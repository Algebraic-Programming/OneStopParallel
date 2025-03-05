/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner   
*/

#include "scheduler/Coarsers/Sarkar.hpp"

void Sarkar::init() {
    if (params.commCostSeq.empty()) {
        int commCost = static_cast<int>(getOriginalInstance()->getArchitecture().synchronisationCosts());
        while (commCost > 0) {
            params.commCostSeq.push_back(commCost);
            commCost /= 2;
        }
    }
    std::sort(params.commCostSeq.begin(), params.commCostSeq.end());
}

std::vector<unsigned> Sarkar::getBotPosetMap() {
    const ComputationalDag &graph = dag_history.back()->getComputationalDag();

    std::vector<unsigned> botPosetMap = graph.get_bottom_node_distance();

    unsigned max = *std::max_element(botPosetMap.begin(), botPosetMap.end());
    max++;

    for (size_t i = 0; i < botPosetMap.size(); i++) {
        botPosetMap[i] = max - botPosetMap[i];
    }

    return botPosetMap;
}

std::vector<int> Sarkar::getTopDistance(int commCost) {
    const ComputationalDag &graph = dag_history.back()->getComputationalDag();

    std::vector<int> topDist(graph.numberOfVertices(), 0);

    for (const auto &vertex : graph.GetTopOrder()) {
        int max_temp = 0;
        for (const auto &j : graph.parents(vertex)) {
            max_temp = std::max(max_temp, topDist[j]);
        }
        if (graph.numberOfParents(vertex) > 0) {
            max_temp += commCost;
        }

        topDist[vertex] = max_temp + graph.nodeWorkWeight(vertex);
    }

    return topDist;
}

std::vector<int> Sarkar::getBotDistance(int commCost) {
    const ComputationalDag &graph = dag_history.back()->getComputationalDag();

    std::vector<int> botDist(graph.numberOfVertices(), 0);

    for (const auto &vertex : graph.dfs_reverse_topoOrder()) {
        int max_temp = 0;
        for (const auto &j : graph.children(vertex)) {
            max_temp = std::max(max_temp, botDist[j]);
        }
        if (graph.numberOfChildren(vertex) > 0) {
            max_temp += commCost;
        }

        botDist[vertex] = max_temp + graph.nodeWorkWeight(vertex);
    }

    return botDist;
}



std::pair<RETURN_STATUS, unsigned> Sarkar::singleContraction(int commCost) {
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();

    const std::vector<std::vector<bool>> nodeNodeCompatabilityMat = dag_history.back()->getNodeNodeCompatabilityMatrix();

    const std::vector<unsigned> vertexPoset = useTopPoset ? graph.get_top_node_distance() : getBotPosetMap();
    const std::vector<int> topDist = getTopDistance(commCost);
    const std::vector<int> botDist = getBotDistance(commCost);

    auto cmp = [](const std::tuple<int, VertexType, VertexType> &lhs, const std::tuple<int, VertexType, VertexType> &rhs) {
        return (std::get<0>(lhs) > std::get<0>(rhs))
                || ((std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) < std::get<1>(rhs)))
                || ((std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs)) && (std::get<2>(lhs) < std::get<2>(rhs)));
    };
    std::set<std::tuple<int, VertexType, VertexType>, decltype(cmp)> edgePriority(cmp);

    for (VertexType edgeSrc = 0; edgeSrc < graph.numberOfVertices(); edgeSrc++) {
        for (const VertexType &edgeTgt : graph.children(edgeSrc)) {

            // if (graph.nodeType(edgeSrc) != graph.nodeType(edgeTgt)) continue;
            if (! nodeNodeCompatabilityMat[graph.nodeType(edgeSrc)][graph.nodeType(edgeTgt)]) continue;
            if (vertexPoset[edgeSrc] + 1 != vertexPoset[edgeTgt]) continue;
            if (topDist[edgeSrc] + commCost + graph.nodeWorkWeight(edgeTgt) != topDist[edgeTgt]) continue;
            if (botDist[edgeTgt] + commCost + graph.nodeWorkWeight(edgeSrc) != botDist[edgeSrc]) continue;

            int maxPath = topDist[edgeSrc] + botDist[edgeTgt] + commCost;
            int maxParentDist = 0;
            int maxChildDist = 0;

            for (const auto &par : graph.parents(edgeSrc)) {
                maxParentDist = std::max(maxParentDist, topDist[par]);
            }
            for (const auto &par : graph.parents(edgeTgt)) {
                if (par == edgeSrc) continue;
                maxParentDist = std::max(maxParentDist, topDist[par]);
            }
            if (graph.numberOfParents(edgeSrc) > 0 || graph.numberOfParents(edgeTgt) > 1) {
                maxParentDist += commCost;
            }

            for (const auto &chld : graph.children(edgeSrc)) {
                if (chld == edgeTgt) continue;
                maxChildDist = std::max(maxChildDist, botDist[chld]);
            }
            for (const auto &chld : graph.children(edgeTgt)) {
                maxChildDist = std::max(maxChildDist, botDist[chld]);
            }
            if (graph.numberOfChildren(edgeSrc) > 1 || graph.numberOfChildren(edgeTgt) > 0) {
                maxChildDist += commCost;
            }

            int newMaxPath = maxParentDist + maxChildDist + graph.nodeWorkWeight(edgeSrc) + graph.nodeWorkWeight(edgeTgt);
            int savings = maxPath - newMaxPath;

            if (savings + static_cast<int>(params.leniency * static_cast<double>(maxPath)) >= 0) {
                edgePriority.emplace(savings, edgeSrc, edgeTgt);
            }
        }
    }

    std::vector<std::unordered_set<VertexType>> partition;
    std::vector<bool> partitionedSourceFlag(graph.numberOfVertices(), false);
    std::vector<bool> partitionedTargetFlag(graph.numberOfVertices(), false);

    unsigned maxCorseningNum = graph.numberOfVertices() - static_cast<unsigned>(static_cast<double>(graph.numberOfVertices()) * params.geomDecay);

    unsigned counter = 0;
    int minSave = 0;
    for (auto prioIter = edgePriority.begin(); prioIter != edgePriority.end(); prioIter++) {
        const int &edgeSave = std::get<0>(*prioIter);
        const VertexType &edgeSrc = std::get<1>(*prioIter);
        const VertexType &edgeTgt = std::get<2>(*prioIter);

        // Iterations halt
        if (edgeSave < minSave) break;

        // Check whether we can glue
        if (partitionedSourceFlag[edgeSrc]) continue;
        if (partitionedSourceFlag[edgeTgt]) continue;
        if (partitionedTargetFlag[edgeSrc]) continue;
        if (partitionedTargetFlag[edgeTgt]) continue;

        bool shouldSkipSrc = false;
        for (const VertexType &chld : graph.children(edgeSrc)) {
            if ((vertexPoset[chld] == vertexPoset[edgeSrc] + 1) && partitionedTargetFlag[chld]) {
                shouldSkipSrc = true;
                break;
            }
        }
        bool shouldSkipTgt = false;
        for (const VertexType &par : graph.children(edgeTgt)) {
            if ((vertexPoset[par] + 1 == vertexPoset[edgeTgt]) && partitionedSourceFlag[par]) {
                shouldSkipTgt = true;
                break;
            }
        }
        if (shouldSkipSrc && shouldSkipTgt) continue;

        // Adding to partition
        partition.emplace_back(std::initializer_list<VertexType>{edgeSrc, edgeTgt});
        counter++;
        if (counter > maxCorseningNum) {
            minSave = edgeSave;
        }
        partitionedSourceFlag[edgeSrc] = true;
        partitionedTargetFlag[edgeTgt] = true;
    }

    RETURN_STATUS status = add_contraction(partition);

    return {status, counter};
}

std::pair<RETURN_STATUS, unsigned> Sarkar::allChildrenContraction(int commCost) {
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();

    const std::vector<std::vector<bool>> nodeNodeCompatabilityMat = dag_history.back()->getNodeNodeCompatabilityMatrix();

    const std::vector<unsigned> vertexPoset = graph.get_top_node_distance();
    const std::vector<int> topDist = getTopDistance(commCost);
    const std::vector<int> botDist = getBotDistance(commCost);

    auto cmp = [](const std::pair<int, VertexType> &lhs, const std::pair<int, VertexType> &rhs) {
        return (lhs.first > rhs.first)
                || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<int, VertexType>, decltype(cmp)> vertPriority(cmp);

    for (VertexType groupHead = 0; groupHead < graph.numberOfVertices(); groupHead++ ) {
        bool shouldSkip = false;
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            if (graph.nodeType(groupHead) != graph.nodeType(groupFoot)) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) continue;
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            if (vertexPoset[groupFoot] != vertexPoset[groupHead] + 1) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) continue;

        int maxPath = topDist[groupHead] + botDist[groupHead] - graph.nodeWorkWeight(groupHead);
        int maxParentDist = 0;
        int maxChildDist = 0;

        for (const VertexType &par : graph.parents(groupHead)) {
            maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
        }
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            for (const VertexType &par : graph.parents(groupFoot)) {
                if (par == groupHead) continue;
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }
        }

        for (const VertexType &groupFoot : graph.children(groupHead)) {
            for (const VertexType &chld : graph.children(groupFoot)) {
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }
        }

        int newMaxPath = maxParentDist + maxChildDist + graph.nodeWorkWeight(groupHead);
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            newMaxPath += graph.nodeWorkWeight(groupFoot);
        }

        int savings = maxPath - newMaxPath;
        if (savings + static_cast<int>(params.leniency * static_cast<double>(maxPath)) >= 0) {
            vertPriority.emplace(savings, groupHead);
        }
    }

    std::vector<std::unordered_set<VertexType>> partition;
    std::vector<bool> partitionedFlag(graph.numberOfVertices(), false);

    unsigned maxCorseningNum = graph.numberOfVertices() - static_cast<unsigned>(static_cast<double>(graph.numberOfVertices()) * params.geomDecay);

    unsigned counter = 0;
    int minSave = 0;
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const int &vertSave = prioIter->first;
        const VertexType &groupHead = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) break;

        // Check whether we can glue
        if (partitionedFlag[groupHead]) continue;
        bool shouldSkip = false;
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            if (partitionedFlag[groupFoot]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) continue;

        // Adding to partition
        std::unordered_set<VertexType> part({groupHead});
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            part.emplace(groupFoot);
        }

        partition.push_back(part);
        counter += graph.numberOfChildren(groupHead);
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFlag[groupHead] = true;
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            partitionedFlag[groupFoot] = true;
        }
    }


    RETURN_STATUS status = add_contraction(partition);
    return {status, counter};
}



std::pair<RETURN_STATUS, unsigned> Sarkar::allParentsContraction(int commCost) {
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();

    const std::vector<std::vector<bool>> nodeNodeCompatabilityMat = dag_history.back()->getNodeNodeCompatabilityMatrix();

    const std::vector<unsigned> vertexPoset = getBotPosetMap();
    const std::vector<int> topDist = getTopDistance(commCost);
    const std::vector<int> botDist = getBotDistance(commCost);

    auto cmp = [](const std::pair<int, VertexType> &lhs, const std::pair<int, VertexType> &rhs) {
        return (lhs.first > rhs.first)
                || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<int, VertexType>, decltype(cmp)> vertPriority(cmp);

    for (VertexType groupFoot = 0; groupFoot < graph.numberOfVertices(); groupFoot++ ) {
        bool shouldSkip = false;
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            if (graph.nodeType(groupHead) != graph.nodeType(groupFoot)) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) continue;
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            if (vertexPoset[groupFoot] != vertexPoset[groupHead] + 1) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) continue;

        int maxPath = topDist[groupFoot] + botDist[groupFoot] - graph.nodeWorkWeight(groupFoot);
        int maxParentDist = 0;
        int maxChildDist = 0;

        for (const VertexType &child : graph.children(groupFoot)) {
            maxChildDist = std::max(maxChildDist, botDist[child] + commCost);
        }
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            for (const VertexType &chld : graph.children(groupHead)) {
                if (chld == groupFoot) continue;
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }
        }

        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            for (const VertexType &par : graph.parents(groupHead)) {
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }
        }

        int newMaxPath = maxParentDist + maxChildDist + graph.nodeWorkWeight(groupFoot);
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            newMaxPath += graph.nodeWorkWeight(groupHead);
        }

        int savings = maxPath - newMaxPath;
        if (savings + static_cast<int>(params.leniency * static_cast<double>(maxPath)) >= 0) {
            vertPriority.emplace(savings, groupFoot);
        }
    }

    std::vector<std::unordered_set<VertexType>> partition;
    std::vector<bool> partitionedFlag(graph.numberOfVertices(), false);

    unsigned maxCorseningNum = graph.numberOfVertices() - static_cast<unsigned>(static_cast<double>(graph.numberOfVertices()) * params.geomDecay);

    unsigned counter = 0;
    int minSave = 0;
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const int &vertSave = prioIter->first;
        const VertexType &groupFoot = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) break;

        // Check whether we can glue
        if (partitionedFlag[groupFoot]) continue;
        bool shouldSkip = false;
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            if (partitionedFlag[groupHead]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) continue;

        // Adding to partition
        std::unordered_set<VertexType> part({groupFoot});
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            part.emplace(groupHead);
        }

        partition.push_back(part);
        counter += graph.numberOfParents(groupFoot);
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFlag[groupFoot] = true;
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            partitionedFlag[groupHead] = true;
        }
    }


    RETURN_STATUS status = add_contraction(partition);
    return {status, counter};
}



RETURN_STATUS Sarkar::run_contractions() {
    init();

    RETURN_STATUS status = SUCCESS;

    for (int commCost : params.commCostSeq) {

        bool change = true;
        while(change) {
            change = false;

            unsigned failedCounter = 0;
            while (failedCounter < 2) {
                std::pair<RETURN_STATUS, unsigned> returnPair = singleContraction(commCost);
                status = std::max(status, returnPair.first);
                if (returnPair.second == 0) {
                    failedCounter++;
                } else {
                    failedCounter = 0;
                    change = true;
                }
                
                useTopPoset = !useTopPoset;
            }

            std::pair<RETURN_STATUS, unsigned> returnPairChildrenContraction = allChildrenContraction(commCost);
            std::pair<RETURN_STATUS, unsigned> returnPairParentsContraction = allParentsContraction(commCost);

            if ((returnPairChildrenContraction.second > 0) || (returnPairParentsContraction.second > 0)) {
                change = true;
            }
        }
    }

    return status;
}