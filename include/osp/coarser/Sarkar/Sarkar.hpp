/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos K. Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "osp/auxiliary/datastructures/union_find.hpp"
#include "osp/auxiliary/hash_util.hpp"
#include "osp/auxiliary/math/divisors.hpp"
#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"

namespace osp {

namespace SarkarParams {

enum class Mode {
    LINES,
    FAN_IN_FULL,
    FAN_IN_PARTIAL,
    FAN_OUT_FULL,
    FAN_OUT_PARTIAL,
    LEVEL_EVEN,
    LEVEL_ODD,
    FAN_IN_BUFFER,
    FAN_OUT_BUFFER,
    HOMOGENEOUS_BUFFER
};

template <typename CommCostType>
struct Parameters {
    double geomDecay_{0.875};
    double leniency_{0.0};
    Mode mode_{Mode::LINES};
    CommCostType commCost_{static_cast<CommCostType>(0)};
    CommCostType maxWeight_{std::numeric_limits<CommCostType>::max()};
    CommCostType smallWeightThreshold_{std::numeric_limits<CommCostType>::lowest()};
    bool useTopPoset_{true};
};

}    // namespace SarkarParams

template <typename GraphTIn, typename GraphTOut>
class Sarkar : public CoarserGenExpansionMap<GraphTIn, GraphTOut> {
  private:
    SarkarParams::Parameters<VWorkwT<GraphTIn>> params_;

    std::vector<VertexIdxT<GraphTIn>> GetBotPosetMap(const GraphTIn &graph) const;
    std::vector<VWorkwT<GraphTIn>> GetTopDistance(VWorkwT<GraphTIn> commCost, const GraphTIn &graph) const;
    std::vector<VWorkwT<GraphTIn>> GetBotDistance(VWorkwT<GraphTIn> commCost, const GraphTIn &graph) const;

    VertexIdxT<GraphTIn> SingleContraction(VWorkwT<GraphTIn> commCost,
                                           const GraphTIn &graph,
                                           std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const;
    VertexIdxT<GraphTIn> AllChildrenContraction(VWorkwT<GraphTIn> commCost,
                                                const GraphTIn &graph,
                                                std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const;
    VertexIdxT<GraphTIn> SomeChildrenContraction(VWorkwT<GraphTIn> commCost,
                                                 const GraphTIn &graph,
                                                 std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const;
    VertexIdxT<GraphTIn> AllParentsContraction(VWorkwT<GraphTIn> commCost,
                                               const GraphTIn &graph,
                                               std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const;
    VertexIdxT<GraphTIn> SomeParentsContraction(VWorkwT<GraphTIn> commCost,
                                                const GraphTIn &graph,
                                                std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const;
    VertexIdxT<GraphTIn> LevelContraction(VWorkwT<GraphTIn> commCost,
                                          const GraphTIn &graph,
                                          std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const;

    VertexIdxT<GraphTIn> HomogeneousBufferMerge(VWorkwT<GraphTIn> commCost,
                                                const GraphTIn &graph,
                                                std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const;
    std::vector<std::size_t> HomogeneousMerge(const std::size_t number, const std::size_t minSize, const std::size_t maxSize) const;

    std::vector<std::size_t> ComputeNodeHashes(const GraphTIn &graph,
                                               const std::vector<VertexIdxT<GraphTIn>> &vertexPoset,
                                               const std::vector<VWorkwT<GraphTIn>> &dist) const;

  public:
    virtual std::vector<std::vector<VertexIdxT<GraphTIn>>> GenerateVertexExpansionMap(const GraphTIn &dagIn) override;
    std::vector<std::vector<VertexIdxT<GraphTIn>>> GenerateVertexExpansionMap(const GraphTIn &dagIn, VertexIdxT<GraphTIn> &diff);

    inline void SetParameters(const SarkarParams::Parameters<VWorkwT<GraphTIn>> &params) { params_ = params; };

    inline SarkarParams::Parameters<VWorkwT<GraphTIn>> &GetParameters() { return params_; };

    inline const SarkarParams::Parameters<VWorkwT<GraphTIn>> &GetParameters() const { return params_; };

    Sarkar(SarkarParams::Parameters<VWorkwT<GraphTIn>> params = SarkarParams::Parameters<VWorkwT<GraphTIn>>())
        : params_(params) {};

    Sarkar(const Sarkar &) = default;
    Sarkar(Sarkar &&) = default;
    Sarkar &operator=(const Sarkar &) = default;
    Sarkar &operator=(Sarkar &&) = default;
    virtual ~Sarkar() override = default;

    std::string GetCoarserName() const override { return "Sarkar"; }
};

template <typename GraphTIn, typename GraphTOut>
std::vector<VertexIdxT<GraphTIn>> Sarkar<GraphTIn, GraphTOut>::GetBotPosetMap(const GraphTIn &graph) const {
    std::vector<VertexIdxT<GraphTIn>> botPosetMap = GetBottomNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(graph);

    VertexIdxT<GraphTIn> max = *std::max_element(botPosetMap.begin(), botPosetMap.end());
    ++max;

    for (std::size_t i = 0; i < botPosetMap.size(); i++) {
        botPosetMap[i] = max - botPosetMap[i];
    }

    return botPosetMap;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<VWorkwT<GraphTIn>> Sarkar<GraphTIn, GraphTOut>::GetTopDistance(VWorkwT<GraphTIn> commCost,
                                                                           const GraphTIn &graph) const {
    std::vector<VWorkwT<GraphTIn>> topDist(graph.NumVertices(), 0);

    for (const auto &vertex : GetTopOrder<GraphTIn>(graph)) {
        VWorkwT<GraphTIn> maxTemp = 0;

        for (const auto &j : graph.Parents(vertex)) {
            maxTemp = std::max(maxTemp, topDist[j]);
        }
        if (graph.InDegree(vertex) > 0) {
            maxTemp += commCost;
        }

        topDist[vertex] = maxTemp + graph.VertexWorkWeight(vertex);
    }

    return topDist;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<VWorkwT<GraphTIn>> Sarkar<GraphTIn, GraphTOut>::GetBotDistance(VWorkwT<GraphTIn> commCost,
                                                                           const GraphTIn &graph) const {
    std::vector<VWorkwT<GraphTIn>> botDist(graph.NumVertices(), 0);

    for (const auto &vertex : GetTopOrderReverse<GraphTIn>(graph)) {
        VWorkwT<GraphTIn> maxTemp = 0;

        for (const auto &j : graph.Children(vertex)) {
            maxTemp = std::max(maxTemp, botDist[j]);
        }
        if (graph.OutDegree(vertex) > 0) {
            maxTemp += commCost;
        }

        botDist[vertex] = maxTemp + graph.VertexWorkWeight(vertex);
    }

    return botDist;
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::SingleContraction(
    VWorkwT<GraphTIn> commCost, const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const {
    using VertexType = VertexIdxT<GraphTIn>;
    assert(expansionMapOutput.size() == 0);

    const std::vector<VertexIdxT<GraphTIn>> vertexPoset
        = params_.useTopPoset_ ? GetTopNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(graph) : GetBotPosetMap(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    auto cmp = [](const std::tuple<long, VertexType, VertexType> &lhs, const std::tuple<long, VertexType, VertexType> &rhs) {
        return (std::get<0>(lhs) > std::get<0>(rhs))
               || ((std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) < std::get<1>(rhs)))
               || ((std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs))
                   && (std::get<2>(lhs) < std::get<2>(rhs)));
    };
    std::set<std::tuple<long, VertexType, VertexType>, decltype(cmp)> edgePriority(cmp);

    for (const VertexType &edgeSrc : graph.Vertices()) {
        for (const VertexType &edgeTgt : graph.Children(edgeSrc)) {
            if constexpr (HasTypedVerticesV<GraphTIn>) {
                if (graph.VertexType(edgeSrc) != graph.VertexType(edgeTgt)) {
                    continue;
                }
            }

            if (vertexPoset[edgeSrc] + 1 != vertexPoset[edgeTgt]) {
                continue;
            }
            if (topDist[edgeSrc] + commCost + graph.VertexWorkWeight(edgeTgt) != topDist[edgeTgt]) {
                continue;
            }
            if (botDist[edgeTgt] + commCost + graph.VertexWorkWeight(edgeSrc) != botDist[edgeSrc]) {
                continue;
            }
            if (graph.VertexWorkWeight(edgeSrc) + graph.VertexWorkWeight(edgeTgt) > params_.maxWeight_) {
                continue;
            }

            VWorkwT<GraphTIn> maxPath = topDist[edgeSrc] + botDist[edgeTgt] + commCost;
            VWorkwT<GraphTIn> maxParentDist = 0;
            VWorkwT<GraphTIn> maxChildDist = 0;

            for (const auto &par : graph.Parents(edgeSrc)) {
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }
            for (const auto &par : graph.Parents(edgeTgt)) {
                if (par == edgeSrc) {
                    continue;
                }
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }

            for (const auto &chld : graph.Children(edgeSrc)) {
                if (chld == edgeTgt) {
                    continue;
                }
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }
            for (const auto &chld : graph.Children(edgeTgt)) {
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }

            VWorkwT<GraphTIn> newMaxPath
                = maxParentDist + maxChildDist + graph.VertexWorkWeight(edgeSrc) + graph.VertexWorkWeight(edgeTgt);
            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);

            // cannot have leniency here as it may destroy symmetries
            if (savings >= 0) {
                edgePriority.emplace(savings, edgeSrc, edgeTgt);
            }
        }
    }

    std::vector<bool> partitionedSourceFlag(graph.NumVertices(), false);
    std::vector<bool> partitionedTargetFlag(graph.NumVertices(), false);

    VertexIdxT<GraphTIn> maxCorseningNum
        = graph.NumVertices() - static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(graph.NumVertices()) * params_.geomDecay_);

    VertexIdxT<GraphTIn> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = edgePriority.begin(); prioIter != edgePriority.end(); prioIter++) {
        const long &edgeSave = std::get<0>(*prioIter);
        const VertexType &edgeSrc = std::get<1>(*prioIter);
        const VertexType &edgeTgt = std::get<2>(*prioIter);

        // Iterations halt
        if (edgeSave < minSave) {
            break;
        }

        // Check whether we can glue
        if (partitionedSourceFlag[edgeSrc]) {
            continue;
        }
        if (partitionedSourceFlag[edgeTgt]) {
            continue;
        }
        if (partitionedTargetFlag[edgeSrc]) {
            continue;
        }
        if (partitionedTargetFlag[edgeTgt]) {
            continue;
        }

        bool shouldSkipSrc = false;
        for (const VertexType &chld : graph.Children(edgeSrc)) {
            if ((vertexPoset[chld] == vertexPoset[edgeSrc] + 1) && partitionedTargetFlag[chld]) {
                shouldSkipSrc = true;
                break;
            }
        }
        bool shouldSkipTgt = false;
        for (const VertexType &par : graph.Parents(edgeTgt)) {
            if ((vertexPoset[par] + 1 == vertexPoset[edgeTgt]) && partitionedSourceFlag[par]) {
                shouldSkipTgt = true;
                break;
            }
        }
        if (shouldSkipSrc && shouldSkipTgt) {
            continue;
        }

        // Adding to partition
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{edgeSrc, edgeTgt});
        counter++;
        if (counter > maxCorseningNum) {
            minSave = edgeSave;
        }
        partitionedSourceFlag[edgeSrc] = true;
        partitionedTargetFlag[edgeTgt] = true;
    }

    expansionMapOutput.reserve(graph.NumVertices() - counter);
    for (const VertexType &vert : graph.Vertices()) {
        if (partitionedSourceFlag[vert]) {
            continue;
        }
        if (partitionedTargetFlag[vert]) {
            continue;
        }

        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::AllChildrenContraction(
    VWorkwT<GraphTIn> commCost, const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const {
    using VertexType = VertexIdxT<GraphTIn>;
    assert(expansionMapOutput.size() == 0);

    const std::vector<VertexIdxT<GraphTIn>> vertexPoset = GetTopNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, VertexType> &lhs, const std::pair<long, VertexType> &rhs) {
        return (lhs.first > rhs.first) || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, VertexType>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType &groupHead : graph.Vertices()) {
        if (graph.OutDegree(groupHead) < 2) {
            continue;
        }

        bool shouldSkip = false;
        if constexpr (HasTypedVerticesV<GraphTIn>) {
            for (const VertexType &groupFoot : graph.Children(groupHead)) {
                if (graph.VertexType(groupHead) != graph.VertexType(groupFoot)) {
                    shouldSkip = true;
                    break;
                }
            }
        }
        if (shouldSkip) {
            continue;
        }
        for (const VertexType &groupFoot : graph.Children(groupHead)) {
            if (vertexPoset[groupFoot] != vertexPoset[groupHead] + 1) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }
        VWorkwT<GraphTIn> combined_weight = graph.VertexWorkWeight(groupHead);
        for (const VertexType &groupFoot : graph.Children(groupHead)) {
            combined_weight += graph.VertexWorkWeight(groupFoot);
        }
        if (combined_weight > params_.maxWeight_) {
            continue;
        }

        VWorkwT<GraphTIn> maxPath = topDist[groupHead] + botDist[groupHead] - graph.VertexWorkWeight(groupHead);
        for (const VertexType &chld : graph.Children(groupHead)) {
            maxPath = std::max(maxPath, topDist[chld] + botDist[chld] - graph.VertexWorkWeight(chld));
        }

        VWorkwT<GraphTIn> maxParentDist = 0;
        VWorkwT<GraphTIn> maxChildDist = 0;

        for (const VertexType &par : graph.Parents(groupHead)) {
            maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
        }
        for (const VertexType &groupFoot : graph.Children(groupHead)) {
            for (const VertexType &par : graph.Parents(groupFoot)) {
                if (par == groupHead) {
                    continue;
                }
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }
        }

        for (const VertexType &groupFoot : graph.Children(groupHead)) {
            for (const VertexType &chld : graph.Children(groupFoot)) {
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }
        }

        VWorkwT<GraphTIn> newMaxPath = maxParentDist + maxChildDist + graph.VertexWorkWeight(groupHead);
        for (const VertexType &groupFoot : graph.Children(groupHead)) {
            newMaxPath += graph.VertexWorkWeight(groupFoot);
        }

        long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);
        if (savings + static_cast<long>(params_.leniency_ * static_cast<double>(maxPath)) >= 0) {
            vertPriority.emplace(savings, groupHead);
        }
    }

    std::vector<bool> partitionedFlag(graph.NumVertices(), false);

    VertexIdxT<GraphTIn> maxCorseningNum
        = graph.NumVertices() - static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(graph.NumVertices()) * params_.geomDecay_);

    VertexIdxT<GraphTIn> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long &vertSave = prioIter->first;
        const VertexType &groupHead = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) {
            break;
        }

        // Check whether we can glue
        if (partitionedFlag[groupHead]) {
            continue;
        }
        bool shouldSkip = false;
        for (const VertexType &groupFoot : graph.Children(groupHead)) {
            if (partitionedFlag[groupFoot]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }

        // Adding to partition
        std::vector<VertexType> part;
        part.reserve(1 + graph.OutDegree(groupHead));
        part.emplace_back(groupHead);
        for (const VertexType &groupFoot : graph.Children(groupHead)) {
            part.emplace_back(groupFoot);
        }

        expansionMapOutput.emplace_back(std::move(part));
        counter += static_cast<VertexIdxT<GraphTIn>>(graph.OutDegree(groupHead));
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFlag[groupHead] = true;
        for (const VertexType &groupFoot : graph.Children(groupHead)) {
            partitionedFlag[groupFoot] = true;
        }
    }

    for (const VertexType &vert : graph.Vertices()) {
        if (partitionedFlag[vert]) {
            continue;
        }
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::AllParentsContraction(
    VWorkwT<GraphTIn> commCost, const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const {
    using VertexType = VertexIdxT<GraphTIn>;
    assert(expansionMapOutput.size() == 0);

    const std::vector<VertexIdxT<GraphTIn>> vertexPoset = GetBotPosetMap(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, VertexType> &lhs, const std::pair<long, VertexType> &rhs) {
        return (lhs.first > rhs.first) || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, VertexType>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType &groupFoot : graph.Vertices()) {
        if (graph.InDegree(groupFoot) < 2) {
            continue;
        }

        bool shouldSkip = false;
        if constexpr (HasTypedVerticesV<GraphTIn>) {
            for (const VertexType &groupHead : graph.Parents(groupFoot)) {
                if (graph.VertexType(groupHead) != graph.VertexType(groupFoot)) {
                    shouldSkip = true;
                    break;
                }
            }
        }
        if (shouldSkip) {
            continue;
        }
        for (const VertexType &groupHead : graph.Parents(groupFoot)) {
            if (vertexPoset[groupFoot] != vertexPoset[groupHead] + 1) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }
        VWorkwT<GraphTIn> combined_weight = graph.VertexWorkWeight(groupFoot);
        for (const VertexType &groupHead : graph.Parents(groupFoot)) {
            combined_weight += graph.VertexWorkWeight(groupHead);
        }
        if (combined_weight > params_.maxWeight_) {
            continue;
        }

        VWorkwT<GraphTIn> maxPath = topDist[groupFoot] + botDist[groupFoot] - graph.VertexWorkWeight(groupFoot);
        for (const VertexType &par : graph.Parents(groupFoot)) {
            maxPath = std::max(maxPath, topDist[par] + botDist[par] - graph.VertexWorkWeight(par));
        }

        VWorkwT<GraphTIn> maxParentDist = 0;
        VWorkwT<GraphTIn> maxChildDist = 0;

        for (const VertexType &child : graph.Children(groupFoot)) {
            maxChildDist = std::max(maxChildDist, botDist[child] + commCost);
        }
        for (const VertexType &groupHead : graph.Parents(groupFoot)) {
            for (const VertexType &chld : graph.Children(groupHead)) {
                if (chld == groupFoot) {
                    continue;
                }
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }
        }

        for (const VertexType &groupHead : graph.Parents(groupFoot)) {
            for (const VertexType &par : graph.Parents(groupHead)) {
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }
        }

        VWorkwT<GraphTIn> newMaxPath = maxParentDist + maxChildDist + graph.VertexWorkWeight(groupFoot);
        for (const VertexType &groupHead : graph.Parents(groupFoot)) {
            newMaxPath += graph.VertexWorkWeight(groupHead);
        }

        long savings = maxPath - newMaxPath;
        if (savings + static_cast<long>(params_.leniency_ * static_cast<double>(maxPath)) >= 0) {
            vertPriority.emplace(savings, groupFoot);
        }
    }

    std::vector<bool> partitionedFlag(graph.NumVertices(), false);

    VertexIdxT<GraphTIn> maxCorseningNum
        = graph.NumVertices() - static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(graph.NumVertices()) * params_.geomDecay_);

    VertexIdxT<GraphTIn> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long &vertSave = prioIter->first;
        const VertexType &groupFoot = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) {
            break;
        }

        // Check whether we can glue
        if (partitionedFlag[groupFoot]) {
            continue;
        }
        bool shouldSkip = false;
        for (const VertexType &groupHead : graph.Parents(groupFoot)) {
            if (partitionedFlag[groupHead]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }

        // Adding to partition
        std::vector<VertexType> part;
        part.reserve(1 + graph.InDegree(groupFoot));
        part.emplace_back(groupFoot);
        for (const VertexType &groupHead : graph.Parents(groupFoot)) {
            part.emplace_back(groupHead);
        }

        expansionMapOutput.emplace_back(std::move(part));
        counter += static_cast<VertexIdxT<GraphTIn>>(graph.InDegree(groupFoot));
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFlag[groupFoot] = true;
        for (const VertexType &groupHead : graph.Parents(groupFoot)) {
            partitionedFlag[groupHead] = true;
        }
    }

    for (const VertexType &vert : graph.Vertices()) {
        if (partitionedFlag[vert]) {
            continue;
        }
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<std::vector<VertexIdxT<GraphTIn>>> Sarkar<GraphTIn, GraphTOut>::GenerateVertexExpansionMap(const GraphTIn &dagIn,
                                                                                                       VertexIdxT<GraphTIn> &diff) {
    std::vector<std::vector<VertexIdxT<GraphTIn>>> expansionMap;

    // std::cout << "Mode: " << static_cast<int>(params.mode) << "\n";
    switch (params_.mode_) {
        case SarkarParams::Mode::LINES: {
            diff = SingleContraction(params_.commCost_, dagIn, expansionMap);
        } break;

        case SarkarParams::Mode::FAN_IN_FULL: {
            diff = AllParentsContraction(params_.commCost_, dagIn, expansionMap);
        } break;

        case SarkarParams::Mode::FAN_IN_PARTIAL: {
            diff = SomeParentsContraction(params_.commCost_, dagIn, expansionMap);
        } break;

        case SarkarParams::Mode::FAN_OUT_FULL: {
            diff = AllChildrenContraction(params_.commCost_, dagIn, expansionMap);
        } break;

        case SarkarParams::Mode::FAN_OUT_PARTIAL: {
            diff = SomeChildrenContraction(params_.commCost_, dagIn, expansionMap);
        } break;

        case SarkarParams::Mode::LEVEL_EVEN:
        case SarkarParams::Mode::LEVEL_ODD: {
            diff = LevelContraction(params_.commCost_, dagIn, expansionMap);
        } break;

        case SarkarParams::Mode::FAN_IN_BUFFER:
        case SarkarParams::Mode::FAN_OUT_BUFFER:
        case SarkarParams::Mode::HOMOGENEOUS_BUFFER: {
            diff = HomogeneousBufferMerge(params_.commCost_, dagIn, expansionMap);
        } break;

        default: {
#ifdef __cpp_lib_unreachable
            std::unreachable();
#endif
            assert(false);
        } break;
    }

    // std::cout << " Diff: " << diff << '\n';

    return expansionMap;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<std::vector<VertexIdxT<GraphTIn>>> Sarkar<GraphTIn, GraphTOut>::GenerateVertexExpansionMap(const GraphTIn &dagIn) {
    VertexIdxT<GraphTIn> dummy;
    return GenerateVertexExpansionMap(dagIn, dummy);
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::SomeChildrenContraction(
    VWorkwT<GraphTIn> commCost, const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const {
    using VertexType = VertexIdxT<GraphTIn>;
    assert(expansionMapOutput.size() == 0);

    const std::vector<VertexIdxT<GraphTIn>> vertexPoset = GetTopNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, std::vector<VertexType>> &lhs, const std::pair<long, std::vector<VertexType>> &rhs) {
        return (lhs.first > rhs.first) || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, std::vector<VertexType>>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType &groupHead : graph.Vertices()) {
        if (graph.OutDegree(groupHead) < 2) {
            continue;
        }

        auto cmp_chld = [&topDist, &botDist](const VertexType &lhs, const VertexType &rhs) {
            return (topDist[lhs] < topDist[rhs]) || ((topDist[lhs] == topDist[rhs]) && (botDist[lhs] > botDist[rhs]))
                   || ((topDist[lhs] == topDist[rhs]) && (botDist[lhs] == botDist[rhs]) && (lhs < rhs));
        };
        std::set<VertexType, decltype(cmp_chld)> childrenPriority(cmp_chld);
        for (const VertexType &chld : graph.Children(groupHead)) {
            if (vertexPoset[chld] == vertexPoset[groupHead] + 1) {
                childrenPriority.emplace(chld);
            }
        }
        if (childrenPriority.size() < 2) {
            continue;
        }

        std::vector<std::pair<typename std::set<VertexType, decltype(cmp_chld)>::const_iterator,
                              typename std::set<VertexType, decltype(cmp_chld)>::const_iterator>>
            admissbleChildrenGroups;
        for (auto chldIterStart = childrenPriority.cbegin(); chldIterStart != childrenPriority.cend();) {
            if constexpr (HasTypedVerticesV<GraphTIn>) {
                if (graph.VertexType(groupHead) != graph.VertexType(*chldIterStart)) {
                    ++chldIterStart;
                    continue;
                }
            }

            const VWorkwT<GraphTIn> t_dist = topDist[*chldIterStart];
            const VWorkwT<GraphTIn> b_dist = botDist[*chldIterStart];
            auto chldIterEnd = chldIterStart;
            while (chldIterEnd != childrenPriority.cend() && t_dist == topDist[*chldIterEnd]
                   && b_dist == botDist[*chldIterEnd]) {
                if constexpr (HasTypedVerticesV<GraphTIn>) {
                    if (graph.VertexType(groupHead) != graph.VertexType(*chldIterEnd)) {
                        break;
                    }
                }
                ++chldIterEnd;
            }

            admissbleChildrenGroups.emplace_back(chldIterStart, chldIterEnd);
            chldIterStart = chldIterEnd;
        }

        std::vector<VertexType> contractionEnsemble;
        std::set<VertexType> contractionChildrenSet;
        contractionEnsemble.reserve(1 + graph.OutDegree(groupHead));
        contractionEnsemble.emplace_back(groupHead);
        VWorkwT<GraphTIn> addedWeight = graph.VertexWorkWeight(groupHead);

        for (std::size_t i = 0U; i < admissbleChildrenGroups.size(); ++i) {
            const auto &first = admissbleChildrenGroups[i].first;
            const auto &last = admissbleChildrenGroups[i].second;

            for (auto it = first; it != last; ++it) {
                contractionEnsemble.emplace_back(*it);
                contractionChildrenSet.emplace(*it);
                addedWeight += graph.VertexWorkWeight(*it);
            }
            if (addedWeight > params_.maxWeight_) {
                break;
            }

            VWorkwT<GraphTIn> maxPath = 0;
            for (const VertexType &vert : contractionEnsemble) {
                maxPath = std::max(maxPath, topDist[vert] + botDist[vert] - graph.VertexWorkWeight(vert));
            }

            VWorkwT<GraphTIn> maxParentDist = 0;
            VWorkwT<GraphTIn> maxChildDist = 0;

            for (const VertexType &vert : contractionEnsemble) {
                for (const VertexType &par : graph.Parents(vert)) {
                    if (par == groupHead) {
                        continue;
                    }
                    maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
                }
            }

            for (const VertexType &chld : graph.Children(groupHead)) {
                if (contractionChildrenSet.find(chld) == contractionChildrenSet.end()) {
                    maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
                }
            }

            for (std::size_t j = 1; j < contractionEnsemble.size(); j++) {
                for (const VertexType &chld : graph.Children(contractionEnsemble[j])) {
                    maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
                }
            }

            VWorkwT<GraphTIn> newMaxPath = maxParentDist + maxChildDist;
            for (const VertexType &vert : contractionEnsemble) {
                newMaxPath += graph.VertexWorkWeight(vert);
            }

            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);
            if (savings + static_cast<long>(params_.leniency_ * static_cast<double>(maxPath)) >= 0) {
                vertPriority.emplace(savings, contractionEnsemble);
            }
        }
    }

    std::vector<bool> partitionedFlag(graph.NumVertices(), false);
    std::vector<bool> partitionedHeadFlag(graph.NumVertices(), false);

    VertexIdxT<GraphTIn> maxCorseningNum
        = graph.NumVertices() - static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(graph.NumVertices()) * params_.geomDecay_);

    VertexIdxT<GraphTIn> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long &vertSave = prioIter->first;
        const VertexType &groupHead = prioIter->second.front();
        const std::vector<VertexType> &contractionEnsemble = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) {
            break;
        }

        // Check whether we can glue
        bool shouldSkip = false;
        for (const VertexType &vert : contractionEnsemble) {
            if (partitionedFlag[vert]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }

        for (const VertexType &chld : graph.Children(groupHead)) {
            if ((std::find(contractionEnsemble.cbegin(), contractionEnsemble.cend(), chld) == contractionEnsemble.cend())
                && (vertexPoset[chld] == vertexPoset[groupHead] + 1)) {
                if ((partitionedFlag[chld]) && (!partitionedHeadFlag[chld])) {
                    shouldSkip = true;
                    break;
                }
            }
        }
        if (shouldSkip) {
            continue;
        }

        // Adding to partition
        expansionMapOutput.emplace_back(contractionEnsemble);
        counter += static_cast<VertexIdxT<GraphTIn>>(contractionEnsemble.size()) - 1;
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedHeadFlag[groupHead] = true;
        for (const VertexType &vert : contractionEnsemble) {
            partitionedFlag[vert] = true;
        }
    }

    for (const VertexType &vert : graph.Vertices()) {
        if (partitionedFlag[vert]) {
            continue;
        }
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::SomeParentsContraction(
    VWorkwT<GraphTIn> commCost, const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const {
    using VertexType = VertexIdxT<GraphTIn>;
    assert(expansionMapOutput.size() == 0);

    const std::vector<VertexIdxT<GraphTIn>> vertexPoset = GetBotPosetMap(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, std::vector<VertexType>> &lhs, const std::pair<long, std::vector<VertexType>> &rhs) {
        return (lhs.first > rhs.first) || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, std::vector<VertexType>>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType &groupFoot : graph.Vertices()) {
        if (graph.InDegree(groupFoot) < 2) {
            continue;
        }

        auto cmp_par = [&topDist, &botDist](const VertexType &lhs, const VertexType &rhs) {
            return (botDist[lhs] < botDist[rhs]) || ((botDist[lhs] == botDist[rhs]) && (topDist[lhs] > topDist[rhs]))
                   || ((botDist[lhs] == botDist[rhs]) && (topDist[lhs] == topDist[rhs]) && (lhs < rhs));
        };
        std::set<VertexType, decltype(cmp_par)> parentsPriority(cmp_par);
        for (const VertexType &par : graph.Parents(groupFoot)) {
            if (vertexPoset[par] + 1 == vertexPoset[groupFoot]) {
                parentsPriority.emplace(par);
            }
        }
        if (parentsPriority.size() < 2) {
            continue;
        }

        std::vector<std::pair<typename std::set<VertexType, decltype(cmp_par)>::const_iterator,
                              typename std::set<VertexType, decltype(cmp_par)>::const_iterator>>
            admissbleParentGroups;
        for (auto parIterStart = parentsPriority.cbegin(); parIterStart != parentsPriority.cend();) {
            if constexpr (HasTypedVerticesV<GraphTIn>) {
                if (graph.VertexType(groupFoot) != graph.VertexType(*parIterStart)) {
                    ++parIterStart;
                    continue;
                }
            }

            const VWorkwT<GraphTIn> t_dist = topDist[*parIterStart];
            const VWorkwT<GraphTIn> b_dist = botDist[*parIterStart];
            auto parIterEnd = parIterStart;
            while (parIterEnd != parentsPriority.cend() && t_dist == topDist[*parIterEnd] && b_dist == botDist[*parIterEnd]) {
                if constexpr (HasTypedVerticesV<GraphTIn>) {
                    if (graph.VertexType(groupFoot) != graph.VertexType(*parIterEnd)) {
                        break;
                    }
                }
                ++parIterEnd;
            }

            admissbleParentGroups.emplace_back(parIterStart, parIterEnd);
            parIterStart = parIterEnd;
        }

        std::vector<VertexType> contractionEnsemble;
        std::set<VertexType> contractionParentsSet;
        contractionEnsemble.reserve(1 + graph.InDegree(groupFoot));
        contractionEnsemble.emplace_back(groupFoot);
        VWorkwT<GraphTIn> addedWeight = graph.VertexWorkWeight(groupFoot);

        for (std::size_t i = 0U; i < admissbleParentGroups.size(); ++i) {
            const auto &first = admissbleParentGroups[i].first;
            const auto &last = admissbleParentGroups[i].second;

            for (auto it = first; it != last; ++it) {
                contractionEnsemble.emplace_back(*it);
                contractionParentsSet.emplace(*it);
                addedWeight += graph.VertexWorkWeight(*it);
            }
            if (addedWeight > params_.maxWeight_) {
                break;
            }

            VWorkwT<GraphTIn> maxPath = 0;
            for (const VertexType &vert : contractionEnsemble) {
                maxPath = std::max(maxPath, topDist[vert] + botDist[vert] - graph.VertexWorkWeight(vert));
            }

            VWorkwT<GraphTIn> maxParentDist = 0;
            VWorkwT<GraphTIn> maxChildDist = 0;

            for (const VertexType &vert : contractionEnsemble) {
                for (const VertexType &chld : graph.Children(vert)) {
                    if (chld == groupFoot) {
                        continue;
                    }
                    maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
                }
            }

            for (const VertexType &par : graph.Parents(groupFoot)) {
                if (contractionParentsSet.find(par) == contractionParentsSet.end()) {
                    maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
                }
            }

            for (std::size_t j = 1; j < contractionEnsemble.size(); j++) {
                for (const VertexType &par : graph.Parents(contractionEnsemble[j])) {
                    maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
                }
            }

            VWorkwT<GraphTIn> newMaxPath = maxParentDist + maxChildDist;
            for (const VertexType &vert : contractionEnsemble) {
                newMaxPath += graph.VertexWorkWeight(vert);
            }

            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);
            if (savings + static_cast<long>(params_.leniency_ * static_cast<double>(maxPath)) >= 0) {
                vertPriority.emplace(savings, contractionEnsemble);
            }
        }
    }

    std::vector<bool> partitionedFlag(graph.NumVertices(), false);
    std::vector<bool> partitionedFootFlag(graph.NumVertices(), false);

    VertexIdxT<GraphTIn> maxCorseningNum
        = graph.NumVertices() - static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(graph.NumVertices()) * params_.geomDecay_);

    VertexIdxT<GraphTIn> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long &vertSave = prioIter->first;
        const VertexType &groupFoot = prioIter->second.front();
        const std::vector<VertexType> &contractionEnsemble = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) {
            break;
        }

        // Check whether we can glue
        bool shouldSkip = false;
        for (const VertexType &vert : contractionEnsemble) {
            if (partitionedFlag[vert]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) {
            continue;
        }

        for (const VertexType &par : graph.Parents(groupFoot)) {
            if ((std::find(contractionEnsemble.cbegin(), contractionEnsemble.cend(), par) == contractionEnsemble.cend())
                && (vertexPoset[par] + 1 == vertexPoset[groupFoot])) {
                if ((partitionedFlag[par]) && (!partitionedFootFlag[par])) {
                    shouldSkip = true;
                    break;
                }
            }
        }
        if (shouldSkip) {
            continue;
        }

        // Adding to partition
        expansionMapOutput.emplace_back(contractionEnsemble);
        counter += static_cast<VertexIdxT<GraphTIn>>(contractionEnsemble.size()) - 1;
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFootFlag[groupFoot] = true;
        for (const VertexType &vert : contractionEnsemble) {
            partitionedFlag[vert] = true;
        }
    }

    for (const VertexType &vert : graph.Vertices()) {
        if (partitionedFlag[vert]) {
            continue;
        }
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::LevelContraction(
    VWorkwT<GraphTIn> commCost, const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const {
    using VertexType = VertexIdxT<GraphTIn>;
    assert(expansionMapOutput.size() == 0);

    const std::vector<VertexIdxT<GraphTIn>> vertexPoset
        = params_.useTopPoset_ ? GetTopNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(graph) : GetBotPosetMap(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, std::vector<VertexType>> &lhs, const std::pair<long, std::vector<VertexType>> &rhs) {
        return (lhs.first > rhs.first) || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, std::vector<VertexType>>, decltype(cmp)> vertPriority(cmp);

    const VertexIdxT<GraphTIn> minLevel = *std::min_element(vertexPoset.cbegin(), vertexPoset.cend());
    const VertexIdxT<GraphTIn> maxLevel = *std::max_element(vertexPoset.cbegin(), vertexPoset.cend());

    const VertexIdxT<GraphTIn> parity = params_.mode_ == SarkarParams::Mode::LEVEL_EVEN ? 0 : 1;

    std::vector<std::vector<VertexIdxT<GraphTIn>>> levels(maxLevel - minLevel + 1);
    for (const VertexType &vert : graph.Vertices()) {
        levels[vertexPoset[vert] - minLevel].emplace_back(vert);
    }

    for (VertexIdxT<GraphTIn> headLevel = minLevel + parity; headLevel < maxLevel; headLevel += 2) {
        const VertexIdxT<GraphTIn> footLevel = headLevel + 1;

        const std::vector<VertexIdxT<GraphTIn>> &headVertices = levels[headLevel - minLevel];
        const std::vector<VertexIdxT<GraphTIn>> &footVertices = levels[footLevel - minLevel];

        UnionFindUniverse<VertexType, std::size_t, VWorkwT<GraphTIn>, VMemwT<GraphTIn>> uf;
        for (const VertexType &vert : headVertices) {
            uf.AddObject(vert, graph.VertexWorkWeight(vert));
        }
        for (const VertexType &vert : footVertices) {
            uf.AddObject(vert, graph.VertexWorkWeight(vert));
        }

        for (const VertexType &srcVert : headVertices) {
            for (const VertexType &tgtVert : graph.Children(srcVert)) {
                if (vertexPoset[tgtVert] != footLevel) {
                    continue;
                }

                if constexpr (HasTypedVerticesV<GraphTIn>) {
                    if (graph.VertexType(srcVert) != graph.VertexType(tgtVert)) {
                        continue;
                    }
                }

                uf.JoinByName(srcVert, tgtVert);
            }
        }

        std::vector<std::vector<VertexType>> components = uf.GetConnectedComponents();
        for (std::vector<VertexType> &comp : components) {
            if (comp.size() < 2) {
                continue;
            }
            if (uf.GetWeightOfComponentByName(comp.at(0)) > params_.maxWeight_) {
                continue;
            }

            std::sort(comp.begin(), comp.end());

            VWorkwT<GraphTIn> maxPath = std::numeric_limits<VWorkwT<GraphTIn>>::lowest();
            for (const VertexType &vert : comp) {
                maxPath = std::max(maxPath, topDist[vert] + botDist[vert] - graph.VertexWorkWeight(vert));
            }

            VWorkwT<GraphTIn> maxParentDist = 0;
            for (const VertexType &vert : comp) {
                for (const VertexType &par : graph.Parents(vert)) {
                    if (std::binary_search(comp.cbegin(), comp.cend(), par)) {
                        continue;
                    }

                    maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
                }
            }

            VWorkwT<GraphTIn> maxChildDist = 0;
            for (const VertexType &vert : comp) {
                for (const VertexType &chld : graph.Children(vert)) {
                    if (std::binary_search(comp.cbegin(), comp.cend(), chld)) {
                        continue;
                    }

                    maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
                }
            }

            VWorkwT<GraphTIn> newMaxPath = maxParentDist + maxChildDist;
            for (const VertexType &vert : comp) {
                newMaxPath += graph.VertexWorkWeight(vert);
            }

            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);

            if (savings + static_cast<long>(params_.leniency_ * static_cast<double>(maxPath)) >= 0) {
                vertPriority.emplace(savings, comp);
            }
        }
    }

    std::vector<bool> partitionedFlag(graph.NumVertices(), false);

    VertexIdxT<GraphTIn> maxCorseningNum
        = graph.NumVertices() - static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(graph.NumVertices()) * params_.geomDecay_);

    VertexIdxT<GraphTIn> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.cbegin(); prioIter != vertPriority.cend(); prioIter++) {
        const long &compSave = prioIter->first;
        const std::vector<VertexType> &comp = prioIter->second;

        // Iterations halt
        if (compSave < minSave) {
            break;
        }

        // Check whether we can glue
        bool shouldSkipHead = false;
        bool shouldSkipFoot = false;
        for (const VertexType &vert : comp) {
            if (((vertexPoset[vert] - minLevel - parity) % 2) == 0) {    // head vertex
                for (const VertexType &chld : graph.Children(vert)) {
                    if ((vertexPoset[chld] == vertexPoset[vert] + 1) && partitionedFlag[chld]) {
                        shouldSkipHead = true;
                    }
                }
            } else {    // foot vertex
                for (const VertexType &par : graph.Parents(vert)) {
                    if ((vertexPoset[par] + 1 == vertexPoset[vert]) && partitionedFlag[par]) {
                        shouldSkipFoot = true;
                    }
                }
            }
        }

        if (shouldSkipHead && shouldSkipFoot) {
            continue;
        }

        // Adding to partition
        expansionMapOutput.emplace_back(comp);
        counter += static_cast<VertexIdxT<GraphTIn>>(comp.size() - 1);
        if (counter > maxCorseningNum) {
            minSave = compSave;
        }

        for (const VertexType &vert : comp) {
            partitionedFlag[vert] = true;
        }
    }

    expansionMapOutput.reserve(graph.NumVertices() - counter);
    for (const VertexType &vert : graph.Vertices()) {
        if (partitionedFlag[vert]) {
            continue;
        }

        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<std::size_t> Sarkar<GraphTIn, GraphTOut>::ComputeNodeHashes(const GraphTIn &graph,
                                                                        const std::vector<VertexIdxT<GraphTIn>> &vertexPoset,
                                                                        const std::vector<VWorkwT<GraphTIn>> &dist) const {
    using VertexType = VertexIdxT<GraphTIn>;

    std::vector<std::size_t> hashes(graph.NumVertices());
    for (const VertexType &vert : graph.Vertices()) {
        std::size_t &hash = hashes[vert];
        hash = std::hash<VWorkwT<GraphTIn>>{}(graph.VertexWorkWeight(vert));
        HashCombine(hash, vertexPoset[vert]);
        HashCombine(hash, dist[vert]);
        if constexpr (HasTypedVerticesV<GraphTIn>) {
            HashCombine(hash, graph.VertexType(vert));
        }
    }

    return hashes;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<std::size_t> Sarkar<GraphTIn, GraphTOut>::HomogeneousMerge(const std::size_t number,
                                                                       const std::size_t minSize,
                                                                       const std::size_t maxSize) const {
    assert(minSize <= maxSize);
    assert(number > 0);

    std::size_t bestDiv = 1U;
    for (std::size_t div : DivisorsList(number)) {
        if (div > maxSize) {
            continue;
        }

        if (div < minSize && bestDiv < div) {
            bestDiv = div;
        }
        if (div >= minSize && ((bestDiv < minSize) || (div < bestDiv))) {
            bestDiv = div;
        }
    }

    if (bestDiv != 1U) {
        return std::vector<std::size_t>(number / bestDiv, bestDiv);
    }

    std::size_t bestScore = 0U;
    std::size_t bestBins = number / minSize;
    for (std::size_t bins = std::max(number / maxSize, static_cast<std::size_t>(2U)); bins <= number / minSize; ++bins) {
        if (number % bins == 0U && number != bins) {
            return std::vector<std::size_t>(bins, number / bins);
        }

        std::size_t score = std::min(DivisorsList(number / bins).size(), DivisorsList((number / bins) + 1).size());
        if (score >= bestScore) {
            bestScore = score;
            bestBins = bins;
        }
    }

    std::size_t remainder = number % bestBins;
    std::size_t size = number / bestBins;

    std::vector<std::size_t> groups;
    for (std::size_t i = 0U; i < bestBins; ++i) {
        if (remainder != 0U) {
            groups.emplace_back(size + 1U);
            --remainder;
        } else {
            groups.emplace_back(size);
        }
    }

    return groups;
}

template <typename GraphTIn, typename GraphTOut>
VertexIdxT<GraphTIn> Sarkar<GraphTIn, GraphTOut>::HomogeneousBufferMerge(
    VWorkwT<GraphTIn> commCost, const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &expansionMapOutput) const {
    using VertexType = VertexIdxT<GraphTIn>;
    assert(expansionMapOutput.size() == 0);

    const std::vector<VertexIdxT<GraphTIn>> vertexTopPoset = GetTopNodeDistance<GraphTIn, VertexIdxT<GraphTIn>>(graph);
    const std::vector<VertexIdxT<GraphTIn>> vertexBotPoset = GetBotPosetMap(graph);
    const std::vector<VWorkwT<GraphTIn>> topDist = GetTopDistance(commCost, graph);
    const std::vector<VWorkwT<GraphTIn>> botDist = GetBotDistance(commCost, graph);

    std::vector<std::size_t> hashValuesCombined(graph.NumVertices(), 1729U);

    if (params_.mode_ == SarkarParams::Mode::FAN_OUT_BUFFER || params_.mode_ == SarkarParams::Mode::HOMOGENEOUS_BUFFER) {
        const std::vector<std::size_t> hashValues = ComputeNodeHashes(graph, vertexTopPoset, topDist);
        std::vector<std::size_t> hashValuesWithParents = hashValues;
        for (const VertexType &par : graph.Vertices()) {
            for (const VertexType &chld : graph.Children(par)) {
                HashCombine(hashValuesWithParents[chld], hashValues[par]);
            }
        }
        for (const VertexType &vert : graph.Vertices()) {
            HashCombine(hashValuesCombined[vert], hashValuesWithParents[vert]);
        }
    }
    if (params_.mode_ == SarkarParams::Mode::FAN_IN_BUFFER || params_.mode_ == SarkarParams::Mode::HOMOGENEOUS_BUFFER) {
        const std::vector<std::size_t> hashValues = ComputeNodeHashes(graph, vertexBotPoset, botDist);
        std::vector<std::size_t> hashValuesWithChildren = hashValues;
        for (const VertexType &chld : graph.Vertices()) {
            for (const VertexType &par : graph.Parents(chld)) {
                HashCombine(hashValuesWithChildren[par], hashValues[chld]);
            }
        }
        for (const VertexType &vert : graph.Vertices()) {
            HashCombine(hashValuesCombined[vert], hashValuesWithChildren[vert]);
        }
    }

    std::unordered_map<std::size_t, std::set<VertexType>> orbits;
    for (const VertexType &vert : graph.Vertices()) {
        if (graph.VertexWorkWeight(vert) > params_.smallWeightThreshold_) {
            continue;
        }

        const std::size_t hash = hashValuesCombined[vert];
        auto foundIter = orbits.find(hash);
        if (foundIter == orbits.end()) {
            orbits.emplace(std::piecewise_construct,
                           std::forward_as_tuple(hash),
                           std::forward_as_tuple(std::initializer_list<VertexIdxT<GraphTIn>>{vert}));
        } else {
            foundIter->second.emplace(vert);
        }
    }

    VertexIdxT<GraphTIn> counter = 0;
    std::vector<bool> partitionedFlag(graph.NumVertices(), false);

    for (const VertexType &vert : graph.Vertices()) {
        if (graph.VertexWorkWeight(vert) > params_.smallWeightThreshold_) {
            continue;
        }
        if (partitionedFlag[vert]) {
            continue;
        }

        const std::set<VertexType> &orb = orbits.at(hashValuesCombined[vert]);
        if (orb.size() <= 1U) {
            continue;
        }

        std::set<VertexType> parents;
        if (params_.mode_ == SarkarParams::Mode::FAN_OUT_BUFFER || params_.mode_ == SarkarParams::Mode::HOMOGENEOUS_BUFFER) {
            for (const VertexType &par : graph.Parents(vert)) {
                parents.emplace(par);
            }
        }

        std::set<VertexType> children;
        if (params_.mode_ == SarkarParams::Mode::FAN_IN_BUFFER || params_.mode_ == SarkarParams::Mode::HOMOGENEOUS_BUFFER) {
            for (const VertexType &chld : graph.Children(vert)) {
                children.emplace(chld);
            }
        }

        std::set<VertexType> secureOrb;
        for (const VertexType &vertCandidate : orb) {
            if (vertexTopPoset[vertCandidate] != vertexTopPoset[vert]) {
                continue;
            }
            if (vertexBotPoset[vertCandidate] != vertexBotPoset[vert]) {
                continue;
            }
            if (graph.VertexWorkWeight(vertCandidate) != graph.VertexWorkWeight(vert)) {
                continue;
            }
            if (topDist[vertCandidate] != topDist[vert]) {
                continue;
            }
            if (botDist[vertCandidate] != botDist[vert]) {
                continue;
            }
            if constexpr (HasTypedVerticesV<GraphTIn>) {
                if (graph.VertexType(vertCandidate) != graph.VertexType(vert)) {
                    continue;
                }
            }

            if (params_.mode_ == SarkarParams::Mode::FAN_OUT_BUFFER || params_.mode_ == SarkarParams::Mode::HOMOGENEOUS_BUFFER) {
                std::set<VertexType> candidateParents;
                for (const VertexType &par : graph.Parents(vertCandidate)) {
                    candidateParents.emplace(par);
                }
                if (candidateParents != parents) {
                    continue;
                }
            }

            if (params_.mode_ == SarkarParams::Mode::FAN_IN_BUFFER || params_.mode_ == SarkarParams::Mode::HOMOGENEOUS_BUFFER) {
                std::set<VertexType> candidateChildren;
                for (const VertexType &chld : graph.Children(vertCandidate)) {
                    candidateChildren.emplace(chld);
                }
                if (candidateChildren != children) {
                    continue;
                }
            }

            secureOrb.emplace(vertCandidate);
        }
        if (secureOrb.size() <= 1U) {
            continue;
        }

        const VWorkwT<GraphTIn> desiredVerticesInGroup = graph.VertexWorkWeight(vert) == 0
                                                             ? std::numeric_limits<VWorkwT<GraphTIn>>::lowest()
                                                             : params_.smallWeightThreshold_ / graph.VertexWorkWeight(vert);
        const VWorkwT<GraphTIn> maxVerticesInGroup = graph.VertexWorkWeight(vert) == 0
                                                         ? std::numeric_limits<VWorkwT<GraphTIn>>::max()
                                                         : params_.maxWeight_ / graph.VertexWorkWeight(vert);

        const std::size_t minDesiredSize = desiredVerticesInGroup < 2 ? 2U : static_cast<std::size_t>(desiredVerticesInGroup);
        const std::size_t maxDesiredSize
            = std::max(minDesiredSize, std::min(minDesiredSize * 2U, static_cast<std::size_t>(maxVerticesInGroup)));

        std::vector<std::size_t> groups = HomogeneousMerge(secureOrb.size(), minDesiredSize, maxDesiredSize);

        auto secureOrbIter = secureOrb.begin();
        for (std::size_t groupSize : groups) {
            std::vector<VertexType> cluster;
            for (std::size_t i = 0; i < groupSize; ++i) {
                cluster.emplace_back(*secureOrbIter);
                ++secureOrbIter;
            }
            expansionMapOutput.emplace_back(std::move(cluster));
            counter += static_cast<VertexType>(groupSize) - 1;
        }

        for (const VertexType &touchedVertex : secureOrb) {
            partitionedFlag[touchedVertex] = true;
        }
    }

    for (const VertexType &vert : graph.Vertices()) {
        if (partitionedFlag[vert]) {
            continue;
        }
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

}    // end namespace osp
