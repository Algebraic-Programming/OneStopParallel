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

enum class Mode { LINES, FAN_IN_FULL, FAN_IN_PARTIAL, FAN_OUT_FULL, FAN_OUT_PARTIAL, LEVEL_EVEN, LEVEL_ODD, FAN_IN_BUFFER, FAN_OUT_BUFFER, HOMOGENEOUS_BUFFER };

template<typename commCostType>
struct Parameters {
    double geomDecay{0.875};
    double leniency{0.0};
    Mode mode{Mode::LINES};
    commCostType commCost{ static_cast<commCostType>(0) };
    commCostType maxWeight{ std::numeric_limits<commCostType>::max() };
    commCostType smallWeightThreshold{ std::numeric_limits<commCostType>::lowest() };
    bool useTopPoset{true};
};
} // end namespace SarkarParams

template<typename Graph_t_in, typename Graph_t_out>
class Sarkar : public CoarserGenExpansionMap<Graph_t_in, Graph_t_out> {
    private:
        SarkarParams::Parameters< v_workw_t<Graph_t_in> > params;
        
        std::vector< vertex_idx_t<Graph_t_in> > getBotPosetMap(const Graph_t_in &graph) const;
        std::vector< v_workw_t<Graph_t_in> > getTopDistance(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph) const;
        std::vector< v_workw_t<Graph_t_in> > getBotDistance(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph) const;
        
        vertex_idx_t<Graph_t_in> singleContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        vertex_idx_t<Graph_t_in> allChildrenContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        vertex_idx_t<Graph_t_in> someChildrenContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        vertex_idx_t<Graph_t_in> allParentsContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        vertex_idx_t<Graph_t_in> someParentsContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        vertex_idx_t<Graph_t_in> levelContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        
        vertex_idx_t<Graph_t_in> homogeneous_buffer_merge(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        vertex_idx_t<Graph_t_in> out_buffer_merge(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        vertex_idx_t<Graph_t_in> in_buffer_merge(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const;
        
        std::vector<std::size_t> homogeneousMerge(const std::size_t number, const std::size_t minSize, const std::size_t maxSize) const;

        std::vector<std::size_t> computeNodeHashes(const Graph_t_in &graph, const std::vector< vertex_idx_t<Graph_t_in> > &vertexPoset, const std::vector< v_workw_t<Graph_t_in> > &dist) const;

    public:
        virtual std::vector<std::vector<vertex_idx_t<Graph_t_in>>> generate_vertex_expansion_map(const Graph_t_in &dag_in) override;
        std::vector<std::vector<vertex_idx_t<Graph_t_in>>> generate_vertex_expansion_map(const Graph_t_in &dag_in, vertex_idx_t<Graph_t_in> &diff);

        inline void setParameters(const SarkarParams::Parameters< v_workw_t<Graph_t_in> >& params_) { params = params_; };
        inline SarkarParams::Parameters< v_workw_t<Graph_t_in> >& getParameters() { return params; };
        inline const SarkarParams::Parameters< v_workw_t<Graph_t_in> >& getParameters() const { return params; };

        Sarkar(SarkarParams::Parameters< v_workw_t<Graph_t_in> > params_ = SarkarParams::Parameters< v_workw_t<Graph_t_in> >()) : params(params_) {};

        Sarkar(const Sarkar &) = default;
        Sarkar(Sarkar &&) = default;
        Sarkar &operator=(const Sarkar &) = default;
        Sarkar &operator=(Sarkar &&) = default;
        virtual ~Sarkar() override = default;

        std::string getCoarserName() const override { return "Sarkar"; }
};






template<typename Graph_t_in, typename Graph_t_out>
std::vector< vertex_idx_t<Graph_t_in> > Sarkar<Graph_t_in, Graph_t_out>::getBotPosetMap(const Graph_t_in &graph) const {
    std::vector< vertex_idx_t<Graph_t_in> > botPosetMap = get_bottom_node_distance<Graph_t_in, vertex_idx_t<Graph_t_in>>(graph);

    vertex_idx_t<Graph_t_in> max = *std::max_element(botPosetMap.begin(), botPosetMap.end());
    ++max;

    for (std::size_t i = 0; i < botPosetMap.size(); i++) {
        botPosetMap[i] = max - botPosetMap[i];
    }

    return botPosetMap;
}

template<typename Graph_t_in, typename Graph_t_out>
std::vector< v_workw_t<Graph_t_in> > Sarkar<Graph_t_in, Graph_t_out>::getTopDistance(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph) const {
    std::vector< v_workw_t<Graph_t_in> > topDist(graph.num_vertices(), 0);

    for (const auto &vertex : GetTopOrder<Graph_t_in>(graph)) {
        v_workw_t<Graph_t_in> max_temp = 0;

        for (const auto &j : graph.parents(vertex)) {
            max_temp = std::max(max_temp, topDist[j]);
        }
        if (graph.in_degree(vertex) > 0) {
            max_temp += commCost;
        }

        topDist[vertex] = max_temp + graph.vertex_work_weight(vertex);
    }

    return topDist;
}

template<typename Graph_t_in, typename Graph_t_out>
std::vector< v_workw_t<Graph_t_in> > Sarkar<Graph_t_in, Graph_t_out>::getBotDistance(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph) const {
    std::vector< v_workw_t<Graph_t_in> > botDist(graph.num_vertices(), 0);

    for (const auto &vertex : GetTopOrderReverse<Graph_t_in>(graph)) {
        v_workw_t<Graph_t_in> max_temp = 0;

        for (const auto &j : graph.children(vertex)) {
            max_temp = std::max(max_temp, botDist[j]);
        }
        if (graph.out_degree(vertex) > 0) {
            max_temp += commCost;
        }

        botDist[vertex] = max_temp + graph.vertex_work_weight(vertex);
    }

    return botDist;
}


template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::singleContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);

    const std::vector< vertex_idx_t<Graph_t_in> > vertexPoset = params.useTopPoset ? get_top_node_distance<Graph_t_in, vertex_idx_t<Graph_t_in>>(graph) : getBotPosetMap(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    auto cmp = [](const std::tuple<long, VertexType, VertexType> &lhs, const std::tuple<long, VertexType, VertexType> &rhs) {
        return (std::get<0>(lhs) > std::get<0>(rhs))
                || ((std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) < std::get<1>(rhs)))
                || ((std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs)) && (std::get<2>(lhs) < std::get<2>(rhs)));
    };
    std::set<std::tuple<long, VertexType, VertexType>, decltype(cmp)> edgePriority(cmp);

    for (const VertexType &edgeSrc : graph.vertices()) {
        for (const VertexType &edgeTgt : graph.children(edgeSrc)) {

            if constexpr (has_typed_vertices_v<Graph_t_in>) {
                if (graph.vertex_type(edgeSrc) != graph.vertex_type(edgeTgt)) continue;
            }

            if (vertexPoset[edgeSrc] + 1 != vertexPoset[edgeTgt]) continue;
            if (topDist[edgeSrc] + commCost + graph.vertex_work_weight(edgeTgt) != topDist[edgeTgt]) continue;
            if (botDist[edgeTgt] + commCost + graph.vertex_work_weight(edgeSrc) != botDist[edgeSrc]) continue;
            if (graph.vertex_work_weight(edgeSrc) + graph.vertex_work_weight(edgeTgt) > params.maxWeight) continue;

            v_workw_t<Graph_t_in> maxPath = topDist[edgeSrc] + botDist[edgeTgt] + commCost;
            v_workw_t<Graph_t_in> maxParentDist = 0;
            v_workw_t<Graph_t_in> maxChildDist = 0;

            for (const auto &par : graph.parents(edgeSrc)) {
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }
            for (const auto &par : graph.parents(edgeTgt)) {
                if (par == edgeSrc) continue;
                maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
            }

            for (const auto &chld : graph.children(edgeSrc)) {
                if (chld == edgeTgt) continue;
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }
            for (const auto &chld : graph.children(edgeTgt)) {
                maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
            }

            v_workw_t<Graph_t_in> newMaxPath = maxParentDist + maxChildDist + graph.vertex_work_weight(edgeSrc) + graph.vertex_work_weight(edgeTgt);
            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);

            // cannot have leniency here as it may destroy symmetries
            if (savings >= 0) {
                edgePriority.emplace(savings, edgeSrc, edgeTgt);
            }
        }
    }

    std::vector<bool> partitionedSourceFlag(graph.num_vertices(), false);
    std::vector<bool> partitionedTargetFlag(graph.num_vertices(), false);

    vertex_idx_t<Graph_t_in> maxCorseningNum = graph.num_vertices() - static_cast< vertex_idx_t<Graph_t_in> >(static_cast<double>(graph.num_vertices()) * params.geomDecay);


    vertex_idx_t<Graph_t_in> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = edgePriority.begin(); prioIter != edgePriority.end(); prioIter++) {
        const long &edgeSave = std::get<0>(*prioIter);
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
        for (const VertexType &par : graph.parents(edgeTgt)) {
            if ((vertexPoset[par] + 1 == vertexPoset[edgeTgt]) && partitionedSourceFlag[par]) {
                shouldSkipTgt = true;
                break;
            }
        }
        if (shouldSkipSrc && shouldSkipTgt) continue;

        // Adding to partition
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{edgeSrc, edgeTgt});
        counter++;
        if (counter > maxCorseningNum) {
            minSave = edgeSave;
        }
        partitionedSourceFlag[edgeSrc] = true;
        partitionedTargetFlag[edgeTgt] = true;
    }

    expansionMapOutput.reserve(graph.num_vertices() - counter);
    for (const VertexType &vert : graph.vertices()) {
        if (partitionedSourceFlag[vert]) continue;
        if (partitionedTargetFlag[vert]) continue;

        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}


template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::allChildrenContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);

    const std::vector< vertex_idx_t<Graph_t_in> > vertexPoset = get_top_node_distance<Graph_t_in, vertex_idx_t<Graph_t_in>>(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, VertexType> &lhs, const std::pair<long, VertexType> &rhs) {
        return (lhs.first > rhs.first)
                || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, VertexType>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType &groupHead : graph.vertices()) {
        if (graph.out_degree(groupHead) < 2) continue;

        bool shouldSkip = false;
        if constexpr (has_typed_vertices_v<Graph_t_in>) {
            for (const VertexType &groupFoot : graph.children(groupHead)) {
                if (graph.vertex_type(groupHead) != graph.vertex_type(groupFoot)) {
                    shouldSkip = true;
                    break;
                }
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
        v_workw_t<Graph_t_in> combined_weight = graph.vertex_work_weight(groupHead);
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            combined_weight += graph.vertex_work_weight(groupFoot);
        }
        if (combined_weight > params.maxWeight) continue;

        v_workw_t<Graph_t_in> maxPath = topDist[groupHead] + botDist[groupHead] - graph.vertex_work_weight(groupHead);
        for (const VertexType &chld : graph.children(groupHead)) {
            maxPath = std::max(maxPath, topDist[chld] + botDist[chld] - graph.vertex_work_weight(chld));
        }

        v_workw_t<Graph_t_in> maxParentDist = 0;
        v_workw_t<Graph_t_in> maxChildDist = 0;

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

        v_workw_t<Graph_t_in> newMaxPath = maxParentDist + maxChildDist + graph.vertex_work_weight(groupHead);
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            newMaxPath += graph.vertex_work_weight(groupFoot);
        }

        long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);
        if (savings + static_cast<long>(params.leniency * static_cast<double>(maxPath)) >= 0) {
            vertPriority.emplace(savings, groupHead);
        }
    }

    std::vector<bool> partitionedFlag(graph.num_vertices(), false);

    vertex_idx_t<Graph_t_in> maxCorseningNum = graph.num_vertices() - static_cast<vertex_idx_t<Graph_t_in>>(static_cast<double>(graph.num_vertices()) * params.geomDecay);

    vertex_idx_t<Graph_t_in> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long &vertSave = prioIter->first;
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
        std::vector<VertexType> part;
        part.reserve(1 + graph.out_degree(groupHead));
        part.emplace_back(groupHead);
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            part.emplace_back(groupFoot);
        }

        expansionMapOutput.emplace_back( std::move(part) );
        counter += static_cast<vertex_idx_t<Graph_t_in>>( graph.out_degree(groupHead) );
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFlag[groupHead] = true;
        for (const VertexType &groupFoot : graph.children(groupHead)) {
            partitionedFlag[groupFoot] = true;
        }
    }

    for (const VertexType &vert : graph.vertices()) {
        if (partitionedFlag[vert]) continue;
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}





template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::allParentsContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);

    const std::vector< vertex_idx_t<Graph_t_in> > vertexPoset = getBotPosetMap(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, VertexType> &lhs, const std::pair<long, VertexType> &rhs) {
        return (lhs.first > rhs.first)
                || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, VertexType>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType &groupFoot : graph.vertices()) {
        if (graph.in_degree(groupFoot) < 2) continue;

        bool shouldSkip = false;
        if constexpr (has_typed_vertices_v<Graph_t_in>) {
            for (const VertexType &groupHead : graph.parents(groupFoot)) {
                if (graph.vertex_type(groupHead) != graph.vertex_type(groupFoot)) {
                    shouldSkip = true;
                    break;
                }
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
        v_workw_t<Graph_t_in> combined_weight = graph.vertex_work_weight(groupFoot);
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            combined_weight += graph.vertex_work_weight(groupHead);
        }
        if (combined_weight > params.maxWeight) continue;

        v_workw_t<Graph_t_in> maxPath = topDist[groupFoot] + botDist[groupFoot] - graph.vertex_work_weight(groupFoot);
        for (const VertexType &par : graph.parents(groupFoot)) {
            maxPath = std::max(maxPath, topDist[par] + botDist[par] - graph.vertex_work_weight(par));
        }

        v_workw_t<Graph_t_in> maxParentDist = 0;
        v_workw_t<Graph_t_in> maxChildDist = 0;

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

        v_workw_t<Graph_t_in> newMaxPath = maxParentDist + maxChildDist + graph.vertex_work_weight(groupFoot);
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            newMaxPath += graph.vertex_work_weight(groupHead);
        }

        long savings = maxPath - newMaxPath;
        if (savings + static_cast<long>(params.leniency * static_cast<double>(maxPath)) >= 0) {
            vertPriority.emplace(savings, groupFoot);
        }
    }

    std::vector<bool> partitionedFlag(graph.num_vertices(), false);

    vertex_idx_t<Graph_t_in> maxCorseningNum = graph.num_vertices() - static_cast<vertex_idx_t<Graph_t_in>>(static_cast<double>(graph.num_vertices()) * params.geomDecay);

    vertex_idx_t<Graph_t_in> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long &vertSave = prioIter->first;
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
        std::vector<VertexType> part;
        part.reserve(1 + graph.in_degree(groupFoot));
        part.emplace_back(groupFoot);
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            part.emplace_back(groupHead);
        }

        expansionMapOutput.emplace_back( std::move(part) );
        counter += static_cast<vertex_idx_t<Graph_t_in>>( graph.in_degree(groupFoot) );
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFlag[groupFoot] = true;
        for (const VertexType &groupHead : graph.parents(groupFoot)) {
            partitionedFlag[groupHead] = true;
        }
    }

    for (const VertexType &vert : graph.vertices()) {
        if (partitionedFlag[vert]) continue;
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}












template<typename Graph_t_in, typename Graph_t_out>
std::vector<std::vector<vertex_idx_t<Graph_t_in>>> Sarkar<Graph_t_in, Graph_t_out>::generate_vertex_expansion_map(const Graph_t_in &dag_in, vertex_idx_t<Graph_t_in> &diff) {
    std::vector<std::vector<vertex_idx_t<Graph_t_in>>> expansionMap;

    // std::cout << "Mode: " << (int) params.mode << "\n";

    switch (params.mode)
    {
        case SarkarParams::Mode::LINES:
            {
                diff = singleContraction(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::FAN_IN_FULL:
            {
                diff = allParentsContraction(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::FAN_IN_PARTIAL:
            {
                diff = someParentsContraction(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::FAN_OUT_FULL:
            {
                diff = allChildrenContraction(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::FAN_OUT_PARTIAL:
            {
                diff = someChildrenContraction(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::LEVEL_EVEN:
            {
                diff = levelContraction(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::LEVEL_ODD:
            {
                diff = levelContraction(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::FAN_IN_BUFFER:
            {
                diff = in_buffer_merge(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::FAN_OUT_BUFFER:
            {
                diff = out_buffer_merge(params.commCost, dag_in, expansionMap);
            }
            break;

        case SarkarParams::Mode::HOMOGENEOUS_BUFFER:
            {
                diff = homogeneous_buffer_merge(params.commCost, dag_in, expansionMap);
            }
            break;
    }

    // std::cout << " Diff: " << diff << '\n';

    return expansionMap;
}




template<typename Graph_t_in, typename Graph_t_out>
std::vector<std::vector<vertex_idx_t<Graph_t_in>>> Sarkar<Graph_t_in, Graph_t_out>::generate_vertex_expansion_map(const Graph_t_in &dag_in) {
    vertex_idx_t<Graph_t_in> dummy;
    return generate_vertex_expansion_map(dag_in, dummy);
}



template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::someChildrenContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);

    const std::vector< vertex_idx_t<Graph_t_in> > vertexPoset = get_top_node_distance<Graph_t_in, vertex_idx_t<Graph_t_in>>(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, std::vector<VertexType>> &lhs, const std::pair<long, std::vector<VertexType>> &rhs) {
        return (lhs.first > rhs.first)
                || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, std::vector<VertexType>>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType &groupHead : graph.vertices()) {
        if (graph.out_degree(groupHead) < 2) continue;

        auto cmp_chld = [&topDist, &botDist](const VertexType &lhs, const VertexType &rhs) {
            return (topDist[lhs] < topDist[rhs])
                    || ((topDist[lhs] == topDist[rhs]) && (botDist[lhs] > botDist[rhs]))
                    || ((topDist[lhs] == topDist[rhs]) && (botDist[lhs] == botDist[rhs]) && (lhs < rhs));
        };
        std::set<VertexType, decltype(cmp_chld)> childrenPriority(cmp_chld);
        for (const VertexType &chld : graph.children(groupHead)) {
            if (vertexPoset[chld] == vertexPoset[groupHead] + 1) {
                childrenPriority.emplace(chld);
            }
        }
        if (childrenPriority.size() < 2) continue;

        std::vector< std::pair< typename std::set<VertexType, decltype(cmp_chld)>::const_iterator, typename std::set<VertexType, decltype(cmp_chld)>::const_iterator > > admissble_children_groups;
        for (auto chld_iter_start = childrenPriority.cbegin(); chld_iter_start != childrenPriority.cend();) {
            if constexpr (has_typed_vertices_v<Graph_t_in>) {
                if (graph.vertex_type(groupHead) != graph.vertex_type(*chld_iter_start)) {
                    ++chld_iter_start;
                    continue;
                }
            }

            const v_workw_t<Graph_t_in> t_dist = topDist[*chld_iter_start];
            const v_workw_t<Graph_t_in> b_dist = botDist[*chld_iter_start];
            auto chld_iter_end = chld_iter_start;
            while (chld_iter_end != childrenPriority.cend() && t_dist == topDist[*chld_iter_end] && b_dist == botDist[*chld_iter_end]) {
                if constexpr (has_typed_vertices_v<Graph_t_in>) {
                    if (graph.vertex_type(groupHead) != graph.vertex_type(*chld_iter_end)) {
                        break;
                    }
                }
                ++chld_iter_end;
            }

            admissble_children_groups.emplace_back(chld_iter_start, chld_iter_end);
            chld_iter_start = chld_iter_end;
        }

        std::vector<VertexType> contractionEnsemble;
        std::set<VertexType> contractionChildrenSet;
        contractionEnsemble.reserve(1 + graph.out_degree(groupHead));
        contractionEnsemble.emplace_back(groupHead);
        v_workw_t<Graph_t_in> added_weight = graph.vertex_work_weight(groupHead);

        for (std::size_t i = 0U; i < admissble_children_groups.size(); ++i) {
            const auto &first = admissble_children_groups[i].first;
            const auto &last = admissble_children_groups[i].second;

            for (auto it = first; it != last; ++it) {
                contractionEnsemble.emplace_back(*it);
                contractionChildrenSet.emplace(*it);
                added_weight += graph.vertex_work_weight(*it);
            }
            if (added_weight > params.maxWeight) break;

            v_workw_t<Graph_t_in> maxPath = 0;
            for (const VertexType &vert : contractionEnsemble) {
                maxPath = std::max(maxPath, topDist[vert] + botDist[vert] - graph.vertex_work_weight(vert));
            }

            v_workw_t<Graph_t_in> maxParentDist = 0;
            v_workw_t<Graph_t_in> maxChildDist = 0;

            for (const VertexType &vert : contractionEnsemble) {
                for (const VertexType &par : graph.parents(vert)) {
                    if (par == groupHead) continue;
                    maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
                }
            }

            for (const VertexType &chld : graph.children(groupHead)) {
                if (contractionChildrenSet.find(chld) == contractionChildrenSet.end()) {
                    maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
                }
            }

            for (std::size_t j = 1; j < contractionEnsemble.size(); j++) {
                for (const VertexType &chld : graph.children(contractionEnsemble[j])) {
                    maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
                }
            }

            v_workw_t<Graph_t_in> newMaxPath = maxParentDist + maxChildDist;
            for (const VertexType &vert : contractionEnsemble) {
                newMaxPath += graph.vertex_work_weight(vert);
            }

            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);
            if (savings + static_cast<long>(params.leniency * static_cast<double>(maxPath)) >= 0) {
                vertPriority.emplace(savings, contractionEnsemble);
            }
        }
    }

    std::vector<bool> partitionedFlag(graph.num_vertices(), false);
    std::vector<bool> partitionedHeadFlag(graph.num_vertices(), false);

    vertex_idx_t<Graph_t_in> maxCorseningNum = graph.num_vertices() - static_cast<vertex_idx_t<Graph_t_in>>(static_cast<double>(graph.num_vertices()) * params.geomDecay);

    vertex_idx_t<Graph_t_in> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long &vertSave = prioIter->first;
        const VertexType &groupHead = prioIter->second.front();
        const std::vector<VertexType> &contractionEnsemble = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) break;

        // Check whether we can glue
        bool shouldSkip = false;
        for (const VertexType &vert : contractionEnsemble) {
            if (partitionedFlag[vert]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) continue;

        for (const VertexType &chld : graph.children(groupHead)) {
            if ((std::find(contractionEnsemble.cbegin(), contractionEnsemble.cend(), chld) == contractionEnsemble.cend()) && (vertexPoset[chld] == vertexPoset[groupHead] + 1)) {
                if ((partitionedFlag[chld]) && (!partitionedHeadFlag[chld])) {
                    shouldSkip = true;
                    break;
                }
            }
        }
        if (shouldSkip) continue;

        // Adding to partition
        expansionMapOutput.emplace_back(contractionEnsemble);
        counter += static_cast<vertex_idx_t<Graph_t_in>>( contractionEnsemble.size() ) - 1;
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedHeadFlag[groupHead] = true;
        for (const VertexType &vert : contractionEnsemble) {
            partitionedFlag[vert] = true;
        }
    }

    for (const VertexType &vert : graph.vertices()) {
        if (partitionedFlag[vert]) continue;
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}









template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::someParentsContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);

    const std::vector< vertex_idx_t<Graph_t_in> > vertexPoset = getBotPosetMap(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, std::vector<VertexType>> &lhs, const std::pair<long, std::vector<VertexType>> &rhs) {
        return (lhs.first > rhs.first)
                || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, std::vector<VertexType>>, decltype(cmp)> vertPriority(cmp);

    for (const VertexType &groupFoot : graph.vertices()) {
        if (graph.in_degree(groupFoot) < 2) continue;

        auto cmp_par = [&topDist, &botDist](const VertexType &lhs, const VertexType &rhs) {
            return (botDist[lhs] < botDist[rhs])
                    || ((botDist[lhs] == botDist[rhs]) && (topDist[lhs] > topDist[rhs]))
                    || ((botDist[lhs] == botDist[rhs]) && (topDist[lhs] == topDist[rhs]) && (lhs < rhs));
        };
        std::set<VertexType, decltype(cmp_par)> parentsPriority(cmp_par);
        for (const VertexType &par : graph.parents(groupFoot)) {
            if (vertexPoset[par] + 1 == vertexPoset[groupFoot]) {
                parentsPriority.emplace(par);
            }
        }
        if (parentsPriority.size() < 2) continue;

        std::vector< std::pair< typename std::set<VertexType, decltype(cmp_par)>::const_iterator, typename std::set<VertexType, decltype(cmp_par)>::const_iterator > > admissble_parent_groups;
        for (auto par_iter_start = parentsPriority.cbegin(); par_iter_start != parentsPriority.cend();) {
            if constexpr (has_typed_vertices_v<Graph_t_in>) {
                if (graph.vertex_type(groupFoot) != graph.vertex_type(*par_iter_start)) {
                    ++par_iter_start;
                    continue;
                }
            }

            const v_workw_t<Graph_t_in> t_dist = topDist[*par_iter_start];
            const v_workw_t<Graph_t_in> b_dist = botDist[*par_iter_start];
            auto par_iter_end = par_iter_start;
            while (par_iter_end != parentsPriority.cend() && t_dist == topDist[*par_iter_end] && b_dist == botDist[*par_iter_end]) {
                if constexpr (has_typed_vertices_v<Graph_t_in>) {
                    if (graph.vertex_type(groupFoot) != graph.vertex_type(*par_iter_end)) {
                        break;
                    }
                }
                ++par_iter_end;
            }

            admissble_parent_groups.emplace_back(par_iter_start, par_iter_end);
            par_iter_start = par_iter_end;
        }

        std::vector<VertexType> contractionEnsemble;
        std::set<VertexType> contractionParentsSet;
        contractionEnsemble.reserve(1 + graph.in_degree(groupFoot));
        contractionEnsemble.emplace_back(groupFoot);
        v_workw_t<Graph_t_in> added_weight = graph.vertex_work_weight(groupFoot);

        for (std::size_t i = 0U; i < admissble_parent_groups.size(); ++i) {
            const auto &first = admissble_parent_groups[i].first;
            const auto &last = admissble_parent_groups[i].second;

            for (auto it = first; it != last; ++it) {
                contractionEnsemble.emplace_back(*it);
                contractionParentsSet.emplace(*it);
                added_weight += graph.vertex_work_weight(*it);
            }
            if (added_weight > params.maxWeight) break;

            v_workw_t<Graph_t_in> maxPath = 0;
            for (const VertexType &vert : contractionEnsemble) {
                maxPath = std::max(maxPath, topDist[vert] + botDist[vert] - graph.vertex_work_weight(vert));
            }
            
            v_workw_t<Graph_t_in> maxParentDist = 0;
            v_workw_t<Graph_t_in> maxChildDist = 0;       

            for (const VertexType &vert : contractionEnsemble) {
                for (const VertexType &chld : graph.children(vert)) {
                    if (chld == groupFoot) continue;
                    maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
                }
            }

            for (const VertexType &par : graph.parents(groupFoot)) {
                if (contractionParentsSet.find(par) == contractionParentsSet.end()) {
                    maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
                }
            }

            for (std::size_t j = 1; j < contractionEnsemble.size(); j++) {
                for (const VertexType &par : graph.parents(contractionEnsemble[j])) {
                    maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
                }
            }

            v_workw_t<Graph_t_in> newMaxPath = maxParentDist + maxChildDist;
            for (const VertexType &vert : contractionEnsemble) {
                newMaxPath += graph.vertex_work_weight(vert);
            }

            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);
            if (savings + static_cast<long>(params.leniency * static_cast<double>(maxPath)) >= 0) {
                vertPriority.emplace(savings, contractionEnsemble);
            }
        }
    }

    std::vector<bool> partitionedFlag(graph.num_vertices(), false);
    std::vector<bool> partitionedFootFlag(graph.num_vertices(), false);

    vertex_idx_t<Graph_t_in> maxCorseningNum = graph.num_vertices() - static_cast<vertex_idx_t<Graph_t_in>>(static_cast<double>(graph.num_vertices()) * params.geomDecay);

    vertex_idx_t<Graph_t_in> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.begin(); prioIter != vertPriority.end(); prioIter++) {
        const long &vertSave = prioIter->first;
        const VertexType &groupFoot = prioIter->second.front();
        const std::vector<VertexType> &contractionEnsemble = prioIter->second;

        // Iterations halt
        if (vertSave < minSave) break;

        // Check whether we can glue
        bool shouldSkip = false;
        for (const VertexType &vert : contractionEnsemble) {
            if (partitionedFlag[vert]) {
                shouldSkip = true;
                break;
            }
        }
        if (shouldSkip) continue;

        for (const VertexType &par : graph.parents(groupFoot)) {
            if ((std::find(contractionEnsemble.cbegin(), contractionEnsemble.cend(), par) == contractionEnsemble.cend()) && (vertexPoset[par] + 1 == vertexPoset[groupFoot])) {
                if ((partitionedFlag[par]) && (!partitionedFootFlag[par])) {
                    shouldSkip = true;
                    break;
                }
            }
        }
        if (shouldSkip) continue;

        // Adding to partition
        expansionMapOutput.emplace_back(contractionEnsemble);
        counter += static_cast<vertex_idx_t<Graph_t_in>>( contractionEnsemble.size() ) - 1;
        if (counter > maxCorseningNum) {
            minSave = vertSave;
        }
        partitionedFootFlag[groupFoot] = true;
        for (const VertexType &vert : contractionEnsemble) {
            partitionedFlag[vert] = true;
        }
    }

    for (const VertexType &vert : graph.vertices()) {
        if (partitionedFlag[vert]) continue;
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}












template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::levelContraction(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);

    const std::vector< vertex_idx_t<Graph_t_in> > vertexPoset = params.useTopPoset ? get_top_node_distance<Graph_t_in, vertex_idx_t<Graph_t_in>>(graph) : getBotPosetMap(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    auto cmp = [](const std::pair<long, std::vector<VertexType>> &lhs, const std::pair<long, std::vector<VertexType>> &rhs) {
        return (lhs.first > rhs.first)
                || ((lhs.first == rhs.first) && (lhs.second < rhs.second));
    };
    std::set<std::pair<long, std::vector<VertexType>>, decltype(cmp)> vertPriority(cmp);

    const vertex_idx_t<Graph_t_in> minLevel = *std::min_element(vertexPoset.cbegin(), vertexPoset.cend());
    const vertex_idx_t<Graph_t_in> maxLevel = *std::max_element(vertexPoset.cbegin(), vertexPoset.cend());

    const vertex_idx_t<Graph_t_in> parity = params.mode == SarkarParams::Mode::LEVEL_EVEN? 0 : 1;

    std::vector<std::vector<vertex_idx_t<Graph_t_in>>> levels(maxLevel - minLevel + 1);
    for (const VertexType &vert : graph.vertices()) {
        levels[ vertexPoset[vert] - minLevel ].emplace_back(vert);
    }

    for (vertex_idx_t<Graph_t_in> headLevel = minLevel + parity; headLevel < maxLevel; headLevel += 2) {
        const vertex_idx_t<Graph_t_in> footLevel = headLevel + 1;
        
        const std::vector<vertex_idx_t<Graph_t_in>> &headVertices = levels[ headLevel - minLevel ];
        const std::vector<vertex_idx_t<Graph_t_in>> &footVertices = levels[ footLevel - minLevel ];

        Union_Find_Universe<VertexType, std::size_t, v_workw_t<Graph_t_in>, v_memw_t<Graph_t_in>> uf;
        for (const VertexType &vert : headVertices) {
            uf.add_object(vert, graph.vertex_work_weight(vert));
        }
        for (const VertexType &vert : footVertices) {
            uf.add_object(vert, graph.vertex_work_weight(vert));
        }

        for (const VertexType &srcVert : headVertices) {
            for (const VertexType &tgtVert : graph.children(srcVert)) {
                if (vertexPoset[tgtVert] != footLevel) continue;
                
                if constexpr (has_typed_vertices_v<Graph_t_in>) {
                    if (graph.vertex_type(srcVert) != graph.vertex_type(tgtVert)) continue;
                }

                uf.join_by_name(srcVert, tgtVert);
            }
        }

        std::vector<std::vector<VertexType>> components = uf.get_connected_components();
        for (std::vector<VertexType> &comp : components) {
            if (comp.size() < 2) continue;
            if (uf.get_weight_of_component_by_name(comp.at(0)) > params.maxWeight) continue; 

            std::sort(comp.begin(), comp.end());

            v_workw_t<Graph_t_in> maxPath = std::numeric_limits<v_workw_t<Graph_t_in>>::lowest();
            for (const VertexType &vert : comp) {
                maxPath = std::max(maxPath, topDist[vert] + botDist[vert] - graph.vertex_work_weight(vert));
            }

            v_workw_t<Graph_t_in> maxParentDist = 0;
            for (const VertexType &vert : comp) {
                for (const VertexType &par : graph.parents(vert)) {
                    if (std::binary_search(comp.cbegin(), comp.cend(), par)) continue;

                    maxParentDist = std::max(maxParentDist, topDist[par] + commCost);
                }
            }

            v_workw_t<Graph_t_in> maxChildDist = 0;
            for (const VertexType &vert : comp) {
                for (const VertexType &chld : graph.children(vert)) {
                    if (std::binary_search(comp.cbegin(), comp.cend(), chld)) continue;

                    maxChildDist = std::max(maxChildDist, botDist[chld] + commCost);
                }
            }


            v_workw_t<Graph_t_in> newMaxPath = maxParentDist + maxChildDist;
            for (const VertexType &vert : comp) {
                newMaxPath += graph.vertex_work_weight(vert);
            }

            long savings = static_cast<long>(maxPath) - static_cast<long>(newMaxPath);
    
            if (savings + static_cast<long>(params.leniency * static_cast<double>(maxPath)) >= 0) {
                vertPriority.emplace(savings, comp);
            }

        }




    }

    std::vector<bool> partitionedFlag(graph.num_vertices(), false);

    vertex_idx_t<Graph_t_in> maxCorseningNum = graph.num_vertices() - static_cast< vertex_idx_t<Graph_t_in> >(static_cast<double>(graph.num_vertices()) * params.geomDecay);


    vertex_idx_t<Graph_t_in> counter = 0;
    long minSave = std::numeric_limits<long>::lowest();
    for (auto prioIter = vertPriority.cbegin(); prioIter != vertPriority.cend(); prioIter++) {
        const long &compSave = prioIter->first;
        const std::vector<VertexType> &comp = prioIter->second;

        // Iterations halt
        if (compSave < minSave) break;

        // Check whether we can glue
        bool shouldSkipHead = false;
        bool shouldSkipFoot = false;
        for (const VertexType &vert : comp) {
            if (((vertexPoset[vert] - minLevel - parity) % 2) == 0) {   // head vertex
                for (const VertexType &chld : graph.children(vert)) {
                    if ((vertexPoset[chld] == vertexPoset[vert] + 1) && partitionedFlag[chld]) {
                        shouldSkipHead = true;
                    }
                }
            } else {    // foot vertex
                for (const VertexType &par : graph.parents(vert)) {
                    if ((vertexPoset[par] + 1 == vertexPoset[vert]) && partitionedFlag[par]) {
                        shouldSkipFoot = true;
                    }
                }
            }
        }

        if (shouldSkipHead && shouldSkipFoot) continue;

        // Adding to partition
        expansionMapOutput.emplace_back(comp);
        counter += static_cast<vertex_idx_t<Graph_t_in>>( comp.size() - 1 );
        if (counter > maxCorseningNum) {
            minSave = compSave;
        }

        for (const VertexType &vert : comp) {
            partitionedFlag[vert] = true;
        }
    }

    expansionMapOutput.reserve(graph.num_vertices() - counter);
    for (const VertexType &vert : graph.vertices()) {
        if (partitionedFlag[vert]) continue;
        
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

template<typename Graph_t_in, typename Graph_t_out>
std::vector<std::size_t> Sarkar<Graph_t_in, Graph_t_out>::computeNodeHashes(const Graph_t_in &graph, const std::vector< vertex_idx_t<Graph_t_in> > &vertexPoset, const std::vector< v_workw_t<Graph_t_in> > &dist) const {
    using VertexType = vertex_idx_t<Graph_t_in>;

    std::vector<std::size_t> hashes(graph.num_vertices());
    for (const VertexType &vert : graph.vertices()) {
        std::size_t &hash = hashes[vert];
        hash = std::hash<v_workw_t<Graph_t_in>>{}(graph.vertex_work_weight(vert));
        hash_combine(hash, vertexPoset[vert]);
        hash_combine(hash, dist[vert]);
        if constexpr (has_typed_vertices_v<Graph_t_in>) {
            hash_combine(hash, graph.vertex_type(vert));
        }
    }

    return hashes;
}

template<typename Graph_t_in, typename Graph_t_out>
std::vector<std::size_t> Sarkar<Graph_t_in, Graph_t_out>::homogeneousMerge(const std::size_t number, const std::size_t minSize, const std::size_t maxSize) const {
    assert(minSize <= maxSize);
    assert(number > 0);

    std::size_t bestDiv = 1U;
    for (std::size_t div : divisorsList(number)) {
        if (div > maxSize) continue;

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
    for (std::size_t bins = std::max( number / maxSize, static_cast<std::size_t>(2U)); bins <= number / minSize; ++bins) {
        if (number % bins == 0U && number != bins) {
            return std::vector<std::size_t>(bins, number / bins);
        }

        std::size_t score = std::min( divisorsList(number / bins).size(), divisorsList((number / bins) + 1).size() );
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

template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::homogeneous_buffer_merge(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);

    const std::vector< vertex_idx_t<Graph_t_in> > vertexTopPoset = get_top_node_distance<Graph_t_in, vertex_idx_t<Graph_t_in>>(graph);
    const std::vector< vertex_idx_t<Graph_t_in> > vertexBotPoset = getBotPosetMap(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    std::vector<std::size_t> hashValues = computeNodeHashes(graph, vertexTopPoset, topDist);
    std::vector<std::size_t> hashValuesWithParents = hashValues;
    for (const VertexType &par : graph.vertices()) {
        for (const VertexType &chld : graph.children(par)) {
            hash_combine(hashValuesWithParents[chld], hashValues[par]);
        }        
    }
    hashValues = computeNodeHashes(graph, vertexBotPoset, botDist);
    std::vector<std::size_t> hashValuesWithChildren = hashValues;
    for (const VertexType &chld : graph.vertices()) {
        for (const VertexType &par : graph.parents(chld)) {
            hash_combine(hashValuesWithChildren[par], hashValues[chld]);
        }        
    }
    for (const VertexType &vert : graph.vertices()) {
        hash_combine(hashValuesWithParents[vert], hashValuesWithChildren[vert]);
    }
    const std::vector<std::size_t> &hashValuesCombined = hashValuesWithParents;

    std::unordered_map<std::size_t, std::set<VertexType>> orbits;
    for (const VertexType &vert : graph.vertices()) {
        if (graph.vertex_work_weight(vert) > params.smallWeightThreshold) continue;

        const std::size_t hash = hashValuesCombined[vert];
        auto found_iter = orbits.find(hash); 
        if (found_iter == orbits.end()) {
            orbits.emplace(std::piecewise_construct, std::forward_as_tuple(hash), std::forward_as_tuple(std::initializer_list< vertex_idx_t<Graph_t_in> >{vert}));
        } else {
            found_iter->second.emplace(vert);
        }
    }

    vertex_idx_t<Graph_t_in> counter = 0;
    std::vector<bool> partitionedFlag(graph.num_vertices(), false);
    
    for (const VertexType &vert : graph.vertices()) {
        if (graph.vertex_work_weight(vert) > params.smallWeightThreshold) continue;
        if (partitionedFlag[vert]) continue;

        const std::set<VertexType> &orb = orbits.at(hashValuesCombined[vert]);
        if (orb.size() <= 1U) continue;

        std::set<VertexType> parents;
        for (const VertexType &par : graph.parents(vert)) {
            parents.emplace(par);
        }
        std::set<VertexType> children;
        for (const VertexType &chld : graph.children(vert)) {
            children.emplace(chld);
        }

        std::set<VertexType> secureOrb;
        for (const VertexType &vertCandidate : orb) {
            if (vertexTopPoset[vertCandidate] != vertexTopPoset[vert]) continue;
            if (vertexBotPoset[vertCandidate] != vertexBotPoset[vert]) continue;
            if (graph.vertex_work_weight(vertCandidate) != graph.vertex_work_weight(vert)) continue;
            if (topDist[vertCandidate] != topDist[vert]) continue;
            if (botDist[vertCandidate] != botDist[vert]) continue;
            if constexpr (has_typed_vertices_v<Graph_t_in>) {
                if (graph.vertex_type(vertCandidate) != graph.vertex_type(vert)) continue;
            }

            std::set<VertexType> candidateParents;
            for (const VertexType &par : graph.parents(vertCandidate)) {
                candidateParents.emplace(par);
            }
            if (candidateParents != parents) continue;

            std::set<VertexType> candidateChildren;
            for (const VertexType &chld : graph.children(vertCandidate)) {
                candidateChildren.emplace(chld);
            }
            if (candidateChildren != children) continue;

            secureOrb.emplace(vertCandidate);
        }
        if (secureOrb.size() <= 1U) continue;

        const v_workw_t<Graph_t_in> desiredVerticesInGroup = graph.vertex_work_weight(vert) == 0 ? std::numeric_limits<v_workw_t<Graph_t_in>>::lowest() : params.smallWeightThreshold / graph.vertex_work_weight(vert);
        const v_workw_t<Graph_t_in> maxVerticesInGroup = graph.vertex_work_weight(vert) == 0 ? std::numeric_limits<v_workw_t<Graph_t_in>>::max() : params.maxWeight / graph.vertex_work_weight(vert);

        const std::size_t minDesiredSize = desiredVerticesInGroup < 2 ? 2U : static_cast<std::size_t>(desiredVerticesInGroup);
        const std::size_t maxDesiredSize = std::max(minDesiredSize, std::min(minDesiredSize * 2U, static_cast<std::size_t>(maxVerticesInGroup)));

        std::vector<std::size_t> groups = homogeneousMerge(secureOrb.size(), minDesiredSize, maxDesiredSize);

        auto secureOrbIter = secureOrb.begin();
        for (std::size_t groupSize : groups) {
            std::vector<VertexType> cluster;
            for (std::size_t i = 0; i < groupSize; ++i) {
                cluster.emplace_back(*secureOrbIter);
                ++secureOrbIter;
            }
            expansionMapOutput.emplace_back( std::move(cluster) );
            counter += static_cast<VertexType>(groupSize) - 1;
        }

        for (const VertexType &touchedVertex : secureOrb) {
            partitionedFlag[touchedVertex] = true;
        }
    }

    for (const VertexType &vert : graph.vertices()) {
        if (partitionedFlag[vert]) continue;
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}




template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::out_buffer_merge(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);

    const std::vector< vertex_idx_t<Graph_t_in> > vertexTopPoset = get_top_node_distance<Graph_t_in, vertex_idx_t<Graph_t_in>>(graph);
    const std::vector< vertex_idx_t<Graph_t_in> > vertexBotPoset = getBotPosetMap(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    std::vector<std::size_t> hashValues = computeNodeHashes(graph, vertexTopPoset, topDist);
    std::vector<std::size_t> hashValuesWithParents = hashValues;
    for (const VertexType &par : graph.vertices()) {
        for (const VertexType &chld : graph.children(par)) {
            hash_combine(hashValuesWithParents[chld], hashValues[par]);
        }        
    }
    
    const std::vector<std::size_t> &hashValuesCombined = hashValuesWithParents;

    std::unordered_map<std::size_t, std::set<VertexType>> orbits;
    for (const VertexType &vert : graph.vertices()) {
        if (graph.vertex_work_weight(vert) > params.smallWeightThreshold) continue;

        const std::size_t hash = hashValuesCombined[vert];
        auto found_iter = orbits.find(hash); 
        if (found_iter == orbits.end()) {
            orbits.emplace(std::piecewise_construct, std::forward_as_tuple(hash), std::forward_as_tuple(std::initializer_list< vertex_idx_t<Graph_t_in> >{vert}));
        } else {
            found_iter->second.emplace(vert);
        }
    }

    vertex_idx_t<Graph_t_in> counter = 0;
    std::vector<bool> partitionedFlag(graph.num_vertices(), false);
    
    for (const VertexType &vert : graph.vertices()) {
        if (graph.vertex_work_weight(vert) > params.smallWeightThreshold) continue;
        if (partitionedFlag[vert]) continue;

        const std::set<VertexType> &orb = orbits.at(hashValuesCombined[vert]);
        if (orb.size() <= 1U) continue;

        std::set<VertexType> parents;
        for (const VertexType &par : graph.parents(vert)) {
            parents.emplace(par);
        }

        std::set<VertexType> secureOrb;
        for (const VertexType &vertCandidate : orb) {
            if (vertexTopPoset[vertCandidate] != vertexTopPoset[vert]) continue;
            if (vertexBotPoset[vertCandidate] != vertexBotPoset[vert]) continue;
            if (graph.vertex_work_weight(vertCandidate) != graph.vertex_work_weight(vert)) continue;
            if (topDist[vertCandidate] != topDist[vert]) continue;
            if (botDist[vertCandidate] != botDist[vert]) continue;
            if constexpr (has_typed_vertices_v<Graph_t_in>) {
                if (graph.vertex_type(vertCandidate) != graph.vertex_type(vert)) continue;
            }

            std::set<VertexType> candidateParents;
            for (const VertexType &par : graph.parents(vertCandidate)) {
                candidateParents.emplace(par);
            }
            if (candidateParents != parents) continue;

            secureOrb.emplace(vertCandidate);
        }
        if (secureOrb.size() <= 1U) continue;

        const v_workw_t<Graph_t_in> desiredVerticesInGroup = graph.vertex_work_weight(vert) == 0 ? std::numeric_limits<v_workw_t<Graph_t_in>>::lowest() : params.smallWeightThreshold / graph.vertex_work_weight(vert);
        const v_workw_t<Graph_t_in> maxVerticesInGroup = graph.vertex_work_weight(vert) == 0 ? std::numeric_limits<v_workw_t<Graph_t_in>>::max() : params.maxWeight / graph.vertex_work_weight(vert);

        const std::size_t minDesiredSize = desiredVerticesInGroup < 2 ? 2U : static_cast<std::size_t>(desiredVerticesInGroup);
        const std::size_t maxDesiredSize = std::max(minDesiredSize, std::min(minDesiredSize * 2U, static_cast<std::size_t>(maxVerticesInGroup)));

        // TODO: do this smarter depending on the connectivity of the children
        std::vector<std::size_t> groups = homogeneousMerge(secureOrb.size(), minDesiredSize, maxDesiredSize);

        auto secureOrbIter = secureOrb.begin();
        for (std::size_t groupSize : groups) {
            std::vector<VertexType> cluster;
            for (std::size_t i = 0; i < groupSize; ++i) {
                cluster.emplace_back(*secureOrbIter);
                ++secureOrbIter;
            }
            expansionMapOutput.emplace_back( std::move(cluster) );
            counter += static_cast<VertexType>(groupSize) - 1;
        }

        for (const VertexType &touchedVertex : secureOrb) {
            partitionedFlag[touchedVertex] = true;
        }
    }

    for (const VertexType &vert : graph.vertices()) {
        if (partitionedFlag[vert]) continue;
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}





template<typename Graph_t_in, typename Graph_t_out>
vertex_idx_t<Graph_t_in> Sarkar<Graph_t_in, Graph_t_out>::in_buffer_merge(v_workw_t<Graph_t_in> commCost, const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &expansionMapOutput) const {
    using VertexType = vertex_idx_t<Graph_t_in>;
    assert(expansionMapOutput.size() == 0);
    
    const std::vector< vertex_idx_t<Graph_t_in> > vertexTopPoset = get_top_node_distance<Graph_t_in, vertex_idx_t<Graph_t_in>>(graph);
    const std::vector< vertex_idx_t<Graph_t_in> > vertexBotPoset = getBotPosetMap(graph);
    const std::vector< v_workw_t<Graph_t_in> > topDist = getTopDistance(commCost, graph);
    const std::vector< v_workw_t<Graph_t_in> > botDist = getBotDistance(commCost, graph);

    std::vector<std::size_t> hashValues = computeNodeHashes(graph, vertexBotPoset, botDist);
    std::vector<std::size_t> hashValuesWithChildren = hashValues;
    for (const VertexType &chld : graph.vertices()) {
        for (const VertexType &par : graph.parents(chld)) {
            hash_combine(hashValuesWithChildren[par], hashValues[chld]);
        }        
    }
    const std::vector<std::size_t> &hashValuesCombined = hashValuesWithChildren;

    std::unordered_map<std::size_t, std::set<VertexType>> orbits;
    for (const VertexType &vert : graph.vertices()) {
        if (graph.vertex_work_weight(vert) > params.smallWeightThreshold) continue;

        const std::size_t hash = hashValuesCombined[vert];
        auto found_iter = orbits.find(hash); 
        if (found_iter == orbits.end()) {
            orbits.emplace(std::piecewise_construct, std::forward_as_tuple(hash), std::forward_as_tuple(std::initializer_list< vertex_idx_t<Graph_t_in> >{vert}));
        } else {
            found_iter->second.emplace(vert);
        }
    }

    vertex_idx_t<Graph_t_in> counter = 0;
    std::vector<bool> partitionedFlag(graph.num_vertices(), false);
    
    for (const VertexType &vert : graph.vertices()) {
        if (graph.vertex_work_weight(vert) > params.smallWeightThreshold) continue;
        if (partitionedFlag[vert]) continue;

        const std::set<VertexType> &orb = orbits.at(hashValuesCombined[vert]);
        if (orb.size() <= 1U) continue;

        std::set<VertexType> children;
        for (const VertexType &chld : graph.children(vert)) {
            children.emplace(chld);
        }

        std::set<VertexType> secureOrb;
        for (const VertexType &vertCandidate : orb) {
            if (vertexTopPoset[vertCandidate] != vertexTopPoset[vert]) continue;
            if (vertexBotPoset[vertCandidate] != vertexBotPoset[vert]) continue;
            if (graph.vertex_work_weight(vertCandidate) != graph.vertex_work_weight(vert)) continue;
            if (topDist[vertCandidate] != topDist[vert]) continue;
            if (botDist[vertCandidate] != botDist[vert]) continue;
            if constexpr (has_typed_vertices_v<Graph_t_in>) {
                if (graph.vertex_type(vertCandidate) != graph.vertex_type(vert)) continue;
            }

            std::set<VertexType> candidateChildren;
            for (const VertexType &chld : graph.children(vertCandidate)) {
                candidateChildren.emplace(chld);
            }
            if (candidateChildren != children) continue;

            secureOrb.emplace(vertCandidate);
        }
        if (secureOrb.size() <= 1U) continue;

        const v_workw_t<Graph_t_in> desiredVerticesInGroup = graph.vertex_work_weight(vert) == 0 ? std::numeric_limits<v_workw_t<Graph_t_in>>::lowest() : params.smallWeightThreshold / graph.vertex_work_weight(vert);
        const v_workw_t<Graph_t_in> maxVerticesInGroup = graph.vertex_work_weight(vert) == 0 ? std::numeric_limits<v_workw_t<Graph_t_in>>::max() : params.maxWeight / graph.vertex_work_weight(vert);

        const std::size_t minDesiredSize = desiredVerticesInGroup < 2 ? 2U : static_cast<std::size_t>(desiredVerticesInGroup);
        const std::size_t maxDesiredSize = std::max(minDesiredSize, std::min(minDesiredSize * 2U, static_cast<std::size_t>(maxVerticesInGroup)));

        // TODO: do this smarter depending on the connectivity of the parents
        std::vector<std::size_t> groups = homogeneousMerge(secureOrb.size(), minDesiredSize, maxDesiredSize);

        auto secureOrbIter = secureOrb.begin();
        for (std::size_t groupSize : groups) {
            std::vector<VertexType> cluster;
            for (std::size_t i = 0; i < groupSize; ++i) {
                cluster.emplace_back(*secureOrbIter);
                ++secureOrbIter;
            }
            expansionMapOutput.emplace_back( std::move(cluster) );
            counter += static_cast<VertexType>(groupSize) - 1;
        }

        for (const VertexType &touchedVertex : secureOrb) {
            partitionedFlag[touchedVertex] = true;
        }
    }

    for (const VertexType &vert : graph.vertices()) {
        if (partitionedFlag[vert]) continue;
        expansionMapOutput.emplace_back(std::initializer_list<VertexType>{vert});
    }

    return counter;
}

} // end namespace osp