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

#pragma once
#include <limits>

#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util_parallel.hpp"

namespace osp {

/**
 * @brief Acyclic graph contractor that contracts groups of nodes with only one vertex with incoming/outgoing edges
 * (from outside the group)
 *
 */
template <typename GraphTIn, typename GraphTOut, bool useArchitectureMemoryContraints = false>
class FunnelBfs : public CoarserGenExpansionMap<GraphTIn, GraphTOut> {
  public:
    /**
     * @brief Parameters for Funnel coarsener
     *
     */
    struct FunnelBfsParameters {
        bool funnelIncoming_;

        bool useApproxTransitiveReduction_;

        VWorkwT<Graph_t_in> maxWorkWeight_;
        VMemwT<Graph_t_in> maxMemoryWeight_;

        unsigned maxDepth_;

        FunnelBfsParameters(vWorkwT_<Graph_t_in> max_work_weight_ = std::numeric_limits<VWorkwT<Graph_t_in>>::max(),
                            VMemwT<Graph_t_in> max_memory_weight_ = std::numeric_limits<VMemwT<Graph_t_in>>::max(),
                            unsigned max_depth_ = std::numeric_limits<unsigned>::max(),
                            bool funnel_incoming_ = true,
                            bool use_approx_transitive_reduction_ = true)
            : funnel_incoming(funnel_incoming_),
              UseApproxTransitiveReduction(use_approx_transitive_reduction_),
              MaxWorkWeight(max_work_weight_),
              MaxMemoryWeight(max_memory_weight_),
              MaxDepth(max_depth_) {};

        ~FunnelBfsParameters() = default;
    };

    FunnelBfs(FunnelBfsParameters parameters = FunnelBfsParameters()) : parameters_(parameters) {}

    virtual ~FunnelBfs() = default;

    virtual std::vector<std::vector<vertex_idx_t<Graph_t_in>>> generate_vertex_expansion_map(const GraphTIn &graph) override {
        if constexpr (useArchitectureMemoryContraints) {
            if (max_memory_per_vertex_type.size() < graph.NumVertexTypes()) {
                throw std::runtime_error("FunnelBfs: max_memory_per_vertex_type has insufficient size.");
            }
        }

        std::vector<std::vector<vertex_idx_t<Graph_t_in>>> partition;

        if (parameters_.funnelIncoming_) {
            run_in_contraction(graph, partition);
        } else {
            run_out_contraction(graph, partition);
        }

        return partition;
    }

    std::string getCoarserName() const override { return "FunnelBfs"; }

    std::vector<VMemwT<Graph_t_in>> &GetMaxMemoryPerVertexType() { return max_memory_per_vertex_type; }

  private:
    FunnelBfsParameters parameters_;

    std::vector<VMemwT<Graph_t_in>> maxMemoryPerVertexType_;

    void RunInContraction(const GraphTIn &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &partition) {
        using vertex_idx_t = vertex_idx_t<Graph_t_in>;

        const std::unordered_set<edge_desc_t<Graph_t_in>> edgeMask = parameters.use_approx_transitive_reduction
                                                                         ? long_edges_in_triangles_parallel(graph)
                                                                         : std::unordered_set<edge_desc_t<Graph_t_in>>();

        std::vector<bool> visited(graph.NumVertices(), false);

        const std::vector<vertex_idx_t> topOrder = GetTopOrder(graph);

        for (auto revTopIt = top_order.rbegin(); revTopIt != top_order.crend(); rev_top_it++) {
            const vertex_idx_t &bottomNode = *rev_top_it;

            if (visited[bottom_node]) {
                continue;
            }

            VWorkwT<Graph_t_in> workWeightOfGroup = 0;
            VMemwT<Graph_t_in> memoryWeightOfGroup = 0;

            std::unordered_map<vertex_idx_t, vertex_idx_t> childrenNotInGroup;
            std::vector<vertex_idx_t> group;

            std::deque<vertex_idx_t> vertexProcessingFifo({bottom_node});
            std::deque<vertex_idx_t> nextVertexProcessingFifo;

            unsigned depthCounter = 0;

            while ((not vertex_processing_fifo.empty()) || (not next_vertex_processing_fifo.empty())) {
                if (vertexProcessingFifo.empty()) {
                    vertexProcessingFifo = next_vertex_processing_fifo;
                    nextVertexProcessingFifo.clear();
                    depthCounter++;
                    if (depthCounter > parameters_.maxDepth_) {
                        break;
                    }
                }

                vertex_idx_t activeNode = vertex_processing_fifo.front();
                vertexProcessingFifo.pop_front();

                if (graph.VertexType(active_node) != graph.VertexType(bottom_node)) {
                    continue;
                }

                if (workWeightOfGroup + graph.VertexWorkWeight(active_node) > parameters_.maxWorkWeight_) {
                    continue;
                }

                if (memoryWeightOfGroup + graph.VertexMemWeight(active_node) > parameters_.maxMemoryWeight_) {
                    continue;
                }

                if constexpr (useArchitectureMemoryContraints) {
                    if (memory_weight_of_group + graph.VertexMemWeight(active_node)
                        > max_memory_per_vertex_type[graph.VertexType(bottom_node)]) {
                        continue;
                    }
                }

                group.emplace_back(active_node);
                workWeightOfGroup += graph.VertexWorkWeight(active_node);
                memoryWeightOfGroup += graph.VertexMemWeight(active_node);

                for (const auto &in_edge : InEdges(active_node, graph)) {
                    if (parameters.use_approx_transitive_reduction && (edge_mask.find(in_edge) != edge_mask.cend())) {
                        continue;
                    }

                    const vertex_idx_t &par = Source(in_edge, graph);

                    if (children_not_in_group.find(par) != children_not_in_group.cend()) {
                        children_not_in_group[par] -= 1;

                    } else {
                        if (parameters.use_approx_transitive_reduction) {
                            children_not_in_group[par] = 0;

                            for (const auto out_edge : OutEdges(par, graph)) {
                                if (edge_mask.find(out_edge) != edge_mask.cend()) {
                                    continue;
                                }
                                children_not_in_group[par] += 1;
                            }

                        } else {
                            children_not_in_group[par] = graph.OutDegree(par);
                        }
                        children_not_in_group[par] -= 1;
                    }
                }
                for (const auto &in_edge : InEdges(active_node, graph)) {
                    if (parameters.use_approx_transitive_reduction && (edge_mask.find(in_edge) != edge_mask.cend())) {
                        continue;
                    }

                    const vertex_idx_t &par = Source(in_edge, graph);
                    if (children_not_in_group[par] == 0) {
                        next_vertex_processing_fifo.emplace_back(par);
                    }
                }
            }

            partition.push_back(group);

            for (const auto &node : group) {
                visited[node] = true;
            }
        }
    }

    void RunOutContraction(const GraphTIn &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &partition) {
        using vertex_idx_t = vertex_idx_t<Graph_t_in>;

        const std::unordered_set<edge_desc_t<Graph_t_in>> edgeMask = parameters.use_approx_transitive_reduction
                                                                         ? long_edges_in_triangles_parallel(graph)
                                                                         : std::unordered_set<edge_desc_t<Graph_t_in>>();

        std::vector<bool> visited(graph.NumVertices(), false);

        for (const auto &topNode : top_sort_view(graph)) {
            if (visited[topNode]) {
                continue;
            }

            VWorkwT<Graph_t_in> workWeightOfGroup = 0;
            VMemwT<Graph_t_in> memoryWeightOfGroup = 0;

            std::unordered_map<vertex_idx_t, vertex_idx_t> parentsNotInGroup;
            std::vector<vertex_idx_t> group;

            std::deque<vertex_idx_t> vertexProcessingFifo({topNode});
            std::deque<vertex_idx_t> nextVertexProcessingFifo;

            unsigned depthCounter = 0;

            while ((not vertex_processing_fifo.empty()) || (not next_vertex_processing_fifo.empty())) {
                if (vertexProcessingFifo.empty()) {
                    vertexProcessingFifo = next_vertex_processing_fifo;
                    nextVertexProcessingFifo.clear();
                    depthCounter++;
                    if (depthCounter > parameters_.maxDepth_) {
                        break;
                    }
                }

                vertex_idx_t activeNode = vertex_processing_fifo.front();
                vertexProcessingFifo.pop_front();

                if (graph.VertexType(active_node) != graph.VertexType(topNode)) {
                    continue;
                }

                if (workWeightOfGroup + graph.VertexWorkWeight(active_node) > parameters_.maxWorkWeight_) {
                    continue;
                }

                if (memoryWeightOfGroup + graph.VertexMemWeight(active_node) > parameters_.maxMemoryWeight_) {
                    continue;
                }

                if constexpr (useArchitectureMemoryContraints) {
                    if (memory_weight_of_group + graph.VertexMemWeight(active_node)
                        > max_memory_per_vertex_type[graph.VertexType(top_node)]) {
                        continue;
                    }
                }

                group.emplace_back(active_node);
                workWeightOfGroup += graph.VertexWorkWeight(active_node);
                memoryWeightOfGroup += graph.VertexMemWeight(active_node);

                for (const auto &out_edge : OutEdges(active_node, graph)) {
                    if (parameters.use_approx_transitive_reduction && (edge_mask.find(out_edge) != edge_mask.cend())) {
                        continue;
                    }

                    const vertex_idx_t &child = Traget(out_edge, graph);

                    if (parents_not_in_group.find(child) != parents_not_in_group.cend()) {
                        parents_not_in_group[child] -= 1;

                    } else {
                        if (parameters.use_approx_transitive_reduction) {
                            parents_not_in_group[child] = 0;

                            for (const auto in_edge : InEdges(child, graph)) {
                                if (edge_mask.find(in_edge) != edge_mask.cend()) {
                                    continue;
                                }
                                parents_not_in_group[child] += 1;
                            }

                        } else {
                            parents_not_in_group[child] = graph.in_degree(child);
                        }
                        parents_not_in_group[child] -= 1;
                    }
                }
                for (const auto &out_edge : OutEdges(active_node, graph)) {
                    if (parameters.use_approx_transitive_reduction && (edge_mask.find(out_edge) != edge_mask.cend())) {
                        continue;
                    }

                    const vertex_idx_t &child = Traget(out_edge, graph);
                    if (parents_not_in_group[child] == 0) {
                        next_vertex_processing_fifo.emplace_back(child);
                    }
                }
            }

            partition.push_back(group);

            for (const auto &node : group) {
                visited[node] = true;
            }
        }
    }
};

}    // namespace osp
