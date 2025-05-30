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
#include "coarser/Coarser_gen_exp_map.hpp"
#include "graph_algorithms/directed_graph_edge_desc_util_parallel.hpp"
#include <limits>

namespace osp {

/**
 * @brief Acyclic graph contractor that contracts groups of nodes with only one vertex with incoming/outgoing edges
 * (from outside the group)
 *
 */
template<typename Graph_t_in, typename Graph_t_out, bool use_architecture_memory_contraints = false>
class FunnelBfs : public CoarserGenExpansionMap<Graph_t_in, Graph_t_out> {

  public:
    /**
     * @brief Parameters for Funnel coarsener
     *
     */
    struct FunnelBfs_parameters {

        bool funnel_incoming;

        bool use_approx_transitive_reduction;

        v_workw_t<Graph_t_in> max_work_weight;
        v_memw_t<Graph_t_in> max_memory_weight;

        unsigned max_depth;

        FunnelBfs_parameters(v_workw_t<Graph_t_in> max_work_weight_ = std::numeric_limits<v_workw_t<Graph_t_in>>::max(),
                             v_memw_t<Graph_t_in> max_memory_weight_ = std::numeric_limits<v_memw_t<Graph_t_in>>::max(),
                             unsigned max_depth_ = std::numeric_limits<unsigned>::max(), 
                             bool funnel_incoming_ = true,
                             bool use_approx_transitive_reduction_ = true)
            : funnel_incoming(funnel_incoming_), use_approx_transitive_reduction(use_approx_transitive_reduction_),
              max_work_weight(max_work_weight_), max_memory_weight(max_memory_weight_), max_depth(max_depth_) {};

        ~FunnelBfs_parameters() = default;
    };

    FunnelBfs(FunnelBfs_parameters parameters_ = FunnelBfs_parameters())
        : parameters(parameters_) {}

    virtual ~FunnelBfs() = default;

    virtual std::vector<std::vector<vertex_idx_t<Graph_t_in>>>
    generate_vertex_expansion_map(const Graph_t_in &graph) override {

        if constexpr (use_architecture_memory_contraints) {
            if (max_memory_per_vertex_type.size() < graph.num_vertex_types()) {
                throw std::runtime_error("FunnelBfs: max_memory_per_vertex_type has insufficient size.");
            }
        }

        std::vector<std::vector<vertex_idx_t<Graph_t_in>>> partition;

        if (parameters.funnel_incoming) {
            run_in_contraction(graph, partition);
        } else {
            run_out_contraction(graph, partition);
        }

        return partition;
    }

    std::string getCoarserName() const override { return "FunnelBfs"; }

    std::vector<v_memw_t<Graph_t_in>> &get_max_memory_per_vertex_type() { return max_memory_per_vertex_type; }

  private:
    FunnelBfs_parameters parameters;

    std::vector<v_memw_t<Graph_t_in>> max_memory_per_vertex_type;

    void run_in_contraction(const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &partition) {

        using vertex_idx_t = vertex_idx_t<Graph_t_in>;

        const std::unordered_set<edge_desc_t<Graph_t_in>> edge_mask = parameters.use_approx_transitive_reduction
                                                                       ? long_edges_in_triangles_parallel(graph)
                                                                       : std::unordered_set<edge_desc_t<Graph_t_in>>();

        std::vector<bool> visited(graph.num_vertices(), false);

        const std::vector<vertex_idx_t> top_order = GetTopOrder(graph);

        for (auto rev_top_it = top_order.rbegin(); rev_top_it != top_order.crend(); rev_top_it++) {

            const vertex_idx_t &bottom_node = *rev_top_it;

            if (visited[bottom_node])
                continue;

            v_workw_t<Graph_t_in> work_weight_of_group = 0;
            v_memw_t<Graph_t_in> memory_weight_of_group = 0;

            std::unordered_map<vertex_idx_t, vertex_idx_t> children_not_in_group;
            std::vector<vertex_idx_t> group;

            std::deque<vertex_idx_t> vertex_processing_fifo({bottom_node});
            std::deque<vertex_idx_t> next_vertex_processing_fifo;

            unsigned depth_counter = 0;

            while ((not vertex_processing_fifo.empty()) || (not next_vertex_processing_fifo.empty())) {

                if (vertex_processing_fifo.empty()) {
                    vertex_processing_fifo = next_vertex_processing_fifo;
                    next_vertex_processing_fifo.clear();
                    depth_counter++;
                    if (depth_counter > parameters.max_depth) {
                        break;
                    }
                }

                vertex_idx_t active_node = vertex_processing_fifo.front();
                vertex_processing_fifo.pop_front();

                if (graph.vertex_type(active_node) != graph.vertex_type(bottom_node))
                    continue;

                if (work_weight_of_group + graph.vertex_work_weight(active_node) > parameters.max_work_weight)
                    continue;

                if (memory_weight_of_group + graph.vertex_mem_weight(active_node) > parameters.max_memory_weight)
                    continue;

                if constexpr (use_architecture_memory_contraints) {
                    if (memory_weight_of_group + graph.vertex_mem_weight(active_node) >
                        max_memory_per_vertex_type[graph.vertex_type(bottom_node)])
                        continue;
                }

                group.emplace_back(active_node);
                work_weight_of_group += graph.vertex_work_weight(active_node);
                memory_weight_of_group += graph.vertex_mem_weight(active_node);

                for (const auto &in_edge : graph.in_edges(active_node)) {

                    if (parameters.use_approx_transitive_reduction && (edge_mask.find(in_edge) != edge_mask.cend()))
                        continue;

                    const vertex_idx_t &par = source(in_edge, graph);

                    if (children_not_in_group.find(par) != children_not_in_group.cend()) {
                        children_not_in_group[par] -= 1;

                    } else {

                        if (parameters.use_approx_transitive_reduction) {

                            children_not_in_group[par] = 0;

                            for (const auto out_edge : graph.out_edges(par)) {
                                if (edge_mask.find(out_edge) != edge_mask.cend())
                                    continue;
                                children_not_in_group[par] += 1;
                            }

                        } else {
                            children_not_in_group[par] = graph.out_degree(par);
                        }
                        children_not_in_group[par] -= 1;
                    }
                }
                for (const auto &in_edge : graph.in_edges(active_node)) {

                    if (parameters.use_approx_transitive_reduction && (edge_mask.find(in_edge) != edge_mask.cend()))
                        continue;

                    const vertex_idx_t &par = source(in_edge, graph);
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

    void run_out_contraction(const Graph_t_in &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &partition) {

        using vertex_idx_t = vertex_idx_t<Graph_t_in>;

        const std::unordered_set<edge_desc_t<Graph_t_in>> edge_mask = parameters.use_approx_transitive_reduction
                                                                       ? long_edges_in_triangles_parallel(graph)
                                                                       : std::unordered_set<edge_desc_t<Graph_t_in>>();

        std::vector<bool> visited(graph.num_vertices(), false);

        for (const auto &top_node : top_sort_view(graph)) {

            if (visited[top_node])
                continue;

            v_workw_t<Graph_t_in> work_weight_of_group = 0;
            v_memw_t<Graph_t_in> memory_weight_of_group = 0;

            std::unordered_map<vertex_idx_t, vertex_idx_t> parents_not_in_group;
            std::vector<vertex_idx_t> group;

            std::deque<vertex_idx_t> vertex_processing_fifo({top_node});
            std::deque<vertex_idx_t> next_vertex_processing_fifo;

            unsigned depth_counter = 0;

            while ((not vertex_processing_fifo.empty()) || (not next_vertex_processing_fifo.empty())) {

                if (vertex_processing_fifo.empty()) {
                    vertex_processing_fifo = next_vertex_processing_fifo;
                    next_vertex_processing_fifo.clear();
                    depth_counter++;
                    if (depth_counter > parameters.max_depth) {
                        break;
                    }
                }

                vertex_idx_t active_node = vertex_processing_fifo.front();
                vertex_processing_fifo.pop_front();

                if (graph.vertex_type(active_node) != graph.vertex_type(top_node))
                    continue;

                if (work_weight_of_group + graph.vertex_work_weight(active_node) > parameters.max_work_weight)
                    continue;

                if (memory_weight_of_group + graph.vertex_mem_weight(active_node) > parameters.max_memory_weight)
                    continue;

                if constexpr (use_architecture_memory_contraints) {
                    if (memory_weight_of_group + graph.vertex_mem_weight(active_node) >
                        max_memory_per_vertex_type[graph.vertex_type(top_node)])
                        continue;
                }

                group.emplace_back(active_node);
                work_weight_of_group += graph.vertex_work_weight(active_node);
                memory_weight_of_group += graph.vertex_mem_weight(active_node);

                for (const auto &out_edge : graph.out_edges(active_node)) {

                    if (parameters.use_approx_transitive_reduction && (edge_mask.find(out_edge) != edge_mask.cend()))
                        continue;

                    const vertex_idx_t &child = target(out_edge, graph);

                    if (parents_not_in_group.find(child) != parents_not_in_group.cend()) {
                        parents_not_in_group[child] -= 1;

                    } else {

                        if (parameters.use_approx_transitive_reduction) {

                            parents_not_in_group[child] = 0;

                            for (const auto in_edge : graph.in_edges(child)) {
                                if (edge_mask.find(in_edge) != edge_mask.cend())
                                    continue;
                                parents_not_in_group[child] += 1;
                            }

                        } else {
                            parents_not_in_group[child] = graph.in_degree(child);
                        }
                        parents_not_in_group[child] -= 1;
                    }
                }
                for (const auto &out_edge : graph.out_edges(active_node)) {

                    if (parameters.use_approx_transitive_reduction && (edge_mask.find(out_edge) != edge_mask.cend()))
                        continue;

                    const vertex_idx_t &child = target(out_edge, graph);
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
} // namespace osp