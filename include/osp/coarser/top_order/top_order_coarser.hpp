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

#include <vector>

#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

template <typename Graph_t_in, typename Graph_t_out, std::vector<vertex_idx_t<Graph_t_in>> (*top_sort_func)(const Graph_t_in &)>
class top_order_coarser : public Coarser<Graph_t_in, Graph_t_out> {
  private:
    using VertexType = vertex_idx_t<Graph_t_in>;

    // parameters
    v_workw_t<Graph_t_in> work_threshold = std::numeric_limits<v_workw_t<Graph_t_in>>::max();
    v_memw_t<Graph_t_in> memory_threshold = std::numeric_limits<v_memw_t<Graph_t_in>>::max();
    v_commw_t<Graph_t_in> communication_threshold = std::numeric_limits<v_commw_t<Graph_t_in>>::max();
    unsigned degree_threshold = std::numeric_limits<unsigned>::max();
    unsigned node_dist_threshold = std::numeric_limits<unsigned>::max();
    VertexType super_node_size_threshold = std::numeric_limits<VertexType>::max();

    // internal data strauctures
    v_memw_t<Graph_t_in> current_memory = 0;
    v_workw_t<Graph_t_in> current_work = 0;
    v_commw_t<Graph_t_in> current_communication = 0;
    VertexType current_super_node_idx = 0;

    void finish_super_node_add_edges(const Graph_t_in &dag_in,
                                     Graph_t_out &dag_out,
                                     const std::vector<VertexType> &nodes,
                                     std::vector<vertex_idx_t<Graph_t_out>> &reverse_vertex_map) {
        dag_out.set_vertex_mem_weight(current_super_node_idx, current_memory);
        dag_out.set_vertex_work_weight(current_super_node_idx, current_work);
        dag_out.set_vertex_comm_weight(current_super_node_idx, current_communication);

        for (const auto &node : nodes) {
            if constexpr (has_edge_weights_v<Graph_t_in> && has_edge_weights_v<Graph_t_out>) {
                for (const auto &in_edge : in_edges(node, dag_in)) {
                    const VertexType parent_rev = reverse_vertex_map[source(in_edge, dag_in)];
                    if (parent_rev != current_super_node_idx && parent_rev != std::numeric_limits<VertexType>::max()) {
                        auto pair = edge_desc(parent_rev, current_super_node_idx, dag_out);
                        if (pair.second) {
                            dag_out.set_edge_comm_weight(pair.first,
                                                         dag_out.edge_comm_weight(pair.first) + dag_in.edge_comm_weight(in_edge));
                        } else {
                            dag_out.add_edge(parent_rev, current_super_node_idx, dag_in.edge_comm_weight(in_edge));
                        }
                    }
                }
            } else {
                for (const auto &parent : dag_in.parents(node)) {
                    const VertexType parent_rev = reverse_vertex_map[parent];
                    if (parent_rev != current_super_node_idx && parent_rev != std::numeric_limits<VertexType>::max()) {
                        if (not edge(parent_rev, current_super_node_idx, dag_out)) {
                            dag_out.add_edge(parent_rev, current_super_node_idx);
                        }
                    }
                }
            }
        }
    }

    void add_new_super_node(const Graph_t_in &dag_in, Graph_t_out &dag_out, VertexType node) {
        // int node_mem = dag_in.nodeMemoryWeight(node);

        // if (memory_constraint_type == LOCAL_INC_EDGES_2) {

        //     if (not dag_in.isSource(node)) {
        //         node_mem = 0;
        //     }
        // }

        current_memory = dag_in.vertex_mem_weight(node);
        current_work = dag_in.vertex_work_weight(node);
        current_communication = dag_in.vertex_comm_weight(node);

        if constexpr (is_computational_dag_typed_vertices_v<Graph_t_in> && is_computational_dag_typed_vertices_v<Graph_t_out>) {
            current_super_node_idx
                = dag_out.add_vertex(current_work, current_communication, current_memory, dag_in.vertex_type(node));
        } else {
            current_super_node_idx = dag_out.add_vertex(current_work, current_communication, current_memory);
        }
    }

  public:
    top_order_coarser() {};
    virtual ~top_order_coarser() = default;

    inline void set_degree_threshold(unsigned degree_threshold_) { degree_threshold = degree_threshold_; }

    inline void set_work_threshold(v_workw_t<Graph_t_in> work_threshold_) { work_threshold = work_threshold_; }

    inline void set_memory_threshold(v_memw_t<Graph_t_in> memory_threshold_) { memory_threshold = memory_threshold_; }

    inline void set_communication_threshold(v_commw_t<Graph_t_in> communication_threshold_) {
        communication_threshold = communication_threshold_;
    }

    inline void set_super_node_size_threshold(VertexType super_node_size_threshold_) {
        super_node_size_threshold = super_node_size_threshold_;
    }

    inline void set_node_dist_threshold(unsigned node_dist_threshold_) { node_dist_threshold = node_dist_threshold_; }

    // inline void set_memory_constraint_type(MEMORY_CONSTRAINT_TYPE memory_constraint_type_) { memory_constraint_type =
    // memory_constraint_type_; }

    virtual std::string getCoarserName() const override { return "top_order_coarser"; };

    virtual bool coarsenDag(const Graph_t_in &dag_in,
                            Graph_t_out &dag_out,
                            std::vector<vertex_idx_t<Graph_t_out>> &reverse_vertex_map) override {
        assert(dag_out.num_vertices() == 0);
        if (dag_in.num_vertices() == 0) {
            reverse_vertex_map = std::vector<vertex_idx_t<Graph_t_out>>();
            return true;
        }

        std::vector<VertexType> top_ordering = top_sort_func(dag_in);

        std::vector<unsigned> source_node_dist = get_top_node_distance(dag_in);

        reverse_vertex_map.resize(dag_in.num_vertices(), std::numeric_limits<VertexType>::max());

        std::vector<std::vector<VertexType>> vertex_map;
        vertex_map.push_back(std::vector<VertexType>({top_ordering[0]}));

        add_new_super_node(dag_in, dag_out, top_ordering[0]);
        reverse_vertex_map[top_ordering[0]] = current_super_node_idx;

        for (size_t i = 1; i < top_ordering.size(); i++) {
            const auto v = top_ordering[i];

            // int node_mem = dag_in.vertex_mem_weight(v);

            // if (memory_constraint_type == LOCAL_INC_EDGES_2) {

            //     if (not dag_in.isSource(v)) {
            //         node_mem = 0;
            //     }
            // }

            const unsigned dist = source_node_dist[v] - source_node_dist[top_ordering[i - 1]];

            // start new super node if thresholds are exceeded
            if (((current_memory + dag_in.vertex_mem_weight(v) > memory_threshold)
                 || (current_work + dag_in.vertex_work_weight(v) > work_threshold)
                 || (vertex_map.back().size() >= super_node_size_threshold)
                 || (current_communication + dag_in.vertex_comm_weight(v) > communication_threshold))
                || (dist > node_dist_threshold) ||
                // or prev node high out degree
                (dag_in.out_degree(top_ordering[i - 1]) > degree_threshold)) {
                finish_super_node_add_edges(dag_in, dag_out, vertex_map.back(), reverse_vertex_map);
                vertex_map.push_back(std::vector<VertexType>({v}));
                add_new_super_node(dag_in, dag_out, v);

            } else {    // grow current super node

                if constexpr (is_computational_dag_typed_vertices_v<Graph_t_in>
                              && is_computational_dag_typed_vertices_v<Graph_t_out>) {
                    if (dag_out.vertex_type(current_super_node_idx) != dag_in.vertex_type(v)) {
                        finish_super_node_add_edges(dag_in, dag_out, vertex_map.back(), reverse_vertex_map);
                        vertex_map.push_back(std::vector<VertexType>({v}));
                        add_new_super_node(dag_in, dag_out, v);

                    } else {
                        current_memory += dag_in.vertex_mem_weight(v);
                        current_work += dag_in.vertex_work_weight(v);
                        current_communication += dag_in.vertex_comm_weight(v);

                        vertex_map.back().push_back(v);
                    }

                } else {
                    current_memory += dag_in.vertex_mem_weight(v);
                    current_work += dag_in.vertex_work_weight(v);
                    current_communication += dag_in.vertex_comm_weight(v);

                    vertex_map.back().push_back(v);
                }
            }

            reverse_vertex_map[v] = current_super_node_idx;
        }

        if (!vertex_map.back().empty()) {
            finish_super_node_add_edges(dag_in, dag_out, vertex_map.back(), reverse_vertex_map);
        }

        return true;
    }
};

}    // namespace osp
