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

#include "coarser/Coarser.hpp"

#include "graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "graph_algorithms/directed_graph_util.hpp"
#include <limits>

namespace osp {

template<typename Graph_t1, typename Graph_t2>
class hdagg_coarser : public Coarser<Graph_t1, typename Graph_t2> {

    static_assert(std::is_same_v<v_workw_t<Graph_t1>, v_workw_t<Graph_t2>>,
                  "Graph_t1 and Graph_t2 must have the same work weight type");
    static_assert(std::is_same_v<v_memw_t<Graph_t1>, v_memw_t<Graph_t2>>,
                  "Graph_t1 and Graph_t2 must have the same memory weight type");
    static_assert(std::is_same_v<v_commw_t<Graph_t1>, v_commw_t<Graph_t2>>,
                  "Graph_t1 and Graph_t2 must have the same communication weight type");

  private:
    using VertexType = vertex_idx_t<Graph_t1>;

  protected:
    v_workw_t<Graph_t1> work_threshold = std::numeric_limits<v_workw_t<Graph_t1>>::max();
    v_memw_t<Graph_t1> memory_threshold = std::numeric_limits<v_memw_t<Graph_t1>>::max();
    v_commw_t<Graph_t1> communication_threshold = std::numeric_limits<v_commw_t<Graph_t1>>::max();

    std::size_t super_node_size_threshold = std::numeric_limits<std::size_t>::max();

    // MEMORY_CONSTRAINT_TYPE memory_constraint_type = NONE;

    // internal data strauctures
    v_memw_t<Graph_t1> current_memory = 0;
    v_workw_t<Graph_t1> current_work = 0;
    v_commw_t<Graph_t1> current_communication = 0;
    VertexType current_super_node_idx = 0;

    void finish_super_node(Graph_t2 &dag_out) {

        dag_out.set_vertex_mem_weight(current_super_node_idx, current_memory);
        dag_out.set_vertex_work_weight(current_super_node_idx, current_work);
        dag_out.set_vertex_comm_weight(current_super_node_idx, current_communication);
    }

    void add_edges_between_super_nodes(const Graph_t1 &dag_in, Graph_t2 &dag_out,
                                       std::vector<std::vector<VertexType>> &vertex_map,
                                       std::vector<VertexType> &reverse_vertex_map) {

        current_super_node_idx = 0;

        for (const auto &super_node : vertex_map) {
            for (const auto &node : super_node) {

                for (const auto &in_edge : dag_in.in_edges(node)) {
                    const VertexType parent_rev = reverse_vertex_map[in_edge.m_source];
                    if (parent_rev != current_super_node_idx && parent_rev != std::numeric_limits<VertexType>::max()) {

                        const auto pair = has_edge_desc(parent_rev, current_super_node_idx, dag_out);

                        if (pair.second) {
                            dag_out.set_edge_comm_weight(pair.first, dag_out.edge_comm_weight(pair.first) +
                                                                         dag_in.edge_comm_weight(in_edge));
                        } else {
                            dag_out.add_edge(parent_rev, current_super_node_idx, dag_in.edge_comm_weight(in_edge));
                        }
                    }
                }
            }
            current_super_node_idx++;
        }
    }

    void add_new_super_node(const Graph_t &dag_in, Graph_t &dag_out, VertexType node) {

        v_memw_t<Graph_t1> node_mem = dag_in.vertex_mem_weight(node);

        // if (memory_constraint_type == LOCAL_INC_EDGES_2) {

        //     if (not dag_in.isSource(node)) {
        //         node_mem = 0;
        //     }
        // }

        current_memory = node_mem;
        current_work = dag_in.vertex_work_weight(node);
        current_communication = dag_in.vertex_comm_weight_weight(node);
        current_super_node_idx =
            dag_out.add_vertex(current_work, current_communication, current_memory, dag_in.vertex_type(node));
    }

  public:
    hdagg_coarser() {};

    virtual ~hdagg_coarser() = default;

    virtual std::string getCoarserName() const override { return "hdagg_coarser"; };

    virtual bool coarseDag(const Graph_t1 &dag_in, Graph_t2 &dag_out,
                           std::vector<std::vector<VertexType>> &vertex_map) override {

        std::vector<bool> visited(dag_in.num_vertices(), false);
        std::vector<VertexType> reverse_vertex_map(dag_in.num_vertices(), 0);

        auto edge_mask = long_edges_in_triangles_parallel(dag_in);
        const auto edge_mast_end = edge_mask.cend();

        for (const auto &sink : sink_vertices_view(dag_in)) {
            vertex_map.push_back(std::vector<VertexType>({sink}));
        }

        std::size_t part_ind = 0;
        std::size_t partition_size = vertex_map.size();
        while (part_ind < partition_size) {
            std::size_t vert_ind = 0;
            std::size_t part_size = vertex_map[part_ind].size();

            add_new_super_node(dag_in, dag_out, vertex_map[part_ind][vert_ind]);

            while (vert_ind < part_size) {

                const VertexType vert = vertex_map[part_ind][vert_ind];
                reverse_vertex_map[vert] = current_super_node_idx;
                bool indegree_one = true;

                for (const auto &in_edge : dag_in.in_edges(vert)) {

                    if (edge_mask.find(in_edge) != edge_mast_end)
                        continue;

                    unsigned count = 0;
                    for (const auto &out_edge : dag_in.out_edges(in_edge.m_source)) {

                        if (edge_mask.find(out_edge) != edge_mast_end)
                            continue;

                        count++;
                        if (count > 1) {
                            indegree_one = false;
                            break;
                        }
                    }

                    if (not indegree_one) {
                        break;
                    }
                }

                if (indegree_one) {
                    for (const auto &in_edge : dag_in.in_edges(vert)) {

                        if (edge_mask.find(in_edge) != edge_mast_end)
                            continue;

                        const auto &source = in_edge.m_source;

                        v_memw_t<Graph_t1> node_mem = dag_in.vertex_mem_weight(source);

                        // if (memory_constraint_type == LOCAL_INC_EDGES_2) {

                        //     if (not dag_in.isSource(source)) {
                        //         node_mem = 0;
                        //     }
                        // }

                        if (((current_memory + node_mem > memory_threshold) ||
                             (current_work + dag_in.vertex_work_weight(source) > work_threshold) ||
                             (vertex_map[part_ind].size() >= super_node_size_threshold) ||
                             (current_communication + dag_in.vertex_comm_weight(source) > communication_threshold)) ||
                            // or node type changes
                            (dag_out.vertex_type(current_super_node_idx) != dag_in.vertex_type(source))) {

                            if (!visited[in_edge.m_source]) {
                                vertex_map.push_back(std::vector<VertexType>({in_edge.m_source}));
                                partition_size++;
                                visited[in_edge.m_source] = true;
                            }

                        } else {

                            current_memory += node_mem;
                            current_work += dag_in.vertex_work_weight(source);
                            current_communication += dag_in.vertex_comm_weight(source);

                            vertex_map[part_ind].push_back(source);
                            part_size++;
                        }
                    }
                } else {
                    for (const auto &in_edge : dag_in.in_edges(vert)) {

                        if (edge_mask.find(in_edge) != edge_mast_end)
                            continue;

                        if (!visited[in_edge.m_source]) {
                            vertex_map.push_back(std::vector<VertexType>({in_edge.m_source}));
                            partition_size++;
                            visited[in_edge.m_source] = true;
                        }
                    }
                }
                vert_ind++;
            }

            finish_super_node(dag_out);

            part_ind++;
        }

        add_edges_between_super_nodes(dag_in, dag_out, vertex_map, reverse_vertex_map);

        return true;
    }

    inline void set_work_threshold(v_workw_t<Graph_t1> work_threshold_) { work_threshold = work_threshold_; }
    inline void set_memory_threshold(v_memw_t<Graph_t1> memory_threshold_) { memory_threshold = memory_threshold_; }
    inline void set_communication_threshold(v_commw_t<Graph_t1> communication_threshold_) {
        communication_threshold = communication_threshold_;
    }
    inline void set_super_node_size_threshold(std::size_t super_node_size_threshold_) {
        super_node_size_threshold = super_node_size_threshold_;
    }
    // inline void set_memory_constraint_type(MEMORY_CONSTRAINT_TYPE memory_constraint_type_) { memory_constraint_type =
    // memory_constraint_type_; }
};

} // namespace osp