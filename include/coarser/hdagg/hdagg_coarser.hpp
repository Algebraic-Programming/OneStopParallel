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

template<typename Graph_t_in, typename Graph_t_out>
class hdagg_coarser : public CoarserGenContractionMap<Graph_t_in, Graph_t_out> {

    static_assert(is_directed_graph_edge_desc_v<Graph_t_in>,
                  "Graph_t_in must satisfy the directed_graph edge desc concept");
    static_assert(has_hashable_edge_desc_v<Graph_t_in>, "Graph_t_in must satisfy the has_hashable_edge_desc concept");
    static_assert(has_typed_vertices_v<Graph_t_in>, "Graph_t_in must have typed vertices");

  private:
    using VertexType_in = vertex_idx_t<Graph_t_in>;
    using VertexType_out = vertex_idx_t<Graph_t_out>;

  protected:
    v_workw_t<Graph_t_in> work_threshold = std::numeric_limits<v_workw_t<Graph_t_in>>::max();
    v_memw_t<Graph_t_in> memory_threshold = std::numeric_limits<v_memw_t<Graph_t_in>>::max();
    v_commw_t<Graph_t_in> communication_threshold = std::numeric_limits<v_commw_t<Graph_t_in>>::max();

    std::size_t super_node_size_threshold = std::numeric_limits<std::size_t>::max();

    // MEMORY_CONSTRAINT_TYPE memory_constraint_type = NONE;

    // internal data strauctures
    v_memw_t<Graph_t_in> current_memory = 0;
    v_workw_t<Graph_t_in> current_work = 0;
    v_commw_t<Graph_t_in> current_communication = 0;
    VertexType_out current_super_node_idx = 0;
    v_type_t<Graph_t_in> current_v_type = 0;

    void add_new_super_node(const Graph_t_in &dag_in, VertexType_in node) {

        v_memw_t<Graph_t_in> node_mem = dag_in.vertex_mem_weight(node);

        current_memory = node_mem;
        current_work = dag_in.vertex_work_weight(node);
        current_communication = dag_in.vertex_comm_weight(node);
        current_v_type = dag_in.vertex_type(node);
    }

  public:
    hdagg_coarser() {};

    virtual ~hdagg_coarser() = default;

    virtual std::string getCoarserName() const override { return "hdagg_coarser"; };

    virtual std::vector<vertex_idx_t<Graph_t_out>> generate_vertex_contraction_map(const Graph_t_in &dag_in) override {

        std::vector<bool> visited(dag_in.num_vertices(), false);
        std::vector<VertexType_out> reverse_vertex_map(dag_in.num_vertices());

        std::vector<std::vector<VertexType_in>> vertex_map;

        auto edge_mask = long_edges_in_triangles(dag_in);
        const auto edge_mast_end = edge_mask.cend();

        for (const auto &sink : sink_vertices_view(dag_in)) {
            vertex_map.push_back(std::vector<VertexType_in>({sink}));
        }

        std::size_t part_ind = 0;
        std::size_t partition_size = vertex_map.size();
        while (part_ind < partition_size) {
            std::size_t vert_ind = 0;
            std::size_t part_size = vertex_map[part_ind].size();

            add_new_super_node(dag_in, vertex_map[part_ind][vert_ind]);

            while (vert_ind < part_size) {

                const VertexType_in vert = vertex_map[part_ind][vert_ind];
                reverse_vertex_map[vert] = current_super_node_idx;
                bool indegree_one = true;

                for (const auto &in_edge : dag_in.in_edges(vert)) {

                    if (edge_mask.find(in_edge) != edge_mast_end)
                        continue;

                    unsigned count = 0;
                    for (const auto &out_edge : dag_in.out_edges(source(in_edge, dag_in))) {

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

                        const auto &edge_source = source(in_edge, dag_in);

                        v_memw_t<Graph_t_in> node_mem = dag_in.vertex_mem_weight(edge_source);

                        if (((current_memory + node_mem > memory_threshold) ||
                             (current_work + dag_in.vertex_work_weight(edge_source) > work_threshold) ||
                             (vertex_map[part_ind].size() >= super_node_size_threshold) ||
                             (current_communication + dag_in.vertex_comm_weight(edge_source) >
                              communication_threshold)) ||
                            // or node type changes
                            (current_v_type != dag_in.vertex_type(edge_source))) {

                            if (!visited[edge_source]) {
                                vertex_map.push_back(std::vector<VertexType_in>({edge_source}));
                                partition_size++;
                                visited[edge_source] = true;
                            }

                        } else {

                            current_memory += node_mem;
                            current_work += dag_in.vertex_work_weight(edge_source);
                            current_communication += dag_in.vertex_comm_weight(edge_source);

                            vertex_map[part_ind].push_back(edge_source);
                            part_size++;
                        }
                    }
                } else {
                    for (const auto &in_edge : dag_in.in_edges(vert)) {

                        if (edge_mask.find(in_edge) != edge_mast_end)
                            continue;

                        const auto &edge_source = source(in_edge, dag_in);

                        if (!visited[edge_source]) {
                            vertex_map.push_back(std::vector<VertexType_in>({edge_source}));
                            partition_size++;
                            visited[edge_source] = true;
                        }
                    }
                }
                vert_ind++;
            }

            part_ind++;
        }

        return reverse_vertex_map;
    }

    inline void set_work_threshold(v_workw_t<Graph_t_in> work_threshold_) { work_threshold = work_threshold_; }
    inline void set_memory_threshold(v_memw_t<Graph_t_in> memory_threshold_) { memory_threshold = memory_threshold_; }
    inline void set_communication_threshold(v_commw_t<Graph_t_in> communication_threshold_) {
        communication_threshold = communication_threshold_;
    }
    inline void set_super_node_size_threshold(std::size_t super_node_size_threshold_) {
        super_node_size_threshold = super_node_size_threshold_;
    }
};

} // namespace osp