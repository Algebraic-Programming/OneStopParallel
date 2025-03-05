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

#include "coarser/hdagg/hdagg_coarser_variant.hpp"

RETURN_STATUS hdagg_coarser_variant::coarseDag(const ComputationalDag &dag_in, ComputationalDag &dag_out,
                                               std::vector<std::vector<VertexType>> &vertex_map) {

    std::vector<bool> visited(dag_in.numberOfVertices(), false);
    std::vector<VertexType> reverse_vertex_map(dag_in.numberOfVertices(), 0);

    std::unordered_set<EdgeType, EdgeType_hash> edge_mask = dag_in.long_edges_in_triangles_parallel();
    const auto edge_mast_end = edge_mask.cend();

    for (const auto &sink : dag_in.sinkVertices()) {
        vertex_map.push_back(std::vector<VertexType>({sink}));
    }

    size_t part_ind = 0;
    size_t partition_size = vertex_map.size();
    while (part_ind < partition_size) {
        size_t vert_ind = 0;
        size_t part_size = vertex_map[part_ind].size();

        add_new_super_node(dag_in, dag_out, vertex_map[part_ind][vert_ind]);

        while (vert_ind < part_size) {

            const VertexType vert = vertex_map[part_ind][vert_ind];
            reverse_vertex_map[vert] = current_super_node_idx;

            for (const auto &in_edge : dag_in.in_edges(vert)) {

                if (edge_mask.find(in_edge) != edge_mast_end && not visited[in_edge.m_source])
                    continue;

                bool indegree_one = true;

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

                if (indegree_one) {

                    const auto &source = in_edge.m_source;
                    int node_mem = dag_in.nodeMemoryWeight(source);

                    if (memory_constraint_type == LOCAL_INC_EDGES_2) {

                        if (not dag_in.isSource(source)) {
                            node_mem = 0;
                        }
                    }

                    if (((current_memory + node_mem > memory_threshold) ||
                         (current_work + dag_in.nodeWorkWeight(source) > work_threshold) ||
                         (vertex_map[part_ind].size() >= super_node_size_threshold) ||
                         (current_communication + dag_in.nodeCommunicationWeight(source) > communication_threshold)) ||
                        // or node type changes
                        (dag_out.nodeType(current_super_node_idx) != dag_in.nodeType(source))) {

                        if (!visited[in_edge.m_source]) {
                            vertex_map.push_back(std::vector<VertexType>({in_edge.m_source}));
                            partition_size++;
                            visited[in_edge.m_source] = true;
                        }

                    } else {

                        current_memory += node_mem;
                        current_work += dag_in.nodeWorkWeight(source);
                        current_communication += dag_in.nodeCommunicationWeight(source);

                        vertex_map[part_ind].push_back(source);
                        part_size++;
                    }

                } else {

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

    return SUCCESS;
}