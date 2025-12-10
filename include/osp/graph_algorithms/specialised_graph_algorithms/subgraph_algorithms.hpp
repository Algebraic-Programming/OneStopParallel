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

#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"

namespace osp {

template<typename Graph_t_in, typename vert_t, typename edge_t, typename work_weight_type, typename comm_weight_type, typename mem_weight_type, typename vertex_type_template_type>
std::unordered_map<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_in>> create_induced_subgraph_map(const Graph_t_in &dag, Compact_Sparse_Graph<true, true, true, true, true, vert_t, edge_t, work_weight_type, comm_weight_type, mem_weight_type, vertex_type_template_type> &dag_out,
                                                                                                   const std::vector<vertex_idx_t<Graph_t_in>> &selected_nodes) {

    using Graph_t_out = Compact_Sparse_Graph<true, true, true, true, true, vert_t, edge_t, work_weight_type, comm_weight_type, mem_weight_type, vertex_type_template_type>;

    static_assert(std::is_same_v<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_out>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    const std::vector<vertex_idx_t<Graph_t_in>> topOrder = GetTopOrder(instance.getComputationalDag());
    std::vector<vertex_idx_t<Graph_t_in>> topOrderPosition(topOrder.size());
    for (vertex_idx_t<Graph_t_in> pos = 0; pos < dag.numbernum_vertices(); ++pos) {
        topOrderPosition[topOrder[pos]] = pos;
    }

    auto topCmp = [&topOrderPosition](const &vertex_idx_t<Graph_t_in> lhs, const &vertex_idx_t<Graph_t_in> rhs) { return topOrderPosition[lhs] < topOrderPosition[rhs]; };

    std::set<vertex_idx_t<Graph_t_in>, decltype(topCmp)> selectedVerticesOrdered(selected_nodes.begin(), selected_nodes.end(), topCmp);

    std::unordered_map<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_in>> local_idx;
    local_idx.reserve(selected_nodes.size());

    vertex_idx_t<Graph_t_in> nodeCntr = 0;
    for (const auto &node : selectedVerticesOrdered) {
        local_idx[node] = nodeCntr++;
    }

    std::vector<std::pair<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_in>>> edges;
    for (const auto &node : selectedVerticesOrdered) {
        for (const auto &chld : dag.children(node)) {
            if (selectedVerticesOrdered.find(chld) != selectedVerticesOrdered.end()) {
                edges.emplace(node, chld);
            }
        }
    }

    dag_out = Graph_t_out(nodeCntr, edges);

    for (const auto &[oriVert, outVert] : local_idx) {
        dag_out.set_vertex_work_weight(outVert, dag.vertex_work_weight(oriVert));
        dag_out.set_vertex_comm_weight(outVert, dag.vertex_comm_weight(oriVert));
        dag_out.set_vertex_mem_weight(outVert, dag.vertex_mem_weight(oriVert));
        dag_out.set_vertex_type(outVert, dag.vertex_type(oriVert));
    }

    return local_idx;
}

} // end namespace osp
