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

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_concept.hpp"

namespace osp {

template <typename GraphTIn, typename GraphTOut>
void CreateInducedSubgraph(const GraphTIn &dag,
                           GraphTOut &dagOut,
                           const std::set<vertex_idx_t<Graph_t_in>> &selectedNodes,
                           const std::set<vertex_idx_t<Graph_t_in>> &extraSources = {}) {
    static_assert(std::is_same_v<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_out>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    static_assert(is_constructable_cdag_vertex_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_vertex concept");

    static_assert(is_constructable_cdag_edge_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_edge concept");

    assert(dagOut.num_vertices() == 0);

    std::map<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_in>> local_idx;

    for (const auto &node : extra_sources) {
        local_idx[node] = dag_out.num_vertices();
        if constexpr (is_constructable_cdag_typed_vertex_v<Graph_t_out> and has_typed_vertices_v<Graph_t_in>) {
            // add extra source with type
            dag_out.add_vertex(0, dag.vertex_comm_weight(node), dag.vertex_mem_weight(node), dag.vertex_type(node));
        } else {
            // add extra source without type
            dag_out.add_vertex(0, dag.vertex_comm_weight(node), dag.vertex_mem_weight(node));
        }
    }

    for (const auto &node : selected_nodes) {
        local_idx[node] = dag_out.num_vertices();

        if constexpr (is_constructable_cdag_typed_vertex_v<Graph_t_out> and has_typed_vertices_v<Graph_t_in>) {
            // add vertex with type
            dag_out.add_vertex(
                dag.vertex_work_weight(node), dag.vertex_comm_weight(node), dag.vertex_mem_weight(node), dag.vertex_type(node));
        } else {
            // add vertex without type
            dag_out.add_vertex(dag.vertex_work_weight(node), dag.vertex_comm_weight(node), dag.vertex_mem_weight(node));
        }
    }

    if constexpr (has_edge_weights_v<Graph_t_in> and has_edge_weights_v<Graph_t_out>) {
        // add edges with edge comm weights
        for (const auto &node : selected_nodes) {
            for (const auto &in_edge : in_edges(node, dag)) {
                const auto &pred = source(in_edge, dag);
                if (selected_nodes.find(pred) != selected_nodes.end() || extra_sources.find(pred) != extra_sources.end()) {
                    dag_out.add_edge(local_idx[pred], local_idx[node], dag.edge_comm_weight(in_edge));
                }
            }
        }

    } else {
        // add edges without edge comm weights
        for (const auto &node : selected_nodes) {
            for (const auto &pred : dag.parents(node)) {
                if (selected_nodes.find(pred) != selected_nodes.end() || extra_sources.find(pred) != extra_sources.end()) {
                    dag_out.add_edge(local_idx[pred], local_idx[node]);
                }
            }
        }
    }
}

template <typename GraphTIn, typename GraphTOut>
void CreateInducedSubgraph(const GraphTIn &dag, GraphTOut &dagOut, const std::vector<vertex_idx_t<Graph_t_in>> &selectedNodes) {
    return create_induced_subgraph(dag, dag_out, std::set<vertex_idx_t<Graph_t_in>>(selected_nodes.begin(), selected_nodes.end()));
}

template <typename GraphT>
bool CheckOrderedIsomorphism(const GraphT &first, const GraphT &second) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    if (first.num_vertices() != second.num_vertices() || first.num_edges() != second.num_edges()) {
        return false;
    }

    for (const auto &node : first.vertices()) {
        if (first.vertex_work_weight(node) != second.vertex_work_weight(node)
            || first.vertex_mem_weight(node) != second.vertex_mem_weight(node)
            || first.vertex_comm_weight(node) != second.vertex_comm_weight(node)
            || first.vertex_type(node) != second.vertex_type(node)) {
            return false;
        }

        if (first.in_degree(node) != second.in_degree(node) || first.out_degree(node) != second.out_degree(node)) {
            return false;
        }

        if constexpr (has_edge_weights_v<Graph_t>) {
            std::set<std::pair<vertex_idx_t<Graph_t>, e_commw_t<Graph_t>>> first_children, second_children;

            for (const auto &outEdge : out_edges(node, first)) {
                first_children.emplace(target(out_edge, first), first.edge_comm_weight(out_edge));
            }

            for (const auto &outEdge : out_edges(node, second)) {
                second_children.emplace(target(out_edge, second), second.edge_comm_weight(out_edge));
            }

            auto itr = first_children.begin(), secondItr = second_children.begin();
            for (; itr != first_children.end() && second_itr != second_children.end(); ++itr) {
                if (*itr != *second_itr) {
                    return false;
                }
                ++second_itr;
            }

        } else {
            std::set<vertex_idx_t<Graph_t>> firstChildren, second_children;

            for (const auto &child : first.children(node)) {
                firstChildren.emplace(child);
            }

            for (const auto &child : second.children(node)) {
                second_children.emplace(child);
            }

            auto itr = first_children.begin(), secondItr = second_children.begin();
            for (; itr != first_children.end() && second_itr != second_children.end(); ++itr) {
                if (*itr != *second_itr) {
                    return false;
                }
                ++second_itr;
            }
        }
    }

    return true;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<GraphTOut> CreateInducedSubgraphs(const GraphTIn &dagIn, const std::vector<unsigned> &partitionIDs) {
    // assumes that input partition IDs are consecutive and starting from 0

    static_assert(std::is_same_v<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_out>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    static_assert(is_constructable_cdag_vertex_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_vertex concept");

    static_assert(is_constructable_cdag_edge_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_edge concept");

    unsigned numberOfParts = 0;
    for (const auto id : partitionIDs) {
        numberOfParts = std::max(numberOfParts, id + 1);
    }

    std::vector<GraphTOut> splitDags(numberOfParts);

    std::vector<vertex_idx_t<Graph_t_out>> localIdx(dagIn.num_vertices());

    for (const auto node : dagIn.vertices()) {
        localIdx[node] = splitDags[partitionIDs[node]].num_vertices();

        if constexpr (is_constructable_cdag_typed_vertex_v<Graph_t_out> and has_typed_vertices_v<Graph_t_in>) {
            splitDags[partitionIDs[node]].add_vertex(dagIn.vertex_work_weight(node),
                                                     dagIn.vertex_comm_weight(node),
                                                     dagIn.vertex_mem_weight(node),
                                                     dagIn.vertex_type(node));
        } else {
            splitDags[partitionIDs[node]].add_vertex(
                dagIn.vertex_work_weight(node), dagIn.vertex_comm_weight(node), dagIn.vertex_mem_weight(node));
        }
    }

    if constexpr (has_edge_weights_v<Graph_t_in> and has_edge_weights_v<Graph_t_out>) {
        for (const auto node : dagIn.vertices()) {
            for (const auto &outEdge : out_edges(node, dagIn)) {
                auto succ = target(outEdge, dagIn);

                if (partitionIDs[node] == partitionIDs[succ]) {
                    splitDags[partitionIDs[node]].add_edge(local_idx[node], local_idx[succ], dagIn.edge_comm_weight(outEdge));
                }
            }
        }
    } else {
        for (const auto node : dagIn.vertices()) {
            for (const auto &child : dagIn.children(node)) {
                if (partitionIDs[node] == partitionIDs[child]) {
                    splitDags[partitionIDs[node]].add_edge(local_idx[node], local_idx[child]);
                }
            }
        }
    }

    return splitDags;
}

template <typename Graph_t_in, typename Graph_t_out>
std::unordered_map<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_in>> create_induced_subgraph_map(
    const Graph_t_in &dag, Graph_t_out &dag_out, const std::vector<vertex_idx_t<Graph_t_in>> &selected_nodes) {
    static_assert(std::is_same_v<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_out>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    static_assert(is_constructable_cdag_vertex_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_vertex concept");

    static_assert(is_constructable_cdag_edge_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_edge concept");

    assert(dag_out.num_vertices() == 0);

    std::unordered_map<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_in>> local_idx;
    local_idx.reserve(selected_nodes.size());

    for (const auto &node : selected_nodes) {
        local_idx[node] = dag_out.num_vertices();

        if constexpr (is_constructable_cdag_typed_vertex_v<Graph_t_out> and has_typed_vertices_v<Graph_t_in>) {
            // add vertex with type
            dag_out.add_vertex(
                dag.vertex_work_weight(node), dag.vertex_comm_weight(node), dag.vertex_mem_weight(node), dag.vertex_type(node));
        } else {
            // add vertex without type
            dag_out.add_vertex(dag.vertex_work_weight(node), dag.vertex_comm_weight(node), dag.vertex_mem_weight(node));
        }
    }

    if constexpr (has_edge_weights_v<Graph_t_in> and has_edge_weights_v<Graph_t_out>) {
        // add edges with edge comm weights
        for (const auto &node : selected_nodes) {
            for (const auto &in_edge : in_edges(node, dag)) {
                const auto &pred = source(in_edge, dag);
                if (local_idx.count(pred)) {
                    dag_out.add_edge(local_idx[pred], local_idx[node], dag.edge_comm_weight(in_edge));
                }
            }
        }

    } else {
        // add edges without edge comm weights
        for (const auto &node : selected_nodes) {
            for (const auto &pred : dag.parents(node)) {
                if (local_idx.count(pred)) {
                    dag_out.add_edge(local_idx[pred], local_idx[node]);
                }
            }
        }
    }

    return local_idx;
}

}    // end namespace osp
