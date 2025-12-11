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
#include <vector>

#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_concept.hpp"

namespace osp {

template <typename Graph_t_in, typename Graph_t_out>
void create_induced_subgraph(const Graph_t_in &dag,
                             Graph_t_out &dag_out,
                             const std::set<vertex_idx_t<Graph_t_in>> &selected_nodes,
                             const std::set<vertex_idx_t<Graph_t_in>> &extra_sources = {}) {
    static_assert(std::is_same_v<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_out>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    static_assert(is_constructable_cdag_vertex_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_vertex concept");

    static_assert(is_constructable_cdag_edge_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_edge concept");

    assert(dag_out.num_vertices() == 0);

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

template <typename Graph_t_in, typename Graph_t_out>
void create_induced_subgraph(const Graph_t_in &dag,
                             Graph_t_out &dag_out,
                             const std::vector<vertex_idx_t<Graph_t_in>> &selected_nodes) {
    return create_induced_subgraph(dag, dag_out, std::set<vertex_idx_t<Graph_t_in>>(selected_nodes.begin(), selected_nodes.end()));
}

template <typename Graph_t>
bool checkOrderedIsomorphism(const Graph_t &first, const Graph_t &second) {
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

            for (const auto &out_edge : out_edges(node, first)) {
                first_children.emplace(target(out_edge, first), first.edge_comm_weight(out_edge));
            }

            for (const auto &out_edge : out_edges(node, second)) {
                second_children.emplace(target(out_edge, second), second.edge_comm_weight(out_edge));
            }

            auto itr = first_children.begin(), second_itr = second_children.begin();
            for (; itr != first_children.end() && second_itr != second_children.end(); ++itr) {
                if (*itr != *second_itr) {
                    return false;
                }
                ++second_itr;
            }

        } else {
            std::set<vertex_idx_t<Graph_t>> first_children, second_children;

            for (const auto &child : first.children(node)) {
                first_children.emplace(child);
            }

            for (const auto &child : second.children(node)) {
                second_children.emplace(child);
            }

            auto itr = first_children.begin(), second_itr = second_children.begin();
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

template <typename Graph_t_in, typename Graph_t_out>
std::vector<Graph_t_out> create_induced_subgraphs(const Graph_t_in &dag_in, const std::vector<unsigned> &partition_IDs) {
    // assumes that input partition IDs are consecutive and starting from 0

    static_assert(std::is_same_v<vertex_idx_t<Graph_t_in>, vertex_idx_t<Graph_t_out>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    static_assert(is_constructable_cdag_vertex_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_vertex concept");

    static_assert(is_constructable_cdag_edge_v<Graph_t_out>, "Graph_t_out must satisfy the constructable_cdag_edge concept");

    unsigned number_of_parts = 0;
    for (const auto id : partition_IDs) {
        number_of_parts = std::max(number_of_parts, id + 1);
    }

    std::vector<Graph_t_out> split_dags(number_of_parts);

    std::vector<vertex_idx_t<Graph_t_out>> local_idx(dag_in.num_vertices());

    for (const auto node : dag_in.vertices()) {
        local_idx[node] = split_dags[partition_IDs[node]].num_vertices();

        if constexpr (is_constructable_cdag_typed_vertex_v<Graph_t_out> and has_typed_vertices_v<Graph_t_in>) {
            split_dags[partition_IDs[node]].add_vertex(dag_in.vertex_work_weight(node),
                                                       dag_in.vertex_comm_weight(node),
                                                       dag_in.vertex_mem_weight(node),
                                                       dag_in.vertex_type(node));
        } else {
            split_dags[partition_IDs[node]].add_vertex(
                dag_in.vertex_work_weight(node), dag_in.vertex_comm_weight(node), dag_in.vertex_mem_weight(node));
        }
    }

    if constexpr (has_edge_weights_v<Graph_t_in> and has_edge_weights_v<Graph_t_out>) {
        for (const auto node : dag_in.vertices()) {
            for (const auto &out_edge : out_edges(node, dag_in)) {
                auto succ = target(out_edge, dag_in);

                if (partition_IDs[node] == partition_IDs[succ]) {
                    split_dags[partition_IDs[node]].add_edge(local_idx[node], local_idx[succ], dag_in.edge_comm_weight(out_edge));
                }
            }
        }
    } else {
        for (const auto node : dag_in.vertices()) {
            for (const auto &child : dag_in.children(node)) {
                if (partition_IDs[node] == partition_IDs[child]) {
                    split_dags[partition_IDs[node]].add_edge(local_idx[node], local_idx[child]);
                }
            }
        }
    }

    return split_dags;
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
