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

#include <numeric>

#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "directed_graph_top_sort.hpp"

namespace osp {

template<typename Graph_from, typename Graph_to>
bool construct_computational_dag(const Graph_from &from, Graph_to &to) {

    static_assert(is_computational_dag_v<Graph_from>, "Graph_from must satisfy the computational_dag concept");
    static_assert(is_constructable_cdag_vertex_v<Graph_to>, "Graph_to must satisfy the constructable_cdag_vertex concept");

    for (const auto &v_idx : from.vertices()) {

        if constexpr (has_typed_vertices_v<Graph_from> and has_typed_vertices_v<Graph_to>) {
            to.add_vertex(from.vertex_work_weight(v_idx), from.vertex_comm_weight(v_idx),
                          from.vertex_mem_weight(v_idx), from.vertex_type(v_idx));
        } else {
            to.add_vertex(from.vertex_work_weight(v_idx), from.vertex_comm_weight(v_idx),
                          from.vertex_mem_weight(v_idx));
        }
    }

    if constexpr (has_edge_weights_v<Graph_from> and has_edge_weights_v<Graph_to>) {

        for (const auto &e : edges(from)) {
            to.add_edge(source(e, from), target(e, from), from.edge_comm_weight(e));
        }

    } else {
        for (const auto &v : from.vertices()) {
            for (const auto &child : from.children(v)) {
                to.add_edge(v, child);
            }
        }
    }

    return true;
}

} // namespace osp
