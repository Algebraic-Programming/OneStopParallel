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

#include <algorithm>
#include <set>
#include <vector>

#include "bsp/model/BspSchedule.hpp"
#include "concepts/computational_dag_concept.hpp"
#include "concepts/constructable_computational_dag_concept.hpp"
#include "concepts/graph_traits.hpp"
#include "graph_algorithms/directed_graph_edge_desc_util.hpp"

namespace osp { namespace coarser_util {

template<typename Graph_t_out>
bool check_valid_contraction_map(const std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) {
    std::set<vertex_idx_t<Graph_t_out>> image(vertex_contraction_map.cbegin(), vertex_contraction_map.cend());
    const vertex_idx_t<Graph_t_out> image_size = static_cast<vertex_idx_t<Graph_t_out>>(image.size());
    return std::all_of(image.cbegin(), image.cend(), [image_size](const vertex_idx_t<Graph_t_out> &vert) {
        return (vert >= static_cast<vertex_idx_t<Graph_t_out>>(0)) && (vert < image_size);
    });
};

template<typename T>
struct acc_sum {

    T operator()(const T &a, const T &b) { return a + b; }
};

template<typename T>
struct acc_max {

    T operator()(const T &a, const T &b) { return std::max(a, b); }
};

/**
 * @brief Coarsens the input computational DAG into a simplified version.
 *
 * @param dag_in The input computational DAG to be coarsened. It is expected to be a valid graph structure.
 * @param coarsened_dag The output computational DAG after coarsening. It will be populated by this method.
 * @param vertex_contraction_map Output mapping from dag_in to coarsened_dag.
 * @return A status code indicating the success or failure of the coarsening operation.
 */
template<typename Graph_t_in, typename Graph_t_out, typename v_work_acc_method = acc_sum<v_workw_t<Graph_t_in>>,
         typename v_comm_acc_method = acc_sum<v_commw_t<Graph_t_in>>,
         typename v_mem_acc_method = acc_sum<v_memw_t<Graph_t_in>>,
         typename e_comm_acc_method = acc_sum<e_commw_t<Graph_t_in>>>
bool construct_coarse_dag(const Graph_t_in &dag_in, Graph_t_out &coarsened_dag,
                         const std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) {

    assert(check_valid_contraction_map(vertex_contraction_map));

    // todo
    // if constexpr (has_quotient_graph_construction_method<Graph_t_in, Graph_t_out>) {
    //     return true;
    // }

    if constexpr (is_constructable_cdag_v<Graph_t_out>) {
        coarsened_dag = Graph_t_out();

        const vertex_idx_t<Graph_t_out> num_vert_quotient =
            (*std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend())) + 1;

        for (vertex_idx_t<Graph_t_out> vert = 0; vert < num_vert_quotient; ++vert) {
            coarsened_dag.add_vertex(0, 0, 0);
        }

        for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {

            coarsened_dag.set_vertex_work_weight(
                vertex_contraction_map[vert],
                v_work_acc_method()(coarsened_dag.vertex_work_weight(vertex_contraction_map[vert]),
                                  dag_in.vertex_work_weight(vert)));

            coarsened_dag.set_vertex_comm_weight(
                vertex_contraction_map[vert],
                v_comm_acc_method()(coarsened_dag.vertex_comm_weight(vertex_contraction_map[vert]),
                                  dag_in.vertex_comm_weight(vert)));

            coarsened_dag.set_vertex_mem_weight(
                vertex_contraction_map[vert],
                v_mem_acc_method()(coarsened_dag.vertex_mem_weight(vertex_contraction_map[vert]),
                                 dag_in.vertex_mem_weight(vert)));
        }

        if constexpr (has_typed_vertices_v<Graph_t_in> && is_constructable_cdag_typed_vertex_v<Graph_t_out>) {
            static_assert(std::is_same_v<v_type_t<Graph_t_in>, v_type_t<Graph_t_out>>,
                          "Vertex type types of in graph and out graph must be the same!");

            for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                coarsened_dag.set_vertex_type(vertex_contraction_map[vert], dag_in.vertex_type(vert));
            }
            // assert(std::all_of(dag_in.vertices().begin(), dag_in.vertices().end(),
            //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return
            //         dag_in.vertex_type(vert) ==  coarsened_dag.vertex_type(vertex_contraction_map[vert]); })
            //                 && "Contracted vertices must be of the same type");
        }

        for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
            for (const vertex_idx_t<Graph_t_in> &chld : dag_in.children(vert)) {
                if (vertex_contraction_map[vert] == vertex_contraction_map[chld]) {
                    continue;
                }

                if constexpr (has_edge_weights_v<Graph_t_in> && is_constructable_cdag_comm_edge_v<Graph_t_out>) {
                    static_assert(std::is_same_v<e_commw_t<Graph_t_in>, e_commw_t<Graph_t_out>>,
                                  "Edge weight type of in graph and out graph must be the same!");

                    edge_desc_t<Graph_t_in> ori_edge = edge_desc(vert, chld, dag_in).first;
                    const auto pair =
                        edge_desc(vertex_contraction_map[vert], vertex_contraction_map[chld], coarsened_dag);
                    if (pair.second) {

                        coarsened_dag.set_edge_comm_weight(pair.first,
                                                           e_comm_acc_method()(coarsened_dag.edge_comm_weight(pair.first),
                                                                             dag_in.edge_comm_weight(ori_edge)));
                    } else {

                        coarsened_dag.add_edge(vertex_contraction_map[vert], vertex_contraction_map[chld],
                                               dag_in.edge_comm_weight(ori_edge));
                    }

                } else {

                    if (not edge(vertex_contraction_map[vert], vertex_contraction_map[chld], coarsened_dag)) {
                        coarsened_dag.add_edge(vertex_contraction_map[vert], vertex_contraction_map[chld]);
                    }
                }
            }
        }

        return true;
    }

    return false;
};

template<typename Graph_t_in>
bool check_valid_expansion_map(const std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertex_expansion_map) {
    std::size_t cntr = 0;

    std::vector<bool> preImage;
    for (const std::vector<vertex_idx_t<Graph_t_in>> &group : vertex_expansion_map) {
        if (group.size() == 0) {
            return false;
        }

        for (const vertex_idx_t<Graph_t_in> vert : group) {
            if (vert < static_cast<vertex_idx_t<Graph_t_in>>(0)) {
                return false;
            }

            if (static_cast<std::size_t>(vert) >= preImage.size()) {
                preImage.resize(vert + 1, false);
            }

            if (preImage[vert]) {
                return false;
            }

            preImage[vert] = true;
            cntr++;
        }
    }

    return (cntr == preImage.size());
};

template<typename Graph_t_in, typename Graph_t_out>
std::vector<std::vector<vertex_idx_t<Graph_t_in>>>
invert_vertex_contraction_map(const std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) {
    assert(check_valid_contraction_map(vertex_contraction_map));

    vertex_idx_t<Graph_t_out> max_vert =
        *std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend());

    std::vector<std::vector<vertex_idx_t<Graph_t_in>>> expansion_map(max_vert + 1);

    for (std::size_t i = 0; i < vertex_contraction_map.size(); ++i) {
        expansion_map[vertex_contraction_map[i]].push_back(i);
    }

    return expansion_map;
};

template<typename Graph_t_in, typename Graph_t_out>
std::vector<vertex_idx_t<Graph_t_out>>
invert_vertex_expansion_map(const std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertex_expansion_map) {
    assert(check_valid_expansion_map(vertex_expansion_map));

    vertex_idx_t<Graph_t_in> num_vert = 0;
    for (const auto &group : vertex_expansion_map) {
        for (const vertex_idx_t<Graph_t_in> &vert : group) {
            num_vert = std::max(num_vert, vert + 1);
        }
    }

    std::vector<vertex_idx_t<Graph_t_out>> vertex_contraction_map(num_vert);
    for (std::size_t i = 0; i < vertex_expansion_map.size(); i++) {
        for (const vertex_idx_t<Graph_t_in> &vert : vertex_expansion_map[i]) {
            vertex_contraction_map[vert] = i;
        }
    }

    return vertex_contraction_map;
};

}} // namespace osp::coarser_util