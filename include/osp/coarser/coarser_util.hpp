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
#include <queue>
#include <set>
#include <vector>

#include "osp/auxiliary/permute.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/concepts/graph_traits.hpp"
#include "osp/concepts/specific_graph_impl.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"

namespace osp {
namespace coarser_util {

template <typename GraphTOut>
bool CheckValidContractionMap(const std::vector<vertex_idx_t<Graph_t_out>> &vertexContractionMap) {
    std::set<vertex_idx_t<Graph_t_out>> image(vertexContractionMap.cbegin(), vertex_contraction_map.cend());
    const vertex_idx_t<Graph_t_out> imageSize = static_cast<vertex_idx_t<Graph_t_out>>(image.size());
    return std::all_of(image.cbegin(), image.cend(), [image_size](const vertex_idx_t<Graph_t_out> &vert) {
        return (vert >= static_cast<vertex_idx_t<Graph_t_out>>(0)) && (vert < image_size);
    });
}

template <typename T>
struct AccSum {
    T operator()(const T &a, const T &b) { return a + b; }
};

template <typename T>
struct AccMax {
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

template <typename GraphTIn, class GraphTOut, typename VWorkAccMethod, typename VCommAccMethod, typename VMemAccMethod, typename ECommAccMethod>
bool ConstructCoarseDag(const GraphTIn &dagIn,
                        GraphTOut &coarsenedDag,
                        const std::vector<vertex_idx_t<Graph_t_out>> &vertexContractionMap) {
    static_assert(is_directed_graph_v<Graph_t_in> && is_directed_graph_v<Graph_t_out>,
                  "Graph types need to satisfy the is_directed_graph concept.");
    static_assert(IsComputationalDagV<Graph_t_in>, "Graph_t_in must be a computational DAG");
    static_assert(IsConstructableCdagV<Graph_t_out> || IsDirectConstructableCdagV<Graph_t_out>,
                  "Graph_t_out must be a (direct) constructable computational DAG");

    assert(check_valid_contraction_map<GraphTOut>(vertex_contraction_map));

    if (vertexContractionMap.size() == 0) {
        coarsenedDag = GraphTOut();
        return true;
    }

    if constexpr (IsDirectConstructableCdagV<Graph_t_out>) {
        const vertex_idx_t<Graph_t_out> numVertQuotient
            = (*std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend())) + 1;

        std::set<std::pair<vertex_idx_t<Graph_t_out>, vertex_idx_t<Graph_t_out>>> quotient_edges;

        for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
            for (const vertex_idx_t<Graph_t_in> &chld : dag_in.children(vert)) {
                if (vertex_contraction_map[vert] == vertex_contraction_map[chld]) {
                    continue;
                }
                quotient_edges.emplace(vertex_contraction_map[vert], vertex_contraction_map[chld]);
            }
        }

        coarsened_dag = Graph_t_out(num_vert_quotient, quotient_edges);

        if constexpr (HasVertexWeightsV<Graph_t_in> && IsModifiableCdagVertexV<Graph_t_out>) {
            static_assert(std::is_same_v<v_workw_t<Graph_t_in>, v_workw_t<Graph_t_out>>,
                          "Work weight types of in-graph and out-graph must be the same.");
            static_assert(std::is_same_v<v_commw_t<Graph_t_in>, v_commw_t<Graph_t_out>>,
                          "Vertex communication types of in-graph and out-graph must be the same.");
            static_assert(std::is_same_v<v_memw_t<Graph_t_in>, v_memw_t<Graph_t_out>>,
                          "Memory weight types of in-graph and out-graph must be the same.");

            for (const vertex_idx_t<Graph_t_in> &vert : coarsened_dag.vertices()) {
                coarsened_dag.SetVertexWorkWeight(vert, 0);
                coarsened_dag.SetVertexCommWeight(vert, 0);
                coarsened_dag.SetVertexMemWeight(vert, 0);
            }

            for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                coarsened_dag.SetVertexWorkWeight(vertex_contraction_map[vert],
                                                  v_work_acc_method()(coarsened_dag.VertexWorkWeight(vertex_contraction_map[vert]),
                                                                      dag_in.VertexWorkWeight(vert)));

                coarsened_dag.SetVertexCommWeight(vertex_contraction_map[vert],
                                                  v_comm_acc_method()(coarsened_dag.VertexCommWeight(vertex_contraction_map[vert]),
                                                                      dag_in.VertexCommWeight(vert)));

                coarsened_dag.SetVertexMemWeight(
                    vertex_contraction_map[vert],
                    v_mem_acc_method()(coarsened_dag.VertexMemWeight(vertex_contraction_map[vert]), dag_in.VertexMemWeight(vert)));
            }
        }

        if constexpr (HasTypedVerticesV<Graph_t_in> && is_modifiable_cdag_typed_vertex_v<Graph_t_out>) {
            static_assert(std::is_same_v<v_type_t<Graph_t_in>, v_type_t<Graph_t_out>>,
                          "Vertex type types of in graph and out graph must be the same!");

            for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                coarsened_dag.SetVertexType(vertex_contraction_map[vert], dag_in.VertexType(vert));
            }
            // assert(std::all_of(dag_in.vertices().begin(), dag_in.vertices().end(),
            //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return
            //         dag_in.VertexType(vert) ==  coarsened_dag.VertexType(vertex_contraction_map[vert]); })
            //                 && "Contracted vertices must be of the same type");
        }

        if constexpr (HasEdgeWeightsV<Graph_t_in> && IsModifiableCdagCommEdgeV<Graph_t_out>) {
            static_assert(std::is_same_v<e_commw_t<Graph_t_in>, e_commw_t<Graph_t_out>>,
                          "Edge weight type of in graph and out graph must be the same!");

            for (const auto &edge : Edges(coarsenedDag)) {
                coarsenedDag.SetEdgeCommWeight(edge, 0);
            }

            for (const auto &oriEdge : Edges(dagIn)) {
                vertex_idx_t<Graph_t_out> src = vertex_contraction_map[Source(oriEdge, dagIn)];
                vertex_idx_t<Graph_t_out> tgt = vertex_contraction_map[Traget(oriEdge, dagIn)];

                if (src == tgt) {
                    continue;
                }

                const auto [cont_edge, found] = edge_desc(src, tgt, coarsenedDag);
                assert(found && "The edge should already exist");
                coarsenedDag.SetEdgeCommWeight(
                    cont_edge, ECommAccMethod()(coarsenedDag.EdgeCommWeight(cont_edge), dagIn.EdgeCommWeight(oriEdge)));
            }
        }
        return true;
    }

    if constexpr (IsConstructableCdagV<Graph_t_out>) {
        coarsenedDag = GraphTOut();

        const vertex_idx_t<Graph_t_out> numVertQuotient
            = (*std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend())) + 1;

        for (vertex_idx_t<Graph_t_out> vert = 0; vert < num_vert_quotient; ++vert) {
            coarsenedDag.add_vertex(0, 0, 0);
        }

        for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
            coarsened_dag.SetVertexWorkWeight(
                vertex_contraction_map[vert],
                v_work_acc_method()(coarsened_dag.VertexWorkWeight(vertex_contraction_map[vert]), dag_in.VertexWorkWeight(vert)));

            coarsened_dag.SetVertexCommWeight(
                vertex_contraction_map[vert],
                v_comm_acc_method()(coarsened_dag.VertexCommWeight(vertex_contraction_map[vert]), dag_in.VertexCommWeight(vert)));

            coarsened_dag.SetVertexMemWeight(
                vertex_contraction_map[vert],
                v_mem_acc_method()(coarsened_dag.VertexMemWeight(vertex_contraction_map[vert]), dag_in.VertexMemWeight(vert)));
        }

        if constexpr (HasTypedVerticesV<Graph_t_in> && is_constructable_cdag_typed_vertex_v<Graph_t_out>) {
            static_assert(std::is_same_v<v_type_t<Graph_t_in>, v_type_t<Graph_t_out>>,
                          "Vertex type types of in graph and out graph must be the same!");

            for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                coarsened_dag.SetVertexType(vertex_contraction_map[vert], dag_in.VertexType(vert));
            }
            // assert(std::all_of(dag_in.vertices().begin(), dag_in.vertices().end(),
            //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return
            //         dag_in.VertexType(vert) ==  coarsened_dag.VertexType(vertex_contraction_map[vert]); })
            //                 && "Contracted vertices must be of the same type");
        }

        for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
            for (const vertex_idx_t<Graph_t_in> &chld : dag_in.children(vert)) {
                if (vertex_contraction_map[vert] == vertex_contraction_map[chld]) {
                    continue;
                }

                if constexpr (HasEdgeWeightsV<Graph_t_in> && is_constructable_cdag_comm_edge_v<Graph_t_out>) {
                    static_assert(std::is_same_v<e_commw_t<Graph_t_in>, e_commw_t<Graph_t_out>>,
                                  "Edge weight type of in graph and out graph must be the same!");

                    edge_desc_t<Graph_t_in> ori_edge = edge_desc(vert, chld, dag_in).first;
                    const auto pair = edge_desc(vertex_contraction_map[vert], vertex_contraction_map[chld], coarsened_dag);
                    if (pair.second) {
                        coarsened_dag.SetEdgeCommWeight(
                            pair.first,
                            e_comm_acc_method()(coarsened_dag.EdgeCommWeight(pair.first), dag_in.EdgeCommWeight(ori_edge)));
                    } else {
                        coarsened_dag.add_edge(
                            vertex_contraction_map[vert], vertex_contraction_map[chld], dag_in.EdgeCommWeight(ori_edge));
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
}

template <typename GraphTIn,
          class GraphTOut,
          typename VWorkAccMethod = acc_sum<v_workw_t<Graph_t_in>>,
          typename VCommAccMethod = acc_sum<v_commw_t<Graph_t_in>>,
          typename VMemAccMethod = acc_sum<v_memw_t<Graph_t_in>>,
          typename ECommAccMethod = acc_sum<e_commw_t<Graph_t_in>>>
bool ConstructCoarseDag(const GraphTIn &dagIn,
                        GraphTOut &coarsenedDag,
                        std::vector<vertex_idx_t<Graph_t_out>> &vertexContractionMap) {
    if constexpr (is_Compact_Sparse_Graph_reorder_v<GraphTOut>) {
        static_assert(is_directed_graph_v<Graph_t_in> && is_directed_graph_v<Graph_t_out>,
                      "Graph types need to satisfy the is_directed_graph concept.");
        static_assert(IsComputationalDagV<Graph_t_in>, "Graph_t_in must be a computational DAG");
        static_assert(IsConstructableCdagV<Graph_t_out> || IsDirectConstructableCdagV<Graph_t_out>,
                      "Graph_t_out must be a (direct) constructable computational DAG");

        assert(check_valid_contraction_map<GraphTOut>(vertex_contraction_map));

        if (vertexContractionMap.size() == 0) {
            coarsenedDag = GraphTOut();
            return true;
        }
        const vertex_idx_t<Graph_t_out> numVertQuotient
            = (*std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend())) + 1;

        std::set<std::pair<vertex_idx_t<Graph_t_out>, vertex_idx_t<Graph_t_out>>> quotient_edges;

        for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
            for (const vertex_idx_t<Graph_t_in> &chld : dag_in.children(vert)) {
                if (vertex_contraction_map[vert] == vertex_contraction_map[chld]) {
                    continue;
                }
                quotient_edges.emplace(vertex_contraction_map[vert], vertex_contraction_map[chld]);
            }
        }

        coarsened_dag = Graph_t_out(num_vert_quotient, quotient_edges);

        const auto &pushforwardMap = coarsenedDag.get_pushforward_permutation();
        std::vector<vertex_idx_t<Graph_t_out>> combinedExpansionMap(dagIn.NumVertices());
        for (const auto &vert : dagIn.vertices()) {
            combinedExpansionMap[vert] = pushforwardMap[vertex_contraction_map[vert]];
        }

        if constexpr (HasVertexWeightsV<Graph_t_in> && IsModifiableCdagVertexV<Graph_t_out>) {
            static_assert(std::is_same_v<v_workw_t<Graph_t_in>, v_workw_t<Graph_t_out>>,
                          "Work weight types of in-graph and out-graph must be the same.");
            static_assert(std::is_same_v<v_commw_t<Graph_t_in>, v_commw_t<Graph_t_out>>,
                          "Vertex communication types of in-graph and out-graph must be the same.");
            static_assert(std::is_same_v<v_memw_t<Graph_t_in>, v_memw_t<Graph_t_out>>,
                          "Memory weight types of in-graph and out-graph must be the same.");

            for (const vertex_idx_t<Graph_t_in> &vert : coarsened_dag.vertices()) {
                coarsened_dag.SetVertexWorkWeight(vert, 0);
                coarsened_dag.SetVertexCommWeight(vert, 0);
                coarsened_dag.SetVertexMemWeight(vert, 0);
            }

            for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                coarsened_dag.SetVertexWorkWeight(vertex_contraction_map[vert],
                                                  v_work_acc_method()(coarsened_dag.VertexWorkWeight(combined_expansion_map[vert]),
                                                                      dag_in.VertexWorkWeight(vert)));

                coarsened_dag.SetVertexCommWeight(vertex_contraction_map[vert],
                                                  v_comm_acc_method()(coarsened_dag.VertexCommWeight(combined_expansion_map[vert]),
                                                                      dag_in.VertexCommWeight(vert)));

                coarsened_dag.SetVertexMemWeight(
                    vertex_contraction_map[vert],
                    v_mem_acc_method()(coarsened_dag.VertexMemWeight(combined_expansion_map[vert]), dag_in.VertexMemWeight(vert)));
            }
        }

        if constexpr (HasTypedVerticesV<Graph_t_in> && is_modifiable_cdag_typed_vertex_v<Graph_t_out>) {
            static_assert(std::is_same_v<v_type_t<Graph_t_in>, v_type_t<Graph_t_out>>,
                          "Vertex type types of in graph and out graph must be the same!");

            for (const vertex_idx_t<Graph_t_in> &vert : dag_in.vertices()) {
                coarsened_dag.SetVertexType(vertex_contraction_map[vert], dag_in.VertexType(vert));
            }
            // assert(std::all_of(dag_in.vertices().begin(), dag_in.vertices().end(),
            //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return
            //         dag_in.VertexType(vert) ==  coarsened_dag.VertexType(vertex_contraction_map[vert]); })
            //                 && "Contracted vertices must be of the same type");
        }

        if constexpr (HasEdgeWeightsV<Graph_t_in> && HasEdgeWeightsV<Graph_t_out>) {
            static_assert(std::is_same_v<e_commw_t<Graph_t_in>, e_commw_t<Graph_t_out>>,
                          "Edge weight type of in graph and out graph must be the same!");

            for (const auto &oriEdge : Edges(dagIn)) {
                vertex_idx_t<Graph_t_out> src = vertex_contraction_map[Source(oriEdge, dagIn)];
                vertex_idx_t<Graph_t_out> tgt = vertex_contraction_map[Traget(oriEdge, dagIn)];

                if (src == tgt) {
                    continue;
                }

                coarsenedDag.SetEdgeCommWeight(src, tgt, 0);
            }

            for (const auto &oriEdge : Edges(dagIn)) {
                vertex_idx_t<Graph_t_out> src = vertex_contraction_map[Source(oriEdge, dagIn)];
                vertex_idx_t<Graph_t_out> tgt = vertex_contraction_map[Traget(oriEdge, dagIn)];

                if (src == tgt) {
                    continue;
                }

                const auto contEdge = coarsenedDag.edge(pushforwardMap[src], pushforwardMap[tgt]);
                assert(Source(cont_edge, coarsenedDag) == pushforwardMap[src]
                       && Traget(cont_edge, coarsenedDag) == pushforwardMap[tgt]);
                coarsenedDag.SetEdgeCommWeight(
                    src, tgt, ECommAccMethod()(coarsenedDag.EdgeCommWeight(cont_edge), dagIn.EdgeCommWeight(oriEdge)));
            }
        }

        std::swap(vertex_contraction_map, combined_expansion_map);
        return true;
    } else {
        return construct_coarse_dag<GraphTIn, GraphTOut, VWorkAccMethod, VCommAccMethod, VMemAccMethod, ECommAccMethod>(
            dagIn, coarsenedDag, static_cast<const std::vector<vertex_idx_t<Graph_t_out>> &>(vertex_contraction_map));
    }
}

template <typename GraphTIn>
bool CheckValidExpansionMap(const std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertexExpansionMap) {
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
}

template <typename GraphTIn, typename GraphTOut>
std::vector<std::vector<vertex_idx_t<Graph_t_in>>> InvertVertexContractionMap(
    const std::vector<vertex_idx_t<Graph_t_out>> &vertexContractionMap) {
    assert(check_valid_contraction_map<GraphTOut>(vertex_contraction_map));

    vertex_idx_t<Graph_t_out> numVert = vertex_contraction_map.size() == 0
                                            ? 0
                                            : *std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend()) + 1;

    std::vector<std::vector<vertex_idx_t<Graph_t_in>>> expansionMap(numVert);

    for (std::size_t i = 0; i < vertexContractionMap.size(); ++i) {
        expansionMap[vertex_contraction_map[i]].push_back(i);
    }

    return expansion_map;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<vertex_idx_t<Graph_t_out>> InvertVertexExpansionMap(
    const std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertexExpansionMap) {
    assert(check_valid_expansion_map<GraphTIn>(vertex_expansion_map));

    vertex_idx_t<Graph_t_in> numVert = 0;
    for (const auto &group : vertex_expansion_map) {
        for (const vertex_idx_t<Graph_t_in> &vert : group) {
            num_vert = std::max(num_vert, vert + 1);
        }
    }

    std::vector<vertex_idx_t<Graph_t_out>> vertexContractionMap(numVert);
    for (std::size_t i = 0; i < vertexExpansionMap.size(); i++) {
        for (const vertex_idx_t<Graph_t_in> &vert : vertex_expansion_map[i]) {
            vertex_contraction_map[vert] = static_cast<vertex_idx_t<Graph_t_out>>(i);
        }
    }

    return vertex_contraction_map;
}

template <typename GraphTIn>
void ReorderExpansionMap(const GraphTIn &graph, std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertexExpansionMap) {
    assert(check_valid_expansion_map<GraphTIn>(vertex_expansion_map));

    std::vector<std::size_t> vertexContractionMap(graph.NumVertices());
    for (std::size_t i = 0; i < vertexExpansionMap.size(); i++) {
        for (const vertex_idx_t<Graph_t_in> &vert : vertex_expansion_map[i]) {
            vertex_contraction_map[vert] = i;
        }
    }

    std::vector<std::size_t> prec(vertexExpansionMap.size(), 0);
    for (const auto &vert : graph.vertices()) {
        for (const auto &par : graph.parents(vert)) {
            if (vertexContractionMap.at(par) != vertexContractionMap.at(vert)) {
                prec[vertexContractionMap.at(vert)] += 1;
            }
        }
    }

    for (auto &comp : vertex_expansion_map) {
        std::nth_element(comp.begin(), comp.begin(), comp.end());
    }

    auto cmp = [&vertex_expansion_map](const std::size_t &lhs, const std::size_t &rhs) {
        return vertex_expansion_map[lhs] > vertex_expansion_map[rhs];    // because priority queue is a max_priority queue
    };

    std::priority_queue<std::size_t, std::vector<std::size_t>, decltype(cmp)> ready(cmp);
    std::vector<std::size_t> topOrder;
    topOrder.reserve(vertex_expansion_map.size());
    for (std::size_t i = 0; i < vertexExpansionMap.size(); ++i) {
        if (prec[i] == 0) {
            ready.emplace(i);
        }
    }

    while (!ready.empty()) {
        const std::size_t nextGroup = ready.top();
        ready.pop();
        topOrder.emplace_back(nextGroup);

        for (const auto &vert : vertex_expansion_map[next_group]) {
            for (const auto &chld : graph.children(vert)) {
                if (vertex_contraction_map.at(vert) != vertex_contraction_map.at(chld)) {
                    prec[vertex_contraction_map.at(chld)] -= 1;
                    if (prec[vertex_contraction_map.at(chld)] == 0) {
                        ready.emplace(vertex_contraction_map.at(chld));
                    }
                }
            }
        }
    }
    assert(topOrder.size() == vertex_expansion_map.size());

    inverse_permute_inplace(vertex_expansion_map, topOrder);

    return;
}

template <typename GraphTIn, typename GraphTOut>
bool PullBackSchedule(const BspSchedule<GraphTIn> &scheduleIn,
                      const std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertexMap,
                      BspSchedule<GraphTOut> &scheduleOut) {
    for (unsigned v = 0; v < vertexMap.size(); ++v) {
        const auto proc = scheduleIn.assignedProcessor(v);
        const auto step = scheduleIn.assignedSuperstep(v);

        for (const auto &u : vertex_map[v]) {
            schedule_out.setAssignedSuperstep(u, step);
            schedule_out.setAssignedProcessor(u, proc);
        }
    }

    return true;
}

template <typename GraphTIn, typename GraphTOut>
bool PullBackSchedule(const BspSchedule<GraphTIn> &scheduleIn,
                      const std::vector<vertex_idx_t<Graph_t_out>> &reverseVertexMap,
                      BspSchedule<GraphTOut> &scheduleOut) {
    for (unsigned idx = 0; idx < reverseVertexMap.size(); ++idx) {
        const auto &v = reverse_vertex_map[idx];

        scheduleOut.setAssignedSuperstep(idx, scheduleIn.assignedSuperstep(v));
        scheduleOut.setAssignedProcessor(idx, scheduleIn.assignedProcessor(v));
    }

    return true;
}

template <typename IntegralType>
std::vector<IntegralType> ComposeVertexContractionMap(const std::vector<IntegralType> &firstMap,
                                                      const std::vector<IntegralType> &secondMap) {
    static_assert(std::is_integral_v<IntegralType>);
    std::vector<IntegralType> composedMap(firstMap.size());

    for (std::size_t i = 0; i < composedMap.size(); ++i) {
        composedMap[i] = secondMap[firstMap[i]];
    }

    return composedMap;
}

}    // end namespace coarser_util
}    // end namespace osp
