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
bool CheckValidContractionMap(const std::vector<VertexIdxT<GraphTOut>> &vertexContractionMap) {
    std::set<VertexIdxT<GraphTOut>> image(vertexContractionMap.cbegin(), vertexContractionMap.cend());
    const VertexIdxT<GraphTOut> imageSize = static_cast<VertexIdxT<GraphTOut>>(image.size());
    return std::all_of(image.cbegin(), image.cend(), [imageSize](const VertexIdxT<GraphTOut> &vert) {
        return (vert >= static_cast<VertexIdxT<GraphTOut>>(0)) && (vert < imageSize);
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
                        const std::vector<VertexIdxT<GraphTOut>> &vertexContractionMap) {
    static_assert(IsDirectedGraphV<GraphTIn> && IsDirectedGraphV<GraphTOut>,
                  "Graph types need to satisfy the is_directed_graph concept.");
    static_assert(IsComputationalDagV<GraphTIn>, "GraphTIn must be a computational DAG");
    static_assert(IsConstructableCdagV<GraphTOut> || IsDirectConstructableCdagV<GraphTOut>,
                  "GraphTOut must be a (direct) constructable computational DAG");

    assert(check_valid_contraction_map<GraphTOut>(vertexContractionMap));

    if (vertexContractionMap.size() == 0) {
        coarsenedDag = GraphTOut();
        return true;
    }

    if constexpr (IsDirectConstructableCdagV<GraphTOut>) {
        const VertexIdxT<GraphTOut> numVertQuotient
            = (*std::max_element(vertexContractionMap.cbegin(), vertexContractionMap.cend())) + 1;

        std::set<std::pair<VertexIdxT<GraphTOut>, VertexIdxT<GraphTOut>>> quotient_edges;

        for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
            for (const VertexIdxT<GraphTIn> &chld : dagIn.Children(vert)) {
                if (vertexContractionMap[vert] == vertexContractionMap[chld]) {
                    continue;
                }
                quotient_edges.emplace(vertexContractionMap[vert], vertexContractionMap[chld]);
            }
        }

        coarsenedDag = GraphTOut(numVertQuotient, quotient_edges);

        if constexpr (HasVertexWeightsV<GraphTIn> && IsModifiableCdagVertexV<GraphTOut>) {
            static_assert(std::is_same_v<VWorkwT<GraphTIn>, VWorkwT<GraphTOut>>,
                          "Work weight types of in-graph and out-graph must be the same.");
            static_assert(std::is_same_v<VCommwT<GraphTIn>, VCommwT<GraphTOut>>,
                          "Vertex communication types of in-graph and out-graph must be the same.");
            static_assert(std::is_same_v<VMemwT<GraphTIn>, VMemwT<GraphTOut>>,
                          "Memory weight types of in-graph and out-graph must be the same.");

            for (const VertexIdxT<GraphTIn> &vert : coarsenedDag.Vertices()) {
                coarsenedDag.SetVertexWorkWeight(vert, 0);
                coarsenedDag.SetVertexCommWeight(vert, 0);
                coarsenedDag.SetVertexMemWeight(vert, 0);
            }

            for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
                coarsenedDag.SetVertexWorkWeight(
                    vertexContractionMap[vert],
                    VWorkAccMethod()(coarsenedDag.VertexWorkWeight(vertexContractionMap[vert]), dagIn.VertexWorkWeight(vert)));

                coarsenedDag.SetVertexCommWeight(
                    vertexContractionMap[vert],
                    VCommAccMethod()(coarsenedDag.VertexCommWeight(vertexContractionMap[vert]), dagIn.VertexCommWeight(vert)));

                coarsenedDag.SetVertexMemWeight(
                    vertexContractionMap[vert],
                    VMemAccMethod()(coarsenedDag.VertexMemWeight(vertexContractionMap[vert]), dagIn.VertexMemWeight(vert)));
            }
        }

        if constexpr (HasTypedVerticesV<GraphTIn> && IsModifiableCdagTypedVertexV<GraphTOut>) {
            static_assert(std::is_same_v<VTypeT<GraphTIn>, VTypeT<GraphTOut>>,
                          "Vertex type types of in graph and out graph must be the same!");

            for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
                coarsenedDag.SetVertexType(vertexContractionMap[vert], dagIn.VertexType(vert));
            }
            // assert(std::all_of(dag_in.vertices().begin(), dag_in.vertices().end(),
            //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return
            //         dag_in.VertexType(vert) ==  coarsened_dag.VertexType(vertex_contraction_map[vert]); })
            //                 && "Contracted vertices must be of the same type");
        }

        if constexpr (HasEdgeWeightsV<GraphTIn> && IsModifiableCdagCommEdgeV<GraphTOut>) {
            static_assert(std::is_same_v<ECommwT<GraphTIn>, ECommwT<GraphTOut>>,
                          "Edge weight type of in graph and out graph must be the same!");

            for (const auto &edge : Edges(coarsenedDag)) {
                coarsenedDag.SetEdgeCommWeight(edge, 0);
            }

            for (const auto &oriEdge : Edges(dagIn)) {
                VertexIdxT<GraphTOut> src = vertexContractionMap[Source(oriEdge, dagIn)];
                VertexIdxT<GraphTOut> tgt = vertexContractionMap[Traget(oriEdge, dagIn)];

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

    if constexpr (IsConstructableCdagV<GraphTOut>) {
        coarsenedDag = GraphTOut();

        const VertexIdxT<GraphTOut> numVertQuotient
            = (*std::max_element(vertexContractionMap.cbegin(), vertexContractionMap.cend())) + 1;

        for (VertexIdxT<GraphTOut> vert = 0; vert < numVertQuotient; ++vert) {
            coarsenedDag.AddVertex(0, 0, 0);
        }

        for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
            coarsenedDag.SetVertexWorkWeight(
                vertexContractionMap[vert],
                VWorkAccMethod()(coarsenedDag.VertexWorkWeight(vertexContractionMap[vert]), dagIn.VertexWorkWeight(vert)));

            coarsenedDag.SetVertexCommWeight(
                vertexContractionMap[vert],
                VCommAccMethod()(coarsenedDag.VertexCommWeight(vertexContractionMap[vert]), dagIn.VertexCommWeight(vert)));

            coarsenedDag.SetVertexMemWeight(
                vertexContractionMap[vert],
                VMemAccMethod()(coarsenedDag.VertexMemWeight(vertexContractionMap[vert]), dagIn.VertexMemWeight(vert)));
        }

        if constexpr (HasTypedVerticesV<GraphTIn> && IsModifiableCdagTypedVertexV<GraphTOut>) {
            static_assert(std::is_same_v<VTypeT<GraphTIn>, VTypeT<GraphTOut>>,
                          "Vertex type types of in graph and out graph must be the same!");

            for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
                coarsenedDag.SetVertexType(vertexContractionMap[vert], dagIn.VertexType(vert));
            }
            // assert(std::all_of(dag_in.vertices().begin(), dag_in.vertices().end(),
            //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return
            //         dag_in.VertexType(vert) ==  coarsened_dag.VertexType(vertex_contraction_map[vert]); })
            //                 && "Contracted vertices must be of the same type");
        }

        for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
            for (const VertexIdxT<GraphTIn> &chld : dagIn.Children(vert)) {
                if (vertexContractionMap[vert] == vertexContractionMap[chld]) {
                    continue;
                }

                if constexpr (HasEdgeWeightsV<GraphTIn> && IsModifiableCdagCommEdgeV<GraphTOut>) {
                    static_assert(std::is_same_v<ECommwT<GraphTIn>, ECommwT<GraphTOut>>,
                                  "Edge weight type of in graph and out graph must be the same!");

                    EdgeDescT<GraphTIn> ori_edge = EdgeDesc(vert, chld, dagIn).first;
                    const auto pair = EdgeDesc(vertexContractionMap[vert], vertexContractionMap[chld], coarsenedDag);
                    if (pair.second) {
                        coarsenedDag.SetEdgeCommWeight(
                            pair.first, ECommAccMethod()(coarsenedDag.EdgeCommWeight(pair.first), dagIn.EdgeCommWeight(ori_edge)));
                    } else {
                        coarsenedDag.AddEdge(
                            vertexContractionMap[vert], vertexContractionMap[chld], dagIn.EdgeCommWeight(ori_edge));
                    }
                } else {
                    if (not Edge(vertexContractionMap[vert], vertexContractionMap[chld], coarsenedDag)) {
                        coarsenedDag.AddEdge(vertexContractionMap[vert], vertexContractionMap[chld]);
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
          typename VWorkAccMethod = AccSum<VWorkwT<GraphTIn>>,
          typename VCommAccMethod = AccSum<VCommwT<GraphTIn>>,
          typename VMemAccMethod = AccSum<VMemwT<GraphTIn>>,
          typename ECommAccMethod = AccSum<ECommwT<GraphTIn>>>
bool ConstructCoarseDag(const GraphTIn &dagIn, GraphTOut &coarsenedDag, std::vector<VertexIdxT<GraphTOut>> &vertexContractionMap) {
    if constexpr (IsCompactSparseGraphReorderV<GraphTOut>) {
        static_assert(IsDirectedGraphV<GraphTIn> && IsDirectedGraphV<GraphTOut>,
                      "Graph types need to satisfy the is_directed_graph concept.");
        static_assert(IsComputationalDagV<GraphTIn>, "GraphTIn must be a computational DAG");
        static_assert(IsConstructableCdagV<GraphTOut> || IsDirectConstructableCdagV<GraphTOut>,
                      "GraphTOut must be a (direct) constructable computational DAG");

        assert(check_valid_contraction_map<GraphTOut>(vertexContractionMap));

        if (vertexContractionMap.size() == 0) {
            coarsenedDag = GraphTOut();
            return true;
        }
        const VertexIdxT<GraphTOut> numVertQuotient
            = (*std::max_element(vertexContractionMap.cbegin(), vertexContractionMap.cend())) + 1;

        std::set<std::pair<VertexIdxT<GraphTOut>, VertexIdxT<GraphTOut>>> quotient_edges;

        for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
            for (const VertexIdxT<GraphTIn> &chld : dagIn.Children(vert)) {
                if (vertexContractionMap[vert] == vertexContractionMap[chld]) {
                    continue;
                }
                quotient_edges.emplace(vertexContractionMap[vert], vertexContractionMap[chld]);
            }
        }

        coarsenedDag = GraphTOut(numVertQuotient, quotient_edges);

        const auto &pushforwardMap = coarsenedDag.get_pushforward_permutation();
        std::vector<VertexIdxT<GraphTOut>> combinedExpansionMap(dagIn.NumVertices());
        for (const auto &vert : dagIn.Vertices()) {
            combinedExpansionMap[vert] = pushforwardMap[vertexContractionMap[vert]];
        }

        if constexpr (HasVertexWeightsV<GraphTIn> && IsModifiableCdagVertexV<GraphTOut>) {
            static_assert(std::is_same_v<VWorkwT<GraphTIn>, VWorkwT<GraphTOut>>,
                          "Work weight types of in-graph and out-graph must be the same.");
            static_assert(std::is_same_v<VCommwT<GraphTIn>, VCommwT<GraphTOut>>,
                          "Vertex communication types of in-graph and out-graph must be the same.");
            static_assert(std::is_same_v<VMemwT<GraphTIn>, VMemwT<GraphTOut>>,
                          "Memory weight types of in-graph and out-graph must be the same.");

            for (const VertexIdxT<GraphTIn> &vert : coarsenedDag.Vertices()) {
                coarsenedDag.SetVertexWorkWeight(vert, 0);
                coarsenedDag.SetVertexCommWeight(vert, 0);
                coarsenedDag.SetVertexMemWeight(vert, 0);
            }

            for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
                coarsenedDag.SetVertexWorkWeight(
                    vertexContractionMap[vert],
                    VWorkAccMethod()(coarsenedDag.VertexWorkWeight(combinedExpansionMap[vert]), dagIn.VertexWorkWeight(vert)));

                coarsenedDag.SetVertexCommWeight(
                    vertexContractionMap[vert],
                    VCommAccMethod()(coarsenedDag.VertexCommWeight(combinedExpansionMap[vert]), dagIn.VertexCommWeight(vert)));

                coarsenedDag.SetVertexMemWeight(
                    vertexContractionMap[vert],
                    VMemAccMethod()(coarsenedDag.VertexMemWeight(combinedExpansionMap[vert]), dagIn.VertexMemWeight(vert)));
            }
        }

        if constexpr (HasTypedVerticesV<GraphTIn> && IsModifiableCdagTypedVertexV<GraphTOut>) {
            static_assert(std::is_same_v<VTypeT<GraphTIn>, VTypeT<GraphTOut>>,
                          "Vertex type types of in graph and out graph must be the same!");

            for (const VertexIdxT<GraphTIn> &vert : dagIn.Vertices()) {
                coarsenedDag.SetVertexType(vertexContractionMap[vert], dagIn.VertexType(vert));
            }
            // assert(std::all_of(dag_in.vertices().begin(), dag_in.vertices().end(),
            //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return
            //         dag_in.VertexType(vert) ==  coarsened_dag.VertexType(vertex_contraction_map[vert]); })
            //                 && "Contracted vertices must be of the same type");
        }

        if constexpr (HasEdgeWeightsV<GraphTIn> && HasEdgeWeightsV<GraphTOut>) {
            static_assert(std::is_same_v<ECommwT<GraphTIn>, ECommwT<GraphTOut>>,
                          "Edge weight type of in graph and out graph must be the same!");

            for (const auto &oriEdge : Edges(dagIn)) {
                VertexIdxT<GraphTOut> src = vertexContractionMap[Source(oriEdge, dagIn)];
                VertexIdxT<GraphTOut> tgt = vertexContractionMap[Traget(oriEdge, dagIn)];

                if (src == tgt) {
                    continue;
                }

                coarsenedDag.SetEdgeCommWeight(src, tgt, 0);
            }

            for (const auto &oriEdge : Edges(dagIn)) {
                VertexIdxT<GraphTOut> src = vertexContractionMap[Source(oriEdge, dagIn)];
                VertexIdxT<GraphTOut> tgt = vertexContractionMap[Traget(oriEdge, dagIn)];

                if (src == tgt) {
                    continue;
                }

                const auto contEdge = coarsenedDag.edge(pushforwardMap[src], pushforwardMap[tgt]);
                assert(Source(contEdge, coarsenedDag) == pushforwardMap[src]
                       && Traget(contEdge, coarsenedDag) == pushforwardMap[tgt]);
                coarsenedDag.SetEdgeCommWeight(
                    src, tgt, ECommAccMethod()(coarsenedDag.EdgeCommWeight(contEdge), dagIn.EdgeCommWeight(oriEdge)));
            }
        }

        std::swap(vertexContractionMap, combinedExpansionMap);
        return true;
    } else {
        return ConstructCoarseDag<GraphTIn, GraphTOut, VWorkAccMethod, VCommAccMethod, VMemAccMethod, ECommAccMethod>(
            dagIn, coarsenedDag, static_cast<const std::vector<VertexIdxT<GraphTOut>> &>(vertexContractionMap));
    }
}

template <typename GraphTIn>
bool CheckValidExpansionMap(const std::vector<std::vector<VertexIdxT<GraphTIn>>> &vertexExpansionMap) {
    std::size_t cntr = 0;

    std::vector<bool> preImage;
    for (const std::vector<VertexIdxT<GraphTIn>> &group : vertexExpansionMap) {
        if (group.size() == 0) {
            return false;
        }

        for (const VertexIdxT<GraphTIn> vert : group) {
            if (vert < static_cast<VertexIdxT<GraphTIn>>(0)) {
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
std::vector<std::vector<VertexIdxT<GraphTIn>>> InvertVertexContractionMap(
    const std::vector<VertexIdxT<GraphTOut>> &vertexContractionMap) {
    assert(CheckValidContractionMap<GraphTOut>(vertexContractionMap));

    VertexIdxT<GraphTOut> numVert
        = vertexContractionMap.size() == 0 ? 0 : *std::max_element(vertexContractionMap.cbegin(), vertexContractionMap.cend()) + 1;

    std::vector<std::vector<VertexIdxT<GraphTIn>>> expansionMap(numVert);

    for (std::size_t i = 0; i < vertexContractionMap.size(); ++i) {
        expansionMap[vertexContractionMap[i]].push_back(i);
    }

    return expansionMap;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<VertexIdxT<GraphTOut>> InvertVertexExpansionMap(const std::vector<std::vector<VertexIdxT<GraphTIn>>> &vertexExpansionMap) {
    assert(CheckValidExpansionMap<GraphTIn>(vertexExpansionMap));

    VertexIdxT<GraphTIn> numVert = 0;
    for (const auto &group : vertexExpansionMap) {
        for (const VertexIdxT<GraphTIn> &vert : group) {
            numVert = std::max(numVert, vert + 1);
        }
    }

    std::vector<VertexIdxT<GraphTOut>> vertexContractionMap(numVert);
    for (std::size_t i = 0; i < vertexExpansionMap.size(); i++) {
        for (const VertexIdxT<GraphTIn> &vert : vertexExpansionMap[i]) {
            vertexContractionMap[vert] = static_cast<VertexIdxT<GraphTOut>>(i);
        }
    }

    return vertexContractionMap;
}

template <typename GraphTIn>
void ReorderExpansionMap(const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &vertexExpansionMap) {
    assert(CheckValidExpansionMap<GraphTIn>(vertexExpansionMap));

    std::vector<std::size_t> vertexContractionMap(graph.NumVertices());
    for (std::size_t i = 0; i < vertexExpansionMap.size(); i++) {
        for (const VertexIdxT<GraphTIn> &vert : vertexExpansionMap[i]) {
            vertexContractionMap[vert] = i;
        }
    }

    std::vector<std::size_t> prec(vertexExpansionMap.size(), 0);
    for (const auto &vert : graph.vertices()) {
        for (const auto &par : graph.Parents(vert)) {
            if (vertexContractionMap.at(par) != vertexContractionMap.at(vert)) {
                prec[vertexContractionMap.at(vert)] += 1;
            }
        }
    }

    for (auto &comp : vertexExpansionMap) {
        std::nth_element(comp.begin(), comp.begin(), comp.end());
    }

    auto cmp = [&vertexExpansionMap](const std::size_t &lhs, const std::size_t &rhs) {
        return vertexExpansionMap[lhs] > vertexExpansionMap[rhs];    // because priority queue is a max_priority queue
    };

    std::priority_queue<std::size_t, std::vector<std::size_t>, decltype(cmp)> ready(cmp);
    std::vector<std::size_t> topOrder;
    topOrder.reserve(vertexExpansionMap.size());
    for (std::size_t i = 0; i < vertexExpansionMap.size(); ++i) {
        if (prec[i] == 0) {
            ready.emplace(i);
        }
    }

    while (!ready.empty()) {
        const std::size_t nextGroup = ready.top();
        ready.pop();
        topOrder.emplace_back(nextGroup);

        for (const auto &vert : vertexExpansionMap[nextGroup]) {
            for (const auto &chld : graph.Children(vert)) {
                if (vertexContractionMap.at(vert) != vertexContractionMap.at(chld)) {
                    prec[vertexContractionMap.at(chld)] -= 1;
                    if (prec[vertexContractionMap.at(chld)] == 0) {
                        ready.emplace(vertexContractionMap.at(chld));
                    }
                }
            }
        }
    }
    assert(topOrder.size() == vertexExpansionMap.size());

    inverse_permute_inplace(vertexExpansionMap, topOrder);

    return;
}

template <typename GraphTIn, typename GraphTOut>
bool PullBackSchedule(const BspSchedule<GraphTIn> &scheduleIn,
                      const std::vector<std::vector<VertexIdxT<GraphTIn>>> &vertexMap,
                      BspSchedule<GraphTOut> &scheduleOut) {
    for (unsigned v = 0; v < vertexMap.size(); ++v) {
        const auto proc = scheduleIn.AssignedProcessor(v);
        const auto step = scheduleIn.AssignedSuperstep(v);

        for (const auto &u : vertexMap[v]) {
            scheduleOut.setAssignedSuperstep(u, step);
            scheduleOut.SetAssignedProcessor(u, proc);
        }
    }

    return true;
}

template <typename GraphTIn, typename GraphTOut>
bool PullBackSchedule(const BspSchedule<GraphTIn> &scheduleIn,
                      const std::vector<VertexIdxT<GraphTOut>> &reverseVertexMap,
                      BspSchedule<GraphTOut> &scheduleOut) {
    for (unsigned idx = 0; idx < reverseVertexMap.size(); ++idx) {
        const auto &v = reverseVertexMap[idx];

        scheduleOut.setAssignedSuperstep(idx, scheduleIn.AssignedSuperstep(v));
        scheduleOut.SetAssignedProcessor(idx, scheduleIn.AssignedProcessor(v));
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
