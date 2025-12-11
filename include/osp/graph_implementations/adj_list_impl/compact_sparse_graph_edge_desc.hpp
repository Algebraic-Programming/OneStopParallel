/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/
#pragma once

#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"

namespace osp {

template <bool KeepVertexOrder,
          bool UseWorkWeights = false,
          bool UseCommWeights = false,
          bool UseMemWeights = false,
          bool UseEdgeCommWeights = false,
          bool UseVertTypes = false,
          typename VertT = std::size_t,
          typename EdgeT = std::size_t,
          typename WorkWeightType = unsigned,
          typename CommWeightType = unsigned,
          typename MemWeightType = unsigned,
          typename ECommWeightType = unsigned,
          typename VertexTypeTemplateType = unsigned>
class CompactSparseGraphEdgeDesc : public CompactSparseGraph<KeepVertexOrder,
                                                             UseWorkWeights,
                                                             UseCommWeights,
                                                             UseMemWeights,
                                                             UseVertTypes,
                                                             VertT,
                                                             EdgeT,
                                                             WorkWeightType,
                                                             CommWeightType,
                                                             MemWeightType,
                                                             VertexTypeTemplateType> {
  private:
    using ThisT = CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                             UseWorkWeights,
                                             UseCommWeights,
                                             UseMemWeights,
                                             UseEdgeCommWeights,
                                             UseVertTypes,
                                             VertT,
                                             EdgeT,
                                             WorkWeightType,
                                             CommWeightType,
                                             MemWeightType,
                                             ECommWeightType,
                                             VertexTypeTemplateType>;
    using BaseT = CompactSparseGraph<KeepVertexOrder,
                                     UseWorkWeights,
                                     UseCommWeights,
                                     UseMemWeights,
                                     UseVertTypes,
                                     VertT,
                                     EdgeT,
                                     WorkWeightType,
                                     CommWeightType,
                                     MemWeightType,
                                     VertexTypeTemplateType>;

  public:
    using vertex_idx = typename BaseT::VertexIdx;

    using VertexWorkWeightType = typename BaseT::VertexWorkWeightType;
    using VertexCommWeightType = typename BaseT::VertexCommWeightType;
    using VertexMemWeightType = typename BaseT::VertexMemWeightType;
    using VertexTypeType = typename BaseT::VertexTypeType;

    using DirectedEdgeDescriptor = EdgeT;
    using EdgeCommWeightType = ECommWeightType;

  protected:
    std::vector<EdgeCommWeightType> edgeCommWeights_;

    class InEdgesRange {
      private:
        const vertex_idx tgtVert_;
        const typename BaseT::CompactParentEdges::ParentRange parRange_;
        const typename BaseT::CompactChildrenEdges &cscOutEdges_;

        class InEdgesIterator {
          public:
            using iterator_category = std::bidirectional_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = EdgeT;
            using pointer = vertex_idx *;
            using reference = EdgeT &;

          private:
            const vertex_idx targetVert_;
            const typename BaseT::CompactChildrenEdges &cscOutEdges_;

            typename std::vector<vertex_idx>::const_iterator current_;

          public:
            InEdgesIterator(const vertex_idx &targetVert,
                            const typename BaseT::CompactChildrenEdges &cscOutEdges,
                            const typename std::vector<vertex_idx>::const_iterator start)
                : targetVert_(targetVert), cscOutEdges_(cscOutEdges), current_(start) {};
            InEdgesIterator(const InEdgesIterator &other)
                : targetVert_(other.targetVert_), cscOutEdges_(other.cscOutEdges_), current_(other.current_) {};

            InEdgesIterator &operator=(const InEdgesIterator &other) {
                if (this != &other) {
                    targetVert_ = other.targetVert_;
                    cscOutEdges_ = other.cscOutEdges_;
                    current_ = other.current_;
                }
                return *this;
            };

            inline value_type operator*() const {
                const vertex_idx srcVert = *current_;
                typename BaseT::CompactChildrenEdges::ChildrenRange range = cscOutEdges_.Children(srcVert);

                assert(std::binary_search(range.Cbegin(), range.Cend(), targetVert_));
                auto it = std::lower_bound(range.Cbegin(), range.Cend(), targetVert_);

                EdgeT diff = static_cast<EdgeT>(std::distance(range.Cbegin(), it));
                EdgeT edgeDescVal = cscOutEdges_.ChildrenIndxBegin(srcVert) + diff;

                return edgeDescVal;
            };

            inline InEdgesIterator &operator++() {
                ++current_;
                return *this;
            };

            inline InEdgesIterator operator++(int) {
                InEdgesIterator temp = *this;
                ++(*this);
                return temp;
            };

            inline InEdgesIterator &operator--() {
                --current_;
                return *this;
            };

            inline InEdgesIterator operator--(int) {
                InEdgesIterator temp = *this;
                --(*this);
                return temp;
            };

            inline bool operator==(const InEdgesIterator &other) const { return current_ == other.current_; };

            inline bool operator!=(const InEdgesIterator &other) const { return !(*this == other); };

            inline bool operator<=(const InEdgesIterator &other) const { return current_ <= other.current_; };

            inline bool operator<(const InEdgesIterator &other) const { return (*this <= other) && (*this != other); };

            inline bool operator>=(const InEdgesIterator &other) const { return (!(*this <= other)) || (*this == other); };

            inline bool operator>(const InEdgesIterator &other) const { return !(*this <= other); };
        };

      public:
        InEdgesRange() = default;
        InEdgesRange(const InEdgesRange &other) = default;
        InEdgesRange(InEdgesRange &&other) = default;
        InEdgesRange &operator=(const InEdgesRange &other) = default;
        InEdgesRange &operator=(InEdgesRange &&other) = default;
        virtual ~InEdgesRange() = default;

        InEdgesRange(const vertex_idx &tgtVert, const ThisT &graph, const typename BaseT::CompactChildrenEdges &cscOutEdges)
            : tgtVert_(tgtVert), parRange_(graph.Parents(tgtVert)), cscOutEdges_(cscOutEdges) {};

        inline auto Cbegin() const { return InEdgesIterator(tgtVert_, cscOutEdges_, parRange_.Cbegin()); };

        inline auto Cend() const { return InEdgesIterator(tgtVert_, cscOutEdges_, parRange_.Cend()); };

        inline auto begin() const { return Cbegin(); };

        inline auto end() const { return Cend(); };
    };

  public:
    CompactSparseGraphEdgeDesc() = default;
    CompactSparseGraphEdgeDesc(const CompactSparseGraphEdgeDesc &other) = default;
    CompactSparseGraphEdgeDesc(CompactSparseGraphEdgeDesc &&other) = default;
    CompactSparseGraphEdgeDesc &operator=(const CompactSparseGraphEdgeDesc &other) = default;
    CompactSparseGraphEdgeDesc &operator=(CompactSparseGraphEdgeDesc &&other) = default;
    virtual ~CompactSparseGraphEdgeDesc() = default;

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices, const EdgeListType &edges) : BaseT(numVertices, edges) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::NumEdges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices, const EdgeListType &edges, const std::vector<VertexWorkWeightType> &ww)
        : BaseT(numVertices, edges, ww) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices, const EdgeListType &edges, const std::vector<VertexWorkWeightType> &&ww)
        : BaseT(numVertices, edges, std::move(ww)) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices,
                               const EdgeListType &edges,
                               std::vector<VertexWorkWeightType> &ww,
                               std::vector<VertexCommWeightType> &cw)
        : BaseT(numVertices, edges, ww, cw) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices,
                               const EdgeListType &edges,
                               std::vector<VertexWorkWeightType> &&ww,
                               std::vector<VertexCommWeightType> &&cw)
        : BaseT(numVertices, edges, std::move(ww), std::move(cw)) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices,
                               const EdgeListType &edges,
                               const std::vector<VertexWorkWeightType> &ww,
                               const std::vector<VertexCommWeightType> &cw,
                               const std::vector<VertexMemWeightType> &mw)
        : BaseT(numVertices, edges, ww, cw, mw) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices,
                               const EdgeListType &edges,
                               const std::vector<VertexWorkWeightType> &&ww,
                               const std::vector<VertexCommWeightType> &&cw,
                               const std::vector<VertexMemWeightType> &&mw)
        : BaseT(numVertices, edges, std::move(ww), std::move(cw), std::move(mw)) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices,
                               const EdgeListType &edges,
                               const std::vector<VertexWorkWeightType> &ww,
                               const std::vector<VertexCommWeightType> &cw,
                               const std::vector<VertexMemWeightType> &mw,
                               const std::vector<VertexTypeType> &vt)
        : BaseT(numVertices, edges, ww, cw, mw, vt) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename EdgeListType>
    CompactSparseGraphEdgeDesc(vertex_idx numVertices,
                               const EdgeListType &edges,
                               const std::vector<VertexWorkWeightType> &&ww,
                               const std::vector<VertexCommWeightType> &&cw,
                               const std::vector<VertexMemWeightType> &&mw,
                               const std::vector<VertexTypeType> &&vt)
        : BaseT(numVertices, edges, std::move(ww), std::move(cw), std::move(mw), std::move(vt)) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }
    }

    template <typename GraphType>
    CompactSparseGraphEdgeDesc(const GraphType &graph) : BaseT(graph) {
        if constexpr (UseEdgeCommWeights) {
            edgeCommWeights_ = std::vector<EdgeCommWeightType>(BaseT::num_edges(), static_cast<EdgeCommWeightType>(0));
        }

        if constexpr (has_edge_weights_v<GraphType> && UseEdgeCommWeights) {
            for (const auto &edge : Edges(graph)) {
                const auto src = Source(edge, graph);
                const auto tgt = Target(edge, graph);
                set_edge_comm_weight(src, tgt, graph.edge_comm_weight(edge));
            }
        }
    }

    inline auto Edges() const { return IntegralRange<DirectedEdgeDescriptor>(BaseT::numberOfEdges_); };

    inline DirectedEdgeDescriptor Edge(const vertex_idx &src, const vertex_idx &tgt) const {
        typename BaseT::CompactChildrenEdges::ChildrenRange range = BaseT::cscOutEdges_.Children(src);

        assert(std::binary_search(range.Cbegin(), range.Cend(), tgt));
        auto it = std::lower_bound(range.Cbegin(), range.Cend(), tgt);

        DirectedEdgeDescriptor diff = static_cast<DirectedEdgeDescriptor>(std::distance(range.Cbegin(), it));
        DirectedEdgeDescriptor edgeDescVal = BaseT::cscOutEdges_.ChildrenIndxBegin(src) + diff;

        return edgeDescVal;
    };

    inline vertex_idx Source(const DirectedEdgeDescriptor &edge) const { return BaseT::cscOutEdges_.Source(edge); };

    inline vertex_idx Target(const DirectedEdgeDescriptor &edge) const { return BaseT::cscOutEdges_.Target(edge); };

    inline auto OutEdges(const vertex_idx &vert) const {
        return IntegralRange<DirectedEdgeDescriptor>(BaseT::cscOutEdges_.ChildrenIndxBegin(vert),
                                                     BaseT::cscOutEdges_.ChildrenIndxBegin(vert + 1));
    };

    inline auto InEdges(const vertex_idx &vert) const { return InEdgesRange(vert, *this, BaseT::cscOutEdges_); };

    template <typename RetT = EdgeCommWeightType>
    inline std::enable_if_t<UseEdgeCommWeights, RetT> EdgeCommWeight(const DirectedEdgeDescriptor &edge) const {
        return edgeCommWeights_[edge];
    }

    template <typename RetT = EdgeCommWeightType>
    inline std::enable_if_t<not UseEdgeCommWeights, RetT> EdgeCommWeight(const DirectedEdgeDescriptor &edge) const {
        return static_cast<RetT>(1);
    }

    template <typename RetT = void>
    inline std::enable_if_t<UseEdgeCommWeights, RetT> SetEdgeCommWeight(const vertex_idx &src,
                                                                        const vertex_idx &tgt,
                                                                        const EdgeCommWeightType eCommWeight) {
        if constexpr (KeepVertexOrder) {
            edgeCommWeights_[Edge(src, tgt)] = eCommWeight;
        } else {
            const vertex_idx internalSrc = BaseT::vertexPermutationFromOriginalToInternal_[src];
            const vertex_idx internalTgt = BaseT::vertexPermutationFromOriginalToInternal_[tgt];
            edgeCommWeights_[Edge(internalSrc, internalTgt)] = eCommWeight;
        }
    }

    template <typename RetT = void>
    inline std::enable_if_t<not UseEdgeCommWeights, RetT> SetEdgeCommWeight(const vertex_idx &src,
                                                                            const vertex_idx &tgt,
                                                                            const EdgeCommWeightType eCommWeight) {
        static_assert(UseEdgeCommWeights, "To set edge communication weight, graph type must allow edge communication weights.");
    }
};

template <bool KeepVertexOrder,
          bool UseWorkWeights,
          bool UseCommWeights,
          bool UseMemWeights,
          bool UseEdgeCommWeights,
          bool UseVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename ECommWeightType,
          typename VertexTypeTemplateType>
inline auto Edges(const CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                                   UseWorkWeights,
                                                   UseCommWeights,
                                                   UseMemWeights,
                                                   UseEdgeCommWeights,
                                                   UseVertTypes,
                                                   VertT,
                                                   EdgeT,
                                                   WorkWeightType,
                                                   CommWeightType,
                                                   MemWeightType,
                                                   ECommWeightType,
                                                   VertexTypeTemplateType> &graph) {
    return graph.Edges();
}

template <bool KeepVertexOrder,
          bool UseWorkWeights,
          bool UseCommWeights,
          bool UseMemWeights,
          bool UseEdgeCommWeights,
          bool UseVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename ECommWeightType,
          typename VertexTypeTemplateType>
inline auto OutEdges(VertexIdxT<CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                                           UseWorkWeights,
                                                           UseCommWeights,
                                                           UseMemWeights,
                                                           UseEdgeCommWeights,
                                                           UseVertTypes,
                                                           VertT,
                                                           EdgeT,
                                                           WorkWeightType,
                                                           CommWeightType,
                                                           MemWeightType,
                                                           ECommWeightType,
                                                           VertexTypeTemplateType>> v,
                     const CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                                      UseWorkWeights,
                                                      UseCommWeights,
                                                      UseMemWeights,
                                                      UseEdgeCommWeights,
                                                      UseVertTypes,
                                                      VertT,
                                                      EdgeT,
                                                      WorkWeightType,
                                                      CommWeightType,
                                                      MemWeightType,
                                                      ECommWeightType,
                                                      VertexTypeTemplateType> &graph) {
    return graph.OutEdges(v);
}

template <bool KeepVertexOrder,
          bool UseWorkWeights,
          bool UseCommWeights,
          bool UseMemWeights,
          bool UseEdgeCommWeights,
          bool UseVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename ECommWeightType,
          typename VertexTypeTemplateType>
inline auto InEdges(VertexIdxT<CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                                          UseWorkWeights,
                                                          UseCommWeights,
                                                          UseMemWeights,
                                                          UseEdgeCommWeights,
                                                          UseVertTypes,
                                                          VertT,
                                                          EdgeT,
                                                          WorkWeightType,
                                                          CommWeightType,
                                                          MemWeightType,
                                                          ECommWeightType,
                                                          VertexTypeTemplateType>> v,
                    const CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                                     UseWorkWeights,
                                                     UseCommWeights,
                                                     UseMemWeights,
                                                     UseEdgeCommWeights,
                                                     UseVertTypes,
                                                     VertT,
                                                     EdgeT,
                                                     WorkWeightType,
                                                     CommWeightType,
                                                     MemWeightType,
                                                     ECommWeightType,
                                                     VertexTypeTemplateType> &graph) {
    return graph.InEdges(v);
}

template <bool KeepVertexOrder,
          bool UseWorkWeights,
          bool UseCommWeights,
          bool UseMemWeights,
          bool UseEdgeCommWeights,
          bool UseVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename ECommWeightType,
          typename VertexTypeTemplateType>
inline VertexIdxT<CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                             UseWorkWeights,
                                             UseCommWeights,
                                             UseMemWeights,
                                             UseEdgeCommWeights,
                                             UseVertTypes,
                                             VertT,
                                             EdgeT,
                                             WorkWeightType,
                                             CommWeightType,
                                             MemWeightType,
                                             ECommWeightType,
                                             VertexTypeTemplateType>>
Source(const EdgeDescT<CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                                  UseWorkWeights,
                                                  UseCommWeights,
                                                  UseMemWeights,
                                                  UseEdgeCommWeights,
                                                  UseVertTypes,
                                                  VertT,
                                                  EdgeT,
                                                  WorkWeightType,
                                                  CommWeightType,
                                                  MemWeightType,
                                                  ECommWeightType,
                                                  VertexTypeTemplateType>> &edge,
       const CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                        UseWorkWeights,
                                        UseCommWeights,
                                        UseMemWeights,
                                        UseEdgeCommWeights,
                                        UseVertTypes,
                                        VertT,
                                        EdgeT,
                                        WorkWeightType,
                                        CommWeightType,
                                        MemWeightType,
                                        ECommWeightType,
                                        VertexTypeTemplateType> &graph) {
    return graph.Source(edge);
}

template <bool KeepVertexOrder,
          bool UseWorkWeights,
          bool UseCommWeights,
          bool UseMemWeights,
          bool UseEdgeCommWeights,
          bool UseVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename ECommWeightType,
          typename VertexTypeTemplateType>
inline VertexIdxT<CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                             UseWorkWeights,
                                             UseCommWeights,
                                             UseMemWeights,
                                             UseEdgeCommWeights,
                                             UseVertTypes,
                                             VertT,
                                             EdgeT,
                                             WorkWeightType,
                                             CommWeightType,
                                             MemWeightType,
                                             ECommWeightType,
                                             VertexTypeTemplateType>>
Target(const EdgeDescT<CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                                  UseWorkWeights,
                                                  UseCommWeights,
                                                  UseMemWeights,
                                                  UseEdgeCommWeights,
                                                  UseVertTypes,
                                                  VertT,
                                                  EdgeT,
                                                  WorkWeightType,
                                                  CommWeightType,
                                                  MemWeightType,
                                                  ECommWeightType,
                                                  VertexTypeTemplateType>> &edge,
       const CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                        UseWorkWeights,
                                        UseCommWeights,
                                        UseMemWeights,
                                        UseEdgeCommWeights,
                                        UseVertTypes,
                                        VertT,
                                        EdgeT,
                                        WorkWeightType,
                                        CommWeightType,
                                        MemWeightType,
                                        ECommWeightType,
                                        VertexTypeTemplateType> &graph) {
    return graph.Target(edge);
}

template <bool KeepVertexOrder,
          bool UseWorkWeights,
          bool UseCommWeights,
          bool UseMemWeights,
          bool UseEdgeCommWeights,
          bool UseVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename ECommWeightType,
          typename VertexTypeTemplateType>
struct is_Compact_Sparse_Graph<CompactSparseGraphEdgeDesc<KeepVertexOrder,
                                                          UseWorkWeights,
                                                          UseCommWeights,
                                                          UseMemWeights,
                                                          UseEdgeCommWeights,
                                                          UseVertTypes,
                                                          VertT,
                                                          EdgeT,
                                                          WorkWeightType,
                                                          CommWeightType,
                                                          MemWeightType,
                                                          ECommWeightType,
                                                          VertexTypeTemplateType>,
                               void> : std::true_type {};

template <bool UseWorkWeights,
          bool UseCommWeights,
          bool UseMemWeights,
          bool UseEdgeCommWeights,
          bool UseVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename ECommWeightType,
          typename VertexTypeTemplateType>
struct is_Compact_Sparse_Graph_reorder<CompactSparseGraphEdgeDesc<false,
                                                                  UseWorkWeights,
                                                                  UseCommWeights,
                                                                  UseMemWeights,
                                                                  UseEdgeCommWeights,
                                                                  UseVertTypes,
                                                                  VertT,
                                                                  EdgeT,
                                                                  WorkWeightType,
                                                                  CommWeightType,
                                                                  MemWeightType,
                                                                  ECommWeightType,
                                                                  VertexTypeTemplateType>,
                                       void> : std::true_type {};

static_assert(isCompactSparseGraphV<CompactSparseGraphEdgeDesc<true>>);
static_assert(isCompactSparseGraphV<CompactSparseGraphEdgeDesc<false>>);
static_assert(!isCompactSparseGraphReorderV<CompactSparseGraphEdgeDesc<true>>);
static_assert(isCompactSparseGraphReorderV<CompactSparseGraphEdgeDesc<false>>);

static_assert(hasVertexWeightsV<CompactSparseGraphEdgeDesc<true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_vertex_weights concept");

static_assert(hasVertexWeightsV<CompactSparseGraphEdgeDesc<false, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_vertex_weights concept");

static_assert(isDirectedGraphV<CompactSparseGraphEdgeDesc<false, false, false, false, false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed_graph concept");

static_assert(isDirectedGraphV<CompactSparseGraphEdgeDesc<false, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed_graph concept");

static_assert(isDirectedGraphV<CompactSparseGraphEdgeDesc<true, false, false, false, false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed_graph concept");

static_assert(isDirectedGraphV<CompactSparseGraphEdgeDesc<true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed_graph concept");

static_assert(isComputationalDagV<CompactSparseGraphEdgeDesc<false, true, true, true, false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computation_dag concept");

static_assert(isComputationalDagV<CompactSparseGraphEdgeDesc<true, true, true, true, false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computation_dag concept");

static_assert(isComputationalDagTypedVerticesV<CompactSparseGraphEdgeDesc<false, true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computation_dag with types concept");

static_assert(isComputationalDagTypedVerticesV<CompactSparseGraphEdgeDesc<true, true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computation_dag with types concept");

static_assert(isDirectedGraphEdgeDescV<CompactSparseGraphEdgeDesc<true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed graph edge descriptor concept.");

static_assert(isDirectedGraphEdgeDescV<CompactSparseGraphEdgeDesc<false>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the directed graph edge descriptor concept.");

static_assert(
    isComputationalDagTypedVerticesEdgeDescV<CompactSparseGraphEdgeDesc<false, true, true, true, true, true>>,
    "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computational_dag_typed_vertices_edge_desc_v with types concept");

static_assert(
    isComputationalDagTypedVerticesEdgeDescV<CompactSparseGraphEdgeDesc<true, true, true, true, true, true>>,
    "Compact_Sparse_Graph_EdgeDesc must satisfy the is_computational_dag_typed_vertices_edge_desc_v with types concept");

static_assert(hasEdgeWeightsV<CompactSparseGraphEdgeDesc<false, true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_edge_weights concept");

static_assert(hasEdgeWeightsV<CompactSparseGraphEdgeDesc<true, true, true, true, true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_edge_weights concept");

static_assert(hasHashableEdgeDescV<CompactSparseGraphEdgeDesc<true, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_hashable_edge_desc concept");

static_assert(hasHashableEdgeDescV<CompactSparseGraphEdgeDesc<false, true>>,
              "Compact_Sparse_Graph_EdgeDesc must satisfy the has_hashable_edge_desc concept");

using CSGE
    = CompactSparseGraphEdgeDesc<false, true, true, true, true, true, std::size_t, std::size_t, unsigned, unsigned, unsigned, unsigned, unsigned>;

}    // namespace osp
