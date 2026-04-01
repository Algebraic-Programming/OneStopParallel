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

#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <type_traits>
#include <vector>

#include "osp/auxiliary/math/math_helper.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/concepts/specific_graph_impl.hpp"
#include "osp/graph_algorithms/directed_graph_edge_view.hpp"
#include "osp/graph_implementations/integral_range.hpp"

namespace osp {

template <bool keepVertexOrder,
          bool useWorkWeights = false,
          bool useCommWeights = false,
          bool useMemWeights = false,
          bool useVertTypes = false,
          typename VertT = std::size_t,
          typename EdgeT = std::size_t,
          typename WorkWeightType = unsigned,
          typename CommWeightType = unsigned,
          typename MemWeightType = unsigned,
          typename VertexTypeTemplateType = unsigned>
class CompactSparseGraph {
    static_assert(std::is_integral<VertT>::value && std::is_integral<EdgeT>::value,
                  "Vertex and edge type must be of integral nature.");
    static_assert(std::is_arithmetic_v<WorkWeightType> && "Work weight must be of arithmetic type.");
    static_assert(std::is_arithmetic_v<CommWeightType> && "Communication weight must be of arithmetic type.");
    static_assert(std::is_arithmetic_v<MemWeightType> && "Memory weight must be of arithmetic type.");
    static_assert(std::is_integral_v<VertexTypeTemplateType> && "Vertex type type must be of integral type.");

  public:
    using VertexIdx = VertT;

    using VertexWorkWeightType = std::conditional_t<useWorkWeights, WorkWeightType, EdgeT>;
    using VertexCommWeightType = CommWeightType;
    using VertexMemWeightType = MemWeightType;
    using VertexTypeType = VertexTypeTemplateType;

    static bool constexpr verticesInTopOrder_ = true;
    static bool constexpr childrenInTopOrder_ = true;
    static bool constexpr childrenInVertexOrder_ = true;
    static bool constexpr parentsInTopOrder_ = true;
    static bool constexpr parentsInVertexOrder_ = true;

  private:
    using ThisT = CompactSparseGraph<keepVertexOrder,
                                     useWorkWeights,
                                     useCommWeights,
                                     useMemWeights,
                                     useVertTypes,
                                     VertT,
                                     EdgeT,
                                     WorkWeightType,
                                     CommWeightType,
                                     MemWeightType,
                                     VertexTypeTemplateType>;

  protected:
    class CompactParentEdges {
      private:
        // Compressed Sparse Row (CSR)
        std::vector<VertexIdx> csrEdgeParents_;
        std::vector<EdgeT> csrTargetPtr_;

      public:
        CompactParentEdges() = default;
        CompactParentEdges(const CompactParentEdges &other) = default;
        CompactParentEdges(CompactParentEdges &&other) = default;
        CompactParentEdges &operator=(const CompactParentEdges &other) = default;
        CompactParentEdges &operator=(CompactParentEdges &&other) = default;
        virtual ~CompactParentEdges() = default;

        CompactParentEdges(const std::vector<VertexIdx> &csrEdgeParents, const std::vector<EdgeT> &csrTargetPtr)
            : csrEdgeParents_(csrEdgeParents), csrTargetPtr_(csrTargetPtr) {};
        CompactParentEdges(std::vector<VertexIdx> &&csrEdgeParents, std::vector<EdgeT> &&csrTargetPtr)
            : csrEdgeParents_(std::move(csrEdgeParents)), csrTargetPtr_(std::move(csrTargetPtr)) {};

        inline EdgeT NumberOfParents(const VertexIdx v) const { return csrTargetPtr_[v + 1] - csrTargetPtr_[v]; }

        class ParentRange {
          private:
            const std::vector<VertexIdx> &csrEdgeParents_;
            const std::vector<EdgeT> &csrTargetPtr_;
            const VertexIdx vert_;

          public:
            ParentRange(const std::vector<VertexIdx> &csrEdgeParents, const std::vector<EdgeT> &csrTargetPtr, const VertexIdx vert)
                : csrEdgeParents_(csrEdgeParents), csrTargetPtr_(csrTargetPtr), vert_(vert) {};

            inline auto cbegin() const {
                auto it = csrEdgeParents_.cbegin();
                std::advance(it, csrTargetPtr_[vert_]);
                return it;
            }

            inline auto cend() const {
                auto it = csrEdgeParents_.cbegin();
                std::advance(it, csrTargetPtr_[vert_ + 1]);
                return it;
            }

            inline auto begin() const { return cbegin(); }

            inline auto end() const { return cend(); }

            inline auto crbegin() const {
                auto it = csrEdgeParents_.crbegin();
                std::advance(it, csrTargetPtr_[csrTargetPtr_.size() - 1] - csrTargetPtr_[vert_ + 1]);
                return it;
            };

            inline auto crend() const {
                auto it = csrEdgeParents_.crbegin();
                std::advance(it, csrTargetPtr_[csrTargetPtr_.size() - 1] - csrTargetPtr_[vert_]);
                return it;
            };

            inline auto rbegin() const { return crbegin(); };

            inline auto rend() const { return crend(); };
        };

        inline ParentRange Parents(const VertexIdx vert) const { return ParentRange(csrEdgeParents_, csrTargetPtr_, vert); }
    };

    class CompactChildrenEdges {
      private:
        // Compressed Sparse Column (CSC)
        std::vector<VertexIdx> cscEdgeChildren_;
        std::vector<EdgeT> cscSourcePtr_;

      public:
        CompactChildrenEdges() = default;
        CompactChildrenEdges(const CompactChildrenEdges &other) = default;
        CompactChildrenEdges(CompactChildrenEdges &&other) = default;
        CompactChildrenEdges &operator=(const CompactChildrenEdges &other) = default;
        CompactChildrenEdges &operator=(CompactChildrenEdges &&other) = default;
        virtual ~CompactChildrenEdges() = default;

        CompactChildrenEdges(const std::vector<VertexIdx> &cscEdgeChildren, const std::vector<EdgeT> &cscSourcePtr)
            : cscEdgeChildren_(cscEdgeChildren), cscSourcePtr_(cscSourcePtr) {};
        CompactChildrenEdges(std::vector<VertexIdx> &&cscEdgeChildren, std::vector<EdgeT> &&cscSourcePtr)
            : cscEdgeChildren_(std::move(cscEdgeChildren)), cscSourcePtr_(std::move(cscSourcePtr)) {};

        inline EdgeT NumberOfChildren(const VertexIdx v) const { return cscSourcePtr_[v + 1] - cscSourcePtr_[v]; }

        inline VertexIdx Source(const EdgeT &indx) const {
            auto it = std::upper_bound(cscSourcePtr_.cbegin(), cscSourcePtr_.cend(), indx);
            VertexIdx src = static_cast<VertexIdx>(std::distance(cscSourcePtr_.cbegin(), it) - 1);
            return src;
        };

        inline VertexIdx Target(const EdgeT &indx) const { return cscEdgeChildren_[indx]; };

        inline EdgeT ChildrenIndxBegin(const VertexIdx &vert) const { return cscSourcePtr_[vert]; };

        class ChildrenRange {
          private:
            const std::vector<VertexIdx> &cscEdgeChildren_;
            const std::vector<EdgeT> &cscSourcePtr_;
            const VertexIdx vert_;

          public:
            ChildrenRange(const std::vector<VertexIdx> &cscEdgeChildren,
                          const std::vector<EdgeT> &cscSourcePtr,
                          const VertexIdx vert)
                : cscEdgeChildren_(cscEdgeChildren), cscSourcePtr_(cscSourcePtr), vert_(vert) {};

            inline auto cbegin() const {
                auto it = cscEdgeChildren_.cbegin();
                std::advance(it, cscSourcePtr_[vert_]);
                return it;
            };

            inline auto cend() const {
                auto it = cscEdgeChildren_.cbegin();
                std::advance(it, cscSourcePtr_[vert_ + 1]);
                return it;
            };

            inline auto begin() const { return cbegin(); };

            inline auto end() const { return cend(); };

            inline auto crbegin() const {
                auto it = cscEdgeChildren_.crbegin();
                std::advance(it, cscSourcePtr_[cscSourcePtr_.size() - 1] - cscSourcePtr_[vert_ + 1]);
                return it;
            };

            inline auto crend() const {
                auto it = cscEdgeChildren_.crbegin();
                std::advance(it, cscSourcePtr_[cscSourcePtr_.size() - 1] - cscSourcePtr_[vert_]);
                return it;
            };

            inline auto rbegin() const { return crbegin(); };

            inline auto rend() const { return crend(); };
        };

        inline ChildrenRange Children(const VertexIdx vert) const { return ChildrenRange(cscEdgeChildren_, cscSourcePtr_, vert); }
    };

    VertexIdx numberOfVertices_ = static_cast<VertT>(0);
    EdgeT numberOfEdges_ = static_cast<EdgeT>(0);

    CompactParentEdges csrInEdges_;
    CompactChildrenEdges cscOutEdges_;

    VertexTypeType numberOfVertexTypes_ = static_cast<VertexTypeType>(1);

    std::vector<VertexWorkWeightType> vertWorkWeights_;
    std::vector<VertexCommWeightType> vertCommWeights_;
    std::vector<VertexMemWeightType> vertMemWeights_;
    std::vector<VertexTypeType> vertTypes_;

    std::vector<VertexIdx> vertexPermutationFromInternalToOriginal_;
    std::vector<VertexIdx> vertexPermutationFromOriginalToInternal_;

    template <typename RetT = void>
    std::enable_if_t<not useVertTypes, RetT> UpdateNumVertexTypes() {
        numberOfVertexTypes_ = static_cast<VertexTypeType>(1);
    }

    template <typename RetT = void>
    std::enable_if_t<useVertTypes, RetT> UpdateNumVertexTypes() {
        numberOfVertexTypes_ = static_cast<VertexTypeType>(1);
        for (const auto vt : vertTypes_) {
            numberOfVertexTypes_ = std::max(numberOfVertexTypes_, vt);
        }
    }

  public:
    CompactSparseGraph() = default;
    CompactSparseGraph(const CompactSparseGraph &other) = default;
    CompactSparseGraph(CompactSparseGraph &&other) = default;
    CompactSparseGraph &operator=(const CompactSparseGraph &other) = default;
    CompactSparseGraph &operator=(CompactSparseGraph &&other) = default;
    virtual ~CompactSparseGraph() = default;

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices, const EdgeListType &edges)
        : numberOfVertices_(numVertices), numberOfEdges_(static_cast<EdgeT>(edges.size())) {
        static_assert(IsContainerOf<EdgeListType, std::pair<VertexIdx, VertexIdx>>::value
                      || IsEdgeListType<EdgeListType, VertexIdx, EdgeT>::value);

        assert((0 <= numVertices) && "Number of vertices must be non-negative.");
        assert((edges.size() < static_cast<size_t>(std::numeric_limits<EdgeT>::max()))
               && "Number of edges must be strictly smaller than the maximally representable number.");

        if constexpr (IsContainerOf<EdgeListType, std::pair<VertexIdx, VertexIdx>>::value) {
            assert(std::all_of(edges.begin(),
                               edges.end(),
                               [numVertices](const auto &edge) {
                                   return (0 <= edge.first) && (edge.first < numVertices) && (0 <= edge.second)
                                          && (edge.second < numVertices);
                               })
                   && "Source and target of edges must be non-negative and less than the number of vertices.");
        }

        if constexpr (isEdgeListTypeV<EdgeListType, VertexIdx, EdgeT>) {
            assert(std::all_of(edges.begin(),
                               edges.end(),
                               [numVertices](const auto &edge) {
                                   return (0 <= edge.source_) && (edge.source_ < numVertices) && (0 <= edge.target_)
                                          && (edge.target_ < numVertices);
                               })
                   && "Source and target of edges must be non-negative and less than the number of vertices.");
        }

        if constexpr (keepVertexOrder) {
            if constexpr (IsContainerOf<EdgeListType, std::pair<VertexIdx, VertexIdx>>::value) {
                assert(std::all_of(edges.begin(), edges.end(), [](const auto &edge) { return edge.first < edge.second; })
                       && "Vertex order must be a topological order.");
            }
            if constexpr (isEdgeListTypeV<EdgeListType, VertexIdx, EdgeT>) {
                assert(std::all_of(edges.begin(), edges.end(), [](const auto &edge) { return edge.source_ < edge.target_; })
                       && "Vertex order must be a topological order.");
            }
        }

        if constexpr (useWorkWeights) {
            vertWorkWeights_ = std::vector<VertexWorkWeightType>(NumVertices(), 1);
        }
        if constexpr (useCommWeights) {
            vertCommWeights_ = std::vector<VertexCommWeightType>(NumVertices(), 0);
        }
        if constexpr (useMemWeights) {
            vertMemWeights_ = std::vector<VertexMemWeightType>(NumVertices(), 0);
        }
        if constexpr (useVertTypes) {
            numberOfVertexTypes_ = 1;
            vertTypes_ = std::vector<VertexTypeType>(NumVertices(), 0);
        }
        if constexpr (!keepVertexOrder) {
            vertexPermutationFromInternalToOriginal_.reserve(NumVertices());
            vertexPermutationFromOriginalToInternal_.reserve(NumVertices());
        }

        // Construction
        std::vector<std::vector<VertexIdx>> childrenTmp(NumVertices());
        std::vector<EdgeT> numParentsTmp(NumVertices(), 0);

        if constexpr (IsContainerOf<EdgeListType, std::pair<VertexIdx, VertexIdx>>::value) {
            for (const auto &edge : edges) {
                childrenTmp[edge.first].push_back(edge.second);
                numParentsTmp[edge.second]++;
            }
        }
        if constexpr (isEdgeListTypeV<EdgeListType, VertexIdx, EdgeT>) {
            for (const auto &edge : edges) {
                childrenTmp[edge.source_].push_back(edge.target_);
                numParentsTmp[edge.target_]++;
            }
        }

        std::vector<VertexIdx> cscEdgeChildren;
        cscEdgeChildren.reserve(NumEdges());
        std::vector<EdgeT> cscSourcePtr(NumVertices() + 1);
        std::vector<VertexIdx> csrEdgeParents(NumEdges());
        std::vector<EdgeT> csrTargetPtr;
        csrTargetPtr.reserve(NumVertices() + 1);

        if constexpr (keepVertexOrder) {
            for (VertexIdx vert = 0; vert < NumVertices(); ++vert) {
                cscSourcePtr[vert] = static_cast<EdgeT>(cscEdgeChildren.size());

                std::sort(childrenTmp[vert].begin(), childrenTmp[vert].end());
                for (const auto &chld : childrenTmp[vert]) {
                    cscEdgeChildren.emplace_back(chld);
                }
            }
            cscSourcePtr[NumVertices()] = static_cast<EdgeT>(cscEdgeChildren.size());

            csrTargetPtr = std::vector<EdgeT>(NumVertices() + 1, 0);
            for (std::size_t i = 0U; i < numParentsTmp.size(); ++i) {
                csrTargetPtr[i + 1] = csrTargetPtr[i] + numParentsTmp[i];
            }

            std::vector<EdgeT> offset = csrTargetPtr;
            for (VertexIdx vert = 0; vert < NumVertices(); ++vert) {
                for (const auto &chld : childrenTmp[vert]) {
                    csrEdgeParents[offset[chld]++] = vert;
                }
            }

        } else {
            std::vector<std::vector<VertexIdx>> parentsTmp(NumVertices());

            if constexpr (IsContainerOf<EdgeListType, std::pair<VertexIdx, VertexIdx>>::value) {
                for (const auto &edge : edges) {
                    parentsTmp[edge.second].push_back(edge.first);
                }
            }
            if constexpr (isEdgeListTypeV<EdgeListType, VertexIdx, EdgeT>) {
                for (const auto &edge : edges) {
                    parentsTmp[edge.target_].push_back(edge.source_);
                }
            }

            // Generating modified Gorder topological order cf. "Speedup Graph Processing by Graph Ordering" by Hao Wei, Jeffrey
            // Xu Yu, Can Lu, and Xuemin Lin
            const double decay = 8.0;

            std::vector<EdgeT> precRemaining = numParentsTmp;
            std::vector<double> priorities(NumVertices(), 0.0);

            auto vCmp = [&priorities, &childrenTmp](const VertexIdx &lhs, const VertexIdx &rhs) {
                return (priorities[lhs] < priorities[rhs])
                       || ((priorities[lhs] <= priorities[rhs]) && (childrenTmp[lhs].size() < childrenTmp[rhs].size()))
                       || ((priorities[lhs] <= priorities[rhs]) && (childrenTmp[lhs].size() == childrenTmp[rhs].size())
                           && (lhs > rhs));
            };

            std::priority_queue<VertexIdx, std::vector<VertexIdx>, decltype(vCmp)> readyQ(vCmp);
            for (VertexIdx vert = 0; vert < NumVertices(); ++vert) {
                if (precRemaining[vert] == 0) {
                    readyQ.push(vert);
                }
            }

            while (!readyQ.empty()) {
                VertexIdx vert = readyQ.top();
                readyQ.pop();

                double pos = static_cast<double>(vertexPermutationFromInternalToOriginal_.size());
                pos /= decay;

                vertexPermutationFromInternalToOriginal_.push_back(vert);

                // update priorities
                for (VertexIdx chld : childrenTmp[vert]) {
                    priorities[chld] = LogSumExp(priorities[chld], pos);
                }
                for (VertexIdx par : parentsTmp[vert]) {
                    for (VertexIdx sibling : childrenTmp[par]) {
                        priorities[sibling] = LogSumExp(priorities[sibling], pos);
                    }
                }
                for (VertexIdx chld : childrenTmp[vert]) {
                    for (VertexIdx couple : parentsTmp[chld]) {
                        priorities[couple] = LogSumExp(priorities[couple], pos);
                    }
                }

                // update constraints and push to queue
                for (VertexIdx chld : childrenTmp[vert]) {
                    --precRemaining[chld];
                    if (precRemaining[chld] == 0) {
                        readyQ.push(chld);
                    }
                }
            }

            assert(vertexPermutationFromInternalToOriginal_.size() == static_cast<size_t>(NumVertices()));

            // constructing the csr and csc
            vertexPermutationFromOriginalToInternal_ = std::vector<VertexIdx>(NumVertices(), 0);
            for (VertexIdx newPos = 0; newPos < NumVertices(); ++newPos) {
                vertexPermutationFromOriginalToInternal_[vertexPermutationFromInternalToOriginal_[newPos]] = newPos;
            }

            for (VertexIdx vertNewPos = 0; vertNewPos < NumVertices(); ++vertNewPos) {
                cscSourcePtr[vertNewPos] = static_cast<EdgeT>(cscEdgeChildren.size());

                VertexIdx vertOldName = vertexPermutationFromInternalToOriginal_[vertNewPos];

                std::vector<VertexIdx> childrenNewName;
                childrenNewName.reserve(childrenTmp[vertOldName].size());

                for (VertexIdx chldOldName : childrenTmp[vertOldName]) {
                    childrenNewName.push_back(vertexPermutationFromOriginalToInternal_[chldOldName]);
                }

                std::sort(childrenNewName.begin(), childrenNewName.end());
                for (const auto &chld : childrenNewName) {
                    cscEdgeChildren.emplace_back(chld);
                }
            }
            cscSourcePtr[NumVertices()] = static_cast<EdgeT>(cscEdgeChildren.size());

            EdgeT acc = 0;
            for (VertexIdx vertOldName : vertexPermutationFromInternalToOriginal_) {
                csrTargetPtr.push_back(acc);
                acc += numParentsTmp[vertOldName];
            }
            csrTargetPtr.push_back(acc);

            std::vector<EdgeT> offset = csrTargetPtr;
            for (VertexIdx vert = 0; vert < NumVertices(); ++vert) {
                for (EdgeT indx = cscSourcePtr[vert]; indx < cscSourcePtr[vert + 1]; ++indx) {
                    const VertexIdx chld = cscEdgeChildren[indx];
                    csrEdgeParents[offset[chld]++] = vert;
                }
            }
        }

        cscOutEdges_ = CompactChildrenEdges(std::move(cscEdgeChildren), std::move(cscSourcePtr));
        csrInEdges_ = CompactParentEdges(std::move(csrEdgeParents), std::move(csrTargetPtr));
    }

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices, const EdgeListType &edges, const std::vector<VertexWorkWeightType> &ww)
        : CompactSparseGraph(numVertices, edges) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
        assert((ww.size() == static_cast<std::size_t>(NumVertices()))
               && "Work weights vector must have the same length as the number of vertices.");

        if constexpr (keepVertexOrder) {
            vertWorkWeights_ = ww;
        } else {
            for (auto vert : Vertices()) {
                vertWorkWeights_[vert] = ww[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }
    }

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices, EdgeListType &edges, const std::vector<VertexWorkWeightType> &&ww)
        : CompactSparseGraph(numVertices, edges) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
        assert((ww.size() == static_cast<std::size_t>(NumVertices()))
               && "Work weights vector must have the same length as the number of vertices.");

        if constexpr (keepVertexOrder) {
            vertWorkWeights_ = std::move(ww);
        } else {
            for (auto vert : Vertices()) {
                vertWorkWeights_[vert] = ww[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }
    }

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices,
                       const EdgeListType &edges,
                       const std::vector<VertexWorkWeightType> &ww,
                       const std::vector<VertexCommWeightType> &cw)
        : CompactSparseGraph(numVertices, edges) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
        static_assert(useCommWeights, "To set communication weight, graph type must allow communication weights.");
        assert((ww.size() == static_cast<std::size_t>(NumVertices()))
               && "Work weights vector must have the same length as the number of vertices.");
        assert((cw.size() == static_cast<std::size_t>(NumVertices()))
               && "Communication weights vector must have the same length as the number of vertices.");

        if constexpr (keepVertexOrder) {
            vertWorkWeights_ = ww;
        } else {
            for (auto vert : Vertices()) {
                vertWorkWeights_[vert] = ww[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertCommWeights_ = cw;
        } else {
            for (auto vert : Vertices()) {
                vertCommWeights_[vert] = cw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }
    }

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices,
                       const EdgeListType &edges,
                       std::vector<VertexWorkWeightType> &&ww,
                       std::vector<VertexCommWeightType> &&cw)
        : CompactSparseGraph(numVertices, edges) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
        static_assert(useCommWeights, "To set communication weight, graph type must allow communication weights.");
        assert((ww.size() == static_cast<std::size_t>(NumVertices()))
               && "Work weights vector must have the same length as the number of vertices.");
        assert((cw.size() == static_cast<std::size_t>(NumVertices()))
               && "Communication weights vector must have the same length as the number of vertices.");

        if constexpr (keepVertexOrder) {
            vertWorkWeights_ = std::move(ww);
        } else {
            for (auto vert : Vertices()) {
                vertWorkWeights_[vert] = ww[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertCommWeights_ = std::move(cw);
        } else {
            for (auto vert : Vertices()) {
                vertCommWeights_[vert] = cw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }
    }

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices,
                       const EdgeListType &edges,
                       const std::vector<VertexWorkWeightType> &ww,
                       const std::vector<VertexCommWeightType> &cw,
                       const std::vector<VertexMemWeightType> &mw)
        : CompactSparseGraph(numVertices, edges) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
        static_assert(useCommWeights, "To set communication weight, graph type must allow communication weights.");
        static_assert(useMemWeights, "To set memory weight, graph type must allow memory weights.");
        assert((ww.size() == static_cast<std::size_t>(NumVertices()))
               && "Work weights vector must have the same length as the number of vertices.");
        assert((cw.size() == static_cast<std::size_t>(NumVertices()))
               && "Communication weights vector must have the same length as the number of vertices.");
        assert((mw.size() == static_cast<std::size_t>(NumVertices()))
               && "Memory weights vector must have the same length as the number of vertices.");

        if constexpr (keepVertexOrder) {
            vertWorkWeights_ = ww;
        } else {
            for (auto vert : Vertices()) {
                vertWorkWeights_[vert] = ww[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertCommWeights_ = cw;
        } else {
            for (auto vert : Vertices()) {
                vertCommWeights_[vert] = cw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertMemWeights_ = mw;
        } else {
            for (auto vert : Vertices()) {
                vertMemWeights_[vert] = mw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }
    }

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices,
                       const EdgeListType &edges,
                       std::vector<VertexWorkWeightType> &&ww,
                       std::vector<VertexCommWeightType> &&cw,
                       std::vector<VertexMemWeightType> &&mw)
        : CompactSparseGraph(numVertices, edges) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
        static_assert(useCommWeights, "To set communication weight, graph type must allow communication weights.");
        static_assert(useMemWeights, "To set memory weight, graph type must allow memory weights.");
        assert((ww.size() == static_cast<std::size_t>(NumVertices()))
               && "Work weights vector must have the same length as the number of vertices.");
        assert((cw.size() == static_cast<std::size_t>(NumVertices()))
               && "Communication weights vector must have the same length as the number of vertices.");
        assert((mw.size() == static_cast<std::size_t>(NumVertices()))
               && "Memory weights vector must have the same length as the number of vertices.");

        if constexpr (keepVertexOrder) {
            vertWorkWeights_ = std::move(ww);
        } else {
            for (auto vert : Vertices()) {
                vertWorkWeights_[vert] = ww[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertCommWeights_ = std::move(cw);
        } else {
            for (auto vert : Vertices()) {
                vertCommWeights_[vert] = cw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertMemWeights_ = std::move(mw);
        } else {
            for (auto vert : Vertices()) {
                vertMemWeights_[vert] = mw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }
    }

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices,
                       const EdgeListType &edges,
                       const std::vector<VertexWorkWeightType> &ww,
                       const std::vector<VertexCommWeightType> &cw,
                       const std::vector<VertexMemWeightType> &mw,
                       const std::vector<VertexTypeType> &vt)
        : CompactSparseGraph(numVertices, edges) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
        static_assert(useCommWeights, "To set communication weight, graph type must allow communication weights.");
        static_assert(useMemWeights, "To set memory weight, graph type must allow memory weights.");
        static_assert(useVertTypes, "To set vertex types, graph type must allow vertex types.");
        assert((ww.size() == static_cast<std::size_t>(NumVertices()))
               && "Work weights vector must have the same length as the number of vertices.");
        assert((cw.size() == static_cast<std::size_t>(NumVertices()))
               && "Communication weights vector must have the same length as the number of vertices.");
        assert((mw.size() == static_cast<std::size_t>(NumVertices()))
               && "Memory weights vector must have the same length as the number of vertices.");
        assert((vt.size() == static_cast<std::size_t>(NumVertices()))
               && "Vertex type vector must have the same length as the number of vertices.");

        if constexpr (keepVertexOrder) {
            vertWorkWeights_ = ww;
        } else {
            for (auto vert : Vertices()) {
                vertWorkWeights_[vert] = ww[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertCommWeights_ = cw;
        } else {
            for (auto vert : Vertices()) {
                vertCommWeights_[vert] = cw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertMemWeights_ = mw;
        } else {
            for (auto vert : Vertices()) {
                vertMemWeights_[vert] = mw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertTypes_ = vt;
        } else {
            for (auto vert : Vertices()) {
                vertTypes_[vert] = vt[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }
    }

    template <typename EdgeListType>
    CompactSparseGraph(VertexIdx numVertices,
                       const EdgeListType &edges,
                       std::vector<VertexWorkWeightType> &&ww,
                       std::vector<VertexCommWeightType> &&cw,
                       std::vector<VertexMemWeightType> &&mw,
                       std::vector<VertexTypeType> &&vt)
        : CompactSparseGraph(numVertices, edges) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
        static_assert(useCommWeights, "To set communication weight, graph type must allow communication weights.");
        static_assert(useMemWeights, "To set memory weight, graph type must allow memory weights.");
        static_assert(useVertTypes, "To set vertex types, graph type must allow vertex types.");
        assert((ww.size() == static_cast<std::size_t>(NumVertices()))
               && "Work weights vector must have the same length as the number of vertices.");
        assert((cw.size() == static_cast<std::size_t>(NumVertices()))
               && "Communication weights vector must have the same length as the number of vertices.");
        assert((mw.size() == static_cast<std::size_t>(NumVertices()))
               && "Memory weights vector must have the same length as the number of vertices.");
        assert((vt.size() == static_cast<std::size_t>(NumVertices()))
               && "Vertex type vector must have the same length as the number of vertices.");

        if constexpr (keepVertexOrder) {
            vertWorkWeights_ = std::move(ww);
        } else {
            for (auto vert : Vertices()) {
                vertWorkWeights_[vert] = ww[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertCommWeights_ = std::move(cw);
        } else {
            for (auto vert : Vertices()) {
                vertCommWeights_[vert] = cw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertMemWeights_ = std::move(mw);
        } else {
            for (auto vert : Vertices()) {
                vertMemWeights_[vert] = mw[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }

        if constexpr (keepVertexOrder) {
            vertTypes_ = std::move(vt);
        } else {
            for (auto vert : Vertices()) {
                vertTypes_[vert] = vt[vertexPermutationFromInternalToOriginal_[vert]];
            }
        }
    }

    template <typename GraphType>
    CompactSparseGraph(const GraphType &graph) : CompactSparseGraph(graph.NumVertices(), EdgeView(graph)) {
        static_assert(isDirectedGraphV<GraphType>);

        if constexpr (isComputationalDagV<GraphType> && useWorkWeights) {
            for (const auto &vert : graph.Vertices()) {
                SetVertexWorkWeight(vert, graph.VertexWorkWeight(vert));
            }
        }

        if constexpr (isComputationalDagV<GraphType> && useCommWeights) {
            for (const auto &vert : graph.Vertices()) {
                SetVertexCommWeight(vert, graph.VertexCommWeight(vert));
            }
        }

        if constexpr (isComputationalDagV<GraphType> && useMemWeights) {
            for (const auto &vert : graph.Vertices()) {
                SetVertexMemWeight(vert, graph.VertexMemWeight(vert));
            }
        }

        if constexpr (isComputationalDagTypedVerticesV<GraphType> && useVertTypes) {
            for (const auto &vert : graph.Vertices()) {
                SetVertexType(vert, graph.VertexType(vert));
            }
        }
    }

    inline auto Vertices() const { return IntegralRange<VertexIdx>(numberOfVertices_); };

    inline VertT NumVertices() const { return numberOfVertices_; };

    inline EdgeT NumEdges() const { return numberOfEdges_; }

    inline auto Parents(const VertexIdx &v) const { return csrInEdges_.Parents(v); };

    inline auto Children(const VertexIdx &v) const { return cscOutEdges_.Children(v); };

    inline EdgeT InDegree(const VertexIdx &v) const { return csrInEdges_.NumberOfParents(v); };

    inline EdgeT OutDegree(const VertexIdx &v) const { return cscOutEdges_.NumberOfChildren(v); };

    template <typename RetT = VertexWorkWeightType>
    inline std::enable_if_t<useWorkWeights, RetT> VertexWorkWeight(const VertexIdx &v) const {
        return vertWorkWeights_[v];
    }

    template <typename RetT = VertexWorkWeightType>
    inline std::enable_if_t<not useWorkWeights, RetT> VertexWorkWeight(const VertexIdx &v) const {
        return static_cast<RetT>(1) + InDegree(v);
    }

    template <typename RetT = VertexCommWeightType>
    inline std::enable_if_t<useCommWeights, RetT> VertexCommWeight(const VertexIdx &v) const {
        return vertCommWeights_[v];
    }

    template <typename RetT = VertexCommWeightType>
    inline std::enable_if_t<not useCommWeights, RetT> VertexCommWeight(const VertexIdx) const {
        return static_cast<RetT>(1);
    }

    template <typename RetT = VertexMemWeightType>
    inline std::enable_if_t<useMemWeights, RetT> VertexMemWeight(const VertexIdx &v) const {
        return vertMemWeights_[v];
    }

    template <typename RetT = VertexMemWeightType>
    inline std::enable_if_t<not useMemWeights, RetT> VertexMemWeight(const VertexIdx) const {
        return static_cast<RetT>(1);
    }

    template <typename RetT = VertexTypeType>
    inline std::enable_if_t<useVertTypes, RetT> VertexType(const VertexIdx &v) const {
        return vertTypes_[v];
    }

    template <typename RetT = VertexTypeType>
    inline std::enable_if_t<not useVertTypes, RetT> VertexType(const VertexIdx) const {
        return static_cast<RetT>(0);
    }

    inline VertexTypeType NumVertexTypes() const { return numberOfVertexTypes_; };

    template <typename RetT = void>
    inline std::enable_if_t<useWorkWeights, RetT> SetVertexWorkWeight(const VertexIdx &v, const VertexWorkWeightType workWeight) {
        if constexpr (keepVertexOrder) {
            vertWorkWeights_[v] = workWeight;
        } else {
            vertWorkWeights_[vertexPermutationFromOriginalToInternal_[v]] = workWeight;
        }
    }

    template <typename RetT = void>
    inline std::enable_if_t<not useWorkWeights, RetT> SetVertexWorkWeight(const VertexIdx &v,
                                                                          const VertexWorkWeightType workWeight) {
        static_assert(useWorkWeights, "To set work weight, graph type must allow work weights.");
    }

    template <typename RetT = void>
    inline std::enable_if_t<useCommWeights, RetT> SetVertexCommWeight(const VertexIdx &v, const VertexCommWeightType commWeight) {
        if constexpr (keepVertexOrder) {
            vertCommWeights_[v] = commWeight;
        } else {
            vertCommWeights_[vertexPermutationFromOriginalToInternal_[v]] = commWeight;
        }
    }

    template <typename RetT = void>
    inline std::enable_if_t<not useCommWeights, RetT> SetVertexCommWeight(const VertexIdx &v,
                                                                          const VertexCommWeightType commWeight) {
        static_assert(useCommWeights, "To set comm weight, graph type must allow comm weights.");
    }

    template <typename RetT = void>
    inline std::enable_if_t<useMemWeights, RetT> SetVertexMemWeight(const VertexIdx &v, const VertexMemWeightType memWeight) {
        if constexpr (keepVertexOrder) {
            vertMemWeights_[v] = memWeight;
        } else {
            vertMemWeights_[vertexPermutationFromOriginalToInternal_[v]] = memWeight;
        }
    }

    template <typename RetT = void>
    inline std::enable_if_t<not useMemWeights, RetT> SetVertexMemWeight(const VertexIdx &v, const VertexMemWeightType memWeight) {
        static_assert(useMemWeights, "To set mem weight, graph type must allow mem weights.");
    }

    template <typename RetT = void>
    inline std::enable_if_t<useVertTypes, RetT> SetVertexType(const VertexIdx &v, const VertexTypeType vertexType) {
        if constexpr (keepVertexOrder) {
            vertTypes_[v] = vertexType;
        } else {
            vertTypes_[vertexPermutationFromOriginalToInternal_[v]] = vertexType;
        }
        numberOfVertexTypes_ = std::max(numberOfVertexTypes_, vertexType);
    }

    template <typename RetT = void>
    inline std::enable_if_t<not useVertTypes, RetT> SetVertexType(const VertexIdx &v, const VertexTypeType vertexType) {
        static_assert(useVertTypes, "To set vert type, graph type must allow vertex types.");
    }

    template <typename RetT = const std::vector<VertexIdx> &>
    inline std::enable_if_t<keepVertexOrder, RetT> GetPullbackPermutation() const {
        static_assert(!keepVertexOrder, "No permutation was applied. This is a deleted function.");
        return {};
    }

    template <typename RetT = const std::vector<VertexIdx> &>
    inline std::enable_if_t<not keepVertexOrder, RetT> GetPullbackPermutation() const {
        return vertexPermutationFromInternalToOriginal_;
    }

    template <typename RetT = const std::vector<VertexIdx> &>
    inline std::enable_if_t<keepVertexOrder, RetT> GetPushforwardPermutation() const {
        static_assert(!keepVertexOrder, "No permutation was applied. This is a deleted function.");
        return {};
    }

    template <typename RetT = const std::vector<VertexIdx> &>
    inline std::enable_if_t<not keepVertexOrder, RetT> GetPushforwardPermutation() const {
        return vertexPermutationFromOriginalToInternal_;
    }
};

template <bool keepVertexOrder,
          bool useWorkWeights,
          bool useCommWeights,
          bool useMemWeights,
          bool useVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename VertexTypeTemplateType>
struct IsCompactSparseGraph<
    CompactSparseGraph<keepVertexOrder, useWorkWeights, useCommWeights, useMemWeights, useVertTypes, VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType>,
    void> : std::true_type {};

template <bool useWorkWeights,
          bool useCommWeights,
          bool useMemWeights,
          bool useVertTypes,
          typename VertT,
          typename EdgeT,
          typename WorkWeightType,
          typename CommWeightType,
          typename MemWeightType,
          typename VertexTypeTemplateType>
struct IsCompactSparseGraphReorder<
    CompactSparseGraph<false, useWorkWeights, useCommWeights, useMemWeights, useVertTypes, VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType>,
    void> : std::true_type {};

static_assert(isCompactSparseGraphV<CompactSparseGraph<true>>);
static_assert(isCompactSparseGraphV<CompactSparseGraph<false>>);
static_assert(!isCompactSparseGraphReorderV<CompactSparseGraph<true>>);
static_assert(isCompactSparseGraphReorderV<CompactSparseGraph<false>>);

static_assert(hasVertexWeightsV<CompactSparseGraph<true, true>>, "CompactSparseGraph must satisfy the has_vertex_weights concept");

static_assert(hasVertexWeightsV<CompactSparseGraph<false, true>>, "CompactSparseGraph must satisfy the has_vertex_weights concept");

static_assert(isDirectedGraphV<CompactSparseGraph<false, false, false, false, false>>,
              "CompactSparseGraph must satisfy the directed_graph concept");

static_assert(isDirectedGraphV<CompactSparseGraph<false, true, true, true, true>>,
              "CompactSparseGraph must satisfy the directed_graph concept");

static_assert(isDirectedGraphV<CompactSparseGraph<true, false, false, false, false>>,
              "CompactSparseGraph must satisfy the directed_graph concept");

static_assert(isDirectedGraphV<CompactSparseGraph<true, true, true, true, true>>,
              "CompactSparseGraph must satisfy the directed_graph concept");

static_assert(isComputationalDagV<CompactSparseGraph<false, true, true, true, false>>,
              "CompactSparseGraph must satisfy the is_computation_dag concept");

static_assert(isComputationalDagV<CompactSparseGraph<true, true, true, true, false>>,
              "CompactSparseGraph must satisfy the is_computation_dag concept");

static_assert(isComputationalDagTypedVerticesV<CompactSparseGraph<false, true, true, true, true>>,
              "CompactSparseGraph must satisfy the is_computation_dag with types concept");

static_assert(isComputationalDagTypedVerticesV<CompactSparseGraph<true, true, true, true, true>>,
              "CompactSparseGraph must satisfy the is_computation_dag with types concept");

static_assert(isDirectConstructableCdagV<CompactSparseGraph<true, true>>, "CompactSparseGraph must be directly constructable");

static_assert(isDirectConstructableCdagV<CompactSparseGraph<false, true>>, "CompactSparseGraph must be directly constructable");

using CSG = CompactSparseGraph<false, true, true, true, true, std::size_t, std::size_t, unsigned, unsigned, unsigned, unsigned>;

static_assert(isDirectedGraphEdgeDescV<CSG>, "CSG must satisfy the directed_graph_edge_desc concept");

// // Graph specific implementations

// template<typename GraphTIn, typename v_work_acc_method, typename v_comm_acc_method, typename v_mem_acc_method, typename
// e_comm_acc_method,
//          bool useWorkWeights, bool useCommWeights, bool useMemWeights, bool useVertTypes, typename VertT, typename
//          EdgeT, typename WorkWeightType, typename CommWeightType, typename MemWeightType, typename
//          VertexTypeTemplateType>
// bool coarser_util::ConstructCoarseDag(
//             const GraphTIn &dag_in,
//             CompactSparseGraph<false, useWorkWeights, useCommWeights, useMemWeights, useVertTypes, VertT, EdgeT,
//             WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType> &coarsened_dag,
//             std::vector<VertexIdxT<CompactSparseGraph<false, useWorkWeights, useCommWeights, useMemWeights,
//             useVertTypes, VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType>>>
//             &vertex_contraction_map) {

//     using Graph_out_type = CompactSparseGraph<false, useWorkWeights, useCommWeights, useMemWeights, useVertTypes,
//     VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType>;

//     static_assert(isDirectedGraphV<GraphTIn> && isDirectedGraphV<Graph_out_type>, "Graph types need to satisfy the
//     is_directed_graph concept."); static_assert(isComputationalDagV<GraphTIn>, "GraphTIn must be a computational DAG");
//     static_assert(isConstructableCdagV<Graph_out_type> || isDirectConstructableCdagV<Graph_out_type>, "Graph_out_type
//     must be a (direct) constructable computational DAG");

//     assert(CheckValidContractionMap<Graph_out_type>(vertex_contraction_map));

//     const VertexIdxT<Graph_out_type> num_vert_quotient =
//         (*std::max_element(vertex_contraction_map.cbegin(), vertex_contraction_map.cend())) + 1;

//     std::set<std::pair<VertexIdxT<Graph_out_type>, VertexIdxT<Graph_out_type>>> quotient_edges;

//     for (const VertexIdxT<GraphTIn> &vert : dag_in.Vertices()) {
//         for (const VertexIdxT<GraphTIn> &chld : dag_in.Children(vert)) {
//             if (vertex_contraction_map[vert] == vertex_contraction_map[chld]) {
//                 continue;
//             }
//             quotient_edges.emplace(vertex_contraction_map[vert], vertex_contraction_map[chld]);
//         }
//     }

//     coarsened_dag = Graph_out_type(num_vert_quotient, quotient_edges);

//     const auto& pushforward_map = coarsened_dag.GetPushforwardPermutation();
//     std::vector<VertexIdxT<Graph_out_type>> combined_expansion_map(dag_in.NumVertices());
//     for (const auto &vert : dag_in.Vertices()) {
//         combined_expansion_map[vert] = pushforward_map[vertex_contraction_map[vert]];
//     }

//     if constexpr (hasVertexWeightsV<GraphTIn> && isModifiableCdagVertexV<Graph_out_type>) {
//         static_assert(std::is_same_v<VWorkwT<GraphTIn>, VWorkwT<Graph_out_type>>, "Work weight types of in-graph and
//         out-graph must be the same."); static_assert(std::is_same_v<VCommwT<GraphTIn>, VCommwT<Graph_out_type>>, "Vertex
//         communication types of in-graph and out-graph must be the same."); static_assert(std::is_same_v<VMemwT<GraphTIn>,
//         VMemwT<Graph_out_type>>, "Memory weight types of in-graph and out-graph must be the same.");

//         for (const VertexIdxT<GraphTIn> &vert : coarsened_dag.Vertices()) {
//             coarsened_dag.SetVertexWorkWeight(vert, 0);
//             coarsened_dag.SetVertexCommWeight(vert, 0);
//             coarsened_dag.SetVertexMemWeight(vert, 0);
//         }

//         for (const VertexIdxT<GraphTIn> &vert : dag_in.Vertices()) {
//             coarsened_dag.SetVertexWorkWeight(
//                 vertex_contraction_map[vert],
//                 v_work_acc_method()(coarsened_dag.VertexWorkWeight(combined_expansion_map[vert]),
//                                 dag_in.VertexWorkWeight(vert)));

//             coarsened_dag.SetVertexCommWeight(
//                 vertex_contraction_map[vert],
//                 v_comm_acc_method()(coarsened_dag.VertexCommWeight(combined_expansion_map[vert]),
//                                 dag_in.VertexCommWeight(vert)));

//             coarsened_dag.SetVertexMemWeight(
//                 vertex_contraction_map[vert],
//                 v_mem_acc_method()(coarsened_dag.VertexMemWeight(combined_expansion_map[vert]),
//                                 dag_in.VertexMemWeight(vert)));
//         }
//     }

//     if constexpr (hasTypedVerticesV<GraphTIn> && is_modifiable_cdag_typed_vertex_v<Graph_out_type>) {
//         static_assert(std::is_same_v<VTypeT<GraphTIn>, VTypeT<Graph_out_type>>,
//                         "Vertex type types of in graph and out graph must be the same!");

//         for (const VertexIdxT<GraphTIn> &vert : dag_in.Vertices()) {
//             coarsened_dag.SetVertexType(vertex_contraction_map[vert], dag_in.VertexType(vert));
//         }
//         // assert(std::all_of(dag_in.Vertices().begin(), dag_in.Vertices().end(),
//         //         [&dag_in, &vertex_contraction_map, &coarsened_dag](const auto &vert){ return
//         //         dag_in.VertexType(vert) ==  coarsened_dag.VertexType(vertex_contraction_map[vert]); })
//         //                 && "Contracted vertices must be of the same type");
//     }

//     std::swap(vertex_contraction_map, combined_expansion_map);

//     std::cout << "Specific Template construct coarsen dag" << std::endl;

//     return true;
// };

}    // namespace osp
