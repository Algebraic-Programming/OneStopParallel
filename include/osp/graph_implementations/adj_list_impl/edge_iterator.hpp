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

#include <iterator>
#include <vector>

#include "osp/concepts/directed_graph_concept.hpp"

namespace osp {

template <typename GraphT>
class EdgeRangeVectorImpl {
    using DirectedEdgeDescriptor = typename DirectedGraphEdgeDescTraits<GraphT>::DirectedEdgeDescriptor;
    using VertexIdx = typename DirectedGraphTraits<GraphT>::VertexIdx;
    using Iter = typename GraphT::OutEdgesIteratorT;
    const GraphT &graph_;

    struct EdgeIterator {
        VertexIdx currentVertex_;
        std::size_t currentEdgeIdx_;
        Iter currentEdge_;

        const GraphT *graph_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = DirectedEdgeDescriptor;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        EdgeIterator() : currentVertex_(0u), currentEdgeIdx_(0u), graph_(nullptr) {}

        EdgeIterator(const EdgeIterator &other)
            : currentVertex_(other.currentVertex_), currentEdgeIdx_(other.currentEdgeIdx_), graph_(other.graph_) {}

        EdgeIterator &operator=(const EdgeIterator &other) {
            if (this != &other) {
                currentVertex_ = other.currentVertex_;
                currentEdgeIdx_ = other.currentEdgeIdx_;
                graph_ = other.graph_;
            }
            return *this;
        }

        EdgeIterator(const GraphT &graph) : currentVertex_(0u), currentEdgeIdx_(0u), graph_(&graph) {
            while (currentVertex_ != graph_->NumVertices()) {
                if (graph_->OutEdges(currentVertex_).begin() != graph_->OutEdges(currentVertex_).end()) {
                    currentEdge_ = graph_->OutEdges(currentVertex_).begin();
                    break;
                }
                currentVertex_++;
            }
        }

        EdgeIterator(std::size_t currentEdgeIdx, const GraphT &graph)
            : currentVertex_(0u), currentEdgeIdx_(currentEdgeIdx), graph_(&graph) {
            if (currentEdgeIdx_ < graph_->NumEdges()) {
                std::size_t tmp = 0u;

                if (tmp < currentEdgeIdx_) {
                    while (currentVertex_ != graph_->NumVertices()) {
                        currentEdge_ = graph_->OutEdges(currentVertex_).begin();

                        while (currentEdge_ != graph_->OutEdges(currentVertex_).end()) {
                            if (tmp == currentEdgeIdx_) {
                                break;
                            }

                            currentEdge_++;
                            tmp++;
                        }

                        currentVertex_++;
                    }
                }

            } else {
                currentEdgeIdx_ = graph_->NumEdges();
                currentVertex_ = graph_->NumVertices();
            }
        }

        const value_type &operator*() const { return *currentEdge_; }

        const value_type *operator->() const { return &(*currentEdge_); }

        // Prefix increment
        EdgeIterator &operator++() {
            currentEdge_++;
            currentEdgeIdx_++;

            if (currentEdge_ == graph_->OutEdges(currentVertex_).end()) {
                currentVertex_++;

                while (currentVertex_ != graph_->NumVertices()) {
                    if (graph_->OutEdges(currentVertex_).begin() != graph_->OutEdges(currentVertex_).end()) {
                        currentEdge_ = graph_->OutEdges(currentVertex_).begin();
                        break;
                    }

                    currentVertex_++;
                }
            }

            return *this;
        }

        // Postfix increment
        EdgeIterator operator++(int) {
            EdgeIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const EdgeIterator &other) const { return currentEdgeIdx_ == other.currentEdgeIdx_; }

        inline bool operator!=(const EdgeIterator &other) const { return currentEdgeIdx_ != other.currentEdgeIdx_; }
    };

  public:
    EdgeRangeVectorImpl(const GraphT &graph) : graph_(graph) {}

    auto begin() const { return EdgeIterator(graph_); }

    auto end() const { return EdgeIterator(graph_.NumEdges(), graph_); }

    auto size() const { return graph_.NumEdges(); }
};

template <typename GraphT>
class EdgeSourceRange {
    using DirectedEdgeDescriptor = typename DirectedGraphEdgeDescTraits<GraphT>::DirectedEdgeDescriptor;
    using VertexIdx = typename DirectedGraphTraits<GraphT>::VertexIdx;
    using Iter = typename GraphT::InEdgesIteratorT;

    const GraphT &graph_;
    const std::vector<DirectedEdgeDescriptor> &edges_;

    struct SourceIterator {
        const GraphT *graph_;
        Iter currentEdge_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = VertexIdx;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        SourceIterator() : graph_(nullptr) {}

        SourceIterator(const SourceIterator &other) : graph_(other.graph_), currentEdge_(other.currentEdge_) {}

        SourceIterator &operator=(const SourceIterator &other) {
            if (this != &other) {
                graph_ = other.graph_;
                currentEdge_ = other.currentEdge_;
            }
            return *this;
        }

        SourceIterator(Iter currentEdge, const GraphT &graph) : graph_(&graph), currentEdge_(currentEdge) {}

        value_type operator*() const { return Source(*currentEdge_, *graph_); }

        // Prefix increment
        SourceIterator &operator++() {
            currentEdge_++;
            return *this;
        }

        // Postfix increment
        SourceIterator operator++(int) {
            SourceIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const SourceIterator &other) const { return currentEdge_ == other.currentEdge_; }

        inline bool operator!=(const SourceIterator &other) const { return currentEdge_ != other.currentEdge_; }
    };

  public:
    EdgeSourceRange(const std::vector<DirectedEdgeDescriptor> &edges, const GraphT &graph) : graph_(graph), edges_(edges) {}

    auto begin() const { return SourceIterator(edges_.begin(), graph_); }

    auto end() const { return SourceIterator(edges_.end(), graph_); }

    auto size() const { return edges_.size(); }
};

template <typename GraphT>
class EdgeTargetRange {
    using DirectedEdgeDescriptor = typename DirectedGraphEdgeDescTraits<GraphT>::DirectedEdgeDescriptor;
    using VertexIdx = typename DirectedGraphTraits<GraphT>::VertexIdx;
    using Iter = typename GraphT::OutEdgesIteratorT;
    const GraphT &graph_;
    const std::vector<DirectedEdgeDescriptor> &edges_;

    struct TargetIterator {
        const GraphT *graph_;
        Iter currentEdge_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = VertexIdx;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        TargetIterator() : graph_(nullptr) {}

        TargetIterator(const TargetIterator &other) : graph_(other.graph_), currentEdge_(other.currentEdge_) {}

        TargetIterator &operator=(const TargetIterator &other) {
            if (this != &other) {
                graph_ = other.graph_;
                currentEdge_ = other.currentEdge_;
            }
            return *this;
        }

        TargetIterator(Iter currentEdge, const GraphT &graph) : graph_(&graph), currentEdge_(currentEdge) {}

        value_type operator*() const { return Target(*currentEdge_, *graph_); }

        // Prefix increment
        TargetIterator &operator++() {
            currentEdge_++;
            return *this;
        }

        // Postfix increment
        TargetIterator operator++(int) {
            TargetIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const TargetIterator &other) const { return currentEdge_ == other.currentEdge_; }

        inline bool operator!=(const TargetIterator &other) const { return currentEdge_ != other.currentEdge_; }
    };

  public:
    EdgeTargetRange(const std::vector<DirectedEdgeDescriptor> &edges, const GraphT &graph) : graph_(graph), edges_(edges) {}

    auto begin() const { return TargetIterator(edges_.begin(), graph_); }

    auto end() const { return TargetIterator(edges_.end(), graph_); }

    auto size() const { return edges_.size(); }
};

}    // namespace osp
