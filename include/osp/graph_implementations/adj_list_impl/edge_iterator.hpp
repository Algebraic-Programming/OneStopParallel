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
        VertexIdx currentVertex;
        std::size_t currentEdgeIdx;
        Iter currentEdge;

        const GraphT *graph;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = DirectedEdgeDescriptor;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        EdgeIterator() : currentVertex(0u), currentEdgeIdx(0u), graph(nullptr) {}

        EdgeIterator(const EdgeIterator &other)
            : currentVertex(other.currentVertex), currentEdgeIdx(other.currentEdgeIdx), graph(other.graph) {}

        EdgeIterator &operator=(const EdgeIterator &other) {
            if (this != &other) {
                currentVertex = other.currentVertex;
                currentEdgeIdx = other.currentEdgeIdx;
                graph = other.graph;
            }
            return *this;
        }

        EdgeIterator(const GraphT &graph) : currentVertex(0u), currentEdgeIdx(0u), graph(&graph) {
            while (currentVertex != graph->NumVertices()) {
                if (graph->OutEdges(currentVertex).begin() != graph->OutEdges(currentVertex).end()) {
                    currentEdge = graph->OutEdges(currentVertex).begin();
                    break;
                }
                currentVertex++;
            }
        }

        EdgeIterator(std::size_t currentEdgeIdx, const GraphT &graph)
            : currentVertex(0u), currentEdgeIdx(currentEdgeIdx), graph(&graph) {
            if (currentEdgeIdx < graph->NumEdges()) {
                std::size_t tmp = 0u;

                if (tmp < currentEdgeIdx) {
                    while (currentVertex != graph->NumVertices()) {
                        currentEdge = graph->OutEdges(currentVertex).begin();

                        while (currentEdge != graph->OutEdges(currentVertex).end()) {
                            if (tmp == currentEdgeIdx) {
                                break;
                            }

                            currentEdge++;
                            tmp++;
                        }

                        currentVertex++;
                    }
                }

            } else {
                currentEdgeIdx = graph->NumEdges();
                currentVertex = graph->NumVertices();
            }
        }

        const value_type &operator*() const { return *currentEdge; }

        const value_type *operator->() const { return &(*currentEdge); }

        // Prefix increment
        EdgeIterator &operator++() {
            currentEdge++;
            currentEdgeIdx++;

            if (currentEdge == graph->OutEdges(currentVertex).end()) {
                currentVertex++;

                while (currentVertex != graph->NumVertices()) {
                    if (graph->OutEdges(currentVertex).begin() != graph->OutEdges(currentVertex).end()) {
                        currentEdge = graph->OutEdges(currentVertex).begin();
                        break;
                    }

                    currentVertex++;
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

        inline bool operator==(const EdgeIterator &other) const { return currentEdgeIdx == other.currentEdgeIdx; }

        inline bool operator!=(const EdgeIterator &other) const { return currentEdgeIdx != other.currentEdgeIdx; }
    };

  public:
    EdgeRangeVectorImpl(const GraphT &graph) : graph_(graph) {}

    auto begin() const { return EdgeIterator(graph_); }

    auto end() const { return EdgeIterator(graph_.NumEdges(), graph_); }

    auto Size() const { return graph_.NumEdges(); }
};

template <typename GraphT>
class EdgeSourceRange {
    using DirectedEdgeDescriptor = typename DirectedGraphEdgeDescTraits<GraphT>::DirectedEdgeDescriptor;
    using VertexIdx = typename DirectedGraphTraits<GraphT>::VertexIdx;
    using Iter = typename GraphT::InEdgesIteratorT;

    const GraphT &graph_;
    const std::vector<DirectedEdgeDescriptor> &edges_;

    struct SourceIterator {
        const GraphT *graph;
        Iter currentEdge;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = VertexIdx;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        SourceIterator() : graph(nullptr) {}

        SourceIterator(const SourceIterator &other) : graph(other.graph), currentEdge(other.currentEdge) {}

        SourceIterator &operator=(const SourceIterator &other) {
            if (this != &other) {
                graph = other.graph;
                currentEdge = other.currentEdge;
            }
            return *this;
        }

        SourceIterator(Iter currentEdge, const GraphT &graph) : graph(&graph), currentEdge(currentEdge) {}

        value_type operator*() const { return Source(*currentEdge, *graph); }

        // Prefix increment
        SourceIterator &operator++() {
            currentEdge++;
            return *this;
        }

        // Postfix increment
        SourceIterator operator++(int) {
            SourceIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const SourceIterator &other) const { return currentEdge == other.currentEdge; }

        inline bool operator!=(const SourceIterator &other) const { return currentEdge != other.currentEdge; }
    };

  public:
    EdgeSourceRange(const std::vector<DirectedEdgeDescriptor> &edges, const GraphT &graph) : graph_(graph), edges_(edges) {}

    auto begin() const { return SourceIterator(edges_.begin(), graph_); }

    auto end() const { return SourceIterator(edges_.end(), graph_); }

    auto Size() const { return edges_.size(); }
};

template <typename GraphT>
class EdgeTargetRange {
    using DirectedEdgeDescriptor = typename DirectedGraphEdgeDescTraits<GraphT>::DirectedEdgeDescriptor;
    using VertexIdx = typename DirectedGraphTraits<GraphT>::VertexIdx;
    using Iter = typename GraphT::OutEdgesIteratorT;
    const GraphT &graph_;
    const std::vector<DirectedEdgeDescriptor> &edges_;

    struct TargetIterator {
        const GraphT *graph;
        Iter currentEdge;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = VertexIdx;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        TargetIterator() : graph(nullptr) {}

        TargetIterator(const TargetIterator &other) : graph(other.graph), currentEdge(other.currentEdge) {}

        TargetIterator &operator=(const TargetIterator &other) {
            if (this != &other) {
                graph = other.graph;
                currentEdge = other.currentEdge;
            }
            return *this;
        }

        TargetIterator(Iter currentEdge, const GraphT &graph) : graph(&graph), currentEdge(currentEdge) {}

        value_type operator*() const { return Target(*currentEdge, *graph); }

        // Prefix increment
        TargetIterator &operator++() {
            currentEdge++;
            return *this;
        }

        // Postfix increment
        TargetIterator operator++(int) {
            TargetIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const TargetIterator &other) const { return currentEdge == other.currentEdge; }

        inline bool operator!=(const TargetIterator &other) const { return currentEdge != other.currentEdge; }
    };

  public:
    EdgeTargetRange(const std::vector<DirectedEdgeDescriptor> &edges, const GraphT &graph) : graph_(graph), edges_(edges) {}

    auto begin() const { return TargetIterator(edges_.begin(), graph_); }

    auto end() const { return TargetIterator(edges_.end(), graph_); }

    auto Size() const { return edges_.size(); }
};

}    // namespace osp
