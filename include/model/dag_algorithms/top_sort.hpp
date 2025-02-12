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

#include <queue>
#include <vector>

#include "model/ComputationalDag.hpp"

namespace dag_algorithms {

std::vector<VertexType> top_sort_dfs(const ComputationalDag &dag);

std::vector<VertexType> top_sort_bfs(const ComputationalDag &dag);

std::vector<VertexType> top_sort_locality(const ComputationalDag &dag);

std::vector<VertexType> top_sort_max_children(const ComputationalDag &dag);

std::vector<VertexType> top_sort_random(const ComputationalDag &dag);

std::vector<VertexType> top_sort_heavy_edges(const ComputationalDag &dag, bool sum = false);

template<typename T>
std::vector<VertexType> top_sort_priority_node_type(const ComputationalDag &dag, const std::vector<T> &node_priority) {

    std::vector<VertexType> predecessors_count(dag.numberOfVertices(), 0);
    std::vector<VertexType> top_order(dag.numberOfVertices(), 0);

    struct heap_node {

        unsigned node;

        T priority;

        heap_node() : node(0), priority(0) {}
        heap_node(unsigned n, unsigned p) : node(n), priority(p) {}

        bool operator<(heap_node const &rhs) const {
            return (priority > rhs.priority) || (priority == rhs.priority and node > rhs.node);
        }
    };

    std::vector<std::vector<heap_node>> heap(dag.getNumberOfNodeTypes());

    for (const auto &source_vertex : dag.sourceVertices()) {

        heap[dag.nodeType(source_vertex)].emplace_back(source_vertex, node_priority[source_vertex]);
        std::push_heap(heap[dag.nodeType(source_vertex)].begin(), heap[dag.nodeType(source_vertex)].end());
    }

    unsigned idx = 0;

    unsigned current_node_type = 0;

    while (idx < dag.numberOfVertices()) {

        while (not heap[current_node_type].empty()) { // keep the same node type as long as possible

            std::pop_heap(heap[current_node_type].begin(), heap[current_node_type].end());
            const unsigned current_node = heap[current_node_type].back().node;
            heap[current_node_type].pop_back();

            top_order[idx++] = current_node;

            for (const auto &child : dag.children(current_node)) {

                predecessors_count[child]++;
                if (predecessors_count[child] == dag.numberOfParents(child)) {

                    heap[dag.nodeType(child)].emplace_back(child, node_priority[child]);
                    std::push_heap(heap[dag.nodeType(child)].begin(), heap[dag.nodeType(child)].end());
                }
            }
        }

        current_node_type = (current_node_type + 1) % dag.getNumberOfNodeTypes();
    }

    return top_order;
};


template<typename T>
std::vector<VertexType> top_sort_priority(const ComputationalDag &dag, const std::vector<T> &node_priority) {

    std::vector<VertexType> predecessors_count(dag.numberOfVertices(), 0);
    std::vector<VertexType> top_order(dag.numberOfVertices(), 0);

    struct heap_node {

        unsigned node;

        T priority;

        heap_node() : node(0), priority(0) {}
        heap_node(unsigned n, unsigned p) : node(n), priority(p) {}

        bool operator<(heap_node const &rhs) const {
            return (priority > rhs.priority) || (priority == rhs.priority and node > rhs.node);
        }
    };

    std::vector<heap_node> heap;

    for (const auto &source_vertex : dag.sourceVertices()) {

        heap.emplace_back(source_vertex, node_priority[source_vertex]);
        std::push_heap(heap.begin(), heap.end());
    }

    unsigned idx = 0;

    while (not heap.empty()) {

        std::pop_heap(heap.begin(), heap.end());
        const unsigned current_node = heap.back().node;
        heap.pop_back();

        top_order[idx++] = current_node;

        for (const auto &child : dag.children(current_node)) {

            predecessors_count[child]++;
            if (predecessors_count[child] == dag.numberOfParents(child)) {

                heap.emplace_back(child, node_priority[child]);
                std::push_heap(heap.begin(), heap.end());
            }
        }
    }

    return top_order;
};

} // namespace dag_algorithms