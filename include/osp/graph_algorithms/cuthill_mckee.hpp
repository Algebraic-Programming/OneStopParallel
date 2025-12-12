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
#include <unordered_map>
#include <vector>

#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

template <typename GraphT>
struct CmVertex {
    using VertexType = vertex_idx_t<Graph_t>;
    VertexType vertex_;

    VertexType parentPosition_;

    VertexType degree_;

    CmVertex() : vertex(0), parent_position(0), degree(0) {}

    CmVertex(VertexType vertex, VertexType degree, VertexType parentPosition)
        : vertex(vertex_), parent_position(parent_position_), degree(degree_) {}

    bool operator<(CmVertex const &rhs) const {
        return (parent_position < rhs.parent_position) || (parent_position == rhs.parent_position and degree < rhs.degree)
               || (parent_position == rhs.parent_position and degree == rhs.degree and vertex < rhs.vertex);
    }
};

template <typename GraphT>
std::vector<vertex_idx_t<Graph_t>> CuthillMckeeWavefront(const GraphT &dag, bool permutation = false) {
    using VertexType = vertex_idx_t<Graph_t>;
    using CmVertex = CmVertex<GraphT>;

    std::vector<VertexType> result(dag.num_vertices());
    std::vector<unsigned> predecessorsCount(dag.num_vertices(), 0);
    std::vector<VertexType> predecessorsPosition(dag.num_vertices(), dag.num_vertices());

    std::vector<CmVertex> currentWavefront;
    for (const auto &source : source_vertices_view(dag)) {
        currentWavefront.push_back(CmVertex(source, dag.out_degree(source), 0));
    }

    std::vector<CmVertex> newWavefront;
    VertexType nodeCounter = 0;
    while (node_counter < dag.num_vertices()) {
        newWavefront.clear();
        std::sort(currentWavefront.begin(), currentWavefront.end());

        if (permutation) {
            for (VertexType i = 0; i < static_cast<VertexType>(currentWavefront.size()); i++) {
                result[currentWavefront[i].vertex] = node_counter + i;
            }
        } else {
            for (size_t i = 0; i < currentWavefront.size(); i++) {
                result[node_counter + i] = currentWavefront[i].vertex;
            }
        }

        if (nodeCounter + static_cast<VertexType>(currentWavefront.size()) == dag.num_vertices()) {
            break;
        }

        for (VertexType i = 0; i < static_cast<VertexType>(currentWavefront.size()); i++) {
            for (const auto &child : dag.children(current_wavefront[i].vertex)) {
                predecessors_count[child]++;
                predecessors_position[child] = std::min(predecessors_position[child], node_counter + i);

                if (predecessors_count[child] == dag.in_degree(child)) {
                    new_wavefront.push_back(cm_vertex(child, dag.out_degree(child), predecessors_position[child]));
                }
            }
        }

        nodeCounter += static_cast<VertexType>(currentWavefront.size());

        std::swap(currentWavefront, newWavefront);
    }

    return result;
}

template <typename GraphT>
std::vector<vertex_idx_t<Graph_t>> CuthillMckeeUndirected(const GraphT &dag, bool startAtSink, bool perm = false) {
    using VertexType = vertex_idx_t<Graph_t>;
    using CmVertex = CmVertex<GraphT>;

    std::vector<VertexType> cmOrder(dag.num_vertices());

    std::unordered_map<VertexType, unsigned> maxNodeDistances;
    VertexType firstNode = 0;

    // compute bottom or top node distances of sink or source nodes, store node with the largest distance in first_node
    if (startAtSink) {
        unsigned maxDistance = 0;
        const std::vector<unsigned> topNodeDistance = get_top_node_distance(dag);
        for (const auto &i : dag.vertices()) {
            if (is_sink(i, dag)) {
                maxNodeDistances[i] = topNodeDistance[i];

                if (topNodeDistance[i] > maxDistance) {
                    maxDistance = topNodeDistance[i];
                    firstNode = i;
                }
            }
        }
    } else {
        unsigned maxDistance = 0;
        const std::vector<unsigned> bottomNodeDistance = get_bottom_node_distance(dag);
        for (const auto &i : dag.vertices()) {
            if (is_source(i, dag)) {
                maxNodeDistances[i] = bottomNodeDistance[i];

                if (bottomNodeDistance[i] > maxDistance) {
                    maxDistance = bottomNodeDistance[i];
                    firstNode = i;
                }
            }
        }
    }

    if (perm) {
        cmOrder[first_node] = 0;
    } else {
        cmOrder[0] = first_node;
    }

    std::unordered_set<VertexType> visited;
    visited.insert(first_node);

    std::vector<CmVertex> currentLevel;
    currentLevel.reserve(dag.in_degree(first_node) + dag.out_degree(first_node));

    for (const auto &child : dag.children(first_node)) {
        current_level.push_back(cm_vertex(child, dag.in_degree(child) + dag.out_degree(child), 0));
        visited.insert(child);
    }

    for (const auto &parent : dag.parents(first_node)) {
        current_level.push_back(cm_vertex(parent, dag.in_degree(parent) + dag.out_degree(parent), 0));
        visited.insert(parent);
    }

    VertexType nodeCounter = 1;
    while (node_counter < dag.num_vertices()) {
        std::sort(currentLevel.begin(), currentLevel.end());

        if (perm) {
            for (VertexType i = 0; i < currentLevel.size(); i++) {
                cmOrder[currentLevel[i].vertex] = node_counter + i;
            }
        } else {
            for (VertexType i = 0; i < currentLevel.size(); i++) {
                cmOrder[node_counter + i] = currentLevel[i].vertex;
            }
        }

        if (nodeCounter + currentLevel.size() == dag.num_vertices()) {
            break;
        }

        std::unordered_map<VertexType, VertexType> nodePriority;

        for (VertexType i = 0; i < currentLevel.size(); i++) {
            for (const auto &child : dag.children(current_level[i].vertex)) {
                if (visited.find(child) == visited.end()) {
                    if (node_priority.find(child) == node_priority.end()) {
                        node_priority[child] = node_counter + i;
                    } else {
                        node_priority[child] = std::min(node_priority[child], node_counter + i);
                    }
                }
            }

            for (const auto &parent : dag.parents(current_level[i].vertex)) {
                if (visited.find(parent) == visited.end()) {
                    if (node_priority.find(parent) == node_priority.end()) {
                        node_priority[parent] = node_counter + i;
                    } else {
                        node_priority[parent] = std::min(node_priority[parent], node_counter + i);
                    }
                }
            }
        }

        nodeCounter += currentLevel.size();

        if (nodePriority.empty()) {    // the dag has more than one connected components

            unsigned maxDistance = 0;
            for (const auto [node, distance] : max_node_distances) {
                if (visited.find(node) == visited.end() and distance > max_distance) {
                    max_distance = distance;
                    first_node = node;
                }
            }

            if (perm) {
                cmOrder[first_node] = node_counter;
            } else {
                cmOrder[node_counter] = first_node;
            }
            visited.insert(first_node);

            currentLevel.clear();
            currentLevel.reserve(dag.in_degree(first_node) + dag.out_degree(first_node));

            for (const auto &child : dag.children(first_node)) {
                current_level.push_back(cm_vertex(child, dag.in_degree(child) + dag.out_degree(child), node_counter));
                visited.insert(child);
            }

            for (const auto &parent : dag.parents(first_node)) {
                current_level.push_back(cm_vertex(parent, dag.in_degree(parent) + dag.out_degree(parent), node_counter));
                visited.insert(parent);
            }

            nodeCounter++;

        } else {
            currentLevel.clear();
            currentLevel.reserve(node_priority.size());

            for (const auto &[node, priority] : node_priority) {
                current_level.push_back(cm_vertex(node, dag.in_degree(node) + dag.out_degree(node), priority));
                visited.insert(node);
            }
        }
    }

    return cm_order;
}

// Cuthill-McKee Wavefront
template <typename GraphT>
inline std::vector<vertex_idx_t<Graph_t>> GetTopOrderCuthillMcKeeWavefront(const GraphT &dag) {
    std::vector<vertex_idx_t<Graph_t>> order;
    if (dag.num_vertices() > 0) {
        std::vector<vertex_idx_t<Graph_t>> priority = cuthill_mckee_wavefront(dag);
        order.reserve(dag.num_vertices());
        for (const auto &v : priority_vec_top_sort_view(dag, priority)) {
            order.push_back(v);
        }
    }
    return order;
}

// Cuthill-McKee Undirected
template <typename GraphT>
inline std::vector<vertex_idx_t<Graph_t>> GetTopOrderCuthillMcKeeUndirected(const GraphT &dag) {
    std::vector<vertex_idx_t<Graph_t>> order;
    if (dag.num_vertices() > 0) {
        std::vector<vertex_idx_t<Graph_t>> priority = cuthill_mckee_undirected(dag, true, true);
        order.reserve(dag.num_vertices());
        for (const auto &v : priority_vec_top_sort_view(dag, priority)) {
            order.push_back(v);
        }
    }
    return order;
}

}    // namespace osp
