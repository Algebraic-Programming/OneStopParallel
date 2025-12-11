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

template <typename Graph_t>
struct cm_vertex {
    using VertexType = vertex_idx_t<Graph_t>;
    VertexType vertex;

    VertexType parent_position;

    VertexType degree;

    cm_vertex() : vertex(0), parent_position(0), degree(0) {}

    cm_vertex(VertexType vertex_, VertexType degree_, VertexType parent_position_)
        : vertex(vertex_), parent_position(parent_position_), degree(degree_) {}

    bool operator<(cm_vertex const &rhs) const {
        return (parent_position < rhs.parent_position) || (parent_position == rhs.parent_position and degree < rhs.degree)
               || (parent_position == rhs.parent_position and degree == rhs.degree and vertex < rhs.vertex);
    }
};

template <typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> cuthill_mckee_wavefront(const Graph_t &dag, bool permutation = false) {
    using VertexType = vertex_idx_t<Graph_t>;
    using cm_vertex = cm_vertex<Graph_t>;

    std::vector<VertexType> result(dag.num_vertices());
    std::vector<unsigned> predecessors_count(dag.num_vertices(), 0);
    std::vector<VertexType> predecessors_position(dag.num_vertices(), dag.num_vertices());

    std::vector<cm_vertex> current_wavefront;
    for (const auto &source : source_vertices_view(dag)) {
        current_wavefront.push_back(cm_vertex(source, dag.out_degree(source), 0));
    }

    std::vector<cm_vertex> new_wavefront;
    VertexType node_counter = 0;
    while (node_counter < dag.num_vertices()) {
        new_wavefront.clear();
        std::sort(current_wavefront.begin(), current_wavefront.end());

        if (permutation) {
            for (VertexType i = 0; i < static_cast<VertexType>(current_wavefront.size()); i++) {
                result[current_wavefront[i].vertex] = node_counter + i;
            }
        } else {
            for (size_t i = 0; i < current_wavefront.size(); i++) {
                result[node_counter + i] = current_wavefront[i].vertex;
            }
        }

        if (node_counter + static_cast<VertexType>(current_wavefront.size()) == dag.num_vertices()) {
            break;
        }

        for (VertexType i = 0; i < static_cast<VertexType>(current_wavefront.size()); i++) {
            for (const auto &child : dag.children(current_wavefront[i].vertex)) {
                predecessors_count[child]++;
                predecessors_position[child] = std::min(predecessors_position[child], node_counter + i);

                if (predecessors_count[child] == dag.in_degree(child)) {
                    new_wavefront.push_back(cm_vertex(child, dag.out_degree(child), predecessors_position[child]));
                }
            }
        }

        node_counter += static_cast<VertexType>(current_wavefront.size());

        std::swap(current_wavefront, new_wavefront);
    }

    return result;
}

template <typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> cuthill_mckee_undirected(const Graph_t &dag, bool start_at_sink, bool perm = false) {
    using VertexType = vertex_idx_t<Graph_t>;
    using cm_vertex = cm_vertex<Graph_t>;

    std::vector<VertexType> cm_order(dag.num_vertices());

    std::unordered_map<VertexType, unsigned> max_node_distances;
    VertexType first_node = 0;

    // compute bottom or top node distances of sink or source nodes, store node with the largest distance in first_node
    if (start_at_sink) {
        unsigned max_distance = 0;
        const std::vector<unsigned> top_node_distance = get_top_node_distance(dag);
        for (const auto &i : dag.vertices()) {
            if (is_sink(i, dag)) {
                max_node_distances[i] = top_node_distance[i];

                if (top_node_distance[i] > max_distance) {
                    max_distance = top_node_distance[i];
                    first_node = i;
                }
            }
        }
    } else {
        unsigned max_distance = 0;
        const std::vector<unsigned> bottom_node_distance = get_bottom_node_distance(dag);
        for (const auto &i : dag.vertices()) {
            if (is_source(i, dag)) {
                max_node_distances[i] = bottom_node_distance[i];

                if (bottom_node_distance[i] > max_distance) {
                    max_distance = bottom_node_distance[i];
                    first_node = i;
                }
            }
        }
    }

    if (perm) {
        cm_order[first_node] = 0;
    } else {
        cm_order[0] = first_node;
    }

    std::unordered_set<VertexType> visited;
    visited.insert(first_node);

    std::vector<cm_vertex> current_level;
    current_level.reserve(dag.in_degree(first_node) + dag.out_degree(first_node));

    for (const auto &child : dag.children(first_node)) {
        current_level.push_back(cm_vertex(child, dag.in_degree(child) + dag.out_degree(child), 0));
        visited.insert(child);
    }

    for (const auto &parent : dag.parents(first_node)) {
        current_level.push_back(cm_vertex(parent, dag.in_degree(parent) + dag.out_degree(parent), 0));
        visited.insert(parent);
    }

    VertexType node_counter = 1;
    while (node_counter < dag.num_vertices()) {
        std::sort(current_level.begin(), current_level.end());

        if (perm) {
            for (VertexType i = 0; i < current_level.size(); i++) {
                cm_order[current_level[i].vertex] = node_counter + i;
            }
        } else {
            for (VertexType i = 0; i < current_level.size(); i++) {
                cm_order[node_counter + i] = current_level[i].vertex;
            }
        }

        if (node_counter + current_level.size() == dag.num_vertices()) {
            break;
        }

        std::unordered_map<VertexType, VertexType> node_priority;

        for (VertexType i = 0; i < current_level.size(); i++) {
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

        node_counter += current_level.size();

        if (node_priority.empty()) {    // the dag has more than one connected components

            unsigned max_distance = 0;
            for (const auto [node, distance] : max_node_distances) {
                if (visited.find(node) == visited.end() and distance > max_distance) {
                    max_distance = distance;
                    first_node = node;
                }
            }

            if (perm) {
                cm_order[first_node] = node_counter;
            } else {
                cm_order[node_counter] = first_node;
            }
            visited.insert(first_node);

            current_level.clear();
            current_level.reserve(dag.in_degree(first_node) + dag.out_degree(first_node));

            for (const auto &child : dag.children(first_node)) {
                current_level.push_back(cm_vertex(child, dag.in_degree(child) + dag.out_degree(child), node_counter));
                visited.insert(child);
            }

            for (const auto &parent : dag.parents(first_node)) {
                current_level.push_back(cm_vertex(parent, dag.in_degree(parent) + dag.out_degree(parent), node_counter));
                visited.insert(parent);
            }

            node_counter++;

        } else {
            current_level.clear();
            current_level.reserve(node_priority.size());

            for (const auto &[node, priority] : node_priority) {
                current_level.push_back(cm_vertex(node, dag.in_degree(node) + dag.out_degree(node), priority));
                visited.insert(node);
            }
        }
    }

    return cm_order;
}

// Cuthill-McKee Wavefront
template <typename Graph_t>
inline std::vector<vertex_idx_t<Graph_t>> GetTopOrderCuthillMcKeeWavefront(const Graph_t &dag) {
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
template <typename Graph_t>
inline std::vector<vertex_idx_t<Graph_t>> GetTopOrderCuthillMcKeeUndirected(const Graph_t &dag) {
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
