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

#include "model/dag_algorithms/cuthill_mckee.hpp"

struct cm_vertex {
    unsigned vertex;

    unsigned parent_position;

    int degree;

    cm_vertex() : vertex(0), parent_position(0), degree(0) {}
    cm_vertex(unsigned vertex_, int degree_, unsigned parent_position_)
        : vertex(vertex_), parent_position(parent_position_), degree(degree_)  {}

    bool operator<(cm_vertex const &rhs) const {
        return (parent_position < rhs.parent_position) ||
               (parent_position == rhs.parent_position and degree < rhs.degree) ||
               (parent_position == rhs.parent_position and degree == rhs.degree and vertex < rhs.vertex);
    }
};

std::vector<VertexType> dag_algorithms::cuthill_mckee_wavefront(const ComputationalDag &dag, bool perm) {

    std::vector<VertexType> result(dag.numberOfVertices());
    std::vector<unsigned> predecessors_count(dag.numberOfVertices(), 0);
    std::vector<unsigned> predecessors_position(dag.numberOfVertices(), dag.numberOfVertices());

    const auto source_vertices = dag.sourceVertices();

    std::vector<cm_vertex> current_wavefront(source_vertices.size());

    for (unsigned i = 0; i < current_wavefront.size(); i++) {

        current_wavefront[i] = cm_vertex(source_vertices[i], dag.numberOfChildren(source_vertices[i]), 0);
    }

    unsigned node_counter = 0;
    while (node_counter < dag.numberOfVertices()) {

        std::sort(current_wavefront.begin(), current_wavefront.end());

        // std::cout << "node counter: " << node_counter << " Wavefront: ";
        // for (const auto &v : current_wavefront) {
        //     std::cout << v.vertex << " ";
        // }
        // std::cout << std::endl;

        if (perm) {
            for (unsigned i = 0; i < current_wavefront.size(); i++) {

                result[current_wavefront[i].vertex] = node_counter + i;
            }
        } else {
            for (unsigned i = 0; i < current_wavefront.size(); i++) {

                result[node_counter + i] = current_wavefront[i].vertex;
            }
        }

        if (node_counter + current_wavefront.size() == dag.numberOfVertices())
            break;

        std::vector<cm_vertex> new_wavefront;

        for (unsigned i = 0; i < current_wavefront.size(); i++) {

            for (const auto &child : dag.children(current_wavefront[i].vertex)) {

                predecessors_count[child]++;
                predecessors_position[child] = std::min(predecessors_position[child], node_counter + i);

                if (predecessors_count[child] == dag.numberOfParents(child)) {
                    new_wavefront.push_back(
                        cm_vertex(child, dag.numberOfChildren(child), predecessors_position[child]));
                }
            }
        }

        node_counter += current_wavefront.size();

        current_wavefront = std::move(new_wavefront);
    }

    return result;
};

int sum(unsigned a, unsigned b) { return a + b; }
int diff_1(unsigned a, unsigned b) { return a - b; }
int diff_2(unsigned a, unsigned b) { return b - a; }

std::vector<VertexType> dag_algorithms::cuthill_mckee_undirected(const ComputationalDag &dag, bool start_at_sink, bool perm) {

    int (*calc_degree)(unsigned in_degree, unsigned out_degree) = sum;

    std::vector<VertexType> cm_order(dag.numberOfVertices());

    std::unordered_map<unsigned, unsigned> max_node_distances;
    unsigned first_node = 0;

    // compute bottom or top node distances of sink or source nodes, store node with the largest distance in first_node
    if (start_at_sink) {
        unsigned max_distance = 0;
        const std::vector<unsigned> top_node_distance = dag.get_top_node_distance();
        for (unsigned i = 0; i < top_node_distance.size(); i++) {
            if (dag.isSink(i)) {

                max_node_distances[i] = top_node_distance[i];

                if (top_node_distance[i] > max_distance) {
                    max_distance = top_node_distance[i];
                    first_node = i;
                }
            }
        }
    } else {
        unsigned max_distance = 0;
        const std::vector<unsigned> bottom_node_distance = dag.get_bottom_node_distance();
        for (unsigned i = 0; i < bottom_node_distance.size(); i++) {
            if (dag.isSource(i)) {

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
    current_level.reserve(dag.numberOfParents(first_node) + dag.numberOfChildren(first_node));

    for (const auto &child : dag.children(first_node)) {
        current_level.push_back(cm_vertex(child, calc_degree(dag.numberOfParents(child), dag.numberOfChildren(child)), 0));
        visited.insert(child);
    }

    for (const auto &parent : dag.parents(first_node)) {
        current_level.push_back(cm_vertex(parent, calc_degree(dag.numberOfParents(parent), dag.numberOfChildren(parent)), 0));
        visited.insert(parent);
    }

    unsigned node_counter = 1;
    while (node_counter < dag.numberOfVertices()) {

        std::sort(current_level.begin(), current_level.end());

        if (perm) {
            for (unsigned i = 0; i < current_level.size(); i++) {
                cm_order[current_level[i].vertex] = node_counter + i;
            }
        } else {
            for (unsigned i = 0; i < current_level.size(); i++) {
                cm_order[node_counter + i] = current_level[i].vertex;
            }
        }

        if (node_counter + current_level.size() == dag.numberOfVertices()) {
            break;
        }

        std::unordered_map<unsigned, unsigned> node_priority;

        for (unsigned i = 0; i < current_level.size(); i++) {

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

        if (node_priority.empty()) { // the dag has more than one connected components

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
            current_level.reserve(dag.numberOfParents(first_node) + dag.numberOfChildren(first_node));

            for (const auto &child : dag.children(first_node)) {

                current_level.push_back(cm_vertex(child, calc_degree(dag.numberOfParents(child), dag.numberOfChildren(child)),
                    node_counter));
                visited.insert(child);
            }

            for (const auto &parent : dag.parents(first_node)) {

                current_level.push_back(cm_vertex(parent, calc_degree(dag.numberOfParents(parent), dag.numberOfChildren(parent)),
                    node_counter));
                visited.insert(parent);
            }

            node_counter++;

        } else {

            current_level.clear();
            current_level.reserve(node_priority.size());

            for (const auto &[node, priority] : node_priority) {

                current_level.push_back(
                    cm_vertex(node, calc_degree(dag.numberOfParents(node), dag.numberOfChildren(node)), priority));
                visited.insert(node);
            }
        }
    }

    return cm_order;
};


// struct cm_priority_vertex {

//     unsigned vertex;

//     unsigned parent_position;

//     unsigned neighbor_position;

//     int degree;

//     cm_priority_vertex(unsigned vertex_, int degree_, unsigned parent_position_, unsigned neighbor_position_)
//         : vertex(vertex_), degree(degree_), parent_position(parent_position_), neighbor_position(neighbor_position_) {}

//     bool operator<(cm_priority_vertex const &rhs) const {
//         return (parent_position < rhs.parent_position) ||
//                (parent_position == rhs.parent_position and neighbor_position < rhs.neighbor_position) ||
//                (parent_position == rhs.parent_position and neighbor_position == rhs.neighbor_position and
//                 degree < rhs.degree) ||
//                (parent_position == rhs.parent_position and neighbor_position == rhs.neighbor_position and
//                 degree == rhs.degree and vertex < rhs.vertex);
//     }
// };

// std::vector<unsigned> dynamic_cuthill_mckee(const ComputationalDag &dag) {

//     std::vector<unsigned> result(dag.numberOfVertices());

//     const unsigned num_vertices = dag.numberOfVertices();

//     std::set<cm_priority_vertex> current_wavefront;

//     for (const auto &vertex : dag.sourceVertices()) {

//         current_wavefront.emplace(vertex, dag.numberOfChildren(vertex), 0, num_vertices);
//     }

//     // unsigned node_counter = 0;
//     // while (node_counter < dag.numberOfVertices()) {

//     //     const auto top = current_wavefront.begin().operator*;

//     //     result[node_counter++] = top.vertex;

//     //     for (const auto &child : dag.children(top.vertex)) {

//     //         // update neighbor priority of

//     //         for (const auto &parent : dag.parents(child)) {

//     //             if (parent != top.vertex) {
//     //             }
//     //         }
//     //     }

//     //     if (node_counter + current_wavefront.size() == dag.numberOfVertices())
//     //         break;

//     //     std::vector<cm_vertex> new_wavefront;

//     //     std::unordered_set<unsigned> visited;
//     //     for (unsigend i = 0; i < current_wavefront.size(); i++) {

//     //         for (const auto &child : dag.children(current_wavefront[i].vertex)) {

//     //             if (visited.find(child) == visited.end()) {

//     //                 new_wavefront.push_back(cm_vertex(child, dag.numberOfChildren(child), node_counter + i));
//     //                 visited.insert(child);
//     //             }
//     //         }
//     //     }

//     //     node_counter += current_wavefront.size();

//     //     current_wavefront = std::move(new_wavefront);
//     // }
//     return result;
// };
