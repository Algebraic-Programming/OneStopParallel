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

#include "dag_divider/WavefrontComponentDivider.hpp"
#include "model/dag_algorithms/subgraph_algorithms.hpp"

void WavefrontComponentDivider::compute_isomorphism_map() {

    isomorphism_groups = std::vector<std::vector<std::vector<unsigned>>>(vertex_maps.size());

    isomorphism_groups_subgraphs = std::vector<std::vector<ComputationalDag>>(vertex_maps.size());

    for (size_t i = 0; i < vertex_maps.size(); i++) {

        for (unsigned j = 0; j < vertex_maps[i].size(); j++) {

            ComputationalDag current_subgraph = dag_algorithms::create_induced_subgraph_sorted(*dag, vertex_maps[i][j]);

            bool isomorphism_group_found = false;
            for (size_t k = 0; k < isomorphism_groups[i].size(); k++) {

                if (isomorphism_groups_subgraphs[i][k].checkOrderedIsomorphism(current_subgraph)) {

                    isomorphism_groups[i][k].emplace_back(j);
                    isomorphism_group_found = true;
                    break;
                }
            }

            if (!isomorphism_group_found) {

                isomorphism_groups[i].emplace_back(std::vector<unsigned>{j});
                isomorphism_groups_subgraphs[i].emplace_back(current_subgraph);
            }
        }
    }
    print_isomorphism_groups();
}

void WavefrontComponentDivider::print_isomorphism_groups() const {

    std::cout << "Isomorphism groups: " << std::endl;
    for (size_t i = 0; i < isomorphism_groups.size(); i++) {
        std::cout << "Level " << i << std::endl;
        for (size_t j = 0; j < isomorphism_groups[i].size(); j++) {
            std::cout << "Group " << j << ": ";
            for (const auto &vertex : isomorphism_groups[i][j]) {
                std::cout << vertex << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Isomorphism groups end" << std::endl;
}

bool WavefrontComponentDivider::divide() {

    const std::vector<unsigned> bot_distance = dag->get_top_node_distance();

    std::vector<std::vector<unsigned>> level_sets(1);

    for (VertexType v = 0; v < bot_distance.size(); v++) {
        if (bot_distance[v] - 1 >= level_sets.size()) {
            level_sets.resize(bot_distance[v]);
        }
        level_sets[bot_distance[v] - 1].emplace_back(v);
    }

    compute_forward_statistics(level_sets, *dag);
    print_wavefront_statistics(forward_statistics);

    std::cout << " ------------------- " << std::endl;

    compute_backward_statistics(level_sets, *dag);
    print_wavefront_statistics(backward_statistics, true);

    std::vector<unsigned> cut_levels;

    if (forward_structure_changes.size() > 0 && backward_structure_changes.size() > 0) {

        unsigned fwd = 0;
        unsigned bwd = 0;

        while (fwd < forward_structure_changes.size() && bwd < backward_structure_changes.size()) {

            if (forward_structure_changes[fwd] <= backward_structure_changes[bwd]) {

                if (((fwd + bwd) % 2) == 0) {
                    cut_levels.emplace_back(forward_structure_changes[fwd++]);
                } else {
                    cut_levels.emplace_back(backward_structure_changes[bwd++]);
                }

            } else {
                break;
            }
        }
    } else if (forward_structure_changes.size() > 0) {
        cut_levels = forward_structure_changes;
    } else if (backward_structure_changes.size() > 0) {
        cut_levels = backward_structure_changes;
    }

    std::sort(cut_levels.begin(), cut_levels.end());

    std::cout << "Cut levels: ";
    for (const auto level : cut_levels) {
        std::cout << level << " ";
    }
    std::cout << std::endl;

    if (cut_levels.size() > 0) {

        vertex_maps = std::vector<std::vector<std::vector<unsigned>>>(cut_levels.size() + 1);

        unsigned level_set_idx = 0;
        for (unsigned i = 0; i < cut_levels.size(); i++) {

            Union_Find_Universe<unsigned> uf;
            for (; level_set_idx < cut_levels[i]; level_set_idx++) {
                for (const auto vertex : level_sets[level_set_idx]) {
                    uf.add_object(vertex, dag->nodeWorkWeight(vertex), dag->nodeMemoryWeight(vertex));
                }

                for (const auto &node : level_sets[level_set_idx]) {
                    for (const auto &child : dag->children(node)) {

                        if (uf.is_in_universe(child))
                            uf.join_by_name(node, child);
                    }

                    for (const auto &parent : dag->parents(node)) {
                        if (uf.is_in_universe(parent)) {
                            uf.join_by_name(parent, node);
                        }
                    }
                }
            }
            vertex_maps[i] = uf.get_connected_components();
        }

        Union_Find_Universe<unsigned> uf;
        for (; level_set_idx < level_sets.size(); level_set_idx++) {
            for (const auto vertex : level_sets[level_set_idx]) {
                uf.add_object(vertex, dag->nodeWorkWeight(vertex), dag->nodeMemoryWeight(vertex));
            }

            for (const auto &node : level_sets[level_set_idx]) {
                for (const auto &child : dag->children(node)) {

                    if (uf.is_in_universe(child))
                        uf.join_by_name(node, child);
                }

                for (const auto &parent : dag->parents(node)) {
                    if (uf.is_in_universe(parent)) {
                        uf.join_by_name(parent, node);
                    }
                }
            }
        }

        vertex_maps.back() = uf.get_connected_components();
        // print_wavefront_statistics(forward_statistics);
        return true;
    } else if (forward_statistics.back().number_of_connected_components != 1) {

        std::cout << "No cut levels found, but the graph has "
                  << forward_statistics.back().number_of_connected_components << " connected cmponents." << std::endl;

        vertex_maps = std::vector<std::vector<std::vector<unsigned>>>(1);
        vertex_maps[0] = forward_statistics.back().connected_components_vertices;

        return true;
    }

    // print_wavefront_statistics(forward_statistics);
    vertex_maps.clear();
    return false;
    //
}

void WavefrontComponentDivider::compute_forward_statistics(const std::vector<std::vector<unsigned>> &level_sets,
                                                           const ComputationalDag &dag) {

    forward_statistics.resize(level_sets.size());

    unsigned level_set_idx = 0;

    Union_Find_Universe<unsigned> uf;
    for (const auto vertex : level_sets[level_set_idx]) {
        uf.add_object(vertex, dag.nodeWorkWeight(vertex), dag.nodeMemoryWeight(vertex));
    }

    const auto components = uf.get_connected_components_weights_and_memories();

    forward_statistics[level_set_idx].number_of_connected_components = components.size();

    bool components_stable = false;

    level_set_idx++;

    while (level_set_idx < level_sets.size()) {

        for (const auto vertex : level_sets[level_set_idx]) {
            uf.add_object(vertex, dag.nodeWorkWeight(vertex), dag.nodeMemoryWeight(vertex));
        }

        for (const auto &node : level_sets[level_set_idx]) {
            for (const auto &child : dag.children(node)) {

                if (uf.is_in_universe(child))
                    uf.join_by_name(node, child);
            }

            for (const auto &parent : dag.parents(node)) {
                if (uf.is_in_universe(parent)) {
                    uf.join_by_name(parent, node);
                }
            }
        }

        const auto components = uf.get_connected_components_weights_and_memories();

        forward_statistics[level_set_idx].number_of_connected_components = components.size();
        // forward_statistics[level_set_idx].weights_and_memories = components;
        for (unsigned i = 0; i < components.size(); i++) {

            forward_statistics[level_set_idx].connected_components_weights.emplace_back(std::get<1>(components[i]));
            forward_statistics[level_set_idx].connected_components_memories.emplace_back(std::get<2>(components[i]));
            forward_statistics[level_set_idx].connected_components_vertices.emplace_back(std::get<0>(components[i]));
        }

        if (not components_stable) {

            if ((level_set_idx > 1) &&
                (forward_statistics[level_set_idx - 1].number_of_connected_components >
                         forward_statistics[level_set_idx].number_of_connected_components
                     ? forward_statistics[level_set_idx - 1].number_of_connected_components -
                           forward_statistics[level_set_idx].number_of_connected_components
                     : forward_statistics[level_set_idx].number_of_connected_components -
                               forward_statistics[level_set_idx].number_of_connected_components <=
                           param_stableize_threshold) &&
                (forward_statistics[level_set_idx - 1].number_of_connected_components >
                         forward_statistics[level_set_idx - 2].number_of_connected_components
                     ? forward_statistics[level_set_idx - 1].number_of_connected_components -
                           forward_statistics[level_set_idx - 2].number_of_connected_components
                     : forward_statistics[level_set_idx - 2].number_of_connected_components -
                               forward_statistics[level_set_idx].number_of_connected_components <=
                           param_stableize_threshold)) {

                components_stable = true;
                forward_stable_levels.emplace_back(level_set_idx);
                std::cout << "nr of components stabelized " << level_set_idx << std::endl;
            }

        } else {

            if (forward_statistics[level_set_idx - 1].number_of_connected_components >
                param_structure_change_threshold + forward_statistics[level_set_idx].number_of_connected_components) {

                forward_structure_changes.emplace_back(level_set_idx);
                std::cout << "nr of components changed in level " << level_set_idx << std::endl;
                components_stable = false;
            }
        }

        level_set_idx++;
    }
}

void WavefrontComponentDivider::compute_backward_statistics(const std::vector<std::vector<unsigned>> &level_sets,
                                                            const ComputationalDag &dag) {

    backward_statistics.resize(level_sets.size());

    unsigned level_set_idx = level_sets.size() - 1;

    Union_Find_Universe<unsigned> uf;
    for (const auto vertex : level_sets[level_set_idx]) {
        uf.add_object(vertex, dag.nodeWorkWeight(vertex), dag.nodeMemoryWeight(vertex));
    }

    const auto components = uf.get_connected_components_weights_and_memories();

    backward_statistics[level_set_idx].number_of_connected_components = components.size();

    bool components_stable = false;

    size_t min_number_of_components = dag.numberOfVertices();

    while (level_set_idx > 0) {
        level_set_idx--;
        for (const auto vertex : level_sets[level_set_idx]) {
            uf.add_object(vertex, dag.nodeWorkWeight(vertex), dag.nodeMemoryWeight(vertex));
        }

        for (const auto &node : level_sets[level_set_idx]) {
            for (const auto &child : dag.children(node)) {

                if (uf.is_in_universe(child))
                    uf.join_by_name(node, child);
            }

            for (const auto &parent : dag.parents(node)) {

                if (uf.is_in_universe(parent)) {
                    uf.join_by_name(parent, node);
                }
            }
        }

        const auto components = uf.get_connected_components_weights_and_memories();

        backward_statistics[level_set_idx].number_of_connected_components = components.size();

        for (unsigned i = 0; i < components.size(); i++) {

            backward_statistics[level_set_idx].connected_components_weights.emplace_back(std::get<1>(components[i]));
            backward_statistics[level_set_idx].connected_components_memories.emplace_back(std::get<2>(components[i]));
            backward_statistics[level_set_idx].connected_components_vertices.emplace_back(std::get<0>(components[i]));
        }

        if (not components_stable) {

            if ((level_set_idx < level_sets.size() - 2) &&
                (backward_statistics[level_set_idx + 1].number_of_connected_components >
                         backward_statistics[level_set_idx].number_of_connected_components
                     ? backward_statistics[level_set_idx + 1].number_of_connected_components -
                           backward_statistics[level_set_idx].number_of_connected_components
                     : backward_statistics[level_set_idx].number_of_connected_components -
                               backward_statistics[level_set_idx].number_of_connected_components <=
                           param_stableize_threshold) &&
                (backward_statistics[level_set_idx + 1].number_of_connected_components >
                         backward_statistics[level_set_idx + 2].number_of_connected_components
                     ? backward_statistics[level_set_idx + 1].number_of_connected_components -
                           backward_statistics[level_set_idx + 2].number_of_connected_components
                     : backward_statistics[level_set_idx + 2].number_of_connected_components -
                               backward_statistics[level_set_idx].number_of_connected_components <=
                           param_stableize_threshold)) {

                components_stable = true;
                backward_stable_levels.emplace_back(level_set_idx);
                std::cout << "nr of components stabelized " << level_set_idx << std::endl;
            }

        } else {

            if (backward_statistics[level_set_idx + 1].number_of_connected_components >
                    param_structure_change_threshold +
                        backward_statistics[level_set_idx].number_of_connected_components &&
                backward_statistics[level_set_idx].number_of_connected_components < min_number_of_components) {

                backward_structure_changes.emplace_back(level_set_idx + 1);
                std::cout << "nr of components changed in level " << level_set_idx + 1 << std::endl;
                components_stable = false;
            }
        }
        min_number_of_components = std::min(min_number_of_components, components.size());
    }
}

void WavefrontComponentDivider::print_wavefront_statistics(const std::vector<wavefron_statistics> &statistics,
                                                           bool reverse) {
    if (reverse) {

        for (size_t i = 0; i < statistics.size(); i++) {

            std::cout << "Level " << i << " has " << statistics[i].number_of_connected_components
                      << " connected components." << std::endl;
            // for (size_t j = 0; j < statistics[i].connected_components_vertices.size(); j++) {

            //     std::cout << "Component " << j << " has " <<
            //     statistics[i].connected_components_vertices[j].size()
            //               << " vertice(s), weight: " << statistics[i].connected_components_weights[j]
            //               << ", memory: " << statistics[i].connected_components_memories[j] << std::endl;
            // }
        }

    } else {
        for (size_t i = 0; i < statistics.size(); i++) {
            std::cout << "Level " << i << " has " << statistics[i].number_of_connected_components
                      << " connected components." << std::endl;
            // for (size_t j = 0; j < statistics[i].connected_components_vertices.size(); j++) {

            //     std::cout << "Component " << j << " has " <<
            //     statistics[i].connected_components_vertices[j].size()
            //               << " vertice(s), weight: " << statistics[i].connected_components_weights[j]
            //               << ", memory: " << statistics[i].connected_components_memories[j] << std::endl;
            // }
        }
    }
}