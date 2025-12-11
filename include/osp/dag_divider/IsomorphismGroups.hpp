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

#include <iostream>
#include <vector>

#include "osp/concepts/graph_traits.hpp"
#include "osp/dag_divider/isomorphism_divider/MerkleHashComputer.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp {

template <typename Graph_t, typename Constr_Graph_t>
class IsomorphismGroups {
  private:
    std::vector<std::vector<std::vector<std::size_t>>> isomorphism_groups;

    std::vector<std::vector<Constr_Graph_t>> isomorphism_groups_subgraphs;

    void print_isomorphism_groups() const {
        std::cout << "Isomorphism groups: " << std::endl;
        for (std::size_t i = 0; i < isomorphism_groups.size(); i++) {
            std::cout << "Level " << i << std::endl;
            for (size_t j = 0; j < isomorphism_groups[i].size(); j++) {
                std::cout << "Group " << j << " of size " << isomorphism_groups_subgraphs[i][j].num_vertices() << " : ";

                // ComputationalDagWriter writer(isomorphism_groups_subgraphs[i][j]);
                // writer.write_dot("isomorphism_group_" + std::to_string(i) + "_" + std::to_string(j) + ".dot");

                for (const auto &vertex : isomorphism_groups[i][j]) {
                    std::cout << vertex << " ";
                }
                std::cout << std::endl;
            }
        }

        std::cout << "Isomorphism groups end" << std::endl;
    }

  public:
    IsomorphismGroups() = default;

    /**
     * @brief Retrieves the isomorphism groups.
     *
     * This function returns a constant reference to a three-dimensional vector representing the isomorphism groups.
     * - The first dimension represents the sections that the dag is divided into.
     * - The second dimension represents the groups of isomorphic connected components.
     * - The third dimension lists the indices of the components in the vertex map in the group of isomorphic
     * components.
     *
     * @return const std::vector<std::vector<std::vector<unsigned>>>&
     *         A constant reference to the vertex maps.
     */
    const std::vector<std::vector<std::vector<std::size_t>>> &get_isomorphism_groups() const { return isomorphism_groups; }

    std::vector<std::vector<std::vector<std::size_t>>> &get_isomorphism_groups() { return isomorphism_groups; }

    /**
     * @brief Retrieves the isomorphism groups subgraphs.
     *
     * This function returns a constant reference to a three-dimensional vector representing the isomorphism groups
     * subgraphs.
     * - The first dimension represents the sections that the dag is divided into.
     * - The second dimension represents the groups of isomorphic connected components.
     * - The third dimension contains the subgraph of the isomorphism group.
     *
     * @return const std::vector<std::vector<Graph_t>>& A constant reference
     * to the isomorphism groups subgraphs.
     */
    const std::vector<std::vector<Constr_Graph_t>> &get_isomorphism_groups_subgraphs() const {
        return isomorphism_groups_subgraphs;
    }

    std::vector<std::vector<Constr_Graph_t>> &get_isomorphism_groups_subgraphs() { return isomorphism_groups_subgraphs; }

    /**
     * @brief Computes the isomorphism map for a computed division of the current DAG.
     * The resulting isomorphism groups are stored in the isomorphism_groups member variable.
     * And the corresponding subgraphs are stored in the isomorphism_groups_subgraphs member variable.
     *
     * Reqires the dag to be divided before calling this function.
     */
    void compute_isomorphism_groups(std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> &vertex_maps, const Graph_t &dag) {
        isomorphism_groups = std::vector<std::vector<std::vector<std::size_t>>>(vertex_maps.size());

        isomorphism_groups_subgraphs = std::vector<std::vector<Constr_Graph_t>>(vertex_maps.size());

        for (size_t i = 0; i < vertex_maps.size(); i++) {
            for (std::size_t j = 0; j < vertex_maps[i].size(); j++) {
                Constr_Graph_t current_subgraph;
                create_induced_subgraph(dag, current_subgraph, vertex_maps[i][j]);

                bool isomorphism_group_found = false;
                for (size_t k = 0; k < isomorphism_groups[i].size(); k++) {
                    if (are_isomorphic_by_merkle_hash(isomorphism_groups_subgraphs[i][k], current_subgraph)) {
                        isomorphism_groups[i][k].emplace_back(j);
                        isomorphism_group_found = true;
                        break;
                    }
                }

                if (!isomorphism_group_found) {
                    isomorphism_groups[i].emplace_back(std::vector<std::size_t>{j});
                    isomorphism_groups_subgraphs[i].emplace_back(std::move(current_subgraph));
                }
            }
        }

        print_isomorphism_groups();
    }

    /**
     * @brief Merges large isomorphism groups to avoid resource scarcity in the scheduler.
     * * @param vertex_maps The original vertex maps, which will be modified in place.
     * @param dag The full computational DAG.
     * @param merge_threshold If a group has more members than this, it will be merged.
     * @param target_group_count The number of larger groups to create from a single large group.
     */
    void merge_large_isomorphism_groups(std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> &vertex_maps,
                                        const Graph_t &dag,
                                        size_t merge_threshold,
                                        size_t target_group_count = 8) {
        // Ensure the merge logic is sound: the threshold must be larger than the target.
        assert(merge_threshold > target_group_count);

        for (size_t i = 0; i < isomorphism_groups.size(); ++i) {
            std::vector<std::vector<vertex_idx_t<Graph_t>>> new_vertex_maps_for_level;
            std::vector<std::vector<std::size_t>> new_iso_groups_for_level;
            std::vector<Constr_Graph_t> new_iso_subgraphs_for_level;

            size_t new_component_idx = 0;

            for (size_t j = 0; j < isomorphism_groups[i].size(); ++j) {
                const auto &group = isomorphism_groups[i][j];

                if (group.size() <= merge_threshold) {
                    // This group is small enough, copy it over as is.
                    std::vector<std::size_t> new_group;
                    for (const auto &original_comp_idx : group) {
                        new_vertex_maps_for_level.push_back(vertex_maps[i][original_comp_idx]);
                        new_group.push_back(new_component_idx++);
                    }
                    new_iso_groups_for_level.push_back(new_group);
                    new_iso_subgraphs_for_level.push_back(isomorphism_groups_subgraphs[i][j]);
                } else {
                    // This group is too large and needs to be merged.
                    std::cout << "Merging iso group of size " << group.size() << " into " << target_group_count << " new groups."
                              << std::endl;

                    size_t base_mult = group.size() / target_group_count;
                    size_t remainder = group.size() % target_group_count;

                    std::vector<std::size_t> new_merged_group_indices;
                    size_t current_original_idx = 0;

                    for (size_t k = 0; k < target_group_count; ++k) {
                        std::vector<vertex_idx_t<Graph_t>> merged_component;
                        size_t num_to_merge = base_mult + (k < remainder ? 1 : 0);

                        for (size_t m = 0; m < num_to_merge; ++m) {
                            const auto &original_comp = vertex_maps[i][group[current_original_idx++]];
                            merged_component.insert(merged_component.end(), original_comp.begin(), original_comp.end());
                        }
                        std::sort(merged_component.begin(), merged_component.end());
                        new_vertex_maps_for_level.push_back(merged_component);
                        new_merged_group_indices.push_back(new_component_idx++);
                    }

                    new_iso_groups_for_level.push_back(new_merged_group_indices);
                    Constr_Graph_t new_rep_subgraph;
                    create_induced_subgraph(dag, new_rep_subgraph, new_vertex_maps_for_level.back());
                    new_iso_subgraphs_for_level.push_back(new_rep_subgraph);
                }
            }
            // Replace the old level data with the new, potentially merged data.
            vertex_maps[i] = new_vertex_maps_for_level;
            isomorphism_groups[i] = new_iso_groups_for_level;
            isomorphism_groups_subgraphs[i] = new_iso_subgraphs_for_level;
        }
        // print_isomorphism_groups();
    }
};

}    // namespace osp
