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

template <typename GraphT, typename ConstrGraphT>
class IsomorphismGroups {
  private:
    std::vector<std::vector<std::vector<std::size_t>>> isomorphismGroups_;

    std::vector<std::vector<ConstrGraphT>> isomorphismGroupsSubgraphs_;

    void PrintIsomorphismGroups() const {
        std::cout << "Isomorphism groups: " << std::endl;
        for (std::size_t i = 0; i < isomorphismGroups_.size(); i++) {
            std::cout << "Level " << i << std::endl;
            for (size_t j = 0; j < isomorphismGroups_[i].size(); j++) {
                std::cout << "Group " << j << " of size " << isomorphismGroupsSubgraphs_[i][j].NumVertices() << " : ";

                // ComputationalDagWriter writer(isomorphism_groups_subgraphs[i][j]);
                // writer.write_dot("isomorphism_group_" + std::to_string(i) + "_" + std::to_string(j) + ".dot");

                for (const auto &vertex : isomorphismGroups_[i][j]) {
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
    const std::vector<std::vector<std::vector<std::size_t>>> &GetIsomorphismGroups() const { return isomorphismGroups_; }

    std::vector<std::vector<std::vector<std::size_t>>> &GetIsomorphismGroups() { return isomorphismGroups_; }

    /**
     * @brief Retrieves the isomorphism groups subgraphs.
     *
     * This function returns a constant reference to a three-dimensional vector representing the isomorphism groups
     * subgraphs.
     * - The first dimension represents the sections that the dag is divided into.
     * - The second dimension represents the groups of isomorphic connected components.
     * - The third dimension contains the subgraph of the isomorphism group.
     *
     * @return const std::vector<std::vector<GraphT>>& A constant reference
     * to the isomorphism groups subgraphs.
     */
    const std::vector<std::vector<ConstrGraphT>> &GetIsomorphismGroupsSubgraphs() const { return isomorphismGroupsSubgraphs_; }

    std::vector<std::vector<ConstrGraphT>> &GetIsomorphismGroupsSubgraphs() { return isomorphismGroupsSubgraphs_; }

    /**
     * @brief Computes the isomorphism map for a computed division of the current DAG.
     * The resulting isomorphism groups are stored in the isomorphism_groups member variable.
     * And the corresponding subgraphs are stored in the isomorphism_groups_subgraphs member variable.
     *
     * Reqires the dag to be divided before calling this function.
     */
    void ComputeIsomorphismGroups(std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> &vertexMaps, const GraphT &dag) {
        isomorphismGroups_ = std::vector<std::vector<std::vector<std::size_t>>>(vertex_maps.size());

        isomorphismGroupsSubgraphs_ = std::vector<std::vector<ConstrGraphT>>(vertex_maps.size());

        for (size_t i = 0; i < vertexMaps.size(); i++) {
            for (std::size_t j = 0; j < vertexMaps[i].size(); j++) {
                ConstrGraphT currentSubgraph;
                create_induced_subgraph(dag, currentSubgraph, vertex_maps[i][j]);

                bool isomorphismGroupFound = false;
                for (size_t k = 0; k < isomorphismGroups_[i].size(); k++) {
                    if (are_isomorphic_by_merkle_hash(isomorphismGroupsSubgraphs_[i][k], currentSubgraph)) {
                        isomorphismGroups_[i][k].emplace_back(j);
                        isomorphismGroupFound = true;
                        break;
                    }
                }

                if (!isomorphismGroupFound) {
                    isomorphismGroups_[i].emplace_back(std::vector<std::size_t>{j});
                    isomorphismGroupsSubgraphs_[i].emplace_back(std::move(currentSubgraph));
                }
            }
        }

        PrintIsomorphismGroups();
    }

    /**
     * @brief Merges large isomorphism groups to avoid resource scarcity in the scheduler.
     * * @param vertex_maps The original vertex maps, which will be modified in place.
     * @param dag The full computational DAG.
     * @param merge_threshold If a group has more members than this, it will be merged.
     * @param target_group_count The number of larger groups to create from a single large group.
     */
    void MergeLargeIsomorphismGroups(std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> &vertexMaps,
                                     const GraphT &dag,
                                     size_t mergeThreshold,
                                     size_t targetGroupCount = 8) {
        // Ensure the merge logic is sound: the threshold must be larger than the target.
        assert(mergeThreshold > targetGroupCount);

        for (size_t i = 0; i < isomorphismGroups_.size(); ++i) {
            std::vector<std::vector<VertexIdxT<GraphT>>> newVertexMapsForLevel;
            std::vector<std::vector<std::size_t>> newIsoGroupsForLevel;
            std::vector<ConstrGraphT> newIsoSubgraphsForLevel;

            size_t newComponentIdx = 0;

            for (size_t j = 0; j < isomorphismGroups_[i].size(); ++j) {
                const auto &group = isomorphismGroups_[i][j];

                if (group.size() <= mergeThreshold) {
                    // This group is small enough, copy it over as is.
                    std::vector<std::size_t> newGroup;
                    for (const auto &originalCompIdx : group) {
                        newVertexMapsForLevel.push_back(vertex_maps[i][originalCompIdx]);
                        newGroup.push_back(newComponentIdx++);
                    }
                    newIsoGroupsForLevel.push_back(newGroup);
                    newIsoSubgraphsForLevel.push_back(isomorphismGroupsSubgraphs_[i][j]);
                } else {
                    // This group is too large and needs to be merged.
                    std::cout << "Merging iso group of size " << group.size() << " into " << targetGroupCount << " new groups."
                              << std::endl;

                    size_t baseMult = group.size() / targetGroupCount;
                    size_t remainder = group.size() % targetGroupCount;

                    std::vector<std::size_t> newMergedGroupIndices;
                    size_t currentOriginalIdx = 0;

                    for (size_t k = 0; k < targetGroupCount; ++k) {
                        std::vector<VertexIdxT<GraphT>> mergedComponent;
                        size_t numToMerge = baseMult + (k < remainder ? 1 : 0);

                        for (size_t m = 0; m < numToMerge; ++m) {
                            const auto &originalComp = vertex_maps[i][group[currentOriginalIdx++]];
                            mergedComponent.insert(merged_component.end(), original_comp.begin(), original_comp.end());
                        }
                        std::sort(merged_component.begin(), merged_component.end());
                        newVertexMapsForLevel.push_back(merged_component);
                        newMergedGroupIndices.push_back(newComponentIdx++);
                    }

                    newIsoGroupsForLevel.push_back(newMergedGroupIndices);
                    ConstrGraphT newRepSubgraph;
                    create_induced_subgraph(dag, newRepSubgraph, new_vertex_maps_for_level.back());
                    newIsoSubgraphsForLevel.push_back(newRepSubgraph);
                }
            }
            // Replace the old level data with the new, potentially merged data.
            vertexMaps[i] = new_vertex_maps_for_level;
            isomorphismGroups_[i] = newIsoGroupsForLevel;
            isomorphismGroupsSubgraphs_[i] = newIsoSubgraphsForLevel;
        }
        // print_isomorphism_groups();
    }
};

}    // namespace osp
