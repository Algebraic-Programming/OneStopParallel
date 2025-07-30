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
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp {

template<typename Graph_t, typename Constr_Graph_t>
class IsomorphismGroups {

  private:
    std::vector<std::vector<std::vector<std::size_t>>> isomorphism_groups;

    std::vector<std::vector<Constr_Graph_t>> isomorphism_groups_subgraphs;

    void print_isomorphism_groups() const {

        std::cout << "Isomorphism groups: " << std::endl;
        for (std::size_t i = 0; i < isomorphism_groups.size(); i++) {
            std::cout << "Level " << i << std::endl;
            for (size_t j = 0; j < isomorphism_groups[i].size(); j++) {
                std::cout << "Group " << j << " of size " << isomorphism_groups_subgraphs[i][j].num_vertices()
                          << " : ";

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
    const std::vector<std::vector<std::vector<std::size_t>>> &get_isomorphism_groups() const {
        return isomorphism_groups;
    }

    std::vector<std::vector<std::vector<std::size_t>>> &get_isomorphism_groups() {
        return isomorphism_groups;
    }

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
    void compute_isomorphism_groups(std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> &vertex_maps,
                                    const Graph_t &dag) {

        isomorphism_groups = std::vector<std::vector<std::vector<std::size_t>>>(vertex_maps.size());

        isomorphism_groups_subgraphs = std::vector<std::vector<Constr_Graph_t>>(vertex_maps.size());

        for (size_t i = 0; i < vertex_maps.size(); i++) {

            for (std::size_t j = 0; j < vertex_maps[i].size(); j++) {

                Constr_Graph_t current_subgraph;
                create_induced_subgraph(dag, current_subgraph, vertex_maps[i][j]);

                bool isomorphism_group_found = false;
                for (size_t k = 0; k < isomorphism_groups[i].size(); k++) {

                    if (checkOrderedIsomorphism(isomorphism_groups_subgraphs[i][k], current_subgraph)) {

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

        // for (size_t i = 0; i < vertex_maps.size(); i++) {

        //     if (isomorphism_groups[i].size() > 1)
        //         continue;

        //     for (size_t j = 0; j < isomorphism_groups[i].size(); j++) {

        //         const size_t size = static_cast<int>(isomorphism_groups[i][j].size());

        //         if (size > 8u) {

        //             std::cout << "iso group more than 8 components " << size << std::endl;

        //             if ((size & (size - 1)) == 0) {

        //                 size_t mult = size / 8;
        //                 std::cout << "mult: " << mult << std::endl;

        //                 std::vector<std::vector<unsigned>> new_groups(8);

        //                 unsigned idx = 0;
        //                 for (auto& group : new_groups) {

        //                     for (size_t k = 0; k < mult; k++) {
        //                         group.insert(group.end(), vertex_maps[i][isomorphism_groups[i][j][idx]].begin(),
        //                         vertex_maps[i][isomorphism_groups[i][j][idx]].end()); idx++;
        //                     }
        //                     std::sort(group.begin(), group.end());
        //                 }

        //                 vertex_maps[i] = new_groups;
        //                 isomorphism_groups[i] = std::vector<std::vector<unsigned>>(1,
        //                 std::vector<unsigned>({0,1,2,3,4,5,6,7})); isomorphism_groups_subgraphs[i] =
        //                 std::vector<Graph_t>(1, dag_algorithms::create_induced_subgraph_sorted(dag,
        //                 new_groups[0]));

        //             }
        //         }
        //     }
        // }

        print_isomorphism_groups();
    }
};

} // namespace osp