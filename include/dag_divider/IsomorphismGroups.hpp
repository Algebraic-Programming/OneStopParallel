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
#include "scheduler/Scheduler.hpp"
#include <cmath>

#include "file_interactions/ComputationalDagWriter.hpp"
#include "model/dag_algorithms/subgraph_algorithms.hpp"


/**
 * @class WavefrontComponentDivider
 * @brief Divides the wavefronts of a computational DAG into consecutive groups or sections. 
 * The sections are created with the aim of containing a high number of connected components. 
 * The class also provides functionality to detect groups of isomorphic components within the sections. 
 * 
 * 
 */
class IsomorphismGroups {

  private:

    std::vector<std::vector<std::vector<unsigned>>> isomorphism_groups;

    std::vector<std::vector<ComputationalDag>> isomorphism_groups_subgraphs;

    void print_isomorphism_groups() const;


  public:

    IsomorphismGroups() = default;


    /**
     * @brief Retrieves the isomorphism groups.
     * 
     * This function returns a constant reference to a three-dimensional vector representing the isomorphism groups.
     * - The first dimension represents the sections that the dag is divided into. 
     * - The second dimension represents the groups of isomorphic connected components.
     * - The third dimension lists the indices of the components in the vertex map in the group of isomorphic components.
     * 
     * @return const std::vector<std::vector<std::vector<unsigned>>>& 
     *         A constant reference to the vertex maps.
     */
    const std::vector<std::vector<std::vector<unsigned>>> &get_isomorphism_groups() const { return isomorphism_groups; }

    std::vector<std::vector<std::vector<unsigned>>> &get_isomorphism_groups() { return isomorphism_groups; }

    /**
     * @brief Retrieves the isomorphism groups subgraphs.
     * 
     * This function returns a constant reference to a three-dimensional vector representing the isomorphism groups subgraphs.
     * - The first dimension represents the sections that the dag is divided into. 
     * - The second dimension represents the groups of isomorphic connected components.
     * - The third dimension contains the subgraph of the isomorphism group.
     * 
     * @return const std::vector<std::vector<ComputationalDag>>& A constant reference 
     * to the isomorphism groups subgraphs.
     */
    const std::vector<std::vector<ComputationalDag>> &get_isomorphism_groups_subgraphs() const {
        return isomorphism_groups_subgraphs;
    }

    std::vector<std::vector<ComputationalDag>> &get_isomorphism_groups_subgraphs() {
        return isomorphism_groups_subgraphs;
    }


    /**
     * @brief Computes the isomorphism map for a computed division of the current DAG.
     * The resulting isomorphism groups are stored in the isomorphism_groups member variable. 
     * And the corresponding subgraphs are stored in the isomorphism_groups_subgraphs member variable.
     *       
     * Reqires the dag to be divided before calling this function.
     */
    void compute_isomorphism_groups(std::vector<std::vector<std::vector<unsigned>>> &vertex_maps, const ComputationalDag &dag);


};