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
#include "structures/union_find.hpp"
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
class WavefrontComponentDivider {

  private:

    unsigned param_stableize_threshold = 1;
    unsigned param_structure_change_threshold = 1;


    struct wavefron_statistics {

        unsigned number_of_connected_components;
        std::vector<unsigned> connected_components_weights;
        std::vector<unsigned> connected_components_memories;
        std::vector<std::vector<unsigned>> connected_components_vertices;
    };

    const ComputationalDag *dag;

    std::vector<std::vector<std::vector<unsigned>>> vertex_maps;

    std::vector<std::vector<std::vector<unsigned>>> isomorphism_groups;

    std::vector<std::vector<ComputationalDag>> isomorphism_groups_subgraphs;

    std::vector<wavefron_statistics> forward_statistics;
    std::vector<unsigned> forward_stable_levels;
    std::vector<unsigned> forward_structure_changes;

    std::vector<wavefron_statistics> backward_statistics;
    std::vector<unsigned> backward_stable_levels;
    std::vector<unsigned> backward_structure_changes;

    void print_isomorphism_groups() const;

    void print_wavefront_statistics(const std::vector<wavefron_statistics> &statistics, bool reverse = false);

    void compute_forward_statistics(const std::vector<std::vector<unsigned>> &level_sets, const ComputationalDag &dag);

    void compute_backward_statistics(const std::vector<std::vector<unsigned>> &level_sets, const ComputationalDag &dag);

  public:

    WavefrontComponentDivider() = default;

    void set_stableize_threshold(unsigned threshold) { param_stableize_threshold = threshold; }
    unsigned get_stableize_threshold() const { return param_stableize_threshold; }

    void set_structure_change_threshold(unsigned threshold) { param_structure_change_threshold = threshold; }
    unsigned get_structure_change_threshold() const { return param_structure_change_threshold; }


    /**
     * @brief Retrieves the vertex maps.
     * 
     * This function returns a constant reference to a three-dimensional vector
     * containing the vertex maps. 
     * - The first dimension represents the sections that the dag is divided into. 
     * - The second dimension represents the connected components within each section.
     * - The third dimension lists the vertices within each connected component.
     * 
     * @return const std::vector<std::vector<std::vector<unsigned>>>& 
     *         A constant reference to the vertex maps.
     */
    const std::vector<std::vector<std::vector<unsigned>>> &get_vertex_maps() const { return vertex_maps; }

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

    /**
     * @brief Sets the computational directed acyclic graph (DAG) for the divider.
     * 
     * @param dag_ The computational DAG to be set.
     */
    void set_dag(const ComputationalDag &dag_) {
        backward_stable_levels.clear();
        backward_structure_changes.clear();
        forward_stable_levels.clear();
        forward_structure_changes.clear();
        forward_statistics.clear();
        backward_statistics.clear();

        dag = &dag_;
    }

    /**
     * @brief Computes the isomorphism map for a computed division of the current DAG.
     * The resulting isomorphism groups are stored in the isomorphism_groups member variable. 
     * And the corresponding subgraphs are stored in the isomorphism_groups_subgraphs member variable.
     *       
     * Reqires the dag to be divided before calling this function.
     */
    void compute_isomorphism_map();

    /**
     * @brief Divides the current dag into sections based on the wavefront component structure. 
     * The resulting sections are stored in the vertex_maps member variable. 
     * 
     * Reqires the dag to be set before calling this function.
     * 
     * @return true iff some division into section was found. 
     * If no division was found, the vertex_maps composes the connected components of the entire dag.
     */
    bool divide();
};