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

    const std::vector<std::vector<std::vector<unsigned>>> &get_vertex_maps() const { return vertex_maps; }

    const std::vector<std::vector<std::vector<unsigned>>> &get_isomorphism_groups() const { return isomorphism_groups; }

    const std::vector<std::vector<ComputationalDag>> &get_isomorphism_groups_subgraphs() const {
        return isomorphism_groups_subgraphs;
    }

    void set_dag(const ComputationalDag &dag_) {
        backward_stable_levels.clear();
        backward_structure_changes.clear();
        forward_stable_levels.clear();
        forward_structure_changes.clear();
        forward_statistics.clear();
        backward_statistics.clear();

        dag = &dag_;
    }

    void compute_isomorphism_map();

    bool divide();
};