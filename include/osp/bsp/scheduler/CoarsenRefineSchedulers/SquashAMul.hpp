/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include "osp/auxiliary/Balanced_Coin_Flips.hpp"
#include "osp/bsp/scheduler/MultilevelCoarseAndSchedule.hpp"
#include "osp/coarser/SquashA/SquashA.hpp"

namespace osp {

template<typename Graph_t, typename Graph_t_coarse>
class SquashAMul : public MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse> {
    private:
        vertex_idx_t<Graph_t> min_nodes{ 0 };
        Thue_Morse_Sequence thue_coin{};
        Biased_Random balanced_random{};
        
        // Coarser Params
        SquashAParams::Parameters params;
        // Initial coarser
        SquashA<Graph_t, Graph_t_coarse> coarser_initial;
        // Subsequent coarser
        SquashA<Graph_t_coarse, Graph_t_coarse> coarser_secondary;

        void updateParams();
        
        RETURN_STATUS run_contractions() override;
        
    public:
        void setParams(SquashAParams::Parameters params_) { params = params_; };
        void setMinimumNumberVertices(vertex_idx_t<Graph_t> num) { min_nodes = num; };
        
        std::string getCoarserName() const { return "SquashA"; };
};

template<typename Graph_t, typename Graph_t_coarse>
void SquashAMul<Graph_t, Graph_t_coarse>::updateParams() {
    params.use_structured_poset = thue_coin.get_flip();
    params.use_top_poset = balanced_random.get_flip();

    coarser_initial.setParams(params);
    coarser_secondary.setParams(params);
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS SquashAMul<Graph_t, Graph_t_coarse>::run_contractions() {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    if (min_nodes == 0) {
        min_nodes = MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::getOriginalInstance()->numberOfProcessors() * 1000;
    }

    Biased_Random_with_side_bias coin( params.edge_sort_ratio );

    bool first_coarsen = true;
    unsigned no_change_in_a_row = 0;
    vertex_idx_t<Graph_t> current_num_vertices = MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::getOriginalInstance()->numberOfVertices();

    while( no_change_in_a_row < params.num_rep_without_node_decrease && current_num_vertices > min_nodes ) {
        updateParams();

        Graph_t_coarse coarsened_dag;
        std::vector<vertex_idx_t<Graph_t_coarse>> contraction_map;
        bool coarsen_success;

        if (first_coarsen) {
            coarsen_success = coarser_initial.coarsenDag(MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::getOriginalInstance()->getComputationalDag(), coarsened_dag, contraction_map);
            first_coarsen = false;
        } else {
            coarsen_success = coarser_secondary.coarsenDag(MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::dag_history.back()->getComputationalDag(), coarsened_dag, contraction_map);
        }
        
        if (!coarsen_success) {
            status = RETURN_STATUS::ERROR;
        }

        status = std::max(status, MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::add_contraction(std::move(contraction_map), std::move(coarsened_dag)));
        
        vertex_idx_t<Graph_t> new_num_vertices = MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::dag_history.back()->numberOfVertices();

        if (new_num_vertices == current_num_vertices) {
            no_change_in_a_row++;
        } else {
            no_change_in_a_row = 0;
            current_num_vertices = new_num_vertices;
        }
    }

    return status;
}






} // end namespace osp