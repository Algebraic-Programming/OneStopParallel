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
#include "osp/coarser/MultilevelCoarser.hpp"
#include "osp/coarser/Sarkar/Sarkar.hpp"

namespace osp {

namespace SarkarParams {

enum class BufferMergeMode { OFF, FAN_IN, FAN_OUT, HOMOGENEOUS, FULL };

template<typename commCostType>
struct MulParameters {
    std::size_t seed{42U};
    double geomDecay{0.875};
    double leniency{0.0};
    std::vector< commCostType > commCostVec{ std::initializer_list<commCostType>{} };
    commCostType maxWeight{ std::numeric_limits<commCostType>::max() };
    commCostType smallWeightThreshold{ std::numeric_limits<commCostType>::lowest() };
    unsigned max_num_iteration_without_changes{3U};
    BufferMergeMode buffer_merge_mode{BufferMergeMode::OFF};
};
} // end namespace SarkarParams

template<typename Graph_t, typename Graph_t_coarse>
class SarkarMul : public MultilevelCoarser<Graph_t, Graph_t_coarse> {
    private:
        bool first_coarsen{true};
        Thue_Morse_Sequence thue_coin{42U};
        Biased_Random balanced_random{42U};

        // Multilevel coarser parameters
        SarkarParams::MulParameters< v_workw_t<Graph_t> > ml_params;
        // Coarser parameters
        SarkarParams::Parameters< v_workw_t<Graph_t> > params;
        // Initial coarser
        Sarkar<Graph_t, Graph_t_coarse> coarser_initial;
        // Subsequent coarser
        Sarkar<Graph_t_coarse, Graph_t_coarse> coarser_secondary;

        void setSeed();
        void initParams();
        void updateParams();
        
        RETURN_STATUS run_single_contraction_mode(vertex_idx_t<Graph_t> &diff_vertices);
        RETURN_STATUS run_buffer_merges();
        RETURN_STATUS run_contractions(v_workw_t<Graph_t> commCost);
        RETURN_STATUS run_contractions() override;
        
    public:
        void setParameters(SarkarParams::MulParameters< v_workw_t<Graph_t> > ml_params_) { ml_params = std::move(ml_params_); setSeed(); initParams(); };
        
        std::string getCoarserName() const { return "Sarkar"; };
};

template<typename Graph_t, typename Graph_t_coarse>
void SarkarMul<Graph_t, Graph_t_coarse>::setSeed() {
    constexpr std::size_t seedReduction = 4096U;
    thue_coin = Thue_Morse_Sequence(ml_params.seed % seedReduction);
    balanced_random = Biased_Random(ml_params.seed);
}

template<typename Graph_t, typename Graph_t_coarse>
void SarkarMul<Graph_t, Graph_t_coarse>::initParams() {
    first_coarsen = true;

    params.geomDecay = ml_params.geomDecay;
    params.leniency = ml_params.leniency;
    params.maxWeight = ml_params.maxWeight;
    params.smallWeightThreshold = ml_params.smallWeightThreshold;

    if (ml_params.commCostVec.empty()) {
        v_workw_t<Graph_t> syncCosts = 128;
        syncCosts = std::max(syncCosts, static_cast<v_workw_t<Graph_t>>(1));
        
        while (syncCosts >= static_cast<v_workw_t<Graph_t>>(1)) {
            ml_params.commCostVec.emplace_back( syncCosts );
            syncCosts /= 2;
        }
    }

    std::sort(ml_params.commCostVec.begin(), ml_params.commCostVec.end());
    
    updateParams();
};

template<typename Graph_t, typename Graph_t_coarse>
void SarkarMul<Graph_t, Graph_t_coarse>::updateParams() {
    coarser_initial.setParameters(params);
    coarser_secondary.setParameters(params);
};

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS SarkarMul<Graph_t, Graph_t_coarse>::run_single_contraction_mode(vertex_idx_t<Graph_t> &diff_vertices) {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    vertex_idx_t<Graph_t> current_num_vertices;
    if (first_coarsen) {
        current_num_vertices = MultilevelCoarser<Graph_t, Graph_t_coarse>::getOriginalGraph()->num_vertices();
    } else {
        current_num_vertices = MultilevelCoarser<Graph_t, Graph_t_coarse>::dag_history.back()->num_vertices();
    }

    Graph_t_coarse coarsened_dag;
    std::vector<vertex_idx_t<Graph_t_coarse>> contraction_map;
    bool coarsen_success;

    if (first_coarsen) {
        coarsen_success = coarser_initial.coarsenDag(*(MultilevelCoarser<Graph_t, Graph_t_coarse>::getOriginalGraph()), coarsened_dag, contraction_map);
        first_coarsen = false;
    } else {
        coarsen_success = coarser_secondary.coarsenDag(*(MultilevelCoarser<Graph_t, Graph_t_coarse>::dag_history.back()), coarsened_dag, contraction_map);
    }
    
    if (!coarsen_success) {
        status = RETURN_STATUS::ERROR;
    }

    status = std::max(status, MultilevelCoarser<Graph_t, Graph_t_coarse>::add_contraction(std::move(contraction_map), std::move(coarsened_dag)));
    
    vertex_idx_t<Graph_t> new_num_vertices = MultilevelCoarser<Graph_t, Graph_t_coarse>::dag_history.back()->num_vertices();
    diff_vertices = current_num_vertices - new_num_vertices;

    return status;
};

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS SarkarMul<Graph_t, Graph_t_coarse>::run_contractions(v_workw_t<Graph_t> commCost) {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;
    vertex_idx_t<Graph_t> diff = 0;
    
    params.commCost = commCost;
    updateParams();
    
    unsigned outer_no_change = 0;
    while (outer_no_change < ml_params.max_num_iteration_without_changes) {
        unsigned inner_no_change = 0;
        bool outer_change = false;

        // Lines
        while (inner_no_change < ml_params.max_num_iteration_without_changes) {
            params.mode = SarkarParams::Mode::LINES;
            params.useTopPoset = thue_coin.get_flip();
            updateParams();

            status = std::max(status, run_single_contraction_mode(diff));

            if (diff > 0) {
                outer_change = true;
                inner_no_change = 0;
            } else {
                inner_no_change++;
            }
        }
        inner_no_change = 0;

        // Partial Fans
        while (inner_no_change < ml_params.max_num_iteration_without_changes) {
            params.mode = thue_coin.get_flip() ? SarkarParams::Mode::FAN_IN_PARTIAL : SarkarParams::Mode::FAN_OUT_PARTIAL;
            updateParams();

            status = std::max(status, run_single_contraction_mode(diff));

            if (diff > 0) {
                outer_change = true;
                inner_no_change = 0;
            } else {
                inner_no_change++;
            }
        }
        inner_no_change = 0;

        // Full Fans
        while (inner_no_change < ml_params.max_num_iteration_without_changes) {
            params.mode = thue_coin.get_flip() ? SarkarParams::Mode::FAN_IN_FULL : SarkarParams::Mode::FAN_OUT_FULL;
            updateParams();

            status = std::max(status, run_single_contraction_mode(diff));

            if (diff > 0) {
                outer_change = true;
                inner_no_change = 0;
            } else {
                inner_no_change++;
            }
        }
        inner_no_change = 0;

        // Levels
        while (inner_no_change < ml_params.max_num_iteration_without_changes) {
            params.mode = thue_coin.get_flip()? SarkarParams::Mode::LEVEL_EVEN : SarkarParams::Mode::LEVEL_ODD;
            params.useTopPoset = balanced_random.get_flip();
            updateParams();

            status = std::max(status, run_single_contraction_mode(diff));

            if (diff > 0) {
                outer_change = true;
                inner_no_change = 0;
            } else {
                inner_no_change++;
            }
        }



        if (outer_change) {
            outer_no_change = 0;
        } else {
            outer_no_change++;
        }
    }

    return status;
};


template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS SarkarMul<Graph_t, Graph_t_coarse>::run_buffer_merges() {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    unsigned no_change = 0;
    while (no_change < ml_params.max_num_iteration_without_changes) {        
        vertex_idx_t<Graph_t> diff = 0;
        if ((ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::HOMOGENEOUS) || (ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::FULL && diff == 0)) {
            params.mode = SarkarParams::Mode::HOMOGENEOUS_BUFFER;
            updateParams();
            status = std::max(status, run_single_contraction_mode(diff));
        }
        if (ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::FAN_IN) {
            params.mode = SarkarParams::Mode::FAN_IN_BUFFER;
            updateParams();
            status = std::max(status, run_single_contraction_mode(diff));
        }
        if (ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::FAN_OUT) {
            params.mode = SarkarParams::Mode::FAN_OUT_BUFFER;
            updateParams();
            status = std::max(status, run_single_contraction_mode(diff));
        }
        if (ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::FULL && diff == 0) {
            const bool flip = thue_coin.get_flip();
            params.mode = flip ? SarkarParams::Mode::FAN_IN_BUFFER : SarkarParams::Mode::FAN_OUT_BUFFER;
            updateParams();
            status = std::max(status, run_single_contraction_mode(diff));

            if (diff == 0) {
                params.mode = (!flip) ? SarkarParams::Mode::FAN_IN_BUFFER : SarkarParams::Mode::FAN_OUT_BUFFER;
                updateParams();
                status = std::max(status, run_single_contraction_mode(diff));
            }
        }

        if (diff > 0) {
            no_change = 0;
            status = std::max(status, run_contractions( ml_params.commCostVec.back() ));        
        } else {
            no_change++;
        }
    }

    return status;
}


template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS SarkarMul<Graph_t, Graph_t_coarse>::run_contractions() {
    initParams();

    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;
    
    for (const v_workw_t<Graph_t> commCost : ml_params.commCostVec) {
        status = std::max(status, run_contractions(commCost));
    }

    if (ml_params.buffer_merge_mode != SarkarParams::BufferMergeMode::OFF) {
        status = std::max(status, run_buffer_merges());
    }

    return status;
};







} // end namespace osp