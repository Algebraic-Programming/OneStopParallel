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

namespace sarkar_params {

enum class BufferMergeMode { OFF, FAN_IN, FAN_OUT, HOMOGENEOUS, FULL };

template <typename CommCostType>
struct MulParameters {
    std::size_t seed_{42U};
    double geomDecay_{0.875};
    double leniency_{0.0};
    std::vector<CommCostType> commCostVec_{std::initializer_list<CommCostType>{}};
    CommCostType maxWeight_{std::numeric_limits<CommCostType>::max()};
    CommCostType smallWeightThreshold_{std::numeric_limits<CommCostType>::lowest()};
    unsigned maxNumIterationWithoutChanges_{3U};
    BufferMergeMode bufferMergeMode_{BufferMergeMode::OFF};
};

}    // namespace sarkar_params

template <typename GraphT, typename GraphTCoarse>
class SarkarMul : public MultilevelCoarser<GraphT, GraphTCoarse> {
  private:
    bool firstCoarsen_{true};
    Thue_Morse_Sequence thueCoin_{42U};
    Biased_Random balancedRandom_{42U};

    // Multilevel coarser parameters
    SarkarParams::MulParameters<v_workw_t<Graph_t>> mlParams_;
    // Coarser parameters
    SarkarParams::Parameters<v_workw_t<Graph_t>> params_;
    // Initial coarser
    Sarkar<GraphT, GraphTCoarse> coarserInitial_;
    // Subsequent coarser
    Sarkar<GraphTCoarse, GraphTCoarse> coarserSecondary_;

    void SetSeed();
    void InitParams();
    void UpdateParams();

    RETURN_STATUS RunSingleContractionMode(vertex_idx_t<Graph_t> &diffVertices);
    RETURN_STATUS RunBufferMerges();
    RETURN_STATUS RunContractions(v_workw_t<Graph_t> commCost);
    RETURN_STATUS run_contractions() override;

  public:
    void SetParameters(SarkarParams::MulParameters<v_workw_t<Graph_t>> mlParams) {
        ml_params = std::move(ml_params_);
        SetSeed();
        InitParams();
    };

    std::string GetCoarserName() const { return "Sarkar"; };
};

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::SetSeed() {
    constexpr std::size_t seedReduction = 4096U;
    thue_coin = Thue_Morse_Sequence(ml_params.seed % seedReduction);
    balanced_random = Biased_Random(ml_params.seed);
}

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::InitParams() {
    firstCoarsen_ = true;

    params.geomDecay = ml_params.geomDecay;
    params.leniency = ml_params.leniency;
    params.maxWeight = ml_params.maxWeight;
    params.smallWeightThreshold = ml_params.smallWeightThreshold;

    if (ml_params.commCostVec.empty()) {
        v_workw_t<Graph_t> syncCosts = 128;
        syncCosts = std::max(syncCosts, static_cast<v_workw_t<Graph_t>>(1));

        while (syncCosts >= static_cast<v_workw_t<Graph_t>>(1)) {
            ml_params.commCostVec.emplace_back(syncCosts);
            syncCosts /= 2;
        }
    }

    std::sort(ml_params.commCostVec.begin(), ml_params.commCostVec.end());

    UpdateParams();
}

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::UpdateParams() {
    coarser_initial.setParameters(params);
    coarser_secondary.setParameters(params);
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS SarkarMul<GraphT, GraphTCoarse>::RunSingleContractionMode(vertex_idx_t<Graph_t> &diffVertices) {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    vertex_idx_t<Graph_t> currentNumVertices;
    if (firstCoarsen_) {
        currentNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::getOriginalGraph()->num_vertices();
    } else {
        currentNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::dag_history.back()->num_vertices();
    }

    GraphTCoarse coarsenedDag;
    std::vector<vertex_idx_t<Graph_t_coarse>> contractionMap;
    bool coarsenSuccess;

    if (firstCoarsen_) {
        coarsenSuccess = coarserInitial_.coarsenDag(
            *(MultilevelCoarser<GraphT, GraphTCoarse>::getOriginalGraph()), coarsenedDag, contraction_map);
        firstCoarsen_ = false;
    } else {
        coarsenSuccess = coarserSecondary_.coarsenDag(
            *(MultilevelCoarser<GraphT, GraphTCoarse>::dag_history.back()), coarsenedDag, contraction_map);
    }

    if (!coarsenSuccess) {
        status = RETURN_STATUS::ERROR;
    }

    status = std::max(
        status, MultilevelCoarser<GraphT, GraphTCoarse>::add_contraction(std::move(contraction_map), std::move(coarsenedDag)));

    vertex_idx_t<Graph_t> newNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::dag_history.back()->num_vertices();
    diffVertices = current_num_vertices - new_num_vertices;

    return status;
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS SarkarMul<GraphT, GraphTCoarse>::RunContractions(v_workw_t<Graph_t> commCost) {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;
    vertex_idx_t<Graph_t> diff = 0;

    params.commCost = commCost;
    UpdateParams();

    unsigned outerNoChange = 0;
    while (outer_no_change < ml_params.max_num_iteration_without_changes) {
        unsigned innerNoChange = 0;
        bool outerChange = false;

        // Lines
        while (inner_no_change < ml_params.max_num_iteration_without_changes) {
            params.mode = SarkarParams::Mode::LINES;
            params.useTopPoset = thue_coin.get_flip();
            UpdateParams();

            status = std::max(status, run_single_contraction_mode(diff));

            if (diff > 0) {
                outerChange = true;
                innerNoChange = 0;
            } else {
                innerNoChange++;
            }
        }
        innerNoChange = 0;

        // Partial Fans
        while (inner_no_change < ml_params.max_num_iteration_without_changes) {
            params.mode = thue_coin.get_flip() ? SarkarParams::Mode::FAN_IN_PARTIAL : SarkarParams::Mode::FAN_OUT_PARTIAL;
            UpdateParams();

            status = std::max(status, run_single_contraction_mode(diff));

            if (diff > 0) {
                outerChange = true;
                innerNoChange = 0;
            } else {
                innerNoChange++;
            }
        }
        innerNoChange = 0;

        // Full Fans
        while (inner_no_change < ml_params.max_num_iteration_without_changes) {
            params.mode = thue_coin.get_flip() ? SarkarParams::Mode::FAN_IN_FULL : SarkarParams::Mode::FAN_OUT_FULL;
            UpdateParams();

            status = std::max(status, run_single_contraction_mode(diff));

            if (diff > 0) {
                outerChange = true;
                innerNoChange = 0;
            } else {
                innerNoChange++;
            }
        }
        innerNoChange = 0;

        // Levels
        while (inner_no_change < ml_params.max_num_iteration_without_changes) {
            params.mode = thue_coin.get_flip() ? SarkarParams::Mode::LEVEL_EVEN : SarkarParams::Mode::LEVEL_ODD;
            params.useTopPoset = balanced_random.get_flip();
            UpdateParams();

            status = std::max(status, run_single_contraction_mode(diff));

            if (diff > 0) {
                outerChange = true;
                innerNoChange = 0;
            } else {
                innerNoChange++;
            }
        }

        if (outerChange) {
            outerNoChange = 0;
        } else {
            outerNoChange++;
        }
    }

    return status;
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS SarkarMul<GraphT, GraphTCoarse>::RunBufferMerges() {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    unsigned noChange = 0;
    while (no_change < ml_params.max_num_iteration_without_changes) {
        vertex_idx_t<Graph_t> diff = 0;
        if ((ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::HOMOGENEOUS)
            || (ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::FULL && diff == 0)) {
            params.mode = SarkarParams::Mode::HOMOGENEOUS_BUFFER;
            UpdateParams();
            status = std::max(status, run_single_contraction_mode(diff));
        }
        if (ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::FAN_IN) {
            params.mode = SarkarParams::Mode::FAN_IN_BUFFER;
            UpdateParams();
            status = std::max(status, run_single_contraction_mode(diff));
        }
        if (ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::FAN_OUT) {
            params.mode = SarkarParams::Mode::FAN_OUT_BUFFER;
            UpdateParams();
            status = std::max(status, run_single_contraction_mode(diff));
        }
        if (ml_params.buffer_merge_mode == SarkarParams::BufferMergeMode::FULL && diff == 0) {
            const bool flip = thue_coin.get_flip();
            params.mode = flip ? SarkarParams::Mode::FAN_IN_BUFFER : SarkarParams::Mode::FAN_OUT_BUFFER;
            UpdateParams();
            status = std::max(status, run_single_contraction_mode(diff));

            if (diff == 0) {
                params.mode = (!flip) ? SarkarParams::Mode::FAN_IN_BUFFER : SarkarParams::Mode::FAN_OUT_BUFFER;
                UpdateParams();
                status = std::max(status, run_single_contraction_mode(diff));
            }
        }

        if (diff > 0) {
            noChange = 0;
            status = std::max(status, run_contractions(ml_params.commCostVec.back()));
        } else {
            noChange++;
        }
    }

    return status;
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS SarkarMul<GraphT, GraphTCoarse>::RunContractions() {
    InitParams();

    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    for (const v_workw_t<Graph_t> commCost : ml_params.commCostVec) {
        status = std::max(status, run_contractions(commCost));
    }

    if (ml_params.buffer_merge_mode != SarkarParams::BufferMergeMode::OFF) {
        status = std::max(status, run_buffer_merges());
    }

    return status;
}

}    // end namespace osp
