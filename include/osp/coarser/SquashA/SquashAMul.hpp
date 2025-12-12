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
#include "osp/coarser/SquashA/SquashA.hpp"

namespace osp {

template <typename GraphT, typename GraphTCoarse>
class SquashAMul : public MultilevelCoarser<GraphT, GraphTCoarse> {
  private:
    vertex_idx_t<Graph_t> minNodes_{1};
    Thue_Morse_Sequence thueCoin_{};
    Biased_Random balancedRandom_{};

    // Coarser Params
    squash_a_params::Parameters params_;
    // Initial coarser
    SquashA<GraphT, GraphTCoarse> coarserInitial_;
    // Subsequent coarser
    SquashA<GraphTCoarse, GraphTCoarse> coarserSecondary_;

    void UpdateParams();

    RETURN_STATUS run_contractions() override;

  public:
    void SetParams(squash_a_params::Parameters params) { params_ = params; };

    void SetMinimumNumberVertices(vertex_idx_t<Graph_t> num) { min_nodes = num; };

    std::string GetCoarserName() const { return "SquashA"; };
};

template <typename GraphT, typename GraphTCoarse>
void SquashAMul<GraphT, GraphTCoarse>::UpdateParams() {
    params.use_structured_poset = thue_coin.get_flip();
    params.use_top_poset = balanced_random.get_flip();

    coarserInitial_.setParams(params_);
    coarserSecondary_.setParams(params_);
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS SquashAMul<GraphT, GraphTCoarse>::RunContractions() {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    Biased_Random_with_side_bias coin(params_.edgeSortRatio_);

    bool firstCoarsen = true;
    unsigned noChangeInARow = 0;
    vertex_idx_t<Graph_t> currentNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::getOriginalGraph()->NumVertices();

    while (no_change_in_a_row < params.num_rep_without_node_decrease && current_num_vertices > min_nodes) {
        UpdateParams();

        GraphTCoarse coarsenedDag;
        std::vector<vertex_idx_t<Graph_t_coarse>> contractionMap;
        bool coarsenSuccess;

        if (firstCoarsen) {
            coarsenSuccess = coarserInitial_.coarsenDag(
                *(MultilevelCoarser<GraphT, GraphTCoarse>::getOriginalGraph()), coarsenedDag, contraction_map);
            firstCoarsen = false;
        } else {
            coarsenSuccess = coarserSecondary_.coarsenDag(
                *(MultilevelCoarser<GraphT, GraphTCoarse>::dag_history.back()), coarsenedDag, contraction_map);
        }

        if (!coarsenSuccess) {
            status = RETURN_STATUS::ERROR;
        }

        status = std::max(
            status, MultilevelCoarser<GraphT, GraphTCoarse>::add_contraction(std::move(contraction_map), std::move(coarsenedDag)));

        vertex_idx_t<Graph_t> newNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::dag_history.back()->NumVertices();

        if (newNumVertices == current_num_vertices) {
            noChangeInARow++;
        } else {
            noChangeInARow = 0;
            currentNumVertices = new_num_vertices;
        }
    }

    return status;
}

}    // end namespace osp
