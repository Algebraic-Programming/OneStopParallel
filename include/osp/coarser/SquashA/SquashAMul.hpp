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
    VertexIdxT<GraphT> minNodes_{1};
    ThueMorseSequence thueCoin_{};
    BiasedRandom balancedRandom_{};

    // Coarser Params
    squash_a_params::Parameters params_;
    // Initial coarser
    SquashA<GraphT, GraphTCoarse> coarserInitial_;
    // Subsequent coarser
    SquashA<GraphTCoarse, GraphTCoarse> coarserSecondary_;

    void UpdateParams();

    ReturnStatus RunContractions() override;

  public:
    void SetParams(squash_a_params::Parameters params) { params_ = params; };

    void SetMinimumNumberVertices(VertexIdxT<GraphT> num) { minNodes_ = num; };

    std::string getCoarserName() const override { return "SquashA"; };
};

template <typename GraphT, typename GraphTCoarse>
void SquashAMul<GraphT, GraphTCoarse>::UpdateParams() {
    params_.useStructuredPoset_ = thueCoin_.GetFlip();
    params_.useTopPoset_ = balancedRandom_.GetFlip();

    coarserInitial_.SetParams(params_);
    coarserSecondary_.SetParams(params_);
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus SquashAMul<GraphT, GraphTCoarse>::RunContractions() {
    ReturnStatus status = ReturnStatus::OSP_SUCCESS;

    BiasedRandomWithSideBias coin(params_.edgeSortRatio_);

    bool firstCoarsen = true;
    unsigned noChangeInARow = 0;
    VertexIdxT<GraphT> currentNumVertices;
    if (MultilevelCoarser<GraphT, GraphTCoarse>::GetOriginalGraph()) {
        currentNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::GetOriginalGraph()->NumVertices();
    } else {
        return ReturnStatus::ERROR;
    }

    while (noChangeInARow < params_.numRepWithoutNodeDecrease_ && currentNumVertices > minNodes_) {
        UpdateParams();

        GraphTCoarse coarsenedDag;
        std::vector<VertexIdxT<GraphTCoarse>> contractionMap;
        bool coarsenSuccess;

        if (firstCoarsen) {
            coarsenSuccess = coarserInitial_.CoarsenDag(
                *(MultilevelCoarser<GraphT, GraphTCoarse>::GetOriginalGraph()), coarsenedDag, contractionMap);
            firstCoarsen = false;
        } else {
            coarsenSuccess = coarserSecondary_.CoarsenDag(
                *(MultilevelCoarser<GraphT, GraphTCoarse>::dagHistory_.back()), coarsenedDag, contractionMap);
        }

        if (!coarsenSuccess) {
            status = ReturnStatus::ERROR;
        }

        status = std::max(
            status, MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(std::move(contractionMap), std::move(coarsenedDag)));

        VertexIdxT<GraphT> newNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::dagHistory_.back()->NumVertices();

        if (newNumVertices == currentNumVertices) {
            noChangeInARow++;
        } else {
            noChangeInARow = 0;
            currentNumVertices = newNumVertices;
        }
    }

    return status;
}

}    // end namespace osp
