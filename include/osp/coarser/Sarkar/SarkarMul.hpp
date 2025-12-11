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

template <typename CommCostType>
struct MulParameters {
    std::size_t seed{42U};
    double geomDecay{0.875};
    double leniency{0.0};
    std::vector<CommCostType> commCostVec{std::initializer_list<CommCostType>{}};
    CommCostType maxWeight{std::numeric_limits<CommCostType>::max()};
    CommCostType smallWeightThreshold{std::numeric_limits<CommCostType>::lowest()};
    unsigned maxNumIterationWithoutChanges{3U};
    BufferMergeMode bufferMergeMode{BufferMergeMode::OFF};
};

}    // end namespace SarkarParams

template <typename GraphT, typename GraphTCoarse>
class SarkarMul : public MultilevelCoarser<GraphT, GraphTCoarse> {
  private:
    bool firstCoarsen_{true};
    ThueMorseSequence thueCoin_{42U};
    BiasedRandom balancedRandom_{42U};

    // Multilevel coarser parameters
    SarkarParams::MulParameters<VWorkwT<GraphT>> mlParams_;
    // Coarser parameters
    SarkarParams::Parameters<VWorkwT<GraphT>> params_;
    // Initial coarser
    Sarkar<GraphT, GraphTCoarse> coarserInitial_;
    // Subsequent coarser
    Sarkar<GraphTCoarse, GraphTCoarse> coarserSecondary_;

    void SetSeed();
    void InitParams();
    void UpdateParams();

    RETURN_STATUS RunSingleContractionMode(VertexIdxT<GraphT> &diff_vertices);
    RETURN_STATUS RunBufferMerges();
    RETURN_STATUS RunContractions(VWorkwT<GraphT> commCost);
    RETURN_STATUS run_contractions() override;

  public:
    void SetParameters(SarkarParams::MulParameters<VWorkwT<GraphT>> mlParams) {
        mlParams_ = std::move(mlParams);
        SetSeed();
        InitParams();
    };

    std::string GetCoarserName() const { return "Sarkar"; };
};

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::SetSeed() {
    constexpr std::size_t seedReduction = 4096U;
    thueCoin_ = ThueMorseSequence(mlParams_.seed % seedReduction);
    balancedRandom_ = BiasedRandom(mlParams_.seed);
}

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::InitParams() {
    firstCoarsen_ = true;

    params_.geomDecay = mlParams_.geomDecay;
    params_.leniency = mlParams_.leniency;
    params_.maxWeight = mlParams_.maxWeight;
    params_.smallWeightThreshold = mlParams_.smallWeightThreshold;

    if (mlParams_.commCostVec.empty()) {
        VWorkwT<GraphT> syncCosts = 128;
        syncCosts = std::max(syncCosts, static_cast<VWorkwT<GraphT>>(1));

        while (syncCosts >= static_cast<VWorkwT<GraphT>>(1)) {
            mlParams_.commCostVec.emplace_back(syncCosts);
            syncCosts /= 2;
        }
    }

    std::sort(mlParams_.commCostVec.begin(), mlParams_.commCostVec.end());

    UpdateParams();
}

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::UpdateParams() {
    coarserInitial_.SetParameters(params_);
    coarserSecondary_.SetParameters(params_);
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS SarkarMul<GraphT, GraphTCoarse>::RunSingleContractionMode(VertexIdxT<GraphT> &diff_vertices) {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    VertexIdxT<GraphT> currentNumVertices;
    if (firstCoarsen_) {
        currentNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::GetOriginalGraph()->NumVertices();
    } else {
        currentNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::dagHistory_.back()->NumVertices();
    }

    GraphTCoarse coarsenedDag;
    std::vector<VertexIdxT<GraphTCoarse>> contractionMap;
    bool coarsenSuccess;

    if (firstCoarsen_) {
        coarsenSuccess = coarserInitial_.CoarsenDag(
            *(MultilevelCoarser<GraphT, GraphTCoarse>::GetOriginalGraph()), coarsenedDag, contractionMap);
        firstCoarsen_ = false;
    } else {
        coarsenSuccess = coarserSecondary_.CoarsenDag(
            *(MultilevelCoarser<GraphT, GraphTCoarse>::dagHistory_.back()), coarsenedDag, contractionMap);
    }

    if (!coarsenSuccess) {
        status = RETURN_STATUS::ERROR;
    }

    status = std::max(
        status, MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(std::move(contractionMap), std::move(coarsenedDag)));

    VertexIdxT<GraphT> newNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::dagHistory_.back()->NumVertices();
    diff_vertices = currentNumVertices - newNumVertices;

    return status;
}

template <typename GraphT, typename GraphTCoarse>
RETURN_STATUS SarkarMul<GraphT, GraphTCoarse>::RunContractions(VWorkwT<GraphT> commCost) {
    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;
    VertexIdxT<GraphT> diff = 0;

    params_.commCost = commCost;
    UpdateParams();

    unsigned outerNoChange = 0;
    while (outerNoChange < mlParams_.maxNumIterationWithoutChanges) {
        unsigned innerNoChange = 0;
        bool outerChange = false;

        // Lines
        while (innerNoChange < mlParams_.maxNumIterationWithoutChanges) {
            params_.mode = SarkarParams::Mode::LINES;
            params_.useTopPoset = thueCoin_.GetFlip();
            UpdateParams();

            status = std::max(status, RunSingleContractionMode(diff));

            if (diff > 0) {
                outerChange = true;
                innerNoChange = 0;
            } else {
                innerNoChange++;
            }
        }
        innerNoChange = 0;

        // Partial Fans
        while (innerNoChange < mlParams_.maxNumIterationWithoutChanges) {
            params_.mode = thueCoin_.GetFlip() ? SarkarParams::Mode::FAN_IN_PARTIAL : SarkarParams::Mode::FAN_OUT_PARTIAL;
            UpdateParams();

            status = std::max(status, RunSingleContractionMode(diff));

            if (diff > 0) {
                outerChange = true;
                innerNoChange = 0;
            } else {
                innerNoChange++;
            }
        }
        innerNoChange = 0;

        // Full Fans
        while (innerNoChange < mlParams_.maxNumIterationWithoutChanges) {
            params_.mode = thueCoin_.GetFlip() ? SarkarParams::Mode::FAN_IN_FULL : SarkarParams::Mode::FAN_OUT_FULL;
            UpdateParams();

            status = std::max(status, RunSingleContractionMode(diff));

            if (diff > 0) {
                outerChange = true;
                innerNoChange = 0;
            } else {
                innerNoChange++;
            }
        }
        innerNoChange = 0;

        // Levels
        while (innerNoChange < mlParams_.maxNumIterationWithoutChanges) {
            params_.mode = thueCoin_.GetFlip() ? SarkarParams::Mode::LEVEL_EVEN : SarkarParams::Mode::LEVEL_ODD;
            params_.useTopPoset = balancedRandom_.GetFlip();
            UpdateParams();

            status = std::max(status, RunSingleContractionMode(diff));

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
    while (noChange < mlParams_.maxNumIterationWithoutChanges) {
        VertexIdxT<GraphT> diff = 0;
        if ((mlParams_.bufferMergeMode == SarkarParams::BufferMergeMode::HOMOGENEOUS)
            || (mlParams_.bufferMergeMode == SarkarParams::BufferMergeMode::FULL && diff == 0)) {
            params_.mode = SarkarParams::Mode::HOMOGENEOUS_BUFFER;
            UpdateParams();
            status = std::max(status, RunSingleContractionMode(diff));
        }
        if (mlParams_.bufferMergeMode == SarkarParams::BufferMergeMode::FAN_IN) {
            params_.mode = SarkarParams::Mode::FAN_IN_BUFFER;
            UpdateParams();
            status = std::max(status, RunSingleContractionMode(diff));
        }
        if (mlParams_.bufferMergeMode == SarkarParams::BufferMergeMode::FAN_OUT) {
            params_.mode = SarkarParams::Mode::FAN_OUT_BUFFER;
            UpdateParams();
            status = std::max(status, RunSingleContractionMode(diff));
        }
        if (mlParams_.bufferMergeMode == SarkarParams::BufferMergeMode::FULL && diff == 0) {
            const bool flip = thueCoin_.GetFlip();
            params_.mode = flip ? SarkarParams::Mode::FAN_IN_BUFFER : SarkarParams::Mode::FAN_OUT_BUFFER;
            UpdateParams();
            status = std::max(status, RunSingleContractionMode(diff));

            if (diff == 0) {
                params_.mode = (!flip) ? SarkarParams::Mode::FAN_IN_BUFFER : SarkarParams::Mode::FAN_OUT_BUFFER;
                UpdateParams();
                status = std::max(status, RunSingleContractionMode(diff));
            }
        }

        if (diff > 0) {
            noChange = 0;
            status = std::max(status, run_contractions(mlParams_.commCostVec.back()));
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

    for (const VWorkwT<GraphT> commCost : mlParams_.commCostVec) {
        status = std::max(status, run_contractions(commCost));
    }

    if (mlParams_.bufferMergeMode != SarkarParams::BufferMergeMode::OFF) {
        status = std::max(status, RunBufferMerges());
    }

    return status;
}

}    // end namespace osp
