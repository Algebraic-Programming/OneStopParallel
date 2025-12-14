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
    ThueMorseSequence thueCoin_{42U};
    BiasedRandom balancedRandom_{42U};

    // Multilevel coarser parameters
    sarkar_params::MulParameters<VWorkwT<GraphT>> mlParams_;
    // Coarser parameters
    sarkar_params::Parameters<VWorkwT<GraphT>> params_;
    // Initial coarser
    Sarkar<GraphT, GraphTCoarse> coarserInitial_;
    // Subsequent coarser
    Sarkar<GraphTCoarse, GraphTCoarse> coarserSecondary_;

    void SetSeed();
    void InitParams();
    void UpdateParams();

    ReturnStatus RunSingleContractionMode(VertexIdxT<GraphT> &diffVertices);
    ReturnStatus RunBufferMerges();
    ReturnStatus RunContractions(VWorkwT<GraphT> commCost);
    ReturnStatus RunContractions() override;

  public:
    void SetParameters(sarkar_params::MulParameters<VWorkwT<GraphT>> mlParams) {
        mlParams_ = std::move(mlParams);
        SetSeed();
        InitParams();
    };

    std::string getCoarserName() const override { return "Sarkar"; };
};

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::SetSeed() {
    constexpr std::size_t seedReduction = 4096U;
    thueCoin_ = ThueMorseSequence(mlParams_.seed_ % seedReduction);
    balancedRandom_ = BiasedRandom(mlParams_.seed_);
}

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::InitParams() {
    firstCoarsen_ = true;

    params_.geomDecay_ = mlParams_.geomDecay_;
    params_.leniency_ = mlParams_.leniency_;
    params_.maxWeight_ = mlParams_.maxWeight_;
    params_.smallWeightThreshold_ = mlParams_.smallWeightThreshold_;

    if (mlParams_.commCostVec_.empty()) {
        VWorkwT<GraphT> syncCosts = 128;
        syncCosts = std::max(syncCosts, static_cast<VWorkwT<GraphT>>(1));

        while (syncCosts >= static_cast<VWorkwT<GraphT>>(1)) {
            mlParams_.commCostVec_.emplace_back(syncCosts);
            syncCosts /= 2;
        }
    }

    std::sort(mlParams_.commCostVec_.begin(), mlParams_.commCostVec_.end());

    UpdateParams();
}

template <typename GraphT, typename GraphTCoarse>
void SarkarMul<GraphT, GraphTCoarse>::UpdateParams() {
    coarserInitial_.SetParameters(params_);
    coarserSecondary_.SetParameters(params_);
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus SarkarMul<GraphT, GraphTCoarse>::RunSingleContractionMode(VertexIdxT<GraphT> &diffVertices) {
    ReturnStatus status = ReturnStatus::OSP_SUCCESS;

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
        status = ReturnStatus::ERROR;
    }

    status = std::max(
        status, MultilevelCoarser<GraphT, GraphTCoarse>::AddContraction(std::move(contractionMap), std::move(coarsenedDag)));

    VertexIdxT<GraphT> newNumVertices = MultilevelCoarser<GraphT, GraphTCoarse>::dagHistory_.back()->NumVertices();
    diffVertices = currentNumVertices - newNumVertices;

    return status;
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus SarkarMul<GraphT, GraphTCoarse>::RunContractions(VWorkwT<GraphT> commCost) {
    ReturnStatus status = ReturnStatus::OSP_SUCCESS;
    VertexIdxT<GraphT> diff = 0;

    params_.commCost_ = commCost;
    UpdateParams();

    unsigned outerNoChange = 0;
    while (outerNoChange < mlParams_.maxNumIterationWithoutChanges_) {
        unsigned innerNoChange = 0;
        bool outerChange = false;

        // Lines
        while (innerNoChange < mlParams_.maxNumIterationWithoutChanges_) {
            params_.mode_ = sarkar_params::Mode::LINES;
            params_.useTopPoset_ = thueCoin_.GetFlip();
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
        while (innerNoChange < mlParams_.maxNumIterationWithoutChanges_) {
            params_.mode_ = thueCoin_.GetFlip() ? sarkar_params::Mode::FAN_IN_PARTIAL : sarkar_params::Mode::FAN_OUT_PARTIAL;
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
        while (innerNoChange < mlParams_.maxNumIterationWithoutChanges_) {
            params_.mode_ = thueCoin_.GetFlip() ? sarkar_params::Mode::FAN_IN_FULL : sarkar_params::Mode::FAN_OUT_FULL;
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
        while (innerNoChange < mlParams_.maxNumIterationWithoutChanges_) {
            params_.mode_ = thueCoin_.GetFlip() ? sarkar_params::Mode::LEVEL_EVEN : sarkar_params::Mode::LEVEL_ODD;
            params_.useTopPoset_ = balancedRandom_.GetFlip();
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
ReturnStatus SarkarMul<GraphT, GraphTCoarse>::RunBufferMerges() {
    ReturnStatus status = ReturnStatus::OSP_SUCCESS;

    unsigned noChange = 0;
    while (noChange < mlParams_.maxNumIterationWithoutChanges_) {
        VertexIdxT<GraphT> diff = 0;
        if ((mlParams_.bufferMergeMode_ == sarkar_params::BufferMergeMode::HOMOGENEOUS)
            || (mlParams_.bufferMergeMode_ == sarkar_params::BufferMergeMode::FULL && diff == 0)) {
            params_.mode_ = sarkar_params::Mode::HOMOGENEOUS_BUFFER;
            UpdateParams();
            status = std::max(status, RunSingleContractionMode(diff));
        }
        if (mlParams_.bufferMergeMode_ == sarkar_params::BufferMergeMode::FAN_IN) {
            params_.mode_ = sarkar_params::Mode::FAN_IN_BUFFER;
            UpdateParams();
            status = std::max(status, RunSingleContractionMode(diff));
        }
        if (mlParams_.bufferMergeMode_ == sarkar_params::BufferMergeMode::FAN_OUT) {
            params_.mode_ = sarkar_params::Mode::FAN_OUT_BUFFER;
            UpdateParams();
            status = std::max(status, RunSingleContractionMode(diff));
        }
        if (mlParams_.bufferMergeMode_ == sarkar_params::BufferMergeMode::FULL && diff == 0) {
            const bool flip = thueCoin_.GetFlip();
            params_.mode_ = flip ? sarkar_params::Mode::FAN_IN_BUFFER : sarkar_params::Mode::FAN_OUT_BUFFER;
            UpdateParams();
            status = std::max(status, RunSingleContractionMode(diff));

            if (diff == 0) {
                params_.mode_ = (!flip) ? sarkar_params::Mode::FAN_IN_BUFFER : sarkar_params::Mode::FAN_OUT_BUFFER;
                UpdateParams();
                status = std::max(status, RunSingleContractionMode(diff));
            }
        }

        if (diff > 0) {
            noChange = 0;
            status = std::max(status, RunContractions(mlParams_.commCostVec_.back()));
        } else {
            noChange++;
        }
    }

    return status;
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus SarkarMul<GraphT, GraphTCoarse>::RunContractions() {
    InitParams();

    ReturnStatus status = ReturnStatus::OSP_SUCCESS;

    for (const VWorkwT<GraphT> commCost : mlParams_.commCostVec_) {
        status = std::max(status, RunContractions(commCost));
    }

    if (mlParams_.bufferMergeMode_ != sarkar_params::BufferMergeMode::OFF) {
        status = std::max(status, RunBufferMerges());
    }

    return status;
}

}    // end namespace osp
