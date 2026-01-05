
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

#include "comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "comm_cost_modules/kl_hyper_total_comm_cost.hpp"
#include "comm_cost_modules/kl_total_comm_cost.hpp"
#include "kl_improver_mt.hpp"
#include "kl_include.hpp"

namespace osp {

template <typename GraphT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          bool useNodeCommunicationCostsArg = true>
using KlTotalCommImproverMt
    = KlImproverMt<GraphT,
                   KlTotalCommCostFunction<GraphT, double, MemoryConstraintT, windowSize, useNodeCommunicationCostsArg>,
                   MemoryConstraintT,
                   windowSize,
                   double>;

template <typename GraphT, typename MemoryConstraintT = NoLocalSearchMemoryConstraint, unsigned windowSize = 1>
using KlTotalLambdaCommImproverMt
    = KlImproverMt<GraphT, KlHyperTotalCommCostFunction<GraphT, double, MemoryConstraintT, windowSize>, MemoryConstraintT, windowSize, double>;

template <typename GraphT, typename MemoryConstraintT = NoLocalSearchMemoryConstraint, unsigned windowSize = 1>
using KlBspCommImproverMt
    = KlImproverMt<GraphT, KlBspCommCostFunction<GraphT, double, MemoryConstraintT, windowSize>, MemoryConstraintT, windowSize, double>;

}    // namespace osp
