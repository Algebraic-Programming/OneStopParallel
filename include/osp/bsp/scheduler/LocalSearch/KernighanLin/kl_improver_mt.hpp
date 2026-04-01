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
#include "comm_cost_modules/kl_max_bsp_comm_cost.hpp"
#include "comm_cost_modules/kl_total_comm_cost.hpp"
#include "kl_improver_sync_parallel.hpp"

// Memory constraint modules
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"

namespace osp {

using DoubleCostT = double;

template <typename GraphT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          bool useNodeCommunicationCostsArg = true>
using KlTotalCommImproverMt
    = KlSyncParallelImprover<GraphT,
                             KlTotalCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, windowSize, useNodeCommunicationCostsArg>,
                             MemoryConstraintT,
                             windowSize,
                             DoubleCostT>;

template <typename GraphT,
          typename MemoryConstraintT = LsLocalMemoryConstraint<GraphT>,
          unsigned windowSize = 1,
          bool useNodeCommunicationCostsArg = true>
using KlTotalCommImproverLocalMemConstrMt
    = KlSyncParallelImprover<GraphT,
                             KlTotalCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, windowSize, useNodeCommunicationCostsArg>,
                             MemoryConstraintT,
                             windowSize,
                             DoubleCostT>;

template <typename GraphT, typename MemoryConstraintT = NoLocalSearchMemoryConstraint, unsigned windowSize = 1>
using KlTotalLambdaCommImproverMt
    = KlSyncParallelImprover<GraphT,
                             KlHyperTotalCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, windowSize>,
                             MemoryConstraintT,
                             windowSize,
                             DoubleCostT>;

template <typename GraphT, typename MemoryConstraintT = LsLocalMemoryConstraint<GraphT>, unsigned windowSize = 1>
using KlTotalLambdaCommImproverLocalMemConstrMt
    = KlSyncParallelImprover<GraphT,
                             KlHyperTotalCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, windowSize>,
                             MemoryConstraintT,
                             windowSize,
                             DoubleCostT>;

template <typename GraphT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          typename CommPolicy = EagerCommCostPolicy,
          unsigned windowSize = 1>
using KlBspCommImproverMt
    = KlSyncParallelImprover<GraphT,
                             KlBspCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, CommPolicy, windowSize>,
                             MemoryConstraintT,
                             windowSize,
                             DoubleCostT>;

template <typename GraphT,
          typename MemoryConstraintT = LsLocalMemoryConstraint<GraphT>,
          typename CommPolicy = EagerCommCostPolicy,
          unsigned windowSize = 1>
using KlBspCommImproverLocalMemConstrMt
    = KlSyncParallelImprover<GraphT,
                             KlBspCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, CommPolicy, windowSize>,
                             MemoryConstraintT,
                             windowSize,
                             DoubleCostT>;

template <typename GraphT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          typename CommPolicy = EagerCommCostPolicy,
          unsigned windowSize = 1>
using KlMaxBspCommImproverMt
    = KlSyncParallelImprover<GraphT,
                             KlMaxBspCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, CommPolicy, windowSize>,
                             MemoryConstraintT,
                             windowSize,
                             DoubleCostT>;

template <typename GraphT,
          typename MemoryConstraintT = LsLocalMemoryConstraint<GraphT>,
          typename CommPolicy = EagerCommCostPolicy,
          unsigned windowSize = 1>
using KlMaxBspCommImproverLocalMemConstrMt
    = KlSyncParallelImprover<GraphT,
                             KlMaxBspCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, CommPolicy, windowSize>,
                             MemoryConstraintT,
                             windowSize,
                             DoubleCostT>;

}    // namespace osp
