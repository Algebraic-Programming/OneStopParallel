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

// Variant implementations (both pull in kl_improver_base.hpp)
#include "kl_improver_heap.hpp"
#include "kl_improver_scan.hpp"

// Cost function modules
#include "comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "comm_cost_modules/kl_hyper_total_comm_cost.hpp"
#include "comm_cost_modules/kl_total_comm_cost.hpp"

// Memory constraint modules
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"

namespace osp {

// =============================================================================
// FACTORY — compile-time variant selection based on cost function
//
//   isMaxCommCostFunction_ == true  (BSP max-comm):     KlImproverScan
//   isMaxCommCostFunction_ == false (total/totalLambda): KlImproverHeap
// =============================================================================

template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          typename CostT = double>
using KlImprover = std::conditional_t<CommCostFunctionT::isMaxCommCostFunction_,
                                      KlImproverScan<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>,
                                      KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>>;

// =============================================================================
// Convenience aliases  (previously in kl_include.hpp)
// =============================================================================

using DoubleCostT = double;

// --- Total comm cost (resolves to KlImproverHeap) ---

template <typename GraphT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          bool useNodeCommunicationCostsArg = true>
using KlTotalCommImprover
    = KlImprover<GraphT,
                 KlTotalCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, windowSize, useNodeCommunicationCostsArg>,
                 MemoryConstraintT,
                 windowSize,
                 DoubleCostT>;

template <typename GraphT,
          typename MemoryConstraintT = LsLocalMemoryConstraint<GraphT>,
          unsigned windowSize = 1,
          bool useNodeCommunicationCostsArg = true>
using KlTotalCommImproverLocalMemConstr
    = KlImprover<GraphT,
                 KlTotalCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, windowSize, useNodeCommunicationCostsArg>,
                 MemoryConstraintT,
                 windowSize,
                 DoubleCostT>;

// --- Total lambda comm cost / hypergraph-aware (resolves to KlImproverHeap) ---

template <typename GraphT, typename MemoryConstraintT = NoLocalSearchMemoryConstraint, unsigned windowSize = 1>
using KlTotalLambdaCommImprover = KlImprover<GraphT,
                                             KlHyperTotalCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, windowSize>,
                                             MemoryConstraintT,
                                             windowSize,
                                             DoubleCostT>;

template <typename GraphT, typename MemoryConstraintT = LsLocalMemoryConstraint<GraphT>, unsigned windowSize = 1>
using KlTotalLambdaCommImproverLocalMemConstr
    = KlImprover<GraphT,
                 KlHyperTotalCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, windowSize>,
                 MemoryConstraintT,
                 windowSize,
                 DoubleCostT>;

// --- BSP comm cost / Eager|Lazy|Buffered (resolves to KlImproverScan) ---

template <typename GraphT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          typename CommPolicy = EagerCommCostPolicy,
          unsigned windowSize = 1>
using KlBspCommImprover = KlImprover<GraphT,
                                     KlBspCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, CommPolicy, windowSize>,
                                     MemoryConstraintT,
                                     windowSize,
                                     DoubleCostT>;

template <typename GraphT,
          typename MemoryConstraintT = LsLocalMemoryConstraint<GraphT>,
          typename CommPolicy = EagerCommCostPolicy,
          unsigned windowSize = 1>
using KlBspCommImproverLocalMemConstr
    = KlImprover<GraphT,
                 KlBspCommCostFunction<GraphT, DoubleCostT, MemoryConstraintT, CommPolicy, windowSize>,
                 MemoryConstraintT,
                 windowSize,
                 DoubleCostT>;

}    // namespace osp
