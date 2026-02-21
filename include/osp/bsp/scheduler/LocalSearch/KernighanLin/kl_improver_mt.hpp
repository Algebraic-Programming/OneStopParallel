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

// Parallel implementations
#include "kl_improver_sync_parallel.hpp"

// Cost function modules (already pulled in by kl_improver.hpp, but
// explicit for readability)
#include "comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "comm_cost_modules/kl_hyper_total_comm_cost.hpp"
#include "comm_cost_modules/kl_max_bsp_comm_cost.hpp"
#include "comm_cost_modules/kl_total_comm_cost.hpp"

// Memory constraint modules
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"

namespace osp {

// =============================================================================
// Synchronized parallel KL improver aliases
//
// Thread-safety of the underlying cost functions:
//
//   SAFE for multi-threaded use (threadRangeGap >= windowSize recommended):
//     - KlTotalCommCostFunction:      UpdateDatastructureAfterMove is no-op.
//                                     No shared mutable comm state.
//     - KlHyperTotalCommCostFunction: Lambda map mutations scoped to
//                                     [startStep, endStep]. Safe if gap
//                                     prevents cross-range neighbor reads.
//
//   UNSAFE for multi-threaded use (use numThreads=1 only):
//     - KlBspCommCostFunction:        ComputeNodeAffinity reads commDs_
//                                     (lambda map, step max) for nodes
//                                     outside the thread's step range.
//                                     Concurrent mutation causes data races.
//     - KlMaxBspCommCostFunction:     Same issue as KlBspCommCostFunction.
//
// For BSP/MaxBSP parallel optimization, the future async improver
// gives each worker its own schedule copy, avoiding shared-state races.
// =============================================================================

using DoubleCostT = double;

// ---------------------------------------------------------------------------
// Total comm cost — SAFE for sync parallel
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Total lambda comm cost / hypergraph-aware — SAFE for sync parallel
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// BSP comm cost — NOT SAFE for sync parallel (shared commDs_ races)
//
// Provided for single-threaded use (SetMaxNumThreads(1)) and for
// future async parallel where each worker has its own schedule copy.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MaxBSP comm cost — NOT SAFE for sync parallel (shared commDs_ races)
//
// Same constraints as BSP above.
// ---------------------------------------------------------------------------

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
