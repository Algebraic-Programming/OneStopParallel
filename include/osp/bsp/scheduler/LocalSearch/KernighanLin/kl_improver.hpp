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

#include "kl_improver_heap.hpp"
#include "kl_improver_scan.hpp"

namespace osp {

// =============================================================================
// FACTORY — selects the right variant based on cost function
//
//   isMaxCommCostFunction_ == true  (BSP max-comm):  KlImproverScan
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

}    // namespace osp
