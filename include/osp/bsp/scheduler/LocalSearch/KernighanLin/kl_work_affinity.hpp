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

#include <algorithm>
#include <vector>

#include "kl_active_schedule.hpp"
#include "osp/bsp/model/util/CompatibleProcessorRange.hpp"

namespace osp {

/*
 * Standalone work affinity computation, callable by any cost function.
 * Computes the work-cost change of moving @p node from its current
 * (proc, step) to every candidate (proc, step) in the window, and
 * writes the result into @p affinityTableNode[proc][idx].
 *
 * This is shared by all additive cost models (BSP, Total, HyperTotal).
 * Coupled cost models (MaxBSP) compute work inline and do not call this.
 */
template <unsigned windowSize, typename GraphT, typename CostT, typename MemoryConstraintT, typename AffinityTableT>
void ComputeWorkAffinity(VertexIdxT<GraphT> node,
                         AffinityTableT &affinityTableNode,
                         KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &activeSchedule,
                         const GraphT &graph,
                         CompatibleProcessorRange<GraphT> &procRange,
                         const unsigned startStep,
                         const unsigned endStep) {
    using VertexWorkWeightT = VWorkwT<GraphT>;

    const unsigned nodeStep = activeSchedule.AssignedSuperstep(node);
    const VertexWorkWeightT vertexWeight = graph.VertexWorkWeight(node);

    // Window index helpers (same logic as cost function StartIdx/EndIdx)
    const unsigned nodeStartIdx = (nodeStep < windowSize + startStep) ? windowSize - (nodeStep - startStep) : 0;
    const unsigned windowRange = 2 * windowSize + 1;
    const unsigned nodeEndIdx = (nodeStep + windowSize <= endStep) ? windowRange : windowRange - (nodeStep + windowSize - endStep);

    // Different-step entries
    unsigned step = (nodeStep > windowSize) ? (nodeStep - windowSize) : 0;
    for (unsigned idx = nodeStartIdx; idx < nodeEndIdx; ++idx, ++step) {
        if (idx == windowSize) {
            continue;
        }

        const CostT maxWorkForStep = static_cast<CostT>(activeSchedule.GetStepMaxWork(step));

        for (const unsigned proc : procRange.CompatibleProcessorsVertex(node)) {
            const VertexWorkWeightT newWeight = vertexWeight + activeSchedule.GetStepProcessorWork(step, proc);
            const CostT workDiff = static_cast<CostT>(newWeight) - maxWorkForStep;
            affinityTableNode[proc][idx] = std::max(CostT(0), workDiff);
        }
    }

    // Same-step entry (idx == windowSize)
    const unsigned nodeProc = activeSchedule.AssignedProcessor(node);
    const VertexWorkWeightT maxWorkForStep = activeSchedule.GetStepMaxWork(nodeStep);
    const bool isSoleMaxProcessor = (activeSchedule.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                    && (maxWorkForStep == activeSchedule.GetStepProcessorWork(nodeStep, nodeProc));

    const CostT nodeProcAffinity
        = isSoleMaxProcessor
              ? std::min(static_cast<CostT>(vertexWeight),
                         static_cast<CostT>(maxWorkForStep) - static_cast<CostT>(activeSchedule.GetStepSecondMaxWork(nodeStep)))
              : CostT(0);
    affinityTableNode[nodeProc][windowSize] = nodeProcAffinity;

    const CostT maxWorkAfterRemoval = static_cast<CostT>(maxWorkForStep) - nodeProcAffinity;

    for (const unsigned proc : procRange.CompatibleProcessorsVertex(node)) {
        if (proc == nodeProc) {
            continue;
        }

        const VertexWorkWeightT newWeight = vertexWeight + activeSchedule.GetStepProcessorWork(nodeStep, proc);
        if (static_cast<CostT>(newWeight) > maxWorkAfterRemoval) {
            affinityTableNode[proc][windowSize] = static_cast<CostT>(newWeight) - maxWorkAfterRemoval;
        } else {
            affinityTableNode[proc][windowSize] = CostT(0);
        }
    }
}

}    // namespace osp
