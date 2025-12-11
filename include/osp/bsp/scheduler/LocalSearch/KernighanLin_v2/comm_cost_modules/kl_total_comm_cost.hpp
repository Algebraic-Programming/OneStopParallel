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

#include "../kl_active_schedule.hpp"
#include "../kl_improver.hpp"

namespace osp {

template <typename GraphT, typename CostT, typename MemoryConstraintT, unsigned WindowSize = 1, bool UseNodeCommunicationCostsArg = true>
struct KlTotalCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    constexpr static bool isMaxCommCostFunction = false;

    constexpr static unsigned windowRange = 2 * WindowSize + 1;
    constexpr static bool useNodeCommunicationCosts = UseNodeCommunicationCostsArg || not hasEdgeWeightsV<GraphT>;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule;

    CompatibleProcessorRange<GraphT> *procRange;

    const GraphT *graph;
    const BspInstance<GraphT> *instance;

    CostT commMultiplier = 1;
    CostT maxCommWeight = 0;

    inline CostT GetCommMultiplier() { return commMultiplier; }

    inline CostT GetMaxCommWeight() { return maxCommWeight; }

    inline CostT GetMaxCommWeightMultiplied() { return maxCommWeight * commMultiplier; }

    const std::string Name() const { return "toal_comm_cost"; }

    inline bool IsCompatible(VertexType node, unsigned proc) { return activeSchedule->GetInstance().IsCompatible(node, proc); }

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &pRange) {
        activeSchedule = &sched;
        procRange = &pRange;
        instance = &sched.GetInstance();
        graph = &instance->GetComputationalDag();
        commMultiplier = 1.0 / instance->NumberOfProcessors();
    }

    struct EmptyStruct {};

    using PreMoveCommDataT = EmptyStruct;

    inline EmptyStruct GetPreMoveCommData(const KlMove &) { return EmptyStruct(); }

    CostT ComputeScheduleCostTest() { return ComputeScheduleCost(); }

    void UpdateDatastructureAfterMove(const KlMove &, const unsigned, const unsigned) {}

    CostT ComputeScheduleCost() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule->NumSteps(); step++) {
            workCosts += activeSchedule->GetStepMaxWork(step);
        }

        CostT commCosts = 0;
        for (const auto &edge : Edges(*graph)) {
            const auto &sourceV = Source(edge, *graph);
            const auto &targetV = Target(edge, *graph);

            const unsigned &sourceProc = activeSchedule->AssignedProcessor(sourceV);
            const unsigned &targetProc = activeSchedule->AssignedProcessor(targetV);

            if (sourceProc != targetProc) {
                if constexpr (useNodeCommunicationCosts) {
                    const CostT sourceCommCost = graph->VertexCommWeight(sourceV);
                    maxCommWeight = std::max(maxCommWeight, sourceCommCost);
                    commCosts += sourceCommCost * instance->CommunicationCosts(sourceProc, targetProc);
                } else {
                    const CostT sourceCommCost = graph->edge_comm_weight(edge);
                    maxCommWeight = std::max(maxCommWeight, sourceCommCost);
                    commCosts += sourceCommCost * instance->communicationCosts(sourceProc, targetProc);
                }
            }
        }

        return workCosts + commCosts * commMultiplier
               + static_cast<VCommwT<GraphT>>(activeSchedule->NumSteps() - 1) * instance->SynchronisationCosts();
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const KlMove &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &maxGainRecompute,
                                std::vector<VertexType> &newNodes) {
        const unsigned &startStep = threadData.startStep;
        const unsigned &endStep = threadData.endStep;

        for (const auto &target : instance->GetComputationalDag().Children(move.node)) {
            const unsigned targetStep = activeSchedule->AssignedSuperstep(target);
            if (targetStep < startStep || targetStep > endStep) {
                continue;
            }

            if (threadData.lockManager.IsLocked(target)) {
                continue;
            }

            if (not threadData.affinityTable.IsSelected(target)) {
                newNodes.push_back(target);
                continue;
            }

            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
                maxGainRecompute[target].fullUpdate = true;
            } else {
                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
            }

            const unsigned targetProc = activeSchedule->AssignedProcessor(target);
            const unsigned targetStartIdx = StartIdx(targetStep, startStep);
            auto &affinityTableTarget = threadData.affinityTable.At(target);

            if (move.fromStep < targetStep + (move.fromProc == targetProc)) {
                const unsigned diff = targetStep - move.fromStep;
                const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
                unsigned idx = targetStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        affinityTableTarget[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.fromProc)) {
                    affinityTableTarget[move.fromProc][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.fromStep - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(WindowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.fromProc)) {
                    affinityTableTarget[move.fromProc][idx] += reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        affinityTableTarget[p][idx] += reward;
                    }
                }
            }

            if (move.toStep < targetStep + (move.toProc == targetProc)) {
                unsigned idx = targetStartIdx;
                const unsigned diff = targetStep - move.toStep;
                const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        affinityTableTarget[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.toProc)) {
                    affinityTableTarget[move.toProc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = move.toStep - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(WindowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.toProc)) {
                    affinityTableTarget[move.toProc][idx] -= reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        affinityTableTarget[p][idx] -= reward;
                    }
                }
            }

            if (move.toProc != move.fromProc) {
                const auto fromProcTargetCommCost = instance->CommunicationCosts(move.fromProc, targetProc);
                const auto toProcTargetCommCost = instance->CommunicationCosts(move.toProc, targetProc);

                const CostT commGain = graph->VertexCommWeight(move.node) * commMultiplier;

                unsigned idx = targetStartIdx;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        const auto x = ChangeCommCost(instance->CommunicationCosts(p, move.toProc), toProcTargetCommCost, commGain);
                        const auto y
                            = ChangeCommCost(instance->CommunicationCosts(p, move.fromProc), fromProcTargetCommCost, commGain);
                        affinityTableTarget[p][idx] += x - y;
                    }
                }
            }
        }

        for (const auto &source : instance->GetComputationalDag().Parents(move.node)) {
            const unsigned sourceStep = activeSchedule->AssignedSuperstep(source);
            if (sourceStep < startStep || sourceStep > endStep) {
                continue;
            }

            if (threadData.lockManager.IsLocked(source)) {
                continue;
            }

            if (not threadData.affinityTable.IsSelected(source)) {
                newNodes.push_back(source);
                continue;
            }

            if (maxGainRecompute.find(source) != maxGainRecompute.end()) {
                maxGainRecompute[source].fullUpdate = true;
            } else {
                maxGainRecompute[source] = KlGainUpdateInfo(source, true);
            }

            const unsigned sourceProc = activeSchedule->AssignedProcessor(source);
            const unsigned windowBound = EndIdx(sourceStep, endStep);
            auto &affinityTableSource = threadData.affinityTable.At(source);

            if (move.fromStep < sourceStep + (move.fromProc != sourceProc)) {
                const unsigned diff = sourceStep - move.fromStep;
                const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
                unsigned idx = StartIdx(sourceStep, startStep);
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] += reward;
                    }
                }

                if (WindowSize >= diff && IsCompatible(source, move.fromProc)) {
                    affinityTableSource[move.fromProc][idx] += reward;
                }

            } else {
                const unsigned diff = move.fromStep - sourceStep;
                unsigned idx = WindowSize + diff;

                if (idx < windowBound && IsCompatible(source, move.fromProc)) {
                    affinityTableSource[move.fromProc][idx] += penalty;
                }

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] -= penalty;
                    }
                }
            }

            if (move.toStep < sourceStep + (move.toProc != sourceProc)) {
                const unsigned diff = sourceStep - move.toStep;
                const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
                unsigned idx = StartIdx(sourceStep, startStep);
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] -= reward;
                    }
                }

                if (WindowSize >= diff && IsCompatible(source, move.toProc)) {
                    affinityTableSource[move.toProc][idx] -= reward;
                }

            } else {
                const unsigned diff = move.toStep - sourceStep;
                unsigned idx = WindowSize + diff;

                if (idx < windowBound && IsCompatible(source, move.toProc)) {
                    affinityTableSource[move.toProc][idx] -= penalty;
                }
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] += penalty;
                    }
                }
            }

            if (move.toProc != move.fromProc) {
                const auto fromProcSourceCommCost = instance->CommunicationCosts(sourceProc, move.fromProc);
                const auto toProcSourceCommCost = instance->CommunicationCosts(sourceProc, move.toProc);

                const CostT commGain = graph->VertexCommWeight(source) * commMultiplier;

                unsigned idx = StartIdx(sourceStep, startStep);
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(source)) {
                        const CostT x
                            = ChangeCommCost(instance->CommunicationCosts(p, move.toProc), toProcSourceCommCost, commGain);
                        const CostT y
                            = ChangeCommCost(instance->CommunicationCosts(p, move.fromProc), fromProcSourceCommCost, commGain);
                        affinityTableSource[p][idx] += x - y;
                    }
                }
            }
        }
    }

    inline unsigned StartIdx(const unsigned nodeStep, const unsigned startStep) {
        return (nodeStep < WindowSize + startStep) ? WindowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep) {
        return (nodeStep + WindowSize <= endStep) ? windowRange : windowRange - (nodeStep + WindowSize - endStep);
    }

    inline CostT ChangeCommCost(const VCommwT<GraphT> &pTargetCommCost,
                                const VCommwT<GraphT> &nodeTargetCommCost,
                                const CostT &commGain) {
        return pTargetCommCost > nodeTargetCommCost ? (pTargetCommCost - nodeTargetCommCost) * commGain
                                                    : (nodeTargetCommCost - pTargetCommCost) * commGain * -1.0;
    }

    template <typename AffinityTableT>
    void ComputeCommAffinity(VertexType node,
                             AffinityTableT &affinityTableNode,
                             const CostT &penalty,
                             const CostT &reward,
                             const unsigned startStep,
                             const unsigned endStep) {
        const unsigned nodeStep = activeSchedule->AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule->AssignedProcessor(node);
        const unsigned windowBound = EndIdx(nodeStep, endStep);
        const unsigned nodeStartIdx = StartIdx(nodeStep, startStep);

        for (const auto &target : instance->GetComputationalDag().Children(node)) {
            const unsigned targetStep = activeSchedule->AssignedSuperstep(target);
            const unsigned targetProc = activeSchedule->AssignedProcessor(target);

            if (targetStep < nodeStep + (targetProc != nodeProc)) {
                const unsigned diff = nodeStep - targetStep;
                const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
                unsigned idx = nodeStartIdx;

                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }

                if (WindowSize >= diff && IsCompatible(node, targetProc)) {
                    affinityTableNode[targetProc][idx] -= reward;
                }

            } else {
                const unsigned diff = targetStep - nodeStep;
                unsigned idx = WindowSize + diff;

                if (idx < windowBound && IsCompatible(node, targetProc)) {
                    affinityTableNode[targetProc][idx] -= penalty;
                }

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
                }
            }

            const CostT commGain = graph->VertexCommWeight(node) * commMultiplier;
            const auto nodeTargetCommCost = instance->CommunicationCosts(nodeProc, targetProc);

            for (const unsigned p : procRange->CompatibleProcessorsVertex(node)) {
                const CostT commCost = ChangeCommCost(instance->CommunicationCosts(p, targetProc), nodeTargetCommCost, commGain);
                for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                    affinityTableNode[p][idx] += commCost;
                }
            }

        }    // traget

        for (const auto &source : instance->GetComputationalDag().Parents(node)) {
            const unsigned sourceStep = activeSchedule->AssignedSuperstep(source);
            const unsigned sourceProc = activeSchedule->AssignedProcessor(source);

            if (sourceStep < nodeStep + (sourceProc == nodeProc)) {
                const unsigned diff = nodeStep - sourceStep;
                const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
                unsigned idx = nodeStartIdx;

                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(node, sourceProc)) {
                    affinityTableNode[sourceProc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = sourceStep - nodeStep;
                unsigned idx = std::min(WindowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(node, sourceProc)) {
                    affinityTableNode[sourceProc][idx] -= reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }
            }

            const CostT commGain = graph->VertexCommWeight(source) * commMultiplier;
            const auto sourceNodeCommCost = instance->CommunicationCosts(sourceProc, nodeProc);

            for (const unsigned p : procRange->CompatibleProcessorsVertex(node)) {
                const CostT commCost = ChangeCommCost(instance->CommunicationCosts(p, sourceProc), sourceNodeCommCost, commGain);
                for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                    affinityTableNode[p][idx] += commCost;
                }
            }
        }    // source
    }
};

}    // namespace osp
