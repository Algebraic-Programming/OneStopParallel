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
#include "lambda_container.hpp"

namespace osp {

template <typename GraphT, typename CostT, typename MemoryConstraintT, unsigned WindowSize = 1>
struct KlHyperTotalCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    constexpr static unsigned windowRange = 2 * WindowSize + 1;
    constexpr static bool isMaxCommCostFunction = false;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule;

    CompatibleProcessorRange<GraphT> *procRange;

    const GraphT *graph;
    const BspInstance<GraphT> *instance;

    CostT commMultiplier = 1;
    CostT maxCommWeight = 0;

    LambdaVectorContainer<VertexType> nodeLambdaMap;

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
        nodeLambdaMap.Initialize(graph->NumVertices(), instance->NumberOfProcessors());
    }

    struct EmptyStruct {};

    using PreMoveCommDataT = EmptyStruct;

    inline EmptyStruct GetPreMoveCommData(const KlMove &) { return EmptyStruct(); }

    CostT ComputeScheduleCost() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule->NumSteps(); step++) {
            workCosts += activeSchedule->GetStepMaxWork(step);
        }

        CostT commCosts = 0;
        for (const auto vertex : graph->Vertices()) {
            const unsigned vertexProc = activeSchedule->AssignedProcessor(vertex);
            const CostT vCommCost = graph->VertexCommWeight(vertex);
            maxCommWeight = std::max(maxCommWeight, vCommCost);

            nodeLambdaMap.ResetNode(vertex);

            for (const auto &target : instance->GetComputationalDag().Children(vertex)) {
                const unsigned targetProc = activeSchedule->AssignedProcessor(target);

                if (nodeLambdaMap.IncreaseProcCount(vertex, targetProc)) {
                    commCosts += vCommCost
                                 * instance->CommunicationCosts(vertexProc, targetProc);    // is 0 if target_proc == vertex_proc
                }
            }
        }

        return workCosts + commCosts * commMultiplier
               + static_cast<VCommwT<GraphT>>(activeSchedule->NumSteps() - 1) * instance->SynchronisationCosts();
    }

    CostT ComputeScheduleCostTest() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule->NumSteps(); step++) {
            workCosts += activeSchedule->GetStepMaxWork(step);
        }

        CostT commCosts = 0;
        for (const auto vertex : graph->Vertices()) {
            const unsigned vertexProc = activeSchedule->AssignedProcessor(vertex);
            const CostT vCommCost = graph->VertexCommWeight(vertex);
            for (const auto lambdaprocMultPair : nodeLambdaMap.IterateProcEntries(vertex)) {
                const auto &lambdaProc = lambdaprocMultPair.first;
                commCosts += vCommCost * instance->CommunicationCosts(vertexProc, lambdaProc);
            }
        }

        return workCosts + commCosts * commMultiplier
               + static_cast<VCommwT<GraphT>>(activeSchedule->NumSteps() - 1) * instance->SynchronisationCosts();
    }

    inline void UpdateDatastructureAfterMove(const KlMove &move, const unsigned startStep, const unsigned endStep) {
        if (move.toProc != move.fromProc) {
            for (const auto &source : instance->GetComputationalDag().Parents(move.node)) {
                const unsigned sourceStep = activeSchedule->AssignedSuperstep(source);
                if (sourceStep < startStep || sourceStep > endStep) {
                    continue;
                }
                UpdateSourceAfterMove(move, source);
            }
        }
    }

    inline void UpdateSourceAfterMove(const KlMove &move, VertexType source) {
        nodeLambdaMap.DecreaseProcCount(source, move.fromProc);
        nodeLambdaMap.IncreaseProcCount(source, move.toProc);
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const KlMove &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &maxGainRecompute,
                                std::vector<VertexType> &newNodes) {
        const unsigned startStep = threadData.startStep;
        const unsigned endStep = threadData.endStep;

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
            auto &affinityTable = threadData.affinityTable.At(target);

            if (move.fromStep < targetStep + (move.fromProc == targetProc)) {
                const unsigned diff = targetStep - move.fromStep;
                const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
                unsigned idx = targetStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.fromProc)) {
                    affinityTable[move.fromProc][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.fromStep - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(WindowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.fromProc)) {
                    affinityTable[move.fromProc][idx] += reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] += reward;
                    }
                }
            }

            if (move.toStep < targetStep + (move.toProc == targetProc)) {
                unsigned idx = targetStartIdx;
                const unsigned diff = targetStep - move.toStep;
                const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.toProc)) {
                    affinityTable[move.toProc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = move.toStep - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(WindowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.toProc)) {
                    affinityTable[move.toProc][idx] -= reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] -= reward;
                    }
                }
            }

            if (move.toProc != move.fromProc) {
                const CostT commGain = graph->VertexCommWeight(move.node) * commMultiplier;

                const unsigned windowBound = EndIdx(targetStep, endStep);
                for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                    if (p == targetProc) {
                        continue;
                    }
                    if (nodeLambdaMap.GetProcEntry(move.node, targetProc) == 1) {
                        for (unsigned idx = targetStartIdx; idx < windowBound; idx++) {
                            const CostT x = instance->CommunicationCosts(move.fromProc, targetProc) * commGain;
                            const CostT y = instance->CommunicationCosts(move.toProc, targetProc) * commGain;
                            affinityTable[p][idx] += x - y;
                        }
                    }

                    if (nodeLambdaMap.HasNoProcEntry(move.node, p)) {
                        for (unsigned idx = targetStartIdx; idx < windowBound; idx++) {
                            const CostT x = instance->CommunicationCosts(move.fromProc, p) * commGain;
                            const CostT y = instance->CommunicationCosts(move.toProc, p) * commGain;
                            affinityTable[p][idx] -= x - y;
                        }
                    }
                }
            }
        }

        for (const auto &source : instance->GetComputationalDag().Parents(move.node)) {
            if (move.toProc != move.fromProc) {
                const unsigned sourceProc = activeSchedule->AssignedProcessor(source);
                if (nodeLambdaMap.HasNoProcEntry(source, move.fromProc)) {
                    const CostT commGain = graph->VertexCommWeight(source) * commMultiplier;

                    for (const auto &target : instance->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule->AssignedSuperstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node)
                            || (not threadData.affinityTable.IsSelected(target)) || threadData.lockManager.IsLocked(target)) {
                            continue;
                        }

                        if (sourceProc != move.fromProc && IsCompatible(target, move.fromProc)) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {    // todo more specialized update
                                maxGainRecompute[target].fullUpdate = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            auto &affinityTableTargetFromProc = threadData.affinityTable.At(target)[move.fromProc];
                            const unsigned targetWindowBound = EndIdx(targetStep, endStep);
                            const CostT commAff = instance->CommunicationCosts(sourceProc, move.fromProc) * commGain;
                            for (unsigned idx = StartIdx(targetStep, startStep); idx < targetWindowBound; idx++) {
                                affinityTableTargetFromProc[idx] += commAff;
                            }
                        }
                    }
                } else if (nodeLambdaMap.GetProcEntry(source, move.fromProc) == 1) {
                    const CostT commGain = graph->VertexCommWeight(source) * commMultiplier;

                    for (const auto &target : instance->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule->AssignedSuperstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node)
                            || threadData.lockManager.IsLocked(target) || (not threadData.affinityTable.IsSelected(target))) {
                            continue;
                        }

                        const unsigned targetProc = activeSchedule->AssignedProcessor(target);
                        if (targetProc == move.fromProc) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {    // todo more specialized update
                                maxGainRecompute[target].fullUpdate = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            const unsigned targetStartIdx = StartIdx(targetStep, startStep);
                            const unsigned targetWindowBound = EndIdx(targetStep, endStep);
                            auto &affinityTableTarget = threadData.affinityTable.At(target);
                            const CostT commAff = instance->CommunicationCosts(sourceProc, targetProc) * commGain;
                            for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                                if (p == targetProc) {
                                    continue;
                                }

                                for (unsigned idx = targetStartIdx; idx < targetWindowBound; idx++) {
                                    affinityTableTarget[p][idx] -= commAff;
                                }
                            }
                            break;    // since node_lambda_map[source][move.from_proc] == 1
                        }
                    }
                }

                if (nodeLambdaMap.GetProcEntry(source, move.toProc) == 1) {
                    const CostT commGain = graph->VertexCommWeight(source) * commMultiplier;

                    for (const auto &target : instance->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule->AssignedSuperstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node)
                            || (not threadData.affinityTable.IsSelected(target)) || threadData.lockManager.IsLocked(target)) {
                            continue;
                        }

                        if (sourceProc != move.toProc && IsCompatible(target, move.toProc)) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
                                maxGainRecompute[target].fullUpdate = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            const unsigned targetWindowBound = EndIdx(targetStep, endStep);
                            auto &affinityTableTargetToProc = threadData.affinityTable.At(target)[move.toProc];
                            const CostT commAff = instance->CommunicationCosts(sourceProc, move.toProc) * commGain;
                            for (unsigned idx = StartIdx(targetStep, startStep); idx < targetWindowBound; idx++) {
                                affinityTableTargetToProc[idx] -= commAff;
                            }
                        }
                    }
                } else if (nodeLambdaMap.GetProcEntry(source, move.toProc) == 2) {
                    for (const auto &target : instance->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule->AssignedSuperstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node)
                            || (not threadData.affinityTable.IsSelected(target)) || threadData.lockManager.IsLocked(target)) {
                            continue;
                        }

                        const unsigned targetProc = activeSchedule->AssignedProcessor(target);
                        if (targetProc == move.toProc) {
                            if (sourceProc != targetProc) {
                                if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
                                    maxGainRecompute[target].fullUpdate = true;
                                } else {
                                    maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                                }

                                const unsigned targetStartIdx = StartIdx(targetStep, startStep);
                                const unsigned targetWindowBound = EndIdx(targetStep, endStep);
                                auto &affinityTableTarget = threadData.affinityTable.At(target);
                                const CostT commAff = instance->CommunicationCosts(sourceProc, targetProc)
                                                      * graph->VertexCommWeight(source) * commMultiplier;
                                for (const unsigned p : procRange->CompatibleProcessorsVertex(target)) {
                                    if (p == targetProc) {
                                        continue;
                                    }

                                    for (unsigned idx = targetStartIdx; idx < targetWindowBound; idx++) {
                                        affinityTableTarget[p][idx] += commAff;
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }

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
            const unsigned sourceStartIdx = StartIdx(sourceStep, startStep);
            const unsigned windowBound = EndIdx(sourceStep, endStep);
            auto &affinityTableSource = threadData.affinityTable.At(source);

            if (move.fromStep < sourceStep + (move.fromProc != sourceProc)) {
                const unsigned diff = sourceStep - move.fromStep;
                const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
                unsigned idx = sourceStartIdx;
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
                unsigned idx = sourceStartIdx;
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
                if (nodeLambdaMap.HasNoProcEntry(source, move.fromProc)) {
                    const CostT commGain = graph->VertexCommWeight(source) * commMultiplier;

                    for (const unsigned p : procRange->CompatibleProcessorsVertex(source)) {
                        if (p == sourceProc) {
                            continue;
                        }

                        const CostT commCost = ChangeCommCost(instance->CommunicationCosts(p, move.fromProc),
                                                              instance->CommunicationCosts(sourceProc, move.fromProc),
                                                              commGain);
                        for (unsigned idx = sourceStartIdx; idx < windowBound; idx++) {
                            affinityTableSource[p][idx] -= commCost;
                        }
                    }
                }

                if (nodeLambdaMap.GetProcEntry(source, move.toProc) == 1) {
                    const CostT commGain = graph->VertexCommWeight(source) * commMultiplier;

                    for (const unsigned p : procRange->CompatibleProcessorsVertex(source)) {
                        if (p == sourceProc) {
                            continue;
                        }

                        const CostT commCost = ChangeCommCost(instance->CommunicationCosts(p, move.toProc),
                                                              instance->CommunicationCosts(sourceProc, move.toProc),
                                                              commGain);
                        for (unsigned idx = sourceStartIdx; idx < windowBound; idx++) {
                            affinityTableSource[p][idx] += commCost;
                        }
                    }
                }
            }
        }
    }

    inline unsigned StartIdx(const unsigned nodeStep, const unsigned startStep) {
        return nodeStep < WindowSize + startStep ? WindowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep) {
        return nodeStep + WindowSize <= endStep ? windowRange : windowRange - (nodeStep + WindowSize - endStep);
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
        }    // traget

        const CostT commGain = graph->VertexCommWeight(node) * commMultiplier;

        for (const unsigned p : procRange->CompatibleProcessorsVertex(node)) {
            if (p == nodeProc) {
                continue;
            }

            for (const auto lambdaPair : nodeLambdaMap.IterateProcEntries(node)) {
                const auto &lambdaProc = lambdaPair.first;
                const CostT commCost = ChangeCommCost(
                    instance->CommunicationCosts(p, lambdaProc), instance->CommunicationCosts(nodeProc, lambdaProc), commGain);
                for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                    affinityTableNode[p][idx] += commCost;
                }
            }
        }

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

            const CostT sourceCommGain = graph->VertexCommWeight(source) * commMultiplier;
            for (const unsigned p : procRange->CompatibleProcessorsVertex(node)) {
                if (p == nodeProc) {
                    continue;
                }

                if (sourceProc != nodeProc && nodeLambdaMap.GetProcEntry(source, nodeProc) == 1) {
                    for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                        affinityTableNode[p][idx] -= instance->CommunicationCosts(sourceProc, nodeProc) * sourceCommGain;
                    }
                }

                if (sourceProc != p && nodeLambdaMap.HasNoProcEntry(source, p)) {
                    for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                        affinityTableNode[p][idx] += instance->CommunicationCosts(sourceProc, p) * sourceCommGain;
                    }
                }
            }
        }    // source
    }
};

}    // namespace osp
