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

template <typename GraphT, typename CostT, typename MemoryConstraintT, unsigned windowSize = 1>
struct KlHyperTotalCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool isMaxCommCostFunction_ = false;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule_;

    CompatibleProcessorRange<GraphT> *procRange_;

    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    CostT commMultiplier_ = 1;
    CostT maxCommWeight_ = 0;

    LambdaVectorContainer<VertexType> nodeLambdaMap_;

    inline CostT GetCommMultiplier() { return commMultiplier_; }

    inline CostT GetMaxCommWeight() { return maxCommWeight_; }

    inline CostT GetMaxCommWeightMultiplied() { return maxCommWeight_ * commMultiplier_; }

    const std::string Name() const { return "toal_comm_cost"; }

    inline bool IsCompatible(VertexType node, unsigned proc) { return activeSchedule_->GetInstance().IsCompatible(node, proc); }

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &pRange) {
        activeSchedule_ = &sched;
        procRange_ = &pRange;
        instance_ = &sched.GetInstance();
        graph_ = &instance_->GetComputationalDag();
        commMultiplier_ = 1.0 / instance_->NumberOfProcessors();
        nodeLambdaMap_.Initialize(graph_->NumVertices(), instance_->NumberOfProcessors());
    }

    struct EmptyStruct {};

    using PreMoveCommDataT = EmptyStruct;

    inline EmptyStruct GetPreMoveCommData(const KlMove &) { return EmptyStruct(); }

    CostT ComputeScheduleCost() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule_->NumSteps(); step++) {
            workCosts += activeSchedule_->GetStepMaxWork(step);
        }

        CostT commCosts = 0;
        for (const auto vertex : graph_->Vertices()) {
            const unsigned vertexProc = activeSchedule_->AssignedProcessor(vertex);
            const CostT vCommCost = graph_->VertexCommWeight(vertex);
            maxCommWeight_ = std::max(maxCommWeight_, vCommCost);

            nodeLambdaMap_.ResetNode(vertex);

            for (const auto &target : instance_->GetComputationalDag().Children(vertex)) {
                const unsigned targetProc = activeSchedule_->AssignedProcessor(target);

                if (nodeLambdaMap_.IncreaseProcCount(vertex, targetProc)) {
                    commCosts += vCommCost
                                 * instance_->CommunicationCosts(vertexProc, targetProc);    // is 0 if targetProc == vertexProc
                }
            }
        }

        return workCosts + commCosts * commMultiplier_
               + static_cast<VCommwT<GraphT>>(activeSchedule_->NumSteps() - 1) * instance_->SynchronisationCosts();
    }

    CostT ComputeScheduleCostTest() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule_->NumSteps(); step++) {
            workCosts += activeSchedule_->GetStepMaxWork(step);
        }

        CostT commCosts = 0;
        for (const auto vertex : graph_->Vertices()) {
            const unsigned vertexProc = activeSchedule_->AssignedProcessor(vertex);
            const CostT vCommCost = graph_->VertexCommWeight(vertex);
            for (const auto lambdaprocMultPair : nodeLambdaMap_.IterateProcEntries(vertex)) {
                const auto &lambdaProc = lambdaprocMultPair.first;
                commCosts += vCommCost * instance_->CommunicationCosts(vertexProc, lambdaProc);
            }
        }

        return workCosts + commCosts * commMultiplier_
               + static_cast<VCommwT<GraphT>>(activeSchedule_->NumSteps() - 1) * instance_->SynchronisationCosts();
    }

    inline void UpdateDatastructureAfterMove(const KlMove &move, const unsigned startStep, const unsigned endStep) {
        if (move.toProc_ != move.fromProc_) {
            for (const auto &source : instance_->GetComputationalDag().Parents(move.node_)) {
                const unsigned sourceStep = activeSchedule_->AssignedSuperstep(source);
                if (sourceStep < startStep || sourceStep > endStep) {
                    continue;
                }
                UpdateSourceAfterMove(move, source);
            }
        }
    }

    inline void UpdateSourceAfterMove(const KlMove &move, VertexType source) {
        nodeLambdaMap_.DecreaseProcCount(source, move.fromProc_);
        nodeLambdaMap_.IncreaseProcCount(source, move.toProc_);
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const KlMove &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &maxGainRecompute,
                                std::vector<VertexType> &newNodes) {
        const unsigned startStep = threadData.startStep_;
        const unsigned endStep = threadData.endStep_;

        for (const auto &target : instance_->GetComputationalDag().Children(move.node_)) {
            const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
            if (targetStep < startStep || targetStep > endStep) {
                continue;
            }

            if (threadData.lockManager_.IsLocked(target)) {
                continue;
            }

            if (not threadData.affinityTable_.IsSelected(target)) {
                newNodes.push_back(target);
                continue;
            }

            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
                maxGainRecompute[target].fullUpdate_ = true;
            } else {
                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
            }

            const unsigned targetProc = activeSchedule_->AssignedProcessor(target);
            const unsigned targetStartIdx = StartIdx(targetStep, startStep);
            auto &affinityTable = threadData.affinityTable_.At(target);

            if (move.fromStep_ < targetStep + (move.fromProc_ == targetProc)) {
                const unsigned diff = targetStep - move.fromStep_;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                unsigned idx = targetStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.fromProc_)) {
                    affinityTable[move.fromProc_][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.fromStep_ - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.fromProc_)) {
                    affinityTable[move.fromProc_][idx] += reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] += reward;
                    }
                }
            }

            if (move.toStep_ < targetStep + (move.toProc_ == targetProc)) {
                unsigned idx = targetStartIdx;
                const unsigned diff = targetStep - move.toStep_;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.toProc_)) {
                    affinityTable[move.toProc_][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = move.toStep_ - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.toProc_)) {
                    affinityTable[move.toProc_][idx] -= reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] -= reward;
                    }
                }
            }

            if (move.toProc_ != move.fromProc_) {
                const CostT commGain = graph_->VertexCommWeight(move.node_) * commMultiplier_;

                const unsigned windowBound = EndIdx(targetStep, endStep);
                for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                    if (p == targetProc) {
                        continue;
                    }
                    if (nodeLambdaMap_.GetProcEntry(move.node_, targetProc) == 1) {
                        for (unsigned idx = targetStartIdx; idx < windowBound; idx++) {
                            const CostT x = instance_->CommunicationCosts(move.fromProc_, targetProc) * commGain;
                            const CostT y = instance_->CommunicationCosts(move.toProc_, targetProc) * commGain;
                            affinityTable[p][idx] += x - y;
                        }
                    }

                    if (nodeLambdaMap_.HasNoProcEntry(move.node_, p)) {
                        for (unsigned idx = targetStartIdx; idx < windowBound; idx++) {
                            const CostT x = instance_->CommunicationCosts(move.fromProc_, p) * commGain;
                            const CostT y = instance_->CommunicationCosts(move.toProc_, p) * commGain;
                            affinityTable[p][idx] -= x - y;
                        }
                    }
                }
            }
        }

        for (const auto &source : instance_->GetComputationalDag().Parents(move.node_)) {
            if (move.toProc_ != move.fromProc_) {
                const unsigned sourceProc = activeSchedule_->AssignedProcessor(source);
                if (nodeLambdaMap_.HasNoProcEntry(source, move.fromProc_)) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const auto &target : instance_->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node_)
                            || (not threadData.affinityTable_.IsSelected(target)) || threadData.lockManager_.IsLocked(target)) {
                            continue;
                        }

                        if (sourceProc != move.fromProc_ && IsCompatible(target, move.fromProc_)) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {    // todo more specialized update
                                maxGainRecompute[target].fullUpdate_ = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            auto &affinityTableTargetFromProc = threadData.affinityTable_.At(target)[move.fromProc_];
                            const unsigned targetWindowBound = EndIdx(targetStep, endStep);
                            const CostT commAff = instance_->CommunicationCosts(sourceProc, move.fromProc_) * commGain;
                            for (unsigned idx = StartIdx(targetStep, startStep); idx < targetWindowBound; idx++) {
                                affinityTableTargetFromProc[idx] += commAff;
                            }
                        }
                    }
                } else if (nodeLambdaMap_.GetProcEntry(source, move.fromProc_) == 1) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const auto &target : instance_->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node_)
                            || threadData.lockManager_.IsLocked(target) || (not threadData.affinityTable_.IsSelected(target))) {
                            continue;
                        }

                        const unsigned targetProc = activeSchedule_->AssignedProcessor(target);
                        if (targetProc == move.fromProc_) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {    // todo more specialized update
                                maxGainRecompute[target].fullUpdate_ = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            const unsigned targetStartIdx = StartIdx(targetStep, startStep);
                            const unsigned targetWindowBound = EndIdx(targetStep, endStep);
                            auto &affinityTableTarget = threadData.affinityTable_.At(target);
                            const CostT commAff = instance_->CommunicationCosts(sourceProc, targetProc) * commGain;
                            for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                                if (p == targetProc) {
                                    continue;
                                }

                                for (unsigned idx = targetStartIdx; idx < targetWindowBound; idx++) {
                                    affinityTableTarget[p][idx] -= commAff;
                                }
                            }
                            break;    // since nodeLambdaMap_[source][move.fromProc_] == 1
                        }
                    }
                }

                if (nodeLambdaMap_.GetProcEntry(source, move.toProc_) == 1) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const auto &target : instance_->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node_)
                            || (not threadData.affinityTable_.IsSelected(target)) || threadData.lockManager_.IsLocked(target)) {
                            continue;
                        }

                        if (sourceProc != move.toProc_ && IsCompatible(target, move.toProc_)) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
                                maxGainRecompute[target].fullUpdate_ = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            const unsigned targetWindowBound = EndIdx(targetStep, endStep);
                            auto &affinityTableTargetToProc = threadData.affinityTable_.At(target)[move.toProc_];
                            const CostT commAff = instance_->CommunicationCosts(sourceProc, move.toProc_) * commGain;
                            for (unsigned idx = StartIdx(targetStep, startStep); idx < targetWindowBound; idx++) {
                                affinityTableTargetToProc[idx] -= commAff;
                            }
                        }
                    }
                } else if (nodeLambdaMap_.GetProcEntry(source, move.toProc_) == 2) {
                    for (const auto &target : instance_->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node_)
                            || (not threadData.affinityTable_.IsSelected(target)) || threadData.lockManager_.IsLocked(target)) {
                            continue;
                        }

                        const unsigned targetProc = activeSchedule_->AssignedProcessor(target);
                        if (targetProc == move.toProc_) {
                            if (sourceProc != targetProc) {
                                if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
                                    maxGainRecompute[target].fullUpdate_ = true;
                                } else {
                                    maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                                }

                                const unsigned targetStartIdx = StartIdx(targetStep, startStep);
                                const unsigned targetWindowBound = EndIdx(targetStep, endStep);
                                auto &affinityTableTarget = threadData.affinityTable_.At(target);
                                const CostT commAff = instance_->CommunicationCosts(sourceProc, targetProc)
                                                      * graph_->VertexCommWeight(source) * commMultiplier_;
                                for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
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

            const unsigned sourceStep = activeSchedule_->AssignedSuperstep(source);
            if (sourceStep < startStep || sourceStep > endStep) {
                continue;
            }

            if (threadData.lockManager_.IsLocked(source)) {
                continue;
            }

            if (not threadData.affinityTable_.IsSelected(source)) {
                newNodes.push_back(source);
                continue;
            }

            if (maxGainRecompute.find(source) != maxGainRecompute.end()) {
                maxGainRecompute[source].fullUpdate_ = true;
            } else {
                maxGainRecompute[source] = KlGainUpdateInfo(source, true);
            }

            const unsigned sourceProc = activeSchedule_->AssignedProcessor(source);
            const unsigned sourceStartIdx = StartIdx(sourceStep, startStep);
            const unsigned windowBound = EndIdx(sourceStep, endStep);
            auto &affinityTableSource = threadData.affinityTable_.At(source);

            if (move.fromStep_ < sourceStep + (move.fromProc_ != sourceProc)) {
                const unsigned diff = sourceStep - move.fromStep_;
                const unsigned bound = windowSize > diff ? windowSize - diff : 0;
                unsigned idx = sourceStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] += reward;
                    }
                }

                if (windowSize >= diff && IsCompatible(source, move.fromProc_)) {
                    affinityTableSource[move.fromProc_][idx] += reward;
                }

            } else {
                const unsigned diff = move.fromStep_ - sourceStep;
                unsigned idx = windowSize + diff;

                if (idx < windowBound && IsCompatible(source, move.fromProc_)) {
                    affinityTableSource[move.fromProc_][idx] += penalty;
                }

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] -= penalty;
                    }
                }
            }

            if (move.toStep_ < sourceStep + (move.toProc_ != sourceProc)) {
                const unsigned diff = sourceStep - move.toStep_;
                const unsigned bound = windowSize > diff ? windowSize - diff : 0;
                unsigned idx = sourceStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] -= reward;
                    }
                }

                if (windowSize >= diff && IsCompatible(source, move.toProc_)) {
                    affinityTableSource[move.toProc_][idx] -= reward;
                }

            } else {
                const unsigned diff = move.toStep_ - sourceStep;
                unsigned idx = windowSize + diff;

                if (idx < windowBound && IsCompatible(source, move.toProc_)) {
                    affinityTableSource[move.toProc_][idx] -= penalty;
                }
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] += penalty;
                    }
                }
            }

            if (move.toProc_ != move.fromProc_) {
                if (nodeLambdaMap_.HasNoProcEntry(source, move.fromProc_)) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        if (p == sourceProc) {
                            continue;
                        }

                        const CostT commCost = ChangeCommCost(instance_->CommunicationCosts(p, move.fromProc_),
                                                              instance_->CommunicationCosts(sourceProc, move.fromProc_),
                                                              commGain);
                        for (unsigned idx = sourceStartIdx; idx < windowBound; idx++) {
                            affinityTableSource[p][idx] -= commCost;
                        }
                    }
                }

                if (nodeLambdaMap_.GetProcEntry(source, move.toProc_) == 1) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        if (p == sourceProc) {
                            continue;
                        }

                        const CostT commCost = ChangeCommCost(instance_->CommunicationCosts(p, move.toProc_),
                                                              instance_->CommunicationCosts(sourceProc, move.toProc_),
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
        return nodeStep < windowSize + startStep ? windowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep) {
        return nodeStep + windowSize <= endStep ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep);
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
        const unsigned nodeStep = activeSchedule_->AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule_->AssignedProcessor(node);
        const unsigned windowBound = EndIdx(nodeStep, endStep);
        const unsigned nodeStartIdx = StartIdx(nodeStep, startStep);

        for (const auto &target : instance_->GetComputationalDag().Children(node)) {
            const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
            const unsigned targetProc = activeSchedule_->AssignedProcessor(target);

            if (targetStep < nodeStep + (targetProc != nodeProc)) {
                const unsigned diff = nodeStep - targetStep;
                const unsigned bound = windowSize > diff ? windowSize - diff : 0;
                unsigned idx = nodeStartIdx;

                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }

                if (windowSize >= diff && IsCompatible(node, targetProc)) {
                    affinityTableNode[targetProc][idx] -= reward;
                }

            } else {
                const unsigned diff = targetStep - nodeStep;
                unsigned idx = windowSize + diff;

                if (idx < windowBound && IsCompatible(node, targetProc)) {
                    affinityTableNode[targetProc][idx] -= penalty;
                }

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
                }
            }
        }    // traget

        const CostT commGain = graph_->VertexCommWeight(node) * commMultiplier_;

        for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
            if (p == nodeProc) {
                continue;
            }

            for (const auto lambdaPair : nodeLambdaMap_.IterateProcEntries(node)) {
                const auto &lambdaProc = lambdaPair.first;
                const CostT commCost = ChangeCommCost(
                    instance_->CommunicationCosts(p, lambdaProc), instance_->CommunicationCosts(nodeProc, lambdaProc), commGain);
                for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                    affinityTableNode[p][idx] += commCost;
                }
            }
        }

        for (const auto &source : instance_->GetComputationalDag().Parents(node)) {
            const unsigned sourceStep = activeSchedule_->AssignedSuperstep(source);
            const unsigned sourceProc = activeSchedule_->AssignedProcessor(source);

            if (sourceStep < nodeStep + (sourceProc == nodeProc)) {
                const unsigned diff = nodeStep - sourceStep;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                unsigned idx = nodeStartIdx;

                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(node, sourceProc)) {
                    affinityTableNode[sourceProc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = sourceStep - nodeStep;
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(node, sourceProc)) {
                    affinityTableNode[sourceProc][idx] -= reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }
            }

            const CostT sourceCommGain = graph_->VertexCommWeight(source) * commMultiplier_;
            for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                if (p == nodeProc) {
                    continue;
                }

                if (sourceProc != nodeProc && nodeLambdaMap_.GetProcEntry(source, nodeProc) == 1) {
                    for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                        affinityTableNode[p][idx] -= instance_->CommunicationCosts(sourceProc, nodeProc) * sourceCommGain;
                    }
                }

                if (sourceProc != p && nodeLambdaMap_.HasNoProcEntry(source, p)) {
                    for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                        affinityTableNode[p][idx] += instance_->CommunicationCosts(sourceProc, p) * sourceCommGain;
                    }
                }
            }
        }    // source
    }
};

}    // namespace osp
