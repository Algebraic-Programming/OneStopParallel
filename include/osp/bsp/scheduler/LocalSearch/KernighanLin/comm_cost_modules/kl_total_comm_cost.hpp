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

template <typename GraphT, typename CostT, typename MemoryConstraintT, unsigned windowSize = 1, bool useNodeCommunicationCostsArg = true>
struct KlTotalCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using kl_move = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    constexpr static bool isMaxCommCostFunction_ = false;

    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool useNodeCommunicationCosts_ = useNodeCommunicationCostsArg || not HasEdgeWeightsV<GraphT>;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule_;

    CompatibleProcessorRange<GraphT> *procRange_;

    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    CostT commMultiplier_ = 1;
    CostT maxCommWeight_ = 0;

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
    }

    struct EmptyStruct {};

    using PreMoveCommDataT = EmptyStruct;

    inline EmptyStruct GetPreMoveCommData(const kl_move &) { return EmptyStruct(); }

    CostT ComputeScheduleCostTest() { return ComputeScheduleCost(); }

    void UpdateDatastructureAfterMove(const kl_move &, const unsigned, const unsigned) {}

    CostT ComputeScheduleCost() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule_->NumSteps(); step++) {
            workCosts += activeSchedule_->GetStepMaxWork(step);
        }

        CostT commCosts = 0;
        for (const auto &edge : Edges(*graph_)) {
            const auto &sourceV = Source(edge, *graph_);
            const auto &targetV = Target(edge, *graph_);

            const unsigned &sourceProc = activeSchedule_->AssignedProcessor(sourceV);
            const unsigned &targetProc = activeSchedule_->AssignedProcessor(targetV);

            if (sourceProc != targetProc) {
                if constexpr (useNodeCommunicationCosts_) {
                    const CostT sourceCommCost = graph_->VertexCommWeight(sourceV);
                    maxCommWeight_ = std::max(maxCommWeight_, sourceCommCost);
                    commCosts += sourceCommCost * instance_->CommunicationCosts(sourceProc, targetProc);
                } else {
                    const CostT sourceCommCost = graph_->EdgeCommWeight(edge);
                    maxCommWeight_ = std::max(maxCommWeight_, sourceCommCost);
                    commCosts += sourceCommCost * instance_->CommunicationCosts(sourceProc, targetProc);
                }
            }
        }

        return workCosts + commCosts * commMultiplier_
               + static_cast<VCommwT<GraphT>>(activeSchedule_->NumSteps() - 1) * instance_->SynchronisationCosts();
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const kl_move &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &maxGainRecompute,
                                std::vector<VertexType> &newNodes) {
        const unsigned &startStep = threadData.startStep_;
        const unsigned &endStep = threadData.endStep_;

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
            auto &affinity_table_target = threadData.affinityTable_.At(target);

            if (move.fromStep_ < targetStep + (move.fromProc_ == targetProc)) {
                const unsigned diff = targetStep - move.fromStep_;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                unsigned idx = targetStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinity_table_target[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.fromProc_)) {
                    affinity_table_target[move.fromProc_][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.fromStep_ - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.fromProc_)) {
                    affinity_table_target[move.fromProc_][idx] += reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinity_table_target[p][idx] += reward;
                    }
                }
            }

            if (move.toStep_ < targetStep + (move.toProc_ == targetProc)) {
                unsigned idx = targetStartIdx;
                const unsigned diff = targetStep - move.toStep_;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinity_table_target[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.toProc_)) {
                    affinity_table_target[move.toProc_][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = move.toStep_ - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.toProc_)) {
                    affinity_table_target[move.toProc_][idx] -= reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinity_table_target[p][idx] -= reward;
                    }
                }
            }

            if (move.toProc_ != move.fromProc_) {
                const auto fromProcTargetCommCost = instance_->CommunicationCosts(move.fromProc_, targetProc);
                const auto toProcTargetCommCost = instance_->CommunicationCosts(move.toProc_, targetProc);

                const CostT commGain = graph_->VertexCommWeight(move.node_) * commMultiplier_;

                unsigned idx = targetStartIdx;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        const auto x
                            = ChangeCommCost(instance_->CommunicationCosts(p, move.toProc_), toProcTargetCommCost, commGain);
                        const auto y
                            = ChangeCommCost(instance_->CommunicationCosts(p, move.fromProc_), fromProcTargetCommCost, commGain);
                        affinity_table_target[p][idx] += x - y;
                    }
                }
            }
        }

        for (const auto &source : instance_->GetComputationalDag().Parents(move.node_)) {
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
            const unsigned windowBound = EndIdx(sourceStep, endStep);
            auto &affinityTableSource = threadData.affinityTable_.At(source);

            if (move.fromStep_ < sourceStep + (move.fromProc_ != sourceProc)) {
                const unsigned diff = sourceStep - move.fromStep_;
                const unsigned bound = windowSize > diff ? windowSize - diff : 0;
                unsigned idx = StartIdx(sourceStep, startStep);
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
                unsigned idx = StartIdx(sourceStep, startStep);
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
                const auto fromProcSourceCommCost = instance_->CommunicationCosts(sourceProc, move.fromProc_);
                const auto toProcSourceCommCost = instance_->CommunicationCosts(sourceProc, move.toProc_);

                const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                unsigned idx = StartIdx(sourceStep, startStep);
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        const CostT x
                            = ChangeCommCost(instance_->CommunicationCosts(p, move.toProc_), toProcSourceCommCost, commGain);
                        const CostT y
                            = ChangeCommCost(instance_->CommunicationCosts(p, move.fromProc_), fromProcSourceCommCost, commGain);
                        affinityTableSource[p][idx] += x - y;
                    }
                }
            }
        }
    }

    inline unsigned StartIdx(const unsigned nodeStep, const unsigned startStep) {
        return (nodeStep < windowSize + startStep) ? windowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep) {
        return (nodeStep + windowSize <= endStep) ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep);
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

            const CostT commGain = graph_->VertexCommWeight(node) * commMultiplier_;
            const auto nodeTargetCommCost = instance_->CommunicationCosts(nodeProc, targetProc);

            for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                const CostT commCost = ChangeCommCost(instance_->CommunicationCosts(p, targetProc), nodeTargetCommCost, commGain);
                for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                    affinityTableNode[p][idx] += commCost;
                }
            }

        }    // traget

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

            const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;
            const auto sourceNodeCommCost = instance_->CommunicationCosts(sourceProc, nodeProc);

            for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                const CostT commCost = ChangeCommCost(instance_->CommunicationCosts(p, sourceProc), sourceNodeCommCost, commGain);
                for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                    affinityTableNode[p][idx] += commCost;
                }
            }
        }    // source
    }
};

}    // namespace osp
