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

    inline bool IsCompatible(VertexType node, unsigned proc) { return activeSchedule_->GetInstance().isCompatible(node, proc); }

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
        for (unsigned step = 0; step < activeSchedule_->num_steps(); step++) {
            workCosts += activeSchedule_->get_step_max_work(step);
        }

        CostT commCosts = 0;
        for (const auto &edge : Edges(*graph_)) {
            const auto &sourceV = Source(edge, *graph_);
            const auto &targetV = Target(edge, *graph_);

            const unsigned &sourceProc = activeSchedule_->assigned_processor(sourceV);
            const unsigned &targetProc = activeSchedule_->assigned_processor(targetV);

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
               + static_cast<VCommwT<GraphT>>(activeSchedule_->num_steps() - 1) * instance_->SynchronisationCosts();
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const kl_move &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &maxGainRecompute,
                                std::vector<VertexType> &newNodes) {
        const unsigned &startStep = threadData.startStep;
        const unsigned &endStep = threadData.endStep;

        for (const auto &target : instance_->GetComputationalDag().Children(move.node)) {
            const unsigned targetStep = activeSchedule_->assigned_superstep(target);
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
                maxGainRecompute[target].full_update = true;
            } else {
                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
            }

            const unsigned targetProc = activeSchedule_->assigned_processor(target);
            const unsigned targetStartIdx = StartIdx(targetStep, startStep);
            auto &affinity_table_target = threadData.affinityTable.at(target);

            if (move.from_step < targetStep + (move.from_proc == targetProc)) {
                const unsigned diff = targetStep - move.from_step;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                unsigned idx = targetStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinity_table_target[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.from_proc)) {
                    affinity_table_target[move.from_proc][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.from_step - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.from_proc)) {
                    affinity_table_target[move.from_proc][idx] += reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinity_table_target[p][idx] += reward;
                    }
                }
            }

            if (move.to_step < targetStep + (move.to_proc == targetProc)) {
                unsigned idx = targetStartIdx;
                const unsigned diff = targetStep - move.to_step;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinity_table_target[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.to_proc)) {
                    affinity_table_target[move.to_proc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = move.to_step - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && IsCompatible(target, move.to_proc)) {
                    affinity_table_target[move.to_proc][idx] -= reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinity_table_target[p][idx] -= reward;
                    }
                }
            }

            if (move.to_proc != move.from_proc) {
                const auto fromProcTargetCommCost = instance_->CommunicationCosts(move.from_proc, targetProc);
                const auto toProcTargetCommCost = instance_->CommunicationCosts(move.to_proc, targetProc);

                const CostT commGain = graph_->VertexCommWeight(move.node) * commMultiplier_;

                unsigned idx = targetStartIdx;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        const auto x
                            = ChangeCommCost(instance_->CommunicationCosts(p, move.to_proc), toProcTargetCommCost, commGain);
                        const auto y
                            = ChangeCommCost(instance_->CommunicationCosts(p, move.from_proc), fromProcTargetCommCost, commGain);
                        affinity_table_target[p][idx] += x - y;
                    }
                }
            }
        }

        for (const auto &source : instance_->GetComputationalDag().Parents(move.node)) {
            const unsigned sourceStep = activeSchedule_->assigned_superstep(source);
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
                maxGainRecompute[source].full_update = true;
            } else {
                maxGainRecompute[source] = KlGainUpdateInfo(source, true);
            }

            const unsigned sourceProc = activeSchedule_->assigned_processor(source);
            const unsigned windowBound = EndIdx(sourceStep, endStep);
            auto &affinityTableSource = threadData.affinityTable.at(source);

            if (move.from_step < sourceStep + (move.from_proc != sourceProc)) {
                const unsigned diff = sourceStep - move.from_step;
                const unsigned bound = windowSize > diff ? windowSize - diff : 0;
                unsigned idx = StartIdx(sourceStep, startStep);
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] += reward;
                    }
                }

                if (windowSize >= diff && IsCompatible(source, move.from_proc)) {
                    affinityTableSource[move.from_proc][idx] += reward;
                }

            } else {
                const unsigned diff = move.from_step - sourceStep;
                unsigned idx = windowSize + diff;

                if (idx < windowBound && IsCompatible(source, move.from_proc)) {
                    affinityTableSource[move.from_proc][idx] += penalty;
                }

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] -= penalty;
                    }
                }
            }

            if (move.to_step < sourceStep + (move.to_proc != sourceProc)) {
                const unsigned diff = sourceStep - move.to_step;
                const unsigned bound = windowSize > diff ? windowSize - diff : 0;
                unsigned idx = StartIdx(sourceStep, startStep);
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] -= reward;
                    }
                }

                if (windowSize >= diff && IsCompatible(source, move.to_proc)) {
                    affinityTableSource[move.to_proc][idx] -= reward;
                }

            } else {
                const unsigned diff = move.to_step - sourceStep;
                unsigned idx = windowSize + diff;

                if (idx < windowBound && IsCompatible(source, move.to_proc)) {
                    affinityTableSource[move.to_proc][idx] -= penalty;
                }
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] += penalty;
                    }
                }
            }

            if (move.to_proc != move.from_proc) {
                const auto fromProcSourceCommCost = instance_->CommunicationCosts(sourceProc, move.from_proc);
                const auto toProcSourceCommCost = instance_->CommunicationCosts(sourceProc, move.to_proc);

                const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                unsigned idx = StartIdx(sourceStep, startStep);
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        const CostT x
                            = ChangeCommCost(instance_->CommunicationCosts(p, move.to_proc), toProcSourceCommCost, commGain);
                        const CostT y
                            = ChangeCommCost(instance_->CommunicationCosts(p, move.from_proc), fromProcSourceCommCost, commGain);
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
        const unsigned nodeStep = activeSchedule_->assigned_superstep(node);
        const unsigned nodeProc = activeSchedule_->assigned_processor(node);
        const unsigned windowBound = EndIdx(nodeStep, endStep);
        const unsigned nodeStartIdx = StartIdx(nodeStep, startStep);

        for (const auto &target : instance_->GetComputationalDag().Children(node)) {
            const unsigned targetStep = activeSchedule_->assigned_superstep(target);
            const unsigned targetProc = activeSchedule_->assigned_processor(target);

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
            const unsigned sourceStep = activeSchedule_->assigned_superstep(source);
            const unsigned sourceProc = activeSchedule_->assigned_processor(source);

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
