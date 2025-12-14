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
    using kl_move = KlMoveStruct<CostT, VertexType>;
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

    inline bool IsCompatible(VertexType node, unsigned proc) { return activeSchedule_->GetInstance().isCompatible(node, proc); }

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &pRange) {
        activeSchedule_ = &sched;
        procRange_ = &pRange;
        instance_ = &sched.GetInstance();
        graph_ = &instance_->GetComputationalDag();
        commMultiplier_ = 1.0 / instance_->NumberOfProcessors();
        nodeLambdaMap_.initialize(graph_->NumVertices(), instance_->NumberOfProcessors());
    }

    struct EmptyStruct {};

    using PreMoveCommDataT = EmptyStruct;

    inline EmptyStruct GetPreMoveCommData(const kl_move &) { return EmptyStruct(); }

    CostT ComputeScheduleCost() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule_->num_steps(); step++) {
            workCosts += activeSchedule_->get_step_max_work(step);
        }

        CostT commCosts = 0;
        for (const auto vertex : graph_->Vertices()) {
            const unsigned vertexProc = activeSchedule_->assigned_processor(vertex);
            const CostT vCommCost = graph_->VertexCommWeight(vertex);
            maxCommWeight_ = std::max(maxCommWeight_, vCommCost);

            nodeLambdaMap_.reset_node(vertex);

            for (const auto &target : instance_->GetComputationalDag().Children(vertex)) {
                const unsigned targetProc = activeSchedule_->assigned_processor(target);

                if (nodeLambdaMap_.IncreaseProcCount(vertex, targetProc)) {
                    commCosts += vCommCost
                                 * instance_->CommunicationCosts(vertexProc, targetProc);    // is 0 if targetProc == vertexProc
                }
            }
        }

        return workCosts + commCosts * commMultiplier_
               + static_cast<VCommwT<GraphT>>(activeSchedule_->num_steps() - 1) * instance_->SynchronisationCosts();
    }

    CostT ComputeScheduleCostTest() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule_->num_steps(); step++) {
            workCosts += activeSchedule_->get_step_max_work(step);
        }

        CostT commCosts = 0;
        for (const auto vertex : graph_->Vertices()) {
            const unsigned vertexProc = activeSchedule_->assigned_processor(vertex);
            const CostT vCommCost = graph_->VertexCommWeight(vertex);
            for (const auto lambdaproc_mult_pair : nodeLambdaMap_.iterate_proc_entries(vertex)) {
                const auto &lambdaProc = lambdaproc_mult_pair.first;
                commCosts += vCommCost * instance_->CommunicationCosts(vertexProc, lambdaProc);
            }
        }

        return workCosts + commCosts * commMultiplier_
               + static_cast<VCommwT<GraphT>>(activeSchedule_->num_steps() - 1) * instance_->SynchronisationCosts();
    }

    inline void UpdateDatastructureAfterMove(const kl_move &move, const unsigned startStep, const unsigned endStep) {
        if (move.to_proc != move.from_proc) {
            for (const auto &source : instance_->GetComputationalDag().Parents(move.node)) {
                const unsigned sourceStep = activeSchedule_->assigned_superstep(source);
                if (sourceStep < startStep || sourceStep > endStep) {
                    continue;
                }
                UpdateSourceAfterMove(move, source);
            }
        }
    }

    inline void UpdateSourceAfterMove(const kl_move &move, VertexType source) {
        nodeLambdaMap_.DecreaseProcCount(source, move.from_proc);
        nodeLambdaMap_.IncreaseProcCount(source, move.to_proc);
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const kl_move &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &maxGainRecompute,
                                std::vector<VertexType> &newNodes) {
        const unsigned startStep = threadData.startStep;
        const unsigned endStep = threadData.endStep;

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
            auto &affinityTable = threadData.affinityTable.at(target);

            if (move.from_step < targetStep + (move.from_proc == targetProc)) {
                const unsigned diff = targetStep - move.from_step;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                unsigned idx = targetStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && is_compatible(target, move.from_proc)) {
                    affinityTable[move.from_proc][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.from_step - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && is_compatible(target, move.from_proc)) {
                    affinityTable[move.from_proc][idx] += reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] += reward;
                    }
                }
            }

            if (move.to_step < targetStep + (move.to_proc == targetProc)) {
                unsigned idx = targetStartIdx;
                const unsigned diff = targetStep - move.to_step;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && is_compatible(target, move.to_proc)) {
                    affinityTable[move.to_proc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = move.to_step - targetStep;
                const unsigned windowBound = EndIdx(targetStep, endStep);
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && is_compatible(target, move.to_proc)) {
                    affinityTable[move.to_proc][idx] -= reward;
                }

                idx++;

                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                        affinityTable[p][idx] -= reward;
                    }
                }
            }

            if (move.to_proc != move.from_proc) {
                const CostT commGain = graph_->VertexCommWeight(move.node) * commMultiplier_;

                const unsigned windowBound = EndIdx(targetStep, endStep);
                for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                    if (p == targetProc) {
                        continue;
                    }
                    if (nodeLambdaMap_.get_proc_entry(move.node, targetProc) == 1) {
                        for (unsigned idx = targetStartIdx; idx < windowBound; idx++) {
                            const CostT x = instance_->CommunicationCosts(move.from_proc, targetProc) * commGain;
                            const CostT y = instance_->CommunicationCosts(move.to_proc, targetProc) * commGain;
                            affinityTable[p][idx] += x - y;
                        }
                    }

                    if (nodeLambdaMap_.has_no_proc_entry(move.node, p)) {
                        for (unsigned idx = targetStartIdx; idx < windowBound; idx++) {
                            const CostT x = instance_->CommunicationCosts(move.from_proc, p) * commGain;
                            const CostT y = instance_->CommunicationCosts(move.to_proc, p) * commGain;
                            affinityTable[p][idx] -= x - y;
                        }
                    }
                }
            }
        }

        for (const auto &source : instance_->GetComputationalDag().Parents(move.node)) {
            if (move.to_proc != move.from_proc) {
                const unsigned sourceProc = activeSchedule_->assigned_processor(source);
                if (nodeLambdaMap_.has_no_proc_entry(source, move.from_proc)) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const auto &target : instance_->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule_->assigned_superstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node)
                            || (not threadData.affinityTable.IsSelected(target)) || threadData.lockManager.IsLocked(target)) {
                            continue;
                        }

                        if (sourceProc != move.from_proc && is_compatible(target, move.from_proc)) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {    // todo more specialized update
                                maxGainRecompute[target].full_update = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            auto &affinity_table_target_from_proc = threadData.affinityTable.at(target)[move.from_proc];
                            const unsigned target_window_bound = EndIdx(targetStep, endStep);
                            const CostT comm_aff = instance_->CommunicationCosts(sourceProc, move.from_proc) * commGain;
                            for (unsigned idx = StartIdx(targetStep, startStep); idx < target_window_bound; idx++) {
                                affinity_table_target_from_proc[idx] += comm_aff;
                            }
                        }
                    }
                } else if (nodeLambdaMap_.get_proc_entry(source, move.from_proc) == 1) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const auto &target : instance_->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule_->assigned_superstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node)
                            || threadData.lockManager.IsLocked(target) || (not threadData.affinityTable.IsSelected(target))) {
                            continue;
                        }

                        const unsigned targetProc = activeSchedule_->assigned_processor(target);
                        if (targetProc == move.from_proc) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {    // todo more specialized update
                                maxGainRecompute[target].full_update = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            const unsigned targetStartIdx = StartIdx(targetStep, startStep);
                            const unsigned target_window_bound = EndIdx(targetStep, endStep);
                            auto &affinity_table_target = threadData.affinityTable.at(target);
                            const CostT comm_aff = instance_->CommunicationCosts(sourceProc, targetProc) * commGain;
                            for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                                if (p == targetProc) {
                                    continue;
                                }

                                for (unsigned idx = targetStartIdx; idx < target_window_bound; idx++) {
                                    affinity_table_target[p][idx] -= comm_aff;
                                }
                            }
                            break;    // since nodeLambdaMap_[source][move.from_proc] == 1
                        }
                    }
                }

                if (nodeLambdaMap_.get_proc_entry(source, move.to_proc) == 1) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const auto &target : instance_->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule_->assigned_superstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node)
                            || (not threadData.affinityTable.IsSelected(target)) || threadData.lockManager.IsLocked(target)) {
                            continue;
                        }

                        if (sourceProc != move.to_proc && is_compatible(target, move.to_proc)) {
                            if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
                                maxGainRecompute[target].full_update = true;
                            } else {
                                maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                            }

                            const unsigned target_window_bound = EndIdx(targetStep, endStep);
                            auto &affinity_table_target_to_proc = threadData.affinityTable.at(target)[move.to_proc];
                            const CostT comm_aff = instance_->CommunicationCosts(sourceProc, move.to_proc) * commGain;
                            for (unsigned idx = StartIdx(targetStep, startStep); idx < target_window_bound; idx++) {
                                affinity_table_target_to_proc[idx] -= comm_aff;
                            }
                        }
                    }
                } else if (nodeLambdaMap_.get_proc_entry(source, move.to_proc) == 2) {
                    for (const auto &target : instance_->GetComputationalDag().Children(source)) {
                        const unsigned targetStep = activeSchedule_->assigned_superstep(target);
                        if ((targetStep < startStep || targetStep > endStep) || (target == move.node)
                            || (not threadData.affinityTable.IsSelected(target)) || threadData.lockManager.IsLocked(target)) {
                            continue;
                        }

                        const unsigned targetProc = activeSchedule_->assigned_processor(target);
                        if (targetProc == move.to_proc) {
                            if (sourceProc != targetProc) {
                                if (maxGainRecompute.find(target) != maxGainRecompute.end()) {
                                    maxGainRecompute[target].full_update = true;
                                } else {
                                    maxGainRecompute[target] = KlGainUpdateInfo(target, true);
                                }

                                const unsigned targetStartIdx = StartIdx(targetStep, startStep);
                                const unsigned target_window_bound = EndIdx(targetStep, endStep);
                                auto &affinity_table_target = threadData.affinityTable.at(target);
                                const CostT comm_aff = instance_->CommunicationCosts(sourceProc, targetProc)
                                                       * graph_->VertexCommWeight(source) * commMultiplier_;
                                for (const unsigned p : procRange_->CompatibleProcessorsVertex(target)) {
                                    if (p == targetProc) {
                                        continue;
                                    }

                                    for (unsigned idx = targetStartIdx; idx < target_window_bound; idx++) {
                                        affinity_table_target[p][idx] += comm_aff;
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }

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
            const unsigned sourceStartIdx = StartIdx(sourceStep, startStep);
            const unsigned windowBound = EndIdx(sourceStep, endStep);
            auto &affinityTableSource = threadData.affinityTable.at(source);

            if (move.from_step < sourceStep + (move.from_proc != sourceProc)) {
                const unsigned diff = sourceStep - move.from_step;
                const unsigned bound = windowSize > diff ? windowSize - diff : 0;
                unsigned idx = sourceStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] += reward;
                    }
                }

                if (windowSize >= diff && is_compatible(source, move.from_proc)) {
                    affinityTableSource[move.from_proc][idx] += reward;
                }

            } else {
                const unsigned diff = move.from_step - sourceStep;
                unsigned idx = windowSize + diff;

                if (idx < windowBound && is_compatible(source, move.from_proc)) {
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
                unsigned idx = sourceStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] -= reward;
                    }
                }

                if (windowSize >= diff && is_compatible(source, move.to_proc)) {
                    affinityTableSource[move.to_proc][idx] -= reward;
                }

            } else {
                const unsigned diff = move.to_step - sourceStep;
                unsigned idx = windowSize + diff;

                if (idx < windowBound && is_compatible(source, move.to_proc)) {
                    affinityTableSource[move.to_proc][idx] -= penalty;
                }
                for (; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        affinityTableSource[p][idx] += penalty;
                    }
                }
            }

            if (move.to_proc != move.from_proc) {
                if (nodeLambdaMap_.has_no_proc_entry(source, move.from_proc)) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        if (p == sourceProc) {
                            continue;
                        }

                        const CostT comm_cost = ChangeCommCost(instance_->CommunicationCosts(p, move.from_proc),
                                                               instance_->CommunicationCosts(sourceProc, move.from_proc),
                                                               commGain);
                        for (unsigned idx = sourceStartIdx; idx < windowBound; idx++) {
                            affinityTableSource[p][idx] -= comm_cost;
                        }
                    }
                }

                if (nodeLambdaMap_.get_proc_entry(source, move.to_proc) == 1) {
                    const CostT commGain = graph_->VertexCommWeight(source) * commMultiplier_;

                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(source)) {
                        if (p == sourceProc) {
                            continue;
                        }

                        const CostT comm_cost = ChangeCommCost(instance_->CommunicationCosts(p, move.to_proc),
                                                               instance_->CommunicationCosts(sourceProc, move.to_proc),
                                                               commGain);
                        for (unsigned idx = sourceStartIdx; idx < windowBound; idx++) {
                            affinityTableSource[p][idx] += comm_cost;
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

                if (windowSize >= diff && is_compatible(node, targetProc)) {
                    affinityTableNode[targetProc][idx] -= reward;
                }

            } else {
                const unsigned diff = targetStep - nodeStep;
                unsigned idx = windowSize + diff;

                if (idx < windowBound && is_compatible(node, targetProc)) {
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

            for (const auto lambda_pair : nodeLambdaMap_.iterate_proc_entries(node)) {
                const auto &lambdaProc = lambda_pair.first;
                const CostT comm_cost = ChangeCommCost(
                    instance_->CommunicationCosts(p, lambdaProc), instance_->CommunicationCosts(nodeProc, lambdaProc), commGain);
                for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                    affinityTableNode[p][idx] += comm_cost;
                }
            }
        }

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

                if (idx - 1 < bound && is_compatible(node, sourceProc)) {
                    affinityTableNode[sourceProc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = sourceStep - nodeStep;
                unsigned idx = std::min(windowSize + diff, windowBound);

                if (idx < windowBound && is_compatible(node, sourceProc)) {
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

                if (sourceProc != nodeProc && nodeLambdaMap_.get_proc_entry(source, nodeProc) == 1) {
                    for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                        affinityTableNode[p][idx] -= instance_->CommunicationCosts(sourceProc, nodeProc) * sourceCommGain;
                    }
                }

                if (sourceProc != p && nodeLambdaMap_.has_no_proc_entry(source, p)) {
                    for (unsigned idx = nodeStartIdx; idx < windowBound; idx++) {
                        affinityTableNode[p][idx] += instance_->CommunicationCosts(sourceProc, p) * sourceCommGain;
                    }
                }
            }
        }    // source
    }
};

}    // namespace osp
