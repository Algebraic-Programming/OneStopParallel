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

#include <array>

#include "../kl_active_schedule.hpp"
#include "../kl_improver.hpp"
#include "FastDeltaTacker.hpp"
#include "max_comm_datastructure.hpp"

namespace osp {

template <typename GraphT, typename CostT, typename MemoryConstraintT, unsigned windowSize = 1>
struct KlBspCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;
    using CommWeightT = VCommwT<GraphT>;

    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool isMaxCommCostFunction_ = true;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule_;
    CompatibleProcessorRange<GraphT> *procRange_;
    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    MaxCommDatastructure<GraphT, CostT, KlActiveSchedule<GraphT, CostT, MemoryConstraintT>> commDs_;

    inline CostT GetCommMultiplier() { return 1; }

    inline CostT GetMaxCommWeight() { return commDs_.maxCommWeight_; }

    inline CostT GetMaxCommWeightMultiplied() { return commDs_.maxCommWeight_; }

    inline const std::string Name() const { return "bsp_comm"; }

    inline bool IsCompatible(VertexType node, unsigned proc) { return activeSchedule_->GetInstance().IsCompatible(node, proc); }

    inline unsigned StartIdx(const unsigned nodeStep, const unsigned startStep) {
        return (nodeStep < windowSize + startStep) ? windowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep) {
        return (nodeStep + windowSize <= endStep) ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep);
    }

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &pRange) {
        activeSchedule_ = &sched;
        procRange_ = &pRange;
        instance_ = &sched.GetInstance();
        graph_ = &instance_->GetComputationalDag();

        const unsigned numSteps = activeSchedule_->NumSteps();
        commDs_.Initialize(*activeSchedule_);
    }

    using PreMoveCommDataT = PreMoveCommData<CommWeightT>;

    inline PreMoveCommDataT GetPreMoveCommData(const KlMove &move) { return commDs_.GetPreMoveCommData(move); }

    void ComputeSendReceiveDatastructures() { commDs_.ComputeCommDatastructures(0, activeSchedule_->NumSteps() - 1); }

    template <bool computeDatastructures = true>
    CostT ComputeScheduleCost() {
        if constexpr (computeDatastructures) {
            ComputeSendReceiveDatastructures();
        }

        CostT totalCost = 0;
        for (unsigned step = 0; step < activeSchedule_->NumSteps(); step++) {
            totalCost += activeSchedule_->GetStepMaxWork(step);
            totalCost += commDs_.StepMaxComm(step) * instance_->CommunicationCosts();
        }

        if (activeSchedule_->NumSteps() > 1) {
            totalCost += static_cast<CostT>(activeSchedule_->NumSteps() - 1) * instance_->SynchronisationCosts();
        }

        return totalCost;
    }

    CostT ComputeScheduleCostTest() { return ComputeScheduleCost<false>(); }

    void UpdateDatastructureAfterMove(const KlMove &move, const unsigned startStep, const unsigned endStep) {
        commDs_.UpdateDatastructureAfterMove(move, startStep, endStep);
    }

    // Structure to hold thread-local scratchpads to avoid re-allocation.
    struct ScratchData {
        std::vector<FastDeltaTracker<CommWeightT>> sendDeltas_;    // Size: num_steps
        std::vector<FastDeltaTracker<CommWeightT>> recvDeltas_;    // Size: num_steps

        std::vector<unsigned> activeSteps_;    // List of steps touched in current operation
        std::vector<bool> stepIsActive_;       // Fast lookup for active steps

        std::vector<std::pair<unsigned, CommWeightT>> childCostBuffer_;

        void Init(unsigned nSteps, unsigned nProcs) {
            if (sendDeltas_.size() < nSteps) {
                sendDeltas_.resize(nSteps);
                recvDeltas_.resize(nSteps);
                stepIsActive_.resize(nSteps, false);
                activeSteps_.reserve(nSteps);
            }

            for (auto &tracker : sendDeltas_) {
                tracker.Initialize(nProcs);
            }
            for (auto &tracker : recvDeltas_) {
                tracker.Initialize(nProcs);
            }

            childCostBuffer_.reserve(nProcs);
        }

        void ClearAll() {
            for (unsigned step : activeSteps_) {
                sendDeltas_[step].Clear();
                recvDeltas_[step].Clear();
                stepIsActive_[step] = false;
            }
            activeSteps_.clear();
            childCostBuffer_.clear();
        }

        void MarkActive(unsigned step) {
            if (!stepIsActive_[step]) {
                stepIsActive_[step] = true;
                activeSteps_.push_back(step);
            }
        }
    };

    template <typename AffinityTableT>
    void ComputeCommAffinity(VertexType node,
                             AffinityTableT &affinityTableNode,
                             const CostT &penalty,
                             const CostT &reward,
                             const unsigned startStep,
                             const unsigned endStep) {
        // Use static thread_local scratchpad to avoid allocation in hot loop
        static thread_local ScratchData scratch;
        scratch.Init(activeSchedule_->NumSteps(), instance_->NumberOfProcessors());
        scratch.ClearAll();

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
        }

        const CommWeightT commWNode = graph_->VertexCommWeight(node);
        const auto &currentVecSchedule = activeSchedule_->GetVectorSchedule();

        auto AddDelta = [&](bool isRecv, unsigned step, unsigned proc, CommWeightT val) {
            if (val == 0) {
                return;
            }
            if (step < activeSchedule_->NumSteps()) {
                scratch.MarkActive(step);
                if (isRecv) {
                    scratch.recvDeltas_[step].Add(proc, val);
                } else {
                    scratch.sendDeltas_[step].Add(proc, val);
                }
            }
        };

        // 1. Remove Node from Current State (Phase 1 - Invariant for all candidates)

        // Outgoing (Children)
        // Child stops receiving from nodeProc at nodeStep
        auto nodeLambdaEntries = commDs_.nodeLambdaMap_.IterateProcEntries(node);
        CommWeightT totalSendCostRemoved = 0;

        for (const auto [proc, count] : nodeLambdaEntries) {
            if (proc != nodeProc) {
                const CommWeightT cost = commWNode * instance_->SendCosts(nodeProc, proc);
                if (cost > 0) {
                    AddDelta(true, nodeStep, proc, -cost);
                    totalSendCostRemoved += cost;
                }
            }
        }
        if (totalSendCostRemoved > 0) {
            AddDelta(false, nodeStep, nodeProc, -totalSendCostRemoved);
        }

        // Incoming (Parents)
        for (const auto &u : graph_->Parents(node)) {
            const unsigned uProc = activeSchedule_->AssignedProcessor(u);
            const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
            const CommWeightT commWU = graph_->VertexCommWeight(u);

            if (uProc != nodeProc) {
                if (commDs_.nodeLambdaMap_.GetProcEntry(u, nodeProc) == 1) {
                    const CommWeightT cost = commWU * instance_->SendCosts(uProc, nodeProc);
                    if (cost > 0) {
                        AddDelta(true, uStep, nodeProc, -cost);
                        AddDelta(false, uStep, uProc, -cost);
                    }
                }
            }
        }

        // 2. Add Node to Target (Iterate candidates)

        for (const unsigned pTo : procRange_->CompatibleProcessorsVertex(node)) {
            // --- Part A: Incoming Edges (Parents -> pTo) ---
            // These updates are specific to pTo but independent of sTo.
            // We apply them, run the sTo loop, then revert them.

            for (const auto &u : graph_->Parents(node)) {
                const unsigned uProc = activeSchedule_->AssignedProcessor(u);
                const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
                const CommWeightT commWU = graph_->VertexCommWeight(u);

                if (uProc != pTo) {
                    bool alreadySendingToPTo = false;
                    unsigned countOnPTo = commDs_.nodeLambdaMap_.GetProcEntry(u, pTo);

                    if (pTo == nodeProc) {
                        if (countOnPTo > 0) {
                            countOnPTo--;
                        }
                    }

                    if (countOnPTo > 0) {
                        alreadySendingToPTo = true;
                    }

                    if (!alreadySendingToPTo) {
                        const CommWeightT cost = commWU * instance_->SendCosts(uProc, pTo);
                        if (cost > 0) {
                            AddDelta(true, uStep, pTo, cost);
                            AddDelta(false, uStep, uProc, cost);
                        }
                    }
                }
            }

            // --- Part B: Outgoing Edges (Node -> Children) ---
            // These depend on which processors children are on.
            scratch.childCostBuffer_.clear();
            CommWeightT totalSendCostAdded = 0;

            for (const auto [v_proc, count] : commDs_.nodeLambdaMap_.IterateProcEntries(node)) {
                if (v_proc != pTo) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(pTo, v_proc);
                    if (cost > 0) {
                        scratch.childCostBuffer_.push_back({v_proc, cost});
                        totalSendCostAdded += cost;
                    }
                }
            }

            // Iterate Window (sTo)
            for (unsigned sToIdx = nodeStartIdx; sToIdx < windowBound; ++sToIdx) {
                unsigned sTo = nodeStep + sToIdx - windowSize;

                // Apply Outgoing Deltas for this specific step sTo
                for (const auto &[v_proc, cost] : scratch.childCostBuffer_) {
                    AddDelta(true, sTo, v_proc, cost);
                }

                if (totalSendCostAdded > 0) {
                    AddDelta(false, sTo, pTo, totalSendCostAdded);
                }

                CostT totalChange = 0;

                // Only check steps that are active (modified in Phase 1, Part A, or Part B)
                for (unsigned step : scratch.activeSteps_) {
                    // Check if dirtyProcs_ is empty implies no change for this step
                    // FastDeltaTracker ensures dirtyProcs_ is empty if all deltas summed to 0
                    if (!scratch.sendDeltas_[step].dirtyProcs_.empty() || !scratch.recvDeltas_[step].dirtyProcs_.empty()) {
                        totalChange += CalculateStepCostChange(step, scratch.sendDeltas_[step], scratch.recvDeltas_[step]);
                    }
                }

                affinityTableNode[pTo][sToIdx] += totalChange * instance_->CommunicationCosts();

                // Revert Outgoing Deltas for sTo (Inverse of Apply)
                for (const auto &[v_proc, cost] : scratch.childCostBuffer_) {
                    AddDelta(true, sTo, v_proc, -cost);
                }
                if (totalSendCostAdded > 0) {
                    AddDelta(false, sTo, pTo, -totalSendCostAdded);
                }
            }

            // Revert Incoming Deltas (Inverse of Part A)
            for (const auto &u : graph_->Parents(node)) {
                const unsigned uProc = activeSchedule_->AssignedProcessor(u);
                const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
                const CommWeightT commWU = graph_->VertexCommWeight(u);

                if (uProc != pTo) {
                    bool alreadySendingToPTo = false;
                    unsigned countOnPTo = commDs_.nodeLambdaMap_.GetProcEntry(u, pTo);
                    if (pTo == nodeProc) {
                        if (countOnPTo > 0) {
                            countOnPTo--;
                        }
                    }
                    if (countOnPTo > 0) {
                        alreadySendingToPTo = true;
                    }

                    if (!alreadySendingToPTo) {
                        const CommWeightT cost = commWU * instance_->SendCosts(uProc, pTo);
                        if (cost > 0) {
                            AddDelta(true, uStep, pTo, -cost);
                            AddDelta(false, uStep, uProc, -cost);
                        }
                    }
                }
            }
        }
    }

    CommWeightT CalculateStepCostChange(unsigned step,
                                        const FastDeltaTracker<CommWeightT> &deltaSend,
                                        const FastDeltaTracker<CommWeightT> &deltaRecv) {
        CommWeightT oldMax = commDs_.StepMaxComm(step);
        unsigned oldMaxCount = commDs_.StepMaxCommCount(step);

        CommWeightT newGlobalMax = 0;
        unsigned reducedMaxInstances = 0;

        // 1. Check modified sends (Iterate sparse dirty list)
        for (unsigned proc : deltaSend.dirtyProcs_) {
            CommWeightT delta = deltaSend.Get(proc);
            // delta cannot be 0 here due to FastDeltaTracker invariant

            CommWeightT currentVal = commDs_.StepProcSend(step, proc);
            CommWeightT newVal = currentVal + delta;

            if (newVal > newGlobalMax) {
                newGlobalMax = newVal;
            }
            if (delta < 0 && currentVal == oldMax) {
                reducedMaxInstances++;
            }
        }

        // 2. Check modified receives (Iterate sparse dirty list)
        for (unsigned proc : deltaRecv.dirtyProcs_) {
            CommWeightT delta = deltaRecv.Get(proc);

            CommWeightT currentVal = commDs_.StepProcReceive(step, proc);
            CommWeightT newVal = currentVal + delta;

            if (newVal > newGlobalMax) {
                newGlobalMax = newVal;
            }
            if (delta < 0 && currentVal == oldMax) {
                reducedMaxInstances++;
            }
        }

        // 3. Determine result
        if (newGlobalMax > oldMax) {
            return newGlobalMax - oldMax;
        }
        if (reducedMaxInstances < oldMaxCount) {
            return 0;
        }

        CommWeightT maxNonDirty = 0;
        const unsigned numProcs = instance_->NumberOfProcessors();
        for (unsigned p = 0; p < numProcs; ++p) {
            if (!deltaSend.IsDirty(p)) {
                maxNonDirty = std::max(maxNonDirty, commDs_.StepProcSend(step, p));
            }
            if (!deltaRecv.IsDirty(p)) {
                maxNonDirty = std::max(maxNonDirty, commDs_.StepProcReceive(step, p));
            }
        }
        return std::max(newGlobalMax, maxNonDirty) - oldMax;
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const KlMove &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &,
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
        }
    }
};

}    // namespace osp
