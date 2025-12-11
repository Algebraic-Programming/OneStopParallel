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
#include "max_comm_datastructure.hpp"

namespace osp {

// A lightweight helper to track deltas without hash maps or repeated allocations.
// Uses a dense vector for O(1) lookups and a sparse list for fast iteration/clearing.
template <typename CommWeightT>
struct FastDeltaTracker {
    std::vector<CommWeightT> denseVals;      // Size: num_procs
    std::vector<unsigned> dirtyProcs;        // List of modified indices
    std::vector<unsigned> procDirtyIndex;    // Map proc -> index in dirty_procs (num_procs if not dirty)
    unsigned numProcs = 0;

    void Initialize(unsigned nProcs) {
        if (nProcs > numProcs) {
            numProcs = nProcs;
            denseVals.resize(numProcs, 0);
            dirtyProcs.reserve(numProcs);
            procDirtyIndex.resize(numProcs, numProcs);
        }
    }

    inline void Add(unsigned proc, CommWeightT val) {
        if (val == 0) {
            return;
        }

        // If currently 0, it is becoming dirty
        if (denseVals[proc] == 0) {
            procDirtyIndex[proc] = static_cast<unsigned>(dirtyProcs.size());
            dirtyProcs.push_back(proc);
        }

        denseVals[proc] += val;

        // If it returns to 0, remove it from dirty list (Swap and Pop for O(1))
        if (denseVals[proc] == 0) {
            unsigned idx = procDirtyIndex[proc];
            unsigned lastProc = dirtyProcs.back();

            // Move last element to the hole
            dirtyProcs[idx] = lastProc;
            procDirtyIndex[lastProc] = idx;

            // Remove last
            dirtyProcs.pop_back();
            procDirtyIndex[proc] = numProcs;
        }
    }

    inline CommWeightT Get(unsigned proc) const {
        if (proc < denseVals.size()) {
            return denseVals[proc];
        }
        return 0;
    }

    inline void Clear() {
        for (unsigned p : dirtyProcs) {
            denseVals[p] = 0;
            procDirtyIndex[p] = numProcs;
        }
        dirtyProcs.clear();
    }
};

template <typename GraphT, typename CostT, typename MemoryConstraintT, unsigned WindowSize = 1>
struct KlBspCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;
    using CommWeightT = VCommwT<GraphT>;

    constexpr static unsigned windowRange = 2 * WindowSize + 1;
    constexpr static bool isMaxCommCostFunction = true;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule;
    CompatibleProcessorRange<GraphT> *procRange;
    const GraphT *graph;
    const BspInstance<GraphT> *instance;

    MaxCommDatastructure<GraphT, CostT, KlActiveSchedule<GraphT, CostT, MemoryConstraintT>> commDs;

    inline CostT GetCommMultiplier() { return 1; }

    inline CostT GetMaxCommWeight() { return commDs.max_comm_weight; }

    inline CostT GetMaxCommWeightMultiplied() { return commDs.maxCommWeight; }

    inline const std::string Name() const { return "bsp_comm"; }

    inline bool IsCompatible(VertexType node, unsigned proc) { return activeSchedule->GetInstance().IsCompatible(node, proc); }

    inline unsigned StartIdx(const unsigned nodeStep, const unsigned startStep) {
        return (nodeStep < WindowSize + startStep) ? WindowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep) {
        return (nodeStep + WindowSize <= endStep) ? windowRange : windowRange - (nodeStep + WindowSize - endStep);
    }

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &pRange) {
        activeSchedule = &sched;
        procRange = &pRange;
        instance = &sched.GetInstance();
        graph = &instance->GetComputationalDag();

        const unsigned numSteps = activeSchedule->NumSteps();
        commDs.Initialize(*activeSchedule);
    }

    using PreMoveCommDataT = PreMoveCommData<CommWeightT>;

    inline PreMoveCommData<CommWeightT> GetPreMoveCommData(const KlMove &move) { return commDs.GetPreMoveCommData(move); }

    void ComputeSendReceiveDatastructures() { commDs.ComputeCommDatastructures(0, activeSchedule->NumSteps() - 1); }

    template <bool ComputeDatastructures = true>
    CostT ComputeScheduleCost() {
        if constexpr (ComputeDatastructures) {
            ComputeSendReceiveDatastructures();
        }

        CostT totalCost = 0;
        for (unsigned step = 0; step < activeSchedule->NumSteps(); step++) {
            totalCost += activeSchedule->GetStepMaxWork(step);
            totalCost += commDs.StepMaxComm(step) * instance->CommunicationCosts();
        }

        if (activeSchedule->NumSteps() > 1) {
            totalCost += static_cast<CostT>(activeSchedule->NumSteps() - 1) * instance->SynchronisationCosts();
        }

        return totalCost;
    }

    CostT ComputeScheduleCostTest() { return ComputeScheduleCost<false>(); }

    void UpdateDatastructureAfterMove(const KlMove &move, const unsigned startStep, const unsigned endStep) {
        commDs.UpdateDatastructureAfterMove(move, startStep, endStep);
    }

    // Structure to hold thread-local scratchpads to avoid re-allocation.
    struct ScratchData {
        std::vector<FastDeltaTracker<CommWeightT>> sendDeltas;    // Size: num_steps
        std::vector<FastDeltaTracker<CommWeightT>> recvDeltas;    // Size: num_steps

        std::vector<unsigned> activeSteps;    // List of steps touched in current operation
        std::vector<bool> stepIsActive;       // Fast lookup for active steps

        std::vector<std::pair<unsigned, CommWeightT>> childCostBuffer;

        void Init(unsigned nSteps, unsigned nProcs) {
            if (sendDeltas.size() < nSteps) {
                sendDeltas.resize(nSteps);
                recvDeltas.resize(nSteps);
                stepIsActive.resize(nSteps, false);
                activeSteps.reserve(nSteps);
            }

            for (auto &tracker : sendDeltas) {
                tracker.Initialize(nProcs);
            }
            for (auto &tracker : recvDeltas) {
                tracker.Initialize(nProcs);
            }

            childCostBuffer.reserve(nProcs);
        }

        void ClearAll() {
            for (unsigned step : activeSteps) {
                sendDeltas[step].Clear();
                recvDeltas[step].Clear();
                stepIsActive[step] = false;
            }
            activeSteps.clear();
            childCostBuffer.clear();
        }

        void MarkActive(unsigned step) {
            if (!stepIsActive[step]) {
                stepIsActive[step] = true;
                activeSteps.push_back(step);
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
        scratch.Init(activeSchedule->NumSteps(), instance->NumberOfProcessors());
        scratch.ClearAll();

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
        }

        const CommWeightT commWNode = graph->VertexCommWeight(node);
        const auto &currentVecSchedule = activeSchedule->GetVectorSchedule();

        auto addDelta = [&](bool isRecv, unsigned step, unsigned proc, CommWeightT val) {
            if (val == 0) {
                return;
            }
            if (step < activeSchedule->NumSteps()) {
                scratch.MarkActive(step);
                if (isRecv) {
                    scratch.recvDeltas[step].Add(proc, val);
                } else {
                    scratch.sendDeltas[step].Add(proc, val);
                }
            }
        };

        // 1. Remove Node from Current State (Phase 1 - Invariant for all candidates)

        // Outgoing (Children)
        // Child stops receiving from node_proc at node_step
        auto nodeLambdaEntries = commDs.nodeLambdaMap.IterateProcEntries(node);
        CommWeightT totalSendCostRemoved = 0;

        for (const auto [proc, count] : nodeLambdaEntries) {
            if (proc != nodeProc) {
                const CommWeightT cost = commWNode * instance->SendCosts(nodeProc, proc);
                if (cost > 0) {
                    addDelta(true, nodeStep, proc, -cost);
                    totalSendCostRemoved += cost;
                }
            }
        }
        if (totalSendCostRemoved > 0) {
            addDelta(false, nodeStep, nodeProc, -totalSendCostRemoved);
        }

        // Incoming (Parents)
        for (const auto &u : graph->Parents(node)) {
            const unsigned uProc = activeSchedule->AssignedProcessor(u);
            const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
            const CommWeightT commWU = graph->VertexCommWeight(u);

            if (uProc != nodeProc) {
                if (commDs.nodeLambdaMap.GetProcEntry(u, nodeProc) == 1) {
                    const CommWeightT cost = commWU * instance->SendCosts(uProc, nodeProc);
                    if (cost > 0) {
                        addDelta(true, uStep, nodeProc, -cost);
                        addDelta(false, uStep, uProc, -cost);
                    }
                }
            }
        }

        // 2. Add Node to Target (Iterate candidates)

        for (const unsigned pTo : procRange->CompatibleProcessorsVertex(node)) {
            // --- Part A: Incoming Edges (Parents -> p_to) ---
            // These updates are specific to p_to but independent of s_to.
            // We apply them, run the s_to loop, then revert them.

            for (const auto &u : graph->Parents(node)) {
                const unsigned uProc = activeSchedule->AssignedProcessor(u);
                const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
                const CommWeightT commWU = graph->VertexCommWeight(u);

                if (uProc != pTo) {
                    bool alreadySendingToPTo = false;
                    unsigned countOnPTo = commDs.nodeLambdaMap.GetProcEntry(u, pTo);

                    if (pTo == nodeProc) {
                        if (countOnPTo > 0) {
                            countOnPTo--;
                        }
                    }

                    if (countOnPTo > 0) {
                        alreadySendingToPTo = true;
                    }

                    if (!alreadySendingToPTo) {
                        const CommWeightT cost = commWU * instance->SendCosts(uProc, pTo);
                        if (cost > 0) {
                            addDelta(true, uStep, pTo, cost);
                            addDelta(false, uStep, uProc, cost);
                        }
                    }
                }
            }

            // --- Part B: Outgoing Edges (Node -> Children) ---
            // These depend on which processors children are on.
            scratch.childCostBuffer.clear();
            CommWeightT totalSendCostAdded = 0;

            for (const auto [v_proc, count] : commDs.nodeLambdaMap.IterateProcEntries(node)) {
                if (v_proc != pTo) {
                    const CommWeightT cost = commWNode * instance->SendCosts(pTo, v_proc);
                    if (cost > 0) {
                        scratch.childCostBuffer.push_back({v_proc, cost});
                        totalSendCostAdded += cost;
                    }
                }
            }

            // Iterate Window (s_to)
            for (unsigned sToIdx = nodeStartIdx; sToIdx < windowBound; ++sToIdx) {
                unsigned sTo = nodeStep + sToIdx - WindowSize;

                // Apply Outgoing Deltas for this specific step s_to
                for (const auto &[v_proc, cost] : scratch.childCostBuffer) {
                    addDelta(true, sTo, v_proc, cost);
                }

                if (totalSendCostAdded > 0) {
                    addDelta(false, sTo, pTo, totalSendCostAdded);
                }

                CostT totalChange = 0;

                // Only check steps that are active (modified in Phase 1, Part A, or Part B)
                for (unsigned step : scratch.activeSteps) {
                    // Check if dirty_procs is empty implies no change for this step
                    // FastDeltaTracker ensures dirty_procs is empty if all deltas summed to 0
                    if (!scratch.sendDeltas[step].dirtyProcs.empty() || !scratch.recvDeltas[step].dirtyProcs.empty()) {
                        totalChange += CalculateStepCostChange(step, scratch.sendDeltas[step], scratch.recvDeltas[step]);
                    }
                }

                affinityTableNode[pTo][sToIdx] += totalChange * instance->CommunicationCosts();

                // Revert Outgoing Deltas for s_to (Inverse of Apply)
                for (const auto &[v_proc, cost] : scratch.childCostBuffer) {
                    addDelta(true, sTo, v_proc, -cost);
                }
                if (totalSendCostAdded > 0) {
                    addDelta(false, sTo, pTo, -totalSendCostAdded);
                }
            }

            // Revert Incoming Deltas (Inverse of Part A)
            for (const auto &u : graph->Parents(node)) {
                const unsigned uProc = activeSchedule->AssignedProcessor(u);
                const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
                const CommWeightT commWU = graph->VertexCommWeight(u);

                if (uProc != pTo) {
                    bool alreadySendingToPTo = false;
                    unsigned countOnPTo = commDs.nodeLambdaMap.GetProcEntry(u, pTo);
                    if (pTo == nodeProc) {
                        if (countOnPTo > 0) {
                            countOnPTo--;
                        }
                    }
                    if (countOnPTo > 0) {
                        alreadySendingToPTo = true;
                    }

                    if (!alreadySendingToPTo) {
                        const CommWeightT cost = commWU * instance->SendCosts(uProc, pTo);
                        if (cost > 0) {
                            addDelta(true, uStep, pTo, -cost);
                            addDelta(false, uStep, uProc, -cost);
                        }
                    }
                }
            }
        }
    }

    CommWeightT CalculateStepCostChange(unsigned step,
                                        const FastDeltaTracker<CommWeightT> &deltaSend,
                                        const FastDeltaTracker<CommWeightT> &deltaRecv) {
        CommWeightT oldMax = commDs.StepMaxComm(step);
        CommWeightT secondMax = commDs.StepSecondMaxComm(step);
        unsigned oldMaxCount = commDs.StepMaxCommCount(step);

        CommWeightT newGlobalMax = 0;
        unsigned reducedMaxInstances = 0;

        // 1. Check modified sends (Iterate sparse dirty list)
        for (unsigned proc : deltaSend.dirtyProcs) {
            CommWeightT delta = deltaSend.Get(proc);
            // delta cannot be 0 here due to FastDeltaTracker invariant

            CommWeightT currentVal = commDs.StepProcSend(step, proc);
            CommWeightT newVal = currentVal + delta;

            if (newVal > newGlobalMax) {
                newGlobalMax = newVal;
            }
            if (delta < 0 && currentVal == oldMax) {
                reducedMaxInstances++;
            }
        }

        // 2. Check modified receives (Iterate sparse dirty list)
        for (unsigned proc : deltaRecv.dirtyProcs) {
            CommWeightT delta = deltaRecv.Get(proc);

            CommWeightT currentVal = commDs.StepProcReceive(step, proc);
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
        return std::max(newGlobalMax, secondMax) - oldMax;
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const KlMove &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &,
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
        }
    }
};

}    // namespace osp
