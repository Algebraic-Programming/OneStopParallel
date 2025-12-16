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
    std::vector<CommWeightT> denseVals_;      // Size: num_procs
    std::vector<unsigned> dirtyProcs_;        // List of modified indices
    std::vector<unsigned> procDirtyIndex_;    // Map proc -> index in dirtyProcs_ (num_procs if not dirty)
    unsigned numProcs_ = 0;

    void Initialize(unsigned nProcs) {
        if (nProcs > numProcs_) {
            numProcs_ = nProcs;
            denseVals_.resize(numProcs_, 0);
            dirtyProcs_.reserve(numProcs_);
            procDirtyIndex_.resize(numProcs_, numProcs_);
        }
    }

    inline void Add(unsigned proc, CommWeightT val) {
        if (val == 0) {
            return;
        }

        // If currently 0, it is becoming dirty
        if (denseVals_[proc] == 0) {
            procDirtyIndex_[proc] = static_cast<unsigned>(dirtyProcs_.size());
            dirtyProcs_.push_back(proc);
        }

        denseVals_[proc] += val;

        // If it returns to 0, remove it from dirty list (Swap and Pop for O(1))
        if (denseVals_[proc] == 0) {
            unsigned idx = procDirtyIndex_[proc];
            unsigned lastProc = dirtyProcs_.back();

            // Move last element to the hole
            dirtyProcs_[idx] = lastProc;
            procDirtyIndex_[lastProc] = idx;

            // Remove last
            dirtyProcs_.pop_back();
            procDirtyIndex_[proc] = numProcs_;
        }
    }

    inline CommWeightT Get(unsigned proc) const {
        if (proc < denseVals_.size()) {
            return denseVals_[proc];
        }
        return 0;
    }

    inline void Clear() {
        for (unsigned p : dirtyProcs_) {
            denseVals_[p] = 0;
            procDirtyIndex_[p] = numProcs_;
        }
        dirtyProcs_.clear();
    }
};

template <typename GraphT, typename CostT, typename MemoryConstraintT, unsigned windowSize = 1>
struct KlBspCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using kl_move = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;
    using comm_weight_t = VCommwT<GraphT>;

    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool isMaxCommCostFunction_ = true;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule_;
    CompatibleProcessorRange<GraphT> *procRange_;
    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    MaxCommDatastructure<GraphT, CostT, KlActiveSchedule<GraphT, CostT, MemoryConstraintT>> commDs_;

    inline CostT GetCommMultiplier() { return 1; }

    inline CostT GetMaxCommWeight() { return commDs_.max_comm_weight; }

    inline CostT GetMaxCommWeightMultiplied() { return commDs_.max_comm_weight; }

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
        commDs_.initialize(*activeSchedule_);
    }

    using pre_move_comm_data_t = PreMoveCommData<comm_weight_t>;

    inline PreMoveCommData<comm_weight_t> GetPreMoveCommData(const kl_move &move) { return commDs_.get_pre_move_comm_data(move); }

    void ComputeSendReceiveDatastructures() { commDs_.compute_comm_datastructures(0, activeSchedule_->NumSteps() - 1); }

    template <bool computeDatastructures = true>
    CostT ComputeScheduleCost() {
        if constexpr (computeDatastructures) {
            ComputeSendReceiveDatastructures();
        }

        CostT totalCost = 0;
        for (unsigned step = 0; step < activeSchedule_->NumSteps(); step++) {
            totalCost += activeSchedule_->get_step_max_work(step);
            totalCost += commDs_.step_max_comm(step) * instance_->CommunicationCosts();
        }

        if (activeSchedule_->NumSteps() > 1) {
            totalCost += static_cast<CostT>(activeSchedule_->NumSteps() - 1) * instance_->SynchronisationCosts();
        }

        return totalCost;
    }

    CostT ComputeScheduleCostTest() { return ComputeScheduleCost<false>(); }

    void UpdateDatastructureAfterMove(const kl_move &move, const unsigned startStep, const unsigned endStep) {
        commDs_.update_datastructure_after_move(move, startStep, endStep);
    }

    // Structure to hold thread-local scratchpads to avoid re-allocation.
    struct ScratchData {
        std::vector<FastDeltaTracker<comm_weight_t>> sendDeltas_;    // Size: num_steps
        std::vector<FastDeltaTracker<comm_weight_t>> recvDeltas_;    // Size: num_steps

        std::vector<unsigned> activeSteps_;    // List of steps touched in current operation
        std::vector<bool> stepIsActive_;       // Fast lookup for active steps

        std::vector<std::pair<unsigned, comm_weight_t>> childCostBuffer_;

        void Init(unsigned nSteps, unsigned nProcs) {
            if (sendDeltas_.size() < nSteps) {
                sendDeltas_.resize(nSteps);
                recvDeltas_.resize(nSteps);
                stepIsActive_.resize(nSteps, false);
                activeSteps_.reserve(nSteps);
            }

            for (auto &tracker : sendDeltas_) {
                tracker.initialize(nProcs);
            }
            for (auto &tracker : recvDeltas_) {
                tracker.initialize(nProcs);
            }

            childCostBuffer_.reserve(nProcs);
        }

        void ClearAll() {
            for (unsigned step : activeSteps_) {
                sendDeltas_[step].clear();
                recvDeltas_[step].clear();
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(node)) {
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(node)) {
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }
            }
        }

        const comm_weight_t commWNode = graph_->VertexCommWeight(node);
        const auto &currentVecSchedule = activeSchedule_->getVectorSchedule();

        auto addDelta = [&](bool isRecv, unsigned step, unsigned proc, comm_weight_t val) {
            if (val == 0) {
                return;
            }
            if (step < activeSchedule_->NumSteps()) {
                scratch.MarkActive(step);
                if (isRecv) {
                    scratch.recvDeltas_[step].add(proc, val);
                } else {
                    scratch.sendDeltas_[step].add(proc, val);
                }
            }
        };

        // 1. Remove Node from Current State (Phase 1 - Invariant for all candidates)

        // Outgoing (Children)
        // Child stops receiving from nodeProc at nodeStep
        auto nodeLambdaEntries = commDs_.nodeLambdaMap_.iterate_proc_entries(node);
        comm_weight_t totalSendCostRemoved = 0;

        for (const auto [proc, count] : nodeLambdaEntries) {
            if (proc != nodeProc) {
                const comm_weight_t cost = commWNode * instance_->SendCosts(nodeProc, proc);
                if (cost > 0) {
                    add_delta(true, nodeStep, proc, -cost);
                    totalSendCostRemoved += cost;
                }
            }
        }
        if (totalSendCostRemoved > 0) {
            addDelta(false, nodeStep, nodeProc, -totalSendCostRemoved);
        }

        // Incoming (Parents)
        for (const auto &u : graph_->Parents(node)) {
            const unsigned uProc = activeSchedule_->assigned_processor(u);
            const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
            const comm_weight_t commWU = graph_->VertexCommWeight(u);

            if (uProc != nodeProc) {
                if (commDs_.nodeLambdaMap_.get_proc_entry(u, nodeProc) == 1) {
                    const comm_weight_t cost = commWU * instance_->SendCosts(uProc, nodeProc);
                    if (cost > 0) {
                        add_delta(true, uStep, nodeProc, -cost);
                        add_delta(false, uStep, uProc, -cost);
                    }
                }
            }
        }

        // 2. Add Node to Target (Iterate candidates)

        for (const unsigned p_to : procRange_->compatible_processors_vertex(node)) {
            // --- Part A: Incoming Edges (Parents -> p_to) ---
            // These updates are specific to p_to but independent of s_to.
            // We apply them, run the s_to loop, then revert them.

            for (const auto &u : graph_->Parents(node)) {
                const unsigned uProc = activeSchedule_->assigned_processor(u);
                const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
                const comm_weight_t commWU = graph_->VertexCommWeight(u);

                if (uProc != p_to) {
                    bool already_sending_to_p_to = false;
                    unsigned count_on_p_to = commDs_.nodeLambdaMap_.get_proc_entry(u, p_to);

                    if (p_to == nodeProc) {
                        if (count_on_p_to > 0) {
                            count_on_p_to--;
                        }
                    }

                    if (count_on_p_to > 0) {
                        already_sending_to_p_to = true;
                    }

                    if (!already_sending_to_p_to) {
                        const comm_weight_t cost = commWU * instance_->SendCosts(uProc, p_to);
                        if (cost > 0) {
                            add_delta(true, uStep, p_to, cost);
                            add_delta(false, uStep, uProc, cost);
                        }
                    }
                }
            }

            // --- Part B: Outgoing Edges (Node -> Children) ---
            // These depend on which processors children are on.
            scratch.childCostBuffer_.clear();
            comm_weight_t totalSendCostAdded = 0;

            for (const auto [v_proc, count] : commDs_.nodeLambdaMap_.iterate_proc_entries(node)) {
                if (v_proc != p_to) {
                    const comm_weight_t cost = commWNode * instance_->SendCosts(p_to, v_proc);
                    if (cost > 0) {
                        scratch.childCostBuffer_.push_back({v_proc, cost});
                        totalSendCostAdded += cost;
                    }
                }
            }

            // Iterate Window (s_to)
            for (unsigned s_to_idx = nodeStartIdx; s_to_idx < windowBound; ++s_to_idx) {
                unsigned s_to = nodeStep + s_to_idx - windowSize;

                // Apply Outgoing Deltas for this specific step s_to
                for (const auto &[v_proc, cost] : scratch.childCostBuffer_) {
                    add_delta(true, s_to, v_proc, cost);
                }

                if (totalSendCostAdded > 0) {
                    add_delta(false, s_to, p_to, totalSendCostAdded);
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

                affinityTableNode[p_to][s_to_idx] += totalChange * instance_->CommunicationCosts();

                // Revert Outgoing Deltas for s_to (Inverse of Apply)
                for (const auto &[v_proc, cost] : scratch.childCostBuffer_) {
                    add_delta(true, s_to, v_proc, -cost);
                }
                if (totalSendCostAdded > 0) {
                    add_delta(false, s_to, p_to, -totalSendCostAdded);
                }
            }

            // Revert Incoming Deltas (Inverse of Part A)
            for (const auto &u : graph_->Parents(node)) {
                const unsigned uProc = activeSchedule_->assigned_processor(u);
                const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
                const comm_weight_t commWU = graph_->VertexCommWeight(u);

                if (uProc != p_to) {
                    bool already_sending_to_p_to = false;
                    unsigned count_on_p_to = commDs_.nodeLambdaMap_.get_proc_entry(u, p_to);
                    if (p_to == nodeProc) {
                        if (count_on_p_to > 0) {
                            count_on_p_to--;
                        }
                    }
                    if (count_on_p_to > 0) {
                        already_sending_to_p_to = true;
                    }

                    if (!already_sending_to_p_to) {
                        const comm_weight_t cost = commWU * instance_->SendCosts(uProc, p_to);
                        if (cost > 0) {
                            add_delta(true, uStep, p_to, -cost);
                            add_delta(false, uStep, uProc, -cost);
                        }
                    }
                }
            }
        }
    }

    comm_weight_t CalculateStepCostChange(unsigned step,
                                          const FastDeltaTracker<comm_weight_t> &deltaSend,
                                          const FastDeltaTracker<comm_weight_t> &deltaRecv) {
        comm_weight_t oldMax = commDs_.step_max_comm(step);
        comm_weight_t secondMax = commDs_.step_second_max_comm(step);
        unsigned oldMaxCount = commDs_.step_max_comm_count(step);

        comm_weight_t newGlobalMax = 0;
        unsigned reducedMaxInstances = 0;

        // 1. Check modified sends (Iterate sparse dirty list)
        for (unsigned proc : deltaSend.dirtyProcs_) {
            comm_weight_t delta = deltaSend.get(proc);
            // delta cannot be 0 here due to FastDeltaTracker invariant

            comm_weight_t currentVal = commDs_.step_proc_send(step, proc);
            comm_weight_t newVal = currentVal + delta;

            if (newVal > newGlobalMax) {
                newGlobalMax = newVal;
            }
            if (delta < 0 && currentVal == oldMax) {
                reducedMaxInstances++;
            }
        }

        // 2. Check modified receives (Iterate sparse dirty list)
        for (unsigned proc : deltaRecv.dirtyProcs_) {
            comm_weight_t delta = deltaRecv.get(proc);

            comm_weight_t currentVal = commDs_.step_proc_receive(step, proc);
            comm_weight_t newVal = currentVal + delta;

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
    void UpdateNodeCommAffinity(const kl_move &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &,
                                std::vector<VertexType> &newNodes) {
        const unsigned startStep = threadData.startStep;
        const unsigned endStep = threadData.endStep;

        for (const auto &target : instance_->GetComputationalDag().Children(move.node)) {
            const unsigned targetStep = activeSchedule_->assigned_superstep(target);
            if (targetStep < startStep || targetStep > endStep) {
                continue;
            }

            if (threadData.lockManager.is_locked(target)) {
                continue;
            }

            if (not threadData.affinityTable.is_selected(target)) {
                newNodes.push_back(target);
                continue;
            }

            const unsigned targetProc = activeSchedule_->assigned_processor(target);
            const unsigned targetStartIdx = StartIdx(targetStep, startStep);
            auto &affinityTable = threadData.affinityTable.at(target);

            if (move.from_step < targetStep + (move.from_proc == targetProc)) {
                const unsigned diff = targetStep - move.from_step;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                unsigned idx = targetStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->compatible_processors_vertex(target)) {
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(target)) {
                        affinityTable[p][idx] += reward;
                    }
                }
            }

            if (move.to_step < targetStep + (move.to_proc == targetProc)) {
                unsigned idx = targetStartIdx;
                const unsigned diff = targetStep - move.to_step;
                const unsigned bound = windowSize >= diff ? windowSize - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->compatible_processors_vertex(target)) {
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(target)) {
                        affinityTable[p][idx] -= reward;
                    }
                }
            }
        }

        for (const auto &source : instance_->GetComputationalDag().Parents(move.node)) {
            const unsigned sourceStep = activeSchedule_->assigned_superstep(source);
            if (sourceStep < startStep || sourceStep > endStep) {
                continue;
            }

            if (threadData.lockManager.is_locked(source)) {
                continue;
            }

            if (not threadData.affinityTable.is_selected(source)) {
                newNodes.push_back(source);
                continue;
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(source)) {
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(source)) {
                        affinityTableSource[p][idx] -= penalty;
                    }
                }
            }

            if (move.to_step < sourceStep + (move.to_proc != sourceProc)) {
                const unsigned diff = sourceStep - move.to_step;
                const unsigned bound = windowSize > diff ? windowSize - diff : 0;
                unsigned idx = sourceStartIdx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : procRange_->compatible_processors_vertex(source)) {
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
                    for (const unsigned p : procRange_->compatible_processors_vertex(source)) {
                        affinityTableSource[p][idx] += penalty;
                    }
                }
            }
        }
    }
};

}    // namespace osp
