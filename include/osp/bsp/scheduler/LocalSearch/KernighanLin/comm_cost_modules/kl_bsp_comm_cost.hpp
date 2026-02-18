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
#include <array>
#include <iostream>

#include "../kl_active_schedule.hpp"
#include "../kl_improver.hpp"
#include "FastDeltaTacker.hpp"
#include "comm_cost_policies.hpp"
#include "max_comm_datastructure.hpp"

namespace osp {

template <typename GraphT, typename CostT, typename MemoryConstraintT, typename CommPolicy = EagerCommCostPolicy, unsigned windowSize = 1>
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

    MaxCommDatastructure<GraphT, CostT, KlActiveSchedule<GraphT, CostT, MemoryConstraintT>, CommPolicy> commDs_;

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
#ifdef KL_DEBUG_VALIDATE_COMM_DS
        static unsigned moveCounter_ = 0;
        moveCounter_++;
        if (!commDs_.ValidateCommDs(moveCounter_, move)) {
            std::cout << "[KL_DEBUG_VALIDATE_COMM_DS] *** DIVERGENCE at move #" << moveCounter_ << " — ABORTING ***" << std::endl;
            std::abort();
        }
#endif
    }

    void SwapCommSteps(unsigned step1, unsigned step2) { commDs_.SwapSteps(step1, step2); }

    void UpdateLambdaAfterStepRemoval(unsigned removedStep) { commDs_.UpdateLambdaAfterStepRemoval(removedStep); }

    void FixupSendRecvAfterStepRemoval(unsigned removedStep, unsigned oldEndStep) {
        commDs_.FixupSendRecvAfterStepRemoval(removedStep, oldEndStep);
    }

    void UpdateLambdaAfterStepInsertion(unsigned insertedStep) { commDs_.UpdateLambdaAfterStepInsertion(insertedStep); }

    void FixupSendRecvAfterStepInsertion(unsigned insertedStep, unsigned startStep, unsigned endStep) {
        commDs_.FixupSendRecvAfterStepInsertion(insertedStep, startStep, endStep);
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

        // DeltaTracker adapter for CalculateDeltaRemove/CalculateDeltaAdd
        struct DeltaAdapterT {
            decltype(AddDelta) &fn;

            void Add(bool isRecv, unsigned step, unsigned proc, CommWeightT v) { fn(isRecv, step, proc, v); }
        };

        DeltaAdapterT deltaAdapter{AddDelta};

        // Negating adapter for reverting CalculateDeltaAdd
        struct NegDeltaAdapterT {
            decltype(AddDelta) &fn;

            void Add(bool isRecv, unsigned step, unsigned proc, CommWeightT v) { fn(isRecv, step, proc, -v); }
        };

        NegDeltaAdapterT negDeltaAdapter{AddDelta};

        // ========== Phase 1: Remove Node from Current State ==========
        // (Invariant for all candidates)

        // Phase 1 Outgoing: node stops sending to children
        auto nodeLambdaEntries = commDs_.nodeLambdaMap_.IterateProcEntries(node);

        for (const auto [proc, val] : nodeLambdaEntries) {
            if (proc != nodeProc && CommPolicy::HasEntry(val)) {
                const CommWeightT cost = commWNode * instance_->SendCosts(nodeProc, proc);
                if (cost > 0) {
                    int recvStep = CommPolicy::OutgoingRecvStep(nodeStep, val);
                    int sendStep = CommPolicy::OutgoingSendStep(nodeStep, val);
                    if (recvStep >= 0) {
                        AddDelta(true, static_cast<unsigned>(recvStep), proc, -cost);
                    }
                    if (sendStep >= 0) {
                        AddDelta(false, static_cast<unsigned>(sendStep), nodeProc, -cost);
                    }
                }
            }
        }

        // Phase 1 Incoming: parents stop sending to node on nodeProc
        for (const auto &u : graph_->Parents(node)) {
            const unsigned uProc = activeSchedule_->AssignedProcessor(u);
            const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
            const CommWeightT commWU = graph_->VertexCommWeight(u);

            if (uProc != nodeProc) {
                const auto &lambdaVal = commDs_.nodeLambdaMap_.GetProcEntry(u, nodeProc);
                if (CommPolicy::HasEntry(lambdaVal)) {
                    const CommWeightT cost = commWU * instance_->SendCosts(uProc, nodeProc);
                    if (cost > 0) {
                        CommPolicy::CalculateDeltaRemove(lambdaVal, nodeStep, uStep, uProc, nodeProc, cost, deltaAdapter);
                    }
                }
            }
        }

        // ========== Phase 2: Add Node to Each Candidate ==========

        // Helper: compute effective val after conceptually removing one instance of nodeStep.
        // Used for Phase 2A when pTo == nodeProc.
        auto ComputeEffectiveVal = [&](const typename CommPolicy::ValueType &val) -> typename CommPolicy::ValueType {
            if constexpr (std::is_same_v<typename CommPolicy::ValueType, unsigned>) {
                return val > 0 ? val - 1 : 0;
            } else {
                auto result = val;
                auto it = std::find(result.begin(), result.end(), nodeStep);
                if (it != result.end()) {
                    result.erase(it);
                }
                return result;
            }
        };

        // Per-parent precomputed data for Phase 2A incoming additions
        struct ParentAddInfo {
            unsigned uProc;
            unsigned uStep;
            CommWeightT cost;
            typename CommPolicy::ValueType effectiveVal;
        };

        static thread_local std::vector<ParentAddInfo> parentAddInfos;

        // Per-dest-proc precomputed data for Phase 2B outgoing
        struct OutgoingInfo {
            unsigned vProc;
            CommWeightT cost;
            int recvStep;
            int sendStep;
        };

        static thread_local std::vector<OutgoingInfo> outgoingInfos;

        for (const unsigned pTo : procRange_->CompatibleProcessorsVertex(node)) {
            // --- Precompute Phase 2A: parent effective vals ---
            parentAddInfos.clear();
            for (const auto &u : graph_->Parents(node)) {
                const unsigned uProc = activeSchedule_->AssignedProcessor(u);
                if (uProc == pTo) {
                    continue;
                }

                const unsigned uStep = currentVecSchedule.AssignedSuperstep(u);
                const CommWeightT commWU = graph_->VertexCommWeight(u);
                const CommWeightT cost = commWU * instance_->SendCosts(uProc, pTo);
                if (cost <= 0) {
                    continue;
                }

                const auto &valOnPTo = commDs_.nodeLambdaMap_.GetProcEntry(u, pTo);
                typename CommPolicy::ValueType effectiveVal;
                if (pTo == nodeProc) {
                    effectiveVal = ComputeEffectiveVal(valOnPTo);
                } else {
                    effectiveVal = valOnPTo;
                }
                parentAddInfos.push_back({uProc, uStep, cost, std::move(effectiveVal)});
            }

            // --- Precompute Phase 2B: outgoing (node -> children) ---
            outgoingInfos.clear();
            for (const auto [vProc, val] : commDs_.nodeLambdaMap_.IterateProcEntries(node)) {
                if (vProc != pTo && CommPolicy::HasEntry(val)) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(pTo, vProc);
                    if (cost > 0) {
                        int recvStep = -1;
                        int sendStep = -1;
                        if constexpr (!CommPolicy::outgoing_recv_at_parent_step) {
                            recvStep = CommPolicy::OutgoingRecvStep(0, val);
                        }
                        if constexpr (!CommPolicy::outgoing_send_at_parent_step) {
                            sendStep = CommPolicy::OutgoingSendStep(0, val);
                        }
                        outgoingInfos.push_back({vProc, cost, recvStep, sendStep});
                    }
                }
            }

            // --- Iterate Window (sTo) ---
            for (unsigned sToIdx = nodeStartIdx; sToIdx < windowBound; ++sToIdx) {
                unsigned sTo = nodeStep + sToIdx - windowSize;

                // Apply Phase 2A: incoming deltas (policy-aware, sTo-dependent)
                for (const auto &info : parentAddInfos) {
                    CommPolicy::CalculateDeltaAdd(info.effectiveVal, sTo, info.uStep, info.uProc, pTo, info.cost, deltaAdapter);
                }

                // Apply Phase 2B: outgoing deltas (policy-aware)
                for (const auto &info : outgoingInfos) {
                    if constexpr (CommPolicy::outgoing_recv_at_parent_step) {
                        AddDelta(true, sTo, info.vProc, info.cost);
                    } else {
                        if (info.recvStep >= 0) {
                            AddDelta(true, static_cast<unsigned>(info.recvStep), info.vProc, info.cost);
                        }
                    }
                    if constexpr (CommPolicy::outgoing_send_at_parent_step) {
                        AddDelta(false, sTo, pTo, info.cost);
                    } else {
                        if (info.sendStep >= 0) {
                            AddDelta(false, static_cast<unsigned>(info.sendStep), pTo, info.cost);
                        }
                    }
                }

                CostT totalChange = 0;
                for (unsigned step : scratch.activeSteps_) {
                    if (!scratch.sendDeltas_[step].dirtyProcs_.empty() || !scratch.recvDeltas_[step].dirtyProcs_.empty()) {
                        totalChange += CalculateStepCostChange(step, scratch.sendDeltas_[step], scratch.recvDeltas_[step]);
                    }
                }

                affinityTableNode[pTo][sToIdx] += totalChange * instance_->CommunicationCosts();

                // Revert Phase 2B: outgoing deltas
                for (const auto &info : outgoingInfos) {
                    if constexpr (CommPolicy::outgoing_recv_at_parent_step) {
                        AddDelta(true, sTo, info.vProc, -info.cost);
                    } else {
                        if (info.recvStep >= 0) {
                            AddDelta(true, static_cast<unsigned>(info.recvStep), info.vProc, -info.cost);
                        }
                    }
                    if constexpr (CommPolicy::outgoing_send_at_parent_step) {
                        AddDelta(false, sTo, pTo, -info.cost);
                    } else {
                        if (info.sendStep >= 0) {
                            AddDelta(false, static_cast<unsigned>(info.sendStep), pTo, -info.cost);
                        }
                    }
                }

                // Revert Phase 2A: incoming deltas
                for (const auto &info : parentAddInfos) {
                    CommPolicy::CalculateDeltaAdd(info.effectiveVal, sTo, info.uStep, info.uProc, pTo, info.cost, negDeltaAdapter);
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
