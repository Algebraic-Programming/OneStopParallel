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
#include <vector>

#include "../kl_active_schedule.hpp"
#include "../kl_improver_base.hpp"
#include "FastDeltaTacker.hpp"
#include "comm_cost_policies.hpp"
#include "max_comm_datastructure.hpp"

namespace osp {

template <typename GraphT, typename CostT, typename MemoryConstraintT, typename CommPolicy = EagerCommCostPolicy, unsigned windowSize = 1>
struct KlMaxBspCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;
    using CommWeightT = VCommwT<GraphT>;
    using VertexWorkWeightT = VWorkwT<GraphT>;

    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool isMaxCommCostFunction_ = true;

    /// Trait: work and comm are coupled via max(work, comm*g).
    /// Base class uses this to skip separate ComputeWorkAffinity and instead
    /// call ComputeNodeAffinity which handles both in a single pass.
    constexpr static bool coupledWorkComm_ = true;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule_;
    CompatibleProcessorRange<GraphT> *procRange_;
    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    MaxCommDatastructure<GraphT, CostT, KlActiveSchedule<GraphT, CostT, MemoryConstraintT>, CommPolicy> commDs_;

    // =========================================================================
    // Simple accessors
    // =========================================================================

    inline CostT GetCommMultiplier() { return 1; }

    inline CostT GetMaxCommWeight() { return commDs_.maxCommWeight_; }

    inline CostT GetMaxCommWeightMultiplied() { return commDs_.maxCommWeight_; }

    inline const std::string Name() const { return "max_bsp_comm"; }

    inline bool IsCompatible(VertexType node, unsigned proc) { return activeSchedule_->GetInstance().IsCompatible(node, proc); }

    inline unsigned StartIdx(const unsigned nodeStep, const unsigned startStep) {
        return (nodeStep < windowSize + startStep) ? windowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep) {
        return (nodeStep + windowSize <= endStep) ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep);
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &pRange) {
        activeSchedule_ = &sched;
        procRange_ = &pRange;
        instance_ = &sched.GetInstance();
        graph_ = &instance_->GetComputationalDag();

        commDs_.Initialize(*activeSchedule_);
    }

    // =========================================================================
    // Schedule cost computation
    //
    // MaxBSP cost = work[0] + Sigma_{s=1}^{S-1} max(work[s], comm[s-1] * g) + (S-1) * L
    //
    // CRITICAL: g only multiplies comm, not the entire max.
    // =========================================================================

    using PreMoveCommDataT = PreMoveCommData<CommWeightT>;

    inline PreMoveCommDataT GetPreMoveCommData(const KlMove &move) { return commDs_.GetPreMoveCommData(move); }

    void ComputeSendReceiveDatastructures() { commDs_.ComputeCommDatastructures(0, activeSchedule_->NumSteps() - 1); }

    template <bool computeDatastructures = true>
    CostT ComputeScheduleCost() {
        if constexpr (computeDatastructures) {
            ComputeSendReceiveDatastructures();
        }

        const unsigned numSteps = activeSchedule_->NumSteps();
        const CostT g = static_cast<CostT>(instance_->CommunicationCosts());

        // Step 0: pure work (no overlapping communication)
        CostT totalCost = static_cast<CostT>(activeSchedule_->GetStepMaxWork(0));

        // Steps 1..S-1: max(work[s], comm[s-1] * g)
        for (unsigned step = 1; step < numSteps; step++) {
            const CostT work = static_cast<CostT>(activeSchedule_->GetStepMaxWork(step));
            const CostT comm = static_cast<CostT>(commDs_.StepMaxComm(step - 1)) * g;
            totalCost += std::max(work, comm);
        }

        if (numSteps > 1) {
            totalCost += static_cast<CostT>(numSteps - 1) * instance_->SynchronisationCosts();
        }

        return totalCost;
    }

    CostT ComputeScheduleCostTest() { return ComputeScheduleCost<false>(); }

    // =========================================================================
    // Datastructure update delegation
    // =========================================================================

    void UpdateDatastructureAfterMove(const KlMove &move, const unsigned startStep, const unsigned endStep) {
        commDs_.UpdateDatastructureAfterMove(move, startStep, endStep);
#ifdef KL_DEBUG_VALIDATE_COMM_DS
        static unsigned moveCounter_ = 0;
        moveCounter_++;
        if (!commDs_.ValidateCommDs(moveCounter_, move)) {
            std::cout << "[KL_DEBUG_VALIDATE_COMM_DS] *** DIVERGENCE at move #" << moveCounter_ << " — ABORTING ***" << std::endl;
        }
#endif
    }

    void SwapCommSteps(unsigned step1, unsigned step2) { commDs_.SwapSteps(step1, step2); }

    auto StepMaxComm(unsigned step) const { return commDs_.StepMaxComm(step); }

    const std::vector<unsigned> &GetLastAffectedCommSteps() const { return commDs_.GetLastAffectedCommSteps(); }

    bool NodeCommDependsOnChangedSteps(VertexType node, const std::unordered_set<unsigned> &changedSteps) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, unsigned>) {
            return false;
        } else {
            auto checkLambda = [&](VertexType n) -> bool {
                for (const auto [proc, val] : commDs_.nodeLambdaMap_.IterateProcEntries(n)) {
                    for (unsigned v : val) {
                        if (v > 0 && changedSteps.count(v - 1)) {
                            return true;
                        }
                    }
                }
                return false;
            };

            if (checkLambda(node)) {
                return true;
            }

            const auto &graph = instance_->GetComputationalDag();
            for (const auto &parent : graph.Parents(node)) {
                if (checkLambda(parent)) {
                    return true;
                }
            }
            return false;
        }
    }

    // =========================================================================
    // Step removal / insertion (lambda renumbering + send/recv fixup)
    // =========================================================================

    void UpdateLambdaAfterStepRemoval(unsigned removedStep) { commDs_.UpdateLambdaAfterStepRemoval(removedStep); }

    void FixupSendRecvAfterStepRemoval(unsigned removedStep, unsigned oldEndStep) {
        commDs_.FixupSendRecvAfterStepRemoval(removedStep, oldEndStep);
    }

    void UpdateLambdaAfterStepInsertion(unsigned insertedStep) { commDs_.UpdateLambdaAfterStepInsertion(insertedStep); }

    void FixupSendRecvAfterStepInsertion(unsigned insertedStep, unsigned startStep, unsigned endStep) {
        commDs_.FixupSendRecvAfterStepInsertion(insertedStep, startStep, endStep);
    }

    // --- Step removal cost delta ---

    void PrepareStepRemoval(unsigned /*removedStep*/) {}

    /// MaxBSP: coupled cost model prevents analytical delta computation.
    /// Full recomputation after datastructure updates. O(numSteps), called
    /// at most once per outer iteration.
    CostT ComputeStepRemovalCostDelta(unsigned /*removedStep*/, CostT currentCost) {
        return ComputeScheduleCost<false>() - currentCost;
    }

    // =========================================================================
    // Thread-local scratchpads (identical to BSP)
    // =========================================================================

    struct ScratchData {
        std::vector<FastDeltaTracker<CommWeightT>> sendDeltas_;
        std::vector<FastDeltaTracker<CommWeightT>> recvDeltas_;

        std::vector<unsigned> activeSteps_;
        std::vector<bool> stepIsActive_;

        std::vector<std::pair<unsigned, CommWeightT>> childCostBuffer_;

        unsigned lastNumProcs_ = 0;

        void Init(unsigned nSteps, unsigned nProcs) {
            if (sendDeltas_.size() < nSteps) {
                sendDeltas_.resize(nSteps);
                recvDeltas_.resize(nSteps);
                stepIsActive_.resize(nSteps, false);
                activeSteps_.reserve(nSteps);
            }

            // When the number of processors changes between uses (e.g. different
            // test cases sharing this static thread_local), the FastDeltaTracker
            // sentinel values (procDirtyIndex_[p] == old numProcs_) become stale
            // and IsDirty() returns incorrect results. Force a full reset.
            const bool procsChanged = (nProcs != lastNumProcs_);
            lastNumProcs_ = nProcs;

            for (auto &tracker : sendDeltas_) {
                if (procsChanged) {
                    tracker = FastDeltaTracker<CommWeightT>{};
                }
                tracker.Initialize(nProcs);
            }
            for (auto &tracker : recvDeltas_) {
                if (procsChanged) {
                    tracker = FastDeltaTracker<CommWeightT>{};
                }
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

    // =========================================================================
    // ComputeNewMaxComm — returns the ABSOLUTE new max comm at a step
    //
    // Unlike BSP's CalculateStepCostChange (which returns a delta), the coupled
    // formula needs the actual new max comm value: max(work, newMaxComm * g).
    // =========================================================================

    CommWeightT ComputeNewMaxComm(unsigned step,
                                  const FastDeltaTracker<CommWeightT> &deltaSend,
                                  const FastDeltaTracker<CommWeightT> &deltaRecv) {
        const CommWeightT oldMax = commDs_.StepMaxComm(step);
        const unsigned oldMaxCount = commDs_.StepMaxCommCount(step);

        CommWeightT newGlobalMax = 0;
        unsigned reducedMaxInstances = 0;

        for (unsigned proc : deltaSend.dirtyProcs_) {
            const CommWeightT delta = deltaSend.Get(proc);
            const CommWeightT currentVal = commDs_.StepProcSend(step, proc);
            const CommWeightT newVal = currentVal + delta;

            if (newVal > newGlobalMax) {
                newGlobalMax = newVal;
            }
            if (delta < 0 && currentVal == oldMax) {
                reducedMaxInstances++;
            }
        }

        for (unsigned proc : deltaRecv.dirtyProcs_) {
            const CommWeightT delta = deltaRecv.Get(proc);
            const CommWeightT currentVal = commDs_.StepProcReceive(step, proc);
            const CommWeightT newVal = currentVal + delta;

            if (newVal > newGlobalMax) {
                newGlobalMax = newVal;
            }
            if (delta < 0 && currentVal == oldMax) {
                reducedMaxInstances++;
            }
        }

        // Case 1: Some dirty entry exceeds the old max -> new max is the dirty max
        if (newGlobalMax >= oldMax) {
            return newGlobalMax;
        }

        // Case 2: Not all max-holders were reduced -> old max survives
        if (reducedMaxInstances < oldMaxCount) {
            return oldMax;
        }

        // Case 3: All max-holders reduced -> scan non-dirty for the true new max
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
        return std::max(newGlobalMax, maxNonDirty);
    }

    // =========================================================================
    // ComputeNodeAffinity — COUPLED work + comm evaluation
    //
    // This replaces the separate ComputeWorkAffinity + ComputeCommAffinity
    // pipeline used by BSP. The max(work, comm*g) coupling means we cannot
    // compute them independently and sum the results.
    //
    // For each candidate (pTo, sTo), we compute the total schedule cost change
    // when moving node from (nodeProc, nodeStep) to (pTo, sTo). The result is
    // stored in affinityTableNode[pTo][sToIdx].
    //
    // Convention: affinityTableNode[nodeProc][windowSize] = 0 (no cost change
    // from staying). gain = atn[current] - atn[candidate] = -(costChange).
    // =========================================================================

    template <typename AffinityTableT>
    void ComputeNodeAffinity(VertexType node,
                             AffinityTableT &affinityTableNode,
                             const CostT &penalty,
                             const CostT &reward,
                             const unsigned startStep,
                             const unsigned endStep) {
        // Initialize affinity table to zero
        for (auto &procVec : affinityTableNode) {
            std::fill(procVec.begin(), procVec.end(), CostT(0));
        }

        const unsigned nodeStep = activeSchedule_->AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule_->AssignedProcessor(node);
        const unsigned windowBound = EndIdx(nodeStep, endStep);
        const unsigned nodeStartIdx = StartIdx(nodeStep, startStep);
        const unsigned numSteps = activeSchedule_->NumSteps();
        const unsigned staleness = activeSchedule_->GetStaleness();

        // =================================================================
        // Part 1: Dependency constraint penalties/rewards (staleness-aware)
        //
        // For each edge, there are TWO violation thresholds depending on
        // whether the node moves to the same or a different processor
        // than the neighbor:
        //
        //   same-proc threshold:  just precedence (gap = 0)
        //   diff-proc threshold:  staleness offset (gap = staleness)
        //
        // For BSP (staleness=1) these thresholds differ by 1, so the
        // original single-exemption trick works. For staleness >= 2 the
        // two thresholds can span multiple positions, and we must handle
        // the gap zone explicitly.
        //
        // Decomposition:
        //   1. Apply penalty/reward to ALL procs for the diff-proc range
        //   2. Apply correction to the neighbor's proc for the gap zone
        //      between the same-proc and diff-proc boundaries.
        //
        // This yields the correct per-proc per-position prediction that
        // matches UpdateViolations in kl_active_schedule.hpp.
        // =================================================================

        // Clamp a signed index into the valid affinity table range
        auto ClampIdx = [&](int val) -> unsigned {
            return static_cast<unsigned>(std::max(static_cast<int>(nodeStartIdx), std::min(val, static_cast<int>(windowBound))));
        };

        // --- Children: node -> child ---
        //
        // Violation at candidate step s = nodeStep + idx - windowSize:
        //   same-proc:  s > childStep             ⟺  idx > windowSize + gap
        //   diff-proc:  s + staleness > childStep ⟺  idx > windowSize + gap - staleness
        //
        //   where gap = childStep - nodeStep (signed).
        //
        // NOT violated: penalty for positions that CREATE a new violation.
        //   ALL procs get +penalty for [diffCutoff, windowBound)
        //   childProc gets -penalty for [diffCutoff, sameCutoff)  (undo: childProc is clear there)
        //
        // VIOLATED: reward for positions that RESOLVE the existing violation.
        //   ALL procs get -reward for [nodeStartIdx, diffCutoff)
        //   childProc gets -reward for [diffCutoff, sameCutoff)  (extra: childProc resolves there too)

        for (const auto &target : instance_->GetComputationalDag().Children(node)) {
            const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
            const unsigned targetProc = activeSchedule_->AssignedProcessor(target);

            const int gap = static_cast<int>(targetStep) - static_cast<int>(nodeStep);

            // First idx that causes same-proc / diff-proc violation
            const unsigned sameCutoff = ClampIdx(static_cast<int>(windowSize) + gap + 1);
            const unsigned diffCutoff = ClampIdx(static_cast<int>(windowSize) + gap - static_cast<int>(staleness) + 1);

            const unsigned currThreshold = (targetProc != nodeProc) ? staleness : 0u;
            const bool currentlyViolated = (nodeStep + currThreshold > targetStep);

            if (!currentlyViolated) {
                // ALL procs get +penalty for diff-proc violation zone
                for (unsigned idx = diffCutoff; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
                }
                // childProc gets -penalty in the gap zone (same-proc is NOT violated there)
                if (IsCompatible(node, targetProc)) {
                    for (unsigned idx = diffCutoff; idx < sameCutoff; idx++) {
                        affinityTableNode[targetProc][idx] -= penalty;
                    }
                }
            } else {
                // ALL procs get -reward for diff-proc resolution zone
                for (unsigned idx = nodeStartIdx; idx < diffCutoff; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }
                // childProc gets extra -reward in the gap zone (same-proc resolves there too)
                if (IsCompatible(node, targetProc)) {
                    for (unsigned idx = diffCutoff; idx < sameCutoff; idx++) {
                        affinityTableNode[targetProc][idx] -= reward;
                    }
                }
            }
        }

        // --- Parents: parent -> node ---
        //
        // Violation at candidate step s = nodeStep + idx - windowSize:
        //   same-proc:  parentStep > s              ⟺  idx < windowSize - gapP
        //   diff-proc:  parentStep + staleness > s  ⟺  idx < windowSize - gapP + staleness
        //
        //   where gapP = nodeStep - parentStep (signed).
        //
        // NOT violated: penalty for positions that CREATE a new violation.
        //   ALL procs get +penalty for [nodeStartIdx, diffCutoffP)
        //   parentProc gets -penalty for [sameCutoffP, diffCutoffP)  (undo: parentProc is clear there)
        //
        // VIOLATED: reward for positions that RESOLVE the existing violation.
        //   ALL procs get -reward for [diffCutoffP, windowBound)
        //   parentProc gets -reward for [sameCutoffP, diffCutoffP)  (extra: parentProc resolves there too)

        for (const auto &source : instance_->GetComputationalDag().Parents(node)) {
            const unsigned sourceStep = activeSchedule_->AssignedSuperstep(source);
            const unsigned sourceProc = activeSchedule_->AssignedProcessor(source);

            const int gapP = static_cast<int>(nodeStep) - static_cast<int>(sourceStep);

            // First idx that is CLEAR for same-proc / diff-proc
            const unsigned sameCutoffP = ClampIdx(static_cast<int>(windowSize) - gapP);
            const unsigned diffCutoffP = ClampIdx(static_cast<int>(windowSize) - gapP + static_cast<int>(staleness));

            const unsigned currThreshold = (sourceProc != nodeProc) ? staleness : 0u;
            const bool currentlyViolated = (sourceStep + currThreshold > nodeStep);

            if (!currentlyViolated) {
                // ALL procs get +penalty for diff-proc violation zone
                for (unsigned idx = nodeStartIdx; idx < diffCutoffP; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
                }
                // parentProc gets -penalty in the gap zone (same-proc is NOT violated there)
                if (IsCompatible(node, sourceProc)) {
                    for (unsigned idx = sameCutoffP; idx < diffCutoffP; idx++) {
                        affinityTableNode[sourceProc][idx] -= penalty;
                    }
                }
            } else {
                // ALL procs get -reward for diff-proc resolution zone
                for (unsigned idx = diffCutoffP; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }
                // parentProc gets extra -reward in the gap zone (same-proc resolves there too)
                if (IsCompatible(node, sourceProc)) {
                    for (unsigned idx = sameCutoffP; idx < diffCutoffP; idx++) {
                        affinityTableNode[sourceProc][idx] -= reward;
                    }
                }
            }
        }

        // =================================================================
        // Part 2: Precompute work removal at nodeStep (once per node)
        //
        // workRemoval = how much maxWork at nodeStep decreases when node
        // leaves. Only non-zero if node is the sole max-work processor.
        // =================================================================

        const VertexWorkWeightT nodeWeight = graph_->VertexWorkWeight(node);
        const CostT maxWorkAtFrom = static_cast<CostT>(activeSchedule_->GetStepMaxWork(nodeStep));
        const CostT secondMaxWorkAtFrom = static_cast<CostT>(activeSchedule_->GetStepSecondMaxWork(nodeStep));
        const bool isSoleMaxProc
            = (activeSchedule_->GetStepMaxWorkProcessorCount()[nodeStep] == 1)
              && (maxWorkAtFrom == static_cast<CostT>(activeSchedule_->GetStepProcessorWork(nodeStep, nodeProc)));

        const CostT workRemoval = isSoleMaxProc ? std::min(static_cast<CostT>(nodeWeight), maxWorkAtFrom - secondMaxWorkAtFrom)
                                                : CostT(0);
        const CostT maxWorkAfterRemoval = maxWorkAtFrom - workRemoval;

        // =================================================================
        // Part 3: Phase 1 — comm removal deltas (constant across candidates)
        //
        // Remove node's outgoing and incoming communication from current
        // position. These deltas persist across all (pTo, sTo) iterations.
        // =================================================================

        static thread_local ScratchData scratch;
        scratch.Init(numSteps, instance_->NumberOfProcessors());
        scratch.ClearAll();

        const CommWeightT commWNode = graph_->VertexCommWeight(node);
        const auto &currentVecSchedule = activeSchedule_->GetVectorSchedule();
        const CostT g = static_cast<CostT>(instance_->CommunicationCosts());

        auto AddDelta = [&](bool isRecv, unsigned step, unsigned proc, CommWeightT val) {
            if (val == 0) {
                return;
            }
            if (step < numSteps) {
                scratch.MarkActive(step);
                if (isRecv) {
                    scratch.recvDeltas_[step].Add(proc, val);
                } else {
                    scratch.sendDeltas_[step].Add(proc, val);
                }
            }
        };

        struct DeltaAdapterT {
            decltype(AddDelta) &fn;

            void Add(bool isRecv, unsigned step, unsigned proc, CommWeightT v) { fn(isRecv, step, proc, v); }
        };

        DeltaAdapterT deltaAdapter{AddDelta};

        struct NegDeltaAdapterT {
            decltype(AddDelta) &fn;

            void Add(bool isRecv, unsigned step, unsigned proc, CommWeightT v) { fn(isRecv, step, proc, -v); }
        };

        NegDeltaAdapterT negDeltaAdapter{AddDelta};

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

        // =================================================================
        // Part 4: Phase 2 + Coupled evaluation per candidate (pTo, sTo)
        //
        // For each candidate, we:
        //   a) Apply Phase 2 comm deltas (incoming from parents, outgoing)
        //   b) Compute work addition at sTo
        //   c) Evaluate the coupled cost change via max(work, comm*g)
        //   d) Revert Phase 2 deltas
        // =================================================================

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

        struct ParentAddInfo {
            unsigned uProc;
            unsigned uStep;
            CommWeightT cost;
            typename CommPolicy::ValueType effectiveVal;
        };

        static thread_local std::vector<ParentAddInfo> parentAddInfos;

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
                const unsigned sTo = nodeStep + sToIdx - windowSize;

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

                // =========================================================
                // Coupled evaluation: max(work[ws], comm[cs] * g)
                //
                // comm step cs pairs with work step ws = cs + 1.
                // Step 0 is pure work (no paired comm step before it).
                // =========================================================

                // Compute work addition for this candidate
                CostT workAdd;
                if (sTo == nodeStep) {
                    // Same step: removal already happened conceptually
                    if (pTo == nodeProc) {
                        // Same proc, same step: node re-placed where it was.
                        // procWork already includes node. After removal, maxWork drops
                        // to maxWorkAfterRemoval. After re-add, procWork is unchanged.
                        workAdd = std::max(
                            CostT(0), static_cast<CostT>(activeSchedule_->GetStepProcessorWork(sTo, pTo)) - maxWorkAfterRemoval);
                    } else {
                        // Different proc, same step: add nodeWeight to pTo's work
                        workAdd = std::max(CostT(0),
                                           static_cast<CostT>(activeSchedule_->GetStepProcessorWork(sTo, pTo))
                                               + static_cast<CostT>(nodeWeight) - maxWorkAfterRemoval);
                    }
                } else {
                    const CostT maxWorkAtTo = static_cast<CostT>(activeSchedule_->GetStepMaxWork(sTo));
                    workAdd = std::max(CostT(0),
                                       static_cast<CostT>(activeSchedule_->GetStepProcessorWork(sTo, pTo))
                                           + static_cast<CostT>(nodeWeight) - maxWorkAtTo);
                }

                CostT totalChange = 0;
                bool fromCovered = false;
                bool toCovered = false;

                // --- Iterate dirty comm steps ---
                for (unsigned cs : scratch.activeSteps_) {
                    if (scratch.sendDeltas_[cs].dirtyProcs_.empty() && scratch.recvDeltas_[cs].dirtyProcs_.empty()) {
                        continue;
                    }

                    const unsigned ws = cs + 1;    // paired work step

                    if (ws >= numSteps) {
                        continue;
                    }

                    // Work delta at the paired work step
                    CostT wd = 0;
                    if (ws == nodeStep) {
                        wd -= workRemoval;
                        fromCovered = true;
                    }
                    if (ws == sTo) {
                        wd += workAdd;
                        toCovered = true;
                    }

                    const CommWeightT newMaxComm = ComputeNewMaxComm(cs, scratch.sendDeltas_[cs], scratch.recvDeltas_[cs]);
                    const CostT oldWork = static_cast<CostT>(activeSchedule_->GetStepMaxWork(ws));
                    const CostT oldMaxComm = static_cast<CostT>(commDs_.StepMaxComm(cs));

                    const CostT oldContrib = std::max(oldWork, oldMaxComm * g);
                    const CostT newContrib = std::max(oldWork + wd, static_cast<CostT>(newMaxComm) * g);

                    totalChange += newContrib - oldContrib;
                }

                // --- Work-only corrections for uncovered steps ---
                //
                // A work change at step ws pairs with comm step cs = ws - 1.
                // If cs wasn't dirty, we still need to account for the work
                // change against the unchanged comm.

                if (!fromCovered && nodeStep > 0) {
                    const unsigned cs = nodeStep - 1;
                    const CostT oldWork = maxWorkAtFrom;
                    const CostT oldComm = static_cast<CostT>(commDs_.StepMaxComm(cs)) * g;
                    CostT newWork = oldWork - workRemoval;

                    // If sTo == nodeStep and not yet covered, handle addition too
                    if (sTo == nodeStep && !toCovered) {
                        newWork += workAdd;
                        toCovered = true;
                    }

                    totalChange += std::max(newWork, oldComm) - std::max(oldWork, oldComm);
                    fromCovered = true;
                }

                if (!fromCovered && nodeStep == 0) {
                    // Step 0 is pure work, no paired comm step
                    CostT newWork = maxWorkAtFrom - workRemoval;

                    if (sTo == 0 && !toCovered) {
                        newWork += workAdd;
                        toCovered = true;
                    }

                    totalChange += newWork - maxWorkAtFrom;
                    fromCovered = true;
                }

                if (!toCovered && sTo > 0) {
                    const unsigned cs = sTo - 1;
                    const CostT oldWork = static_cast<CostT>(activeSchedule_->GetStepMaxWork(sTo));
                    const CostT oldComm = static_cast<CostT>(commDs_.StepMaxComm(cs)) * g;
                    const CostT newWork = oldWork + workAdd;

                    totalChange += std::max(newWork, oldComm) - std::max(oldWork, oldComm);
                    toCovered = true;
                }

                if (!toCovered && sTo == 0) {
                    // Step 0 is pure work
                    const CostT oldWork = static_cast<CostT>(activeSchedule_->GetStepMaxWork(0));
                    totalChange += (oldWork + workAdd) - oldWork;
                    toCovered = true;
                }

                affinityTableNode[pTo][sToIdx] += totalChange;

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
};

}    // namespace osp
