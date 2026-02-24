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
#include "../kl_improver_base.hpp"
#include "comm_cost_policies.hpp"
#include "kl_comm_delta_helper.hpp"
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

    void UpdateLambdaAfterStepRemoval(unsigned removedStep, unsigned endStep) {
        commDs_.UpdateLambdaAfterStepRemoval(removedStep, endStep);
    }

    void UpdateLambdaAfterStepRemoval(unsigned removedStep) { commDs_.UpdateLambdaAfterStepRemoval(removedStep); }

    void FixupSendRecvAfterStepRemoval(unsigned removedStep, unsigned oldEndStep) {
        commDs_.FixupSendRecvAfterStepRemoval(removedStep, oldEndStep);
    }

    void UpdateLambdaAfterStepInsertion(unsigned insertedStep, unsigned endStep) {
        commDs_.UpdateLambdaAfterStepInsertion(insertedStep, endStep);
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
        // Part 3: Comm delta computation (shared scaffold + MaxBSP evaluator)
        //
        // The shared helper handles Phase 1 (removal) and Phase 2 (apply/
        // revert per candidate). The evaluator implements the coupled
        // max(work, comm*g) cost model specific to MaxBSP.
        // =================================================================

        const CostT g = static_cast<CostT>(instance_->CommunicationCosts());

        auto maxBspEvaluator
            = [&](unsigned pTo, unsigned /*sToIdx*/, unsigned sTo, CommDeltaScratchData<CommWeightT> &scratch) -> CostT {
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
                    workAdd = std::max(CostT(0),
                                       static_cast<CostT>(activeSchedule_->GetStepProcessorWork(sTo, pTo)) - maxWorkAfterRemoval);
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

                const CommWeightT newMaxComm = commDs_.ComputeNewMaxComm(cs, scratch.sendDeltas_[cs], scratch.recvDeltas_[cs]);
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

            return totalChange;
        };

        ComputeCommAffinityDeltas<GraphT, CostT, CommWeightT, CommPolicy>(node,
                                                                          affinityTableNode,
                                                                          commDs_,
                                                                          *activeSchedule_,
                                                                          *graph_,
                                                                          *instance_,
                                                                          *procRange_,
                                                                          nodeStep,
                                                                          nodeProc,
                                                                          nodeStartIdx,
                                                                          windowBound,
                                                                          numSteps,
                                                                          windowSize,
                                                                          startStep,
                                                                          endStep,
                                                                          maxBspEvaluator);
    }
};

}    // namespace osp
