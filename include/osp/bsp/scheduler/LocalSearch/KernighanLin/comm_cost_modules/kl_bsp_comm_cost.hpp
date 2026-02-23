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
#include "../kl_work_affinity.hpp"
#include "comm_cost_policies.hpp"
#include "kl_comm_delta_helper.hpp"
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
    constexpr static bool coupledWorkComm_ = false;

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
            // std::abort();
        }
#endif
    }

    void SwapCommSteps(unsigned step1, unsigned step2) { commDs_.SwapSteps(step1, step2); }

    auto StepMaxComm(unsigned step) const { return commDs_.StepMaxComm(step); }

    /// Returns the steps where send/recv arrays changed during the last move.
    /// For Lazy/Buffered, these include the min(child_steps)-1 comm steps.
    const std::vector<unsigned> &GetLastAffectedCommSteps() const { return commDs_.GetLastAffectedCommSteps(); }

    /// Check if any comm step that node's gain depends on is in changedSteps.
    ///
    /// For Lazy/Buffered, CalculateDeltaRemove/Add produce deltas at positions
    /// v-1 where v is ANY step value in a lambda entry (not just the minimum).
    /// For example, CalculateDeltaRemove with val=[19,30,45] and nodeStep=19
    /// produces deltas at step 18 (min-1) AND step 29 (nextMin-1).
    ///
    /// So the complete dependency set is: {v-1 : v ∈ lambda[N or parent][q]}.
    /// We check if any of these fall in changedSteps.
    bool NodeCommDependsOnChangedSteps(VertexType node, const std::unordered_set<unsigned> &changedSteps) {
        // For Eager (ValueType=unsigned, just a count): comm is always at node
        // steps, fully covered by window and parent-position checks.
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, unsigned>) {
            return false;
        } else {
            // For Lazy/Buffered (ValueType=vector<unsigned>): check if any
            // step value v in any lambda entry has (v-1) in changedSteps.
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

            // Check node's own outgoing comm steps
            if (checkLambda(node)) {
                return true;
            }

            // Check all parents' comm steps
            const auto &graph = instance_->GetComputationalDag();
            for (const auto &parent : graph.Parents(node)) {
                if (checkLambda(parent)) {
                    return true;
                }
            }
            return false;
        }
    }

    void UpdateLambdaAfterStepRemoval(unsigned removedStep) { commDs_.UpdateLambdaAfterStepRemoval(removedStep); }

    void FixupSendRecvAfterStepRemoval(unsigned removedStep, unsigned oldEndStep) {
        commDs_.FixupSendRecvAfterStepRemoval(removedStep, oldEndStep);
    }

    void UpdateLambdaAfterStepInsertion(unsigned insertedStep) { commDs_.UpdateLambdaAfterStepInsertion(insertedStep); }

    void FixupSendRecvAfterStepInsertion(unsigned insertedStep, unsigned startStep, unsigned endStep) {
        commDs_.FixupSendRecvAfterStepInsertion(insertedStep, startStep, endStep);
    }

    /// Unified entry point called by the base class.
    /// Additive cost: compute work affinities, then layer comm affinities on top.
    template <typename AffinityTableT>
    void ComputeNodeAffinity(VertexType node,
                             AffinityTableT &affinityTableNode,
                             const CostT &penalty,
                             const CostT &reward,
                             const unsigned startStep,
                             const unsigned endStep) {
        ComputeWorkAffinity<windowSize>(node, affinityTableNode, *activeSchedule_, *graph_, *procRange_, startStep, endStep);
        ComputeCommAffinity(node, affinityTableNode, penalty, reward, startStep, endStep);
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

        // ========== Violation handling (staleness = 1) ==========

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

        // ========== Comm delta computation (shared scaffold + BSP evaluator) ==========
        //
        // BSP cost = Σ maxComm[step] * g per step (additive, decoupled from work).
        // The evaluator sums the per-step max-comm deltas and scales by g.

        const CostT g = instance_->CommunicationCosts();

        auto bspEvaluator
            = [&](unsigned /*pTo*/, unsigned /*sToIdx*/, unsigned /*sTo*/, CommDeltaScratchData<CommWeightT> &scratch) -> CostT {
            CostT totalChange = 0;
            for (unsigned step : scratch.activeSteps_) {
                if (!scratch.sendDeltas_[step].dirtyProcs_.empty() || !scratch.recvDeltas_[step].dirtyProcs_.empty()) {
                    totalChange += commDs_.ComputeNewMaxComm(step, scratch.sendDeltas_[step], scratch.recvDeltas_[step])
                                   - commDs_.StepMaxComm(step);
                }
            }
            return totalChange * g;
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
                                                                          activeSchedule_->NumSteps(),
                                                                          windowSize,
                                                                          bspEvaluator);
    }
};

}    // namespace osp
