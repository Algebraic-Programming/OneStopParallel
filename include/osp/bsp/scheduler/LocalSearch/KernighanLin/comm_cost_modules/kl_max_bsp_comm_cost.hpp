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

    constexpr static bool coupledWorkComm_ = true;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule_;
    CompatibleProcessorRange<GraphT> *procRange_;
    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    MaxCommDatastructure<GraphT, CostT, KlActiveSchedule<GraphT, CostT, MemoryConstraintT>, CommPolicy> commDs_;

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

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &pRange) {
        activeSchedule_ = &sched;
        procRange_ = &pRange;
        instance_ = &sched.GetInstance();
        graph_ = &instance_->GetComputationalDag();

        commDs_.Initialize(*activeSchedule_);
    }

    void ComputeSendReceiveDatastructures() { commDs_.ComputeCommDatastructures(0, activeSchedule_->NumSteps() - 1); }

    template <bool computeDatastructures = true>
    CostT ComputeScheduleCost() {
        if constexpr (computeDatastructures) {
            ComputeSendReceiveDatastructures();
        }

        const unsigned numSteps = activeSchedule_->NumSteps();
        const CostT g = static_cast<CostT>(instance_->CommunicationCosts());

        CostT totalCost = static_cast<CostT>(activeSchedule_->GetStepMaxWork(0));

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

    void UpdateLambdaAfterStepRemoval(unsigned removedStep, unsigned startStep, unsigned endStep) {
        commDs_.UpdateLambdaAfterStepRemoval(removedStep, startStep, endStep);
    }

    void UpdateLambdaAfterStepRemoval(unsigned removedStep) { commDs_.UpdateLambdaAfterStepRemoval(removedStep); }

    void FixupSendRecvAfterStepRemoval(unsigned removedStep, unsigned oldEndStep) {
        commDs_.FixupSendRecvAfterStepRemoval(removedStep, oldEndStep);
    }

    void UpdateLambdaAfterStepInsertion(unsigned insertedStep, unsigned startStep, unsigned endStep) {
        commDs_.UpdateLambdaAfterStepInsertion(insertedStep, startStep, endStep);
    }

    void UpdateLambdaAfterStepInsertion(unsigned insertedStep) { commDs_.UpdateLambdaAfterStepInsertion(insertedStep); }

    void FixupSendRecvAfterStepInsertion(unsigned insertedStep, unsigned startStep, unsigned endStep) {
        commDs_.FixupSendRecvAfterStepInsertion(insertedStep, startStep, endStep);
    }

    void PrepareStepRemoval(unsigned /*removedStep*/) {}

    CostT ComputeStepRemovalCostDelta(unsigned /*removedStep*/, CostT currentCost) {
        return ComputeScheduleCost<false>() - currentCost;
    }

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

        auto ClampIdx = [&](int val) -> unsigned {
            return static_cast<unsigned>(std::max(static_cast<int>(nodeStartIdx), std::min(val, static_cast<int>(windowBound))));
        };

        for (const auto &target : instance_->GetComputationalDag().Children(node)) {
            const unsigned targetStep = activeSchedule_->AssignedSuperstep(target);
            const unsigned targetProc = activeSchedule_->AssignedProcessor(target);

            const int gap = static_cast<int>(targetStep) - static_cast<int>(nodeStep);

            const unsigned sameCutoff = ClampIdx(static_cast<int>(windowSize) + gap + 1);
            const unsigned diffCutoff = ClampIdx(static_cast<int>(windowSize) + gap - static_cast<int>(staleness) + 1);

            const unsigned currThreshold = (targetProc != nodeProc) ? staleness : 0u;
            const bool currentlyViolated = (nodeStep + currThreshold > targetStep);

            if (!currentlyViolated) {
                for (unsigned idx = diffCutoff; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
                }
                if (IsCompatible(node, targetProc)) {
                    for (unsigned idx = diffCutoff; idx < sameCutoff; idx++) {
                        affinityTableNode[targetProc][idx] -= penalty;
                    }
                }
            } else {
                for (unsigned idx = nodeStartIdx; idx < diffCutoff; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }
                if (IsCompatible(node, targetProc)) {
                    for (unsigned idx = diffCutoff; idx < sameCutoff; idx++) {
                        affinityTableNode[targetProc][idx] -= reward;
                    }
                }
            }
        }

        for (const auto &source : instance_->GetComputationalDag().Parents(node)) {
            const unsigned sourceStep = activeSchedule_->AssignedSuperstep(source);
            const unsigned sourceProc = activeSchedule_->AssignedProcessor(source);

            const int gapP = static_cast<int>(nodeStep) - static_cast<int>(sourceStep);

            const unsigned sameCutoffP = ClampIdx(static_cast<int>(windowSize) - gapP);
            const unsigned diffCutoffP = ClampIdx(static_cast<int>(windowSize) - gapP + static_cast<int>(staleness));

            const unsigned currThreshold = (sourceProc != nodeProc) ? staleness : 0u;
            const bool currentlyViolated = (sourceStep + currThreshold > nodeStep);

            if (!currentlyViolated) {
                for (unsigned idx = nodeStartIdx; idx < diffCutoffP; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] += penalty;
                    }
                }

                if (IsCompatible(node, sourceProc)) {
                    for (unsigned idx = sameCutoffP; idx < diffCutoffP; idx++) {
                        affinityTableNode[sourceProc][idx] -= penalty;
                    }
                }
            } else {
                for (unsigned idx = diffCutoffP; idx < windowBound; idx++) {
                    for (const unsigned p : procRange_->CompatibleProcessorsVertex(node)) {
                        affinityTableNode[p][idx] -= reward;
                    }
                }

                if (IsCompatible(node, sourceProc)) {
                    for (unsigned idx = sameCutoffP; idx < diffCutoffP; idx++) {
                        affinityTableNode[sourceProc][idx] -= reward;
                    }
                }
            }
        }

        const VertexWorkWeightT nodeWeight = graph_->VertexWorkWeight(node);
        const CostT maxWorkAtFrom = static_cast<CostT>(activeSchedule_->GetStepMaxWork(nodeStep));
        const CostT secondMaxWorkAtFrom = static_cast<CostT>(activeSchedule_->GetStepSecondMaxWork(nodeStep));
        const bool isSoleMaxProc
            = (activeSchedule_->GetStepMaxWorkProcessorCount()[nodeStep] == 1)
              && (maxWorkAtFrom == static_cast<CostT>(activeSchedule_->GetStepProcessorWork(nodeStep, nodeProc)));

        const CostT workRemoval = isSoleMaxProc ? std::min(static_cast<CostT>(nodeWeight), maxWorkAtFrom - secondMaxWorkAtFrom)
                                                : CostT(0);
        const CostT maxWorkAfterRemoval = maxWorkAtFrom - workRemoval;
        const CostT g = static_cast<CostT>(instance_->CommunicationCosts());

        auto maxBspEvaluator
            = [&](unsigned pTo, unsigned /*sToIdx*/, unsigned sTo, CommDeltaScratchData<CommWeightT> &scratch) -> CostT {
            CostT workAdd;
            if (sTo == nodeStep) {
                if (pTo == nodeProc) {
                    workAdd = std::max(CostT(0),
                                       static_cast<CostT>(activeSchedule_->GetStepProcessorWork(sTo, pTo)) - maxWorkAfterRemoval);
                } else {
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

            for (unsigned cs : scratch.activeSteps_) {
                if (scratch.sendDeltas_[cs].dirtyProcs_.empty() && scratch.recvDeltas_[cs].dirtyProcs_.empty()) {
                    continue;
                }

                const unsigned ws = cs + 1;

                if (ws >= numSteps) {
                    continue;
                }

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

            if (!fromCovered && nodeStep > 0) {
                const unsigned cs = nodeStep - 1;
                const CostT oldWork = maxWorkAtFrom;
                const CostT oldComm = static_cast<CostT>(commDs_.StepMaxComm(cs)) * g;
                CostT newWork = oldWork - workRemoval;

                if (sTo == nodeStep && !toCovered) {
                    newWork += workAdd;
                    toCovered = true;
                }

                totalChange += std::max(newWork, oldComm) - std::max(oldWork, oldComm);
                fromCovered = true;
            }

            if (!fromCovered && nodeStep == 0) {
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
