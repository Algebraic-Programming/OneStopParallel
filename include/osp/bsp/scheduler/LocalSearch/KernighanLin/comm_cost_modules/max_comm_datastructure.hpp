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
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include "FastDeltaTacker.hpp"
#include "comm_cost_policies.hpp"
#include "generic_lambda_container.hpp"
#include "lambda_container.hpp"
#include "osp/bsp/model/BspInstance.hpp"

namespace osp {

template <typename GraphT, typename CostT, typename KlActiveScheduleT, typename CommPolicy = EagerCommCostPolicy>
struct MaxCommDatastructure {
    using CommWeightT = VCommwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;

    const BspInstance<GraphT> *instance_;
    const KlActiveScheduleT *activeSchedule_;

    std::vector<std::vector<CommWeightT>> stepProcSend_;
    std::vector<std::vector<CommWeightT>> stepProcReceive_;

    std::vector<CommWeightT> stepMaxCommCache_;
    std::vector<CommWeightT> stepSecondMaxCommCache_;
    std::vector<unsigned> stepMaxCommCountCache_;

    CommWeightT maxCommWeight_ = 0;

    using ContainerType = typename std::conditional<std::is_same<typename CommPolicy::ValueType, unsigned>::value,
                                                    LambdaVectorContainer<VertexType>,
                                                    GenericLambdaVectorContainer<VertexType, typename CommPolicy::ValueType>>::type;

    ContainerType nodeLambdaMap_;

    std::vector<unsigned> affectedStepsList_;
    std::vector<bool> stepIsAffected_;

    inline CommWeightT StepProcSend(unsigned step, unsigned proc) const { return stepProcSend_[step][proc]; }

    inline CommWeightT &StepProcSend(unsigned step, unsigned proc) { return stepProcSend_[step][proc]; }

    inline CommWeightT StepProcReceive(unsigned step, unsigned proc) const { return stepProcReceive_[step][proc]; }

    inline CommWeightT &StepProcReceive(unsigned step, unsigned proc) { return stepProcReceive_[step][proc]; }

    inline CommWeightT StepMaxComm(unsigned step) const { return stepMaxCommCache_[step]; }

    inline CommWeightT StepSecondMaxComm(unsigned step) const { return stepSecondMaxCommCache_[step]; }

    inline unsigned StepMaxCommCount(unsigned step) const { return stepMaxCommCountCache_[step]; }

    inline const std::vector<unsigned> &GetLastAffectedCommSteps() const { return affectedStepsList_; }

    CommWeightT ComputeNewMaxComm(unsigned step,
                                  const FastDeltaTracker<CommWeightT> &deltaSend,
                                  const FastDeltaTracker<CommWeightT> &deltaRecv) const {
        const CommWeightT oldMax = stepMaxCommCache_[step];
        const unsigned oldMaxCount = stepMaxCommCountCache_[step];

        CommWeightT newGlobalMax = 0;
        unsigned reducedMaxInstances = 0;

        for (unsigned proc : deltaSend.dirtyProcs_) {
            const CommWeightT delta = deltaSend.Get(proc);
            const CommWeightT currentVal = stepProcSend_[step][proc];
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
            const CommWeightT currentVal = stepProcReceive_[step][proc];
            const CommWeightT newVal = currentVal + delta;

            if (newVal > newGlobalMax) {
                newGlobalMax = newVal;
            }
            if (delta < 0 && currentVal == oldMax) {
                reducedMaxInstances++;
            }
        }

        if (newGlobalMax >= oldMax) {
            return newGlobalMax;
        }

        if (reducedMaxInstances < oldMaxCount) {
            return oldMax;
        }

        CommWeightT maxNonDirty = 0;
        const unsigned numProcs = instance_->NumberOfProcessors();
        for (unsigned p = 0; p < numProcs; ++p) {
            if (!deltaSend.IsDirty(p)) {
                maxNonDirty = std::max(maxNonDirty, stepProcSend_[step][p]);
            }
            if (!deltaRecv.IsDirty(p)) {
                maxNonDirty = std::max(maxNonDirty, stepProcReceive_[step][p]);
            }
        }
        return std::max(newGlobalMax, maxNonDirty);
    }

    inline void Initialize(KlActiveScheduleT &klSched) {
        activeSchedule_ = &klSched;
        instance_ = &activeSchedule_->GetInstance();
        const unsigned numSteps = activeSchedule_->NumSteps();
        const unsigned numProcs = instance_->NumberOfProcessors();
        maxCommWeight_ = 0;

        stepProcSend_.assign(numSteps, std::vector<CommWeightT>(numProcs, 0));
        stepProcReceive_.assign(numSteps, std::vector<CommWeightT>(numProcs, 0));

        stepMaxCommCache_.assign(numSteps, 0);
        stepSecondMaxCommCache_.assign(numSteps, 0);
        stepMaxCommCountCache_.assign(numSteps, 0);

        nodeLambdaMap_.Initialize(instance_->GetComputationalDag().NumVertices(), numProcs);

        // Initialize scratchpad
        stepIsAffected_.assign(numSteps, false);
        affectedStepsList_.reserve(numSteps);
    }

    inline void Clear() {
        stepProcSend_.clear();
        stepProcReceive_.clear();
        stepMaxCommCache_.clear();
        stepSecondMaxCommCache_.clear();
        stepMaxCommCountCache_.clear();
        nodeLambdaMap_.clear();
        affectedStepsList_.clear();
        stepIsAffected_.clear();
    }

    inline void ArrangeSuperstepCommData(const unsigned step) {
        CommWeightT maxSend = 0;
        CommWeightT secondMaxSend = 0;
        unsigned maxSendCount = 0;

        const auto &sends = stepProcSend_[step];
        for (const auto val : sends) {
            if (val > maxSend) {
                secondMaxSend = maxSend;
                maxSend = val;
                maxSendCount = 1;
            } else if (val == maxSend) {
                maxSendCount++;
            } else if (val > secondMaxSend) {
                secondMaxSend = val;
            }
        }

        CommWeightT maxReceive = 0;
        CommWeightT secondMaxReceive = 0;
        unsigned maxReceiveCount = 0;

        const auto &receives = stepProcReceive_[step];
        for (const auto val : receives) {
            if (val > maxReceive) {
                secondMaxReceive = maxReceive;
                maxReceive = val;
                maxReceiveCount = 1;
            } else if (val == maxReceive) {
                maxReceiveCount++;
            } else if (val > secondMaxReceive) {
                secondMaxReceive = val;
            }
        }

        const CommWeightT globalMax = std::max(maxSend, maxReceive);
        stepMaxCommCache_[step] = globalMax;

        unsigned globalCount = 0;
        if (maxSend == globalMax) {
            globalCount += maxSendCount;
        }
        if (maxReceive == globalMax) {
            globalCount += maxReceiveCount;
        }
        stepMaxCommCountCache_[step] = globalCount;

        CommWeightT candSend = (maxSend == globalMax) ? secondMaxSend : maxSend;
        CommWeightT candRecv = (maxReceive == globalMax) ? secondMaxReceive : maxReceive;

        stepSecondMaxCommCache_[step] = std::max(candSend, candRecv);
    }

    void RecomputeMaxSendReceive(unsigned step) { ArrangeSuperstepCommData(step); }

    void UpdateDatastructureAfterMove(const KlMove &move, unsigned startStep, unsigned endStep) {
        const auto &graph = instance_->GetComputationalDag();

        for (unsigned step : affectedStepsList_) {
            if (step < stepIsAffected_.size()) {
                stepIsAffected_[step] = false;
            }
        }
        affectedStepsList_.clear();

        auto MarkStep = [&](unsigned step) -> bool {
            if (step >= startStep && step <= endStep && step < stepIsAffected_.size()) {
                if (!stepIsAffected_[step]) {
                    stepIsAffected_[step] = true;
                    affectedStepsList_.push_back(step);
                }
                return true;
            }
            return false;
        };

        const VertexType node = move.node_;
        const unsigned fromStep = move.fromStep_;
        const unsigned toStep = move.toStep_;
        const unsigned fromProc = move.fromProc_;
        const unsigned toProc = move.toProc_;
        const CommWeightT commWNode = graph.VertexCommWeight(node);

        if (fromStep != toStep) {
            for (const auto [proc, val] : nodeLambdaMap_.IterateProcEntries(node)) {
                if (proc != fromProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::RemoveOutgoingComm(*this, cost, fromStep, fromProc, proc, val, MarkStep);
                    }
                }

                if (proc != toProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::AddOutgoingComm(*this, cost, toStep, toProc, proc, val, MarkStep);
                    }
                }
            }

        } else if (fromProc != toProc) {
            for (const auto [proc, val] : nodeLambdaMap_.IterateProcEntries(node)) {
                // Remove Old (Sender: fromProc, Receiver: proc)
                if (proc != fromProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::RemoveOutgoingComm(*this, cost, fromStep, fromProc, proc, val, MarkStep);
                    }
                }

                // Add New (Sender: toProc, Receiver: proc)
                if (proc != toProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::AddOutgoingComm(*this, cost, fromStep, toProc, proc, val, MarkStep);
                    }
                }
            }
        }

        for (const auto &parent : graph.Parents(node)) {
            const unsigned parentStep = activeSchedule_->AssignedSuperstep(parent);

            // Skip parents outside our step range (thread safety)
            if (parentStep < startStep || parentStep > endStep) {
                continue;
            }

            // Fast boundary check
            if (parentStep >= stepProcSend_.size()) {
                continue;
            }

            const unsigned parentProc = activeSchedule_->AssignedProcessor(parent);
            const CommWeightT commWParent = graph.VertexCommWeight(parent);

            auto &val = nodeLambdaMap_.GetProcEntry(parent, fromProc);
            const bool removedFromProc = CommPolicy::RemoveChild(val, fromStep);

            if (removedFromProc) {
                if (fromProc != parentProc) {
                    const CommWeightT cost = commWParent * instance_->SendCosts(parentProc, fromProc);
                    if (cost > 0) {
                        CommPolicy::UnattributeCommunication(
                            *this, cost, parentStep, parentProc, fromProc, fromStep, val, MarkStep);
                    }
                }
            }

            auto &valTo = nodeLambdaMap_.GetProcEntry(parent, toProc);
            const bool addedToProc = CommPolicy::AddChild(valTo, toStep);

            if (addedToProc) {
                if (toProc != parentProc) {
                    const CommWeightT cost = commWParent * instance_->SendCosts(parentProc, toProc);
                    if (cost > 0) {
                        CommPolicy::AttributeCommunication(*this, cost, parentStep, parentProc, toProc, toStep, valTo, MarkStep);
                    }
                }
            }
        }

        for (unsigned step : affectedStepsList_) {
            ArrangeSuperstepCommData(step);
        }
    }

    void UpdateLambdaAfterStepRemoval(unsigned removedStep, unsigned startStep, unsigned endStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            const auto &psv = activeSchedule_->GetSetSchedule().GetProcessorStepVertices();
            endStep = std::min(endStep, static_cast<unsigned>(psv.size() - 1));

            if (endStep <= removedStep) {
                return;    // Nothing shifted
            }

            const auto &graph = instance_->GetComputationalDag();
            const unsigned numProcs = instance_->NumberOfProcessors();

            for (unsigned newStep = endStep - 1;; newStep--) {
                const unsigned oldStep = newStep + 1;
                for (unsigned p = 0; p < numProcs; p++) {
                    for (const auto &node : psv[newStep][p]) {
                        for (const auto &parent : graph.Parents(node)) {
                            const unsigned parentStep = activeSchedule_->AssignedSuperstep(parent);
                            if (parentStep < startStep || parentStep > endStep) {
                                continue;    // Thread safety: skip parents in other threads' ranges
                            }
                            auto &val = nodeLambdaMap_.GetProcEntry(parent, p);
                            for (auto &entry : val) {
                                if (entry == oldStep) {
                                    entry = newStep;
                                    break;    // Each child appears once per parent-proc pair
                                }
                            }
                        }
                    }
                }
                if (newStep == removedStep) {
                    break;
                }
            }
        }
    }

    void UpdateLambdaAfterStepRemoval(unsigned removedStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            UpdateLambdaAfterStepRemoval(removedStep, 0, std::numeric_limits<unsigned>::max());
        }
    }

    void FixupSendRecvAfterStepRemoval(unsigned removedStep, unsigned oldEndStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            if (removedStep == 0) {
                std::fill(stepProcSend_[oldEndStep].begin(), stepProcSend_[oldEndStep].end(), 0);
                std::fill(stepProcReceive_[oldEndStep].begin(), stepProcReceive_[oldEndStep].end(), 0);
                ArrangeSuperstepCommData(oldEndStep);
                return;
            }
            const unsigned numProcs = static_cast<unsigned>(stepProcSend_[0].size());
            for (unsigned p = 0; p < numProcs; p++) {
                stepProcSend_[removedStep - 1][p] += stepProcSend_[oldEndStep][p];
                stepProcReceive_[removedStep - 1][p] += stepProcReceive_[oldEndStep][p];
                // DON'T clear oldEndStep — it serves as backup for insertion reversal
            }
            ArrangeSuperstepCommData(removedStep - 1);
        }
    }

    void UpdateLambdaAfterStepInsertion(unsigned insertedStep, unsigned startStep, unsigned endStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            const auto &psv = activeSchedule_->GetSetSchedule().GetProcessorStepVertices();
            endStep = std::min(endStep, static_cast<unsigned>(psv.size() - 1));

            const auto &graph = instance_->GetComputationalDag();
            const unsigned numProcs = instance_->NumberOfProcessors();

            for (unsigned newStep = endStep; newStep > insertedStep; newStep--) {
                const unsigned oldStep = newStep - 1;
                for (unsigned p = 0; p < numProcs; p++) {
                    for (const auto &node : psv[newStep][p]) {
                        for (const auto &parent : graph.Parents(node)) {
                            const unsigned parentStep = activeSchedule_->AssignedSuperstep(parent);
                            if (parentStep < startStep || parentStep > endStep) {
                                continue;
                            }
                            auto &val = nodeLambdaMap_.GetProcEntry(parent, p);
                            for (auto &entry : val) {
                                if (entry == oldStep) {
                                    entry = newStep;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void UpdateLambdaAfterStepInsertion(unsigned insertedStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            UpdateLambdaAfterStepInsertion(insertedStep, 0, std::numeric_limits<unsigned>::max());
        }
    }

    void FixupSendRecvAfterStepInsertion(unsigned insertedStep, unsigned startStep, unsigned endStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            if (insertedStep == 0) {
                ComputeCommDatastructures(startStep, endStep);
                return;
            }
            const unsigned numProcs = static_cast<unsigned>(stepProcSend_[0].size());
            for (unsigned p = 0; p < numProcs; p++) {
                stepProcSend_[insertedStep - 1][p] -= stepProcSend_[insertedStep][p];
                stepProcReceive_[insertedStep - 1][p] -= stepProcReceive_[insertedStep][p];
            }
            ArrangeSuperstepCommData(insertedStep - 1);
            ArrangeSuperstepCommData(insertedStep);
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcSend_[step1], stepProcSend_[step2]);
        std::swap(stepProcReceive_[step1], stepProcReceive_[step2]);
        std::swap(stepMaxCommCache_[step1], stepMaxCommCache_[step2]);
        std::swap(stepSecondMaxCommCache_[step1], stepSecondMaxCommCache_[step2]);
        std::swap(stepMaxCommCountCache_[step1], stepMaxCommCountCache_[step2]);
    }

    void ResetSuperstep(unsigned step) {
        std::fill(stepProcSend_[step].begin(), stepProcSend_[step].end(), 0);
        std::fill(stepProcReceive_[step].begin(), stepProcReceive_[step].end(), 0);
        ArrangeSuperstepCommData(step);
    }

    void ComputeCommDatastructures(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            std::fill(stepProcSend_[step].begin(), stepProcSend_[step].end(), 0);
            std::fill(stepProcReceive_[step].begin(), stepProcReceive_[step].end(), 0);
        }

        const auto &vecSched = activeSchedule_->GetVectorSchedule();
        const auto &graph = instance_->GetComputationalDag();

        for (const auto &u : graph.Vertices()) {
            nodeLambdaMap_.ResetNode(u);
            const unsigned uProc = vecSched.AssignedProcessor(u);
            const unsigned uStep = vecSched.AssignedSuperstep(u);
            const CommWeightT commW = graph.VertexCommWeight(u);
            maxCommWeight_ = std::max(maxCommWeight_, commW);

            for (const auto &v : graph.Children(u)) {
                const unsigned vProc = vecSched.AssignedProcessor(v);
                const unsigned vStep = vecSched.AssignedSuperstep(v);

                const CommWeightT commWSendCost = (uProc != vProc) ? commW * instance_->SendCosts(uProc, vProc) : 0;

                auto &val = nodeLambdaMap_.GetProcEntry(u, vProc);
                if (CommPolicy::AddChild(val, vStep)) {
                    if (uProc != vProc && commWSendCost > 0) {
                        CommPolicy::AttributeCommunication(
                            *this, commWSendCost, uStep, uProc, vProc, vStep, val, [](unsigned) { return true; });
                    }
                }
            }
        }

        for (unsigned step = startStep; step <= endStep; step++) {
            if (step >= stepProcSend_.size()) {
                continue;
            }
            ArrangeSuperstepCommData(step);
        }
    }
};

}    // namespace osp
