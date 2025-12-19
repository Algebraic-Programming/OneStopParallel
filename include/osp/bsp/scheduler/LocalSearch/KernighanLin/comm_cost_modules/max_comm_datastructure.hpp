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
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "comm_cost_policies.hpp"
#include "generic_lambda_container.hpp"
#include "lambda_container.hpp"
#include "osp/bsp/model/BspInstance.hpp"

namespace osp {

template <typename CommWeightT>
struct PreMoveCommData {
    struct StepInfo {
        CommWeightT maxComm_;
        CommWeightT secondMaxComm_;
        unsigned maxCommCount_;
    };

    std::unordered_map<unsigned, StepInfo> stepData_;

    PreMoveCommData() = default;

    void AddStep(unsigned step, CommWeightT max, CommWeightT second, unsigned count) { stepData_[step] = {max, second, count}; }

    bool GetStep(unsigned step, StepInfo &info) const {
        auto it = stepData_.find(step);
        if (it != stepData_.end()) {
            info = it->second;
            return true;
        }
        return false;
    }
};

template <typename GraphT, typename CostT, typename KlActiveScheduleT, typename CommPolicy = EagerCommCostPolicy>
struct MaxCommDatastructure {
    using CommWeightT = VCommwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;

    const BspInstance<GraphT> *instance_;
    const KlActiveScheduleT *activeSchedule_;

    std::vector<std::vector<CommWeightT>> stepProcSend_;
    std::vector<std::vector<CommWeightT>> stepProcReceive_;

    // Caches for fast cost calculation (Global Max/Second Max per step)
    std::vector<CommWeightT> stepMaxCommCache_;
    std::vector<CommWeightT> stepSecondMaxCommCache_;
    std::vector<unsigned> stepMaxCommCountCache_;

    CommWeightT maxCommWeight_ = 0;

    // Select the appropriate container type based on the policy's ValueType
    using ContainerType = typename std::conditional<std::is_same<typename CommPolicy::ValueType, unsigned>::value,
                                                    LambdaVectorContainer<VertexType>,
                                                    GenericLambdaVectorContainer<VertexType, typename CommPolicy::ValueType>>::type;

    ContainerType nodeLambdaMap_;

    // Optimization: Scratchpad for update_datastructure_after_move to avoid allocations
    std::vector<unsigned> affectedStepsList_;
    std::vector<bool> stepIsAffected_;

    inline CommWeightT StepProcSend(unsigned step, unsigned proc) const { return stepProcSend_[step][proc]; }

    inline CommWeightT &StepProcSend(unsigned step, unsigned proc) { return stepProcSend_[step][proc]; }

    inline CommWeightT StepProcReceive(unsigned step, unsigned proc) const { return stepProcReceive_[step][proc]; }

    inline CommWeightT &StepProcReceive(unsigned step, unsigned proc) { return stepProcReceive_[step][proc]; }

    inline CommWeightT StepMaxComm(unsigned step) const { return stepMaxCommCache_[step]; }

    inline CommWeightT StepSecondMaxComm(unsigned step) const { return stepSecondMaxCommCache_[step]; }

    inline unsigned StepMaxCommCount(unsigned step) const { return stepMaxCommCountCache_[step]; }

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

    inline PreMoveCommData<CommWeightT> GetPreMoveCommData(const KlMove &move) {
        PreMoveCommData<CommWeightT> data;
        std::unordered_set<unsigned> affectedSteps;

        affectedSteps.insert(move.fromStep_);
        affectedSteps.insert(move.toStep_);

        const auto &graph = instance_->GetComputationalDag();

        for (const auto &parent : graph.Parents(move.node_)) {
            affectedSteps.insert(activeSchedule_->AssignedSuperstep(parent));
        }

        for (unsigned step : affectedSteps) {
            data.AddStep(step, StepMaxComm(step), StepSecondMaxComm(step), StepMaxCommCount(step));
        }

        return data;
    }

    void UpdateDatastructureAfterMove(const KlMove &move, unsigned, unsigned) {
        const auto &graph = instance_->GetComputationalDag();

        // Prepare Scratchpad (Avoids Allocations) ---
        for (unsigned step : affectedStepsList_) {
            if (step < stepIsAffected_.size()) {
                stepIsAffected_[step] = false;
            }
        }
        affectedStepsList_.clear();

        auto MarkStep = [&](unsigned step) {
            if (step < stepIsAffected_.size() && !stepIsAffected_[step]) {
                stepIsAffected_[step] = true;
                affectedStepsList_.push_back(step);
            }
        };

        const VertexType node = move.node_;
        const unsigned fromStep = move.fromStep_;
        const unsigned toStep = move.toStep_;
        const unsigned fromProc = move.fromProc_;
        const unsigned toProc = move.toProc_;
        const CommWeightT commWNode = graph.VertexCommWeight(node);

        // Handle Node Movement (Outgoing Edges: Node -> Children)

        if (fromStep != toStep) {
            // Case 1: Node changes Step
            for (const auto [proc, val] : nodeLambdaMap_.IterateProcEntries(node)) {
                // A. Remove Old (Sender: fromProc, Receiver: proc)
                if (proc != fromProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::UnattributeCommunication(*this, cost, fromStep, fromProc, proc, 0, val);
                    }
                }

                // B. Add New (Sender: toProc, Receiver: proc)
                if (proc != toProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::AttributeCommunication(*this, cost, toStep, toProc, proc, 0, val);
                    }
                }
            }
            MarkStep(fromStep);
            MarkStep(toStep);

        } else if (fromProc != toProc) {
            // Case 2: Node stays in same Step, but changes Processor

            for (const auto [proc, val] : nodeLambdaMap_.IterateProcEntries(node)) {
                // Remove Old (Sender: fromProc, Receiver: proc)
                if (proc != fromProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::UnattributeCommunication(*this, cost, fromStep, fromProc, proc, 0, val);
                    }
                }

                // Add New (Sender: toProc, Receiver: proc)
                if (proc != toProc) {
                    const CommWeightT cost = commWNode * instance_->SendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::AttributeCommunication(*this, cost, fromStep, toProc, proc, 0, val);
                    }
                }
            }
            MarkStep(fromStep);
        }

        // Update Parents' Outgoing Communication (Parents → Node)

        for (const auto &parent : graph.Parents(node)) {
            const unsigned parentStep = activeSchedule_->AssignedSuperstep(parent);
            // Fast boundary check
            if (parentStep >= stepProcSend_.size()) {
                continue;
            }

            const unsigned parentProc = activeSchedule_->AssignedProcessor(parent);
            const CommWeightT commWParent = graph.VertexCommWeight(parent);

            auto &val = nodeLambdaMap_.GetProcEntry(parent, fromProc);
            const bool removedFromProc = CommPolicy::RemoveChild(val, fromStep);

            // 1. Handle Removal from fromProc
            if (removedFromProc) {
                if (fromProc != parentProc) {
                    const CommWeightT cost = commWParent * instance_->SendCosts(parentProc, fromProc);
                    if (cost > 0) {
                        CommPolicy::UnattributeCommunication(*this, cost, parentStep, parentProc, fromProc, fromStep, val);
                    }
                }
            }

            auto &valTo = nodeLambdaMap_.GetProcEntry(parent, toProc);
            const bool addedToProc = CommPolicy::AddChild(valTo, toStep);

            // 2. Handle Addition to toProc
            if (addedToProc) {
                if (toProc != parentProc) {
                    const CommWeightT cost = commWParent * instance_->SendCosts(parentProc, toProc);
                    if (cost > 0) {
                        CommPolicy::AttributeCommunication(*this, cost, parentStep, parentProc, toProc, toStep, valTo);
                    }
                }
            }

            MarkStep(parentStep);
        }

        // Re-arrange Affected Steps
        for (unsigned step : affectedStepsList_) {
            ArrangeSuperstepCommData(step);
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
                        CommPolicy::AttributeCommunication(*this, commWSendCost, uStep, uProc, vProc, vStep, val);
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
