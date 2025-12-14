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
    using comm_weight_t = VCommwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using kl_move = KlMoveStruct<CostT, VertexType>;

    const BspInstance<GraphT> *instance_;
    const KlActiveScheduleT *activeSchedule_;

    std::vector<std::vector<comm_weight_t>> stepProcSend_;
    std::vector<std::vector<comm_weight_t>> stepProcReceive_;

    // Caches for fast cost calculation (Global Max/Second Max per step)
    std::vector<comm_weight_t> stepMaxCommCache_;
    std::vector<comm_weight_t> stepSecondMaxCommCache_;
    std::vector<unsigned> stepMaxCommCountCache_;

    comm_weight_t maxCommWeight_ = 0;

    // Select the appropriate container type based on the policy's ValueType
    using ContainerType = typename std::conditional<std::is_same<typename CommPolicy::ValueType, unsigned>::value,
                                                    LambdaVectorContainer<VertexType>,
                                                    GenericLambdaVectorContainer<VertexType, typename CommPolicy::ValueType>>::type;

    ContainerType nodeLambdaMap_;

    // Optimization: Scratchpad for update_datastructure_after_move to avoid allocations
    std::vector<unsigned> affectedStepsList_;
    std::vector<bool> stepIsAffected_;

    inline comm_weight_t StepProcSend(unsigned step, unsigned proc) const { return stepProcSend_[step][proc]; }

    inline comm_weight_t &StepProcSend(unsigned step, unsigned proc) { return stepProcSend_[step][proc]; }

    inline comm_weight_t StepProcReceive(unsigned step, unsigned proc) const { return stepProcReceive_[step][proc]; }

    inline comm_weight_t &StepProcReceive(unsigned step, unsigned proc) { return stepProcReceive_[step][proc]; }

    inline comm_weight_t StepMaxComm(unsigned step) const { return stepMaxCommCache_[step]; }

    inline comm_weight_t StepSecondMaxComm(unsigned step) const { return stepSecondMaxCommCache_[step]; }

    inline unsigned StepMaxCommCount(unsigned step) const { return stepMaxCommCountCache_[step]; }

    inline void Initialize(KlActiveScheduleT &klSched) {
        activeSchedule_ = &klSched;
        instance_ = &activeSchedule_->GetInstance();
        const unsigned numSteps = activeSchedule_->numSteps();
        const unsigned numProcs = instance_->NumberOfProcessors();
        maxCommWeight_ = 0;

        stepProcSend_.assign(numSteps, std::vector<comm_weight_t>(numProcs, 0));
        stepProcReceive_.assign(numSteps, std::vector<comm_weight_t>(numProcs, 0));

        stepMaxCommCache_.assign(numSteps, 0);
        stepSecondMaxCommCache_.assign(numSteps, 0);
        stepMaxCommCountCache_.assign(numSteps, 0);

        nodeLambdaMap_.initialize(instance_->GetComputationalDag().NumVertices(), numProcs);

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
        comm_weight_t maxSend = 0;
        comm_weight_t secondMaxSend = 0;
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

        comm_weight_t maxReceive = 0;
        comm_weight_t secondMaxReceive = 0;
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

        const comm_weight_t globalMax = std::max(maxSend, maxReceive);
        stepMaxCommCache_[step] = globalMax;

        unsigned globalCount = 0;
        if (maxSend == globalMax) {
            globalCount += maxSendCount;
        }
        if (maxReceive == globalMax) {
            globalCount += maxReceiveCount;
        }
        stepMaxCommCountCache_[step] = globalCount;

        comm_weight_t candSend = (maxSend == globalMax) ? secondMaxSend : maxSend;
        comm_weight_t candRecv = (maxReceive == globalMax) ? secondMaxReceive : maxReceive;

        stepSecondMaxCommCache_[step] = std::max(candSend, candRecv);
    }

    void RecomputeMaxSendReceive(unsigned step) { ArrangeSuperstepCommData(step); }

    inline PreMoveCommData<comm_weight_t> GetPreMoveCommData(const kl_move &move) {
        PreMoveCommData<comm_weight_t> data;
        std::unordered_set<unsigned> affectedSteps;

        affectedSteps.insert(move.fromStep);
        affectedSteps.insert(move.toStep);

        const auto &graph = instance_->GetComputationalDag();

        for (const auto &parent : graph.Parents(move.node)) {
            affectedSteps.insert(activeSchedule_->assigned_superstep(parent));
        }

        for (unsigned step : affectedSteps) {
            data.add_step(step, StepMaxComm(step), StepSecondMaxComm(step), StepMaxCommCount(step));
        }

        return data;
    }

    void UpdateDatastructureAfterMove(const kl_move &move, unsigned, unsigned) {
        const auto &graph = instance_->GetComputationalDag();

        // Prepare Scratchpad (Avoids Allocations) ---
        for (unsigned step : affectedStepsList_) {
            if (step < stepIsAffected_.size()) {
                stepIsAffected_[step] = false;
            }
        }
        affectedStepsList_.clear();

        auto markStep = [&](unsigned step) {
            if (step < stepIsAffected_.size() && !stepIsAffected_[step]) {
                stepIsAffected_[step] = true;
                affectedStepsList_.push_back(step);
            }
        };

        const VertexType node = move.node;
        const unsigned fromStep = move.fromStep;
        const unsigned toStep = move.toStep;
        const unsigned fromProc = move.fromProc;
        const unsigned toProc = move.toProc;
        const comm_weight_t commWNode = graph.VertexCommWeight(node);

        // Handle Node Movement (Outgoing Edges: Node -> Children)

        if (fromStep != toStep) {
            // Case 1: Node changes Step
            for (const auto [proc, val] : nodeLambdaMap_.iterate_proc_entries(node)) {
                // A. Remove Old (Sender: fromProc, Receiver: proc)
                if (proc != fromProc) {
                    const comm_weight_t cost = commWNode * instance_->sendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::unattribute_communication(*this, cost, fromStep, fromProc, proc, 0, val);
                    }
                }

                // B. Add New (Sender: toProc, Receiver: proc)
                if (proc != toProc) {
                    const comm_weight_t cost = commWNode * instance_->sendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::attribute_communication(*this, cost, toStep, toProc, proc, 0, val);
                    }
                }
            }
            markStep(fromStep);
            markStep(toStep);

        } else if (fromProc != toProc) {
            // Case 2: Node stays in same Step, but changes Processor

            for (const auto [proc, val] : nodeLambdaMap_.iterate_proc_entries(node)) {
                // Remove Old (Sender: fromProc, Receiver: proc)
                if (proc != fromProc) {
                    const comm_weight_t cost = commWNode * instance_->sendCosts(fromProc, proc);
                    if (cost > 0) {
                        CommPolicy::unattribute_communication(*this, cost, fromStep, fromProc, proc, 0, val);
                    }
                }

                // Add New (Sender: toProc, Receiver: proc)
                if (proc != toProc) {
                    const comm_weight_t cost = commWNode * instance_->sendCosts(toProc, proc);
                    if (cost > 0) {
                        CommPolicy::attribute_communication(*this, cost, fromStep, toProc, proc, 0, val);
                    }
                }
            }
            markStep(fromStep);
        }

        // Update Parents' Outgoing Communication (Parents → Node)

        for (const auto &parent : graph.Parents(node)) {
            const unsigned parentStep = activeSchedule_->assigned_superstep(parent);
            // Fast boundary check
            if (parentStep >= stepProcSend_.size()) {
                continue;
            }

            const unsigned parentProc = activeSchedule_->assigned_processor(parent);
            const comm_weight_t commWParent = graph.VertexCommWeight(parent);

            auto &val = nodeLambdaMap_.get_proc_entry(parent, fromProc);
            const bool removed_from_proc = CommPolicy::remove_child(val, fromStep);

            // 1. Handle Removal from fromProc
            if (removed_from_proc) {
                if (fromProc != parentProc) {
                    const comm_weight_t cost = commWParent * instance_->sendCosts(parentProc, fromProc);
                    if (cost > 0) {
                        CommPolicy::unattribute_communication(*this, cost, parentStep, parentProc, fromProc, fromStep, val);
                    }
                }
            }

            auto &val_to = nodeLambdaMap_.get_proc_entry(parent, toProc);
            const bool added_to_proc = CommPolicy::add_child(val_to, toStep);

            // 2. Handle Addition to toProc
            if (added_to_proc) {
                if (toProc != parentProc) {
                    const comm_weight_t cost = commWParent * instance_->sendCosts(parentProc, toProc);
                    if (cost > 0) {
                        CommPolicy::attribute_communication(*this, cost, parentStep, parentProc, toProc, toStep, val_to);
                    }
                }
            }

            markStep(parentStep);
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

        const auto &vecSched = activeSchedule_->getVectorSchedule();
        const auto &graph = instance_->GetComputationalDag();

        for (const auto &u : graph.Vertices()) {
            nodeLambdaMap_.reset_node(u);
            const unsigned uProc = vecSched.AssignedProcessor(u);
            const unsigned uStep = vecSched.AssignedSuperstep(u);
            const comm_weight_t commW = graph.VertexCommWeight(u);
            maxCommWeight_ = std::max(maxCommWeight_, commW);

            for (const auto &v : graph.Children(u)) {
                const unsigned vProc = vecSched.AssignedProcessor(v);
                const unsigned vStep = vecSched.AssignedSuperstep(v);

                const comm_weight_t commWSendCost = (uProc != vProc) ? commW * instance_->sendCosts(uProc, vProc) : 0;

                auto &val = nodeLambdaMap_.get_proc_entry(u, vProc);
                if (CommPolicy::add_child(val, vStep)) {
                    if (uProc != vProc && commWSendCost > 0) {
                        CommPolicy::attribute_communication(*this, commWSendCost, uStep, uProc, vProc, vStep, val);
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
