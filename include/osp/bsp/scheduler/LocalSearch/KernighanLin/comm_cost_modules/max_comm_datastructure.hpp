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
    using kl_move = kl_move_struct<cost_t, VertexType>;

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
    using ContainerType =
        typename std::conditional<std::is_same<typename CommPolicy::ValueType, unsigned>::value,
                                  lambda_vector_container<VertexType>,
                                  generic_lambda_vector_container<VertexType, typename CommPolicy::ValueType>>::type;

    ContainerType nodeLambdaMap_;

    // Optimization: Scratchpad for update_datastructure_after_move to avoid allocations
    std::vector<unsigned> affectedStepsList_;
    std::vector<bool> stepIsAffected_;

    inline comm_weight_t StepProcSend(unsigned step, unsigned proc) const { return step_proc_send_[step][proc]; }

    inline comm_weight_t &StepProcSend(unsigned step, unsigned proc) { return step_proc_send_[step][proc]; }

    inline comm_weight_t StepProcReceive(unsigned step, unsigned proc) const { return step_proc_receive_[step][proc]; }

    inline comm_weight_t &StepProcReceive(unsigned step, unsigned proc) { return step_proc_receive_[step][proc]; }

    inline comm_weight_t StepMaxComm(unsigned step) const { return step_max_comm_cache[step]; }

    inline comm_weight_t StepSecondMaxComm(unsigned step) const { return step_second_max_comm_cache[step]; }

    inline unsigned StepMaxCommCount(unsigned step) const { return stepMaxCommCountCache_[step]; }

    inline void Initialize(KlActiveScheduleT &klSched) {
        activeSchedule_ = &klSched;
        instance_ = &activeSchedule_->GetInstance();
        const unsigned numSteps = activeSchedule_->num_steps();
        const unsigned numProcs = instance_->NumberOfProcessors();
        max_comm_weight = 0;

        step_proc_send_.assign(num_steps, std::vector<comm_weight_t>(num_procs, 0));
        step_proc_receive_.assign(num_steps, std::vector<comm_weight_t>(num_procs, 0));

        step_max_comm_cache.assign(num_steps, 0);
        step_second_max_comm_cache.assign(num_steps, 0);
        stepMaxCommCountCache_.assign(numSteps, 0);

        node_lambda_map.initialize(instance->GetComputationalDag().NumVertices(), num_procs);

        // Initialize scratchpad
        stepIsAffected_.assign(numSteps, false);
        affectedStepsList_.reserve(numSteps);
    }

    inline void Clear() {
        step_proc_send_.clear();
        step_proc_receive_.clear();
        step_max_comm_cache.clear();
        step_second_max_comm_cache.clear();
        stepMaxCommCountCache_.clear();
        node_lambda_map.clear();
        affectedStepsList_.clear();
        stepIsAffected_.clear();
    }

    inline void ArrangeSuperstepCommData(const unsigned step) {
        comm_weight_t maxSend = 0;
        comm_weight_t secondMaxSend = 0;
        unsigned maxSendCount = 0;

        const auto &sends = step_proc_send_[step];
        for (const auto val : sends) {
            if (val > max_send) {
                second_max_send = max_send;
                max_send = val;
                max_send_count = 1;
            } else if (val == max_send) {
                max_send_count++;
            } else if (val > second_max_send) {
                second_max_send = val;
            }
        }

        comm_weight_t maxReceive = 0;
        comm_weight_t secondMaxReceive = 0;
        unsigned maxReceiveCount = 0;

        const auto &receives = step_proc_receive_[step];
        for (const auto val : receives) {
            if (val > max_receive) {
                second_max_receive = max_receive;
                max_receive = val;
                max_receive_count = 1;
            } else if (val == max_receive) {
                max_receive_count++;
            } else if (val > second_max_receive) {
                second_max_receive = val;
            }
        }

        const comm_weight_t globalMax = std::max(max_send, max_receive);
        step_max_comm_cache[step] = global_max;

        unsigned globalCount = 0;
        if (maxSend == global_max) {
            globalCount += maxSendCount;
        }
        if (maxReceive == global_max) {
            globalCount += maxReceiveCount;
        }
        stepMaxCommCountCache_[step] = globalCount;

        comm_weight_t candSend = (maxSend == global_max) ? second_max_send : max_send;
        comm_weight_t candRecv = (maxReceive == global_max) ? second_max_receive : max_receive;

        step_second_max_comm_cache[step] = std::max(cand_send, cand_recv);
    }

    void RecomputeMaxSendReceive(unsigned step) { ArrangeSuperstepCommData(step); }

    inline pre_move_comm_data<comm_weight_t> GetPreMoveCommData(const kl_move &move) {
        pre_move_comm_data<comm_weight_t> data;
        std::unordered_set<unsigned> affectedSteps;

        affectedSteps.insert(move.from_step);
        affectedSteps.insert(move.to_step);

        const auto &graph = instance_->GetComputationalDag();

        for (const auto &parent : graph.Parents(move.node)) {
            affected_steps.insert(active_schedule->assigned_superstep(parent));
        }

        for (unsigned step : affectedSteps) {
            data.add_step(step, step_max_comm(step), step_second_max_comm(step), step_max_comm_count(step));
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
        const unsigned fromStep = move.from_step;
        const unsigned toStep = move.to_step;
        const unsigned fromProc = move.from_proc;
        const unsigned toProc = move.to_proc;
        const comm_weight_t commWNode = graph.VertexCommWeight(node);

        // Handle Node Movement (Outgoing Edges: Node -> Children)

        if (fromStep != toStep) {
            // Case 1: Node changes Step
            for (const auto [proc, val] : node_lambda_map.iterate_proc_entries(node)) {
                // A. Remove Old (Sender: from_proc, Receiver: proc)
                if (proc != from_proc) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(from_proc, proc);
                    if (cost > 0) {
                        CommPolicy::unattribute_communication(*this, cost, from_step, from_proc, proc, 0, val);
                    }
                }

                // B. Add New (Sender: to_proc, Receiver: proc)
                if (proc != to_proc) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(to_proc, proc);
                    if (cost > 0) {
                        CommPolicy::attribute_communication(*this, cost, to_step, to_proc, proc, 0, val);
                    }
                }
            }
            markStep(fromStep);
            markStep(toStep);

        } else if (fromProc != toProc) {
            // Case 2: Node stays in same Step, but changes Processor

            for (const auto [proc, val] : node_lambda_map.iterate_proc_entries(node)) {
                // Remove Old (Sender: from_proc, Receiver: proc)
                if (proc != from_proc) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(from_proc, proc);
                    if (cost > 0) {
                        CommPolicy::unattribute_communication(*this, cost, from_step, from_proc, proc, 0, val);
                    }
                }

                // Add New (Sender: to_proc, Receiver: proc)
                if (proc != to_proc) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(to_proc, proc);
                    if (cost > 0) {
                        CommPolicy::attribute_communication(*this, cost, from_step, to_proc, proc, 0, val);
                    }
                }
            }
            markStep(fromStep);
        }

        // Update Parents' Outgoing Communication (Parents → Node)

        for (const auto &parent : graph.Parents(node)) {
            const unsigned parent_step = active_schedule->assigned_superstep(parent);
            // Fast boundary check
            if (parent_step >= step_proc_send_.size()) {
                continue;
            }

            const unsigned parent_proc = active_schedule->assigned_processor(parent);
            const comm_weight_t comm_w_parent = graph.VertexCommWeight(parent);

            auto &val = node_lambda_map.get_proc_entry(parent, from_proc);
            const bool removed_from_proc = CommPolicy::remove_child(val, from_step);

            // 1. Handle Removal from from_proc
            if (removed_from_proc) {
                if (from_proc != parent_proc) {
                    const comm_weight_t cost = comm_w_parent * instance->sendCosts(parent_proc, from_proc);
                    if (cost > 0) {
                        CommPolicy::unattribute_communication(*this, cost, parent_step, parent_proc, from_proc, from_step, val);
                    }
                }
            }

            auto &val_to = node_lambda_map.get_proc_entry(parent, to_proc);
            const bool added_to_proc = CommPolicy::add_child(val_to, to_step);

            // 2. Handle Addition to to_proc
            if (added_to_proc) {
                if (to_proc != parent_proc) {
                    const comm_weight_t cost = comm_w_parent * instance->sendCosts(parent_proc, to_proc);
                    if (cost > 0) {
                        CommPolicy::attribute_communication(*this, cost, parent_step, parent_proc, to_proc, to_step, val_to);
                    }
                }
            }

            mark_step(parent_step);
        }

        // Re-arrange Affected Steps
        for (unsigned step : affectedStepsList_) {
            ArrangeSuperstepCommData(step);
        }
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(step_proc_send_[step1], step_proc_send_[step2]);
        std::swap(step_proc_receive_[step1], step_proc_receive_[step2]);
        std::swap(step_max_comm_cache[step1], step_max_comm_cache[step2]);
        std::swap(step_second_max_comm_cache[step1], step_second_max_comm_cache[step2]);
        std::swap(stepMaxCommCountCache_[step1], stepMaxCommCountCache_[step2]);
    }

    void ResetSuperstep(unsigned step) {
        std::fill(step_proc_send_[step].begin(), step_proc_send_[step].end(), 0);
        std::fill(step_proc_receive_[step].begin(), step_proc_receive_[step].end(), 0);
        ArrangeSuperstepCommData(step);
    }

    void ComputeCommDatastructures(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            std::fill(step_proc_send_[step].begin(), step_proc_send_[step].end(), 0);
            std::fill(step_proc_receive_[step].begin(), step_proc_receive_[step].end(), 0);
        }

        const auto &vecSched = activeSchedule_->getVectorSchedule();
        const auto &graph = instance_->GetComputationalDag();

        for (const auto &u : graph.Vertices()) {
            node_lambda_map.reset_node(u);
            const unsigned uProc = vecSched.AssignedProcessor(u);
            const unsigned uStep = vecSched.AssignedSuperstep(u);
            const comm_weight_t commW = graph.VertexCommWeight(u);
            max_comm_weight = std::max(max_comm_weight, comm_w);

            for (const auto &v : graph.Children(u)) {
                const unsigned vProc = vecSched.AssignedProcessor(v);
                const unsigned vStep = vecSched.AssignedSuperstep(v);

                const comm_weight_t commWSendCost = (uProc != vProc) ? comm_w * instance_->sendCosts(uProc, vProc) : 0;

                auto &val = node_lambda_map.get_proc_entry(u, v_proc);
                if (CommPolicy::add_child(val, vStep)) {
                    if (uProc != vProc && comm_w_send_cost > 0) {
                        CommPolicy::attribute_communication(*this, comm_w_send_cost, uStep, uProc, vProc, vStep, val);
                    }
                }
            }
        }

        for (unsigned step = startStep; step <= endStep; step++) {
            if (step >= step_proc_send_.size()) {
                continue;
            }
            ArrangeSuperstepCommData(step);
        }
    }
};

}    // namespace osp
