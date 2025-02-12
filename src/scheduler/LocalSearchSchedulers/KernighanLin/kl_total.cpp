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

#include "scheduler/LocalSearchSchedulers/KernighanLin/kl_total.hpp"

void kl_total::initialize_datastructures() {

#ifdef KL_DEBUG
    std::cout << "KLTotal initialize datastructures" << std::endl;
#endif

    kl_base::initialize_datastructures();

    int max_edge_weight_ = 0;
    int max_node_weight_ = 0;

    for (const auto vertex : current_schedule.instance->getComputationalDag().vertices()) {

        if (current_schedule.instance->getComputationalDag().isSink(vertex))
            continue;

        max_edge_weight_ = std::max(max_edge_weight_,
                                    current_schedule.instance->getComputationalDag().nodeCommunicationWeight(vertex));

        max_node_weight_ =
            std::max(max_node_weight_, current_schedule.instance->getComputationalDag().nodeWorkWeight(vertex));
    }

    if (not current_schedule.use_node_communication_costs) {

        max_edge_weight_ = 0;

        for (const auto &edge : current_schedule.instance->getComputationalDag().edges()) {
            max_edge_weight_ = std::max(max_edge_weight_,
                                        current_schedule.instance->getComputationalDag().edgeCommunicationWeight(edge));
        }
    }

    max_edge_weight = max_edge_weight_ + max_node_weight_;

    parameters.initial_penalty =
        max_edge_weight * current_schedule.comm_multiplier * current_schedule.instance->communicationCosts();

    parameters.gain_threshold =
        max_edge_weight * current_schedule.comm_multiplier * current_schedule.instance->communicationCosts();
}

void kl_total::update_reward_penalty() {

    if (current_schedule.current_violations.size() <= parameters.violations_threshold) {
        penalty = parameters.initial_penalty;
        reward = 0.0;

    } else {
        parameters.violations_threshold = 0;

        penalty = std::log((current_schedule.current_violations.size())) * max_edge_weight *
                  current_schedule.comm_multiplier * current_schedule.instance->communicationCosts();

        reward = std::sqrt((current_schedule.current_violations.size() + 4)) * max_edge_weight *
                 current_schedule.comm_multiplier * current_schedule.instance->communicationCosts();
    }
}

void kl_total::set_initial_reward_penalty() {

    penalty = parameters.initial_penalty;
    reward = max_edge_weight * current_schedule.comm_multiplier * current_schedule.instance->communicationCosts();
}

void kl_total::select_nodes_comm(unsigned threshold) {

    if (current_schedule.use_node_communication_costs) {

        for (const auto &node : current_schedule.instance->getComputationalDag().vertices()) {

            for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

                if (current_schedule.vector_schedule.assignedProcessor(node) !=
                    current_schedule.vector_schedule.assignedProcessor(source)) {

                    if (current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) >
                        node_comm_selection_threshold) {

                        node_selection.insert(node);
                        break;
                    }
                }
            }

            for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

                if (current_schedule.vector_schedule.assignedProcessor(node) !=
                    current_schedule.vector_schedule.assignedProcessor(target)) {

                    if (current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) >
                        node_comm_selection_threshold) {

                        node_selection.insert(node);
                        break;
                    }
                }
            }
        }
    } else {
        for (const auto &node : current_schedule.instance->getComputationalDag().vertices()) {

            for (const auto &in_edge : current_schedule.instance->getComputationalDag().in_edges(node)) {

                const auto &source = in_edge.m_source;
                if (current_schedule.vector_schedule.assignedProcessor(node) !=
                    current_schedule.vector_schedule.assignedProcessor(source)) {

                    if (current_schedule.instance->getComputationalDag().edgeCommunicationWeight(in_edge) >
                        node_comm_selection_threshold) {

                        node_selection.insert(node);
                        break;
                    }
                }
            }

            for (const auto &out_edge : current_schedule.instance->getComputationalDag().out_edges(node)) {

                const auto &target = out_edge.m_target;
                if (current_schedule.vector_schedule.assignedProcessor(node) !=
                    current_schedule.vector_schedule.assignedProcessor(target)) {

                    if (current_schedule.instance->getComputationalDag().edgeCommunicationWeight(out_edge) >
                        node_comm_selection_threshold) {

                        node_selection.insert(node);
                        break;
                    }
                }
            }
        }
    }
}