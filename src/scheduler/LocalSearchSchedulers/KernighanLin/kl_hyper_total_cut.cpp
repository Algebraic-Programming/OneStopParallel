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

#include "scheduler/LocalSearchSchedulers/KernighanLin/kl_hyper_total_cut.hpp"



double kl_hyper_total_cut::compute_current_costs() {

    double work_costs = 0;
    for (unsigned step = 0; step < current_schedule.num_steps(); step++) {
        work_costs += current_schedule.step_max_work[step];
    }

    double comm_costs = 0;

    for (const auto &node : current_schedule.instance->getComputationalDag().vertices()) {

        if (current_schedule.instance->getComputationalDag().isSink(node))
            continue;

        std::unordered_set<unsigned> intersects;

        for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

            const unsigned &target_proc = current_schedule.vector_schedule.assignedProcessor(target);
            const unsigned &target_step = current_schedule.vector_schedule.assignedSuperstep(target);

            if (current_schedule.vector_schedule.assignedProcessor(node) != target_proc || current_schedule.vector_schedule.assignedSuperstep(node) != target_step) {
                intersects.insert(current_schedule.instance->numberOfProcessors() * target_step + target_proc);
            }
        }

        comm_costs += intersects.size() * current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node);
    }

    current_schedule.current_cost = work_costs + comm_costs * current_schedule.comm_multiplier + (current_schedule.num_steps() - 1) * current_schedule.instance->synchronisationCosts();

    return current_schedule.current_cost;
}

void kl_hyper_total_cut::compute_comm_gain(unsigned node, unsigned current_step, unsigned current_proc,
                                           unsigned new_proc) {

    // TODO
    throw std::runtime_error("Not implemented yet");

    if (current_proc == new_proc) {

        for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

            const double loss = (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) *
                                1.0 * current_schedule.comm_multiplier;

            if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                node_gains[node][new_proc][0] -= loss;
                node_gains[node][new_proc][2] -= loss;
                node_change_in_costs[node][new_proc][0] += loss;
                node_change_in_costs[node][new_proc][2] += loss;

            } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step + 1) {

                node_gains[node][new_proc][2] += loss;
                node_change_in_costs[node][new_proc][2] -= loss;

            } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step - 1) {

                node_gains[node][new_proc][0] += loss;
                node_change_in_costs[node][new_proc][0] -= loss;
            }

            if ((current_step + 1 == current_schedule.vector_schedule.assignedSuperstep(target) &&
                 current_proc != current_schedule.vector_schedule.assignedProcessor(target)) ||
                (current_step == current_schedule.vector_schedule.assignedSuperstep(target) &&
                 current_proc == current_schedule.vector_schedule.assignedProcessor(target))) {
                node_gains[node][current_proc][2] -= penalty;

            } else if ((current_step == current_schedule.vector_schedule.assignedSuperstep(target) &&
                        current_proc != current_schedule.vector_schedule.assignedProcessor(target)) ||
                       (current_step - 1 == current_schedule.vector_schedule.assignedSuperstep(target) &&
                        current_proc == current_schedule.vector_schedule.assignedProcessor(target))) {

                node_gains[node][current_proc][0] +=
                    (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) + reward;
            }
        }

        for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

            const double loss =
                (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) * 1.0 *
                current_schedule.comm_multiplier;

            if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                node_gains[node][new_proc][0] -= loss;
                node_gains[node][new_proc][2] -= loss;
                node_change_in_costs[node][new_proc][0] += loss;
                node_change_in_costs[node][new_proc][2] += loss;

            } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step + 1) {

                node_gains[node][new_proc][2] += loss;
                node_change_in_costs[node][new_proc][2] -= loss;

            } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step - 1) {

                node_gains[node][new_proc][0] += loss;
                node_change_in_costs[node][new_proc][0] -= loss;
            }

            if ((current_step - 1 == current_schedule.vector_schedule.assignedSuperstep(source) &&
                 current_proc != current_schedule.vector_schedule.assignedProcessor(source)) ||
                (current_step == current_schedule.vector_schedule.assignedSuperstep(source) &&
                 current_proc == current_schedule.vector_schedule.assignedProcessor(source))) {
                node_gains[node][current_proc][0] -= penalty;

            } else if ((current_step == current_schedule.vector_schedule.assignedSuperstep(source) &&
                        current_proc != current_schedule.vector_schedule.assignedProcessor(source)) ||
                       (current_step + 1 == current_schedule.vector_schedule.assignedSuperstep(source) &&
                        current_proc == current_schedule.vector_schedule.assignedProcessor(source))) {

                node_gains[node][current_proc][2] +=
                    (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) + reward;
            }
        }
    } else {

        // current_proc != new_proc

        for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

            const unsigned &target_proc = current_schedule.vector_schedule.assignedProcessor(target);
            if (target_proc == current_proc) {

                if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                    const double loss =
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) *
                        current_schedule.instance->communicationCosts(new_proc, target_proc) *
                        current_schedule.comm_multiplier;

                    node_gains[node][new_proc][0] -= loss;
                    node_gains[node][new_proc][1] -= loss;
                    node_gains[node][new_proc][2] -= loss;

                    node_change_in_costs[node][new_proc][0] += loss;
                    node_change_in_costs[node][new_proc][1] += loss;
                    node_change_in_costs[node][new_proc][2] += loss;
                }

                if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                    node_gains[node][new_proc][1] -= penalty;
                    node_gains[node][new_proc][2] -= penalty;

                } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step + 1) {

                    node_gains[node][new_proc][2] -= penalty;
                }

            } else if (target_proc == new_proc) {

                const double gain =
                    (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) *
                    current_schedule.instance->communicationCosts(current_proc, target_proc) *
                    current_schedule.comm_multiplier;

                if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {
                    node_gains[node][new_proc][1] += gain;
                    node_change_in_costs[node][new_proc][1] -= gain;

                } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step + 1) {
                    node_gains[node][new_proc][2] += gain;
                    node_change_in_costs[node][new_proc][2] -= gain;
                } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step - 1) {
                    node_gains[node][new_proc][0] += gain;
                    node_change_in_costs[node][new_proc][0] -= gain;
                }

                if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                    node_gains[node][new_proc][1] +=
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) + reward;

                    node_gains[node][new_proc][0] +=
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) + reward;

                } else if (current_schedule.vector_schedule.assignedSuperstep(target) < current_step) {

                    node_gains[node][new_proc][0] +=
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) + reward;
                }

            } else {

                assert(target_proc != current_proc && target_proc != new_proc);

                const double gain = (double)(current_schedule.instance->communicationCosts(new_proc, target_proc) -
                                             current_schedule.instance->communicationCosts(current_proc, target_proc)) *
                                    current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) *
                                    current_schedule.comm_multiplier;

                node_gains[node][new_proc][0] += gain;
                node_gains[node][new_proc][1] += gain;
                node_gains[node][new_proc][2] += gain;

                node_change_in_costs[node][new_proc][0] -= gain;
                node_change_in_costs[node][new_proc][1] -= gain;
                node_change_in_costs[node][new_proc][2] -= gain;

                if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step + 1) {

                    node_gains[node][new_proc][2] -= penalty;
                } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                    node_gains[node][new_proc][0] +=
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(node) + reward;
                }
            }
        }

        for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

            const unsigned &source_proc = current_schedule.vector_schedule.assignedProcessor(source);
            if (source_proc == current_proc) {

                if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {
                    const double loss =
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) *
                        current_schedule.instance->communicationCosts(current_proc, new_proc) *
                        current_schedule.comm_multiplier;

                    node_gains[node][new_proc][0] -= loss;
                    node_gains[node][new_proc][1] -= loss;
                    node_gains[node][new_proc][2] -= loss;

                    node_change_in_costs[node][new_proc][0] += loss;
                    node_change_in_costs[node][new_proc][1] += loss;
                    node_change_in_costs[node][new_proc][2] += loss;
                }

                if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                    node_gains[node][new_proc][0] -= penalty;
                    node_gains[node][new_proc][1] -= penalty;

                } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step - 1) {

                    node_gains[node][new_proc][0] -= penalty;
                }

            } else if (source_proc == new_proc) {

                assert(source_proc != current_proc);
                const double gain =
                    (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) *
                    current_schedule.instance->communicationCosts(current_proc, new_proc) *
                    current_schedule.comm_multiplier;

                if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {
                    node_gains[node][new_proc][1] += gain;
                    node_change_in_costs[node][new_proc][1] -= gain;

                } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step - 1) {
                    node_gains[node][new_proc][0] += gain;
                    node_change_in_costs[node][new_proc][0] -= gain;
                } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step + 1) {
                    node_gains[node][new_proc][2] += gain;
                    node_change_in_costs[node][new_proc][2] -= gain;
                }

                if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                    node_gains[node][new_proc][1] +=
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) +
                        reward;

                    node_gains[node][new_proc][2] +=
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) +
                        reward;

                } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step + 1) {

                    node_gains[node][new_proc][2] +=
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) +
                        reward;
                }

            } else {

                assert(source_proc != current_proc && source_proc != new_proc);
                const double gain = (double)(current_schedule.instance->communicationCosts(new_proc, source_proc) -
                                             current_schedule.instance->communicationCosts(current_proc, source_proc)) *
                                    current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) *
                                    current_schedule.comm_multiplier;

                node_gains[node][new_proc][0] += gain;
                node_gains[node][new_proc][1] += gain;
                node_gains[node][new_proc][2] += gain;

                node_change_in_costs[node][new_proc][0] -= gain;
                node_change_in_costs[node][new_proc][1] -= gain;
                node_change_in_costs[node][new_proc][2] -= gain;

                if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step - 1) {

                    node_gains[node][new_proc][0] -= penalty;

                } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                    node_gains[node][new_proc][2] +=
                        (double)current_schedule.instance->getComputationalDag().nodeCommunicationWeight(source) +
                        reward;
                }
            }
        }
    }
}
