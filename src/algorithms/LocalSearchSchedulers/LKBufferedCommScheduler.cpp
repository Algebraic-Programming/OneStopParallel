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

#include "algorithms/LocalSearchSchedulers/LKBufferedCommScheduler.hpp"

void LKBufferedCommScheduler::commputeCommGain(unsigned node, unsigned current_step, unsigned current_proc,
                                               unsigned new_proc) {

/*
    if (current_proc == new_proc) {

    } else {

        unsigned out_comm_vol = 0;
        unsigned comm_to_new_proc = 0;
        unsigned comm_to_old_proc = 0;
        unsigned step_needed = num_steps;

        std::vector<bool> communicated_to_proc(num_procs, false);

        // outgoing edges
        for (const auto &target : instance->getComputationalDag().children(node)) {

            if (vector_schedule.assignedProcessor(target) == current_proc) {

                step_needed = std::min(step_needed, vector_schedule.assignedSuperstep(target));

                // node needs to be send only once
                comm_to_old_proc = instance->sendCosts(current_proc, new_proc) *
                                   instance->getComputationalDag().nodeCommunicationWeight(target);

                // charge penalty for eah violated edge
                if (vector_schedule.assignedSuperstep(target) == current_step) {

                    node_gains[node][new_proc][1] -=
                        instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;

                } else if (vector_schedule.assignedSuperstep(target) == current_step + 1) {

                    node_gains[node][new_proc][1] -=
                        instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;
                    node_gains[node][new_proc][2] -=
                        instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;
                }

            } else if (vector_schedule.assignedProcessor(target) == new_proc) {

                // communicated once
                comm_to_new_proc = instance->sendCosts(current_proc, new_proc) *
                                   instance->getComputationalDag().nodeCommunicationWeight(target);

            } else {

                // communicated once per processor
                if (!communicated_to_proc[vector_schedule.assignedProcessor(target)]) {

                    communicated_to_proc[vector_schedule.assignedProcessor(target)] = true;

                    comm_to_old_proc += instance->sendCosts(current_proc, new_proc) *
                                        instance->getComputationalDag().nodeCommunicationWeight(target);
                }

                // charge penalty for each edge that is violated
                if (vector_schedule.assignedSuperstep(target) == current_step + 1) {
                    node_gains[node][new_proc][2] -=
                        instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;
                }
            }
        }

        // charge costs for new communication -- penalty was already charged above
        if (step_needed < num_steps) {
            step_needed--; // needs to be send in the step before it is computed
            const unsigned local_max = std::max(step_max_send[step_needed], step_max_receive[step_needed]);
            if (step_max_receive[step_needed] + comm_to_old_proc > local_max) {
                node_gains[node][new_proc][0] -= step_max_receive[step_needed] + comm_to_old_proc - local_max;
                node_gains[node][new_proc][1] -= step_max_receive[step_needed] + comm_to_old_proc - local_max;
                node_gains[node][new_proc][2] -= step_max_receive[step_needed] + comm_to_old_proc - local_max;
            }
        }

        node_gains[node][new_proc][0] += computeGainMoveCommunication(
            current_proc, new_proc, current_step, current_step - 1, comm_to_old_proc, comm_to_new_proc, out_comm_vol);

        node_gains[node][new_proc][1] += computeGainMoveCommunication(
            current_proc, new_proc, current_step, current_step, comm_to_old_proc, comm_to_new_proc, out_comm_vol);

        node_gains[node][new_proc][2] += computeGainMoveCommunication(
            current_proc, new_proc, current_step + 1, current_step, comm_to_old_proc, comm_to_new_proc, out_comm_vol);

        // unsigned inc_comm_vol = 0;
        unsigned comm_from_new_proc = 0;
        unsigned comm_from_old_proc = 0;

        std::vector<unsigned> new_comm_nodes_from_current_to_new;
        std::vector<unsigned> remove_comm_nodes_from_new_to_current;
        std::vector<unsigned> remove_comm_nodes_from_any_to_current;
        std::vector<unsigned> new_comm_nodes_from_any_to_new;

        // incoming edges
        for (const auto &source : instance->getComputationalDag().parents(node)) {

            const unsigned &source_proc = vector_schedule.assignedProcessor(source);
            // remove communication from source_proc to current_proc
            if (source_proc == current_proc) {

                // check if source is already sent to new_proc
                bool communication_needed = true;
                for (auto const &target : instance->getComputationalDag().children(source)) {
                    if (target != node && vector_schedule.assignedProcessor(target) == new_proc) {
                        communication_needed = false;
                        break;
                    }
                }

                if (communication_needed) {

                    new_comm_nodes_from_current_to_new.push_back(source);
                    comm_from_old_proc += instance->sendCosts(current_proc, new_proc) *
                                          instance->getComputationalDag().nodeCommunicationWeight(source);
                }

            } else if (source_proc == new_proc) {

                // check if source is still needed
                bool remove_communication = true;
                for (auto const &target : instance->getComputationalDag().children(source)) {
                    if (target != node && vector_schedule.assignedProcessor(target) == current_proc) {
                        remove_communication = false;
                        break;
                    }
                }

                if (remove_communication) {

                    remove_comm_nodes_from_new_to_current.push_back(source);
                    comm_from_new_proc += instance->sendCosts(new_proc, current_proc) *
                                          instance->getComputationalDag().nodeCommunicationWeight(source);
                }

            } else {

                // check if source is still needed
                bool remove_communication = true;
                bool communication_needed = true;
                for (auto const &target : instance->getComputationalDag().children(source)) {
                    if (target != node && vector_schedule.assignedProcessor(target) == current_proc) {
                        remove_communication = false;
                    }

                    if (target != node && vector_schedule.assignedProcessor(target) == new_proc) {
                        communication_needed = false;
                    }
                }

                if (remove_communication) {

                    remove_comm_nodes_from_any_to_current.push_back(source);
                    comm_from_new_proc += instance->sendCosts(new_proc, current_proc) *
                                          instance->getComputationalDag().nodeCommunicationWeight(source);
                }

                if (communication_needed) {

                    new_comm_nodes_from_any_to_new.push_back(source);
                    comm_from_old_proc += instance->sendCosts(current_proc, new_proc) *
                                          instance->getComputationalDag().nodeCommunicationWeight(source);
                }
            }
        }
    
    
    }
*/
}

void LKBufferedCommScheduler::compute_superstep_datastructures() {

    for (unsigned node = 0; node < num_nodes; node++) {

        std::vector<unsigned> step_needed(num_procs, num_steps);
        std::vector<unsigned> node_needed(num_procs, 0);
        for (const auto &target : instance->getComputationalDag().children(node)) {

            if (vector_schedule.assignedProcessor(node) != vector_schedule.assignedSuperstep(target)) {
                if (vector_schedule.assignedSuperstep(target) <
                    step_needed[vector_schedule.assignedProcessor(target)]) {
                    step_needed[vector_schedule.assignedProcessor(target)] = vector_schedule.assignedSuperstep(target);
                    node_needed[vector_schedule.assignedProcessor(target)] = target;
                }
            }
        }

        for (unsigned proc = 0; proc < num_procs; proc++) {

            if (step_needed[proc] < num_steps) {

                comm_edges[node].insert(node_needed[proc]);

                step_processor_send[vector_schedule.assignedSuperstep(node)][vector_schedule.assignedProcessor(node)] +=
                    instance->sendCosts(vector_schedule.node_to_processor_assignment[node], proc) *
                    instance->getComputationalDag().nodeCommunicationWeight(node);

                step_processor_receive[step_needed[proc] - 1][proc] +=
                    instance->sendCosts(vector_schedule.node_to_processor_assignment[node], proc) *
                    instance->getComputationalDag().nodeCommunicationWeight(node);
            }
        }
    }

    for (unsigned step = 0; step < num_steps; step++) {

        for (unsigned proc = 0; proc < num_procs; proc++) {

            for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                step_processor_work[step][proc] += instance->getComputationalDag().nodeWorkWeight(node);
            }

            if (step_processor_work[step][proc] > step_max_work[step]) {

                step_second_max_work[step] = step_max_work[step];
                step_max_work[step] = step_processor_work[step][proc];

            } else if (step_processor_work[step][proc] > step_second_max_work[step]) {

                step_second_max_work[step] = step_processor_work[step][proc];
            }

            if (step_processor_send[step][proc] > step_max_send[step]) {

                step_second_max_send[step] = step_max_send[step];
                step_max_send[step] = step_processor_send[step][proc];

            } else if (step_processor_send[step][proc] > step_second_max_send[step]) {

                step_second_max_send[step] = step_processor_send[step][proc];
            }

            if (step_processor_receive[step][proc] > step_max_receive[step]) {

                step_second_max_receive[step] = step_max_receive[step];
                step_max_receive[step] = step_processor_receive[step][proc];

            } else if (step_processor_receive[step][proc] > step_second_max_receive[step]) {

                step_second_max_receive[step] = step_processor_receive[step][proc];
            }
        }
    }
}

void LKBufferedCommScheduler::initalize_superstep_datastructures() {

    step_processor_work = std::vector<std::vector<unsigned>>(num_steps, std::vector<unsigned>(num_procs, 0));
    step_processor_send = std::vector<std::vector<unsigned>>(num_steps, std::vector<unsigned>(num_procs, 0));
    step_processor_receive = std::vector<std::vector<unsigned>>(num_steps, std::vector<unsigned>(num_procs, 0));

    step_max_work = std::vector<unsigned>(num_steps, 0);
    step_max_send = std::vector<unsigned>(num_steps, 0);
    step_max_receive = std::vector<unsigned>(num_steps, 0);

    step_second_max_work = std::vector<unsigned>(num_steps, 0);
    step_second_max_send = std::vector<unsigned>(num_steps, 0);
    step_second_max_receive = std::vector<unsigned>(num_steps, 0);

    comm_edges = std::vector<std::unordered_set<unsigned>>(num_nodes, std::unordered_set<unsigned>());
}

void LKBufferedCommScheduler::cleanup_superstep_datastructures() {

    step_processor_work.clear();
    step_processor_send.clear();
    step_processor_receive.clear();

    step_max_work.clear();
    step_max_send.clear();
    step_max_receive.clear();

    step_second_max_work.clear();
    step_second_max_send.clear();
    step_second_max_receive.clear();

    comm_edges.clear();
}

unsigned LKBufferedCommScheduler::current_costs() {

    unsigned costs = 0;
    for (unsigned step = 0; step < num_steps; step++) {
        costs += step_max_work[step];
        costs += instance->communicationCosts() * std::max(step_max_send[step], step_max_receive[step]);
    }

    return costs + (num_steps - 1) * instance->synchronisationCosts();
}


/*
unsigned LKBufferedCommScheduler::computeGainMoveCommunication(unsigned from_proc, unsigned to_proc, unsigned from_step,
                                                               unsigned to_step, unsigned comm_to_old_proc,
                                                               unsigned comm_to_new_proc, unsigned comm_vol) {

    unsigned gain = 0;

    if (from_step != to_step) {

        const unsigned remove_vol = comm_vol + comm_to_new_proc;
        if (step_max_send[from_step] > step_max_receive[from_step] &&
            step_processor_send[from_step][from_proc] > step_second_max_send[from_step]) {

            if (step_max_send[from_step] - remove_vol > step_second_max_send[from_step]) {
                gain += remove_vol;
            } else {
                gain += step_max_send[from_step] - step_second_max_send[from_step];
            }
        }

        const unsigned add_vol = comm_vol + comm_to_old_proc;
        unsigned local_max = std::max(step_max_receive[to_step], step_max_send[to_step]);
        if (local_max < step_processor_send[to_step][to_proc] + add_vol) {

            gain -= local_max - step_processor_send[to_step][to_proc] - add_vol;
        }
    } else {

        const unsigned new_old_proc_send = step_processor_send[from_step][from_proc] - (comm_vol + comm_to_new_proc);
        const unsigned new_new_proc_send = step_processor_send[from_step][to_proc] + comm_vol + comm_to_old_proc;

        if (step_max_send[from_step] == step_processor_send[from_step][from_proc]) {

            unsigned local_max =
                std::max(step_max_receive[to_step], std::max(step_second_max_send[to_step], new_old_proc_send));
            if (local_max < new_new_proc_send) {

                gain -= local_max - new_new_proc_send;
            }

        } else {

            unsigned local_max = std::max(step_max_receive[to_step], step_max_send[to_step]);
            if (local_max < new_new_proc_send) {

                gain -= local_max - new_new_proc_send;
            }
        }
    }

    return gain;
}

*/