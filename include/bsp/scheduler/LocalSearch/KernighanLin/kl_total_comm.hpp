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
#include <chrono>
#include <climits>
#include <string>
#include <vector>

#include "kl_total.hpp"

namespace osp {
template<typename Graph_t, typename MemoryConstraint_t = no_local_search_memory_constraint>
class kl_total_comm : public kl_total<Graph_t, MemoryConstraint_t> {

  protected:
    virtual void compute_comm_gain(vertex_idx_t<Graph_t> node, unsigned current_step, unsigned current_proc,
                                   unsigned new_proc) override {

        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.use_node_communication_costs) {

            if (current_proc == new_proc) {

                for (const auto &target :
                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().children(node)) {

                    if ((current_step + 1 ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) &&
                         current_proc !=
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target)) ||
                        (current_step ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) &&
                         current_proc ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target))) {
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][current_proc][2] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                    } else if ((current_step ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) &&
                                current_proc !=
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target)) ||
                               (current_step - 1 ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) &&
                                current_proc ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target))) {

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][current_proc][0] +=
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .vertex_comm_weight(node) +
                            kl_total<Graph_t, MemoryConstraint_t>::reward;
                    }
                }

                for (const auto &source :
                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().parents(node)) {

                    if ((current_step - 1 ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) &&
                         current_proc !=
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source)) ||
                        (current_step ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) &&
                         current_proc ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source))) {
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][current_proc][0] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                    } else if ((current_step ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) &&
                                current_proc !=
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source)) ||
                               (current_step + 1 ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) &&
                                current_proc ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source))) {

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][current_proc][2] +=
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .vertex_comm_weight(source) +
                            kl_total<Graph_t, MemoryConstraint_t>::reward;
                    }
                }
            } else {

                // current_proc != new_proc

                for (const auto &target :
                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().children(node)) {

                    const unsigned &target_proc =
                        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target);
                    if (target_proc == current_proc) {

                        const double loss =
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .vertex_comm_weight(node) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(new_proc, target_proc) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] -= loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= loss;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] += loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] += loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] += loss;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) ==
                            current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) ==
                                   current_step + 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                        }

                    } else if (target_proc == new_proc) {

                        const double gain = (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                                .vertex_comm_weight(node) *
                                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(
                                                current_proc, target_proc) *
                                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] += gain;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) ==
                            current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .vertex_comm_weight(node) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .vertex_comm_weight(node) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) <
                                   current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .vertex_comm_weight(node) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                        }

                    } else {

                        assert(target_proc != current_proc && target_proc != new_proc);

                        const double gain =
                            (double)(kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(new_proc,
                                                                                                      target_proc) -
                                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(current_proc,
                                                                                                      target_proc)) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().vertex_comm_weight(
                                node) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] += gain;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) ==
                            current_step + 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target) ==
                                   current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .vertex_comm_weight(node) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                        }
                    }
                }

                for (const auto &source :
                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().parents(node)) {

                    const unsigned &source_proc =
                        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source);
                    if (source_proc == current_proc) {

                        const double loss =
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .vertex_comm_weight(source) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(current_proc, new_proc) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] -= loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= loss;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] += loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] += loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] += loss;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) ==
                            current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) ==
                                   current_step - 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                        }

                    } else if (source_proc == new_proc) {

                        assert(source_proc != current_proc);
                        const double gain =
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .vertex_comm_weight(source) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(current_proc, new_proc) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] += gain;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) ==
                            current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .vertex_comm_weight(source) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .vertex_comm_weight(source) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) ==
                                   current_step + 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .vertex_comm_weight(source) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                        }

                    } else {

                        assert(source_proc != current_proc && source_proc != new_proc);
                        const double gain =
                            (double)(kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(new_proc,
                                                                                                      source_proc) -
                                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(current_proc,
                                                                                                      source_proc)) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().vertex_comm_weight(
                                source) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] += gain;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) ==
                            current_step - 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source) ==
                                   current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .vertex_comm_weight(source) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                        }
                    }
                }
            }
        } else {

            if (current_proc == new_proc) {

                for (const auto &out_edge :
                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().out_edges(node)) {
                    const auto &target_v =
                        target(out_edge, kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag());
                    // for (const auto &target :
                    // kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().children(node)) {

                    if ((current_step + 1 ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) &&
                         current_proc !=
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target_v)) ||
                        (current_step ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) &&
                         current_proc ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target_v))) {

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][current_proc][2] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                    } else if ((current_step ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) &&
                                current_proc !=
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target_v)) ||
                               (current_step - 1 ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) &&
                                current_proc ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target_v))) {

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][current_proc][0] +=
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .edge_comm_weight(out_edge) +
                            kl_total<Graph_t, MemoryConstraint_t>::reward;
                    }
                }

                for (const auto &in_edge :
                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().in_edges(node)) {

                    const auto &source_v =
                        source(in_edge, kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag());
                    // for (const auto &source :
                    // kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().parents(node)) {

                    if ((current_step - 1 ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) &&
                         current_proc !=
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source_v)) ||
                        (current_step ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) &&
                         current_proc ==
                             kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source_v))) {

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][current_proc][0] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                    } else if ((current_step ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) &&
                                current_proc !=
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source_v)) ||
                               (current_step + 1 ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) &&
                                current_proc ==
                                    kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source_v))) {

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][current_proc][2] +=
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .edge_comm_weight(in_edge) +
                            kl_total<Graph_t, MemoryConstraint_t>::reward;
                    }
                }
            } else {

                // current_proc != new_proc

                for (const auto &out_edge :
                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().out_edges(node)) {

                    const auto &target_v =
                        target(out_edge, kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag());
                    const unsigned &target_proc =
                        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(target_v);

                    if (target_proc == current_proc) {

                        const double loss =
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .edge_comm_weight(out_edge) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(new_proc, target_proc) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] -= loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= loss;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] += loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] += loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] += loss;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) ==
                            current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) ==
                                   current_step + 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                        }

                    } else if (target_proc == new_proc) {

                        const double gain = (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                                .edge_comm_weight(out_edge) *
                                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(
                                                current_proc, target_proc) *
                                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] += gain;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) ==
                            current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .edge_comm_weight(out_edge) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .edge_comm_weight(out_edge) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) ==
                                   current_step - 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .edge_comm_weight(out_edge) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                        }

                    } else {

                        assert(target_proc != current_proc && target_proc != new_proc);

                        const double gain =
                            (double)(kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(new_proc,
                                                                                                      target_proc) -
                                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(current_proc,
                                                                                                      target_proc)) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().edge_comm_weight(
                                out_edge) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] += gain;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) ==
                            current_step + 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(target_v) ==
                                   current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .edge_comm_weight(out_edge) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                        }
                    }
                }

                for (const auto &in_edge :
                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().in_edges(node)) {
                    const auto &source_v =
                        source(in_edge, kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag());

                    const unsigned &source_proc =
                        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source_v);
                    if (source_proc == current_proc) {

                        const double loss =
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .edge_comm_weight(in_edge) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(current_proc, new_proc) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] -= loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] -= loss;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] += loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] += loss;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] += loss;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) ==
                            current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) ==
                                   current_step - 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;
                        }
                    } else if (source_proc == new_proc) {

                        assert(source_proc != current_proc);
                        const double gain =
                            (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                .edge_comm_weight(in_edge) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(current_proc, new_proc) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] += gain;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) ==
                            current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .edge_comm_weight(in_edge) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .edge_comm_weight(in_edge) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) ==
                                   current_step + 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .edge_comm_weight(in_edge) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                        }

                    } else {

                        assert(source_proc != current_proc && source_proc != new_proc);
                        const double gain =
                            (double)(kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(new_proc,
                                                                                                      source_proc) -
                                     kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(current_proc,
                                                                                                      source_proc)) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().edge_comm_weight(
                                in_edge) *
                            kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][1] += gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] += gain;

                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][0] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][1] -= gain;
                        kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs[node][new_proc][2] -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) ==
                            current_step - 1) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][0] -= kl_total<Graph_t, MemoryConstraint_t>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedSuperstep(source_v) ==
                                   current_step) {

                            kl_total<Graph_t, MemoryConstraint_t>::node_gains[node][new_proc][2] +=
                                (double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()
                                    .edge_comm_weight(in_edge) +
                                kl_total<Graph_t, MemoryConstraint_t>::reward;
                        }
                    }
                }
            }
        }
    }

    virtual double compute_current_costs() override {

        double work_costs = 0;
        for (unsigned step = 0; step < kl_total<Graph_t, MemoryConstraint_t>::current_schedule.num_steps(); step++) {
            work_costs += kl_total<Graph_t, MemoryConstraint_t>::current_schedule.step_max_work[step];
        }

        double comm_costs = 0;
        for (const auto &edge : kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().edges()) {

            const auto &source_v = source(edge, kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag());
            const unsigned &source_proc = kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(source_v);
            const unsigned &target_proc = kl_total<Graph_t, MemoryConstraint_t>::current_schedule.vector_schedule.assignedProcessor(
                target(edge, kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag()));

            if (source_proc != target_proc) {

                if (kl_total<Graph_t, MemoryConstraint_t>::current_schedule.use_node_communication_costs) {
                    comm_costs +=
                        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().vertex_comm_weight(source_v) *
                        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(source_proc, target_proc);
                } else {
                    comm_costs +=
                        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->getComputationalDag().edge_comm_weight(edge) *
                        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->communicationCosts(source_proc, target_proc);
                }
            }
        }

        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.current_cost =
            work_costs + comm_costs * kl_total<Graph_t, MemoryConstraint_t>::current_schedule.comm_multiplier +
            ((double)kl_total<Graph_t, MemoryConstraint_t>::current_schedule.num_steps() - 1) *
                kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->synchronisationCosts();

        return kl_total<Graph_t, MemoryConstraint_t>::current_schedule.current_cost;
    }

  public:
    kl_total_comm(bool use_node_communication_costs_ = true) : kl_total<Graph_t, MemoryConstraint_t>(use_node_communication_costs_) {}

    virtual ~kl_total_comm() = default;

    virtual std::string getScheduleName() const override { return "KLTotalComm"; }
};

template<typename Graph_t, typename MemoryConstraint_t = no_local_search_memory_constraint>
class kl_total_comm_test : public kl_total_comm<Graph_t, MemoryConstraint_t> {

  public:
    kl_total_comm_test() : kl_total_comm<Graph_t, MemoryConstraint_t>() {}

    virtual ~kl_total_comm_test() = default;

    virtual std::string getScheduleName() const override { return "KLBaseTest"; }

    kl_current_schedule_total<Graph_t, MemoryConstraint_t> &get_current_schedule() {
        return kl_total<Graph_t, MemoryConstraint_t>::current_schedule;
    }

    auto &get_node_gains() { return kl_total<Graph_t, MemoryConstraint_t>::node_gains; }
    auto &get_node_change_in_costs() { return kl_total<Graph_t, MemoryConstraint_t>::node_change_in_costs; }
    auto &get_max_gain_heap() { return kl_total<Graph_t, MemoryConstraint_t>::max_gain_heap; }

    void initialize_gain_heap_test(const std::unordered_set<vertex_idx_t<Graph_t>> &nodes, double reward_ = 0.0,
                                   double penalty_ = 0.0) {
        kl_total<Graph_t, MemoryConstraint_t>::reward = reward_;
        kl_total<Graph_t, MemoryConstraint_t>::penalty = penalty_;

        kl_total<Graph_t, MemoryConstraint_t>::initialize_gain_heap(nodes);
    }

    void test_setup_schedule(BspSchedule<Graph_t> &schedule) {

        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance = &schedule.getInstance();

        kl_total<Graph_t, MemoryConstraint_t>::best_schedule = &schedule;

        kl_total<Graph_t, MemoryConstraint_t>::num_nodes = kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->numberOfVertices();
        kl_total<Graph_t, MemoryConstraint_t>::num_procs = kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->numberOfProcessors();

        kl_total<Graph_t, MemoryConstraint_t>::set_parameters();
        kl_total<Graph_t, MemoryConstraint_t>::initialize_datastructures();
    }

    RETURN_STATUS improve_schedule_test_1(BspSchedule<Graph_t> &schedule) {

        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance = &schedule.getInstance();

        kl_total<Graph_t, MemoryConstraint_t>::best_schedule = &schedule;
        kl_total<Graph_t, MemoryConstraint_t>::num_nodes = kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->numberOfVertices();
        kl_total<Graph_t, MemoryConstraint_t>::num_procs = kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->numberOfProcessors();

        kl_total<Graph_t, MemoryConstraint_t>::set_parameters();
        kl_total<Graph_t, MemoryConstraint_t>::initialize_datastructures();

        bool improvement_found = kl_total<Graph_t, MemoryConstraint_t>::run_local_search_simple();

        

        if (improvement_found)
            return SUCCESS;
        else
            return BEST_FOUND;
    }

    RETURN_STATUS improve_schedule_test_2(BspSchedule<Graph_t> &schedule) {

        kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance = &schedule.getInstance();

        kl_total<Graph_t, MemoryConstraint_t>::best_schedule = &schedule;
        kl_total<Graph_t, MemoryConstraint_t>::num_nodes = kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->numberOfVertices();
        kl_total<Graph_t, MemoryConstraint_t>::num_procs = kl_total<Graph_t, MemoryConstraint_t>::current_schedule.instance->numberOfProcessors();

        kl_total<Graph_t, MemoryConstraint_t>::set_parameters();
        kl_total<Graph_t, MemoryConstraint_t>::initialize_datastructures();

        bool improvement_found = kl_total<Graph_t, MemoryConstraint_t>::run_local_search_unlock_delay();

       

        if (improvement_found)
            return SUCCESS;
        else
            return BEST_FOUND;
    }
};

} // namespace osp