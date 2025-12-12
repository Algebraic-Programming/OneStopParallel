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

template <typename GraphT, typename MemoryConstraintT = no_local_search_memory_constraint, bool useNodeCommunicationCostsArg = true>
class KlTotalComm : public KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg> {
  protected:
    virtual void compute_comm_gain(vertex_idx_t<Graph_t> node, unsigned currentStep, unsigned currentProc, unsigned newProc) override {
        if constexpr (KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.use_node_communication_costs) {
            if (currentProc == newProc) {
                for (const auto &target :
                     kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule.instance
                         ->getComputationalDag()
                         .children(node)) {
                    if ((current_step + 1
                             == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .vector_schedule.assignedSuperstep(target)
                         && current_proc
                                != kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedProcessor(target))
                        || (current_step
                                == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                            && current_proc
                                   == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                          .vector_schedule.assignedProcessor(target))) {
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][current_proc][2]
                            -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                    } else if ((current_step
                                    == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .vector_schedule.assignedSuperstep(target)
                                && current_proc
                                       != kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                              .vector_schedule.assignedProcessor(target))
                               || (current_step - 1
                                       == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                              .vector_schedule.assignedSuperstep(target)
                                   && current_proc
                                          == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                                 .vector_schedule.assignedProcessor(target))) {
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][current_proc][0]
                            += static_cast<double>(
                                   kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .instance->getComputationalDag()
                                       .VertexCommWeight(node))
                               + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                    }
                }

                for (const auto &source :
                     kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule.instance
                         ->getComputationalDag()
                         .parents(node)) {
                    if ((current_step - 1
                             == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .vector_schedule.assignedSuperstep(source)
                         && current_proc
                                != kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedProcessor(source))
                        || (current_step
                                == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                            && current_proc
                                   == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                          .vector_schedule.assignedProcessor(source))) {
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][current_proc][0]
                            -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                    } else if ((current_step
                                    == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .vector_schedule.assignedSuperstep(source)
                                && current_proc
                                       != kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                              .vector_schedule.assignedProcessor(source))
                               || (current_step + 1
                                       == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                              .vector_schedule.assignedSuperstep(source)
                                   && current_proc
                                          == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                                 .vector_schedule.assignedProcessor(source))) {
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][current_proc][2]
                            += static_cast<double>(
                                   kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .instance->getComputationalDag()
                                       .VertexCommWeight(source))
                               + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                    }
                }
            } else {
                // current_proc != new_proc

                for (const auto &target :
                     kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule.instance
                         ->getComputationalDag()
                         .children(node)) {
                    const unsigned &target_proc
                        = kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                              .vector_schedule.assignedProcessor(target);
                    if (target_proc == current_proc) {
                        const double loss
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->getComputationalDag()
                                      .VertexCommWeight(node))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->communicationCosts(new_proc, target_proc)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            -= loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            -= loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            -= loss;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            += loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            += loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            += loss;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(target)
                            == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                                   == current_step + 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                        }

                    } else if (target_proc == new_proc) {
                        const double gain
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->getComputationalDag()
                                      .VertexCommWeight(node))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->communicationCosts(current_proc, target_proc)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            += gain;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(target)
                            == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .VertexCommWeight(node))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;

                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .VertexCommWeight(node))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                                   < current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .VertexCommWeight(node))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                        }

                    } else {
                        assert(target_proc != current_proc && target_proc != new_proc);

                        const double gain
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->communicationCosts(new_proc, target_proc)
                                  - kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                        .instance->communicationCosts(current_proc, target_proc))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->getComputationalDag()
                                    .VertexCommWeight(node)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            += gain;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(target)
                            == current_step + 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                                   == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .VertexCommWeight(node))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                        }
                    }
                }

                for (const auto &source :
                     kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule.instance
                         ->getComputationalDag()
                         .parents(node)) {
                    const unsigned &source_proc
                        = kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                              .vector_schedule.assignedProcessor(source);
                    if (source_proc == current_proc) {
                        const double loss
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->getComputationalDag()
                                      .VertexCommWeight(source))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->communicationCosts(current_proc, new_proc)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            -= loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            -= loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            -= loss;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            += loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            += loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            += loss;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(source)
                            == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                                   == current_step - 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                        }

                    } else if (source_proc == new_proc) {
                        assert(source_proc != current_proc);
                        const double gain
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->getComputationalDag()
                                      .VertexCommWeight(source))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->communicationCosts(current_proc, new_proc)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            += gain;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(source)
                            == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .VertexCommWeight(source))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;

                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .VertexCommWeight(source))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                                   == current_step + 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .VertexCommWeight(source))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                        }

                    } else {
                        assert(source_proc != current_proc && source_proc != new_proc);
                        const double gain
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->communicationCosts(new_proc, source_proc)
                                  - kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                        .instance->communicationCosts(current_proc, source_proc))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->getComputationalDag()
                                    .VertexCommWeight(source)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            += gain;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(source)
                            == current_step - 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                                   == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .VertexCommWeight(source))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                        }
                    }
                }
            }
        } else {
            if (currentProc == newProc) {
                for (const auto &out_edge :
                     out_edges(node,
                               kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule.instance
                                   ->getComputationalDag())) {
                    const auto &target_v
                        = Traget(out_edge,
                                 kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                     .instance->getComputationalDag());
                    // for (const auto &target :
                    // kl_total<Graph_t,
                    // MemoryConstraint_t,use_node_communication_costs_arg>::current_schedule.instance->getComputationalDag().children(node)) {

                    if ((current_step + 1
                             == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .vector_schedule.assignedSuperstep(target_v)
                         && current_proc
                                != kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedProcessor(target_v))
                        || (current_step
                                == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target_v)
                            && current_proc
                                   == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                          .vector_schedule.assignedProcessor(target_v))) {
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][current_proc][2]
                            -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                    } else if ((current_step
                                    == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .vector_schedule.assignedSuperstep(target_v)
                                && current_proc
                                       != kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                              .vector_schedule.assignedProcessor(target_v))
                               || (current_step - 1
                                       == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                              .vector_schedule.assignedSuperstep(target_v)
                                   && current_proc
                                          == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                                 .vector_schedule.assignedProcessor(target_v))) {
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][current_proc][0]
                            += static_cast<double>(
                                   kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .instance->getComputationalDag()
                                       .EdgeCommWeight(out_edge))
                               + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                    }
                }

                for (const auto &in_edge :
                     in_edges(node,
                              kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule.instance
                                  ->getComputationalDag())) {
                    const auto &source_v
                        = Source(in_edge,
                                 kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                     .instance->getComputationalDag());
                    // for (const auto &source :
                    // kl_total<Graph_t,
                    // MemoryConstraint_t,use_node_communication_costs_arg>::current_schedule.instance->getComputationalDag().parents(node)) {

                    if ((current_step - 1
                             == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .vector_schedule.assignedSuperstep(source_v)
                         && current_proc
                                != kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedProcessor(source_v))
                        || (current_step
                                == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source_v)
                            && current_proc
                                   == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                          .vector_schedule.assignedProcessor(source_v))) {
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][current_proc][0]
                            -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                    } else if ((current_step
                                    == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .vector_schedule.assignedSuperstep(source_v)
                                && current_proc
                                       != kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                              .vector_schedule.assignedProcessor(source_v))
                               || (current_step + 1
                                       == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                              .vector_schedule.assignedSuperstep(source_v)
                                   && current_proc
                                          == kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                                 .vector_schedule.assignedProcessor(source_v))) {
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][current_proc][2]
                            += static_cast<double>(
                                   kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .instance->getComputationalDag()
                                       .EdgeCommWeight(in_edge))
                               + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                    }
                }
            } else {
                // current_proc != new_proc

                for (const auto &out_edge :
                     out_edges(node,
                               kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule.instance
                                   ->getComputationalDag())) {
                    const auto &target_v
                        = Traget(out_edge,
                                 kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                     .instance->getComputationalDag());
                    const unsigned &target_proc
                        = kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                              .vector_schedule.assignedProcessor(target_v);

                    if (target_proc == current_proc) {
                        const double loss
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->getComputationalDag()
                                      .EdgeCommWeight(out_edge))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->communicationCosts(new_proc, target_proc)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            -= loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            -= loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            -= loss;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            += loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            += loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            += loss;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(target_v)
                            == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target_v)
                                   == current_step + 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                        }

                    } else if (target_proc == new_proc) {
                        const double gain
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->getComputationalDag()
                                      .EdgeCommWeight(out_edge))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->communicationCosts(current_proc, target_proc)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            += gain;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(target_v)
                            == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .EdgeCommWeight(out_edge))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .EdgeCommWeight(out_edge))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target_v)
                                   == current_step - 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .EdgeCommWeight(out_edge))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                        }

                    } else {
                        assert(target_proc != current_proc && target_proc != new_proc);

                        const double gain
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->communicationCosts(new_proc, target_proc)
                                  - kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                        .instance->communicationCosts(current_proc, target_proc))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->getComputationalDag()
                                    .EdgeCommWeight(out_edge)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            += gain;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(target_v)
                            == current_step + 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target_v)
                                   == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .EdgeCommWeight(out_edge))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                        }
                    }
                }

                for (const auto &in_edge :
                     in_edges(node,
                              kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule.instance
                                  ->getComputationalDag())) {
                    const auto &source_v
                        = Source(in_edge,
                                 kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                     .instance->getComputationalDag());

                    const unsigned &source_proc
                        = kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                              .vector_schedule.assignedProcessor(source_v);
                    if (source_proc == current_proc) {
                        const double loss
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->getComputationalDag()
                                      .EdgeCommWeight(in_edge))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->communicationCosts(current_proc, new_proc)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            -= loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            -= loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            -= loss;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            += loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            += loss;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            += loss;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(source_v)
                            == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source_v)
                                   == current_step - 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;
                        }
                    } else if (source_proc == new_proc) {
                        assert(source_proc != current_proc);
                        const double gain
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->getComputationalDag()
                                      .EdgeCommWeight(in_edge))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->communicationCosts(current_proc, new_proc)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            += gain;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(source_v)
                            == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .EdgeCommWeight(in_edge))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;

                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .EdgeCommWeight(in_edge))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source_v)
                                   == current_step + 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .EdgeCommWeight(in_edge))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                        }

                    } else {
                        assert(source_proc != current_proc && source_proc != new_proc);
                        const double gain
                            = static_cast<double>(
                                  kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                      .instance->communicationCosts(new_proc, source_proc)
                                  - kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                        .instance->communicationCosts(current_proc, source_proc))
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .instance->getComputationalDag()
                                    .EdgeCommWeight(in_edge)
                              * kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                    .comm_multiplier;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][1]
                            += gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                            += gain;

                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][0]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][1]
                            -= gain;
                        kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_change_in_costs[node]
                                                                                                                     [new_proc][2]
                            -= gain;

                        if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                .vector_schedule.assignedSuperstep(source_v)
                            == current_step - 1) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][0]
                                -= kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::penalty;

                        } else if (kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source_v)
                                   == current_step) {
                            kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::node_gains[node][new_proc][2]
                                += static_cast<double>(
                                       kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::current_schedule
                                           .instance->getComputationalDag()
                                           .EdgeCommWeight(in_edge))
                                   + kl_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg>::reward;
                        }
                    }
                }
            }
        }
    }

    virtual double compute_current_costs() override {
        double workCosts = 0;
        for (unsigned step = 0;
             step < KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.num_steps();
             step++) {
            workCosts += KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.step_max_work[step];
        }

        double commCosts = 0;
        for (const auto &edge : Edges(KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance
                                          ->getComputationalDag())) {
            const auto &sourceV = Source(
                edge,
                KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance->getComputationalDag());
            const unsigned &sourceProc = KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule
                                             .vector_schedule.assignedProcessor(sourceV);
            const unsigned &targetProc
                = KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.vector_schedule
                      .assignedProcessor(Traget(edge,
                                                KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule
                                                    .instance->getComputationalDag()));

            if (sourceProc != targetProc) {
                if constexpr (KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule
                                  .use_node_communication_costs) {
                    commCosts += KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance
                                     ->getComputationalDag()
                                     .VertexCommWeight(sourceV)
                                 * KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance
                                       ->communicationCosts(sourceProc, targetProc);
                } else {
                    commCosts += KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance
                                     ->getComputationalDag()
                                     .EdgeCommWeight(edge)
                                 * KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance
                                       ->communicationCosts(sourceProc, targetProc);
                }
            }
        }

        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.current_cost
            = workCosts
              + commCosts * KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.comm_multiplier
              + (static_cast<double>(KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.num_steps())
                 - 1)
                    * KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance
                          ->synchronisationCosts();

        return KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.current_cost;
    }

  public:
    KlTotalComm() : KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>() {}

    virtual ~KlTotalComm() = default;

    virtual std::string getScheduleName() const override { return "KLTotalComm"; }
};

template <typename GraphT, typename MemoryConstraintT = no_local_search_memory_constraint, bool useNodeCommunicationCostsArg = true>
class KlTotalCommTest : public KlTotalComm<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg> {
  public:
    KlTotalCommTest() : KlTotalComm<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>() {}

    virtual ~KlTotalCommTest() = default;

    virtual std::string getScheduleName() const override { return "KLBaseTest"; }

    KlCurrentScheduleTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg> &GetCurrentSchedule() {
        return KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule;
    }

    auto &GetNodeGains() { return KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::node_gains; }

    auto &GetNodeChangeInCosts() {
        return KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::node_change_in_costs;
    }

    auto &GetMaxGainHeap() { return KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::max_gain_heap; }

    void InitializeGainHeapTest(const std::unordered_set<vertex_idx_t<Graph_t>> &nodes, double reward = 0.0, double penalty = 0.0) {
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::reward = reward;
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::penalty = penalty;

        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::initialize_gain_heap(nodes);
    }

    void TestSetupSchedule(BspSchedule<GraphT> &schedule) {
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance = &schedule.getInstance();

        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::best_schedule = &schedule;

        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::num_nodes
            = KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance->numberOfVertices();
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::num_procs
            = KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance->numberOfProcessors();

        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::set_parameters();
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::initialize_datastructures();
    }

    RETURN_STATUS ImproveScheduleTest1(BspSchedule<GraphT> &schedule) {
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance = &schedule.getInstance();

        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::best_schedule = &schedule;
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::num_nodes
            = KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance->numberOfVertices();
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::num_procs
            = KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::current_schedule.instance->numberOfProcessors();

        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::set_parameters();
        KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::initialize_datastructures();

        bool improvementFound = KlTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg>::run_local_search_simple();

        if (improvementFound) {
            return RETURN_STATUS::OSP_SUCCESS;
        } else {
            return RETURN_STATUS::BEST_FOUND;
        }
    }

    RETURN_STATUS ImproveScheduleTest2(BspSchedule<GraphT> &schedule) {
        KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance = &schedule.getInstance();

        KlTotal<GraphT, MemoryConstraintT, true>::best_schedule = &schedule;
        KlTotal<GraphT, MemoryConstraintT, true>::num_nodes
            = KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance->numberOfVertices();
        KlTotal<GraphT, MemoryConstraintT, true>::num_procs
            = KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance->numberOfProcessors();

        KlTotal<GraphT, MemoryConstraintT, true>::set_parameters();
        KlTotal<GraphT, MemoryConstraintT, true>::initialize_datastructures();

        bool improvementFound = KlTotal<GraphT, MemoryConstraintT, true>::run_local_search_unlock_delay();

        if (improvementFound) {
            return RETURN_STATUS::OSP_SUCCESS;
        } else {
            return RETURN_STATUS::BEST_FOUND;
        }
    }
};

}    // namespace osp
