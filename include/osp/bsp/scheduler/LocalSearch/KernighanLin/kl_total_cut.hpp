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

template <typename GraphT, typename MemoryConstraintT = NoLocalSearchMemoryConstraint, bool UseNodeCommunicationCostsArg = true>
class KlTotalCut : public KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg> {
  protected:
    double maxEdgeWeight_ = 0.0;

    virtual void compute_comm_gain(VertexIdxT<GraphT> node, unsigned currentStep, unsigned currentProc, unsigned newProc) override {
        if constexpr (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.use_node_communication_costs) {
            if (currentProc == newProc) {
                for (const auto &target :
                     KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                         ->getComputationalDag()
                         .children(node)) {
                    const unsigned &targetProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .vector_schedule.assignedProcessor(target);
                    const double loss
                        = static_cast<double>(
                              KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                  ->getComputationalDag()
                                  .vertex_comm_weight(node))
                          * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                ->communicationCosts(newProc, targetProc)
                          * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                    if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                            .assignedSuperstep(target)
                        == currentStep) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] -= loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] -= loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            += loss;

                    } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                   .assignedSuperstep(target)
                               == currentStep + 1) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            -= loss;

                    } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                   .assignedSuperstep(target)
                               == currentStep - 1) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            -= loss;
                    }

                    if ((currentStep + 1
                             == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                    .assignedSuperstep(target)
                         && currentProc
                                != KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedProcessor(target))
                        || (currentStep
                                == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                            && currentProc
                                   == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                          .vector_schedule.assignedProcessor(target))) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][currentProc][2]
                            -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                    } else if ((currentStep
                                    == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .vector_schedule.assignedSuperstep(target)
                                && currentProc
                                       != KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .vector_schedule.assignedProcessor(target))
                               || (currentStep - 1
                                       == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .vector_schedule.assignedSuperstep(target)
                                   && currentProc
                                          == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                 .vector_schedule.assignedProcessor(target))) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][currentProc][0]
                            += static_cast<double>(
                                   KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                       ->getComputationalDag()
                                       .vertex_comm_weight(node))
                               + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                    }
                }

                for (const auto &source :
                     KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                         ->getComputationalDag()
                         .parents(node)) {
                    const unsigned &sourceProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .vector_schedule.assignedProcessor(source);
                    const double loss
                        = static_cast<double>(
                              KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                  ->getComputationalDag()
                                  .vertex_comm_weight(source))
                          * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                ->communicationCosts(newProc, sourceProc)
                          * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                    if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                            .assignedSuperstep(source)
                        == currentStep) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] -= loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] -= loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            += loss;

                    } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                   .assignedSuperstep(source)
                               == currentStep + 1) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            -= loss;

                    } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                   .assignedSuperstep(source)
                               == currentStep - 1) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            -= loss;
                    }

                    if ((currentStep - 1
                             == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                    .assignedSuperstep(source)
                         && currentProc
                                != KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedProcessor(source))
                        || (currentStep
                                == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                            && currentProc
                                   == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                          .vector_schedule.assignedProcessor(source))) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][currentProc][0]
                            -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                    } else if ((currentStep
                                    == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .vector_schedule.assignedSuperstep(source)
                                && currentProc
                                       != KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .vector_schedule.assignedProcessor(source))
                               || (currentStep + 1
                                       == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .vector_schedule.assignedSuperstep(source)
                                   && currentProc
                                          == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                 .vector_schedule.assignedProcessor(source))) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][currentProc][2]
                            += static_cast<double>(
                                   KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                       ->getComputationalDag()
                                       .vertex_comm_weight(source))
                               + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                    }
                }
            } else {
                // current_proc != new_proc

                for (const auto &target :
                     KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                         ->getComputationalDag()
                         .children(node)) {
                    const unsigned &targetProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .vector_schedule.assignedProcessor(target);
                    if (targetProc == currentProc) {
                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(target)
                            == currentStep) {
                            const double loss
                                = static_cast<double>(
                                      KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                          ->getComputationalDag()
                                          .vertex_comm_weight(node))
                                  * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                        ->communicationCosts(newProc, targetProc)
                                  * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] -= loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] -= loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] -= loss;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                                += loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                                += loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                                += loss;
                        }

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(target)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                                   == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                        }

                    } else if (targetProc == newProc) {
                        const double gain
                            = static_cast<double>(
                                  KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                      ->getComputationalDag()
                                      .vertex_comm_weight(node))
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                    ->communicationCosts(currentProc, targetProc)
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(target)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                                -= gain;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                                   == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                                -= gain;
                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                                   == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                                -= gain;
                        }

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(target)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .vertex_comm_weight(node))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .vertex_comm_weight(node))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                                   < currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .vertex_comm_weight(node))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                        }

                    } else {
                        assert(targetProc != currentProc && targetProc != newProc);

                        const double gain
                            = static_cast<double>(
                                  KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                      ->communicationCosts(newProc, targetProc)
                                  - KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                        ->communicationCosts(currentProc, targetProc))
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                    ->getComputationalDag()
                                    .vertex_comm_weight(node)
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] += gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += gain;

                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            -= gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                            -= gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            -= gain;

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(target)
                            == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(target)
                                   == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .vertex_comm_weight(node))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                        }
                    }
                }

                for (const auto &source :
                     KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                         ->getComputationalDag()
                         .parents(node)) {
                    const unsigned &sourceProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .vector_schedule.assignedProcessor(source);
                    if (sourceProc == currentProc) {
                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(source)
                            == currentStep) {
                            const double loss
                                = static_cast<double>(
                                      KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                          ->getComputationalDag()
                                          .vertex_comm_weight(source))
                                  * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                        ->communicationCosts(currentProc, newProc)
                                  * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] -= loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] -= loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] -= loss;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                                += loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                                += loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                                += loss;
                        }

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(source)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                                   == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                        }

                    } else if (sourceProc == newProc) {
                        assert(sourceProc != currentProc);
                        const double gain
                            = static_cast<double>(
                                  KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                      ->getComputationalDag()
                                      .vertex_comm_weight(source))
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                    ->communicationCosts(currentProc, newProc)
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(source)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                                -= gain;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                                   == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                                -= gain;
                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                                   == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                                -= gain;
                        }

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(source)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .vertex_comm_weight(source))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .vertex_comm_weight(source))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                                   == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .vertex_comm_weight(source))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                        }

                    } else {
                        assert(sourceProc != currentProc && sourceProc != newProc);
                        const double gain
                            = static_cast<double>(
                                  KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                      ->communicationCosts(newProc, sourceProc)
                                  - KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                        ->communicationCosts(currentProc, sourceProc))
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                    ->getComputationalDag()
                                    .vertex_comm_weight(source)
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] += gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += gain;

                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            -= gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                            -= gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            -= gain;

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(source)
                            == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(source)
                                   == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .vertex_comm_weight(source))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                        }
                    }
                }
            }
        } else {
            if (currentProc == newProc) {
                for (const auto &outEdge :
                     out_edges(node,
                               KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                   ->getComputationalDag())) {
                    const auto &targetV = target(outEdge,
                                                 KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .instance->getComputationalDag());
                    const unsigned &targetProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .vector_schedule.assignedProcessor(targetV);

                    const double loss
                        = static_cast<double>(
                              KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                  ->getComputationalDag()
                                  .edge_comm_weight(outEdge))
                          * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                ->communicationCosts(newProc, targetProc)
                          * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                    if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                            .assignedSuperstep(targetV)
                        == currentStep) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] -= loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] -= loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            += loss;

                    } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                   .assignedSuperstep(targetV)
                               == currentStep + 1) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            -= loss;

                    } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                   .assignedSuperstep(targetV)
                               == currentStep - 1) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            -= loss;
                    }

                    if ((currentStep + 1
                             == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                    .assignedSuperstep(targetV)
                         && currentProc
                                != KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedProcessor(targetV))
                        || (currentStep
                                == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(targetV)
                            && currentProc
                                   == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                          .vector_schedule.assignedProcessor(targetV))) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][currentProc][2]
                            -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                    } else if ((currentStep
                                    == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .vector_schedule.assignedSuperstep(targetV)
                                && currentProc
                                       != KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .vector_schedule.assignedProcessor(targetV))
                               || (currentStep - 1
                                       == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .vector_schedule.assignedSuperstep(targetV)
                                   && currentProc
                                          == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                 .vector_schedule.assignedProcessor(targetV))) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][currentProc][0]
                            += static_cast<double>(
                                   KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                       ->getComputationalDag()
                                       .edge_comm_weight(outEdge))
                               + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                    }
                }

                for (const auto &inEdge : in_edges(node,
                                                   KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                       .instance->getComputationalDag())) {
                    const auto &sourceV = source(inEdge,
                                                 KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .instance->getComputationalDag());
                    const unsigned &sourceProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .vector_schedule.assignedProcessor(sourceV);

                    const double loss
                        = static_cast<double>(
                              KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                  ->getComputationalDag()
                                  .edge_comm_weight(inEdge))
                          * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                ->communicationCosts(newProc, sourceProc)
                          * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                    if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                            .assignedSuperstep(sourceV)
                        == currentStep) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] -= loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] -= loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            += loss;

                    } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                   .assignedSuperstep(sourceV)
                               == currentStep + 1) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            -= loss;

                    } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                   .assignedSuperstep(sourceV)
                               == currentStep - 1) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += loss;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            -= loss;
                    }

                    if ((currentStep - 1
                             == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                    .assignedSuperstep(sourceV)
                         && currentProc
                                != KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedProcessor(sourceV))
                        || (currentStep
                                == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(sourceV)
                            && currentProc
                                   == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                          .vector_schedule.assignedProcessor(sourceV))) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][currentProc][0]
                            -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                    } else if ((currentStep
                                    == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .vector_schedule.assignedSuperstep(sourceV)
                                && currentProc
                                       != KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .vector_schedule.assignedProcessor(sourceV))
                               || (currentStep + 1
                                       == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .vector_schedule.assignedSuperstep(sourceV)
                                   && currentProc
                                          == KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                 .vector_schedule.assignedProcessor(sourceV))) {
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][currentProc][2]
                            += static_cast<double>(
                                   KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                       ->getComputationalDag()
                                       .edge_comm_weight(inEdge))
                               + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                    }
                }
            } else {
                // current_proc != new_proc

                for (const auto &outEdge :
                     out_edges(node,
                               KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                   ->getComputationalDag())) {
                    const auto &targetV = target(outEdge,
                                                 KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .instance->getComputationalDag());
                    const unsigned &targetProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .vector_schedule.assignedProcessor(targetV);

                    if (targetProc == currentProc) {
                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(targetV)
                            == currentStep) {
                            const double loss
                                = static_cast<double>(
                                      KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                          ->getComputationalDag()
                                          .edge_comm_weight(outEdge))
                                  * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                        ->communicationCosts(newProc, targetProc)
                                  * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] -= loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] -= loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] -= loss;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                                += loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                                += loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                                += loss;
                        }

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(targetV)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(targetV)
                                   == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                        }

                    } else if (targetProc == newProc) {
                        const double gain
                            = static_cast<double>(
                                  KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                      ->getComputationalDag()
                                      .edge_comm_weight(outEdge))
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                    ->communicationCosts(currentProc, targetProc)
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(targetV)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                                -= gain;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(targetV)
                                   == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                                -= gain;
                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(targetV)
                                   == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                                -= gain;
                        }

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(targetV)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .edge_comm_weight(outEdge))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .edge_comm_weight(outEdge))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(targetV)
                                   == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .edge_comm_weight(outEdge))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                        }

                    } else {
                        assert(targetProc != currentProc && targetProc != newProc);

                        const double gain
                            = static_cast<double>(
                                  KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                      ->communicationCosts(newProc, targetProc)
                                  - KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                        ->communicationCosts(currentProc, targetProc))
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                    ->getComputationalDag()
                                    .edge_comm_weight(outEdge)
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] += gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += gain;

                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            -= gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                            -= gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            -= gain;

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(targetV)
                            == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(targetV)
                                   == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .edge_comm_weight(outEdge))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                        }
                    }
                }

                for (const auto &inEdge : in_edges(node,
                                                   KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                       .instance->getComputationalDag())) {
                    const auto &sourceV = source(inEdge,
                                                 KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .instance->getComputationalDag());

                    const unsigned &sourceProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                                     .vector_schedule.assignedProcessor(sourceV);
                    if (sourceProc == currentProc) {
                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(sourceV)
                            == currentStep) {
                            const double loss
                                = static_cast<double>(
                                      KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                          ->getComputationalDag()
                                          .edge_comm_weight(inEdge))
                                  * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                        ->communicationCosts(currentProc, newProc)
                                  * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] -= loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] -= loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] -= loss;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                                += loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                                += loss;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                                += loss;
                        }

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(sourceV)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(sourceV)
                                   == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;
                        }
                    } else if (sourceProc == newProc) {
                        assert(sourceProc != currentProc);
                        const double gain
                            = static_cast<double>(
                                  KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                      ->getComputationalDag()
                                      .edge_comm_weight(inEdge))
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                    ->communicationCosts(currentProc, newProc)
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(sourceV)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                                -= gain;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(sourceV)
                                   == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                                -= gain;
                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(sourceV)
                                   == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += gain;
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                                -= gain;
                        }

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(sourceV)
                            == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .edge_comm_weight(inEdge))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;

                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .edge_comm_weight(inEdge))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(sourceV)
                                   == currentStep + 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .edge_comm_weight(inEdge))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                        }

                    } else {
                        assert(sourceProc != currentProc && sourceProc != newProc);
                        const double gain
                            = static_cast<double>(
                                  KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                      ->communicationCosts(newProc, sourceProc)
                                  - KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                        ->communicationCosts(currentProc, sourceProc))
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                    ->getComputationalDag()
                                    .edge_comm_weight(inEdge)
                              * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier;

                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0] += gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][1] += gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2] += gain;

                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][0]
                            -= gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][1]
                            -= gain;
                        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_change_in_costs[node][newProc][2]
                            -= gain;

                        if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.vector_schedule
                                .assignedSuperstep(sourceV)
                            == currentStep - 1) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][0]
                                -= KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::penalty;

                        } else if (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                       .vector_schedule.assignedSuperstep(sourceV)
                                   == currentStep) {
                            KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::node_gains[node][newProc][2]
                                += static_cast<double>(
                                       KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                           .instance->getComputationalDag()
                                           .edge_comm_weight(inEdge))
                                   + KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::reward;
                        }
                    }
                }
            }
        }
    }

    virtual double compute_current_costs() override {
        double workCosts = 0;
        for (unsigned step = 0;
             step < KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.num_steps();
             step++) {
            workCosts += KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.step_max_work[step];
        }

        double commCosts = 0;
        for (const auto &edge : edges(KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                          ->getComputationalDag())) {
            const VertexIdxT<GraphT> &sourceV = source(
                edge,
                KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance->getComputationalDag());
            const VertexIdxT<GraphT> &targetV = target(
                edge,
                KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance->getComputationalDag());
            const unsigned &sourceProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                             .vector_schedule.assignedProcessor(sourceV);
            const unsigned &targetProc = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                             .vector_schedule.assignedProcessor(targetV);
            const unsigned &sourceStep = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                             .vector_schedule.assignedSuperstep(sourceV);
            const unsigned &targetStep = KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                             .vector_schedule.assignedSuperstep(targetV);

            if (sourceProc != targetProc || sourceStep != targetStep) {
                if constexpr (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                  .use_node_communication_costs) {
                    commCosts += KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                     ->getComputationalDag()
                                     .vertex_comm_weight(sourceV)
                                 * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                       ->communicationCosts(sourceProc, targetProc);
                } else {
                    commCosts += KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                     ->getComputationalDag()
                                     .edge_comm_weight(edge)
                                 * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.instance
                                       ->communicationCosts(sourceProc, targetProc);
                }
            }
        }

        KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.current_cost
            = workCosts
              + commCosts * KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.comm_multiplier
              + (KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.num_steps() - 1)
                    * static_cast<double>(KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule
                                              .instance->synchronisationCosts());

        return KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>::current_schedule.current_cost;
    }

  public:
    KlTotalCut() : KlTotal<GraphT, MemoryConstraintT, UseNodeCommunicationCostsArg>() {}

    virtual ~KlTotalCut() = default;

    virtual std::string getScheduleName() const override { return "KLTotalCut"; }
};

template <typename GraphT, typename MemoryConstraintT = NoLocalSearchMemoryConstraint>
class KlTotalCutTest : public KlTotalCut<GraphT, MemoryConstraintT, true> {
  public:
    KlTotalCutTest() : KlTotalCut<GraphT, MemoryConstraintT, true>() {}

    virtual ~KlTotalCutTest() = default;

    virtual std::string getScheduleName() const override { return "KLTotalCutTest"; }

    KlCurrentScheduleTotal<GraphT, MemoryConstraintT, true> &GetCurrentSchedule() {
        return KlTotal<GraphT, MemoryConstraintT, true>::current_schedule;
    }

    auto &GetNodeGains() { return KlTotal<GraphT, MemoryConstraintT, true>::node_gains; }

    auto &GetNodeChangeInCosts() { return KlTotal<GraphT, MemoryConstraintT, true>::node_change_in_costs; }

    auto &GetMaxGainHeap() { return KlTotal<GraphT, MemoryConstraintT, true>::max_gain_heap; }

    void InitializeGainHeapTest(const std::unordered_set<VertexIdxT<GraphT>> &nodes, double reward = 0.0, double penalty = 0.0) {
        KlTotal<GraphT, MemoryConstraintT, true>::reward = reward;
        KlTotal<GraphT, MemoryConstraintT, true>::penalty = penalty;

        KlTotal<GraphT, MemoryConstraintT, true>::initialize_gain_heap(nodes);
    }

    void TestSetupSchedule(BspSchedule<GraphT> &schedule) {
        KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance = &schedule.getInstance();

        KlTotal<GraphT, MemoryConstraintT, true>::best_schedule = &schedule;

        KlTotal<GraphT, MemoryConstraintT, true>::num_nodes
            = KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance->numberOfVertices();
        KlTotal<GraphT, MemoryConstraintT, true>::num_procs
            = KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance->numberOfProcessors();

        KlTotal<GraphT, MemoryConstraintT, true>::set_parameters();
        KlTotal<GraphT, MemoryConstraintT, true>::initialize_datastructures();
    }

    RETURN_STATUS ImproveScheduleTest1(BspSchedule<GraphT> &schedule) {
        KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance = &schedule.getInstance();

        KlTotal<GraphT, MemoryConstraintT, true>::best_schedule = &schedule;
        KlTotal<GraphT, MemoryConstraintT, true>::num_nodes
            = KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance->numberOfVertices();
        KlTotal<GraphT, MemoryConstraintT, true>::num_procs
            = KlTotal<GraphT, MemoryConstraintT, true>::current_schedule.instance->numberOfProcessors();

        KlTotal<GraphT, MemoryConstraintT, true>::set_parameters();
        KlTotal<GraphT, MemoryConstraintT, true>::initialize_datastructures();

        bool improvementFound = KlTotal<GraphT, MemoryConstraintT, true>::run_local_search_simple();

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
