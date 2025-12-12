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

#include "kl_base.hpp"

namespace osp {

template <typename GraphT, typename MemoryConstraintT, bool useNodeCommunicationCostsArg>
class KlCurrentScheduleTotal : public KlCurrentSchedule<GraphT, MemoryConstraintT> {
  public:
    KlCurrentScheduleTotal(IklCostFunction *costF) : KlCurrentSchedule<GraphT, MemoryConstraintT>(costF) {}

    double commMultiplier_ = 1.0;
    constexpr static bool useNodeCommunicationCosts_ = use_node_communication_costs_arg || not HasEdgeWeightsV<Graph_t>;
};

template <typename GraphT, typename MemoryConstraintT, bool useNodeCommunicationCostsArg>
class KlTotal : public KlBase<GraphT, MemoryConstraintT> {
  protected:
    KlCurrentScheduleTotal<GraphT, MemoryConstraintT, useNodeCommunicationCostsArg> currentSchedule_;

    v_commw_t<Graph_t> nodeCommSelectionThreshold_ = 0;
    double maxEdgeWeight_ = 0.0;

    virtual void initialize_datastructures() override {
#ifdef KL_DEBUG
        std::cout << "KLTotal initialize datastructures" << std::endl;
#endif

        KlBase<GraphT, MemoryConstraintT>::initialize_datastructures();

        v_commw_t<Graph_t> maxEdgeWeight = 0;
        v_workw_t<Graph_t> maxNodeWeight = 0;

        for (const auto vertex : currentSchedule_.instance->getComputationalDag().vertices()) {
            if (is_sink(vertex, currentSchedule_.instance->getComputationalDag())) {
                continue;
            }

            maxEdgeWeight = std::max(max_edge_weight_, currentSchedule_.instance->getComputationalDag().VertexCommWeight(vertex));

            maxNodeWeight = std::max(max_node_weight_, currentSchedule_.instance->getComputationalDag().VertexWorkWeight(vertex));
        }

        if constexpr (not currentSchedule_.use_node_communication_costs) {
            maxEdgeWeight = 0;

            for (const auto &edge : Edges(currentSchedule_.instance->getComputationalDag())) {
                maxEdgeWeight = std::max(max_edge_weight_, currentSchedule_.instance->getComputationalDag().EdgeCommWeight(edge));
            }
        }

        maxEdgeWeight_ = max_edge_weight_ + max_node_weight_;

        KlBase<GraphT, MemoryConstraintT>::parameters.initial_penalty
            = maxEdgeWeight_ * currentSchedule_.comm_multiplier * currentSchedule_.instance->CommunicationCosts();

        KlBase<GraphT, MemoryConstraintT>::parameters.gain_threshold
            = maxEdgeWeight_ * currentSchedule_.comm_multiplier * currentSchedule_.instance->CommunicationCosts();
    }

    virtual void update_reward_penalty() override {
        if (currentSchedule_.current_violations.size() <= KlBase<GraphT, MemoryConstraintT>::parameters.violations_threshold) {
            KlBase<GraphT, MemoryConstraintT>::penalty = KlBase<GraphT, MemoryConstraintT>::parameters.initial_penalty;
            KlBase<GraphT, MemoryConstraintT>::reward = 0.0;

        } else {
            KlBase<GraphT, MemoryConstraintT>::parameters.violations_threshold = 0;

            KlBase<GraphT, MemoryConstraintT>::penalty = std::log((currentSchedule_.current_violations.size())) * maxEdgeWeight_
                                                         * currentSchedule_.comm_multiplier
                                                         * currentSchedule_.instance->CommunicationCosts();

            KlBase<GraphT, MemoryConstraintT>::reward = std::sqrt((currentSchedule_.current_violations.size() + 4))
                                                        * maxEdgeWeight_ * currentSchedule_.comm_multiplier
                                                        * currentSchedule_.instance->CommunicationCosts();
        }
    }

    virtual void set_initial_reward_penalty() override {
        KlBase<GraphT, MemoryConstraintT>::penalty = KlBase<GraphT, MemoryConstraintT>::parameters.initial_penalty;
        KlBase<GraphT, MemoryConstraintT>::reward
            = maxEdgeWeight_ * currentSchedule_.comm_multiplier * currentSchedule_.instance->CommunicationCosts();
    }

    virtual void select_nodes_comm() override {
        if constexpr (currentSchedule_.use_node_communication_costs) {
            for (const auto &node : currentSchedule_.instance->getComputationalDag().vertices()) {
                for (const auto &source : currentSchedule_.instance->getComputationalDag().parents(node)) {
                    if (currentSchedule_.vector_schedule.assignedProcessor(node)
                        != currentSchedule_.vector_schedule.assignedProcessor(source)) {
                        if (current_schedule.instance->getComputationalDag().VertexCommWeight(node)
                            > node_comm_selection_threshold) {
                            KlBase<GraphT, MemoryConstraintT>::node_selection.insert(node);
                            break;
                        }
                    }
                }

                for (const auto &target : currentSchedule_.instance->getComputationalDag().children(node)) {
                    if (currentSchedule_.vector_schedule.assignedProcessor(node)
                        != currentSchedule_.vector_schedule.assignedProcessor(target)) {
                        if (current_schedule.instance->getComputationalDag().VertexCommWeight(node)
                            > node_comm_selection_threshold) {
                            KlBase<GraphT, MemoryConstraintT>::node_selection.insert(node);
                            break;
                        }
                    }
                }
            }

        } else {
            for (const auto &node : currentSchedule_.instance->getComputationalDag().vertices()) {
                for (const auto &inEdge : InEdges(node, currentSchedule_.instance->getComputationalDag())) {
                    const auto &sourceV = Source(inEdge, currentSchedule_.instance->getComputationalDag());
                    if (currentSchedule_.vector_schedule.assignedProcessor(node)
                        != currentSchedule_.vector_schedule.assignedProcessor(sourceV)) {
                        if (current_schedule.instance->getComputationalDag().EdgeCommWeight(in_edge)
                            > node_comm_selection_threshold) {
                            KlBase<GraphT, MemoryConstraintT>::node_selection.insert(node);
                            break;
                        }
                    }
                }

                for (const auto &outEdge : OutEdges(node, currentSchedule_.instance->getComputationalDag())) {
                    const auto &targetV = Traget(outEdge, currentSchedule_.instance->getComputationalDag());
                    if (currentSchedule_.vector_schedule.assignedProcessor(node)
                        != currentSchedule_.vector_schedule.assignedProcessor(targetV)) {
                        if (current_schedule.instance->getComputationalDag().EdgeCommWeight(out_edge)
                            > node_comm_selection_threshold) {
                            KlBase<GraphT, MemoryConstraintT>::node_selection.insert(node);
                            break;
                        }
                    }
                }
            }
        }
    }

  public:
    KlTotal() : KlBase<GraphT, MemoryConstraintT>(currentSchedule_), currentSchedule_(this) {}

    virtual ~KlTotal() = default;
};

}    // namespace osp
