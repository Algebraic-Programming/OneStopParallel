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

template<typename Graph_t, typename MemoryConstraint_t, bool use_node_communication_costs_arg>
class kl_current_schedule_total : public kl_current_schedule<Graph_t, MemoryConstraint_t> {
 
  public:
    kl_current_schedule_total(Ikl_cost_function *cost_f_)
        : kl_current_schedule<Graph_t, MemoryConstraint_t>(cost_f_) {}

    double comm_multiplier = 1.0;    
    constexpr static bool use_node_communication_costs = use_node_communication_costs_arg || not has_edge_weights_v<Graph_t>;

};

template<typename Graph_t, typename MemoryConstraint_t, bool use_node_communication_costs_arg>
class kl_total : public kl_base<Graph_t, MemoryConstraint_t> {

  protected:
    kl_current_schedule_total<Graph_t, MemoryConstraint_t, use_node_communication_costs_arg> current_schedule;

    v_commw_t<Graph_t> node_comm_selection_threshold = 0;
    double max_edge_weight = 0.0;
    virtual void initialize_datastructures() override {

#ifdef KL_DEBUG
        std::cout << "KLTotal initialize datastructures" << std::endl;
#endif

        kl_base<Graph_t, MemoryConstraint_t>::initialize_datastructures();

        v_commw_t<Graph_t> max_edge_weight_ = 0;
        v_workw_t<Graph_t> max_node_weight_ = 0;

        for (const auto vertex : current_schedule.instance->getComputationalDag().vertices()) {

            if (is_sink(vertex, current_schedule.instance->getComputationalDag()))
                continue;

            max_edge_weight_ =
                std::max(max_edge_weight_, current_schedule.instance->getComputationalDag().vertex_comm_weight(vertex));

            max_node_weight_ =
                std::max(max_node_weight_, current_schedule.instance->getComputationalDag().vertex_work_weight(vertex));
        }
       

        if constexpr (not current_schedule.use_node_communication_costs) {

            max_edge_weight_ = 0;

            for (const auto &edge : edges(current_schedule.instance->getComputationalDag())) {
                max_edge_weight_ = std::max(max_edge_weight_, current_schedule.instance->getComputationalDag().edge_comm_weight(edge));
            }
        }
        

        max_edge_weight = max_edge_weight_ + max_node_weight_;

        kl_base<Graph_t, MemoryConstraint_t>::parameters.initial_penalty =
            max_edge_weight * current_schedule.comm_multiplier * current_schedule.instance->communicationCosts();

        kl_base<Graph_t, MemoryConstraint_t>::parameters.gain_threshold =
            max_edge_weight * current_schedule.comm_multiplier * current_schedule.instance->communicationCosts();
    }

    virtual void update_reward_penalty() override {

        if (current_schedule.current_violations.size() <= kl_base<Graph_t, MemoryConstraint_t>::parameters.violations_threshold) {
            kl_base<Graph_t, MemoryConstraint_t>::penalty = kl_base<Graph_t, MemoryConstraint_t>::parameters.initial_penalty;
            kl_base<Graph_t, MemoryConstraint_t>::reward = 0.0;

        } else {
            kl_base<Graph_t, MemoryConstraint_t>::parameters.violations_threshold = 0;

            kl_base<Graph_t, MemoryConstraint_t>::penalty = std::log((current_schedule.current_violations.size())) * max_edge_weight *
                                        current_schedule.comm_multiplier *
                                        current_schedule.instance->communicationCosts();

            kl_base<Graph_t, MemoryConstraint_t>::reward = std::sqrt((current_schedule.current_violations.size() + 4)) * max_edge_weight *
                                       current_schedule.comm_multiplier *
                                       current_schedule.instance->communicationCosts();
        }
    }

    virtual void set_initial_reward_penalty() override {

        kl_base<Graph_t, MemoryConstraint_t>::penalty = kl_base<Graph_t, MemoryConstraint_t>::parameters.initial_penalty;
        kl_base<Graph_t, MemoryConstraint_t>::reward =
            max_edge_weight * current_schedule.comm_multiplier * current_schedule.instance->communicationCosts();
    }

    virtual void select_nodes_comm() override {

        if constexpr (current_schedule.use_node_communication_costs) {

            for (const auto &node : current_schedule.instance->getComputationalDag().vertices()) {

                for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

                    if (current_schedule.vector_schedule.assignedProcessor(node) !=
                        current_schedule.vector_schedule.assignedProcessor(source)) {

                        if (current_schedule.instance->getComputationalDag().vertex_comm_weight(node) >
                            node_comm_selection_threshold) {

                              kl_base<Graph_t, MemoryConstraint_t>::node_selection.insert(node);
                            break;
                        }
                    }
                }

                for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

                    if (current_schedule.vector_schedule.assignedProcessor(node) !=
                        current_schedule.vector_schedule.assignedProcessor(target)) {

                        if (current_schedule.instance->getComputationalDag().vertex_comm_weight(node) >
                            node_comm_selection_threshold) {

                              kl_base<Graph_t, MemoryConstraint_t>::node_selection.insert(node);
                            break;
                        }
                    }
                }
            }

        } else {
            for (const auto &node : current_schedule.instance->getComputationalDag().vertices()) {

                for (const auto &in_edge : in_edges(node, current_schedule.instance->getComputationalDag())) {

                    const auto &source_v = source(in_edge, current_schedule.instance->getComputationalDag());
                    if (current_schedule.vector_schedule.assignedProcessor(node) !=
                        current_schedule.vector_schedule.assignedProcessor(source_v)) {

                        if (current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) >
                            node_comm_selection_threshold) {

                            kl_base<Graph_t, MemoryConstraint_t>::node_selection.insert(node);
                            break;
                        }
                    }
                }

                for (const auto &out_edge : out_edges(node, current_schedule.instance->getComputationalDag())) {

                    const auto &target_v = target(out_edge, current_schedule.instance->getComputationalDag());
                    if (current_schedule.vector_schedule.assignedProcessor(node) !=
                        current_schedule.vector_schedule.assignedProcessor(target_v)) {

                        if (current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) >
                            node_comm_selection_threshold) {

                            kl_base<Graph_t, MemoryConstraint_t>::node_selection.insert(node);
                            break;
                        }
                    }
                }
            }
        }
    }

  public:
    kl_total()
        : kl_base<Graph_t, MemoryConstraint_t>(current_schedule), current_schedule(this) {}

    virtual ~kl_total() = default;

};

} // namespace osp