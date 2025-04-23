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


class kl_current_schedule_total : public kl_current_schedule {

  public:

    kl_current_schedule_total(Ikl_cost_function *cost_f_, bool use_node_communication_costs_ = false) : kl_current_schedule(cost_f_), use_node_communication_costs(use_node_communication_costs_) {}

    double comm_multiplier = 1.0;
    bool use_node_communication_costs = true;

};

class kl_total : public kl_base {

  protected:
    kl_current_schedule_total current_schedule;

    int node_comm_selection_threshold = 0;
    double max_edge_weight = 0.0;
    virtual void initialize_datastructures() override;

    virtual void update_reward_penalty() override;
    virtual void set_initial_reward_penalty() override;

    virtual void select_nodes_comm(unsigned threshold) override;

  public:
    kl_total(bool use_node_communication_costs_)
        : kl_base(current_schedule), current_schedule(this, use_node_communication_costs_) {}

    virtual ~kl_total() = default;

};