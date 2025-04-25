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

template<typename Graph_t>
class kl_hyper_total_comm : public kl_total<Graph_t> {

  protected:
    
    virtual void compute_comm_gain(vertex_idx_t<Graph_t> node, unsigned current_step, unsigned current_proc,
                                   unsigned new_proc) override {
        throw std::runtime_error("Not implemented yet");
    }

    virtual double compute_current_costs() override {

        double work_costs = 0;
        for (unsigned step = 0; step < current_schedule.num_steps(); step++) {
            work_costs += current_schedule.step_max_work[step];
        }

        double comm_costs = 0;

        for (const auto &node : current_schedule.instance->getComputationalDag().vertices()) {

            if (is_sink(node, current_schedule.instance->getComputationalDag()))
                continue;

            std::unordered_set<unsigned> intersects;

            for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

                const unsigned &target_proc = current_schedule.vector_schedule.assignedProcessor(target);

                if (current_schedule.vector_schedule.assignedProcessor(node) != target_proc) {
                    intersects.insert(target_proc);
                }
            }

            comm_costs +=
                intersects.size() * current_schedule.instance->getComputationalDag().vertex_comm_weight(node);
        }

        current_schedule.current_cost =
            work_costs + comm_costs * current_schedule.comm_multiplier +
            (current_schedule.num_steps() - 1) * current_schedule.instance->synchronisationCosts();

        return current_schedule.current_cost;
    }

  public:
    kl_hyper_total_comm(bool use_node_communication_costs_ = false) : kl_total(use_node_communication_costs_) {}

    virtual ~kl_hyper_total_comm() = default;

    virtual std::string getScheduleName() const override { return "KLHyperTotalComm"; }
};

} // namespace osp