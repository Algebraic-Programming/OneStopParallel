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
class kl_total_comm : public kl_total {

  protected:

    virtual void compute_comm_gain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc) override;
    virtual double compute_current_costs() override;

  public:
    kl_total_comm(bool use_node_communication_costs_ = false) : kl_total(use_node_communication_costs_) {}

    virtual ~kl_total_comm() = default;

    virtual std::string getScheduleName() const override { return "KLTotalComm"; }
};



class kl_total_comm_test : public kl_total_comm {

  public:

    kl_total_comm_test() : kl_total_comm() {}

    virtual ~kl_total_comm_test() = default;

    virtual std::string getScheduleName() const override { return "KLBaseTest"; }

    kl_current_schedule_total& get_current_schedule() { return current_schedule; }

    auto& get_node_gains() { return node_gains; }
    auto& get_node_change_in_costs() { return node_change_in_costs; }
    auto& get_max_gain_heap() { return max_gain_heap; } 

    void initialize_gain_heap_test(const std::unordered_set<VertexType> &nodes, double reward_ = 0.0, double penalty_ = 0.0) {
        reward = reward_;
        penalty = penalty_;

      initialize_gain_heap(nodes);
    }

    void test_setup_schedule(BspSchedule &schedule) {

      current_schedule.instance = &schedule.getInstance();

      best_schedule = &schedule;

      num_nodes = current_schedule.instance->numberOfVertices();
      num_procs = current_schedule.instance->numberOfProcessors();

      set_parameters();
      initialize_datastructures();
    }


    RETURN_STATUS improve_schedule_test_1(BspSchedule &schedule) {

        current_schedule.instance = &schedule.getInstance();

        best_schedule = &schedule;
        num_nodes = current_schedule.instance->numberOfVertices();
        num_procs = current_schedule.instance->numberOfProcessors();

        set_parameters();
        initialize_datastructures();

        bool improvement_found = run_local_search_simple();

        assert(best_schedule->satisfiesPrecedenceConstraints());

        schedule.setImprovedLazyCommunicationSchedule();

        if (improvement_found)
            return SUCCESS;
        else
            return BEST_FOUND;
    
    }


    RETURN_STATUS improve_schedule_test_2(BspSchedule &schedule) {

        current_schedule.instance = &schedule.getInstance();

        best_schedule = &schedule;
        num_nodes = current_schedule.instance->numberOfVertices();
        num_procs = current_schedule.instance->numberOfProcessors();

        set_parameters();
        initialize_datastructures();

        bool improvement_found = run_local_search_unlock_delay();

        assert(best_schedule->satisfiesPrecedenceConstraints());

        schedule.setImprovedLazyCommunicationSchedule();

        if (improvement_found)
            return SUCCESS;
        else
            return BEST_FOUND;
    
    }

};

}