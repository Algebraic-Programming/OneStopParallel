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

#include <climits>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

#include "scheduler/Scheduler.hpp"

class MemConstListScheduler : public Scheduler {

  protected:
    MEMORY_CONSTRAINT_TYPE memory_const_type = NONE;

    bool use_memory_constraint = false;

    std::vector<int> current_proc_persistent_memory;
    std::vector<int> current_proc_transient_memory;

    std::vector<std::unordered_set<VertexType>> current_proc_predec;

    virtual void init_mem_const_data_structures(const BspArchitecture &arch);
    virtual void reset_mem_const_datastructures_new_superstep(unsigned proc);
    virtual bool check_can_add(const BspSchedule &schedule, const BspInstance &instance, unsigned node, unsigned succ,
                               unsigned supstepIdx);

    virtual bool check_choose_node(const BspSchedule &schedule, const BspInstance &instance, VertexType node, unsigned proc, unsigned current_superstep);
 

    virtual std::vector<VertexType>
    update_mem_const_datastructure_after_assign(const BspSchedule &schedule, const BspInstance &instance,
                                                unsigned nextNode, unsigned nextProc, unsigned supstepIdx,
                                                std::vector<std::set<VertexType>> &procReady);

    // virtual bool choose_mem_const(const BspInstance &instance, const BspSchedule &schedule, const VertexType top_node,
    //                                unsigned proc, unsigned current_superstep);

  public:
    /**
     * @brief Default constructor for GreedyBspLocking.
     */
    MemConstListScheduler() : Scheduler() {}

    /**
     * @brief Default destructor for GreedyBspLocking.
     */
    virtual ~MemConstListScheduler() = default;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) = 0;
    virtual std::string getScheduleName() const = 0;

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override {
        use_memory_constraint = use_memory_constraint_;
    }
};