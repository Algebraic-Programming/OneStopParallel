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

#include "scheduler/Scheduler.hpp"
#include "model/BspInstance.hpp"

class SubproblemMultiScheduling : public Scheduler {

    std::vector<unsigned> last_node_on_proc;
    std::vector<std::vector<unsigned> > proc_task_lists;
    std::vector<std::set<unsigned> > processors_to_nodes;
    std::vector<int> longest_outgoing_path;

  public:
    SubproblemMultiScheduling() {}

    virtual ~SubproblemMultiScheduling() = default;

    std::pair<RETURN_STATUS, std::vector<std::set<unsigned> > > computeMultiSchedule(const BspInstance &instance);

    std::vector<std::pair<unsigned, unsigned> > makeAssignment(const BspInstance &instance,
                                                    const std::set<std::pair<unsigned, unsigned> > &nodes_available,
                                                    const std::set<unsigned> &procs_available) const;

    std::vector<int> static get_longest_path(const ComputationalDag &graph);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;
  
    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "SubproblemMultiScheduling"; }

    inline const std::vector<std::vector<unsigned> >& getProcTaskLists() const { return proc_task_lists; }

};
