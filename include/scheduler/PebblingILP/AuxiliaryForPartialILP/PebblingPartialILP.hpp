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
#include "model/BspMemSchedule.hpp"
#include "scheduler/PebblingILP/AuxiliaryForPartialILP/AcyclicDagDivider.hpp"
#include "scheduler/PebblingILP/AuxiliaryForPartialILP/SubproblemMultiScheduling.hpp"
#include "scheduler/PebblingILP/MultiProcessorPebbling.hpp"
#include "scheduler/GreedySchedulers/GreedyBspFillupScheduler.hpp"

class PebblingPartialILP : public Scheduler {

    unsigned minPartitionSize = 50, maxPartitionSize = 100;
    unsigned time_seconds_for_subILPs = 600;

    bool asynchronous = false;
    bool verbose = false;

    std::map<std::pair<unsigned, unsigned>, unsigned> part_and_nodetype_to_new_index;

  public:
    PebblingPartialILP() {}

    virtual ~PebblingPartialILP() = default;

    std::pair<RETURN_STATUS, BspMemSchedule> computePebbling(const BspInstance &instance);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    ComputationalDag contractByPartition(const BspInstance &instance, const std::vector<unsigned> &node_to_part_assignment);

    BspSchedule computeGreedyBaselineForSubDag(const BspInstance &subInstance, const ComputationalDag& subDagWithoutExternalSources, const std::set<unsigned>& nodes_in_part, unsigned nr_extra_sources) const;
  
    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "PebblingPartialILP"; }

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> getMinAndMaxSize() const { return std::make_pair(minPartitionSize, maxPartitionSize); }
    inline void setMinSize(const unsigned min_size) {minPartitionSize = min_size; maxPartitionSize = 2*min_size; }
    inline void setMinAndMaxSize(const std::pair<unsigned, unsigned> min_and_max) {minPartitionSize = min_and_max.first; maxPartitionSize = min_and_max.second; }
    inline void setAsync(const bool async_) {asynchronous = async_; }
    inline void setSecondsForSubILP(const unsigned seconds_) {time_seconds_for_subILPs = seconds_; }
    inline void setVerbose(const bool verbose_) {verbose = verbose_; }
};