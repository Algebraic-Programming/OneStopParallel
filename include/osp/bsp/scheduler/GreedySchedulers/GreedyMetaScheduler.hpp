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

#include "osp/bsp/model/BspScheduleCostEvaluator.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include <vector>
#include <string>

namespace osp {

template<typename Graph_t>
class GreedyMetaScheduler : public Scheduler<Graph_t> {

    Serial<Graph_t> serial_scheduler_;
    std::vector<Scheduler<Graph_t>*> schedulers_;

    static constexpr bool verbose = true;

  public:
    /**
     * @brief Default constructor for MetaScheduler.
     */
    GreedyMetaScheduler() : Scheduler<Graph_t>() {}

    /**
     * @brief Default destructor for MetaScheduler.
     */
    ~GreedyMetaScheduler() override = default;

    void addSerialScheduler() { schedulers_.push_back(&serial_scheduler_); }
    void addScheduler(Scheduler<Graph_t> & s) { schedulers_.push_back(&s); }
    void resetScheduler() { schedulers_.clear(); }

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {
        if (schedule.getInstance().getArchitecture().numberOfProcessors() == 1) {
            if constexpr (verbose) std::cout << "Using serial scheduler for P=1." << std::endl;
            serial_scheduler_.computeSchedule(schedule);
            return RETURN_STATUS::OSP_SUCCESS;
        }

        v_workw_t<Graph_t> best_schedule_cost = std::numeric_limits<v_workw_t<Graph_t>>::max(); 
        BspSchedule<Graph_t> current_schedule(schedule.getInstance());

        for (Scheduler<Graph_t>* scheduler : schedulers_) {
            scheduler->computeSchedule(current_schedule);
            BspScheduleCostEvaluator<Graph_t> evaluator(current_schedule);
            const v_workw_t<Graph_t> schedule_cost = evaluator.computeCosts();

            if constexpr (verbose) std::cout << "Executed scheduler " << scheduler->getScheduleName() << ", costs: " << schedule_cost << ", nr. supersteps: " << current_schedule.numberOfSupersteps() << std::endl;

            if (schedule_cost < best_schedule_cost) {
                best_schedule_cost = schedule_cost;
                schedule = current_schedule;
                if constexpr (verbose) std::cout << "New best schedule!" << std::endl;     
            }

        }

        return RETURN_STATUS::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return "GreedyMetaScheduler"; }
};

} // namespace osp