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

#include <string>
#include <vector>

#include "osp/bsp/model/cost/LazyCommunicationCost.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/bsp/scheduler/Serial.hpp"

namespace osp {

/**
 * @class GreedyMetaScheduler
 * @brief The GreedyMetaScheduler class represents a meta-scheduler that selects the best schedule produced from a list of
 * added schedulers.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method iterates through a list of schedulers, computes a schedule using each one,
 * and returns the schedule with the minimum cost.
 *
 * @tparam Graph_t The graph type representing the computational DAG.
 * @tparam CostModel The cost model functor to evaluate schedules. Defaults to LazyCommunicationCost.
 */
template <typename GraphT, typename CostModel = LazyCommunicationCost<GraphT>>
class GreedyMetaScheduler : public Scheduler<GraphT> {
    Serial<GraphT> serialScheduler_;
    std::vector<Scheduler<GraphT> *> schedulers_;

    static constexpr bool verbose_ = false;

  public:
    /**
     * @brief Default constructor for GreedyMetaScheduler.
     */
    GreedyMetaScheduler() : Scheduler<GraphT>() {}

    /**
     * @brief Default destructor for MetaScheduler.
     */
    ~GreedyMetaScheduler() override = default;

    void AddSerialScheduler() { schedulers_.push_back(&serialScheduler_); }

    void AddScheduler(Scheduler<GraphT> &s) { schedulers_.push_back(&s); }

    void ResetScheduler() { schedulers_.clear(); }

    ReturnStatus computeSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().GetArchitecture().NumberOfProcessors() == 1) {
            if constexpr (verbose_) {
                std::cout << "Using serial scheduler for P=1." << std::endl;
            }
            serialScheduler_.computeSchedule(schedule);
            return ReturnStatus::OSP_SUCCESS;
        }

        VWorkwT<GraphT> bestScheduleCost = std::numeric_limits<VWorkwT<GraphT>>::max();
        BspSchedule<GraphT> currentSchedule(schedule.GetInstance());

        for (Scheduler<GraphT> *scheduler : schedulers_) {
            scheduler->computeSchedule(currentSchedule);
            const VWorkwT<GraphT> scheduleCost = CostModel()(currentSchedule);

            if constexpr (verbose_) {
                std::cout << "Executed scheduler " << scheduler->getScheduleName() << ", costs: " << schedule_cost
                          << ", nr. supersteps: " << currentSchedule.NumberOfSupersteps() << std::endl;
            }

            if (schedule_cost < best_schedule_cost) {
                bestScheduleCost = schedule_cost;
                schedule = currentSchedule;
                if constexpr (verbose_) {
                    std::cout << "New best schedule!" << std::endl;
                }
            }
        }

        return ReturnStatus::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return "GreedyMetaScheduler"; }
};

}    // namespace osp
