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
#include "Scheduler.hpp"

namespace osp {

/**
 * @class ImprovementScheduler
 * @brief Abstract base class for improvement scheduling scheduler.
 *
 * The ImprovementScheduler class provides a common interface for improvement scheduling scheduler.
 * Subclasses of this class can implement specific improvement scheduler by overriding the virtual methods.
 */
template<typename Graph_t>
class ImprovementScheduler {

  protected:
    unsigned timeLimitSeconds; /**< The time limit in seconds for the improvement algorithm. */

  public:
    /**
     * @brief Constructor for ImprovementScheduler.
     * @param timelimit The time limit in seconds for the improvement algorithm. Default is 3600 seconds (1 hour).
     */
    ImprovementScheduler(unsigned timelimit = 3600) : timeLimitSeconds(timelimit) {}

    /**
     * @brief Destructor for ImprovementScheduler.
     */
    virtual ~ImprovementScheduler() = default;

    /**
     * @brief Set the time limit in seconds for the improvement algorithm.
     * @param limit The time limit in seconds.
     */
    virtual void setTimeLimitSeconds(unsigned int limit) { timeLimitSeconds = limit; }

    /**
     * @brief Set the time limit in hours for the improvement algorithm.
     * @param limit The time limit in hours.
     */
    virtual void setTimeLimitHours(unsigned int limit) { timeLimitSeconds = limit * 3600; }

    /**
     * @brief Get the time limit in seconds for the improvement algorithm.
     * @return The time limit in seconds.
     */
    inline unsigned int getTimeLimitSeconds() const { return timeLimitSeconds; }

    /**
     * @brief Get the time limit in hours for the improvement algorithm.
     * @return The time limit in hours.
     */
    inline unsigned int getTimeLimitHours() const { return timeLimitSeconds / 3600; }

    /**
     * @brief Get the name of the improvement scheduling algorithm.
     * @return The name of the algorithm as a string.
     */
    virtual std::string getScheduleName() const = 0;

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) {
        throw std::runtime_error("Not implemented " + use_memory_constraint_);
    }

    /**
     * @brief Improve the given BspSchedule.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual RETURN_STATUS improveSchedule(BspSchedule<Graph_t> &schedule) = 0;

    /**
     * @brief Construct an improved BspSchedule based on the given schedule.
     * @param schedule The BspSchedule to be improved.
     * @return A pair containing the status of the improvement operation and the improved BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule<Graph_t>>
    constructImprovedSchedule(const BspSchedule<Graph_t> &schedule) {

        BspSchedule<Graph_t> improvedSchedule = schedule;
        RETURN_STATUS status = improveSchedule(improvedSchedule);
        return {status, improvedSchedule};
    }

    /**
     * @brief Improve the given BspSchedule within the time limit.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule<Graph_t> &schedule) { return TIMEOUT; }

    /**
     * @brief Construct an improved BspSchedule based on the given schedule within the time limit.
     * @param schedule The BspSchedule to be improved.
     * @return A pair containing the status of the improvement operation and the improved BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule<Graph_t>>
    constructImprovedScheduleWithTimeLimit(const BspSchedule<Graph_t> &schedule) {

        std::packaged_task<std::pair<RETURN_STATUS, BspSchedule<Graph_t>>(const BspSchedule<Graph_t> &)> task(
            [this](const BspSchedule<Graph_t> &schedule) -> std::pair<RETURN_STATUS, BspSchedule<Graph_t>> {
                return constructImprovedSchedule(schedule);
            });
        auto future = task.get_future();
        std::thread thr(std::move(task), schedule);
        if (future.wait_for(std::chrono::seconds(getTimeLimitSeconds())) == std::future_status::timeout) {
            thr.detach(); // we leave the thread still running
            std::cerr << "Timeout reached, execution of computeSchedule() aborted" << std::endl;
            return std::make_pair(TIMEOUT, BspSchedule<Graph_t>());
        }
        thr.join();
        try {
            const auto result = future.get();
            return result;
        } catch (const std::exception &e) {
            std::cerr << "Exception caught in computeScheduleWithTimeLimit(): " << e.what() << std::endl;
            return std::make_pair(ERROR, BspSchedule<Graph_t>());
        }
    }
};

template<typename Graph_t>
class ComboScheduler : public Scheduler<Graph_t> {

  private:
    Scheduler<Graph_t> &base_scheduler;
    ImprovementScheduler<Graph_t> &improvement_scheduler;

  public:
    ComboScheduler(Scheduler<Graph_t> &base, ImprovementScheduler<Graph_t> &improvement)
        : Scheduler<Graph_t>(), base_scheduler(base), improvement_scheduler(improvement) {}

    virtual void setTimeLimitSeconds(unsigned int limit) override {
      
        Scheduler<Graph_t>::timeLimitSeconds = limit;
        base_scheduler.setTimeLimitSeconds(limit);
        improvement_scheduler.setTimeLimitSeconds(limit);
    }

    virtual void setTimeLimitHours(unsigned int limit) override {

        Scheduler<Graph_t>::timeLimitSeconds = limit * 3600;
        base_scheduler.setTimeLimitHours(limit);
        improvement_scheduler.setTimeLimitHours(limit);
    }

    virtual ~ComboScheduler() = default;

    virtual std::string getScheduleName() const override {
        return base_scheduler->getScheduleName() + "+" + improvement_scheduler->getScheduleName();
    }

    virtual std::pair<RETURN_STATUS, BspSchedule<Graph_t>>
    computeSchedule(const BspInstance<Graph_t> &instance) override {

        std::pair<RETURN_STATUS, BspSchedule<Graph_t>> base_schedule = base_scheduler.computeSchedule(instance);
        if (base_schedule.first != SUCCESS) {
            return base_schedule;
        }
        RETURN_STATUS improve_status = improvement_scheduler.improveSchedule(base_schedule.second);
        return {improve_status, base_schedule.second};
    }
};

} // namespace osp