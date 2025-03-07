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

/**
 * @class ImprovementScheduler
 * @brief Abstract base class for improvement scheduling scheduler.
 *
 * The ImprovementScheduler class provides a common interface for improvement scheduling scheduler.
 * Subclasses of this class can implement specific improvement scheduler by overriding the virtual methods.
 */
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


    virtual void setUseMemoryConstraint(bool use_memory_constraint_) { throw std::runtime_error("Not implemented");}

    /**
     * @brief Improve the given BspSchedule.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual RETURN_STATUS improveSchedule(BspSchedule &schedule) = 0;

    /**
     * @brief Construct an improved BspSchedule based on the given schedule.
     * @param schedule The BspSchedule to be improved.
     * @return A pair containing the status of the improvement operation and the improved BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> constructImprovedSchedule(const BspSchedule &schedule) {

        BspSchedule improvedSchedule = schedule;
        RETURN_STATUS status = improveSchedule(improvedSchedule);
        return std::make_pair(status, improvedSchedule);
    }

    /**
     * @brief Improve the given BspSchedule within the time limit.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule &schedule) { return TIMEOUT; }

    /**
     * @brief Construct an improved BspSchedule based on the given schedule within the time limit.
     * @param schedule The BspSchedule to be improved.
     * @return A pair containing the status of the improvement operation and the improved BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> constructImprovedScheduleWithTimeLimit(const BspSchedule &schedule);
};

class ComboScheduler : public Scheduler {

  private:
    Scheduler *base_scheduler;
    ImprovementScheduler *improvement_scheduler;

  public:
    virtual void setTimeLimitSeconds(unsigned int limit) override;
    virtual void setTimeLimitHours(unsigned int limit) override;

    ComboScheduler(Scheduler& base, ImprovementScheduler& improvement)
        : base_scheduler(&base), improvement_scheduler(&improvement) {}

    virtual ~ComboScheduler() = default;

    virtual std::string getScheduleName() const override {
        return base_scheduler->getScheduleName() + "+" + improvement_scheduler->getScheduleName();
    }

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {

        std::pair<RETURN_STATUS, BspSchedule> base_schedule = base_scheduler->computeSchedule(instance);
        if (base_schedule.first != SUCCESS) {
            return base_schedule;
        }
        RETURN_STATUS improve_status = improvement_scheduler->improveSchedule(base_schedule.second);
        return {improve_status, base_schedule.second};
    }
};