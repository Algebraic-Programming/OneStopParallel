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
#include "ImprovementScheduler.hpp"
#include "SSPScheduler.hpp"

/**
 * @class SSPImprovementScheduler
 * @brief Abstract base class for improvement scheduling SSP scheduler.
 *
 * The SSPImprovementScheduler class provides a common interface for improvement scheduling SSP scheduler.
 * Subclasses of this class can implement specific improvement scheduler by overriding the virtual methods.
 */
class SSPImprovementScheduler : public ImprovementScheduler {

  private:
    /**
     * @brief Improve the given SspSchedule.
     * @param schedule The SspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual RETURN_STATUS improveSSPSchedule(BspSchedule &schedule, unsigned stale) = 0;

  public:
    /**
     * @brief Constructor for SSPImprovementScheduler.
     * @param timelimit The time limit in seconds for the improvement algorithm. Default is 3600 seconds (1 hour).
     */
    SSPImprovementScheduler(unsigned timelimit = 3600) : ImprovementScheduler(timelimit) {}

    /**
     * @brief Destructor for SSPImprovementScheduler.
     */
    virtual ~SSPImprovementScheduler() = default;

    /**
     * @brief Set the time limit in seconds for the improvement algorithm.
     * @param limit The time limit in seconds.
     */
    virtual void setTimeLimitSeconds(unsigned int limit) override { timeLimitSeconds = limit; }

    /**
     * @brief Set the time limit in hours for the improvement algorithm.
     * @param limit The time limit in hours.
     */
    virtual void setTimeLimitHours(unsigned int limit) override { timeLimitSeconds = limit * 3600; }

    /**
     * @brief Get the name of the improvement scheduling algorithm.
     * @return The name of the algorithm as a string.
     */
    virtual std::string getScheduleName() const override = 0;


    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override { throw std::runtime_error("Not implemented");}

    /**
     * @brief Improve the given BspSchedule.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual RETURN_STATUS improveSchedule(BspSchedule &schedule) override {
        return improveSSPSchedule(schedule, 1);
    };

    /**
     * @brief Improve the given SspSchedule.
     * @param schedule The SspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual RETURN_STATUS improveSSPSchedule(SspSchedule &schedule) {
        return improveSSPSchedule(schedule, schedule.getStaleness());
    };

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
     * @brief Construct an improved BspSchedule based on the given schedule.
     * @param schedule The BspSchedule to be improved.
     * @return A pair containing the status of the improvement operation and the improved BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> constructImprovedSSPSchedule(const SspSchedule &schedule) {

        SspSchedule improvedSchedule = schedule;
        RETURN_STATUS status = improveSSPSchedule(improvedSchedule);
        return std::make_pair(status, improvedSchedule);
    }

    /**
     * @brief Improve the given BspSchedule within the time limit.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule &schedule) override { return TIMEOUT; }

    /**
     * @brief Construct an improved BspSchedule based on the given schedule within the time limit.
     * @param schedule The BspSchedule to be improved.
     * @return A pair containing the status of the improvement operation and the improved BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> constructImprovedScheduleWithTimeLimit(const BspSchedule &schedule) override;
};

class ComboSSPScheduler : public SSPScheduler {

  private:
    SSPScheduler *base_scheduler;
    SSPImprovementScheduler *improvement_scheduler;

  public:
    virtual void setTimeLimitSeconds(unsigned int limit) override;
    virtual void setTimeLimitHours(unsigned int limit) override;

    ComboSSPScheduler(SSPScheduler *base, SSPImprovementScheduler *improvement)
        : base_scheduler(base), improvement_scheduler(improvement) {}

    virtual ~ComboSSPScheduler() = default;

    virtual std::string getScheduleName() const override {
        return base_scheduler->getScheduleName() + "+" + improvement_scheduler->getScheduleName();
    }

    virtual std::pair<RETURN_STATUS, SspSchedule> computeSspSchedule(const BspInstance &instance, unsigned stale) override {

        std::pair<RETURN_STATUS, SspSchedule> base_schedule = base_scheduler->computeSspSchedule(instance, stale);
        if (base_schedule.first != SUCCESS) {
            return base_schedule;
        }
        RETURN_STATUS improve_status = improvement_scheduler->improveSSPSchedule(base_schedule.second);
        return {improve_status, base_schedule.second};
    }
};