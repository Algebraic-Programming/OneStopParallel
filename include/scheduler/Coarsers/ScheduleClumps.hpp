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

#include "scheduler/InstanceContractor.hpp"
#include "structures/union_find.hpp"

/**
 * @brief Contracts serial subgraphs
 * 
 */
class ScheduleClumps : public InstanceContractor {
    private:
        Scheduler *clumpingScheduler;

    protected:
        RETURN_STATUS run_contractions() override;

    public:
        ScheduleClumps() : InstanceContractor() { }
        ScheduleClumps(Scheduler *clumpingScheduler_, Scheduler* sched_) : ScheduleClumps(clumpingScheduler_, sched_, nullptr)  { }
        ScheduleClumps(Scheduler *clumpingScheduler_, Scheduler* sched_, ImprovementScheduler* improver_) : InstanceContractor(sched_, improver_), clumpingScheduler(clumpingScheduler_) { }
        ScheduleClumps(unsigned timelimit, Scheduler *clumpingScheduler_, Scheduler* sched_) : ScheduleClumps(timelimit, clumpingScheduler_, sched_, nullptr) { }
        ScheduleClumps(unsigned timelimit, Scheduler *clumpingScheduler_, Scheduler* sched_, ImprovementScheduler* improver_) : InstanceContractor(timelimit, sched_, improver_), clumpingScheduler(clumpingScheduler_) { }
        virtual ~ScheduleClumps() = default;

        void setUseMemoryConstraint(bool use_memory_constraint_) override;

        void setClumpingScheduler(Scheduler *clumpingScheduler_) { clumpingScheduler = clumpingScheduler_; }

        std::string getCoarserName() const override { return clumpingScheduler->getScheduleName() + "Clumps"; }
};