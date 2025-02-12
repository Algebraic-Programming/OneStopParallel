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

#include <tuple>

#include "Scheduler.hpp"

class ReverseScheduler : public Scheduler {
    private:
        Scheduler *base_scheduler;

    public:
        virtual void setTimeLimitSeconds(unsigned int limit) override;
        virtual void setTimeLimitHours(unsigned int limit) override;

        ReverseScheduler(Scheduler *base) : base_scheduler(base) {}
        virtual ~ReverseScheduler() = default;

        virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

        virtual std::string getScheduleName() const override {
            return "Reverse" + base_scheduler->getScheduleName();
        }

};