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

#include <stdexcept>

#include "scheduler/SSPImprovementScheduler.hpp"

std::pair<RETURN_STATUS, BspSchedule>
SSPImprovementScheduler::constructImprovedScheduleWithTimeLimit(const BspSchedule &schedule) {

    std::packaged_task<std::pair<RETURN_STATUS, BspSchedule>(const BspSchedule &)> task(
        [this](const BspSchedule &schedule) -> std::pair<RETURN_STATUS, BspSchedule> {
            return constructImprovedSchedule(schedule);
        });
    auto future = task.get_future();
    std::thread thr(std::move(task), schedule);
    if (future.wait_for(std::chrono::seconds(getTimeLimitSeconds())) == std::future_status::timeout) {
        thr.detach(); // we leave the thread still running
        std::cerr << "Timeout reached, execution of computeSchedule() aborted" << std::endl;
        return std::make_pair(TIMEOUT, BspSchedule());
    }
    thr.join();
    try {
        const auto result = future.get();
        return result;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in computeScheduleWithTimeLimit(): " << e.what() << std::endl;
        return std::make_pair(ERROR, BspSchedule());
    }
}

/*
RETURN_STATUS ImprovementScheduler::improveScheduleWithTimeLimit(BspSchedule &schedule) {


    std::packaged_task<RETURN_STATUS(BspSchedule&)> task(
        [this](BspSchedule &schedule) -> RETURN_STATUS {
            return improveSchedule(schedule);
        });
    auto future = task.get_future();
    std::thread thr(std::move(task), schedule);
    if (future.wait_for(std::chrono::seconds(getTimeLimitSeconds())) == std::future_status::timeout) {
        thr.detach(); // we leave the thread still running
        std::cerr << "Timeout reached, execution of computeSchedule() aborted" << std::endl;
        return TIMEOUT;
    }
    thr.join();
    try {
        const auto result = future.get();
        return result;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in computeScheduleWithTimeLimit(): " << e.what() << std::endl;
        return ERROR;
    }
}
*/

void ComboSSPScheduler::setTimeLimitSeconds(unsigned int limit) {
    timeLimitSeconds = limit;
    if (base_scheduler) base_scheduler->setTimeLimitHours(limit);
    if (improvement_scheduler) improvement_scheduler->setTimeLimitSeconds(limit);
}

void ComboSSPScheduler::setTimeLimitHours(unsigned int limit) {
    timeLimitSeconds = limit * 3600;
    if (base_scheduler) base_scheduler->setTimeLimitHours(limit);
    if (improvement_scheduler) improvement_scheduler->setTimeLimitHours(limit);
}