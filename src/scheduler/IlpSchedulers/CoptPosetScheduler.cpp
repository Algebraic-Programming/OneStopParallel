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

#include "scheduler/IlpSchedulers/CoptPosetScheduler.hpp"

std::pair<RETURN_STATUS, BspSchedule> CoptPosetScheduler::computeSchedule(const BspInstance& instance) {

    BspSchedule schedule(instance);
    schedule.setAssignedSupersteps(poset_map);
    schedule.setAutoCommunicationSchedule();

    CoptPartialScheduler partial_scheduler(0, 0, max_number_supersteps_iter);

    auto status = partial_scheduler.improveSchedule(schedule);

    if (status != SUCCESS) {
        return {status, schedule};
    }

    unsigned old_supersteps = num_posets;
    unsigned new_start = schedule.numberOfSupersteps() - old_supersteps + 1;

    unsigned i = 1;
    while (i < num_posets) {

        partial_scheduler.setStartSuperstep(new_start);
        partial_scheduler.setEndSuperstep(new_start);

        old_supersteps = schedule.numberOfSupersteps();

        status = partial_scheduler.improveSchedule(schedule);

        if (status != SUCCESS) {
            return {status, schedule};
        }

        new_start = schedule.numberOfSupersteps() - old_supersteps + 1;

        i++;
    }

    return {SUCCESS, schedule};
}