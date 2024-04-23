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

#include "algorithms/LocalSearchSchedulers/HillClimbingScheduler.hpp"

RETURN_STATUS HillClimbingScheduler::improveSchedule(BspSchedule &schedule) {

    Schedule S;
    S.ConvertFromNewSchedule(schedule);

    HillClimbing hillClimb(S);

    hillClimb.HillClimb(timeLimitSeconds);
    S = hillClimb.getSchedule();

    HillClimbingCS hillClimbCS(S);
    hillClimbCS.HillClimb(timeLimitSeconds);
    S = hillClimbCS.getSchedule();

    BspSchedule schedule_new = S.ConvertToNewSchedule(schedule.getInstance());
    schedule_new.setAutoCommunicationSchedule();

    if (schedule_new.computeCosts() < schedule.computeCosts()) {
        schedule = schedule_new;
    }

    return SUCCESS;
}