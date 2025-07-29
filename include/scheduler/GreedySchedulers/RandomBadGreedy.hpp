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

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <iterator>

#include "scheduler/Scheduler.hpp"

class RandomBadGreedy : public Scheduler {
  public:
    RandomBadGreedy() : Scheduler(){};
    RandomBadGreedy(unsigned time_limit) : Scheduler(time_limit){};

    std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    std::string getScheduleName() const override { return "RandomBadGreedy"; }
};