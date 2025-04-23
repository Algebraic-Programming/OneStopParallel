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

#include "CoptPartialScheduler.hpp"
#include "scheduler/Scheduler.hpp"


class CoptPosetScheduler : public Scheduler {

  private:
    std::vector<unsigned> poset_map;
    unsigned num_posets;
    unsigned max_number_supersteps_iter;

  public:

    CoptPosetScheduler(std::vector<unsigned> map_, unsigned num_posets_,
                          unsigned max_num_step_iter_)
        : poset_map(map_), num_posets(num_posets_), max_number_supersteps_iter(max_num_step_iter_) {}

    virtual ~CoptPosetScheduler() = default;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance& instance) override;

    virtual std::string getScheduleName() const override { return "PosetIlp"; }

};
