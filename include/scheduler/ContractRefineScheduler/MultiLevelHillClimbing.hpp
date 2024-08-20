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

#include <chrono>
#include <climits>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "scheduler/GreedySchedulers/GreedyLayers.hpp"
#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"

#include "scheduler/Scheduler.hpp"
#include "auxiliary/auxiliary.hpp"

#include "multilevel.hpp"
#include "scheduler/LocalSearchSchedulers/hill_climbing.hpp"


class MultiLevelHillClimbingScheduler : public Scheduler {

    private:

    bool fast_coarsification;
    unsigned hc_steps;
    double contraction_factor;


  public:
    MultiLevelHillClimbingScheduler(bool fast_coars = true, unsigned hc_steps_ = 100, double contraction_factor_ = 0.25) : Scheduler(), fast_coarsification(fast_coars),  hc_steps(hc_steps_), contraction_factor(contraction_factor_) {}

    virtual ~MultiLevelHillClimbingScheduler() = default;

    void setContractionFactor(double factor_) { contraction_factor = factor_; }
    void setHcSteps(unsigned step_) { hc_steps = step_; }
    void setFastCoarsification(bool fast_coars) { fast_coarsification = fast_coars; }

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    virtual std::string getScheduleName() const override { return "MultiHC" + std::to_string(static_cast<int>(contraction_factor*100+0.5)); }
};
