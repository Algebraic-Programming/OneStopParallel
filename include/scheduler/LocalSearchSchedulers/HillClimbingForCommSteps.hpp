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

#include "model/BspSchedule.hpp"

#include "scheduler/ImprovementScheduler.hpp"
#include "auxiliary/auxiliary.hpp"

class HillClimbingForCommSteps : public ImprovementScheduler {

    BspSchedule *schedule;
    int cost=0;

    // Main parameters for runnign algorithm
    bool steepestAscent = false;

    // aux data for comm schedule hill climbing
    std::vector<std::vector<int>> commSchedule;
    std::vector<std::vector<std::list<int>>> supsteplists;
    std::vector<std::set<intPair>> commCostList;
    std::vector<std::vector<std::set<intPair>::iterator>> commCostPointer;
    std::vector<std::vector<int>> sent, received, commCost;
    std::vector<std::vector<intPair>> commBounds;
    std::vector<std::vector<std::list<intPair>>> commSchedSendLists;
    std::vector<std::vector<std::list<intPair>::iterator>> commSchedSendListPointer;
    std::vector<std::vector<std::list<intPair>>> commSchedRecLists;
    std::vector<std::vector<std::list<intPair>::iterator>> commSchedRecListPointer;
    int nextSupstep;

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupstepLists();

    // Initialize data structures (based on current schedule)
    void Init();

        // compute cost change incurred by a potential move
    int moveCostChange(int node, int p, int step);

    // execute a move, updating the comm. schedule and the data structures
    void executeMove(int node, int p, int step, int changeCost);

    // Single comm. schedule hill climbing step
    bool Improve();

    // Convert communication schedule to new format in the end
    void ConvertCommSchedule();

  public:
    HillClimbingForCommSteps() : ImprovementScheduler() {}

    virtual ~HillClimbingForCommSteps() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule &input_schedule) override;

    //call with time limit
    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule &input_schedule, const unsigned timeLimit);

    //setting parameters
    void setSteepestAscend(bool steepestAscent_) {steepestAscent = steepestAscent_;}

    virtual std::string getScheduleName() const override { return "HCcs"; }
};
