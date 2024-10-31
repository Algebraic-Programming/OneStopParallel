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

#include "hill_climbing.hpp"

class HillClimbingScheduler : public ImprovementScheduler {

    BspSchedule *schedule;
    unsigned cost=0;

    // Main parameters for runnign algorithm
    bool shrink = true;
    bool steepestAscent = false;

    // aux data structures
    std::vector<std::vector<std::list<int>>> supsteplists;
    std::vector<std::vector<std::vector<bool>>> canMove;
    std::vector<std::list<intPair>> moveOptions;
    std::vector<std::vector<std::vector<std::list<intPair>::iterator>>> movePointer;
    std::vector<std::vector<std::map<int, int>>> succSteps;
    std::vector<std::vector<int>> workCost, sent, received, commCost;
    std::vector<std::set<intPair>> workCostList, commCostList;
    std::vector<std::vector<std::set<intPair>::iterator>> workCostPointer, commCostPointer;
    std::vector<std::list<int>::iterator> supStepListPointer;
    std::pair<int, std::list<intPair>::iterator> nextMove;
    bool HCwithLatency = true;

    // for improved candidate selection
    std::deque<intTriple> promisingMoves;
    bool findPromisingMoves = true;

    // Initialize data structures (based on current schedule)
    void Init();
    void updatePromisingMoves();

    // Functions to compute and update the std::list of possible moves
    void updateNodeMovesEarlier(int node);
    void updateNodeMovesAt(int node);
    void updateNodeMovesLater(int node);
    void updateNodeMoves(int node);
    void updateMoveOptions(int node, int where);

    void addMoveOption(int node, int p, Direction dir);

    void eraseMoveOption(int node, int p, Direction dir);
    void eraseMoveOptionsEarlier(int node);
    void eraseMoveOptionsAt(int node);
    void eraseMoveOptionsLater(int node);
    void eraseMoveOptions(int node);

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupstepLists();

    // Combine subsequent supersteps whenever there is no communication inbetween
    void RemoveNeedlessSupSteps();

    // For memory constraints
    bool use_memory_constraint = false;
    std::vector<std::vector<int>> memory_used;
    bool violatesMemConstraint(int node, int processor, int where);

    // Compute the cost change incurred by a potential move
    int moveCostChange(int node, int p, int where, stepAuxData &changing);

    // Execute a chosen move, updating the schedule and the data structures
    void executeMove(int node, int newProc, int where, const stepAuxData &changing);

    // Single hill climbing step
    bool Improve();

  public:
    HillClimbingScheduler() : ImprovementScheduler() {}

    virtual ~HillClimbingScheduler() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule &input_schedule) override;

    //call with time/step limits
    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule &input_schedule, const unsigned timeLimit);
    virtual RETURN_STATUS improveScheduleWithStepLimit(BspSchedule &input_schedule, const unsigned stepLimit = 10);

    //setting parameters
    void setSteepestAscend(bool steepestAscent_) {steepestAscent = steepestAscent_;}
    void setShrink(bool shrink_) {shrink = shrink_;}

    virtual std::string getScheduleName() const override { return "HC"; }
};
