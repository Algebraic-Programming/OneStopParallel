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
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "structures/schedule.hpp"

enum Direction { EARLIER = 0, AT, LATER };
static const int NumDirections = 3;

// aux structure for efficiently storing the changes incurred by a potential HC
// step
struct stepAuxData {
    int newCost;
    std::map<intPair, int> sentChange, recChange;
    bool canShrink = false;
};

struct Schedule;

struct HillClimbing {

    Schedule schedule;
    HillClimbing(const Schedule &s) : schedule(s) {}

    Schedule getSchedule() { return schedule; }

    // aux data structures
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

    // For memory constraints
    bool use_memory_constraint = false;
    std::vector<std::vector<int>> memory_used;
    bool violatesMemConstraint(int node, int processor, int where);

    // Compute the cost change incurred by a potential move
    int moveCostChange(int node, int p, int where, stepAuxData &changing);

    // Execute a chosen move, updating the schedule and the data structures
    void executeMove(int node, int newProc, int where, const stepAuxData &changing);

    // Single hill climbing step
    bool Improve(bool SteepestAscent = false, bool shrink = true);

    // Main method for hill climbing (with time limit)
    void HillClimb(int TimeLimit = 600, bool SteepestAscent = false, bool shrink = true);

    // Hill climbing for limited number of steps
    void HillClimbSteps(int StepsLimit = 10, bool SteepestAscent = false, bool shrink = true);

};

struct HillClimbingCS {

    Schedule schedule;
    HillClimbingCS(const Schedule &s) : schedule(s) {}

    Schedule getSchedule() { return schedule; }

    // aux data for comm schedule hill climbing
    std::vector<std::set<intPair>> commCostList;
    std::vector<std::vector<std::set<intPair>::iterator>> commCostPointer;
    std::vector<std::vector<int>> sent, received, commCost;
    std::vector<std::vector<intPair>> commBounds;
    std::vector<std::vector<std::list<intPair>>> commSchedSendLists;
    std::vector<std::vector<std::list<intPair>::iterator>> commSchedSendListPointer;
    std::vector<std::vector<std::list<intPair>>> commSchedRecLists;
    std::vector<std::vector<std::list<intPair>::iterator>> commSchedRecListPointer;
    int nextSupstep;

    // Initialization for comm. schedule hill climbing
    void Init();

    // compute cost change incurred by a potential move
    int moveCostChange(int node, int p, int step);

    // execute a move, updating the comm. schedule and the data structures
    void executeMove(int node, int p, int step, int changeCost);

    // Single comm. schedule hill climbing step
    bool Improve(bool SteepestAscent = false);

    // Main function for comm. schedule hill climbing
    void HillClimb(int TimeLimit = 60, bool SteepestAscent = false);
};
