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

#include <climits>
#include <deque>
#include <list>
#include <vector>

#include "model/BspInstance.hpp"
#include "model/BspSchedule.hpp"
#include "bsp.hpp"
#include "dag.hpp"

// schedule base class
struct Schedule_Base {
    // scheduling problem
    DAG G;
    BSPproblem params;

    int cost;
    virtual int GetCost() const = 0;
    virtual bool IsValid() const = 0;

    bool readDataFromFile(const std::string &filename, std::vector<int> &processor_assignment,
                          std::vector<int> &time_assignment, bool NoNUMA = false);

    virtual ~Schedule_Base() {}
};

// BSP schedule class
struct Schedule : Schedule_Base {

    // main ingredients of BSP` schedule
    std::vector<int> proc, supstep;
    std::vector<std::vector<std::list<int>>> supsteplists;

    // comm schedule
    std::vector<std::vector<int>> commSchedule;

    // reading schedule from file
    bool read(const std::string &filename, bool NoNUMA = false);

    // check if BSP schedule is valid
    bool IsValid() const;

    // write BSP (problem and) schedule to file
    bool WriteToFile(const std::string &filename, bool NoNUMA = false) const;

    // compute BSP schedule cost
    int GetCost() const;

    // FURTHER AUXILIARY
    // create superstep lists (for convenience) for a BSP schedule
    void CreateSupStepLists();  
    // Combine subsequent supersteps whenever there is no communication inbetween
    void RemoveNeedlessSupSteps();

    ~Schedule() {}

    //conversion to new BSP schedule format
    BspSchedule ConvertToNewSchedule(const BspInstance& instance) const;
    void ConvertFromNewSchedule(const BspSchedule& new_bsp);
};

// classical schedule (assigning to concrete time steps)
struct ClassicalSchedule : Schedule_Base {

    // main ingredients of classical schedule
    std::vector<int> proc, time;

    // reading schedule from file
    bool read(const std::string &filename, bool NoNUMA = false);

    // check if a classical (non-BSP) schedule is valid
    bool IsValid() const;
    // auxiliary for classical schedule validity check
    bool CheckJobOverlap() const;

    // compute classical schedule makespan
    int GetCost() const;

    // convert to BSP schedule
    Schedule ConvertToBSP(const std::vector<std::deque<int>> &procAssignmentLists) const;
    Schedule ConvertToBSP() const;
    // aux for conversion
    std::vector<std::deque<int>> getProcAssignmentLists() const;

    ~ClassicalSchedule() {}
};
