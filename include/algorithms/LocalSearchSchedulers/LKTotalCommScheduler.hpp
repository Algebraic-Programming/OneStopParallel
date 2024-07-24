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
#include <omp.h>
#include <algorithm>

#include "LKBase.hpp"


class LKTotalCommScheduler : public LKBase<double> {

  private:

    double comm_multiplier = 1.0;
 
    virtual void initalize_superstep_datastructures() override;
    virtual void cleanup_superstep_datastructures() override;
    virtual void compute_superstep_datastructures() override;

    virtual void update_superstep_datastructures(Move move) override;

    virtual void commputeCommGain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc) override;

    virtual double compute_current_costs() override;

    virtual void initializeRewardPenaltyFactors() override;
    virtual void updateRewardPenaltyFactors() override;

    virtual bool start() override;

    virtual void insertSuperstep(unsigned step) override;

    virtual void concurrent_updateNodesGain(const std::vector<VertexType> &nodes);

    virtual void select_nodes() override;

  public:
    LKTotalCommScheduler() : LKBase() {}

    virtual ~LKTotalCommScheduler() = default;

    virtual std::string getScheduleName() const override { return "LKTotalComm"; }
};