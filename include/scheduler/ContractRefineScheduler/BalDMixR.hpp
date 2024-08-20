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

#include <cmath>
#include <memory>
#include <unordered_set>
#include <vector>

#include "scheduler/Scheduler.hpp"
#include "model/BspInstance.hpp"
#include "model/BspSchedule.hpp"
#include "refine/superstep_clumps.hpp"
#include "scheduler/Serial/Serial.hpp"

/**
 * @brief BalDMixR stands for Balance, Divide, Mix, Repeat
 *
 */
class BalDMixR : public Scheduler {
  private:
    struct support_structure {
        DAG G;
        SubDAG graph;

        LooseSchedule loose_schedule;

        support_structure() = delete;
        support_structure( const ComputationalDag& cdag, const Coarse_Scheduler_Params& para );
        ~support_structure() = default;
    };

    std::unique_ptr<support_structure> helper_vessel;

    std::unique_ptr<Coarse_Scheduler_Params> params;

    unsigned mixing_loops;

    void auto_mixing_loops(const BspInstance& instance);

    void run_initialise();
    void run_mix_loops();
    void run_processor_assingment(const std::vector<std::vector<unsigned>>& processsor_comm_costs);
    void run_superstep_collapses(const BspInstance& instance);
    BspSchedule produce_bsp_schedule(const BspInstance& instance);

  public:
    BalDMixR() : BalDMixR(Coarse_Scheduler_Params(1)) { };
    BalDMixR(const Coarse_Scheduler_Params params_) : BalDMixR(params_, 0) { };
    BalDMixR(const Coarse_Scheduler_Params params_, unsigned mixing_loops_);
    BalDMixR(unsigned timelimit) : BalDMixR(timelimit, Coarse_Scheduler_Params(1)) { };
    BalDMixR(unsigned timelimit, const Coarse_Scheduler_Params params_) : BalDMixR(timelimit, params_, 0) { };
    BalDMixR(unsigned timelimit, const Coarse_Scheduler_Params params_, unsigned mixing_loops_);
    virtual ~BalDMixR() = default;

    void set_mixing_loops(unsigned mixing_loops_) { mixing_loops = mixing_loops_; };

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    virtual std::string getScheduleName() const override {
        return "BalDMixR" + std::to_string(int(round((params->balance_threshhold-1)*100)));
        // removed for now
        //_balthresh" + std::to_string(params->balance_threshhold) + "NodesPerPartition" +
        //       std::to_string(std::round(params->nodes_per_partition));
    }
};