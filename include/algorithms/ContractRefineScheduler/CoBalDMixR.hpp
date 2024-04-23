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

#include "algorithms/ContractRefineScheduler/contract_refine_scheduler.hpp"
#include "algorithms/Scheduler.hpp"
#include "model/BspInstance.hpp"
#include "model/BspSchedule.hpp"
#include "refine/superstep_clumps.hpp"
#include "algorithms/Serial/Serial.hpp"

/**
 * @brief CoBalDMixR stands for Coarsen, Balance, Divide, Mix, Repeat
 *
 */
class CoBalDMixR : public Scheduler {
  private:
    struct support_structure {
      const DAG original_graph;
      CoarseRefineScheduler coarse_refiner;

      support_structure() = delete;
      support_structure( const ComputationalDag& cdag, const CoarseRefineScheduler_parameters& para );
      ~support_structure() = default;
    };

    std::unique_ptr<support_structure> helper_vessel;

    std::unique_ptr<CoarseRefineScheduler_parameters> params;

    void run_processor_assingment(const std::vector<std::vector<unsigned>>& processsor_comm_costs);
    void run_superstep_collapses(const BspInstance& instance);
    BspSchedule produce_bsp_schedule(const BspInstance& instance);

  public:
    CoBalDMixR() : CoBalDMixR( CoarseRefineScheduler_parameters( Coarse_Scheduler_Params(1) ) ) { };
    CoBalDMixR(const CoarseRefineScheduler_parameters params_);
    CoBalDMixR(unsigned timelimit) : CoBalDMixR(timelimit, CoarseRefineScheduler_parameters( Coarse_Scheduler_Params(1) )) { };
    CoBalDMixR(unsigned timelimit, const CoarseRefineScheduler_parameters params_);
    virtual ~CoBalDMixR() = default;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    virtual std::string getScheduleName() const override {
        return "CoBalDMixR"+std::to_string(int(round((params->coarse_schedule_params_initial.balance_threshhold-1)*100)));
    }
};