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

#include <string>

#include "algorithms/Scheduler.hpp"
#include "structures/union_find.hpp"
#include "structures/dag.hpp"
#include "algorithms/Partitioners/partitioners.hpp"
#include "algorithms/Minimal_matching/Hungarian_alg_process_permuter.hpp"

/**
 * @brief Parameters used for HDagg_simple
 * 
 */
struct HDagg_parameters {
    enum BALANCE_FUNC { MAXIMUM, XLOGX };

    float balance_threshhold;
    unsigned hillclimb_balancer_iterations;
    bool hungarian_alg;
    BALANCE_FUNC balance_function;

    HDagg_parameters(float balance_threshhold_ = 1.1, unsigned hillclimb_balancer_iterations_ = 5,
                     bool hungarian_alg_ = true, BALANCE_FUNC balance_function_ = MAXIMUM)
        : balance_threshhold(balance_threshhold_), hillclimb_balancer_iterations(hillclimb_balancer_iterations_),
          hungarian_alg(hungarian_alg_), balance_function(balance_function_) {}
};

/**
 * @brief Scheduler based on HDagg without the coarsening step. Additional locality and weight balancing efforts were implemented.
 * @brief Zarebavani, Behrooz, et al. "HDagg: hybrid aggregation of loop-carried dependence iterations in sparse matrix computations." 2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2022.
 * 
 */
class HDagg_simple : public Scheduler {
  private:
    const HDagg_parameters params;

  public:
    HDagg_simple() : HDagg_simple(HDagg_parameters()){};
    HDagg_simple(HDagg_parameters params_) : Scheduler(), params(params_){};
    HDagg_simple(unsigned timelimit) : HDagg_simple(timelimit, HDagg_parameters()){};
    HDagg_simple(unsigned timelimit, HDagg_parameters params_) : Scheduler(timelimit), params(params_){};
    virtual ~HDagg_simple() = default;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    virtual std::string getScheduleName() const override {
        return "HDaggSimple" + std::to_string(int(round((params.balance_threshhold-1)*100)));
        // removed for now
        //Bal" + std::to_string(params.balance_threshhold);
    }
};