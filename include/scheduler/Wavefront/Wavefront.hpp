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

#include "scheduler/Scheduler.hpp"
#include "structures/union_find.hpp"
#include "structures/dag.hpp"
#include "scheduler/Partitioners/partitioners.hpp"
#include "scheduler/Minimal_matching/Hungarian_alg_process_permuter.hpp"

/**
 * @brief Parameters used for HDagg_simple
 * 
 */
struct Wavefront_parameters {
    unsigned hillclimb_balancer_iterations;
    bool hungarian_alg;

    Wavefront_parameters(unsigned hillclimb_balancer_iterations_ = 5, bool hungarian_alg_ = true)
        : hillclimb_balancer_iterations(hillclimb_balancer_iterations_),
          hungarian_alg(hungarian_alg_) {}
};

/**
 * @brief Scheduler based on HDagg without the coarsening step. Additional locality and weight balancing efforts were implemented.
 * @brief Zarebavani, Behrooz, et al. "HDagg: hybrid aggregation of loop-carried dependence iterations in sparse matrix computations." 2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2022.
 * 
 */
class Wavefront : public Scheduler {
  private:
    const Wavefront_parameters params;

  public:
    Wavefront() : Wavefront(Wavefront_parameters()){};
    Wavefront(Wavefront_parameters params_) : Scheduler(), params(params_){};
    Wavefront(unsigned timelimit) : Wavefront(timelimit, Wavefront_parameters()){};
    Wavefront(unsigned timelimit, Wavefront_parameters params_) : Scheduler(timelimit), params(params_){};
    virtual ~Wavefront() = default;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    virtual std::string getScheduleName() const override {
        return "Wavefront";
        // removed for now
        //Bal" + std::to_string(params.balance_threshhold);
    }
};