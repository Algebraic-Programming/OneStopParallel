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

#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "BspScheduleCS.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

template<typename Graph_t>
class MaxBspScheduleCS : public BspScheduleCS<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t> >, "BspSchedule requires work and comm. weights to have the same type.");

  protected:
    using vertex_idx = vertex_idx_t<Graph_t>;
   

  public:
  
    MaxBspScheduleCS() = delete;

    /**
     * @brief Constructs a BspSchedule object with the specified Bspinstance->
     *
     * @param inst The BspInstance for the schedule.
     */
    MaxBspScheduleCS(const BspInstance<Graph_t> &inst) : BspScheduleCS<Graph_t>(inst) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep
     * assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */
    MaxBspScheduleCS(const BspInstance<Graph_t> &inst, const std::vector<unsigned> &processor_assignment_, const std::vector<unsigned> &superstep_assignment_)
        : BspScheduleCS<Graph_t>(inst, processor_assignment_, superstep_assignment_) {}

    MaxBspScheduleCS(const BspScheduleCS<Graph_t> &schedule) : BspScheduleCS<Graph_t>(schedule) {}
    MaxBspScheduleCS(BspScheduleCS<Graph_t> &&schedule) : BspScheduleCS<Graph_t>(std::move(schedule)) {}

    MaxBspScheduleCS(const MaxBspScheduleCS<Graph_t> &schedule) = default;
    MaxBspScheduleCS(MaxBspScheduleCS<Graph_t> &&schedule) = default;

    MaxBspScheduleCS<Graph_t> &operator=(const MaxBspScheduleCS<Graph_t> &schedule) = default;
    MaxBspScheduleCS<Graph_t> &operator=(MaxBspScheduleCS<Graph_t> &&schedule) = default;

    template<typename Graph_t_other>
    MaxBspScheduleCS(const BspInstance<Graph_t> &instance_, const MaxBspScheduleCS<Graph_t_other> &schedule)
        : BspScheduleCS<Graph_t>(instance_, schedule) {}

    /**
     * @brief Destructor for the BspSchedule class.
     */
    virtual ~MaxBspScheduleCS() = default;
   
    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u
     * and v are assigned to different processors, the superstep assigned to node u is less than the superstep assigned
     * to node v.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    virtual bool satisfiesPrecedenceConstraints() const override {
        return this->template satisfies_precedence_constraints_staleness<2>();
    };

    virtual v_workw_t<Graph_t> computeCosts() const override { 
        
        std::vector<std::vector<v_commw_t<Graph_t>>> rec(this->getInstance().numberOfProcessors(),
                                                         std::vector<v_commw_t<Graph_t>>(this->number_of_supersteps, 0));

        std::vector<std::vector<v_commw_t<Graph_t>>> send(this->getInstance().numberOfProcessors(),
                                                          std::vector<v_commw_t<Graph_t>>(this->number_of_supersteps, 0));

        this->compute_cs_communication_costs_helper(rec, send);
        const std::vector<v_commw_t<Graph_t>> max_comm_per_step = this->compute_max_comm_per_step_helper(rec, send);
        const std::vector<v_workw_t<Graph_t>> max_work_per_step = this->compute_max_work_per_step_helper();

        v_workw_t<Graph_t> costs = 0U;
        for (unsigned step = 0U; step < this->number_of_supersteps; step++) {
            v_commw_t<Graph_t> step_comm_cost = (step == 0U) ? static_cast<v_commw_t<Graph_t>>(0) : max_comm_per_step[step - 1U];
            if (step_comm_cost > static_cast<v_commw_t<Graph_t>>(0)) {
                step_comm_cost += this->instance->synchronisationCosts();
            }
            costs += std::max(step_comm_cost, max_work_per_step[step]);

        }
        return costs;
    }

    virtual bool isMaxBsp() const override { return true; }
};

} // namespace osp