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

#include "BspSchedule.hpp"
#include "osp/bsp/model/cost/LazyCommunicationCost.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

/**
 * @class MaxBspSchedule
 * @brief Represents a schedule for the Bulk Synchronous Parallel (BSP) model.
 *
 * @see BspInstance
 */
template <typename GraphT>
class MaxBspSchedule : public BspSchedule<GraphT> {
    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>,
                  "BspSchedule requires work and comm. weights to have the same type.");

  protected:
    using vertex_idx = vertex_idx_t<Graph_t>;

  public:
    MaxBspSchedule() = delete;

    /**
     * @brief Constructs a BspSchedule object with the specified Bspinstance->
     *
     * @param inst The BspInstance for the schedule.
     */
    MaxBspSchedule(const BspInstance<GraphT> &inst) : BspSchedule<GraphT>(inst) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep
     * assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */
    MaxBspSchedule(const BspInstance<GraphT> &inst,
                   const std::vector<unsigned> &processorAssignment,
                   const std::vector<unsigned> &superstepAssignment)
        : BspSchedule<GraphT>(inst, processorAssignment, superstepAssignment) {}

    MaxBspSchedule(const IBspSchedule<GraphT> &schedule) : BspSchedule<GraphT>(schedule) {}

    MaxBspSchedule(IBspSchedule<GraphT> &&schedule) : BspSchedule<GraphT>(std::move(schedule)) {}

    MaxBspSchedule(const MaxBspSchedule<GraphT> &schedule) = default;

    MaxBspSchedule<GraphT> &operator=(const MaxBspSchedule<GraphT> &schedule) = default;

    MaxBspSchedule(MaxBspSchedule<GraphT> &&schedule) noexcept = default;

    MaxBspSchedule<GraphT> &operator=(MaxBspSchedule<GraphT> &&schedule) noexcept = default;

    template <typename GraphTOther>
    MaxBspSchedule(const BspInstance<GraphT> &instance, const MaxBspSchedule<GraphTOther> &schedule)
        : BspSchedule<GraphT>(instance, schedule) {}

    /**
     * @brief Destructor for the BspSchedule class.
     */
    virtual ~MaxBspSchedule() = default;

    virtual v_workw_t<Graph_t> computeCosts() const override {
        std::vector<std::vector<v_commw_t<Graph_t>>> rec(this->instance->numberOfProcessors(),
                                                         std::vector<v_commw_t<Graph_t>>(this->number_of_supersteps, 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(this->instance->numberOfProcessors(),
                                                          std::vector<v_commw_t<Graph_t>>(this->number_of_supersteps, 0));

        compute_lazy_communication_costs(*this, rec, send);
        const std::vector<v_commw_t<Graph_t>> maxCommPerStep = cost_helpers::compute_max_comm_per_step(*this, rec, send);
        const std::vector<v_workw_t<Graph_t>> maxWorkPerStep = cost_helpers::compute_max_work_per_step(*this);

        v_workw_t<Graph_t> costs = 0U;
        for (unsigned step = 0U; step < this->number_of_supersteps; step++) {
            const v_commw_t<Graph_t> stepCommCost = (step == 0U) ? static_cast<v_commw_t<Graph_t>>(0)
                                                                 : max_comm_per_step[step - 1U];
            costs += std::max(step_comm_cost, max_work_per_step[step]);

            if (stepCommCost > static_cast<v_commw_t<Graph_t>>(0)) {
                costs += this->instance->synchronisationCosts();
            }
        }
        return costs;
    }

    unsigned virtual getStaleness() const override { return 2; }
};

}    // namespace osp
