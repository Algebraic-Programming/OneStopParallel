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

template <typename GraphT>
class MaxBspScheduleCS : public BspScheduleCS<GraphT> {
    static_assert(IsComputationalDagV<Graph_t>, "BspSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>,
                  "BspSchedule requires work and comm. weights to have the same type.");

  protected:
    using vertex_idx = vertex_idx_t<Graph_t>;

  public:
    MaxBspScheduleCS() = delete;

    /**
     * @brief Constructs a BspSchedule object with the specified Bspinstance->
     *
     * @param inst The BspInstance for the schedule.
     */
    MaxBspScheduleCS(const BspInstance<GraphT> &inst) : BspScheduleCS<GraphT>(inst) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep
     * assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */
    MaxBspScheduleCS(const BspInstance<GraphT> &inst,
                     const std::vector<unsigned> &processorAssignment,
                     const std::vector<unsigned> &superstepAssignment)
        : BspScheduleCS<GraphT>(inst, processorAssignment, superstepAssignment) {}

    MaxBspScheduleCS(const BspScheduleCS<GraphT> &schedule) : BspScheduleCS<GraphT>(schedule) {}

    MaxBspScheduleCS(BspScheduleCS<GraphT> &&schedule) : BspScheduleCS<GraphT>(std::move(schedule)) {}

    MaxBspScheduleCS(const MaxBspSchedule<GraphT> &schedule) : BspScheduleCS<GraphT>(schedule) {
        this->setAutoCommunicationSchedule();
    }

    MaxBspScheduleCS(MaxBspSchedule<GraphT> &&schedule) : BspScheduleCS<GraphT>(std::move(schedule)) {
        this->setAutoCommunicationSchedule();
    }

    MaxBspScheduleCS(const MaxBspScheduleCS<GraphT> &schedule) = default;
    MaxBspScheduleCS(MaxBspScheduleCS<GraphT> &&schedule) = default;

    MaxBspScheduleCS<GraphT> &operator=(const MaxBspScheduleCS<GraphT> &schedule) = default;
    MaxBspScheduleCS<GraphT> &operator=(MaxBspScheduleCS<GraphT> &&schedule) = default;

    template <typename GraphTOther>
    MaxBspScheduleCS(const BspInstance<GraphT> &instance, const MaxBspScheduleCS<GraphTOther> &schedule)
        : BspScheduleCS<GraphT>(instance, schedule) {}

    /**
     * @brief Destructor for the BspSchedule class.
     */
    virtual ~MaxBspScheduleCS() = default;

    virtual v_workw_t<Graph_t> computeCosts() const override {
        std::vector<std::vector<v_commw_t<Graph_t>>> rec(this->getInstance().numberOfProcessors(),
                                                         std::vector<v_commw_t<Graph_t>>(this->number_of_supersteps, 0));

        std::vector<std::vector<v_commw_t<Graph_t>>> send(this->getInstance().numberOfProcessors(),
                                                          std::vector<v_commw_t<Graph_t>>(this->number_of_supersteps, 0));

        this->compute_cs_communication_costs_helper(rec, send);
        const std::vector<v_commw_t<Graph_t>> maxCommPerStep = cost_helpers::compute_max_comm_per_step(*this, rec, send);
        const std::vector<v_workw_t<Graph_t>> maxWorkPerStep = cost_helpers::compute_max_work_per_step(*this);

        v_workw_t<Graph_t> costs = 0U;
        for (unsigned step = 0U; step < this->number_of_supersteps; step++) {
            const auto stepCommCost = (step == 0U) ? static_cast<v_commw_t<Graph_t>>(0) : max_comm_per_step[step - 1U];
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
