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
    static_assert(isComputationalDagV<GraphT>, "BspSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "BspSchedule requires work and comm. weights to have the same type.");

  protected:
    using VertexIdx = VertexIdxT<GraphT>;

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
        this->SetAutoCommunicationSchedule();
    }

    MaxBspScheduleCS(MaxBspSchedule<GraphT> &&schedule) : BspScheduleCS<GraphT>(std::move(schedule)) {
        this->SetAutoCommunicationSchedule();
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

    virtual VWorkwT<GraphT> ComputeCosts() const override {
        std::vector<std::vector<VCommwT<GraphT>>> rec(this->instance_->NumberOfProcessors(),
                                                      std::vector<VCommwT<GraphT>>(this->NumberOfSupersteps(), 0));

        std::vector<std::vector<VCommwT<GraphT>>> send(this->instance_->NumberOfProcessors(),
                                                       std::vector<VCommwT<GraphT>>(this->NumberOfSupersteps(), 0));

        this->ComputeCsCommunicationCostsHelper(rec, send);
        const std::vector<VCommwT<GraphT>> maxCommPerStep = cost_helpers::ComputeMaxCommPerStep(*this, rec, send);
        const std::vector<VWorkwT<GraphT>> maxWorkPerStep = cost_helpers::ComputeMaxWorkPerStep(*this);

        VWorkwT<GraphT> costs = 0U;
        for (unsigned step = 0U; step < this->NumberOfSupersteps(); step++) {
            const auto stepCommCost = (step == 0U) ? static_cast<VCommwT<GraphT>>(0) : maxCommPerStep[step - 1U];
            costs += std::max(stepCommCost, maxWorkPerStep[step]);

            if (stepCommCost > static_cast<VCommwT<GraphT>>(0)) {
                costs += this->instance_->SynchronisationCosts();
            }
        }
        return costs;
    }

    unsigned virtual GetStaleness() const override { return 2; }
};

}    // namespace osp
