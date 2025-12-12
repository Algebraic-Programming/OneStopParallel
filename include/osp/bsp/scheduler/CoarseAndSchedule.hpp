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

#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/coarser/Coarser.hpp"
#include "osp/coarser/coarser_util.hpp"

namespace osp {

template <typename GraphT, typename GraphTCoarse>
class CoarseAndSchedule : public Scheduler<GraphT> {
  private:
    Coarser<GraphT, GraphTCoarse> &coarser_;
    Scheduler<GraphTCoarse> &scheduler_;

  public:
    CoarseAndSchedule(Coarser<GraphT, GraphTCoarse> &coarser, Scheduler<GraphTCoarse> &scheduler)
        : coarser_(coarser), scheduler_(scheduler) {}

    std::string getScheduleName() const override {
        return "Coarse(" + coarser_.getCoarserName() + ")AndSchedule(" + scheduler_.getScheduleName() + ")";
    }

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();

        BspInstance<GraphTCoarse> instanceCoarse;

        std::vector<VertexIdxT<Graph_t_coarse>> reverseVertexMap;

        bool status = coarser_.coarsenDag(instance.GetComputationalDag(), instanceCoarse.GetComputationalDag(), reverse_vertex_map);

        if (!status) {
            return ReturnStatus::ERROR;
        }

        instanceCoarse.GetArchitecture() = instance.GetArchitecture();
        instanceCoarse.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

        BspSchedule<GraphTCoarse> scheduleCoarse(instanceCoarse);

        const auto statusCoarse = scheduler_.ComputeSchedule(scheduleCoarse);

        if (status_coarse != ReturnStatus::OSP_SUCCESS and status_coarse != ReturnStatus::BEST_FOUND) {
            return statusCoarse;
        }

        coarser_util::pull_back_schedule(scheduleCoarse, reverse_vertex_map, schedule);

        return ReturnStatus::OSP_SUCCESS;
    }
};

}    // namespace osp
