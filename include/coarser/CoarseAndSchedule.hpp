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

#include "Coarser.hpp"
#include "bsp/scheduler/Scheduler.hpp"

namespace osp {

template<typename Graph_t, typename Graph_t_coarse>
class CoarseAndSchedule : public Scheduler<Graph_t> {

  private:
    Coarser<Graph_t, Graph_t_coarse> &coarser;
    Scheduler<Graph_t_coarse> &scheduler;

  public:
    CoarseAndSchedule(Coarser<Graph_t, Graph_t_coarse> &coarser_, Scheduler<Graph_t_coarse> &scheduler_)
        : coarser(coarser_), scheduler(scheduler_) {}

    std::string getScheduleName() const override { return "CoarseAndSchedule"; }

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {

        const auto &instance = schedule.getInstance();

        BspInstance<Graph_t_coarse> instance_coarse;

        std::vector<std::vector<vertex_idx_t<Graph_t>>> vertex_map;
        std::vector<vertex_idx_t<Graph_t_coarse>> reverse_vertex_map;

        bool status = coarser.coarseDag(instance.getComputationalDag(), instance_coarse.getComputationalDag(),
                                        vertex_map, reverse_vertex_map);

        if (!status) {
            return ERROR;
        }

        instance_coarse.setArchitecture(instance.getArchitecture());
        instance_coarse.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

        BspSchedule<Graph_t_coarse> schedule_coarse(instance_coarse);

        const auto status_coarse = scheduler.computeSchedule(schedule_coarse);

        if (status_coarse != SUCCESS and status_coarse != BEST_FOUND) {
            return status_coarse;
        }

        pull_back_schedule(schedule_coarse, vertex_map, schedule);

        return SUCCESS;
    }
};

} // namespace osp