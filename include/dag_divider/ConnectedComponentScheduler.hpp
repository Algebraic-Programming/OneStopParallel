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

#include "ConnectedComponentDivider.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/computational_dag_util.hpp"

namespace osp {

template<typename Graph_t>
class ConnectedComponentScheduler : public Scheduler<Graph_t> {

    Scheduler<Graph_t> *scheduler;

  public:
    ConnectedComponentScheduler(Scheduler<Graph_t> &scheduler) : scheduler(&scheduler) {}

    std::string getScheduleName() const override { return "SubDagScheduler"; }

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {

        const auto &instance = schedule.getInstance();

        const Graph_t &dag = instance.getComputationalDag();
        ConnectedComponentDivider<Graph_t> partitioner;

        partitioner.compute_connected_components(dag);

        v_workw_t<Graph_t> total_work_weight = sumOfVerticesWorkWeights(dag);

        unsigned num_processors_offset = 0;

        for (unsigned i = 0; i < partitioner.get_sub_dags().size(); i++) {
            const auto &sub_dag = partitioner.get_sub_dags()[i];
            const auto &mapping = partitioner.get_vertex_mapping()[i];

            v_workw_t<Graph_t> sub_dag_work_weight = sumOfVerticesWorkWeights(sub_dag);

            BspInstance<Graph_t> sub_instance(sub_dag, instance.getArchitecture());
            BspArchitecture<Graph_t> &sub_architecture = sub_instance.getArchitecture();

            const double sub_dag_work_weight_percent = (double)sub_dag_work_weight / (double)total_work_weight;
            const unsigned sub_dag_processors = (unsigned)(sub_dag_work_weight_percent * sub_architecture.numberOfProcessors());

            sub_architecture.setNumberOfProcessors(sub_dag_processors);

            BspSchedule<Graph_t> sub_schedule(sub_instance);
            auto status = scheduler->computeSchedule(sub_schedule);

            if (status != SUCCESS && status != BEST_FOUND) {
                return status;
            }

            for (const auto &v : sub_instance.vertices()) {
                schedule.setAssignedProcessor(mapping.at(v), sub_schedule.assignedProcessor(v) + num_processors_offset);
                schedule.setAssignedSuperstep(mapping.at(v), sub_schedule.assignedSuperstep(v));
            }

            num_processors_offset += sub_architecture.numberOfProcessors();
        }

        return SUCCESS;
    }
};

} // namespace osp