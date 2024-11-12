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
#include "scheduler/Scheduler.hpp"
#include "dag_partitioners/ConnectedComponentPartitioner.hpp"

class ConnectedComponentScheduler : public Scheduler {

    Scheduler *scheduler;

  public:

    ConnectedComponentScheduler(Scheduler& scheduler) : scheduler(&scheduler) {}

    std::string getScheduleName() const override { return "SubDagScheduler"; }

    std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {

        ComputationalDag dag = instance.getComputationalDag();
        ConnectedComponentPartitioner partitioner;
        partitioner.compute_connected_components(dag);

        BspSchedule schedule(instance);

        unsigned total_work_weight = dag.sumOfVerticesWorkWeights(dag.vertices().begin(), dag.vertices().end());

        unsigned num_processors_offset = 0;

        for (unsigned i = 0; i < partitioner.get_sub_dags().size(); i++) {
            const auto &sub_dag = partitioner.get_sub_dags()[i];
            const auto &mapping = partitioner.get_vertex_mapping()[i];

            unsigned sub_dag_work_weight = sub_dag.sumOfVerticesWorkWeights(sub_dag.vertices().begin(), sub_dag.vertices().end());

            BspArchitecture sub_architecture = instance.getArchitecture();
            const double sub_dag_work_weight_percent = (double) sub_dag_work_weight / (double) total_work_weight;
            const unsigned sub_dag_processors = sub_dag_work_weight_percent * sub_architecture.numberOfProcessors();
            
            sub_architecture.setNumberOfProcessors(sub_dag_processors);


            BspInstance sub_instance(sub_dag, sub_architecture);

            auto [status, sub_schedule] = scheduler->computeSchedule(sub_instance);

            for (unsigned v = 0; v < sub_instance.numberOfVertices(); v++) {
                schedule.setAssignedProcessor(mapping.at(v), sub_schedule.assignedProcessor(v) + num_processors_offset);
                schedule.setAssignedSuperstep(mapping.at(v), sub_schedule.assignedSuperstep(v));
            }

            num_processors_offset += sub_architecture.numberOfProcessors();

        }

        schedule.updateNumberOfSupersteps();
        schedule.setLazyCommunicationSchedule();

        return {SUCCESS, schedule};
    }
};
