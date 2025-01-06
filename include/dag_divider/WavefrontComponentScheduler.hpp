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
#include "WavefrontComponentDivider.hpp"
#include "scheduler/Scheduler.hpp"

class WavefrontComponentScheduler : public Scheduler {

    bool set_num_proc_crit_path = false;

    Scheduler *scheduler;

  public:
    WavefrontComponentScheduler(Scheduler &scheduler) : scheduler(&scheduler) {}

    std::string getScheduleName() const override { return "WavefrontComponentScheduler"; }

    std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {

        WavefrontComponentDivider divider;

        divider.set_dag(instance.getComputationalDag());

        divider.divide();
        divider.compute_isomorphism_map();

        BspSchedule schedule(instance);

        const std::vector<std::vector<std::vector<unsigned>>> iosmorphism_groups = divider.get_isomorphism_groups();

        unsigned superstep_offset = 0;

        for (size_t i = 0; i < iosmorphism_groups.size(); i++) {

            std::vector<int> subgraph_work_weights(iosmorphism_groups[i].size());
            int total_step_work = 0;

            for (size_t j = 0; j < iosmorphism_groups[i].size(); j++) {

                const auto &sub_dag = divider.get_isomorphism_groups_subgraphs()[i][j];
                subgraph_work_weights[j] =
                    sub_dag.sumOfVerticesWorkWeights(sub_dag.vertices().begin(), sub_dag.vertices().end());
                total_step_work += subgraph_work_weights[j] * iosmorphism_groups[i][j].size();
            }

            unsigned processors_offset = 0;
            unsigned max_number_supersteps = 0;

            for (size_t j = 0; j < iosmorphism_groups[i].size(); j++) {

                const auto &sub_dag = divider.get_isomorphism_groups_subgraphs()[i][j];

                BspArchitecture sub_architecture = instance.getArchitecture();
                unsigned sub_dag_processors = 1u;

                if (set_num_proc_crit_path) {
                    const int critical_path_w = sub_dag.critical_path_weight();
                    const double parallelism = (double)total_step_work / (double)critical_path_w;

                    const unsigned sub_dag_processors = std::max(1u, (unsigned)std::floor(parallelism));
                } else {

                    const double sub_dag_work_weight_percent =
                        (double)subgraph_work_weights[j] / (double)total_step_work;

                    sub_dag_processors =
                        std::max(1u, (unsigned)(sub_dag_work_weight_percent * sub_architecture.numberOfProcessors()));
                }

                sub_architecture.setNumberOfProcessors(sub_dag_processors);

                BspInstance sub_instance(sub_dag, sub_architecture);

                auto [status, sub_schedule] = scheduler->computeSchedule(sub_instance);

                for (const auto &group_member_idx : iosmorphism_groups[i][j]) {

                    VertexType subdag_vertex = 0;
                    for (const auto &vertex : divider.get_vertex_maps()[i][group_member_idx]) {
                        schedule.setAssignedProcessor(
                            vertex, (processors_offset + sub_schedule.assignedProcessor(subdag_vertex)) %
                                        instance.getArchitecture().numberOfProcessors());
                        schedule.setAssignedSuperstep(vertex,
                                                      superstep_offset + sub_schedule.assignedSuperstep(subdag_vertex));
                        subdag_vertex++;
                    }

                    processors_offset += sub_dag_processors;
                }

                max_number_supersteps = std::max(max_number_supersteps, sub_schedule.numberOfSupersteps());
            }

            superstep_offset += max_number_supersteps;
        }

        schedule.updateNumberOfSupersteps();
        schedule.setLazyCommunicationSchedule();

        return {SUCCESS, schedule};
    }
};
