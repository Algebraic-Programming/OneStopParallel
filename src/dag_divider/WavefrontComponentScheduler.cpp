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

#include "dag_divider/WavefrontComponentScheduler.hpp"

BspArchitecture WavefrontComponentScheduler::setup_sub_architecture(const BspArchitecture &original,
                                                                    const ComputationalDag &sub_dag,
                                                                    const double subgraph_work_weight,
                                                                    const double total_step_work) {


    BspArchitecture sub_architecture(original);

    unsigned sub_dag_processors = 1u;
    std::vector<unsigned> sub_dag_processors_type_count = sub_architecture.getProcessorTypeCount();

    std::vector<unsigned> sub_dag_processor_types(sub_dag_processors_type_count.size(), 1u);

    if (set_num_proc_crit_path) {
        const double critical_path_w = sub_dag.critical_path_weight();
        const double parallelism = total_step_work / critical_path_w;
        
        for (unsigned i = 0; i < sub_dag_processor_types.size(); i++) {
            sub_dag_processor_types[i] = std::max(1u, (unsigned)std::floor(parallelism * (double)sub_dag_processors_type_count[i] / (double)original.numberOfProcessors()));
        }

    } else {

        const double sub_dag_work_weight_percent = subgraph_work_weight / total_step_work;

        for (unsigned i = 0; i < sub_dag_processor_types.size(); i++) {
                
                sub_dag_processor_types[i] =
                    std::max(1u, (unsigned)std::floor(sub_dag_processors_type_count[i] * sub_dag_work_weight_percent));
        }
    }

    std::vector<unsigned> sub_dag_processor_memory(sub_dag_processors_type_count.size(), std::numeric_limits<unsigned>::max());

    for (unsigned i = 0; i < original.numberOfProcessors(); i++) {
        sub_dag_processor_memory[original.processorType(i)] = std::min(original.memoryBound(i), sub_dag_processor_memory[original.processorType(i)]);
    }

    sub_architecture.set_processors_consequ_types(sub_dag_processor_types, sub_dag_processor_memory);
    //sub_architecture.print_architecture(std::cout);

    return sub_architecture;
}

std::pair<RETURN_STATUS, BspSchedule> WavefrontComponentScheduler::computeSchedule(const BspInstance &instance) {

    WavefrontComponentDivider divider;

    divider.set_dag(instance.getComputationalDag());

    divider.divide();
    divider.compute_isomorphism_map();

    BspSchedule schedule(instance);

    const std::vector<std::vector<std::vector<unsigned>>> &iosmorphism_groups = divider.get_isomorphism_groups();

    unsigned superstep_offset = 0;

    for (size_t i = 0; i < iosmorphism_groups.size(); i++) { // iterate through wavefront sets

        std::vector<int> subgraph_work_weights(iosmorphism_groups[i].size());
        int total_step_work = 0;

        for (size_t j = 0; j < iosmorphism_groups[i].size(); j++) { // iterate through isomorphism groups

            const ComputationalDag &sub_dag = divider.get_isomorphism_groups_subgraphs()[i][j];
            subgraph_work_weights[j] =
                sub_dag.sumOfVerticesWorkWeights(sub_dag.vertices().begin(), sub_dag.vertices().end());
            total_step_work += subgraph_work_weights[j] * iosmorphism_groups[i][j].size();
        }

        unsigned processors_offset = 0;
        unsigned max_number_supersteps = 0;

        for (size_t j = 0; j < iosmorphism_groups[i].size(); j++) { // iterate through isomorphism groups

            const ComputationalDag &sub_dag = divider.get_isomorphism_groups_subgraphs()[i][j];

            BspArchitecture sub_architecture =
                setup_sub_architecture(instance.getArchitecture(), sub_dag, subgraph_work_weights[j], total_step_work);

            BspInstance sub_instance(sub_dag, sub_architecture);

            auto [status, sub_schedule] = scheduler->computeSchedule(sub_instance);

            for (const auto &group_member_idx : iosmorphism_groups[i][j]) {

                VertexType subdag_vertex = 0;
                for (const auto &vertex : divider.get_vertex_maps()[i][group_member_idx]) {
                    schedule.setAssignedProcessor(vertex,
                                                  (processors_offset + sub_schedule.assignedProcessor(subdag_vertex)) %
                                                      instance.getArchitecture().numberOfProcessors());
                    schedule.setAssignedSuperstep(vertex,
                                                  superstep_offset + sub_schedule.assignedSuperstep(subdag_vertex));
                    subdag_vertex++;
                }

                processors_offset += sub_architecture.numberOfProcessors();
            }

            max_number_supersteps = std::max(max_number_supersteps, sub_schedule.numberOfSupersteps());
        }

        superstep_offset += max_number_supersteps;
    }

    schedule.updateNumberOfSupersteps();
    schedule.setLazyCommunicationSchedule();

    return {SUCCESS, schedule};
}