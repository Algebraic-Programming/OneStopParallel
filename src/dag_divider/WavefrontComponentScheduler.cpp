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
            sub_dag_processor_types[i] =
                std::max(1u, (unsigned)std::floor(parallelism * (double)sub_dag_processors_type_count[i] /
                                                  (double)original.numberOfProcessors()));
        }

    } else {

        const double sub_dag_work_weight_percent = subgraph_work_weight / total_step_work;

        for (unsigned i = 0; i < sub_dag_processor_types.size(); i++) {

            sub_dag_processor_types[i] =
                std::max(1u, (unsigned)std::floor(sub_dag_processors_type_count[i] * sub_dag_work_weight_percent));
        }
    }

    std::vector<unsigned> sub_dag_processor_memory(sub_dag_processors_type_count.size(),
                                                   std::numeric_limits<unsigned>::max());

    for (unsigned i = 0; i < original.numberOfProcessors(); i++) {
        sub_dag_processor_memory[original.processorType(i)] =
            std::min(original.memoryBound(i), sub_dag_processor_memory[original.processorType(i)]);
    }

    sub_architecture.set_processors_consequ_types(sub_dag_processor_types, sub_dag_processor_memory);
    // sub_architecture.print_architecture(std::cout);

    return sub_architecture;
}

std::pair<RETURN_STATUS, BspSchedule> WavefrontComponentScheduler::computeSchedule(const BspInstance &instance) {

    if (check_isomorphism_groups) {
        return computeSchedule_with_isomorphism_groups(instance);
    } else {
        return computeSchedule_without_isomorphism_groups(instance);
    }
}

std::pair<RETURN_STATUS, BspSchedule>
WavefrontComponentScheduler::computeSchedule_without_isomorphism_groups(const BspInstance &instance) {

    auto vertex_maps = divider->divide(instance.getComputationalDag());

    BspSchedule schedule(instance);
    const auto &proc_type_count = instance.getArchitecture().getProcessorTypeCount();

    unsigned superstep_offset = 0;

    for (size_t i = 0; i < vertex_maps.size(); i++) {

        unsigned processors_offset = 0;
        unsigned max_number_supersteps = 0;

        for (size_t j = 0; j < vertex_maps[i].size(); j++) {

            ComputationalDag sub_dag =
                dag_algorithms::create_induced_subgraph_sorted(instance.getComputationalDag(), vertex_maps[i][j]);

            if (i > 0 && instance.getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {
                for (const auto &source_vertex : sub_dag.sourceVertices()) {
                    sub_dag.setNodeCommunicationWeight(source_vertex, sub_dag.nodeCommunicationWeight(source_vertex) +
                                                                          sub_dag.nodeMemoryWeight(source_vertex));
                }
            }


            // BspArchitecture sub_architecture =
            //     setup_sub_architecture(instance.getArchitecture(), sub_dag,
            //     sub_dag.sumOfVerticesWorkWeights(sub_dag.vertices().begin(), sub_dag.vertices().end()),
            //                            instance.getComputationalDag().sumOfVerticesWorkWeights(sub_dag.vertices().begin(),
            //                            sub_dag.vertices().end()));

            std::cout << "Subdag " << i << " " << j << " " << sub_dag.numberOfVertices() << " "
                      << sub_dag.numberOfEdges() << std::endl;

            BspInstance sub_instance(sub_dag, instance.getArchitecture());
            // sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

            auto [status, sub_schedule] = scheduler->computeSchedule(sub_instance);

            for (const auto &vertex : vertex_maps[i][j]) {

                const unsigned proc_orig = sub_schedule.assignedProcessor(vertex);
                const unsigned proc_type = instance.getArchitecture().processorType(proc_orig);
                const unsigned proc = proc_orig - processors_offset;

                schedule.setAssignedProcessor(vertex, proc);
                schedule.setAssignedSuperstep(vertex, superstep_offset + sub_schedule.assignedSuperstep(vertex));
            }

            processors_offset += instance.getArchitecture().numberOfProcessors();
            max_number_supersteps = std::max(max_number_supersteps, sub_schedule.numberOfSupersteps());
        }

        superstep_offset += max_number_supersteps;
    }

    schedule.updateNumberOfSupersteps();
    schedule.setLazyCommunicationSchedule();

    return {SUCCESS, schedule};
}

std::pair<RETURN_STATUS, BspSchedule>
WavefrontComponentScheduler::computeSchedule_with_isomorphism_groups(const BspInstance &instance) {

    IsomorphismGroups iso_groups;
    auto vertex_maps = divider->divide(instance.getComputationalDag());
    iso_groups.compute_isomorphism_groups(vertex_maps, instance.getComputationalDag());

    BspSchedule schedule(instance);
    const auto &proc_type_count = instance.getArchitecture().getProcessorTypeCount();
    const std::vector<std::vector<std::vector<unsigned>>> &iosmorphism_groups = iso_groups.get_isomorphism_groups();
    unsigned superstep_offset = 0;

    for (size_t i = 0; i < iosmorphism_groups.size(); i++) { // iterate through wavefront sets

        std::vector<int> subgraph_work_weights(iosmorphism_groups[i].size());
        int total_step_work = 0;

        for (size_t j = 0; j < iosmorphism_groups[i].size(); j++) { // iterate through isomorphism groups

            const ComputationalDag &sub_dag = iso_groups.get_isomorphism_groups_subgraphs()[i][j];
            subgraph_work_weights[j] =
                sub_dag.sumOfVerticesWorkWeights(sub_dag.vertices().begin(), sub_dag.vertices().end());
            total_step_work += subgraph_work_weights[j] * iosmorphism_groups[i][j].size();
        }

        // unsigned processors_offset = 0;
        unsigned max_number_supersteps = 0;

        std::vector<unsigned> proc_type_offsets(instance.getArchitecture().getNumberOfProcessorTypes(), 0);

        for (size_t j = 1; j < proc_type_offsets.size(); j++) {
            proc_type_offsets[j] = proc_type_offsets[j - 1] + proc_type_count[j - 1];
        }

        std::vector<unsigned> proc_type_offsets_reset(proc_type_offsets);
        proc_type_offsets_reset.push_back(instance.numberOfProcessors());
        bool reset_offsets = false;

        for (size_t j = 0; j < iosmorphism_groups[i].size(); j++) { // iterate through isomorphism groups

            ComputationalDag &sub_dag = iso_groups.get_isomorphism_groups_subgraphs()[i][j];

            if (i > 0 && instance.getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {

                // for (const auto &vertex : sub_dag.vertices()) {
                //     const auto vertex_original = iosmorphism_groups[i][j][vertex];

                //     for (const auto& pred : instance.getComputationalDag().parents(vertex_original)) {

                //         if (pred < vertex) {
                //             sub_dag.setNodeCommunicationWeight(vertex, sub_dag.nodeCommunicationWeight(vertex) +
                //                                                   instance.getComputationalDag().nodeCommunicationWeight(pred));
                //         }
                //     }

                // }

                // for (const auto &source_vertex : sub_dag.sourceVertices()) {
                    
                //     sub_dag.setNodeCommunicationWeight(source_vertex, sub_dag.nodeCommunicationWeight(source_vertex) +
                //                                                           sub_dag.nodeMemoryWeight(source_vertex));
                // }
            }

            BspArchitecture sub_architecture =
                setup_sub_architecture(instance.getArchitecture(), sub_dag, subgraph_work_weights[j], total_step_work);

            BspInstance sub_instance(sub_dag, sub_architecture);
            sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

            int max_mem = 0;
            for (const auto &node : sub_dag.vertices()) {
    
                int node_mem = 0;
    
                if (sub_dag.isSource(node)) {
                    node_mem = sub_dag.nodeMemoryWeight(node);
                }
    
                for (const auto &parents : sub_dag.parents(node)) {
                    node_mem += sub_dag.nodeCommunicationWeight(parents);
                }
    
                max_mem = std::max(max_mem, node_mem);
            }
    
            std::cout << "sub inst node max mem: " << max_mem << std::endl;


            const auto sub_proc_type_count = sub_architecture.getProcessorTypeCount();
            std::vector<unsigned> proc_type_corrections(sub_architecture.getNumberOfProcessorTypes(), 0);
            for (size_t k = 1; k < proc_type_corrections.size(); k++) {
                proc_type_corrections[k] = proc_type_corrections[k - 1] + sub_proc_type_count[k - 1];
            }

            auto [status, sub_schedule] = scheduler->computeSchedule(sub_instance);

            for (const auto &group_member_idx : iosmorphism_groups[i][j]) {

                VertexType subdag_vertex = 0;
                for (const auto &vertex : vertex_maps[i][group_member_idx]) {

                    const unsigned proc_orig = sub_schedule.assignedProcessor(subdag_vertex);
                    const unsigned proc_type = sub_architecture.processorType(proc_orig);
                    const unsigned proc = proc_orig - proc_type_corrections[proc_type];

                    unsigned assign_proc = proc_type_offsets[proc_type] + proc;

                    if (assign_proc >= proc_type_offsets_reset[proc_type + 1]) {
                        assign_proc = proc_type_offsets_reset[proc_type] + proc;
                        reset_offsets = true;
                    }

                    schedule.setAssignedProcessor(vertex, assign_proc);
                    schedule.setAssignedSuperstep(vertex,
                                                  superstep_offset + sub_schedule.assignedSuperstep(subdag_vertex));
                    subdag_vertex++;
                }

                if (reset_offsets) {
                    reset_offsets = false;
                    for (size_t k = 0; k < sub_proc_type_count.size(); k++) {
                        proc_type_offsets[k] = proc_type_offsets_reset[k];
                    }
                }

                for (size_t k = 0; k < sub_proc_type_count.size(); k++) {
                    proc_type_offsets[k] += sub_proc_type_count[k];
                }
                // processors_offset += sub_architecture.numberOfProcessors();
            }

            max_number_supersteps = std::max(max_number_supersteps, sub_schedule.numberOfSupersteps());
        }

        superstep_offset += max_number_supersteps;
    }

    schedule.updateNumberOfSupersteps();
    schedule.setLazyCommunicationSchedule();

    return {SUCCESS, schedule};
}