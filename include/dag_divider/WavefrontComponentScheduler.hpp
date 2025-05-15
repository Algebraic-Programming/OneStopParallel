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
#include "DagDivider.hpp"
#include "IsomorphismGroups.hpp"
#include "WavefrontComponentDivider.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/computational_dag_util.hpp"
#include "graph_algorithms/subgraph_algorithms.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"

namespace osp {

template<typename Graph_t, typename constr_graph_t>
class WavefrontComponentScheduler : public Scheduler<Graph_t> {

    bool set_num_proc_crit_path = false;

    IDagDivider<Graph_t> *divider;

    Scheduler<constr_graph_t> *scheduler;

    bool check_isomorphism_groups = true;

    BspArchitecture<Graph_t> setup_sub_architecture(const BspArchitecture<Graph_t> &original,
                                                    const constr_graph_t &sub_dag, const double subgraph_work_weight,
                                                    const double total_step_work) {

        BspArchitecture sub_architecture(original);

        std::vector<unsigned> sub_dag_processors_type_count = sub_architecture.getProcessorTypeCount();

        std::vector<unsigned> sub_dag_processor_types(sub_dag_processors_type_count.size(), 1u);

        if (set_num_proc_crit_path) {
            const double critical_path_w = critical_path_weight(sub_dag);
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

        std::vector<v_memw_t<Graph_t>> sub_dag_processor_memory(sub_dag_processors_type_count.size(),
                                                                std::numeric_limits<v_memw_t<Graph_t>>::max());

        for (unsigned i = 0; i < original.numberOfProcessors(); i++) {
            sub_dag_processor_memory[original.processorType(i)] =
                std::min(original.memoryBound(i), sub_dag_processor_memory[original.processorType(i)]);
        }

        sub_architecture.set_processors_consequ_types(sub_dag_processor_types, sub_dag_processor_memory);
        // sub_architecture.print_architecture(std::cout);

        return sub_architecture;
    }

    RETURN_STATUS computeSchedule_with_isomorphism_groups(BspSchedule<Graph_t> &schedule) {

        const auto &instance = schedule.getInstance();

        IsomorphismGroups<Graph_t, constr_graph_t> iso_groups;
        auto vertex_maps = divider->divide(instance.getComputationalDag());
        iso_groups.compute_isomorphism_groups(vertex_maps, instance.getComputationalDag());

        const auto &proc_type_count = instance.getArchitecture().getProcessorTypeCount();
        const auto &iosmorphism_groups = iso_groups.get_isomorphism_groups();
        unsigned superstep_offset = 0;

        for (std::size_t i = 0; i < iosmorphism_groups.size(); i++) { // iterate through wavefront sets

            std::vector<v_workw_t<constr_graph_t>> subgraph_work_weights(iosmorphism_groups[i].size());
            v_workw_t<constr_graph_t> total_step_work = 0;

            for (std::size_t j = 0; j < iosmorphism_groups[i].size(); j++) { // iterate through isomorphism groups

                const constr_graph_t &sub_dag = iso_groups.get_isomorphism_groups_subgraphs()[i][j];

                subgraph_work_weights[j] = sumOfVerticesWorkWeights(sub_dag);
                total_step_work +=
                    subgraph_work_weights[j] * static_cast<v_workw_t<constr_graph_t>>(iosmorphism_groups[i][j].size());
            }

            // unsigned processors_offset = 0;
            unsigned max_number_supersteps = 0;

            std::vector<unsigned> proc_type_offsets(instance.getArchitecture().getNumberOfProcessorTypes(), 0);

            for (std::size_t j = 1; j < proc_type_offsets.size(); j++) {
                proc_type_offsets[j] = proc_type_offsets[j - 1] + proc_type_count[j - 1];
            }

            std::vector<unsigned> proc_type_offsets_reset(proc_type_offsets);
            proc_type_offsets_reset.push_back(instance.numberOfProcessors());
            bool reset_offsets = false;

            for (std::size_t j = 0; j < iosmorphism_groups[i].size(); j++) { // iterate through isomorphism groups

                constr_graph_t &sub_dag = iso_groups.get_isomorphism_groups_subgraphs()[i][j];

                if (i > 0 && instance.getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {

                    for (const auto &source_vertex : source_vertices_view(sub_dag)) {

                        sub_dag.set_vertex_comm_weight(source_vertex, sub_dag.vertex_comm_weight(source_vertex) +
                                                                          sub_dag.vertex_mem_weight(source_vertex));
                    }
                }

                BspInstance<constr_graph_t> sub_instance(
                    sub_dag, setup_sub_architecture(instance.getArchitecture(), sub_dag, subgraph_work_weights[j],
                                                    total_step_work));

                sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

                const BspArchitecture<constr_graph_t> &sub_architecture = sub_instance.getArchitecture();

                BspSchedule<constr_graph_t> sub_schedule(sub_instance);
                auto status = scheduler->computeSchedule(sub_schedule);

                if (status != SUCCESS && status != BEST_FOUND) {
                    return status;
                }

                const auto sub_proc_type_count = sub_architecture.getProcessorTypeCount();
                std::vector<unsigned> proc_type_corrections(sub_architecture.getNumberOfProcessorTypes(), 0);
                for (std::size_t k = 1; k < proc_type_corrections.size(); k++) {
                    proc_type_corrections[k] = proc_type_corrections[k - 1] + sub_proc_type_count[k - 1];
                }

                for (const auto &group_member_idx : iosmorphism_groups[i][j]) {

                    vertex_idx_t<constr_graph_t> subdag_vertex = 0;
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
                        for (std::size_t k = 0; k < sub_proc_type_count.size(); k++) {
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

        return SUCCESS;
    }

    RETURN_STATUS computeSchedule_without_isomorphism_groups(BspSchedule<Graph_t> &schedule) {

        auto instance = schedule.getInstance();
        auto vertex_maps = divider->divide(instance.getComputationalDag());
        const auto &proc_type_count = instance.getArchitecture().getProcessorTypeCount();
        unsigned superstep_offset = 0;

        for (std::size_t i = 0; i < vertex_maps.size(); i++) {

            BspInstance<constr_graph_t> sub_instance;
            sub_instance.setArchitecture(instance.getArchitecture());
            sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

            for (std::size_t j = 0; j < vertex_maps[i].size(); j++) {

                create_induced_subgraph(instance.getComputationalDag(), sub_instance.getComputationalDag(),
                                        vertex_maps[i][j]);
            }

            auto &sub_dag = sub_instance.getComputationalDag();

            if (i > 0 && instance.getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {

                for (const auto &source_vertex : source_vertices_view(sub_dag)) {
                    sub_dag.set_vertex_comm_weight(source_vertex, sub_dag.vertex_comm_weight(source_vertex) +
                                                                      sub_dag.vertex_mem_weight(source_vertex));
                }
            }

            BspSchedule<constr_graph_t> sub_schedule(sub_instance);
            const auto status = scheduler->computeSchedule(sub_schedule);

            if (status != SUCCESS && status != BEST_FOUND) {
                return status;
            }

            vertex_idx_t<constr_graph_t> subdag_vertex = 0;
            for (std::size_t j = 0; j < vertex_maps[i].size(); j++) {

                std::vector<vertex_idx_t<Graph_t>> subdag_vertices(vertex_maps[i][j].begin(), vertex_maps[i][j].end());
                std::sort(subdag_vertices.begin(), subdag_vertices.end());

                for (size_t k = 0; k < subdag_vertices.size(); k++) {
                    schedule.setAssignedProcessor(subdag_vertices[k], sub_schedule.assignedProcessor(subdag_vertex));

                    schedule.setAssignedSuperstep(subdag_vertices[k],
                                                  superstep_offset + sub_schedule.assignedSuperstep(subdag_vertex));

                    subdag_vertex++;
                }
            }

            superstep_offset += sub_schedule.numberOfSupersteps();
        }

        return SUCCESS;
    }

  public:
    WavefrontComponentScheduler(IDagDivider<Graph_t> &div, Scheduler<constr_graph_t> &scheduler)
        : divider(&div), scheduler(&scheduler) {}

    void set_check_isomorphism_groups(bool check) { check_isomorphism_groups = check; }

    std::string getScheduleName() const override { return "WavefrontComponentScheduler"; }

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {

        if (check_isomorphism_groups) {
            return computeSchedule_with_isomorphism_groups(schedule);
        } else {
            return computeSchedule_without_isomorphism_groups(schedule);
        }
    }
};

template<typename Graph_t>
using WavefrontComponentScheduler_def_t = WavefrontComponentScheduler<Graph_t, boost_graph>;

} // namespace osp