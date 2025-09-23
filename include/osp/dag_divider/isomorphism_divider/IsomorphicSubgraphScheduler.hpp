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

#include <iostream>
#include "WavefrontOrbitProcessor.hpp"
#include "EftSubgraphScheduler.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp {


template<typename Graph_t, typename Constr_Graph_t>
class IsomorphicSubgraphScheduler {

    private:

    static constexpr bool verbose = true;
    
    size_t symmetry_ = 2;
    Scheduler<Graph_t> * bsp_scheduler_;

    bool plot_dot_graphs_ = false;

    public:

    IsomorphicSubgraphScheduler(Scheduler<Graph_t> & bsp_scheduler) : symmetry_(2), bsp_scheduler_(&bsp_scheduler), plot_dot_graphs_(false) {}
    virtual ~IsomorphicSubgraphScheduler() {}

    void set_symmetry(size_t symmetry) {
        symmetry_ = symmetry;
    }

    void set_plot_dot_graphs(bool plot) {
        plot_dot_graphs_ = plot;
    }

    std::vector<vertex_idx_t<Graph_t>> compute_partition(const BspInstance<Graph_t>& instance) {
        WavefrontOrbitProcessor<Graph_t> wavefront(symmetry_);
        wavefront.discover_isomorphic_groups(instance.getComputationalDag());
        auto isomorphic_groups = wavefront.get_isomorphic_groups();
        auto finalized_subgraphs = wavefront.get_finalized_subgraphs();

        if (plot_dot_graphs_) {
            DotFileWriter writer;
            writer.write_colored_graph("isomorphic_groups.dot", instance.getComputationalDag(), wavefront.get_vertex_color_map());
        }

        const unsigned min_proc_type_count = instance.getArchitecture().getMinProcessorTypeCount();
        trim_subgraph_groups(finalized_subgraphs, isomorphic_groups, min_proc_type_count);

        subgrah_scheduler_input<Graph_t> input;
        input.prepare_subgraph_scheduling_input(instance, finalized_subgraphs, isomorphic_groups);

        if (plot_dot_graphs_) {
            DotFileWriter writer;
            writer.write_graph("iso_groups_contracted.dot", input.instance.getComputationalDag());
        }

        EftSubgraphScheduler<Graph_t> etf_scheduler;
        SubgraphSchedule subgraph_schedule = etf_scheduler.run(input.instance, input.multiplicities, input.required_proc_types);

        std::vector<vertex_idx_t<Graph_t>> partition(instance.numberOfVertices(), 0);

        schedule_isomorphic_group(instance, finalized_subgraphs, isomorphic_groups, subgraph_schedule, partition);

        if (plot_dot_graphs_) {
            DotFileWriter writer;
            writer.write_colored_graph("graph_partition.dot", instance.getComputationalDag(), partition);
        }

        return partition;
    }

    private:

    void trim_subgraph_groups(std::vector<subgraph<Graph_t>>& finalized_subgraphs,
                              std::vector<std::vector<unsigned>>& isomorphic_groups,
                              const unsigned min_proc_type_count) {
        std::vector<std::vector<unsigned>> new_isomorphic_groups;
        std::vector<subgraph<Graph_t>> new_finalized_subgraphs;

        for (size_t i = 0; i < isomorphic_groups.size(); ++i) {
            auto& sgs = isomorphic_groups[i];
            const unsigned group_size = static_cast<unsigned>(sgs.size());
            if (group_size == 0)
                continue;
            const unsigned gcd = std::gcd(group_size, min_proc_type_count);

            if (gcd == group_size) {
                std::vector<unsigned> new_indices;
                for (unsigned old_idx : sgs) {
                    new_indices.push_back(static_cast<unsigned>(new_finalized_subgraphs.size()));
                    new_finalized_subgraphs.push_back(finalized_subgraphs[old_idx]);
                }
                new_isomorphic_groups.push_back(new_indices);
            } else {
                const unsigned merge_size = group_size / gcd;

                const size_t original_hash = finalized_subgraphs[sgs[0]].current_hash;
                size_t new_merged_hash = 0;
                for (unsigned k = 0; k < merge_size; ++k) {
                    hash_combine(new_merged_hash, original_hash);
                }

                std::vector<unsigned> new_group_indices;
                size_t original_sg_cursor = 0;

                for (unsigned j = 0; j < gcd; ++j) {
                    const auto& first_sg_to_merge = finalized_subgraphs[sgs[original_sg_cursor]];
                    subgraph<Graph_t> merged_sg = first_sg_to_merge;
                    original_sg_cursor++;

                    for (unsigned k = 1; k < merge_size; ++k) {
                        const auto& sg_to_merge = finalized_subgraphs[sgs[original_sg_cursor]];
                        original_sg_cursor++;

                        merged_sg.vertices.insert(merged_sg.vertices.end(), sg_to_merge.vertices.begin(),
                                                  sg_to_merge.vertices.end());
                        merged_sg.work_weight += sg_to_merge.work_weight;
                        merged_sg.memory_weight += sg_to_merge.memory_weight;
                        merged_sg.start_wavefront = std::min(merged_sg.start_wavefront, sg_to_merge.start_wavefront);
                        merged_sg.end_wavefront = std::max(merged_sg.end_wavefront, sg_to_merge.end_wavefront);
                    }

                    merged_sg.current_hash = new_merged_hash;

                    new_group_indices.push_back(static_cast<unsigned>(new_finalized_subgraphs.size()));
                    new_finalized_subgraphs.push_back(std::move(merged_sg));
                }
                new_isomorphic_groups.push_back(new_group_indices);
            }
       }

        finalized_subgraphs = std::move(new_finalized_subgraphs);
        isomorphic_groups = std::move(new_isomorphic_groups);
    }

    
    void schedule_isomorphic_group(const BspInstance<Graph_t>& instance, const std::vector<subgraph<Graph_t>> & finalized_subgraphs, const std::vector<std::vector<unsigned>> & isomorphic_groups, const SubgraphSchedule & sub_sched, std::vector<vertex_idx_t<Graph_t>> & partition) {
        vertex_idx_t<Graph_t> current_partition_idx = 0;

        for (size_t grou_idx = 0; grou_idx < isomorphic_groups.size(); ++grou_idx) {
            auto & sgs = isomorphic_groups[grou_idx];
            if (sgs.empty()) {
                continue;
            }

            // --- Schedule Representative and Create Pattern ---
            const auto & rep_finalized_subgraph = finalized_subgraphs[sgs[0]];
            BspInstance<Constr_Graph_t> representative_instance;

            std::vector<vertex_idx_t<Graph_t>> vertices_local = rep_finalized_subgraph.vertices;
            std::sort(vertices_local.begin(), vertices_local.end());

            create_induced_subgraph(instance.getComputationalDag(), representative_instance.getComputationalDag(), vertices_local);

            representative_instance.setArchitecture(instance.getArchitecture());
            std::vector<v_memw_t<Graph_t>> dummy_mem_weights(sub_sched.node_assigned_worker_per_type[grou_idx].size(), 0);
            for (unsigned proc_type = 0; proc_type < sub_sched.node_assigned_worker_per_type[grou_idx].size(); ++proc_type)
                dummy_mem_weights[proc_type] = instance.getArchitecture().maxMemoryBoundProcType(proc_type);
            representative_instance.getArchitecture().set_processors_consequ_types(sub_sched.node_assigned_worker_per_type[grou_idx], dummy_mem_weights);
            representative_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

            BspSchedule<Constr_Graph_t> bsp_schedule(representative_instance);

            if constexpr (verbose) {
                std::cout << "--- Scheduling representative for group " << grou_idx << " ---" << std::endl;
                std::cout << "  Number of subgraphs in group: " << sgs.size() << std::endl;
                std::cout << "  Representative subgraph size: " << vertices_local.size() << " vertices" << std::endl;
                const auto& sub_arch = representative_instance.getArchitecture();
                std::cout << "  Sub-architecture for scheduling:" << std::endl;
                std::cout << "    Processors: " << sub_arch.numberOfProcessors() << std::endl;
                std::cout << "    Processor types counts: ";
                const auto& type_counts = sub_arch.getProcessorTypeCount();
                for (size_t type_idx = 0; type_idx < type_counts.size(); ++type_idx) {
                    std::cout << "T" << type_idx << ":" << type_counts[type_idx] << " ";
                }
                std::cout << std::endl;
                std::cout << "    Sync cost: " << sub_arch.synchronisationCosts() << ", Comm cost: " << sub_arch.communicationCosts() << std::endl;
            }
            bsp_scheduler_->computeSchedule(bsp_schedule);
            SetSchedule<Constr_Graph_t> set_schedule(bsp_schedule);

            // --- Build Pattern Map ---
            std::map<std::pair<unsigned, unsigned>, vertex_idx_t<Graph_t>> sp_proc_to_relative_partition;
            vertex_idx_t<Graph_t> num_partitions_per_subgraph = 0;
            for (unsigned s = 0; s < set_schedule.step_processor_vertices.size(); ++s) {
                const auto& procs = set_schedule.step_processor_vertices[s];
                for (unsigned p = 0; p < procs.size(); ++p) {
                    if (!procs[p].empty()) {
                        sp_proc_to_relative_partition[{s, p}] = num_partitions_per_subgraph;
                        num_partitions_per_subgraph++;
                    }
                }
            }

            // --- Replicate Pattern for ALL Subgraphs in the Group ---
            for (size_t i = 0; i < sgs.size(); ++i) {
                const auto & current_finalized_subgraph = finalized_subgraphs[sgs[i]];
                std::vector<vertex_idx_t<Graph_t>> sg_vertices_local = current_finalized_subgraph.vertices;
                std::sort(sg_vertices_local.begin(), sg_vertices_local.end());

                for (size_t j = 0; j < sg_vertices_local.size(); ++j) {
                    vertex_idx_t<Graph_t> relative_partition = sp_proc_to_relative_partition.at({bsp_schedule.assignedSuperstep(j), bsp_schedule.assignedProcessor(j)});
                    partition[sg_vertices_local[j]] = current_partition_idx + relative_partition;
                }
                current_partition_idx += num_partitions_per_subgraph;
            }
        }
    }





























};

}