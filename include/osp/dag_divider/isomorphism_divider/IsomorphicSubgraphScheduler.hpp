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
#include "OrbitGraphProcessor.hpp"
#include "EftSubgraphScheduler.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp {

template<typename Graph_t, typename Constr_Graph_t>
class IsomorphicSubgraphScheduler {
    static_assert(is_computational_dag_v<Graph_t>, "Graph must be a computational DAG");
    static_assert(is_computational_dag_v<Constr_Graph_t>, "Constr_Graph_t must be a computational DAG");
    static_assert(is_constructable_cdag_v<Constr_Graph_t>,
                  "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>>,
                  "Graph_t and Constr_Graph_t must have the same vertex_idx types");

    private:

    static constexpr bool verbose = true;
    
    size_t symmetry_ = 2;
    Scheduler<Constr_Graph_t> * bsp_scheduler_;

    bool plot_dot_graphs_ = false;

    public:

    IsomorphicSubgraphScheduler(Scheduler<Constr_Graph_t> & bsp_scheduler) : symmetry_(2), bsp_scheduler_(&bsp_scheduler), plot_dot_graphs_(false) {}
    virtual ~IsomorphicSubgraphScheduler() {}

    void set_symmetry(size_t symmetry) {
        symmetry_ = symmetry;
    }

    void set_plot_dot_graphs(bool plot) {
        plot_dot_graphs_ = plot;
    }

    std::vector<vertex_idx_t<Graph_t>> compute_partition(const BspInstance<Graph_t>& instance) {
        OrbitGraphProcessor<Graph_t, Constr_Graph_t> orbit_processor(symmetry_);
        orbit_processor.discover_isomorphic_groups(instance.getComputationalDag());
        auto isomorphic_groups = orbit_processor.get_final_groups();
        
        if (plot_dot_graphs_) {
            DotFileWriter writer;
            writer.write_colored_graph("isomorphic_groups.dot", instance.getComputationalDag(), orbit_processor.get_final_contraction_map());
        }

        const unsigned min_proc_type_count = instance.getArchitecture().getMinProcessorTypeCount();
        trim_subgraph_groups(isomorphic_groups, min_proc_type_count);

        auto input = prepare_subgraph_scheduling_input(instance, isomorphic_groups);

        if (plot_dot_graphs_) {
            DotFileWriter writer;
            writer.write_graph("iso_groups_contracted.dot", input.instance.getComputationalDag());
        }

        EftSubgraphScheduler<Constr_Graph_t> etf_scheduler;
        SubgraphSchedule subgraph_schedule = etf_scheduler.run(input.instance, input.multiplicities, input.required_proc_types);

        std::vector<vertex_idx_t<Graph_t>> partition(instance.numberOfVertices(), 0);

        schedule_isomorphic_group(instance, isomorphic_groups, subgraph_schedule, partition);

        if (plot_dot_graphs_) {
            DotFileWriter writer;
            writer.write_colored_graph("graph_partition.dot", instance.getComputationalDag(), partition);
        }

        return partition;
    }

    protected:

    template<typename G_t, typename C_G_t>
    struct subgraph_scheduler_input {
        BspInstance<C_G_t> instance;
        std::vector<unsigned> multiplicities;
        std::vector<std::vector<v_workw_t<G_t>>> required_proc_types;
    };

    void trim_subgraph_groups(std::vector<typename OrbitGraphProcessor<Graph_t, Constr_Graph_t>::Group>& isomorphic_groups,
                              const unsigned min_proc_type_count) {
        if (min_proc_type_count <= 1) return;

        for (auto& group : isomorphic_groups) {
            const unsigned group_size = static_cast<unsigned>(group.subgraphs.size());
            if (group_size == 0)
                continue;
            const unsigned gcd = std::gcd(group_size, min_proc_type_count);

            if (gcd < group_size) {
                const unsigned merge_size = group_size / gcd;
                std::vector<std::vector<vertex_idx_t<Graph_t>>> new_subgraphs;
                new_subgraphs.reserve(gcd);

                size_t original_sg_cursor = 0;

                for (unsigned j = 0; j < gcd; ++j) {
                    std::vector<vertex_idx_t<Graph_t>> merged_sg_vertices = group.subgraphs[original_sg_cursor];
                    original_sg_cursor++;

                    for (unsigned k = 1; k < merge_size; ++k) {
                        const auto& sg_to_merge_vertices = group.subgraphs[original_sg_cursor];
                        original_sg_cursor++;
                        merged_sg_vertices.insert(merged_sg_vertices.end(), sg_to_merge_vertices.begin(), sg_to_merge_vertices.end());
                    }
                    new_subgraphs.push_back(std::move(merged_sg_vertices));
                }
                group.subgraphs = std::move(new_subgraphs);
            }
       }
    }

    subgraph_scheduler_input<Graph_t, Constr_Graph_t> prepare_subgraph_scheduling_input(
        const BspInstance<Graph_t>& original_instance,
        const std::vector<typename OrbitGraphProcessor<Graph_t, Constr_Graph_t>::Group>& isomorphic_groups) {
        
        subgraph_scheduler_input<Graph_t, Constr_Graph_t> result;
        result.instance.setArchitecture(original_instance.getArchitecture());
        const unsigned num_proc_types = original_instance.getArchitecture().getNumberOfProcessorTypes();

        result.multiplicities.resize(isomorphic_groups.size());
        result.required_proc_types.resize(isomorphic_groups.size());
        std::vector<vertex_idx_t<Constr_Graph_t>> contraction_map(original_instance.numberOfVertices());

        size_t coarse_node_idx = 0;
        for (const auto &group : isomorphic_groups) {
            result.multiplicities[coarse_node_idx] = static_cast<unsigned>(group.subgraphs.size());
            result.required_proc_types[coarse_node_idx].assign(num_proc_types, 0);

            for (const auto &subgraph : group.subgraphs) {
                for (const auto &vertex : subgraph) {
                    contraction_map[vertex] = static_cast<vertex_idx_t<Constr_Graph_t>>(coarse_node_idx);
                    const auto vertex_work = original_instance.getComputationalDag().vertex_work_weight(vertex);
                    const auto vertex_type = original_instance.getComputationalDag().vertex_type(vertex);
                    for (unsigned j = 0; j < num_proc_types; ++j) {
                        if (original_instance.isCompatibleType(vertex_type, j)) {
                            result.required_proc_types[coarse_node_idx][j] += vertex_work;
                        }
                    }
                }
            }
            ++coarse_node_idx;
        }
        coarser_util::construct_coarse_dag(original_instance.getComputationalDag(), result.instance.getComputationalDag(),
                                        contraction_map);

        if constexpr (verbose) {
            std::cout << "\n--- Preparing Subgraph Scheduling Input ---\n";
            std::cout << "Found " << isomorphic_groups.size() << " isomorphic groups to schedule as coarse nodes.\n";
            for (size_t j = 0; j < isomorphic_groups.size(); ++j) {
                std::cout << "  - Coarse Node " << j << " (from " << isomorphic_groups[j].subgraphs.size()
                        << " isomorphic subgraphs):\n";
                std::cout << "    - Multiplicity for scheduling: " << result.multiplicities[j] << "\n";
                std::cout << "    - Total Work (in coarse graph): " << result.instance.getComputationalDag().vertex_work_weight(j) << "\n";
            }
        }
        return result;
    }

    void schedule_isomorphic_group(const BspInstance<Graph_t>& instance, 
                                   const std::vector<typename OrbitGraphProcessor<Graph_t, Constr_Graph_t>::Group>& isomorphic_groups, 
                                   const SubgraphSchedule & sub_sched, 
                                   std::vector<vertex_idx_t<Graph_t>> & partition) {
        vertex_idx_t<Graph_t> current_partition_idx = 0;

        for (size_t grou_idx = 0; grou_idx < isomorphic_groups.size(); ++grou_idx) {
            const auto& group = isomorphic_groups[grou_idx];
            if (group.subgraphs.empty()) {
                continue;
            }

            // --- Schedule Representative and Create Pattern ---
            auto rep_subgraph_vertices_sorted = group.subgraphs[0];
            std::sort(rep_subgraph_vertices_sorted.begin(), rep_subgraph_vertices_sorted.end());

            BspInstance<Constr_Graph_t> representative_instance;

            create_induced_subgraph(instance.getComputationalDag(), representative_instance.getComputationalDag(), rep_subgraph_vertices_sorted);
            
            // Create a map from original vertex ID to its local index in the induced subgraph
            std::unordered_map<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>> vertex_to_local_idx;
            for (size_t j = 0; j < rep_subgraph_vertices_sorted.size(); ++j) {
                vertex_to_local_idx[rep_subgraph_vertices_sorted[j]] = static_cast<vertex_idx_t<Constr_Graph_t>>(j);
            }

            representative_instance.setArchitecture(instance.getArchitecture());
            std::vector<v_memw_t<Constr_Graph_t>> dummy_mem_weights(sub_sched.node_assigned_worker_per_type[grou_idx].size(), 0);
            for (unsigned proc_type = 0; proc_type < sub_sched.node_assigned_worker_per_type[grou_idx].size(); ++proc_type) {
                dummy_mem_weights[proc_type] = static_cast<v_memw_t<Constr_Graph_t>>(instance.getArchitecture().maxMemoryBoundProcType(proc_type));
            }
            representative_instance.getArchitecture().set_processors_consequ_types(sub_sched.node_assigned_worker_per_type[grou_idx], dummy_mem_weights);
            representative_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

            BspSchedule<Constr_Graph_t> bsp_schedule(representative_instance);

            if constexpr (verbose) {
                std::cout << "--- Scheduling representative for group " << grou_idx << " ---" << std::endl;
                std::cout << "  Number of subgraphs in group: " << group.subgraphs.size() << std::endl;
                std::cout << "  Representative subgraph size: " << rep_subgraph_vertices_sorted.size() << " vertices" << std::endl;
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
            for (size_t i = 0; i < group.subgraphs.size(); ++i) {
                auto current_subgraph_vertices_sorted = group.subgraphs[i];
                std::sort(current_subgraph_vertices_sorted.begin(), current_subgraph_vertices_sorted.end());

                for (size_t j = 0; j < current_subgraph_vertices_sorted.size(); ++j) {
                    vertex_idx_t<Graph_t> original_rep_vertex = rep_subgraph_vertices_sorted[j];
                    vertex_idx_t<Constr_Graph_t> local_idx = vertex_to_local_idx.at(original_rep_vertex);
                    vertex_idx_t<Graph_t> relative_partition = sp_proc_to_relative_partition.at({bsp_schedule.assignedSuperstep(local_idx), bsp_schedule.assignedProcessor(local_idx)});
                    partition[current_subgraph_vertices_sorted[j]] = current_partition_idx + relative_partition;
                }
                current_partition_idx += num_partitions_per_subgraph;
            }
        }
    }





























};

}