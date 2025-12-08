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

#include "EftSubgraphScheduler.hpp"
#include "HashComputer.hpp"
#include "MerkleHashComputer.hpp"
#include "OrbitGraphProcessor.hpp"
#include "TrimmedGroupScheduler.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

namespace osp {

/**
 * @brief A scheduler that leverages isomorphic subgraphs to partition a DAG.
 *
 * @class IsomorphicSubgraphScheduler
 *
 * This scheduler first identifies isomorphic subgraphs within the input DAG using a hash-based approach.
 * It then groups these isomorphic subgraphs into "orbits". Each orbit is treated as a single node in a
 * coarser graph. The scheduler then uses an ETF-like approach to schedule these coarse nodes (orbits)
 * onto available processors. Finally, the schedule for each orbit is "unrolled" back to the original
 * DAG, assigning a partition ID to each original vertex.
 *
 * The scheduler supports trimming of isomorphic groups to better fit processor counts, and can
 * dynamically switch between a standard BSP scheduler and a specialized TrimmedGroupScheduler
 * for these trimmed groups.
 *
 * @tparam Graph_t The type of the input computational DAG.
 * @tparam Constr_Graph_t The type of the constructable computational DAG used for internal representations.
 */
template<typename Graph_t, typename Constr_Graph_t>
class IsomorphicSubgraphScheduler {
    static_assert(is_computational_dag_v<Graph_t>, "Graph must be a computational DAG");
    static_assert(is_computational_dag_v<Constr_Graph_t>, "Constr_Graph_t must be a computational DAG");
    static_assert(is_constructable_cdag_v<Constr_Graph_t>,
                  "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>>,
                  "Graph_t and Constr_Graph_t must have the same vertex_idx types");

  private:
    static constexpr bool verbose = false;
    const HashComputer<vertex_idx_t<Graph_t>> *hash_computer_;
    size_t symmetry_ = 4;
    Scheduler<Constr_Graph_t> *bsp_scheduler_;
    bool use_max_group_size_ = false;
    unsigned max_group_size_ = 0;
    bool plot_dot_graphs_ = false;
    v_workw_t<Constr_Graph_t> work_threshold_ = 10;
    v_workw_t<Constr_Graph_t> critical_path_threshold_ = 10;
    double orbit_lock_ratio_ = 0.4;
    double natural_breaks_count_percentage_ = 0.1;
    bool merge_different_node_types = true;
    bool allow_use_trimmed_scheduler = true;
    bool use_max_bsp = false;
    bool use_adaptive_symmetry_threshold = true;

  public:
    explicit IsomorphicSubgraphScheduler(Scheduler<Constr_Graph_t> &bsp_scheduler)
        : hash_computer_(nullptr), bsp_scheduler_(&bsp_scheduler), plot_dot_graphs_(false) {}

    IsomorphicSubgraphScheduler(Scheduler<Constr_Graph_t> &bsp_scheduler, const HashComputer<vertex_idx_t<Graph_t>> &hash_computer)
        : hash_computer_(&hash_computer), bsp_scheduler_(&bsp_scheduler), plot_dot_graphs_(false) {}

    virtual ~IsomorphicSubgraphScheduler() {}

    void setMergeDifferentTypes(bool flag) { merge_different_node_types = flag; }
    void setWorkThreshold(v_workw_t<Constr_Graph_t> work_threshold) { work_threshold_ = work_threshold; }
    void setCriticalPathThreshold(v_workw_t<Constr_Graph_t> critical_path_threshold) { critical_path_threshold_ = critical_path_threshold; }
    void setOrbitLockRatio(double orbit_lock_ratio) { orbit_lock_ratio_ = orbit_lock_ratio; }
    void setNaturalBreaksCountPercentage(double natural_breaks_count_percentage) { natural_breaks_count_percentage_ = natural_breaks_count_percentage; }
    void setAllowTrimmedScheduler(bool flag) { allow_use_trimmed_scheduler = flag; }
    void set_plot_dot_graphs(bool plot) { plot_dot_graphs_ = plot; }
    void disable_use_max_group_size() { use_max_group_size_ = false; }
    void setUseMaxBsp(bool flag) { use_max_bsp = flag; }
    void enable_use_max_group_size(const unsigned max_group_size) {
        use_max_group_size_ = true;
        max_group_size_ = max_group_size;
    }
    void setEnableAdaptiveSymmetryThreshold() { use_adaptive_symmetry_threshold = true; }
    void setUseStaticSymmetryLevel(size_t static_symmetry_level) {
        use_adaptive_symmetry_threshold = false;
        symmetry_ = static_symmetry_level;
    }

    std::vector<vertex_idx_t<Graph_t>> compute_partition(const BspInstance<Graph_t> &instance) {
        OrbitGraphProcessor<Graph_t, Constr_Graph_t> orbit_processor;
        orbit_processor.set_work_threshold(work_threshold_);
        orbit_processor.setMergeDifferentNodeTypes(merge_different_node_types);
        orbit_processor.setCriticalPathThreshold(critical_path_threshold_);
        orbit_processor.setLockRatio(orbit_lock_ratio_);
        orbit_processor.setNaturalBreaksCountPercentage(natural_breaks_count_percentage_);
        if (not use_adaptive_symmetry_threshold) {
            orbit_processor.setUseStaticSymmetryLevel(symmetry_);
        }

        std::unique_ptr<HashComputer<vertex_idx_t<Graph_t>>> local_hasher;
        if (!hash_computer_) {
            local_hasher = std::make_unique<MerkleHashComputer<Graph_t, bwd_merkle_node_hash_func<Graph_t>, true>>(instance.getComputationalDag(), instance.getComputationalDag());
            hash_computer_ = local_hasher.get();
        }

        orbit_processor.discover_isomorphic_groups(instance.getComputationalDag(), *hash_computer_);

        auto isomorphic_groups = orbit_processor.get_final_groups();

        std::vector<bool> was_trimmed(isomorphic_groups.size(), false);
        trim_subgraph_groups(isomorphic_groups, instance, was_trimmed); // Apply trimming and record which groups were affected

        auto input = prepare_subgraph_scheduling_input(instance, isomorphic_groups, was_trimmed);

        EftSubgraphScheduler<Constr_Graph_t> etf_scheduler;
        SubgraphSchedule subgraph_schedule = etf_scheduler.run(input.instance, input.multiplicities, input.required_proc_types, input.max_num_processors);
        subgraph_schedule.was_trimmed = std::move(was_trimmed); // Pass through trimming info

        std::vector<vertex_idx_t<Graph_t>> partition(instance.numberOfVertices(), 0);
        schedule_isomorphic_group(instance, isomorphic_groups, subgraph_schedule, partition);

        if (plot_dot_graphs_) {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
            std::string timestamp = ss.str() + "_";

            DotFileWriter writer;
            writer.write_colored_graph(timestamp + "isomorphic_groups.dot", instance.getComputationalDag(), orbit_processor.get_final_contraction_map());
            writer.write_colored_graph(timestamp + "orbits_colored.dot", instance.getComputationalDag(), orbit_processor.get_contraction_map());
            writer.write_graph(timestamp + "iso_groups_contracted.dot", input.instance.getComputationalDag());
            writer.write_colored_graph(timestamp + "graph_partition.dot", instance.getComputationalDag(), partition);
            Constr_Graph_t corase_graph;
            coarser_util::construct_coarse_dag(instance.getComputationalDag(), corase_graph, partition);
            writer.write_graph(timestamp + "block_graph.dot", corase_graph);
        }
        return partition;
    }

  protected:
    template<typename G_t, typename C_G_t>
    struct subgraph_scheduler_input {
        BspInstance<C_G_t> instance;
        std::vector<unsigned> multiplicities;
        std::vector<unsigned> max_num_processors;
        std::vector<std::vector<v_workw_t<G_t>>> required_proc_types;
    };

    void trim_subgraph_groups(std::vector<typename OrbitGraphProcessor<Graph_t, Constr_Graph_t>::Group> &isomorphic_groups,
                              const BspInstance<Graph_t> &instance,
                              std::vector<bool> &was_trimmed) {
        if constexpr (verbose) {
            std::cout << "\n--- Trimming Isomorphic Subgraph Groups ---" << std::endl;
        }
        for (size_t group_idx = 0; group_idx < isomorphic_groups.size(); ++group_idx) {
            auto &group = isomorphic_groups[group_idx];
            const unsigned group_size = static_cast<unsigned>(group.size());
            if (group_size <= 1)
                continue;

            unsigned effective_min_proc_type_count = 0;

            if (use_max_group_size_) {
                if constexpr (verbose) {
                    std::cout << "Group " << group_idx << " (size " << group_size << "): Using fixed max_group_size_ = " << max_group_size_ << " for trimming." << std::endl;
                }
                effective_min_proc_type_count = max_group_size_;
            } else {
                // Determine if the group consists of a single node type
                bool is_single_type_group = true;
                v_type_t<Graph_t> common_node_type = 0;

                if constexpr (has_typed_vertices_v<Graph_t>) {
                    if (!group.subgraphs.empty() && !group.subgraphs[0].empty()) {
                        common_node_type = instance.getComputationalDag().vertex_type(group.subgraphs[0][0]);
                        const auto &rep_subgraph = group.subgraphs[0];
                        for (const auto &vertex : rep_subgraph) {
                            if (instance.getComputationalDag().vertex_type(vertex) != common_node_type) {
                                is_single_type_group = false;
                                break;
                            }
                        }
                    } else {
                        is_single_type_group = false;
                    }
                } else {
                    is_single_type_group = false;
                }

                if (is_single_type_group) {
                    // Dynamically determine min_proc_type_count based on compatible processors for this type
                    unsigned min_compatible_processors = std::numeric_limits<unsigned>::max();
                    const auto &proc_type_counts = instance.getArchitecture().getProcessorTypeCount();

                    bool found_compatible_processor = false;
                    for (unsigned proc_type_idx = 0; proc_type_idx < proc_type_counts.size(); ++proc_type_idx) {
                        if (instance.isCompatibleType(common_node_type, proc_type_idx)) {
                            min_compatible_processors = std::min(min_compatible_processors, proc_type_counts[proc_type_idx]);
                            found_compatible_processor = true;
                        }
                    }
                    if (found_compatible_processor) {
                        if constexpr (verbose) {
                            std::cout << "Group " << group_idx << " (size " << group_size << "): Single node type (" << common_node_type
                                      << "). Min compatible processors: " << min_compatible_processors << "." << std::endl;
                        }
                        effective_min_proc_type_count = min_compatible_processors;
                    } else {
                        if constexpr (verbose) {
                            std::cout << "Group " << group_idx << " (size " << group_size << "): Single node type (" << common_node_type
                                      << ") but no compatible processors found. Disabling trimming." << std::endl;
                        }
                        // If no compatible processors found for this type, effectively disable trimming for this group.
                        effective_min_proc_type_count = 1;
                    }
                } else {
                    // Fallback to a default min_proc_type_count if not a single-type group or no typed vertices.
                    const auto &type_count = instance.getArchitecture().getProcessorTypeCount();
                    if (type_count.empty()) {
                        effective_min_proc_type_count = 0;
                    }
                    effective_min_proc_type_count = *std::min_element(type_count.begin(), type_count.end());
                    if constexpr (verbose) {
                        std::cout << "Group " << group_idx << " (size " << group_size << "): Multi-type or untyped group. Using default min_proc_type_count: " << effective_min_proc_type_count << "." << std::endl;
                    }
                }
            }

            // Ensure effective_min_proc_type_count is at least 1 for valid GCD calculation.
            if (effective_min_proc_type_count == 0) {
                effective_min_proc_type_count = 1;
            }

            // If effective_min_proc_type_count is 1, no trimming is needed as gcd(X, 1) = 1.
            if (effective_min_proc_type_count <= 1) {
                continue;
            }

            unsigned gcd = std::gcd(group_size, effective_min_proc_type_count);

            if (gcd < group_size) {
                if constexpr (verbose) {
                    std::cout << "  -> Trimming group " << group_idx << ". GCD(" << group_size << ", " << effective_min_proc_type_count
                              << ") = " << gcd << ". Merging " << group_size / gcd << " subgraphs at a time." << std::endl;
                }

                if (allow_use_trimmed_scheduler)
                    gcd = 1;

                was_trimmed[group_idx] = true;
                const unsigned merge_size = group_size / gcd;
                std::vector<std::vector<vertex_idx_t<Graph_t>>> new_subgraphs;
                new_subgraphs.reserve(gcd);

                size_t original_sg_cursor = 0;

                for (unsigned j = 0; j < gcd; ++j) {
                    std::vector<vertex_idx_t<Graph_t>> merged_sg_vertices;
                    // Estimate capacity for efficiency. Assuming subgraphs have similar sizes.
                    if (!group.subgraphs.empty()) {
                        merged_sg_vertices.reserve(group.subgraphs[0].size() * merge_size);
                    }

                    for (unsigned k = 0; k < merge_size; ++k) {
                        const auto &sg_to_merge_vertices = group.subgraphs[original_sg_cursor];
                        original_sg_cursor++;
                        merged_sg_vertices.insert(merged_sg_vertices.end(), sg_to_merge_vertices.begin(), sg_to_merge_vertices.end());
                    }
                    new_subgraphs.push_back(std::move(merged_sg_vertices));
                }
                group.subgraphs = std::move(new_subgraphs);
            } else {
                if constexpr (verbose) {
                    std::cout << "  -> No trim needed for group " << group_idx << "." << std::endl;
                }
                was_trimmed[group_idx] = false;
            }
        }
    }

    subgraph_scheduler_input<Graph_t, Constr_Graph_t> prepare_subgraph_scheduling_input(
        const BspInstance<Graph_t> &original_instance,
        const std::vector<typename OrbitGraphProcessor<Graph_t, Constr_Graph_t>::Group> &isomorphic_groups,
        const std::vector<bool> &was_trimmed) {

        subgraph_scheduler_input<Graph_t, Constr_Graph_t> result;
        result.instance.setArchitecture(original_instance.getArchitecture());
        const unsigned num_proc_types = original_instance.getArchitecture().getNumberOfProcessorTypes();

        result.multiplicities.resize(isomorphic_groups.size());
        result.max_num_processors.resize(isomorphic_groups.size());
        result.required_proc_types.resize(isomorphic_groups.size());
        std::vector<vertex_idx_t<Constr_Graph_t>> contraction_map(original_instance.numberOfVertices());

        size_t coarse_node_idx = 0;
        for (const auto &group : isomorphic_groups) {

            result.max_num_processors[coarse_node_idx] = static_cast<unsigned>(group.size() * group.subgraphs[0].size());
            result.multiplicities[coarse_node_idx] = (was_trimmed[coarse_node_idx] && allow_use_trimmed_scheduler) ? 1 : static_cast<unsigned>(group.subgraphs.size());
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
                std::cout << "    - Required Processor Types: ";
                for (unsigned k = 0; k < num_proc_types; ++k) {
                    std::cout << result.required_proc_types[j][k] << " ";
                }
                std::cout << "\n";
                std::cout << "    - Max number of processors: " << result.max_num_processors[j] << "\n";
            }
        }
        return result;
    }

    void schedule_isomorphic_group(const BspInstance<Graph_t> &instance,
                                   const std::vector<typename OrbitGraphProcessor<Graph_t, Constr_Graph_t>::Group> &isomorphic_groups,
                                   const SubgraphSchedule &sub_sched,
                                   std::vector<vertex_idx_t<Graph_t>> &partition) {
        vertex_idx_t<Graph_t> current_partition_idx = 0;

        for (size_t group_idx = 0; group_idx < isomorphic_groups.size(); ++group_idx) {
            const auto &group = isomorphic_groups[group_idx];
            if (group.subgraphs.empty()) {
                continue;
            }

            // Schedule the Representative Subgraph to get a BSP schedule pattern ---
            auto rep_subgraph_vertices_sorted = group.subgraphs[0];
            std::sort(rep_subgraph_vertices_sorted.begin(), rep_subgraph_vertices_sorted.end());

            BspInstance<Constr_Graph_t> representative_instance;
            auto rep_global_to_local_map = create_induced_subgraph_map(instance.getComputationalDag(), representative_instance.getComputationalDag(), rep_subgraph_vertices_sorted);

            representative_instance.setArchitecture(instance.getArchitecture());
            const auto &procs_for_group = sub_sched.node_assigned_worker_per_type[group_idx];
            std::vector<v_memw_t<Constr_Graph_t>> mem_weights(procs_for_group.size(), 0);
            for (unsigned proc_type = 0; proc_type < procs_for_group.size(); ++proc_type) {
                mem_weights[proc_type] = static_cast<v_memw_t<Constr_Graph_t>>(instance.getArchitecture().maxMemoryBoundProcType(proc_type));
            }
            representative_instance.getArchitecture().SetProcessorsConsequTypes(procs_for_group, mem_weights);
            representative_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

            // --- Decide which scheduler to use ---
            unsigned min_non_zero_procs = std::numeric_limits<unsigned>::max();
            for (const auto &proc_count : procs_for_group) {
                if (proc_count > 0) {
                    min_non_zero_procs = std::min(min_non_zero_procs, proc_count);
                }
            }

            bool use_trimmed_scheduler = sub_sched.was_trimmed[group_idx] && min_non_zero_procs > 1 && allow_use_trimmed_scheduler;

            Scheduler<Constr_Graph_t> *scheduler_for_group_ptr;
            std::unique_ptr<Scheduler<Constr_Graph_t>> trimmed_scheduler_owner;
            if (use_trimmed_scheduler) {
                if constexpr (verbose)
                    std::cout << "Using TrimmedGroupScheduler for group " << group_idx << std::endl;
                trimmed_scheduler_owner = std::make_unique<TrimmedGroupScheduler<Constr_Graph_t>>(*bsp_scheduler_, min_non_zero_procs);
                scheduler_for_group_ptr = trimmed_scheduler_owner.get();
            } else {
                if constexpr (verbose)
                    std::cout << "Using standard BSP scheduler for group " << group_idx << std::endl;
                scheduler_for_group_ptr = bsp_scheduler_;
            }

            // --- Schedule the representative to get the pattern ---
            BspSchedule<Constr_Graph_t> bsp_schedule(representative_instance);

            if constexpr (verbose) {
                std::cout << "--- Scheduling representative for group " << group_idx << " ---" << std::endl;
                std::cout << "  Number of subgraphs in group: " << group.subgraphs.size() << std::endl;
                const auto &rep_dag = representative_instance.getComputationalDag();
                std::cout << "  Representative subgraph size: " << rep_dag.num_vertices() << " vertices" << std::endl;
                std::vector<unsigned> node_type_counts(rep_dag.num_vertex_types(), 0);
                for (const auto &v : rep_dag.vertices()) {
                    node_type_counts[rep_dag.vertex_type(v)]++;
                }
                std::cout << "    Node type counts: ";
                for (size_t type_idx = 0; type_idx < node_type_counts.size(); ++type_idx) {
                    if (node_type_counts[type_idx] > 0) {
                        std::cout << "T" << type_idx << ":" << node_type_counts[type_idx] << " ";
                    }
                }
                std::cout << std::endl;

                const auto &sub_arch = representative_instance.getArchitecture();
                std::cout << "  Sub-architecture for scheduling:" << std::endl;
                std::cout << "    Processors: " << sub_arch.numberOfProcessors() << std::endl;
                std::cout << "    Processor types counts: ";
                const auto &type_counts = sub_arch.getProcessorTypeCount();
                for (size_t type_idx = 0; type_idx < type_counts.size(); ++type_idx) {
                    std::cout << "T" << type_idx << ":" << type_counts[type_idx] << " ";
                }
                std::cout << std::endl;
                std::cout << "    Sync cost: " << sub_arch.synchronisationCosts() << ", Comm cost: " << sub_arch.communicationCosts() << std::endl;
            }

            scheduler_for_group_ptr->computeSchedule(bsp_schedule);

            if constexpr (verbose) {
                std::cout << "  Schedule satisfies precedence constraints: ";
                std::cout << bsp_schedule.satisfiesPrecedenceConstraints() << std::endl;
                std::cout << "  Schedule satisfies node type constraints: ";
                std::cout << bsp_schedule.satisfiesNodeTypeConstraints() << std::endl;
            }

            if (plot_dot_graphs_) {
                const auto &rep_dag = bsp_schedule.getInstance().getComputationalDag();
                std::vector<unsigned> colors(rep_dag.num_vertices());
                std::map<std::pair<unsigned, unsigned>, unsigned> proc_ss_to_color;
                unsigned next_color = 0;

                for (const auto &v : rep_dag.vertices()) {
                    const auto assignment = std::make_pair(bsp_schedule.assignedProcessor(v), bsp_schedule.assignedSuperstep(v));
                    if (proc_ss_to_color.find(assignment) == proc_ss_to_color.end()) {
                        proc_ss_to_color[assignment] = next_color++;
                    }
                    colors[v] = proc_ss_to_color[assignment];
                }

                auto now = std::chrono::system_clock::now();
                auto in_time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
                std::string timestamp = ss.str() + "_";

                DotFileWriter writer;
                writer.write_colored_graph(timestamp + "iso_group_rep_" + std::to_string(group_idx) + ".dot", rep_dag, colors);
            }

            const bool max_bsp = use_max_bsp && (representative_instance.getComputationalDag().num_edges() == 0) && (representative_instance.getComputationalDag().vertex_type(0) == 0);

            // Build data structures for applying the pattern ---
            // Map (superstep, processor) -> relative partition ID
            std::map<std::pair<unsigned, unsigned>, vertex_idx_t<Graph_t>> sp_proc_to_relative_partition;
            vertex_idx_t<Graph_t> num_partitions_per_subgraph = 0;
            for (vertex_idx_t<Graph_t> j = 0; j < static_cast<vertex_idx_t<Graph_t>>(rep_subgraph_vertices_sorted.size()); ++j) {
                auto sp_pair = std::make_pair(bsp_schedule.assignedSuperstep(j), bsp_schedule.assignedProcessor(j));

                if (max_bsp)
                    sp_pair = std::make_pair(j, 0);

                if (sp_proc_to_relative_partition.find(sp_pair) == sp_proc_to_relative_partition.end()) {
                    sp_proc_to_relative_partition[sp_pair] = num_partitions_per_subgraph++;
                }
            }

            // Pre-compute hashes for the representative to use for mapping
            MerkleHashComputer<Constr_Graph_t> rep_hasher(representative_instance.getComputationalDag());

            // Replicate the schedule pattern for ALL subgraphs in the group ---
            for (vertex_idx_t<Graph_t> i = 0; i < static_cast<vertex_idx_t<Graph_t>>(group.subgraphs.size()); ++i) {
                auto current_subgraph_vertices_sorted = group.subgraphs[i];
                std::sort(current_subgraph_vertices_sorted.begin(), current_subgraph_vertices_sorted.end());

                // Map from a vertex in the current subgraph to its corresponding local index (0, 1, ...) in the representative's schedule
                std::unordered_map<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>> current_vertex_to_rep_local_idx;

                if (i == 0) { // The first subgraph is the representative itself
                    current_vertex_to_rep_local_idx = std::move(rep_global_to_local_map);
                } else { // For other subgraphs, build the isomorphic mapping
                    Constr_Graph_t current_subgraph_graph;
                    create_induced_subgraph(instance.getComputationalDag(), current_subgraph_graph, current_subgraph_vertices_sorted);

                    MerkleHashComputer<Constr_Graph_t> current_hasher(current_subgraph_graph);

                    for (const auto &[hash, rep_orbit_nodes] : rep_hasher.get_orbits()) {
                        const auto &current_orbit_nodes = current_hasher.get_orbit_from_hash(hash);
                        for (size_t k = 0; k < rep_orbit_nodes.size(); ++k) {
                            // Map: current_subgraph_vertex -> representative_subgraph_local_idx
                            current_vertex_to_rep_local_idx[current_subgraph_vertices_sorted[current_orbit_nodes[k]]] = static_cast<vertex_idx_t<Constr_Graph_t>>(rep_orbit_nodes[k]);
                        }
                    }
                }

                // Apply the partition pattern
                for (const auto &current_vertex : current_subgraph_vertices_sorted) {
                    const auto rep_local_idx = current_vertex_to_rep_local_idx.at(current_vertex);
                    auto sp_pair = std::make_pair(bsp_schedule.assignedSuperstep(rep_local_idx), bsp_schedule.assignedProcessor(rep_local_idx));

                    if (max_bsp)
                        sp_pair = std::make_pair(rep_local_idx, 0);

                    partition[current_vertex] = current_partition_idx + sp_proc_to_relative_partition.at(sp_pair);
                }
                current_partition_idx += num_partitions_per_subgraph;
            }
        }
    }
};

} // namespace osp