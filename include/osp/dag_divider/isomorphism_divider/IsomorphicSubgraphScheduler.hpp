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

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "EftSubgraphScheduler.hpp"
#include "HashComputer.hpp"
#include "MerkleHashComputer.hpp"
#include "OrbitGraphProcessor.hpp"
#include "TrimmedGroupScheduler.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

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
template <typename GraphT, typename ConstrGraphT>
class IsomorphicSubgraphScheduler {
    static_assert(IsComputationalDagV<Graph_t>, "Graph must be a computational DAG");
    static_assert(IsComputationalDagV<Constr_Graph_t>, "Constr_Graph_t must be a computational DAG");
    static_assert(is_constructable_cdag_v<Constr_Graph_t>, "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>>,
                  "Graph_t and Constr_Graph_t must have the same vertex_idx types");

  private:
    static constexpr bool verbose_ = false;
    const HashComputer<vertex_idx_t<Graph_t>> *hashComputer_;
    size_t symmetry_ = 4;
    Scheduler<ConstrGraphT> *bspScheduler_;
    bool useMaxGroupSize_ = false;
    unsigned maxGroupSize_ = 0;
    bool plotDotGraphs_ = false;
    v_workw_t<Constr_Graph_t> workThreshold_ = 10;
    v_workw_t<Constr_Graph_t> criticalPathThreshold_ = 10;
    double orbitLockRatio_ = 0.4;
    double naturalBreaksCountPercentage_ = 0.1;
    bool mergeDifferentNodeTypes_ = true;
    bool allowUseTrimmedScheduler_ = true;
    bool useMaxBsp_ = false;
    bool useAdaptiveSymmetryThreshold_ = true;

  public:
    explicit IsomorphicSubgraphScheduler(Scheduler<ConstrGraphT> &bspScheduler)
        : hash_computer_(nullptr), bspScheduler_(&bspScheduler), plotDotGraphs_(false) {}

    IsomorphicSubgraphScheduler(Scheduler<ConstrGraphT> &bspScheduler, const HashComputer<vertex_idx_t<Graph_t>> &hashComputer)
        : hash_computer_(&hash_computer), bspScheduler_(&bspScheduler), plotDotGraphs_(false) {}

    virtual ~IsomorphicSubgraphScheduler() {}

    void SetMergeDifferentTypes(bool flag) { mergeDifferentNodeTypes_ = flag; }

    void SetWorkThreshold(v_workw_t<Constr_Graph_t> workThreshold) { work_threshold_ = work_threshold; }

    void SetCriticalPathThreshold(v_workw_t<Constr_Graph_t> criticalPathThreshold) {
        critical_path_threshold_ = critical_path_threshold;
    }

    void SetOrbitLockRatio(double orbitLockRatio) { orbitLockRatio_ = orbitLockRatio; }

    void SetNaturalBreaksCountPercentage(double naturalBreaksCountPercentage) {
        naturalBreaksCountPercentage_ = naturalBreaksCountPercentage;
    }

    void SetAllowTrimmedScheduler(bool flag) { allowUseTrimmedScheduler_ = flag; }

    void SetPlotDotGraphs(bool plot) { plotDotGraphs_ = plot; }

    void DisableUseMaxGroupSize() { useMaxGroupSize_ = false; }

    void SetUseMaxBsp(bool flag) { useMaxBsp_ = flag; }

    void EnableUseMaxGroupSize(const unsigned maxGroupSize) {
        useMaxGroupSize_ = true;
        maxGroupSize_ = maxGroupSize;
    }

    void SetEnableAdaptiveSymmetryThreshold() { useAdaptiveSymmetryThreshold_ = true; }

    void SetUseStaticSymmetryLevel(size_t staticSymmetryLevel) {
        useAdaptiveSymmetryThreshold_ = false;
        symmetry_ = staticSymmetryLevel;
    }

    std::vector<vertex_idx_t<Graph_t>> ComputePartition(const BspInstance<GraphT> &instance) {
        OrbitGraphProcessor<GraphT, ConstrGraphT> orbitProcessor;
        orbit_processor.set_work_threshold(work_threshold_);
        orbitProcessor.setMergeDifferentNodeTypes(mergeDifferentNodeTypes_);
        orbit_processor.setCriticalPathThreshold(critical_path_threshold_);
        orbitProcessor.setLockRatio(orbitLockRatio_);
        orbitProcessor.setNaturalBreaksCountPercentage(naturalBreaksCountPercentage_);
        if (not useAdaptiveSymmetryThreshold_) {
            orbitProcessor.setUseStaticSymmetryLevel(symmetry_);
        }

        std::unique_ptr<HashComputer<vertex_idx_t<Graph_t>>> localHasher;
        if (!hash_computer_) {
            localHasher = std::make_unique<MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true>>(
                instance.getComputationalDag(), instance.getComputationalDag());
            hash_computer_ = local_hasher.get();
        }

        orbit_processor.discover_isomorphic_groups(instance.getComputationalDag(), *hash_computer_);

        auto isomorphicGroups = orbitProcessor.get_final_groups();

        std::vector<bool> wasTrimmed(isomorphicGroups.size(), false);
        TrimSubgraphGroups(isomorphicGroups, instance, wasTrimmed);    // Apply trimming and record which groups were affected

        auto input = PrepareSubgraphSchedulingInput(instance, isomorphicGroups, wasTrimmed);

        EftSubgraphScheduler<ConstrGraphT> etfScheduler;
        SubgraphSchedule subgraphSchedule
            = etfScheduler.run(input.instance, input.multiplicities, input.required_proc_types, input.max_num_processors);
        subgraphSchedule.wasTrimmed_ = std::move(wasTrimmed);    // Pass through trimming info

        std::vector<vertex_idx_t<Graph_t>> partition(instance.numberOfVertices(), 0);
        schedule_isomorphic_group(instance, isomorphic_groups, subgraph_schedule, partition);

        if (plotDotGraphs_) {
            auto now = std::chrono::system_clock::now();
            auto inTimeT = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&inTimeT), "%Y%m%d_%H%M%S");
            std::string timestamp = ss.str() + "_";

            DotFileWriter writer;
            writer.write_colored_graph(
                timestamp + "isomorphic_groups.dot", instance.getComputationalDag(), orbitProcessor.get_final_contraction_map());
            writer.write_colored_graph(
                timestamp + "orbits_colored.dot", instance.getComputationalDag(), orbitProcessor.get_contraction_map());
            writer.write_graph(timestamp + "iso_groups_contracted.dot", input.instance.getComputationalDag());
            writer.write_colored_graph(timestamp + "graph_partition.dot", instance.getComputationalDag(), partition);
            ConstrGraphT coraseGraph;
            coarser_util::construct_coarse_dag(instance.getComputationalDag(), corase_graph, partition);
            writer.write_graph(timestamp + "block_graph.dot", coraseGraph);
        }
        return partition;
    }

  protected:
    template <typename GT, typename CGT>
    struct SubgraphSchedulerInput {
        BspInstance<CGT> instance_;
        std::vector<unsigned> multiplicities_;
        std::vector<unsigned> maxNumProcessors_;
        std::vector<std::vector<v_workw_t<G_t>>> requiredProcTypes_;
    };

    void TrimSubgraphGroups(std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group> &isomorphicGroups,
                            const BspInstance<GraphT> &instance,
                            std::vector<bool> &wasTrimmed) {
        if constexpr (verbose_) {
            std::cout << "\n--- Trimming Isomorphic Subgraph Groups ---" << std::endl;
        }
        for (size_t groupIdx = 0; groupIdx < isomorphicGroups.size(); ++groupIdx) {
            auto &group = isomorphicGroups[groupIdx];
            const unsigned groupSize = static_cast<unsigned>(group.size());
            if (groupSize <= 1) {
                continue;
            }

            unsigned effectiveMinProcTypeCount = 0;

            if (useMaxGroupSize_) {
                if constexpr (verbose_) {
                    std::cout << "Group " << groupIdx << " (size " << groupSize
                              << "): Using fixed max_group_size_ = " << maxGroupSize_ << " for trimming." << std::endl;
                }
                effectiveMinProcTypeCount = maxGroupSize_;
            } else {
                // Determine if the group consists of a single node type
                bool isSingleTypeGroup = true;
                v_type_t<Graph_t> commonNodeType = 0;

                if constexpr (HasTypedVerticesV<Graph_t>) {
                    if (!group.subgraphs.empty() && !group.subgraphs[0].empty()) {
                        commonNodeType = instance.getComputationalDag().vertex_type(group.subgraphs[0][0]);
                        const auto &repSubgraph = group.subgraphs[0];
                        for (const auto &vertex : repSubgraph) {
                            if (instance.getComputationalDag().vertex_type(vertex) != common_node_type) {
                                isSingleTypeGroup = false;
                                break;
                            }
                        }
                    } else {
                        isSingleTypeGroup = false;
                    }
                } else {
                    isSingleTypeGroup = false;
                }

                if (isSingleTypeGroup) {
                    // Dynamically determine min_proc_type_count based on compatible processors for this type
                    unsigned minCompatibleProcessors = std::numeric_limits<unsigned>::max();
                    const auto &procTypeCounts = instance.getArchitecture().getProcessorTypeCount();

                    bool foundCompatibleProcessor = false;
                    for (unsigned procTypeIdx = 0; procTypeIdx < procTypeCounts.size(); ++procTypeIdx) {
                        if (instance.isCompatibleType(common_node_type, procTypeIdx)) {
                            minCompatibleProcessors = std::min(minCompatibleProcessors, procTypeCounts[procTypeIdx]);
                            foundCompatibleProcessor = true;
                        }
                    }
                    if (foundCompatibleProcessor) {
                        if constexpr (verbose_) {
                            std::cout << "Group " << groupIdx << " (size " << groupSize << "): Single node type ("
                                      << common_node_type << "). Min compatible processors: " << minCompatibleProcessors << "."
                                      << std::endl;
                        }
                        effectiveMinProcTypeCount = minCompatibleProcessors;
                    } else {
                        if constexpr (verbose_) {
                            std::cout << "Group " << groupIdx << " (size " << groupSize << "): Single node type ("
                                      << common_node_type << ") but no compatible processors found. Disabling trimming."
                                      << std::endl;
                        }
                        // If no compatible processors found for this type, effectively disable trimming for this group.
                        effectiveMinProcTypeCount = 1;
                    }
                } else {
                    // Fallback to a default min_proc_type_count if not a single-type group or no typed vertices.
                    const auto &typeCount = instance.getArchitecture().getProcessorTypeCount();
                    if (typeCount.empty()) {
                        effectiveMinProcTypeCount = 0;
                    }
                    effectiveMinProcTypeCount = *std::min_element(typeCount.begin(), typeCount.end());
                    if constexpr (verbose_) {
                        std::cout << "Group " << groupIdx << " (size " << groupSize
                                  << "): Multi-type or untyped group. Using default min_proc_type_count: "
                                  << effectiveMinProcTypeCount << "." << std::endl;
                    }
                }
            }

            // Ensure effective_min_proc_type_count is at least 1 for valid GCD calculation.
            if (effectiveMinProcTypeCount == 0) {
                effectiveMinProcTypeCount = 1;
            }

            // If effective_min_proc_type_count is 1, no trimming is needed as gcd(X, 1) = 1.
            if (effectiveMinProcTypeCount <= 1) {
                continue;
            }

            unsigned gcd = std::gcd(groupSize, effectiveMinProcTypeCount);

            if (gcd < groupSize) {
                if constexpr (verbose_) {
                    std::cout << "  -> Trimming group " << groupIdx << ". GCD(" << groupSize << ", " << effectiveMinProcTypeCount
                              << ") = " << gcd << ". Merging " << groupSize / gcd << " subgraphs at a time." << std::endl;
                }

                if (allowUseTrimmedScheduler_) {
                    gcd = 1;
                }

                wasTrimmed[groupIdx] = true;
                const unsigned mergeSize = groupSize / gcd;
                std::vector<std::vector<vertex_idx_t<Graph_t>>> newSubgraphs;
                newSubgraphs.reserve(gcd);

                size_t originalSgCursor = 0;

                for (unsigned j = 0; j < gcd; ++j) {
                    std::vector<vertex_idx_t<Graph_t>> mergedSgVertices;
                    // Estimate capacity for efficiency. Assuming subgraphs have similar sizes.
                    if (!group.subgraphs.empty()) {
                        mergedSgVertices.reserve(group.subgraphs[0].size() * mergeSize);
                    }

                    for (unsigned k = 0; k < mergeSize; ++k) {
                        const auto &sgToMergeVertices = group.subgraphs[originalSgCursor];
                        originalSgCursor++;
                        mergedSgVertices.insert(merged_sg_vertices.end(), sgToMergeVertices.begin(), sgToMergeVertices.end());
                    }
                    newSubgraphs.push_back(std::move(merged_sg_vertices));
                }
                group.subgraphs = std::move(new_subgraphs);
            } else {
                if constexpr (verbose_) {
                    std::cout << "  -> No trim needed for group " << groupIdx << "." << std::endl;
                }
                wasTrimmed[groupIdx] = false;
            }
        }
    }

    SubgraphSchedulerInput<GraphT, ConstrGraphT> PrepareSubgraphSchedulingInput(
        const BspInstance<GraphT> &originalInstance,
        const std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group> &isomorphicGroups,
        const std::vector<bool> &wasTrimmed) {
        SubgraphSchedulerInput<GraphT, ConstrGraphT> result;
        result.instance.getArchitecture() = originalInstance.getArchitecture();
        const unsigned numProcTypes = originalInstance.getArchitecture().getNumberOfProcessorTypes();

        result.multiplicities.resize(isomorphicGroups.size());
        result.max_num_processors.resize(isomorphicGroups.size());
        result.required_proc_types.resize(isomorphicGroups.size());
        std::vector<vertex_idx_t<Constr_Graph_t>> contractionMap(originalInstance.numberOfVertices());

        size_t coarseNodeIdx = 0;
        for (const auto &group : isomorphicGroups) {
            result.max_num_processors[coarseNodeIdx] = static_cast<unsigned>(group.size() * group.subgraphs[0].size());
            result.multiplicities[coarseNodeIdx]
                = (wasTrimmed[coarseNodeIdx] && allowUseTrimmedScheduler_) ? 1 : static_cast<unsigned>(group.subgraphs.size());
            result.required_proc_types[coarseNodeIdx].assign(numProcTypes, 0);

            for (const auto &subgraph : group.subgraphs) {
                for (const auto &vertex : subgraph) {
                    contractionMap[vertex] = static_cast<vertex_idx_t<Constr_Graph_t>>(coarseNodeIdx);
                    const auto vertexWork = originalInstance.getComputationalDag().vertex_work_weight(vertex);
                    const auto vertexType = originalInstance.getComputationalDag().vertex_type(vertex);
                    for (unsigned j = 0; j < numProcTypes; ++j) {
                        if (originalInstance.isCompatibleType(vertexType, j)) {
                            result.required_proc_types[coarseNodeIdx][j] += vertexWork;
                        }
                    }
                }
            }

            ++coarseNodeIdx;
        }
        coarser_util::construct_coarse_dag(
            original_instance.getComputationalDag(), result.instance.getComputationalDag(), contraction_map);

        if constexpr (verbose_) {
            std::cout << "\n--- Preparing Subgraph Scheduling Input ---\n";
            std::cout << "Found " << isomorphicGroups.size() << " isomorphic groups to schedule as coarse nodes.\n";
            for (size_t j = 0; j < isomorphicGroups.size(); ++j) {
                std::cout << "  - Coarse Node " << j << " (from " << isomorphicGroups[j].subgraphs.size()
                          << " isomorphic subgraphs):\n";
                std::cout << "    - Multiplicity for scheduling: " << result.multiplicities[j] << "\n";
                std::cout << "    - Total Work (in coarse graph): " << result.instance.getComputationalDag().vertex_work_weight(j)
                          << "\n";
                std::cout << "    - Required Processor Types: ";
                for (unsigned k = 0; k < numProcTypes; ++k) {
                    std::cout << result.required_proc_types[j][k] << " ";
                }
                std::cout << "\n";
                std::cout << "    - Max number of processors: " << result.max_num_processors[j] << "\n";
            }
        }
        return result;
    }

    void ScheduleIsomorphicGroup(const BspInstance<GraphT> &instance,
                                 const std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group> &isomorphicGroups,
                                 const SubgraphSchedule &subSched,
                                 std::vector<vertex_idx_t<Graph_t>> &partition) {
        vertex_idx_t<Graph_t> currentPartitionIdx = 0;

        for (size_t groupIdx = 0; groupIdx < isomorphicGroups.size(); ++groupIdx) {
            const auto &group = isomorphicGroups[groupIdx];
            if (group.subgraphs.empty()) {
                continue;
            }

            // Schedule the Representative Subgraph to get a BSP schedule pattern ---
            auto repSubgraphVerticesSorted = group.subgraphs[0];
            std::sort(repSubgraphVerticesSorted.begin(), repSubgraphVerticesSorted.end());

            BspInstance<ConstrGraphT> representativeInstance;
            auto repGlobalToLocalMap = create_induced_subgraph_map(
                instance.getComputationalDag(), representativeInstance.getComputationalDag(), repSubgraphVerticesSorted);

            representativeInstance.getArchitecture() = instance.getArchitecture();
            const auto &procsForGroup = subSched.nodeAssignedWorkerPerType_[groupIdx];
            std::vector<v_memw_t<Constr_Graph_t>> memWeights(procsForGroup.size(), 0);
            for (unsigned procType = 0; procType < procsForGroup.size(); ++procType) {
                memWeights[procType]
                    = static_cast<v_memw_t<Constr_Graph_t>>(instance.getArchitecture().maxMemoryBoundProcType(procType));
            }
            representativeInstance.getArchitecture().SetProcessorsConsequTypes(procsForGroup, mem_weights);
            representativeInstance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

            // --- Decide which scheduler to use ---
            unsigned minNonZeroProcs = std::numeric_limits<unsigned>::max();
            for (const auto &procCount : procsForGroup) {
                if (procCount > 0) {
                    minNonZeroProcs = std::min(minNonZeroProcs, procCount);
                }
            }

            bool useTrimmedScheduler = subSched.wasTrimmed_[groupIdx] && minNonZeroProcs > 1 && allowUseTrimmedScheduler_;

            Scheduler<ConstrGraphT> *schedulerForGroupPtr;
            std::unique_ptr<Scheduler<ConstrGraphT>> trimmedSchedulerOwner;
            if (useTrimmedScheduler) {
                if constexpr (verbose_) {
                    std::cout << "Using TrimmedGroupScheduler for group " << groupIdx << std::endl;
                }
                trimmedSchedulerOwner = std::make_unique<TrimmedGroupScheduler<ConstrGraphT>>(*bspScheduler_, minNonZeroProcs);
                schedulerForGroupPtr = trimmedSchedulerOwner.get();
            } else {
                if constexpr (verbose_) {
                    std::cout << "Using standard BSP scheduler for group " << groupIdx << std::endl;
                }
                schedulerForGroupPtr = bspScheduler_;
            }

            // --- Schedule the representative to get the pattern ---
            BspSchedule<ConstrGraphT> bspSchedule(representativeInstance);

            if constexpr (verbose_) {
                std::cout << "--- Scheduling representative for group " << groupIdx << " ---" << std::endl;
                std::cout << "  Number of subgraphs in group: " << group.subgraphs.size() << std::endl;
                const auto &repDag = representativeInstance.getComputationalDag();
                std::cout << "  Representative subgraph size: " << repDag.num_vertices() << " vertices" << std::endl;
                std::vector<unsigned> nodeTypeCounts(repDag.num_vertex_types(), 0);
                for (const auto &v : repDag.vertices()) {
                    nodeTypeCounts[repDag.vertex_type(v)]++;
                }
                std::cout << "    Node type counts: ";
                for (size_t typeIdx = 0; typeIdx < nodeTypeCounts.size(); ++typeIdx) {
                    if (nodeTypeCounts[typeIdx] > 0) {
                        std::cout << "T" << typeIdx << ":" << nodeTypeCounts[typeIdx] << " ";
                    }
                }
                std::cout << std::endl;

                const auto &subArch = representativeInstance.getArchitecture();
                std::cout << "  Sub-architecture for scheduling:" << std::endl;
                std::cout << "    Processors: " << subArch.numberOfProcessors() << std::endl;
                std::cout << "    Processor types counts: ";
                const auto &typeCounts = subArch.getProcessorTypeCount();
                for (size_t typeIdx = 0; typeIdx < typeCounts.size(); ++typeIdx) {
                    std::cout << "T" << typeIdx << ":" << typeCounts[typeIdx] << " ";
                }
                std::cout << std::endl;
                std::cout << "    Sync cost: " << subArch.synchronisationCosts()
                          << ", Comm cost: " << subArch.communicationCosts() << std::endl;
            }

            schedulerForGroupPtr->computeSchedule(bspSchedule);

            if constexpr (verbose_) {
                std::cout << "  Schedule satisfies precedence constraints: ";
                std::cout << bspSchedule.satisfiesPrecedenceConstraints() << std::endl;
                std::cout << "  Schedule satisfies node type constraints: ";
                std::cout << bspSchedule.satisfiesNodeTypeConstraints() << std::endl;
            }

            if (plotDotGraphs_) {
                const auto &repDag = bspSchedule.getInstance().getComputationalDag();
                std::vector<unsigned> colors(repDag.num_vertices());
                std::map<std::pair<unsigned, unsigned>, unsigned> procSsToColor;
                unsigned nextColor = 0;

                for (const auto &v : repDag.vertices()) {
                    const auto assignment = std::make_pair(bspSchedule.assignedProcessor(v), bspSchedule.assignedSuperstep(v));
                    if (procSsToColor.find(assignment) == procSsToColor.end()) {
                        procSsToColor[assignment] = nextColor++;
                    }
                    colors[v] = procSsToColor[assignment];
                }

                auto now = std::chrono::system_clock::now();
                auto inTimeT = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&inTimeT), "%Y%m%d_%H%M%S");
                std::string timestamp = ss.str() + "_";

                DotFileWriter writer;
                writer.write_colored_graph(timestamp + "iso_group_rep_" + std::to_string(groupIdx) + ".dot", repDag, colors);
            }

            const bool maxBsp = useMaxBsp_ && (representativeInstance.getComputationalDag().num_edges() == 0)
                                && (representativeInstance.getComputationalDag().vertex_type(0) == 0);

            // Build data structures for applying the pattern ---
            // Map (superstep, processor) -> relative partition ID
            std::map<std::pair<unsigned, unsigned>, vertex_idx_t<Graph_t>> spProcToRelativePartition;
            vertex_idx_t<Graph_t> numPartitionsPerSubgraph = 0;
            for (vertex_idx_t<Graph_t> j = 0; j < static_cast<vertex_idx_t<Graph_t>>(repSubgraphVerticesSorted.size()); ++j) {
                auto spPair = std::make_pair(bspSchedule.assignedSuperstep(j), bspSchedule.assignedProcessor(j));

                if (maxBsp) {
                    spPair = std::make_pair(j, 0);
                }

                if (spProcToRelativePartition.find(sp_pair) == sp_proc_to_relative_partition.end()) {
                    spProcToRelativePartition[sp_pair] = num_partitions_per_subgraph++;
                }
            }

            // Pre-compute hashes for the representative to use for mapping
            MerkleHashComputer<Constr_Graph_t> repHasher(representativeInstance.getComputationalDag());

            // Replicate the schedule pattern for ALL subgraphs in the group ---
            for (vertex_idx_t<Graph_t> i = 0; i < static_cast<vertex_idx_t<Graph_t>>(group.subgraphs.size()); ++i) {
                auto currentSubgraphVerticesSorted = group.subgraphs[i];
                std::sort(current_subgraph_vertices_sorted.begin(), current_subgraph_vertices_sorted.end());

                // Map from a vertex in the current subgraph to its corresponding local index (0, 1, ...) in the representative's schedule
                std::unordered_map<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>> current_vertex_to_rep_local_idx;

                if (i == 0) {    // The first subgraph is the representative itself
                    current_vertex_to_rep_local_idx = std::move(rep_global_to_local_map);
                } else {    // For other subgraphs, build the isomorphic mapping
                    ConstrGraphT currentSubgraphGraph;
                    create_induced_subgraph(instance.getComputationalDag(), currentSubgraphGraph, current_subgraph_vertices_sorted);

                    MerkleHashComputer<Constr_Graph_t> currentHasher(currentSubgraphGraph);

                    for (const auto &[hash, rep_orbit_nodes] : rep_hasher.get_orbits()) {
                        const auto &current_orbit_nodes = current_hasher.get_orbit_from_hash(hash);
                        for (size_t k = 0; k < rep_orbit_nodes.size(); ++k) {
                            // Map: current_subgraph_vertex -> representative_subgraph_local_idx
                            current_vertex_to_rep_local_idx[current_subgraph_vertices_sorted[current_orbit_nodes[k]]]
                                = static_cast<vertex_idx_t<Constr_Graph_t>>(rep_orbit_nodes[k]);
                        }
                    }
                }

                // Apply the partition pattern
                for (const auto &current_vertex : current_subgraph_vertices_sorted) {
                    const auto rep_local_idx = current_vertex_to_rep_local_idx.at(current_vertex);
                    auto sp_pair = std::make_pair(bsp_schedule.assignedSuperstep(rep_local_idx),
                                                  bsp_schedule.assignedProcessor(rep_local_idx));

                    if (max_bsp) {
                        sp_pair = std::make_pair(rep_local_idx, 0);
                    }

                    partition[current_vertex] = current_partition_idx + sp_proc_to_relative_partition.at(sp_pair);
                }
                currentPartitionIdx += num_partitions_per_subgraph;
            }
        }
    }
};

}    // namespace osp
