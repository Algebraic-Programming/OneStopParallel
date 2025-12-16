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
 * @tparam ConstrGraphT The type of the constructable computational DAG used for internal representations.
 */
template <typename GraphT, typename ConstrGraphT>
class IsomorphicSubgraphScheduler {
    static_assert(isComputationalDagV<GraphT>, "Graph must be a computational DAG");
    static_assert(isComputationalDagV<ConstrGraphT>, "ConstrGraphT must be a computational DAG");
    static_assert(isConstructableCdagV<ConstrGraphT>, "ConstrGraphT must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>>,
                  "Graph_t and ConstrGraphT must have the same VertexIdx types");

  private:
    static constexpr bool verbose_ = false;
    const HashComputer<VertexIdxT<GraphT>> *hashComputer_;
    size_t symmetry_ = 4;
    Scheduler<ConstrGraphT> *bspScheduler_;
    bool useMaxGroupSize_ = false;
    unsigned maxGroupSize_ = 0;
    bool plotDotGraphs_ = false;
    VWorkwT<ConstrGraphT> workThreshold_ = 10;
    VWorkwT<ConstrGraphT> criticalPathThreshold_ = 10;
    double orbitLockRatio_ = 0.4;
    double naturalBreaksCountPercentage_ = 0.1;
    bool mergeDifferentNodeTypes_ = true;
    bool allowUseTrimmedScheduler_ = true;
    bool useMaxBsp_ = false;
    bool useAdaptiveSymmetryThreshold_ = true;

  public:
    IsomorphicSubgraphScheduler(Scheduler<ConstrGraphT> &bspScheduler)
        : hashComputer_(nullptr), bspScheduler_(&bspScheduler), plotDotGraphs_(false) {}

    IsomorphicSubgraphScheduler(Scheduler<ConstrGraphT> &bspScheduler, const HashComputer<VertexIdxT<GraphT>> &hashComputer)
        : hashComputer_(&hashComputer), bspScheduler_(&bspScheduler), plotDotGraphs_(false) {}

    virtual ~IsomorphicSubgraphScheduler() {}

    void SetMergeDifferentTypes(bool flag) { mergeDifferentNodeTypes_ = flag; }

    void SetWorkThreshold(VWorkwT<ConstrGraphT> workThreshold) { workThreshold_ = workThreshold; }

    void SetCriticalPathThreshold(VWorkwT<ConstrGraphT> criticalPathThreshold) { criticalPathThreshold_ = criticalPathThreshold; }

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

    std::vector<VertexIdxT<GraphT>> ComputePartition(const BspInstance<GraphT> &instance) {
        OrbitGraphProcessor<GraphT, ConstrGraphT> orbitProcessor;
        orbitProcessor.SetWorkThreshold(workThreshold_);
        orbitProcessor.SetMergeDifferentNodeTypes(mergeDifferentNodeTypes_);
        orbitProcessor.SetCriticalPathThreshold(criticalPathThreshold_);
        orbitProcessor.SetLockRatio(orbitLockRatio_);
        orbitProcessor.SetNaturalBreaksCountPercentage(naturalBreaksCountPercentage_);
        if (not useAdaptiveSymmetryThreshold_) {
            orbitProcessor.SetUseStaticSymmetryLevel(symmetry_);
        }

        std::unique_ptr<HashComputer<VertexIdxT<GraphT>>> localHasher;
        if (!hashComputer_) {
            localHasher = std::make_unique<MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true>>(
                instance.GetComputationalDag(), instance.GetComputationalDag());
            hashComputer_ = localHasher.get();
        }

        orbitProcessor.DiscoverIsomorphicGroups(instance.GetComputationalDag(), *hashComputer_);

        auto isomorphicGroups = orbitProcessor.GetFinalGroups();

        std::vector<bool> wasTrimmed(isomorphicGroups.size(), false);
        TrimSubgraphGroups(isomorphicGroups, instance, wasTrimmed);    // Apply trimming and record which groups were affected

        auto input = PrepareSubgraphSchedulingInput(instance, isomorphicGroups, wasTrimmed);

        EftSubgraphScheduler<ConstrGraphT> etfScheduler;
        SubgraphSchedule subgraphSchedule
            = etfScheduler.Run(input.instance_, input.multiplicities_, input.requiredProcTypes_, input.maxNumProcessors_);
        subgraphSchedule.wasTrimmed_ = std::move(wasTrimmed);    // Pass through trimming info

        std::vector<VertexIdxT<GraphT>> partition(instance.NumberOfVertices(), 0);
        ScheduleIsomorphicGroup(instance, isomorphicGroups, subgraphSchedule, partition);

        if (plotDotGraphs_) {
            auto now = std::chrono::system_clock::now();
            auto inTimeT = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&inTimeT), "%Y%m%d_%H%M%S");
            std::string timestamp = ss.str() + "_";

            DotFileWriter writer;
            writer.WriteColoredGraph(
                timestamp + "isomorphic_groups.dot", instance.GetComputationalDag(), orbitProcessor.GetFinalContractionMap());
            writer.WriteColoredGraph(
                timestamp + "orbits_colored.dot", instance.GetComputationalDag(), orbitProcessor.GetContractionMap());
            writer.WriteGraph(timestamp + "iso_groups_contracted.dot", input.instance_.GetComputationalDag());
            writer.WriteColoredGraph(timestamp + "graph_partition.dot", instance.GetComputationalDag(), partition);
            ConstrGraphT coarseGraph;
            coarser_util::ConstructCoarseDag(instance.GetComputationalDag(), coarseGraph, partition);
            writer.WriteGraph(timestamp + "block_graph.dot", coarseGraph);
        }
        return partition;
    }

  protected:
    template <typename GT, typename CGT>
    struct SubgraphSchedulerInput {
        BspInstance<CGT> instance_;
        std::vector<unsigned> multiplicities_;
        std::vector<unsigned> maxNumProcessors_;
        std::vector<std::vector<VWorkwT<GT>>> requiredProcTypes_;
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
                VTypeT<GraphT> commonNodeType = 0;

                if constexpr (hasTypedVerticesV<GraphT>) {
                    if (!group.subgraphs_.empty() && !group.subgraphs_[0].empty()) {
                        commonNodeType = instance.GetComputationalDag().VertexType(group.subgraphs_[0][0]);
                        const auto &repSubgraph = group.subgraphs_[0];
                        for (const auto &vertex : repSubgraph) {
                            if (instance.GetComputationalDag().VertexType(vertex) != commonNodeType) {
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
                    const auto &procTypeCounts = instance.GetArchitecture().GetProcessorTypeCount();

                    bool foundCompatibleProcessor = false;
                    for (unsigned procTypeIdx = 0; procTypeIdx < procTypeCounts.size(); ++procTypeIdx) {
                        if (instance.IsCompatibleType(commonNodeType, procTypeIdx)) {
                            minCompatibleProcessors = std::min(minCompatibleProcessors, procTypeCounts[procTypeIdx]);
                            foundCompatibleProcessor = true;
                        }
                    }
                    if (foundCompatibleProcessor) {
                        if constexpr (verbose_) {
                            std::cout << "Group " << groupIdx << " (size " << groupSize << "): Single node type ("
                                      << commonNodeType << "). Min compatible processors: " << minCompatibleProcessors << "."
                                      << std::endl;
                        }
                        effectiveMinProcTypeCount = minCompatibleProcessors;
                    } else {
                        if constexpr (verbose_) {
                            std::cout << "Group " << groupIdx << " (size " << groupSize << "): Single node type ("
                                      << commonNodeType << ") but no compatible processors found. Disabling trimming."
                                      << std::endl;
                        }
                        // If no compatible processors found for this type, effectively disable trimming for this group.
                        effectiveMinProcTypeCount = 1;
                    }
                } else {
                    // Fallback to a default min_proc_type_count if not a single-type group or no typed vertices.
                    const auto &typeCount = instance.GetArchitecture().GetProcessorTypeCount();
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
                std::vector<std::vector<VertexIdxT<GraphT>>> newSubgraphs;
                newSubgraphs.reserve(gcd);

                size_t originalSgCursor = 0;

                for (unsigned j = 0; j < gcd; ++j) {
                    std::vector<VertexIdxT<GraphT>> mergedSgVertices;
                    // Estimate capacity for efficiency. Assuming subgraphs have similar sizes.
                    if (!group.subgraphs_.empty()) {
                        mergedSgVertices.reserve(group.subgraphs_[0].size() * mergeSize);
                    }

                    for (unsigned k = 0; k < mergeSize; ++k) {
                        const auto &sgToMergeVertices = group.subgraphs_[originalSgCursor];
                        originalSgCursor++;
                        mergedSgVertices.insert(mergedSgVertices.end(), sgToMergeVertices.begin(), sgToMergeVertices.end());
                    }
                    newSubgraphs.push_back(std::move(mergedSgVertices));
                }
                group.subgraphs_ = std::move(newSubgraphs);
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
        result.instance_.GetArchitecture() = originalInstance.GetArchitecture();
        const unsigned numProcTypes = originalInstance.GetArchitecture().GetNumberOfProcessorTypes();

        result.multiplicities_.resize(isomorphicGroups.size());
        result.maxNumProcessors_.resize(isomorphicGroups.size());
        result.requiredProcTypes_.resize(isomorphicGroups.size());
        std::vector<VertexIdxT<ConstrGraphT>> contractionMap(originalInstance.NumberOfVertices());

        size_t coarseNodeIdx = 0;
        for (const auto &group : isomorphicGroups) {
            result.maxNumProcessors_[coarseNodeIdx] = static_cast<unsigned>(group.size() * group.subgraphs_[0].size());
            result.multiplicities_[coarseNodeIdx]
                = (wasTrimmed[coarseNodeIdx] && allowUseTrimmedScheduler_) ? 1 : static_cast<unsigned>(group.subgraphs_.size());
            result.requiredProcTypes_[coarseNodeIdx].assign(numProcTypes, 0);

            for (const auto &subgraph : group.subgraphs_) {
                for (const auto &vertex : subgraph) {
                    contractionMap[vertex] = static_cast<VertexIdxT<ConstrGraphT>>(coarseNodeIdx);
                    const auto vertexWork = originalInstance.GetComputationalDag().VertexWorkWeight(vertex);
                    const auto vertexType = originalInstance.GetComputationalDag().VertexType(vertex);
                    for (unsigned j = 0; j < numProcTypes; ++j) {
                        if (originalInstance.IsCompatibleType(vertexType, j)) {
                            result.requiredProcTypes_[coarseNodeIdx][j] += vertexWork;
                        }
                    }
                }
            }

            ++coarseNodeIdx;
        }
        coarser_util::ConstructCoarseDag(
            originalInstance.GetComputationalDag(), result.instance_.GetComputationalDag(), contractionMap);

        if constexpr (verbose_) {
            std::cout << "\n--- Preparing Subgraph Scheduling Input ---\n";
            std::cout << "Found " << isomorphicGroups.size() << " isomorphic groups to schedule as coarse nodes.\n";
            for (size_t j = 0; j < isomorphicGroups.size(); ++j) {
                std::cout << "  - Coarse Node " << j << " (from " << isomorphicGroups[j].subgraphs_.size()
                          << " isomorphic subgraphs):\n";
                std::cout << "    - Multiplicity for scheduling: " << result.multiplicities_[j] << "\n";
                std::cout << "    - Total Work (in coarse graph): " << result.instance_.GetComputationalDag().VertexWorkWeight(j)
                          << "\n";
                std::cout << "    - Required Processor Types: ";
                for (unsigned k = 0; k < numProcTypes; ++k) {
                    std::cout << result.requiredProcTypes_[j][k] << " ";
                }
                std::cout << "\n";
                std::cout << "    - Max number of processors: " << result.maxNumProcessors_[j] << "\n";
            }
        }
        return result;
    }

    void ScheduleIsomorphicGroup(const BspInstance<GraphT> &instance,
                                 const std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group> &isomorphicGroups,
                                 const SubgraphSchedule &subSched,
                                 std::vector<VertexIdxT<GraphT>> &partition) {
        VertexIdxT<GraphT> currentPartitionIdx = 0;

        for (size_t groupIdx = 0; groupIdx < isomorphicGroups.size(); ++groupIdx) {
            const auto &group = isomorphicGroups[groupIdx];
            if (group.subgraphs_.empty()) {
                continue;
            }

            // Schedule the Representative Subgraph to get a BSP schedule pattern ---
            auto repSubgraphVerticesSorted = group.subgraphs_[0];
            std::sort(repSubgraphVerticesSorted.begin(), repSubgraphVerticesSorted.end());

            BspInstance<ConstrGraphT> representativeInstance;
            auto repGlobalToLocalMap = CreateInducedSubgraphMap(
                instance.GetComputationalDag(), representativeInstance.GetComputationalDag(), repSubgraphVerticesSorted);

            representativeInstance.GetArchitecture() = instance.GetArchitecture();
            const auto &procsForGroup = subSched.nodeAssignedWorkerPerType_[groupIdx];
            std::vector<VMemwT<ConstrGraphT>> memWeights(procsForGroup.size(), 0);
            for (unsigned procType = 0; procType < procsForGroup.size(); ++procType) {
                memWeights[procType]
                    = static_cast<VMemwT<ConstrGraphT>>(instance.GetArchitecture().MaxMemoryBoundProcType(procType));
            }
            representativeInstance.GetArchitecture().SetProcessorsConsequTypes(procsForGroup, memWeights);
            representativeInstance.SetNodeProcessorCompatibility(instance.GetProcessorCompatibilityMatrix());

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
                std::cout << "  Number of subgraphs in group: " << group.subgraphs_.size() << std::endl;
                const auto &repDag = representativeInstance.GetComputationalDag();
                std::cout << "  Representative subgraph size: " << repDag.NumVertices() << " vertices" << std::endl;
                std::vector<unsigned> nodeTypeCounts(repDag.NumVertexTypes(), 0);
                for (const auto &v : repDag.Vertices()) {
                    nodeTypeCounts[repDag.VertexType(v)]++;
                }
                std::cout << "    Node type counts: ";
                for (size_t typeIdx = 0; typeIdx < nodeTypeCounts.size(); ++typeIdx) {
                    if (nodeTypeCounts[typeIdx] > 0) {
                        std::cout << "T" << typeIdx << ":" << nodeTypeCounts[typeIdx] << " ";
                    }
                }
                std::cout << std::endl;

                const auto &subArch = representativeInstance.GetArchitecture();
                std::cout << "  Sub-architecture for scheduling:" << std::endl;
                std::cout << "    Processors: " << subArch.NumberOfProcessors() << std::endl;
                std::cout << "    Processor types counts: ";
                const auto &typeCounts = subArch.GetProcessorTypeCount();
                for (size_t typeIdx = 0; typeIdx < typeCounts.size(); ++typeIdx) {
                    std::cout << "T" << typeIdx << ":" << typeCounts[typeIdx] << " ";
                }
                std::cout << std::endl;
                std::cout << "    Sync cost: " << subArch.SynchronisationCosts()
                          << ", Comm cost: " << subArch.CommunicationCosts() << std::endl;
            }

            schedulerForGroupPtr->ComputeSchedule(bspSchedule);

            if constexpr (verbose_) {
                std::cout << "  Schedule satisfies precedence constraints: ";
                std::cout << bspSchedule.SatisfiesPrecedenceConstraints() << std::endl;
                std::cout << "  Schedule satisfies node type constraints: ";
                std::cout << bspSchedule.SatisfiesNodeTypeConstraints() << std::endl;
            }

            if (plotDotGraphs_) {
                const auto &repDag = bspSchedule.GetInstance().GetComputationalDag();
                std::vector<unsigned> colors(repDag.NumVertices());
                std::map<std::pair<unsigned, unsigned>, unsigned> procSsToColor;
                unsigned nextColor = 0;

                for (const auto &v : repDag.Vertices()) {
                    const auto assignment = std::make_pair(bspSchedule.AssignedProcessor(v), bspSchedule.AssignedSuperstep(v));
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
                writer.WriteColoredGraph(timestamp + "iso_group_rep_" + std::to_string(groupIdx) + ".dot", repDag, colors);
            }

            const bool maxBsp = useMaxBsp_ && (representativeInstance.GetComputationalDag().NumEdges() == 0)
                                && (representativeInstance.GetComputationalDag().VertexType(0) == 0);

            // Build data structures for applying the pattern ---
            // Map (superstep, processor) -> relative partition ID
            std::map<std::pair<unsigned, unsigned>, VertexIdxT<GraphT>> spProcToRelativePartition;
            VertexIdxT<GraphT> numPartitionsPerSubgraph = 0;
            for (VertexIdxT<GraphT> j = 0; j < static_cast<VertexIdxT<GraphT>>(repSubgraphVerticesSorted.size()); ++j) {
                auto spPair = std::make_pair(bspSchedule.AssignedSuperstep(j), bspSchedule.AssignedProcessor(j));

                if (maxBsp) {
                    spPair = std::make_pair(j, 0);
                }

                if (spProcToRelativePartition.find(spPair) == spProcToRelativePartition.end()) {
                    spProcToRelativePartition[spPair] = numPartitionsPerSubgraph++;
                }
            }

            // Pre-compute hashes for the representative to use for mapping
            MerkleHashComputer<ConstrGraphT> repHasher(representativeInstance.GetComputationalDag());

            // Replicate the schedule pattern for ALL subgraphs in the group ---
            for (VertexIdxT<GraphT> i = 0; i < static_cast<VertexIdxT<GraphT>>(group.subgraphs_.size()); ++i) {
                auto currentSubgraphVerticesSorted = group.subgraphs_[i];
                std::sort(currentSubgraphVerticesSorted.begin(), currentSubgraphVerticesSorted.end());

                // Map from a vertex in the current subgraph to its corresponding local index (0, 1, ...) in the representative's schedule
                std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>> currentVertexToRepLocalIdx;

                if (i == 0) {    // The first subgraph is the representative itself
                    currentVertexToRepLocalIdx = std::move(repGlobalToLocalMap);
                } else {    // For other subgraphs, build the isomorphic mapping
                    ConstrGraphT currentSubgraphGraph;
                    CreateInducedSubgraph(instance.GetComputationalDag(), currentSubgraphGraph, currentSubgraphVerticesSorted);

                    MerkleHashComputer<ConstrGraphT> currentHasher(currentSubgraphGraph);

                    for (const auto &[hash, repOrbitNodes] : repHasher.GetOrbits()) {
                        const auto &currentOrbitNodes = currentHasher.GetOrbitFromHash(hash);
                        for (size_t k = 0; k < repOrbitNodes.size(); ++k) {
                            // Map: current_subgraph_vertex -> representative_subgraph_local_idx
                            currentVertexToRepLocalIdx[currentSubgraphVerticesSorted[currentOrbitNodes[k]]]
                                = static_cast<VertexIdxT<ConstrGraphT>>(repOrbitNodes[k]);
                        }
                    }
                }

                // Apply the partition pattern
                for (const auto &currentVertex : currentSubgraphVerticesSorted) {
                    const auto repLocalIdx = currentVertexToRepLocalIdx.at(currentVertex);
                    auto spPair
                        = std::make_pair(bspSchedule.AssignedSuperstep(repLocalIdx), bspSchedule.AssignedProcessor(repLocalIdx));

                    if (maxBsp) {
                        spPair = std::make_pair(repLocalIdx, 0);
                    }

                    partition[currentVertex] = currentPartitionIdx + spProcToRelativePartition.at(spPair);
                }
                currentPartitionIdx += numPartitionsPerSubgraph;
            }
        }
    }
};

}    // namespace osp
