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

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/pebbling/PebblingSchedule.hpp"
#include "osp/pebbling/pebblers/pebblingILP/MultiProcessorPebbling.hpp"
#include "osp/pebbling/pebblers/pebblingILP/partialILP/AcyclicDagDivider.hpp"
#include "osp/pebbling/pebblers/pebblingILP/partialILP/SubproblemMultiScheduling.hpp"

namespace osp {

template <typename GraphT>
class PebblingPartialILP : public Scheduler<GraphT> {
    static_assert(is_computational_dag_v<Graph_t>, "PebblingSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>,
                  "PebblingSchedule requires work and comm. weights to have the same type.");

    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = v_workw_t<Graph_t>;

    unsigned minPartitionSize_ = 50, maxPartitionSize_ = 100;
    unsigned timeSecondsForSubIlPs_ = 600;

    bool asynchronous_ = false;
    bool verbose_ = false;

    std::map<std::pair<unsigned, unsigned>, unsigned> partAndNodetypeToNewIndex_;

  public:
    PebblingPartialILP() {}

    virtual ~PebblingPartialILP() = default;

    RETURN_STATUS ComputePebbling(PebblingSchedule<GraphT> &schedule);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    virtual RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override;

    GraphT ContractByPartition(const BspInstance<GraphT> &instance, const std::vector<unsigned> &nodeToPartAssignment);

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "PebblingPartialILP"; }

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> GetMinAndMaxSize() const { return std::make_pair(minPartitionSize_, maxPartitionSize_); }

    inline void SetMinSize(const unsigned minSize) {
        minPartitionSize_ = minSize;
        maxPartitionSize_ = 2 * minSize;
    }

    inline void SetMinAndMaxSize(const std::pair<unsigned, unsigned> minAndMax) {
        minPartitionSize_ = min_and_max.first;
        maxPartitionSize_ = min_and_max.second;
    }

    inline void SetAsync(const bool async) { asynchronous_ = async; }

    inline void SetSecondsForSubIlp(const unsigned seconds) { timeSecondsForSubIlPs_ = seconds; }

    inline void SetVerbose(const bool verbose) { verbose_ = verbose; }
};

template <typename GraphT>
RETURN_STATUS PebblingPartialILP<GraphT>::ComputePebbling(PebblingSchedule<GraphT> &schedule) {
    const BspInstance<GraphT> &instance = schedule.getInstance();

    if (!PebblingSchedule<GraphT>::hasValidSolution(instance)) {
        return RETURN_STATUS::ERROR;
    }

    // STEP 1: divide DAG acyclicly with partitioning ILP

    AcyclicDagDivider<GraphT> dagDivider;
    dagDivider.setMinAndMaxSize({minPartitionSize_, maxPartitionSize_});
    std::vector<unsigned> assignmentToParts = dagDivider.computePartitioning(instance);
    unsigned nrParts = *std::max_element(assignment_to_parts.begin(), assignment_to_parts.end()) + 1;

    // TODO remove source nodes before this?
    GraphT contractedDag = contractByPartition(instance, assignment_to_parts);

    // STEP 2: develop high-level multischedule on parts

    BspInstance<GraphT> contractedInstance(
        contractedDag, instance.getArchitecture(), instance.getNodeProcessorCompatibilityMatrix());

    SubproblemMultiScheduling<GraphT> multiScheduler;
    std::vector<std::set<unsigned>> processorsToPartsAndTypes;
    multiScheduler.computeMultiSchedule(contractedInstance, processors_to_parts_and_types);

    std::vector<std::set<unsigned>> processorsToParts(nrParts);
    for (unsigned part = 0; part < nrParts; ++part) {
        for (unsigned type = 0; type < instance.getComputationalDag().num_vertex_types(); ++type) {
            if (part_and_nodetype_to_new_index.find({part, type}) != part_and_nodetype_to_new_index.end()) {
                unsigned newIndex = part_and_nodetype_to_new_index[{part, type}];
                for (unsigned proc : processors_to_parts_and_types[new_index]) {
                    processors_to_parts[part].insert(proc);
                }
            }
        }
    }

    // AUX: check for isomorphism

    // create set of nodes & external sources for all parts, and the nodes that need to have blue pebble at the end
    std::vector<std::set<vertex_idx>> nodesInPart(nrParts), extra_sources(nr_parts);
    std::vector<std::map<vertex_idx, vertex_idx>> originalNodeId(nrParts);
    std::vector<std::map<unsigned, unsigned>> originalProcId(nrParts);
    for (vertex_idx node = 0; node < instance.numberOfVertices(); ++node) {
        if (instance.getComputationalDag().in_degree(node) > 0) {
            nodesInPart[assignment_to_parts[node]].insert(node);
        } else {
            extra_sources[assignment_to_parts[node]].insert(node);
        }
        for (const vertex_idx &pred : instance.getComputationalDag().parents(node)) {
            if (assignment_to_parts[node] != assignment_to_parts[pred]) {
                extra_sources[assignment_to_parts[node]].insert(pred);
            }
        }
    }

    std::vector<Graph_t> subDags;
    for (unsigned part = 0; part < nrParts; ++part) {
        GraphT dag;
        create_induced_subgraph(instance.getComputationalDag(), dag, nodes_in_part[part], extra_sources[part]);
        subDags.push_back(dag);

        // set source nodes to a new type, so that they are compatible with any processor
        unsigned artificialTypeForSources = subDags.back().num_vertex_types();
        for (vertex_idx nodeIdx = 0; node_idx < extra_sources[part].size(); ++node_idx) {
            subDags.back().set_vertex_type(node_idx, artificialTypeForSources);
        }
    }

    std::vector<unsigned> isomorphicTo(nrParts, UINT_MAX);

    std::cout << "Number of parts: " << nrParts << std::endl;

    for (unsigned part = 0; part < nrParts; ++part) {
        for (unsigned otherPart = part + 1; otherPart < nrParts; ++otherPart) {
            if (isomorphicTo[otherPart] < UINT_MAX) {
                continue;
            }

            bool isomorphic = true;
            if (!checkOrderedIsomorphism(subDags[part], subDags[otherPart])) {
                continue;
            }

            std::vector<unsigned> procAssignedPerType(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
            std::vector<unsigned> otherProcAssignedPerType(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
            for (unsigned proc : processors_to_parts[part]) {
                ++proc_assigned_per_type[instance.getArchitecture().processorType(proc)];
            }
            for (unsigned proc : processors_to_parts[other_part]) {
                ++other_proc_assigned_per_type[instance.getArchitecture().processorType(proc)];
            }

            for (unsigned procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes(); ++procType) {
                if (procAssignedPerType[procType] != other_proc_assigned_per_type[procType]) {
                    isomorphic = false;
                }
            }

            if (!isomorphic) {
                continue;
            }

            isomorphicTo[otherPart] = part;
            std::cout << "Part " << otherPart << " is isomorphic to " << part << std::endl;
        }
    }

    // PART 3: solve a small ILP for each part
    std::vector<std::set<vertex_idx>> inFastMem(instance.numberOfProcessors());
    std::vector<PebblingSchedule<Graph_t>> pebbling(nrParts);
    std::vector<BspArchitecture<Graph_t>> subArch(nrParts);
    std::vector<BspInstance<Graph_t>> subInstance(nrParts);

    // to handle the initial memory content for isomorphic parts
    std::vector<std::vector<std::set<vertex_idx>>> hasRedsInBeginning(
        nr_parts, std::vector<std::set<vertex_idx>>(instance.numberOfProcessors()));

    for (unsigned part = 0; part < nrParts; ++part) {
        std::cout << "part " << part << std::endl;

        // set up sub-DAG
        GraphT &subDag = subDags[part];
        std::map<vertex_idx, vertex_idx> localId;
        vertex_idx nodeIdx = 0;
        for (vertex_idx node : extra_sources[part]) {
            local_id[node] = node_idx;
            original_node_id[part][node_idx] = node;
            ++node_idx;
        }
        for (vertex_idx node : nodes_in_part[part]) {
            local_id[node] = node_idx;
            original_node_id[part][node_idx] = node;
            ++node_idx;
        }

        std::set<vertex_idx> needsBlueAtEnd;
        for (vertex_idx node : nodes_in_part[part]) {
            for (const vertex_idx &succ : instance.getComputationalDag().children(node)) {
                if (assignment_to_parts[node] != assignment_to_parts[succ]) {
                    needs_blue_at_end.insert(local_id[node]);
                }
            }

            if (instance.getComputationalDag().out_degree(node) == 0) {
                needs_blue_at_end.insert(local_id[node]);
            }
        }

        // set up sub-architecture
        subArch[part].setNumberOfProcessors(static_cast<unsigned>(processors_to_parts[part].size()));
        unsigned procIndex = 0;
        for (unsigned proc : processors_to_parts[part]) {
            subArch[part].setProcessorType(proc_index, instance.getArchitecture().processorType(proc));
            subArch[part].setMemoryBound(instance.getArchitecture().memoryBound(proc), proc_index);
            original_proc_id[part][proc_index] = proc;
            ++proc_index;
        }
        subArch[part].setCommunicationCosts(instance.getArchitecture().communicationCosts());
        subArch[part].setSynchronisationCosts(instance.getArchitecture().synchronisationCosts());
        // no NUMA parameters for now

        // skip if isomorphic to previous part
        if (isomorphicTo[part] < UINT_MAX) {
            pebbling[part] = pebbling[isomorphicTo[part]];
            hasRedsInBeginning[part] = has_reds_in_beginning[isomorphicTo[part]];
            continue;
        }

        // set node-processor compatibility matrix
        std::vector<std::vector<bool>> compMatrix = instance.getNodeProcessorCompatibilityMatrix();
        compMatrix.emplace_back(instance.getArchitecture().getNumberOfProcessorTypes(), true);
        subInstance[part] = BspInstance(subDag, subArch[part], comp_matrix);

        // currently we only allow the input laoding scenario - the case where this is false is unmaintained/untested
        bool needToLoadInputs = true;

        // keep in fast memory what's relevant, remove the rest
        for (unsigned proc = 0; proc < processorsToParts[part].size(); ++proc) {
            hasRedsInBeginning[part][proc].clear();
            std::set<vertex_idx> newContentFastMem;
            for (vertex_idx node : in_fast_mem[original_proc_id[part][proc]]) {
                if (local_id.find(node) != local_id.end()) {
                    has_reds_in_beginning[part][proc].insert(local_id[node]);
                    new_content_fast_mem.insert(node);
                }
            }

            inFastMem[original_proc_id[part][proc]] = new_content_fast_mem;
        }

        // heuristic solution for baseline
        PebblingSchedule<GraphT> heuristicPebbling;
        GreedyBspScheduler<Graph_t> greedyScheduler;
        BspSchedule<GraphT> bspHeuristic(subInstance[part]);
        greedyScheduler.computeSchedule(bspHeuristic);

        std::set<vertex_idx> extraSourceIds;
        for (vertex_idx idx = 0; idx < extra_sources[part].size(); ++idx) {
            extraSourceIds.insert(idx);
        }

        heuristicPebbling.setNeedToLoadInputs(true);
        heuristicPebbling.SetExternalSources(extra_source_ids);
        heuristicPebbling.SetNeedsBlueAtEnd(needs_blue_at_end);
        heuristicPebbling.SetHasRedInBeginning(has_reds_in_beginning[part]);
        heuristicPebbling.ConvertFromBsp(bspHeuristic, PebblingSchedule<GraphT>::CACHE_EVICTION_STRATEGY::FORESIGHT);

        heuristicPebbling.removeEvictStepsFromEnd();
        pebbling[part] = heuristicPebbling;
        cost_type heuristicCost = asynchronous_ ? heuristicPebbling.computeAsynchronousCost() : heuristicPebbling.computeCost();

        if (!heuristicPebbling.isValid()) {
            std::cout << "ERROR: Pebbling heuristic INVALID!" << std::endl;
        }

        // solution with subILP
        MultiProcessorPebbling<GraphT> mpp;
        mpp.setVerbose(verbose_);
        mpp.setTimeLimitSeconds(timeSecondsForSubIlPs_);
        mpp.setMaxTime(2 * maxPartitionSize_);    // just a heuristic choice, does not guarantee feasibility!
        mpp.setNeedsBlueAtEnd(needs_blue_at_end);
        mpp.setNeedToLoadInputs(needToLoadInputs);
        mpp.setHasRedInBeginning(has_reds_in_beginning[part]);

        PebblingSchedule<GraphT> pebblingILP(subInstance[part]);
        RETURN_STATUS status = mpp.computePebblingWithInitialSolution(heuristicPebbling, pebblingILP, asynchronous_);
        if (status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND) {
            if (!pebblingILP.isValid()) {
                std::cout << "ERROR: Pebbling ILP INVALID!" << std::endl;
            }

            pebblingILP.removeEvictStepsFromEnd();
            cost_type ilpCost = asynchronous_ ? pebblingILP.computeAsynchronousCost() : pebblingILP.computeCost();
            if (ILP_cost < heuristicCost) {
                pebbling[part] = pebblingILP;
                std::cout << "ILP chosen instead of greedy. (" << ILP_cost << " < " << heuristicCost << ")" << std::endl;
            } else {
                std::cout << "Greedy chosen instead of ILP. (" << heuristicCost << " < " << ILP_cost << ")" << std::endl;
            }

            // save fast memory content for next subproblem
            std::vector<std::set<vertex_idx>> fastMemContentAtEnd = pebbling[part].getMemContentAtEnd();
            for (unsigned proc = 0; proc < processorsToParts[part].size(); ++proc) {
                inFastMem[original_proc_id[part][proc]].clear();
                for (vertex_idx node : fast_mem_content_at_end[proc]) {
                    in_fast_mem[original_proc_id[part][proc]].insert(original_node_id[part][node]);
                }
            }
        } else {
            std::cout << "ILP found no solution; using greedy instead (cost = " << heuristicCost << ")." << std::endl;
        }
    }

    // AUX: assemble final schedule from subschedules
    schedule.CreateFromPartialPebblings(
        instance, pebbling, processors_to_parts, original_node_id, original_proc_id, has_reds_in_beginning);
    schedule.cleanSchedule();
    return schedule.isValid() ? RETURN_STATUS::OSP_SUCCESS : RETURN_STATUS::ERROR;
}

template <typename GraphT>
GraphT PebblingPartialILP<GraphT>::ContractByPartition(const BspInstance<GraphT> &instance,
                                                       const std::vector<unsigned> &nodeToPartAssignment) {
    const auto &g = instance.getComputationalDag();

    part_and_nodetype_to_new_index.clear();

    unsigned nrNewNodes = 0;
    for (vertex_idx node = 0; node < instance.numberOfVertices(); ++node) {
        if (part_and_nodetype_to_new_index.find({node_to_part_assignment[node], G.vertex_type(node)})
            == part_and_nodetype_to_new_index.end()) {
            part_and_nodetype_to_new_index[{node_to_part_assignment[node], G.vertex_type(node)}] = nr_new_nodes;
            ++nrNewNodes;
        }
    }

    GraphT contracted;
    for (vertex_idx node = 0; node < nrNewNodes; ++node) {
        contracted.add_vertex(0, 0, 0);
    }

    std::set<std::pair<vertex_idx, vertex_idx>> edges;

    for (vertex_idx node = 0; node < instance.numberOfVertices(); ++node) {
        vertex_idx nodeNewIndex = part_and_nodetype_to_new_index[{node_to_part_assignment[node], G.vertex_type(node)}];
        for (const vertex_idx &succ : instance.getComputationalDag().children(node)) {
            if (node_to_part_assignment[node] != node_to_part_assignment[succ]) {
                edges.emplace(node_new_index, part_and_nodetype_to_new_index[{node_to_part_assignment[succ], G.vertex_type(succ)}]);
            }
        }

        contracted.set_vertex_work_weight(node_new_index,
                                          contracted.vertex_work_weight(node_new_index) + g.vertex_work_weight(node));
        contracted.set_vertex_comm_weight(node_new_index,
                                          contracted.vertex_comm_weight(node_new_index) + g.vertex_comm_weight(node));
        contracted.set_vertex_mem_weight(node_new_index, contracted.vertex_mem_weight(node_new_index) + g.vertex_mem_weight(node));
        contracted.set_vertex_type(node_new_index, g.vertex_type(node));
    }

    for (auto edge : edges) {
        if constexpr (has_edge_weights_v<Graph_t>) {
            contracted.add_edge(edge.first, edge.second, 1);
        } else {
            contracted.add_edge(edge.first, edge.second);
        }
    }

    return contracted;
}

template <typename GraphT>
RETURN_STATUS PebblingPartialILP<GraphT>::ComputeSchedule(BspSchedule<GraphT> &) {
    return RETURN_STATUS::ERROR;
}

}    // namespace osp
