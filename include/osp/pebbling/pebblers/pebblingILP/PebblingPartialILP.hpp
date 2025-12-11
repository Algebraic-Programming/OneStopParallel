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

template <typename Graph_t>
class PebblingPartialILP : public Scheduler<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>, "PebblingSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>,
                  "PebblingSchedule requires work and comm. weights to have the same type.");

    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = v_workw_t<Graph_t>;

    unsigned minPartitionSize = 50, maxPartitionSize = 100;
    unsigned time_seconds_for_subILPs = 600;

    bool asynchronous = false;
    bool verbose = false;

    std::map<std::pair<unsigned, unsigned>, unsigned> part_and_nodetype_to_new_index;

  public:
    PebblingPartialILP() {}

    virtual ~PebblingPartialILP() = default;

    RETURN_STATUS computePebbling(PebblingSchedule<Graph_t> &schedule);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override;

    Graph_t contractByPartition(const BspInstance<Graph_t> &instance, const std::vector<unsigned> &node_to_part_assignment);

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "PebblingPartialILP"; }

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> getMinAndMaxSize() const { return std::make_pair(minPartitionSize, maxPartitionSize); }

    inline void setMinSize(const unsigned min_size) {
        minPartitionSize = min_size;
        maxPartitionSize = 2 * min_size;
    }

    inline void setMinAndMaxSize(const std::pair<unsigned, unsigned> min_and_max) {
        minPartitionSize = min_and_max.first;
        maxPartitionSize = min_and_max.second;
    }

    inline void setAsync(const bool async_) { asynchronous = async_; }

    inline void setSecondsForSubILP(const unsigned seconds_) { time_seconds_for_subILPs = seconds_; }

    inline void setVerbose(const bool verbose_) { verbose = verbose_; }
};

template <typename Graph_t>
RETURN_STATUS PebblingPartialILP<Graph_t>::computePebbling(PebblingSchedule<Graph_t> &schedule) {
    const BspInstance<Graph_t> &instance = schedule.getInstance();

    if (!PebblingSchedule<Graph_t>::hasValidSolution(instance)) {
        return RETURN_STATUS::ERROR;
    }

    // STEP 1: divide DAG acyclicly with partitioning ILP

    AcyclicDagDivider<Graph_t> dag_divider;
    dag_divider.setMinAndMaxSize({minPartitionSize, maxPartitionSize});
    std::vector<unsigned> assignment_to_parts = dag_divider.computePartitioning(instance);
    unsigned nr_parts = *std::max_element(assignment_to_parts.begin(), assignment_to_parts.end()) + 1;

    // TODO remove source nodes before this?
    Graph_t contracted_dag = contractByPartition(instance, assignment_to_parts);

    // STEP 2: develop high-level multischedule on parts

    BspInstance<Graph_t> contracted_instance(
        contracted_dag, instance.getArchitecture(), instance.getNodeProcessorCompatibilityMatrix());

    SubproblemMultiScheduling<Graph_t> multi_scheduler;
    std::vector<std::set<unsigned>> processors_to_parts_and_types;
    multi_scheduler.computeMultiSchedule(contracted_instance, processors_to_parts_and_types);

    std::vector<std::set<unsigned>> processors_to_parts(nr_parts);
    for (unsigned part = 0; part < nr_parts; ++part) {
        for (unsigned type = 0; type < instance.getComputationalDag().num_vertex_types(); ++type) {
            if (part_and_nodetype_to_new_index.find({part, type}) != part_and_nodetype_to_new_index.end()) {
                unsigned new_index = part_and_nodetype_to_new_index[{part, type}];
                for (unsigned proc : processors_to_parts_and_types[new_index]) {
                    processors_to_parts[part].insert(proc);
                }
            }
        }
    }

    // AUX: check for isomorphism

    // create set of nodes & external sources for all parts, and the nodes that need to have blue pebble at the end
    std::vector<std::set<vertex_idx>> nodes_in_part(nr_parts), extra_sources(nr_parts);
    std::vector<std::map<vertex_idx, vertex_idx>> original_node_id(nr_parts);
    std::vector<std::map<unsigned, unsigned>> original_proc_id(nr_parts);
    for (vertex_idx node = 0; node < instance.numberOfVertices(); ++node) {
        if (instance.getComputationalDag().in_degree(node) > 0) {
            nodes_in_part[assignment_to_parts[node]].insert(node);
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
    for (unsigned part = 0; part < nr_parts; ++part) {
        Graph_t dag;
        create_induced_subgraph(instance.getComputationalDag(), dag, nodes_in_part[part], extra_sources[part]);
        subDags.push_back(dag);

        // set source nodes to a new type, so that they are compatible with any processor
        unsigned artificial_type_for_sources = subDags.back().num_vertex_types();
        for (vertex_idx node_idx = 0; node_idx < extra_sources[part].size(); ++node_idx) {
            subDags.back().set_vertex_type(node_idx, artificial_type_for_sources);
        }
    }

    std::vector<unsigned> isomorphicTo(nr_parts, UINT_MAX);

    std::cout << "Number of parts: " << nr_parts << std::endl;

    for (unsigned part = 0; part < nr_parts; ++part) {
        for (unsigned other_part = part + 1; other_part < nr_parts; ++other_part) {
            if (isomorphicTo[other_part] < UINT_MAX) {
                continue;
            }

            bool isomorphic = true;
            if (!checkOrderedIsomorphism(subDags[part], subDags[other_part])) {
                continue;
            }

            std::vector<unsigned> proc_assigned_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
            std::vector<unsigned> other_proc_assigned_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
            for (unsigned proc : processors_to_parts[part]) {
                ++proc_assigned_per_type[instance.getArchitecture().processorType(proc)];
            }
            for (unsigned proc : processors_to_parts[other_part]) {
                ++other_proc_assigned_per_type[instance.getArchitecture().processorType(proc)];
            }

            for (unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type) {
                if (proc_assigned_per_type[proc_type] != other_proc_assigned_per_type[proc_type]) {
                    isomorphic = false;
                }
            }

            if (!isomorphic) {
                continue;
            }

            isomorphicTo[other_part] = part;
            std::cout << "Part " << other_part << " is isomorphic to " << part << std::endl;
        }
    }

    // PART 3: solve a small ILP for each part
    std::vector<std::set<vertex_idx>> in_fast_mem(instance.numberOfProcessors());
    std::vector<PebblingSchedule<Graph_t>> pebbling(nr_parts);
    std::vector<BspArchitecture<Graph_t>> subArch(nr_parts);
    std::vector<BspInstance<Graph_t>> subInstance(nr_parts);

    // to handle the initial memory content for isomorphic parts
    std::vector<std::vector<std::set<vertex_idx>>> has_reds_in_beginning(
        nr_parts, std::vector<std::set<vertex_idx>>(instance.numberOfProcessors()));

    for (unsigned part = 0; part < nr_parts; ++part) {
        std::cout << "part " << part << std::endl;

        // set up sub-DAG
        Graph_t &subDag = subDags[part];
        std::map<vertex_idx, vertex_idx> local_id;
        vertex_idx node_idx = 0;
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

        std::set<vertex_idx> needs_blue_at_end;
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
        unsigned proc_index = 0;
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
            has_reds_in_beginning[part] = has_reds_in_beginning[isomorphicTo[part]];
            continue;
        }

        // set node-processor compatibility matrix
        std::vector<std::vector<bool>> comp_matrix = instance.getNodeProcessorCompatibilityMatrix();
        comp_matrix.emplace_back(instance.getArchitecture().getNumberOfProcessorTypes(), true);
        subInstance[part] = BspInstance(subDag, subArch[part], comp_matrix);

        // currently we only allow the input laoding scenario - the case where this is false is unmaintained/untested
        bool need_to_load_inputs = true;

        // keep in fast memory what's relevant, remove the rest
        for (unsigned proc = 0; proc < processors_to_parts[part].size(); ++proc) {
            has_reds_in_beginning[part][proc].clear();
            std::set<vertex_idx> new_content_fast_mem;
            for (vertex_idx node : in_fast_mem[original_proc_id[part][proc]]) {
                if (local_id.find(node) != local_id.end()) {
                    has_reds_in_beginning[part][proc].insert(local_id[node]);
                    new_content_fast_mem.insert(node);
                }
            }

            in_fast_mem[original_proc_id[part][proc]] = new_content_fast_mem;
        }

        // heuristic solution for baseline
        PebblingSchedule<Graph_t> heuristic_pebbling;
        GreedyBspScheduler<Graph_t> greedy_scheduler;
        BspSchedule<Graph_t> bsp_heuristic(subInstance[part]);
        greedy_scheduler.computeSchedule(bsp_heuristic);

        std::set<vertex_idx> extra_source_ids;
        for (vertex_idx idx = 0; idx < extra_sources[part].size(); ++idx) {
            extra_source_ids.insert(idx);
        }

        heuristic_pebbling.setNeedToLoadInputs(true);
        heuristic_pebbling.SetExternalSources(extra_source_ids);
        heuristic_pebbling.SetNeedsBlueAtEnd(needs_blue_at_end);
        heuristic_pebbling.SetHasRedInBeginning(has_reds_in_beginning[part]);
        heuristic_pebbling.ConvertFromBsp(bsp_heuristic, PebblingSchedule<Graph_t>::CACHE_EVICTION_STRATEGY::FORESIGHT);

        heuristic_pebbling.removeEvictStepsFromEnd();
        pebbling[part] = heuristic_pebbling;
        cost_type heuristicCost = asynchronous ? heuristic_pebbling.computeAsynchronousCost() : heuristic_pebbling.computeCost();

        if (!heuristic_pebbling.isValid()) {
            std::cout << "ERROR: Pebbling heuristic INVALID!" << std::endl;
        }

        // solution with subILP
        MultiProcessorPebbling<Graph_t> mpp;
        mpp.setVerbose(verbose);
        mpp.setTimeLimitSeconds(time_seconds_for_subILPs);
        mpp.setMaxTime(2 * maxPartitionSize);    // just a heuristic choice, does not guarantee feasibility!
        mpp.setNeedsBlueAtEnd(needs_blue_at_end);
        mpp.setNeedToLoadInputs(need_to_load_inputs);
        mpp.setHasRedInBeginning(has_reds_in_beginning[part]);

        PebblingSchedule<Graph_t> pebblingILP(subInstance[part]);
        RETURN_STATUS status = mpp.computePebblingWithInitialSolution(heuristic_pebbling, pebblingILP, asynchronous);
        if (status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND) {
            if (!pebblingILP.isValid()) {
                std::cout << "ERROR: Pebbling ILP INVALID!" << std::endl;
            }

            pebblingILP.removeEvictStepsFromEnd();
            cost_type ILP_cost = asynchronous ? pebblingILP.computeAsynchronousCost() : pebblingILP.computeCost();
            if (ILP_cost < heuristicCost) {
                pebbling[part] = pebblingILP;
                std::cout << "ILP chosen instead of greedy. (" << ILP_cost << " < " << heuristicCost << ")" << std::endl;
            } else {
                std::cout << "Greedy chosen instead of ILP. (" << heuristicCost << " < " << ILP_cost << ")" << std::endl;
            }

            // save fast memory content for next subproblem
            std::vector<std::set<vertex_idx>> fast_mem_content_at_end = pebbling[part].getMemContentAtEnd();
            for (unsigned proc = 0; proc < processors_to_parts[part].size(); ++proc) {
                in_fast_mem[original_proc_id[part][proc]].clear();
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

template <typename Graph_t>
Graph_t PebblingPartialILP<Graph_t>::contractByPartition(const BspInstance<Graph_t> &instance,
                                                         const std::vector<unsigned> &node_to_part_assignment) {
    const auto &G = instance.getComputationalDag();

    part_and_nodetype_to_new_index.clear();

    unsigned nr_new_nodes = 0;
    for (vertex_idx node = 0; node < instance.numberOfVertices(); ++node) {
        if (part_and_nodetype_to_new_index.find({node_to_part_assignment[node], G.vertex_type(node)})
            == part_and_nodetype_to_new_index.end()) {
            part_and_nodetype_to_new_index[{node_to_part_assignment[node], G.vertex_type(node)}] = nr_new_nodes;
            ++nr_new_nodes;
        }
    }

    Graph_t contracted;
    for (vertex_idx node = 0; node < nr_new_nodes; ++node) {
        contracted.add_vertex(0, 0, 0);
    }

    std::set<std::pair<vertex_idx, vertex_idx>> edges;

    for (vertex_idx node = 0; node < instance.numberOfVertices(); ++node) {
        vertex_idx node_new_index = part_and_nodetype_to_new_index[{node_to_part_assignment[node], G.vertex_type(node)}];
        for (const vertex_idx &succ : instance.getComputationalDag().children(node)) {
            if (node_to_part_assignment[node] != node_to_part_assignment[succ]) {
                edges.emplace(node_new_index, part_and_nodetype_to_new_index[{node_to_part_assignment[succ], G.vertex_type(succ)}]);
            }
        }

        contracted.set_vertex_work_weight(node_new_index,
                                          contracted.vertex_work_weight(node_new_index) + G.vertex_work_weight(node));
        contracted.set_vertex_comm_weight(node_new_index,
                                          contracted.vertex_comm_weight(node_new_index) + G.vertex_comm_weight(node));
        contracted.set_vertex_mem_weight(node_new_index, contracted.vertex_mem_weight(node_new_index) + G.vertex_mem_weight(node));
        contracted.set_vertex_type(node_new_index, G.vertex_type(node));
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

template <typename Graph_t>
RETURN_STATUS PebblingPartialILP<Graph_t>::computeSchedule(BspSchedule<Graph_t> &) {
    return RETURN_STATUS::ERROR;
}

}    // namespace osp
