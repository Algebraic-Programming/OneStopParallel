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
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

template <typename GraphT>
class SubproblemMultiScheduling : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<Graph_t>, "PebblingSchedule can only be used with computational DAGs.");

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using commweight_type = v_commw_t<Graph_t>;
    using workweight_type = VWorkwT<Graph_t>;

    std::vector<vertex_idx> lastNodeOnProc_;
    std::vector<std::vector<vertex_idx>> procTaskLists_;
    std::vector<workweight_type> longestOutgoingPath_;

  public:
    SubproblemMultiScheduling() {}

    virtual ~SubproblemMultiScheduling() = default;

    RETURN_STATUS ComputeMultiSchedule(const BspInstance<GraphT> &instance, std::vector<std::set<unsigned>> &processorsToNode);

    std::vector<std::pair<vertex_idx, unsigned>> MakeAssignment(const BspInstance<GraphT> &instance,
                                                                const std::set<std::pair<unsigned, vertex_idx>> &nodesAvailable,
                                                                const std::set<unsigned> &procsAvailable) const;

    std::vector<workweight_type> static GetLongestPath(const GraphT &graph);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override;

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "SubproblemMultiScheduling"; }

    inline const std::vector<std::vector<unsigned>> &GetProcTaskLists() const { return proc_task_lists; }
};

// currently duplicated from BSP locking scheduler's code
template <typename GraphT>
std::vector<VWorkwT<Graph_t>> SubproblemMultiScheduling<GraphT>::GetLongestPath(const GraphT &graph) {
    std::vector<workweight_type> longestPath(graph.NumVertices(), 0);

    std::vector<vertex_idx> topOrder = GetTopOrder(graph);

    for (auto rIter = top_order.rbegin(); rIter != top_order.crend(); r_iter++) {
        longestPath[*r_iter] = graph.VertexWorkWeight(*r_iter);
        if (graph.OutDegree(*r_iter) > 0) {
            workweight_type max = 0;
            for (const auto &child : graph.Children(*r_iter)) {
                if (max <= longest_path[child]) {
                    max = longest_path[child];
                }
            }
            longestPath[*r_iter] += max;
        }
    }

    return longest_path;
}

template <typename GraphT>
RETURN_STATUS SubproblemMultiScheduling<GraphT>::ComputeMultiSchedule(const BspInstance<GraphT> &instance,
                                                                      std::vector<std::set<unsigned>> &processorsToNode) {
    const unsigned &n = static_cast<unsigned>(instance.NumberOfVertices());
    const unsigned &p = instance.NumberOfProcessors();
    const auto &g = instance.GetComputationalDag();

    processorsToNode.clear();
    processorsToNode.resize(n);

    proc_task_lists.clear();
    proc_task_lists.resize(P);

    last_node_on_proc.clear();
    last_node_on_proc.resize(P, UINT_MAX);

    longest_outgoing_path = get_longest_path(G);

    std::set<std::pair<unsigned, vertex_idx>> readySet;

    std::vector<unsigned> nrPredecRemain(n);
    for (vertex_idx node = 0; node < n; node++) {
        nrPredecRemain[node] = static_cast<unsigned>(g.in_degree(node));
        if (g.in_degree(node) == 0) {
            readySet.emplace(-longest_outgoing_path[node], node);
        }
    }

    std::set<unsigned> freeProcs;
    for (unsigned proc = 0; proc < p; ++proc) {
        freeProcs.insert(proc);
    }

    std::vector<double> nodeFinishTime(n, 0);

    std::set<std::pair<double, vertex_idx>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<unsigned>::max());

    while (!readySet.empty() || !finishTimes.empty()) {
        const double time = finishTimes.begin()->first;

        // Find new ready jobs
        while (!finishTimes.empty() && fabs(finishTimes.begin()->first - time) < 0.0001) {
            const vertex_idx node = finishTimes.begin()->second;
            finishTimes.erase(finishTimes.begin());

            if (node != std::numeric_limits<unsigned>::max()) {
                for (const vertex_idx &succ : G.Children(node)) {
                    nrPredecRemain[succ]--;
                    if (nrPredecRemain[succ] == 0) {
                        readySet.emplace(-longest_outgoing_path[succ], succ);
                    }
                }
                for (unsigned proc : processors_to_node[node]) {
                    free_procs.insert(proc);
                }
            }
        }

        // Assign new jobs to idle processors

        // first assign free processors to ready nodes
        std::vector<std::pair<vertex_idx, unsigned>> newAssingments = makeAssignment(instance, readySet, free_procs);

        for (auto entry : new_assingments) {
            vertex_idx node = entry.first;
            unsigned proc = entry.second;

            processors_to_node[node].insert(proc);
            proc_task_lists[proc].push_back(node);
            finishTimes.emplace(time + G.VertexWorkWeight(node), node);
            node_finish_time[node] = time + G.VertexWorkWeight(node);
            last_node_on_proc[proc] = node;
            free_procs.erase(proc);
            readySet.erase({-longest_outgoing_path[node], node});
        }

        // assign remaining free processors to already started nodes, if it helps
        decltype(finishTimes.rbegin()) itr = finishTimes.rbegin();
        while (!free_procs.empty() && itr != finishTimes.rend()) {
            double lastFinishTime = itr->first;

            decltype(finishTimes.rbegin()) itrLatest = itr;
            std::set<std::pair<workweight_type, vertex_idx>> possibleNodes;
            while (itrLatest != finishTimes.rend() && itr_latest->first + 0.0001 > lastFinishTime) {
                vertex_idx node = itr_latest->second;
                double newFinishTime = time
                                       + static_cast<double>(g.VertexWorkWeight(node))
                                             / (static_cast<double>(processors_to_node[node].size()) + 1);
                if (newFinishTime + 0.0001 < itr_latest->first) {
                    possible_nodes.emplace(-longest_outgoing_path[node], node);
                }

                ++itr_latest;
            }
            new_assingments = makeAssignment(instance, possible_nodes, free_procs);
            for (auto entry : new_assingments) {
                vertex_idx node = entry.first;
                unsigned proc = entry.second;

                processors_to_node[node].insert(proc);
                proc_task_lists[proc].push_back(node);
                finishTimes.erase({node_finish_time[node], node});
                double new_finish_time
                    = time + static_cast<double>(G.VertexWorkWeight(node)) / (static_cast<double>(processors_to_node[node].size()));
                finishTimes.emplace(new_finish_time, node);
                node_finish_time[node] = new_finish_time;
                last_node_on_proc[proc] = node;
                free_procs.erase(proc);
            }
            if (newAssingments.empty()) {
                itr = itr_latest;
            }
        }
    }

    return RETURN_STATUS::OSP_SUCCESS;
}

template <typename GraphT>
std::vector<std::pair<vertex_idx_t<Graph_t>, unsigned>> SubproblemMultiScheduling<GraphT>::MakeAssignment(
    const BspInstance<GraphT> &instance,
    const std::set<std::pair<unsigned, vertex_idx>> &nodesAvailable,
    const std::set<unsigned> &procsAvailable) const {
    std::vector<std::pair<vertex_idx, unsigned>> assignments;
    if (nodesAvailable.empty() || procs_available.empty()) {
        return assignments;
    }

    std::set<vertex_idx> assignedNodes;
    std::vector<bool> assignedProcs(instance.NumberOfProcessors(), false);

    for (unsigned proc : procs_available) {
        if (last_node_on_proc[proc] == UINT_MAX) {
            continue;
        }

        for (const auto &succ : instance.GetComputationalDag().Children(last_node_on_proc[proc])) {
            if (nodes_available.find({-longest_outgoing_path[succ], succ}) != nodes_available.end()
                && instance.isCompatible(succ, proc) && assigned_nodes.find(succ) == assigned_nodes.end()) {
                assignments.emplace_back(succ, proc);
                assigned_nodes.insert(succ);
                assigned_procs[proc] = true;
                break;
            }
        }
    }

    for (unsigned proc : procs_available) {
        if (!assigned_procs[proc]) {
            for (auto itr = nodes_available.begin(); itr != nodes_available.end(); ++itr) {
                vertex_idx node = itr->second;
                if (instance.isCompatible(node, proc) && assigned_nodes.find(node) == assigned_nodes.end()) {
                    assignments.emplace_back(node, proc);
                    assigned_nodes.insert(node);
                    break;
                }
            }
        }
    }

    return assignments;
}

template <typename GraphT>
RETURN_STATUS SubproblemMultiScheduling<GraphT>::ComputeSchedule(BspSchedule<GraphT> &) {
    return RETURN_STATUS::ERROR;
}

}    // namespace osp
