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

#include "osp/bsp/model/BspScheduleRecomp.hpp"

namespace osp {

/**
 * @brief The GreedyReccomputer class applies a greedy algorithm to remove some of the communciation steps in
 * a BspSchedule by recomputation steps if this decreases the cost.
 */
template <typename GraphT>
class GreedyRecomputer {
    static_assert(IsComputationalDagV<GraphT>, "GreedyRecomputer can only be used with computational DAGs.");

  private:
    using vertex_idx = VertexIdxT<GraphT>;
    using cost_type = VWorkwT<GraphT>;
    using KeyTriple = std::tuple<VertexIdxT<GraphT>, unsigned int, unsigned int>;

    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "GreedyRecomputer requires work and comm. weights to have the same type.");

  public:
    /**
     * @brief Default destructor for GreedyRecomputer.
     */
    virtual ~GreedyRecomputer() = default;

    ReturnStatus ComputeRecompSchedule(BspScheduleCS<GraphT> &initialSchedule, BspScheduleRecomp<GraphT> &outSchedule) const;
};

template <typename GraphT>
ReturnStatus GreedyRecomputer<GraphT>::ComputeRecompSchedule(BspScheduleCS<GraphT> &initialSchedule,
                                                             BspScheduleRecomp<GraphT> &outSchedule) const {
    const vertex_idx &n = initialSchedule.GetInstance().NumberOfVertices();
    const unsigned &p = initialSchedule.GetInstance().NumberOfProcessors();
    const unsigned &s = initialSchedule.NumberOfSupersteps();
    const GraphT &g = initialSchedule.GetInstance().GetComputationalDag();

    outSchedule = BspScheduleRecomp<GraphT>(initialSchedule.GetInstance());
    outSchedule.setNumberOfSupersteps(initialSchedule.NumberOfSupersteps());

    // Initialize required data structures
    std::vector<std::vector<cost_type>> workCost(P, std::vector<cost_type>(S, 0)), send_cost(P, std::vector<cost_type>(S, 0)),
        rec_cost(P, std::vector<cost_type>(S, 0));

    std::vector<std::vector<unsigned>> firstComputable(n, std::vector<unsigned>(p, 0U)),
        firstPresent(n, std::vector<unsigned>(p, std::numeric_limits<unsigned>::max()));

    std::vector<std::vector<std::multiset<unsigned>>> neededOnProc(n, std::vector<std::multiset<unsigned>>(p, {s}));

    std::vector<cost_type> maxWork(s, 0), max_comm(S, 0);

    std::vector<std::set<KeyTriple>> commSteps(s);

    for (vertex_idx node = 0; node < N; ++node) {
        const unsigned &proc = initialSchedule.assignedProcessor(node);
        const unsigned &step = initialSchedule.assignedSuperstep(node);

        workCost[proc][step] += g.VertexWorkWeight(node);
        firstPresent[node][proc] = std::min(firstPresent[node][proc], step);
        for (vertex_idx pred : G.Parents(node)) {
            needed_on_proc[pred][proc].insert(step);
        }

        outSchedule.assignments(node).emplace_back(proc, step);
    }
    for (const std::pair<KeyTriple, unsigned> item : initial_schedule.getCommunicationSchedule()) {
        const vertex_idx &node = std::get<0>(item.first);
        const unsigned &from_proc = std::get<1>(item.first);
        const unsigned &to_proc = std::get<2>(item.first);
        const unsigned &step = item.second;
        send_cost[from_proc][step]
            += G.VertexCommWeight(node) * initial_schedule.GetInstance().GetArchitecture().communicationCosts(from_proc, to_proc);
        rec_cost[to_proc][step]
            += G.VertexCommWeight(node) * initial_schedule.GetInstance().GetArchitecture().communicationCosts(from_proc, to_proc);

        comm_steps[step].emplace(item.first);
        needed_on_proc[node][from_proc].insert(step);
        first_present[node][to_proc] = std::min(first_present[node][to_proc], step + 1);
    }
    for (unsigned step = 0; step < s; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            maxWork[step] = std::max(max_work[step], work_cost[proc][step]);
            max_comm[step] = std::max(max_comm[step], send_cost[proc][step]);
            max_comm[step] = std::max(max_comm[step], rec_cost[proc][step]);
        }
    }

    for (vertex_idx node = 0; node < N; ++node) {
        for (const vertex_idx &pred : G.Parents(node)) {
            for (unsigned proc = 0; proc < P; ++proc) {
                first_computable[node][proc] = std::max(first_computable[node][proc], first_present[pred][proc]);
            }
        }
    }

    // Find improvement steps
    bool stillImproved = true;
    while (stillImproved) {
        stillImproved = false;

        for (unsigned step = 0; step < s; ++step) {
            std::vector<KeyTriple> toErase;
            for (const KeyTriple &entry : comm_steps[step]) {
                const vertex_idx &node = std::get<0>(entry);
                const unsigned &from_proc = std::get<1>(entry);
                const unsigned &to_proc = std::get<2>(entry);

                // check how much comm cost we save by removing comm schedule entry
                cost_type comm_induced = G.VertexCommWeight(node)
                                         * initial_schedule.GetInstance().GetArchitecture().communicationCosts(from_proc, to_proc);

                cost_type new_max_comm = 0;
                for (unsigned proc = 0; proc < P; ++proc) {
                    if (proc == from_proc) {
                        new_max_comm = std::max(new_max_comm, send_cost[proc][step] - comm_induced);
                    } else {
                        new_max_comm = std::max(new_max_comm, send_cost[proc][step]);
                    }
                    if (proc == to_proc) {
                        new_max_comm = std::max(new_max_comm, rec_cost[proc][step] - comm_induced);
                    } else {
                        new_max_comm = std::max(new_max_comm, rec_cost[proc][step]);
                    }
                }
                if (new_max_comm == max_comm[step]) {
                    continue;
                }

                if (!initial_schedule.GetInstance().isCompatible(node, to_proc)) {
                    continue;
                }

                cost_type decrease = max_comm[step] - new_max_comm;
                if (max_comm[step] > 0 && new_max_comm == 0) {
                    decrease += initial_schedule.GetInstance().GetArchitecture().SynchronisationCosts();
                }

                // check how much it would increase the work cost instead
                unsigned best_step = S;
                cost_type smallest_increase = std::numeric_limits<cost_type>::max();
                for (unsigned comp_step = first_computable[node][to_proc]; comp_step <= *needed_on_proc[node][to_proc].begin();
                     ++comp_step) {
                    cost_type increase = work_cost[to_proc][comp_step] + G.VertexWorkWeight(node) > max_work[comp_step]
                                             ? work_cost[to_proc][comp_step] + G.VertexWorkWeight(node) - max_work[comp_step]
                                             : 0;

                    if (increase < smallest_increase) {
                        best_step = comp_step;
                        smallest_increase = increase;
                    }
                }

                // check if this modification is beneficial
                if (best_step == S || smallest_increase > decrease) {
                    continue;
                }

                // execute the modification
                to_erase.emplace_back(entry);
                out_schedule.assignments(node).emplace_back(to_proc, best_step);

                send_cost[from_proc][step] -= comm_induced;
                rec_cost[to_proc][step] -= comm_induced;
                max_comm[step] = new_max_comm;

                work_cost[to_proc][best_step] += G.VertexWorkWeight(node);
                max_work[best_step] += smallest_increase;

                // update movability bounds
                for (const vertex_idx &pred : G.Parents(node)) {
                    needed_on_proc[pred][to_proc].insert(best_step);
                }

                needed_on_proc[node][from_proc].erase(needed_on_proc[node][from_proc].lower_bound(step));

                first_present[node][to_proc] = best_step;
                for (const vertex_idx &succ : G.Children(node)) {
                    for (const vertex_idx &pred : G.Parents(node)) {
                        first_computable[succ][to_proc] = std::max(first_computable[succ][to_proc], first_present[pred][to_proc]);
                    }
                }

                still_improved = true;
            }
            for (const KeyTriple &entry : to_erase) {
                comm_steps[step].erase(entry);
            }
        }
    }

    for (unsigned step = 0; step < s; ++step) {
        for (const KeyTriple &entry : comm_steps[step]) {
            out_schedule.getCommunicationSchedule().emplace(entry, step);
        }
    }

    outSchedule.mergeSupersteps();

    return ReturnStatus::OSP_SUCCESS;
}

}    // namespace osp
