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

#include "osp/bsp/model/BspScheduleCS.hpp"
#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/model/MaxBspScheduleCS.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

template <typename GraphT>
class GreedyBspToMaxBspConverter {
    static_assert(IsComputationalDagV<Graph_t>, "GreedyBspToMaxBspConverter can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>,
                  "GreedyBspToMaxBspConverter requires work and comm. weights to have the same type.");

  protected:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = v_workw_t<Graph_t>;
    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;

    double latencyCoefficient_ = 1.25;
    double decayFactor_ = 0.5;

    std::vector<std::vector<std::deque<vertex_idx_t<Graph_t>>>> CreateSuperstepLists(const BspScheduleCS<GraphT> &schedule,
                                                                                     std::vector<double> &priorities) const;

  public:
    MaxBspSchedule<GraphT> Convert(const BspSchedule<GraphT> &schedule) const;
    MaxBspScheduleCS<GraphT> Convert(const BspScheduleCS<GraphT> &schedule) const;
};

template <typename GraphT>
MaxBspSchedule<GraphT> GreedyBspToMaxBspConverter<GraphT>::Convert(const BspSchedule<GraphT> &schedule) const {
    BspScheduleCS<GraphT> scheduleCs(schedule);
    return Convert(scheduleCs);
}

template <typename GraphT>
MaxBspScheduleCS<GraphT> GreedyBspToMaxBspConverter<GraphT>::Convert(const BspScheduleCS<GraphT> &schedule) const {
    const GraphT &dag = schedule.getInstance().getComputationalDag();

    // Initialize data structures
    std::vector<double> priorities;
    std::vector<std::vector<std::deque<vertex_idx>>> procList = createSuperstepLists(schedule, priorities);
    std::vector<std::vector<cost_type>> workRemainingProcSuperstep(schedule.getInstance().numberOfProcessors(),
                                                                   std::vector<cost_type>(schedule.numberOfSupersteps(), 0));
    std::vector<vertex_idx> nodesRemainingSuperstep(schedule.numberOfSupersteps(), 0);

    MaxBspScheduleCS<GraphT> scheduleMax(schedule.getInstance());
    for (vertex_idx node = 0; node < schedule.getInstance().numberOfVertices(); node++) {
        workRemainingProcSuperstep[schedule.assignedProcessor(node)][schedule.assignedSuperstep(node)]
            += dag.vertex_work_weight(node);
        ++nodes_remaining_superstep[schedule.assignedSuperstep(node)];
        scheduleMax.setAssignedProcessor(node, schedule.assignedProcessor(node));
    }

    std::vector<std::vector<cost_type>> sendCommRemainingProcSuperstep(schedule.getInstance().numberOfProcessors(),
                                                                       std::vector<cost_type>(schedule.numberOfSupersteps(), 0));
    std::vector<std::vector<cost_type>> recCommRemainingProcSuperstep(schedule.getInstance().numberOfProcessors(),
                                                                      std::vector<cost_type>(schedule.numberOfSupersteps(), 0));

    std::vector<std::set<std::pair<KeyTriple, unsigned>>> freeCommStepsForSuperstep(schedule.numberOfSupersteps());
    std::vector<std::vector<std::pair<KeyTriple, unsigned>>> dependentCommStepsForNode(schedule.getInstance().numberOfVertices());
    for (auto const &[key, val] : schedule.getCommunicationSchedule()) {
        if (schedule.assignedSuperstep(std::get<0>(key)) == val) {
            dependentCommStepsForNode[std::get<0>(key)].emplace_back(key, val);

            cost_type commCost = dag.vertex_comm_weight(std::get<0>(key))
                                 * schedule.getInstance().getArchitecture().sendCosts(std::get<1>(key), std::get<2>(key));
            sendCommRemainingProcSuperstep[std::get<1>(key)][val] += comm_cost;
            recCommRemainingProcSuperstep[std::get<2>(key)][val] += comm_cost;
        } else {
            freeCommStepsForSuperstep[val].emplace(key, val);
        }
    }

    // Iterate through supersteps
    unsigned currentStep = 0;
    for (unsigned step = 0; step < schedule.numberOfSupersteps(); ++step) {
        std::vector<cost_type> workDoneOnProc(schedule.getInstance().numberOfProcessors(), 0);
        cost_type maxWorkDone = 0;
        std::vector<std::pair<KeyTriple, unsigned>> newlyFreedCommSteps;
        std::vector<cost_type> sendSumOfNewlyFreeOnProc(schedule.getInstance().numberOfProcessors(), 0),
            rec_sum_of_newly_free_on_proc(schedule.getInstance().numberOfProcessors(), 0);

        std::vector<std::pair<KeyTriple, unsigned>> commInCurrentStep;

        std::vector<cost_type> sendOnProc(schedule.getInstance().numberOfProcessors(), 0),
            rec_on_proc(schedule.getInstance().numberOfProcessors(), 0);
        bool emptySuperstep = (nodesRemainingSuperstep[step] == 0);

        while (nodesRemainingSuperstep[step] > 0) {
            // I. Select the next node (from any proc) with highest priority
            unsigned chosenProc = schedule.getInstance().numberOfProcessors();
            double bestPrio = std::numeric_limits<double>::max();

            for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
                if (!proc_list[proc][step].empty()
                    && (chosenProc == schedule.getInstance().numberOfProcessors()
                        || priorities[proc_list[proc][step].front()] < bestPrio)) {
                    chosenProc = proc;
                    bestPrio = priorities[proc_list[proc][step].front()];
                }
            }
            if (chosenProc == schedule.getInstance().numberOfProcessors()) {
                break;
            }

            vertex_idx chosenNode = proc_list[chosenProc][step].front();
            procList[chosenProc][step].pop_front();
            workDoneOnProc[chosenProc] += dag.vertex_work_weight(chosen_node);
            workRemainingProcSuperstep[chosenProc][step] -= dag.vertex_work_weight(chosen_node);
            maxWorkDone = std::max(max_work_done, work_done_on_proc[chosenProc]);
            scheduleMax.setAssignedSuperstep(chosen_node, currentStep);
            --nodes_remaining_superstep[step];
            for (const std::pair<KeyTriple, unsigned> &entry : dependent_comm_steps_for_node[chosen_node]) {
                newly_freed_comm_steps.push_back(entry);
                cost_type comm_cost
                    = dag.vertex_comm_weight(chosen_node)
                      * schedule.getInstance().getArchitecture().sendCosts(std::get<1>(entry.first), std::get<2>(entry.first));
                send_sum_of_newly_free_on_proc[std::get<1>(entry.first)] += comm_cost;
                rec_sum_of_newly_free_on_proc[std::get<2>(entry.first)] += comm_cost;
            }

            // II. Add nodes on all other processors if this doesn't increase work cost
            for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
                if (proc == chosenProc) {
                    continue;
                }
                while (!proc_list[proc][step].empty()
                       && work_done_on_proc[proc] + dag.vertex_work_weight(proc_list[proc][step].front()) <= max_work_done) {
                    vertex_idx node = proc_list[proc][step].front();
                    procList[proc][step].pop_front();
                    workDoneOnProc[proc] += dag.vertex_work_weight(node);
                    workRemainingProcSuperstep[proc][step] -= dag.vertex_work_weight(node);
                    scheduleMax.setAssignedSuperstep(node, currentStep);
                    --nodes_remaining_superstep[step];
                    for (const std::pair<KeyTriple, unsigned> &entry : dependent_comm_steps_for_node[node]) {
                        newly_freed_comm_steps.push_back(entry);
                        cost_type comm_cost = dag.vertex_comm_weight(node)
                                              * schedule.getInstance().getArchitecture().sendCosts(std::get<1>(entry.first),
                                                                                                   std::get<2>(entry.first));
                        send_sum_of_newly_free_on_proc[std::get<1>(entry.first)] += comm_cost;
                        rec_sum_of_newly_free_on_proc[std::get<2>(entry.first)] += comm_cost;
                    }
                }
            }

            // III. Add communication steps that are already available
            for (auto itr = free_comm_steps_for_superstep[step].begin(); itr != free_comm_steps_for_superstep[step].end();) {
                if (send_on_proc[std::get<1>(itr->first)] < max_work_done && rec_on_proc[std::get<2>(itr->first)] < max_work_done) {
                    cost_type commCost
                        = dag.vertex_comm_weight(std::get<0>(itr->first))
                          * schedule.getInstance().getArchitecture().sendCosts(std::get<1>(itr->first), std::get<2>(itr->first))
                          * schedule.getInstance().getArchitecture().communicationCosts();
                    sendOnProc[std::get<1>(itr->first)] += comm_cost;
                    rec_on_proc[std::get<2>(itr->first)] += comm_cost;
                    if (currentStep - 1 >= scheduleMax.numberOfSupersteps()) {
                        scheduleMax.setNumberOfSupersteps(currentStep);
                    }
                    scheduleMax.addCommunicationScheduleEntry(itr->first, currentStep - 1);
                    commInCurrentStep.emplace_back(*itr);
                    freeCommStepsForSuperstep[step].erase(itr++);
                } else {
                    ++itr;
                }
            }

            // IV. Decide whether to split superstep here
            if (!free_comm_steps_for_superstep[step].empty() || nodes_remaining_superstep[step] == 0) {
                continue;
            }

            cost_type maxWorkRemaining = 0, max_comm_remaining = 0, comm_after_reduction = 0;
            for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
                maxWorkRemaining = std::max(max_work_remaining, work_remaining_proc_superstep[proc][step]);
                max_comm_remaining = std::max(max_comm_remaining, send_comm_remaining_proc_superstep[proc][step]);
                max_comm_remaining = std::max(max_comm_remaining, rec_comm_remaining_proc_superstep[proc][step]);
                comm_after_reduction = std::max(
                    comm_after_reduction, send_comm_remaining_proc_superstep[proc][step] - send_sum_of_newly_free_on_proc[proc]);
                comm_after_reduction = std::max(
                    comm_after_reduction, rec_comm_remaining_proc_superstep[proc][step] - rec_sum_of_newly_free_on_proc[proc]);
            }
            cost_type commReduction
                = (max_comm_remaining - comm_after_reduction) * schedule.getInstance().getArchitecture().communicationCosts();

            cost_type gain = std::min(comm_reduction, max_work_remaining);
            if (gain > 0
                && static_cast<double>(gain) >= static_cast<double>(schedule.getInstance().getArchitecture().synchronisationCosts())
                                                    * latencyCoefficient_) {
                // Split superstep
                for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
                    workDoneOnProc[proc] = 0;
                    sendOnProc[proc] = 0;
                    rec_on_proc[proc] = 0;
                    sendSumOfNewlyFreeOnProc[proc] = 0;
                    rec_sum_of_newly_free_on_proc[proc] = 0;
                }
                maxWorkDone = 0;
                for (const std::pair<KeyTriple, unsigned> &entry : newly_freed_comm_steps) {
                    free_comm_steps_for_superstep[step].insert(entry);

                    cost_type comm_cost = dag.vertex_comm_weight(std::get<0>(entry.first))
                                          * schedule.getInstance().getArchitecture().sendCosts(std::get<1>(entry.first),
                                                                                               std::get<2>(entry.first));
                    send_comm_remaining_proc_superstep[std::get<1>(entry.first)][step] -= comm_cost;
                    rec_comm_remaining_proc_superstep[std::get<2>(entry.first)][step] -= comm_cost;
                }
                newlyFreedCommSteps.clear();
                commInCurrentStep.clear();
                ++currentStep;
            }
        }

        if (!emptySuperstep) {
            ++currentStep;
        }

        for (const std::pair<KeyTriple, unsigned> &entry : newly_freed_comm_steps) {
            free_comm_steps_for_superstep[step].insert(entry);
        }

        if (freeCommStepsForSuperstep[step].empty()) {
            continue;
        }

        // Handle the remaining communication steps: creating a new superstep afterwards with no work
        cost_type maxCommCurrent = 0;
        for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
            maxCommCurrent = std::max(max_comm_current, send_on_proc[proc]);
            max_comm_current = std::max(max_comm_current, rec_on_proc[proc]);
        }
        sendOnProc.clear();
        sendOnProc.resize(schedule.getInstance().numberOfProcessors(), 0);
        rec_on_proc.clear();
        rec_on_proc.resize(schedule.getInstance().numberOfProcessors(), 0);

        std::set<std::pair<vertex_idx, unsigned>> lateArrivingNodes;
        for (const std::pair<KeyTriple, unsigned> &entry : free_comm_steps_for_superstep[step]) {
            schedule_max.addCommunicationScheduleEntry(entry.first, current_step - 1);
            cost_type comm_cost
                = dag.vertex_comm_weight(std::get<0>(entry.first))
                  * schedule.getInstance().getArchitecture().sendCosts(std::get<1>(entry.first), std::get<2>(entry.first))
                  * schedule.getInstance().getArchitecture().communicationCosts();
            send_on_proc[std::get<1>(entry.first)] += comm_cost;
            rec_on_proc[std::get<2>(entry.first)] += comm_cost;
            late_arriving_nodes.emplace(std::get<0>(entry.first), std::get<2>(entry.first));
        }

        // Edge case - check if it is worth moving all communications from the current superstep to the next one instead (thus
        // saving a sync cost) (for this we need to compute the h-relation-max in the current superstep, the next superstep, and
        // also their union)
        cost_type maxCommAfter = 0;
        for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
            maxCommAfter = std::max(max_comm_after, send_on_proc[proc]);
            max_comm_after = std::max(max_comm_after, rec_on_proc[proc]);
        }

        for (const std::pair<KeyTriple, unsigned> &entry : comm_in_current_step) {
            cost_type comm_cost
                = dag.vertex_comm_weight(std::get<0>(entry.first))
                  * schedule.getInstance().getArchitecture().sendCosts(std::get<1>(entry.first), std::get<2>(entry.first))
                  * schedule.getInstance().getArchitecture().communicationCosts();
            send_on_proc[std::get<1>(entry.first)] += comm_cost;
            rec_on_proc[std::get<2>(entry.first)] += comm_cost;
        }
        cost_type maxCommTogether = 0;
        for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
            maxCommTogether = std::max(max_comm_together, send_on_proc[proc]);
            max_comm_together = std::max(max_comm_together, rec_on_proc[proc]);
        }

        cost_type workLimit = max_comm_after;
        if (maxCommTogether + max_work_done <= max_comm_after + std::max(max_work_done, max_comm_current)
                                                   + schedule.getInstance().getArchitecture().synchronisationCosts()) {
            workLimit = max_comm_together;
            for (const std::pair<KeyTriple, unsigned> &entry : comm_in_current_step) {
                if (current_step - 1 >= schedule_max.numberOfSupersteps()) {
                    schedule_max.setNumberOfSupersteps(current_step);
                }
                schedule_max.addCommunicationScheduleEntry(entry.first, current_step - 1);
                late_arriving_nodes.emplace(std::get<0>(entry.first), std::get<2>(entry.first));
            }
        }

        // Bring computation steps into the extra superstep from the next superstep, if possible,a s long as it does not increase cost
        if (step == schedule.numberOfSupersteps() - 1) {
            continue;
        }

        for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
            cost_type workSoFar = 0;
            std::set<vertex_idx> broughtForward;
            for (vertex_idx node : proc_list[proc][step + 1]) {
                if (work_so_far + dag.vertex_work_weight(node) > work_limit) {
                    continue;
                }

                bool has_dependency = false;

                for (const vertex_idx &parent : dag.parents(node)) {
                    if (schedule.assignedProcessor(node) != schedule.assignedProcessor(parent)
                        && late_arriving_nodes.find(std::make_pair(parent, proc)) != late_arriving_nodes.end()) {
                        has_dependency = true;
                    }

                    if (schedule.assignedProcessor(node) == schedule.assignedProcessor(parent)
                        && schedule.assignedSuperstep(parent) == step + 1
                        && brought_forward.find(parent) == brought_forward.end()) {
                        has_dependency = true;
                    }
                }

                if (has_dependency) {
                    continue;
                }

                brought_forward.insert(node);
                work_so_far += dag.vertex_work_weight(node);
                schedule_max.setAssignedSuperstep(node, current_step);
                work_remaining_proc_superstep[proc][step + 1] -= dag.vertex_work_weight(node);
                --nodes_remaining_superstep[step + 1];

                for (const std::pair<KeyTriple, unsigned> &entry : dependent_comm_steps_for_node[node]) {
                    free_comm_steps_for_superstep[step + 1].insert(entry);
                }
            }

            std::deque<vertex_idx> remaining;
            for (vertex_idx node : proc_list[proc][step + 1]) {
                if (brought_forward.find(node) == brought_forward.end()) {
                    remaining.push_back(node);
                }
            }

            procList[proc][step + 1] = remaining;
        }

        ++currentStep;
    }

    return scheduleMax;
}

// Auxiliary function: creates a separate vectors for each proc-supstep combination, collecting the nodes in a priority-based
// topological order
template <typename GraphT>
std::vector<std::vector<std::deque<vertex_idx_t<Graph_t>>>> GreedyBspToMaxBspConverter<GraphT>::CreateSuperstepLists(
    const BspScheduleCS<GraphT> &schedule, std::vector<double> &priorities) const {
    const GraphT &dag = schedule.getInstance().getComputationalDag();
    std::vector<vertex_idx> topOrder = GetTopOrder(dag);
    priorities.clear();
    priorities.resize(dag.num_vertices());
    std::vector<vertex_idx> localInDegree(dag.num_vertices(), 0);

    // compute for each node the amount of dependent send cost in the same superstep
    std::vector<cost_type> commDependency(dag.num_vertices(), 0);
    for (auto const &[key, val] : schedule.getCommunicationSchedule()) {
        if (schedule.assignedSuperstep(std::get<0>(key)) == val) {
            commDependency[std::get<0>(key)]
                += dag.vertex_comm_weight(std::get<0>(key))
                   * schedule.getInstance().getArchitecture().sendCosts(std::get<1>(key), std::get<2>(key));
        }
    }

    // assign priority to nodes - based on their own work/comm ratio, and that of its successors in the same proc/supstep
    for (auto itr = top_order.rbegin(); itr != top_order.rend(); ++itr) {
        vertex_idx node = *itr;
        double base = static_cast<double>(dag.vertex_work_weight(node));
        if (commDependency[node] > 0) {
            base /= static_cast<double>(2 * comm_dependency[node]);
        }

        double successors = 0;
        unsigned numChildren = 0;
        for (const vertex_idx &child : dag.children(node)) {
            if (schedule.assignedProcessor(node) == schedule.assignedProcessor(child)
                && schedule.assignedSuperstep(node) == schedule.assignedSuperstep(child)) {
                ++num_children;
                successors += priorities[child];
                ++local_in_degree[child];
            }
        }
        if (numChildren > 0) {
            successors = successors * decayFactor_ / static_cast<double>(numChildren);
        }
        priorities[node] = base + successors;
    }

    // create lists for each processor-superstep pair, in a topological order, sorted by priority
    std::vector<std::vector<std::deque<vertex_idx>>> superstepLists(
        schedule.getInstance().numberOfProcessors(), std::vector<std::deque<vertex_idx>>(schedule.numberOfSupersteps()));

    std::set<std::pair<double, vertex_idx>> free;
    for (vertex_idx node = 0; node < schedule.getInstance().numberOfVertices(); node++) {
        if (localInDegree[node] == 0) {
            free.emplace(priorities[node], node);
        }
    }
    while (!free.empty()) {
        vertex_idx node = free.begin()->second;
        free.erase(free.begin());
        superstepLists[schedule.assignedProcessor(node)][schedule.assignedSuperstep(node)].push_back(node);
        for (const vertex_idx &child : dag.children(node)) {
            if (schedule.assignedProcessor(node) == schedule.assignedProcessor(child)
                && schedule.assignedSuperstep(node) == schedule.assignedSuperstep(child)) {
                if (--local_in_degree[child] == 0) {
                    free.emplace(priorities[child], child);
                }
            }
        }
    }

    return superstep_lists;
}

}    // namespace osp
