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

#include <chrono>
#include <climits>
#include <list>
#include <queue>
#include <map>
#include <unordered_set>
#include <string>
#include <vector>

#include "MemoryConstraintModules.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

template<typename weight_t>
struct GrowLocalAutoCores_Params {
    unsigned minSuperstepSize = 20;
    weight_t syncCostMultiplierMinSuperstepWeight = 1;
    weight_t syncCostMultiplierParallelCheck = 4;
};

/**
 * @brief The GreedyBspGrowLocalAutoCores class represents a scheduler that uses a greedy algorithm to compute
 * schedules for BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "GreedyBspGrowLocalAutoCores" in this
 * case.
 */
template<typename Graph_t, typename MemoryConstraint_t = no_memory_constraint>
class GrowLocalAutoCores : public Scheduler<Graph_t> {

  private:
    GrowLocalAutoCores_Params<v_workw_t<Graph_t>> params;

    constexpr static bool use_memory_constraint =
        is_memory_constraint_v<MemoryConstraint_t> or is_memory_constraint_schedule_v<MemoryConstraint_t>;

    static_assert(not use_memory_constraint or std::is_same_v<Graph_t, typename MemoryConstraint_t::Graph_impl_t>,
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

    static_assert(not use_memory_constraint or not (std::is_same_v<MemoryConstraint_t, persistent_transient_memory_constraint<Graph_t>> or std::is_same_v<MemoryConstraint_t, global_memory_constraint<Graph_t>>), 
                  "MemoryConstraint_t must not be persistent_transient_memory_constraint or global_memory_constraint. Not supported in GrowLocalAutoCores.");
               

    MemoryConstraint_t local_memory_constraint;
 

  public:
    /**
     * @brief Default constructor for GreedyBspGrowLocalAutoCores.
     */
    GrowLocalAutoCores(
        GrowLocalAutoCores_Params<v_workw_t<Graph_t>> params_ = GrowLocalAutoCores_Params<v_workw_t<Graph_t>>())
        : params(params_) {}

    /**
     * @brief Default destructor for GreedyBspGrowLocalAutoCores.
     */
    virtual ~GrowLocalAutoCores() = default;

    /**
     * @brief Compute a schedule for the given BspInstance.
     *
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     *
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {

        using vertex_idx = typename Graph_t::vertex_idx;
        const auto &instance = schedule.getInstance();

        for (const auto &v : instance.getComputationalDag().vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            schedule.setAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
        }

        unsigned supstep = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraint_t>) {
            local_memory_constraint.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraint_t>) {
            local_memory_constraint.initialize(schedule, supstep);
        }

        auto &node_to_proc = schedule.assignedProcessors();
        auto &node_to_supstep = schedule.assignedSupersteps();

        const auto N = instance.numberOfVertices();
        const unsigned P = instance.numberOfProcessors();
        const auto &G = instance.getComputationalDag();

        std::unordered_set<vertex_idx> ready;  

        std::vector<vertex_idx> allReady;
        std::vector<std::vector<vertex_idx>> procReady(P);

        std::vector<vertex_idx> predec(N);

        for (const auto &node : G.vertices()) {
            predec[node] = G.in_degree(node);

            if (predec[node] == 0) {
                ready.insert(node);
            }
        }

        std::vector<std::vector<vertex_idx>> new_assignments(P);
        std::vector<std::vector<vertex_idx>> best_new_assignments(P);

        std::vector<vertex_idx> new_ready;
        std::vector<vertex_idx> best_new_ready;

        const v_workw_t<Graph_t> minWeightParallelCheck =
            params.syncCostMultiplierParallelCheck * instance.synchronisationCosts();
        const v_workw_t<Graph_t> minSuperstepWeight =
            params.syncCostMultiplierMinSuperstepWeight * instance.synchronisationCosts();

        double desiredParallelism = static_cast<double>(P);

        vertex_idx total_assigned = 0;
        while (total_assigned < N) {

            unsigned limit = params.minSuperstepSize;
            double best_score = 0;
            double best_parallelism = 0;

            bool continueSuperstepAttempts = true;

            while (continueSuperstepAttempts) {

                for (unsigned p = 0; p < P; p++) {
                    new_assignments[p].clear();
                    procReady[p].clear();
                }

                new_ready.clear();
                allReady.assign(ready.begin(), ready.end());
                std::make_heap(allReady.begin(), allReady.end(), std::greater<vertex_idx>());

                vertex_idx new_total_assigned = 0;
                v_workw_t<Graph_t> weight_limit = 0, total_weight_assigned = 0;

                bool early_memory_break = false;

                // Processor 0
                while (new_assignments[0].size() < limit) {
                    vertex_idx chosen_node = std::numeric_limits<vertex_idx>::max();

                    if constexpr (use_memory_constraint) {
                        if (!procReady[0].empty() && local_memory_constraint.can_add(procReady[0].front(), 0)) {
                            chosen_node = procReady[0].front();
                            std::pop_heap(procReady[0].begin(), procReady[0].end(), std::greater<vertex_idx>());
                            procReady[0].pop_back();
                        } else if (!allReady.empty() && local_memory_constraint.can_add(allReady.front(), 0)) {
                            chosen_node = allReady.front();
                            std::pop_heap(allReady.begin(), allReady.end(), std::greater<vertex_idx>());
                            allReady.pop_back();
                        } else {
                            early_memory_break = true;
                            break;
                        }
                    } else {
                        if (!procReady[0].empty()) {
                            chosen_node = procReady[0].front();
                            std::pop_heap(procReady[0].begin(), procReady[0].end(), std::greater<vertex_idx>());
                            procReady[0].pop_back();
                        } else if (!allReady.empty()) {
                            chosen_node = allReady.front();
                            std::pop_heap(allReady.begin(), allReady.end(), std::greater<vertex_idx>());
                            allReady.pop_back();
                        } else {
                            break;
                        }
                    }

                    new_assignments[0].push_back(chosen_node);
                    node_to_proc[chosen_node] = 0;
                    new_total_assigned++;
                    weight_limit += G.vertex_work_weight(chosen_node);

                    if constexpr (use_memory_constraint) {
                        local_memory_constraint.add(chosen_node, 0);
                    }

                    for (const auto &succ : G.children(chosen_node)) {
                        if (node_to_proc[succ] == std::numeric_limits<unsigned>::max()) {
                            node_to_proc[succ] = 0;
                        } else if (node_to_proc[succ] != 0) {
                            node_to_proc[succ] = P;
                        }

                        predec[succ]--;
                        if (predec[succ] == 0) {
                            new_ready.push_back(succ);

                            if (node_to_proc[succ] == 0) {
                                procReady[0].push_back(succ);
                                std::push_heap(procReady[0].begin(), procReady[0].end(), std::greater<vertex_idx>());
                            }
                        }
                    }
                }

                total_weight_assigned += weight_limit;

                // Processors 1 through P-1
                for (unsigned proc = 1; proc < P; ++proc) {
                    v_workw_t<Graph_t> current_weight_assigned = 0;
                    while (current_weight_assigned < weight_limit) {
                        vertex_idx chosen_node = std::numeric_limits<vertex_idx>::max();

                        if constexpr (use_memory_constraint) {
                            if (!procReady[proc].empty() && local_memory_constraint.can_add(procReady[proc].front(), proc)) {
                                chosen_node = procReady[proc].front();
                                std::pop_heap(procReady[proc].begin(), procReady[proc].end(), std::greater<vertex_idx>());
                                procReady[proc].pop_back();
                            } else if (!allReady.empty() && local_memory_constraint.can_add(allReady.front(), proc)) {
                                chosen_node = allReady.front();
                                std::pop_heap(allReady.begin(), allReady.end(), std::greater<vertex_idx>());
                                allReady.pop_back();
                            } else {
                                early_memory_break = true;
                                break;
                            }
                        } else {
                            if (!procReady[proc].empty()) {
                                chosen_node = procReady[proc].front();
                                std::pop_heap(procReady[proc].begin(), procReady[proc].end(), std::greater<vertex_idx>());
                                procReady[proc].pop_back();
                            } else if (!allReady.empty()) {
                                chosen_node = allReady.front();
                                std::pop_heap(allReady.begin(), allReady.end(), std::greater<vertex_idx>());
                                allReady.pop_back();
                            } else {
                                break;
                            }
                        }

                        new_assignments[proc].push_back(chosen_node);
                        node_to_proc[chosen_node] = proc;
                        new_total_assigned++;
                        current_weight_assigned += G.vertex_work_weight(chosen_node);

                        if constexpr (use_memory_constraint) {
                            local_memory_constraint.add(chosen_node, proc);
                        }

                        for (const auto &succ : G.children(chosen_node)) {
                            if (node_to_proc[succ] == std::numeric_limits<unsigned>::max()) {
                                node_to_proc[succ] = proc;
                            } else if (node_to_proc[succ] != proc) {
                                node_to_proc[succ] = P;
                            }
                            predec[succ]--;
                            if (predec[succ] == 0) {
                                new_ready.push_back(succ);

                                if (node_to_proc[succ] == proc) {
                                    procReady[proc].push_back(succ);
                                    std::push_heap(procReady[proc].begin(), procReady[proc].end(), std::greater<vertex_idx>());
                                }
                            }
                        }
                    }

                    weight_limit = std::max(weight_limit, current_weight_assigned);
                    total_weight_assigned += current_weight_assigned;
                }

                bool accept_step = false;

                double score = static_cast<double>(total_weight_assigned) /
                               static_cast<double>(weight_limit + instance.synchronisationCosts());
                double parallelism = 0;
                if (weight_limit > 0) {
                    parallelism = static_cast<double>(total_weight_assigned) / static_cast<double>(weight_limit);
                }

                if (score > 0.97 * best_score) {
                    best_score = std::max(best_score, score);
                    best_parallelism = parallelism;
                    accept_step = true;
                } else {
                    continueSuperstepAttempts = false;
                }

                if (weight_limit >= minWeightParallelCheck) {
                    if (parallelism < std::max(2.0, 0.8 * desiredParallelism)) {
                        continueSuperstepAttempts = false;
                    }
                }

                if (weight_limit <= minSuperstepWeight) {
                    continueSuperstepAttempts = true;
                    if (total_assigned + new_total_assigned == N) {
                        accept_step = true;
                        continueSuperstepAttempts = false;
                    }
                }

                if (total_assigned + new_total_assigned == N) {
                    continueSuperstepAttempts = false;
                }

                if constexpr (use_memory_constraint) {
                    if (early_memory_break) {
                        continueSuperstepAttempts = false;
                    }
                }

                // undo proc assingments and predec decreases in any case
                for (unsigned proc = 0; proc < P; ++proc) {
                    for (const auto &node : new_assignments[proc]) {
                        node_to_proc[node] = std::numeric_limits<unsigned>::max();

                        for (const auto &succ : G.children(node)) {
                            predec[succ]++;
                            node_to_proc[succ] = std::numeric_limits<unsigned>::max();
                        }
                    }

                    if constexpr (use_memory_constraint) {
                        local_memory_constraint.reset(proc);
                    }
                }

                if (accept_step) {
                    best_new_assignments.swap(new_assignments);
                    best_new_ready.swap(new_ready);
                }

                limit++;
                limit += (limit / 2);
            }

            // apply best iteration
            for (const auto &node : best_new_ready) {
                ready.insert(node);
            }

            for (unsigned proc = 0; proc < P; ++proc) {
                for (const auto &node : best_new_assignments[proc]) {
                    node_to_proc[node] = proc;
                    node_to_supstep[node] = supstep;
                    ready.erase(node);
                    ++total_assigned;

                    for (const auto &succ : G.children(node)) {
                        predec[succ]--;
                    }
                }
            }

            desiredParallelism = (0.3 * desiredParallelism) + (0.6 * best_parallelism) +
                                 (0.1 * static_cast<double>(P)); // weights should sum up to one

            ++supstep;
        }

        schedule.updateNumberOfSupersteps();

        return RETURN_STATUS::OSP_SUCCESS;
    }

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "GreedyBspGrowLocalAutoCores" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "GrowLocalAutoCores"; }
};

} // namespace osp