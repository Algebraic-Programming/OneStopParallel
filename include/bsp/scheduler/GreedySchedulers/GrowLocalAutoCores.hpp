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
#include <map>
#include <set>
#include <string>
#include <vector>

#include "auxiliary/misc.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "bsp/scheduler/Scheduler.hpp"

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
template<typename Graph_t>
class GrowLocalAutoCores : public Scheduler<Graph_t> {

  private:
    GrowLocalAutoCores_Params<v_workw_t<Graph_t>> params;

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

        auto &node_to_proc = schedule.assignedProcessors();
        auto &node_to_supstep = schedule.assignedSupersteps();

        const auto N = instance.numberOfVertices();
        const unsigned P = instance.numberOfProcessors();
        const auto &G = instance.getComputationalDag();

        std::set<vertex_idx> ready;
        std::vector<std::set<std::size_t>::iterator> place_in_ready(N);

        std::set<vertex_idx> allReady;
        std::vector<std::set<vertex_idx>> procReady(P);

        std::vector<vertex_idx> predec(N);

        for (const auto &node : G.vertices()) {
            predec[node] = G.in_degree(node);

            if (predec[node] == 0) {
                place_in_ready[node] = ready.insert(node).first;
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

        unsigned supstep = 0;
        vertex_idx total_assigned = 0;
        while (total_assigned < N) {

            unsigned limit = params.minSuperstepSize;
            double best_score = 0;
            double best_parallelism = 0;

            bool continueSuperstepAttempts = true;

            while (continueSuperstepAttempts) {
                for (unsigned p = 0; p < P; p++) {
                    new_assignments[p].clear();
                }
                new_ready.clear();

                for (unsigned p = 0; p < P; p++) {
                    procReady[p].clear();
                }

                allReady = ready;

                vertex_idx new_total_assigned = 0;
                v_workw_t<Graph_t> weight_limit = 0, total_weight_assigned = 0;

                // Processor 0
                while (new_assignments[0].size() < limit) {
                    vertex_idx chosen_node = std::numeric_limits<vertex_idx>::max();
                    if (!procReady[0].empty()) {
                        chosen_node = *procReady[0].begin();
                        procReady[0].erase(procReady[0].begin());
                    } else if (!allReady.empty()) {
                        chosen_node = *allReady.begin();
                        allReady.erase(allReady.begin());
                    } else {
                        break;
                    }

                    new_assignments[0].push_back(chosen_node);
                    node_to_proc[chosen_node] = 0;
                    new_total_assigned++;
                    weight_limit += G.vertex_work_weight(chosen_node);

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
                                procReady[0].insert(succ);
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
                        if (!procReady[proc].empty()) {
                            chosen_node = *procReady[proc].begin();
                            procReady[proc].erase(procReady[proc].begin());
                        } else if (!allReady.empty()) {
                            chosen_node = *allReady.begin();
                            allReady.erase(allReady.begin());
                        } else
                            break;

                        new_assignments[proc].push_back(chosen_node);
                        node_to_proc[chosen_node] = proc;
                        new_total_assigned++;
                        current_weight_assigned += G.vertex_work_weight(chosen_node);

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
                                    procReady[proc].insert(succ);
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

                // undo proc assingments and predec decreases in any case
                for (unsigned proc = 0; proc < P; ++proc) {
                    for (const auto &node : new_assignments[proc]) {
                        node_to_proc[node] = std::numeric_limits<unsigned>::max();
                    }
                }

                for (unsigned proc = 0; proc < P; ++proc) {
                    for (const auto &node : new_assignments[proc]) {
                        for (const auto &succ : G.children(node)) {
                            predec[succ]++;
                        }
                    }
                }

                for (unsigned proc = 0; proc < P; ++proc) {
                    for (const auto &node : new_assignments[proc]) {
                        for (const auto &succ : G.children(node)) {
                            node_to_proc[succ] = std::numeric_limits<unsigned>::max();
                        }
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
                place_in_ready[node] = ready.insert(node).first;
            }

            for (unsigned proc = 0; proc < P; ++proc) {
                for (const auto &node : best_new_assignments[proc]) {
                    node_to_proc[node] = proc;
                    node_to_supstep[node] = supstep;
                    ready.erase(place_in_ready[node]);
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
       
        return SUCCESS;
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