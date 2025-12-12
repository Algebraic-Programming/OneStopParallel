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
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "MemoryConstraintModules.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

template <typename WeightT>
struct GrowLocalAutoCoresParams {
    unsigned minSuperstepSize_ = 20;
    WeightT syncCostMultiplierMinSuperstepWeight_ = 1;
    WeightT syncCostMultiplierParallelCheck_ = 4;
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
template <typename GraphT, typename MemoryConstraintT = NoMemoryConstraint>
class GrowLocalAutoCores : public Scheduler<GraphT> {
  private:
    GrowLocalAutoCores_Params<v_workw_t<Graph_t>> params_;

    constexpr static bool useMemoryConstraint_ = is_memory_constraint_v<MemoryConstraintT>
                                                 or is_memory_constraint_schedule_v<MemoryConstraintT>;

    static_assert(not useMemoryConstraint_ or std::is_same_v<GraphT, typename MemoryConstraintT::Graph_impl_t>,
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

    static_assert(not useMemoryConstraint_
                      or not(std::is_same_v<MemoryConstraintT, PersistentTransientMemoryConstraint<GraphT>>
                             or std::is_same_v<MemoryConstraintT, GlobalMemoryConstraint<GraphT>>),
                  "MemoryConstraint_t must not be persistent_transient_memory_constraint or global_memory_constraint. Not "
                  "supported in GrowLocalAutoCores.");

    MemoryConstraintT localMemoryConstraint_;

  public:
    /**
     * @brief Default constructor for GreedyBspGrowLocalAutoCores.
     */
    GrowLocalAutoCores(GrowLocalAutoCores_Params<v_workw_t<Graph_t>> params = GrowLocalAutoCores_Params<v_workw_t<Graph_t>>())
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
    virtual RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override {
        using VertexIdx = typename GraphT::vertex_idx;
        const auto &instance = schedule.getInstance();

        for (const auto &v : instance.getComputationalDag().vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            schedule.setAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
        }

        unsigned supstep = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraintT>) {
            localMemoryConstraint_.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraintT>) {
            localMemoryConstraint_.initialize(schedule, supstep);
        }

        auto &nodeToProc = schedule.assignedProcessors();
        auto &nodeToSupstep = schedule.assignedSupersteps();

        const auto n = instance.numberOfVertices();
        const unsigned p = instance.numberOfProcessors();
        const auto &g = instance.getComputationalDag();

        std::unordered_set<VertexIdx> ready;

        std::vector<VertexIdx> allReady;
        std::vector<std::vector<VertexIdx>> procReady(p);

        std::vector<VertexIdx> predec(n);

        for (const auto &node : g.vertices()) {
            predec[node] = g.in_degree(node);

            if (predec[node] == 0) {
                ready.insert(node);
            }
        }

        std::vector<std::vector<VertexIdx>> newAssignments(p);
        std::vector<std::vector<VertexIdx>> bestNewAssignments(p);

        std::vector<VertexIdx> newReady;
        std::vector<VertexIdx> bestNewReady;

        const v_workw_t<Graph_t> minWeightParallelCheck = params.syncCostMultiplierParallelCheck * instance.synchronisationCosts();
        const v_workw_t<Graph_t> minSuperstepWeight = params.syncCostMultiplierMinSuperstepWeight * instance.synchronisationCosts();

        double desiredParallelism = static_cast<double>(p);

        VertexIdx totalAssigned = 0;
        while (totalAssigned < n) {
            unsigned limit = params.minSuperstepSize;
            double bestScore = 0;
            double bestParallelism = 0;

            bool continueSuperstepAttempts = true;

            while (continueSuperstepAttempts) {
                for (unsigned p = 0; p < p; p++) {
                    newAssignments[p].clear();
                    procReady[p].clear();
                }

                newReady.clear();
                allReady.assign(ready.begin(), ready.end());
                std::make_heap(allReady.begin(), allReady.end(), std::greater<VertexIdx>());

                VertexIdx newTotalAssigned = 0;
                v_workw_t<Graph_t> weightLimit = 0, total_weight_assigned = 0;

                bool earlyMemoryBreak = false;

                // Processor 0
                while (newAssignments[0].size() < limit) {
                    VertexIdx chosenNode = std::numeric_limits<VertexIdx>::max();

                    if constexpr (useMemoryConstraint_) {
                        if (!procReady[0].empty() && localMemoryConstraint_.can_add(procReady[0].front(), 0)) {
                            chosenNode = procReady[0].front();
                            std::pop_heap(procReady[0].begin(), procReady[0].end(), std::greater<VertexIdx>());
                            procReady[0].pop_back();
                        } else if (!allReady.empty() && localMemoryConstraint_.can_add(allReady.front(), 0)) {
                            chosenNode = allReady.front();
                            std::pop_heap(allReady.begin(), allReady.end(), std::greater<VertexIdx>());
                            allReady.pop_back();
                        } else {
                            earlyMemoryBreak = true;
                            break;
                        }
                    } else {
                        if (!procReady[0].empty()) {
                            chosenNode = procReady[0].front();
                            std::pop_heap(procReady[0].begin(), procReady[0].end(), std::greater<VertexIdx>());
                            procReady[0].pop_back();
                        } else if (!allReady.empty()) {
                            chosenNode = allReady.front();
                            std::pop_heap(allReady.begin(), allReady.end(), std::greater<VertexIdx>());
                            allReady.pop_back();
                        } else {
                            break;
                        }
                    }

                    newAssignments[0].push_back(chosenNode);
                    nodeToProc[chosenNode] = 0;
                    newTotalAssigned++;
                    weightLimit += g.VertexWorkWeight(chosenNode);

                    if constexpr (useMemoryConstraint_) {
                        localMemoryConstraint_.add(chosenNode, 0);
                    }

                    for (const auto &succ : g.children(chosenNode)) {
                        if (nodeToProc[succ] == std::numeric_limits<unsigned>::max()) {
                            nodeToProc[succ] = 0;
                        } else if (nodeToProc[succ] != 0) {
                            nodeToProc[succ] = p;
                        }

                        predec[succ]--;
                        if (predec[succ] == 0) {
                            newReady.push_back(succ);

                            if (nodeToProc[succ] == 0) {
                                procReady[0].push_back(succ);
                                std::push_heap(procReady[0].begin(), procReady[0].end(), std::greater<VertexIdx>());
                            }
                        }
                    }
                }

                total_weight_assigned += weight_limit;

                // Processors 1 through P-1
                for (unsigned proc = 1; proc < p; ++proc) {
                    v_workw_t<Graph_t> currentWeightAssigned = 0;
                    while (current_weight_assigned < weight_limit) {
                        VertexIdx chosenNode = std::numeric_limits<VertexIdx>::max();

                        if constexpr (useMemoryConstraint_) {
                            if (!procReady[proc].empty() && localMemoryConstraint_.can_add(procReady[proc].front(), proc)) {
                                chosenNode = procReady[proc].front();
                                std::pop_heap(procReady[proc].begin(), procReady[proc].end(), std::greater<VertexIdx>());
                                procReady[proc].pop_back();
                            } else if (!allReady.empty() && localMemoryConstraint_.can_add(allReady.front(), proc)) {
                                chosenNode = allReady.front();
                                std::pop_heap(allReady.begin(), allReady.end(), std::greater<VertexIdx>());
                                allReady.pop_back();
                            } else {
                                earlyMemoryBreak = true;
                                break;
                            }
                        } else {
                            if (!procReady[proc].empty()) {
                                chosenNode = procReady[proc].front();
                                std::pop_heap(procReady[proc].begin(), procReady[proc].end(), std::greater<VertexIdx>());
                                procReady[proc].pop_back();
                            } else if (!allReady.empty()) {
                                chosenNode = allReady.front();
                                std::pop_heap(allReady.begin(), allReady.end(), std::greater<VertexIdx>());
                                allReady.pop_back();
                            } else {
                                break;
                            }
                        }

                        newAssignments[proc].push_back(chosenNode);
                        nodeToProc[chosenNode] = proc;
                        newTotalAssigned++;
                        currentWeightAssigned += g.VertexWorkWeight(chosenNode);

                        if constexpr (useMemoryConstraint_) {
                            localMemoryConstraint_.add(chosenNode, proc);
                        }

                        for (const auto &succ : g.children(chosenNode)) {
                            if (nodeToProc[succ] == std::numeric_limits<unsigned>::max()) {
                                nodeToProc[succ] = proc;
                            } else if (nodeToProc[succ] != proc) {
                                nodeToProc[succ] = p;
                            }
                            predec[succ]--;
                            if (predec[succ] == 0) {
                                newReady.push_back(succ);

                                if (nodeToProc[succ] == proc) {
                                    procReady[proc].push_back(succ);
                                    std::push_heap(procReady[proc].begin(), procReady[proc].end(), std::greater<VertexIdx>());
                                }
                            }
                        }
                    }

                    weightLimit = std::max(weight_limit, current_weight_assigned);
                    total_weight_assigned += current_weight_assigned;
                }

                bool acceptStep = false;

                double score = static_cast<double>(total_weight_assigned)
                               / static_cast<double>(weight_limit + instance.synchronisationCosts());
                double parallelism = 0;
                if (weightLimit > 0) {
                    parallelism = static_cast<double>(total_weight_assigned) / static_cast<double>(weight_limit);
                }

                if (score > 0.97 * bestScore) {
                    bestScore = std::max(bestScore, score);
                    bestParallelism = parallelism;
                    acceptStep = true;
                } else {
                    continueSuperstepAttempts = false;
                }

                if (weightLimit >= minWeightParallelCheck) {
                    if (parallelism < std::max(2.0, 0.8 * desiredParallelism)) {
                        continueSuperstepAttempts = false;
                    }
                }

                if (weightLimit <= minSuperstepWeight) {
                    continueSuperstepAttempts = true;
                    if (totalAssigned + newTotalAssigned == n) {
                        acceptStep = true;
                        continueSuperstepAttempts = false;
                    }
                }

                if (totalAssigned + newTotalAssigned == n) {
                    continueSuperstepAttempts = false;
                }

                if constexpr (useMemoryConstraint_) {
                    if (earlyMemoryBreak) {
                        continueSuperstepAttempts = false;
                    }
                }

                // undo proc assingments and predec decreases in any case
                for (unsigned proc = 0; proc < p; ++proc) {
                    for (const auto &node : newAssignments[proc]) {
                        nodeToProc[node] = std::numeric_limits<unsigned>::max();

                        for (const auto &succ : g.children(node)) {
                            predec[succ]++;
                            nodeToProc[succ] = std::numeric_limits<unsigned>::max();
                        }
                    }

                    if constexpr (useMemoryConstraint_) {
                        localMemoryConstraint_.reset(proc);
                    }
                }

                if (acceptStep) {
                    bestNewAssignments.swap(newAssignments);
                    bestNewReady.swap(newReady);
                }

                limit++;
                limit += (limit / 2);
            }

            // apply best iteration
            for (const auto &node : bestNewReady) {
                ready.insert(node);
            }

            for (unsigned proc = 0; proc < p; ++proc) {
                for (const auto &node : bestNewAssignments[proc]) {
                    nodeToProc[node] = proc;
                    nodeToSupstep[node] = supstep;
                    ready.erase(node);
                    ++totalAssigned;

                    for (const auto &succ : g.children(node)) {
                        predec[succ]--;
                    }
                }
            }

            desiredParallelism = (0.3 * desiredParallelism) + (0.6 * bestParallelism)
                                 + (0.1 * static_cast<double>(p));    // weights should sum up to one

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

}    // namespace osp
