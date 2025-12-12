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

#include <omp.h>

#include <climits>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

// #define TIME_THREADS_GROW_LOCAL_PARALLEL
#ifdef TIME_THREADS_GROW_LOCAL_PARALLEL
#    include <chrono>
#    include <iostream>
#endif

#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

static constexpr unsigned cacheLineSize = 64;

template <typename VertT, typename WeightT>
struct GrowLocalAutoCoresParallelParams {
    VertT minSuperstepSize_ = 20;
    WeightT syncCostMultiplierMinSuperstepWeight_ = 1;
    WeightT syncCostMultiplierParallelCheck_ = 4;

    unsigned numThreads_ = 0;              // 0 for auto
    unsigned maxNumThreads_ = UINT_MAX;    // used when auto num threads
};

/**
 * @brief The GrowLocalAutoCoresParallel class represents a scheduler that uses a greedy algorithm to compute
 * schedules for BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "GrowLocalAutoCoresParallel" in this
 * case.
 */
template <typename GraphT>
class GrowLocalAutoCoresParallel : public Scheduler<GraphT> {
    static_assert(is_directed_graph_v<Graph_t>);
    static_assert(has_vertex_weights_v<Graph_t>);

  private:
    using VertexType = vertex_idx_t<Graph_t>;

    GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>> params;

  public:
    /**
     * @brief Default constructor for GrowLocalAutoCoresParallel.
     */
    GrowLocalAutoCoresParallel(GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>> params_
                               = GrowLocalAutoCoresParallel_Params<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>>())
        : params(params_) {}

    /**
     * @brief Default destructor for GrowLocalAutoCoresParallel.
     */
    virtual ~GrowLocalAutoCoresParallel() = default;

    void ComputePartialSchedule(BspSchedule<GraphT> &schedule,
                                const std::vector<VertexType> &topOrder,
                                const std::vector<VertexType> &posInTopOrder,
                                const VertexType startNode,
                                const VertexType endNode,
                                unsigned &supstep) const {
#ifdef TIME_THREADS_GROW_LOCAL_PARALLEL
        double startTime = omp_get_wtime();
#endif
        const BspInstance<GraphT> &instance = schedule.getInstance();
        const GraphT &graph = instance.getComputationalDag();

        const VertexType n = endNode - startNode;
        const unsigned p = instance.numberOfProcessors();

        std::set<VertexType> ready;

        std::vector<VertexType> futureReady;
        std::vector<VertexType> bestFutureReady;

        std::vector<std::set<VertexType>> procReady(p);
        std::vector<std::set<VertexType>> bestProcReady(p);

        std::vector<VertexType> predec(n, 0);

        if constexpr (has_vertices_in_top_order_v<Graph_t>) {
            if constexpr (has_children_in_vertex_order_v<Graph_t>) {
                for (VertexType vert = startNode; vert < endNode; ++vert) {
                    for (const VertexType &chld : graph.children(vert)) {
                        if (chld >= endNode) {
                            break;
                        }
                        ++predec[chld - startNode];
                    }
                }
            } else {
                for (VertexType vert = startNode; vert < endNode; ++vert) {
                    for (const VertexType &chld : graph.children(vert)) {
                        if (chld < endNode) {
                            ++predec[chld - startNode];
                        }
                    }
                }
            }
        } else {
            for (VertexType index = startNode; index < endNode; ++index) {
                VertexType vert = topOrder[index];
                for (const VertexType &par : graph.parents(vert)) {
                    VertexType posPar = posInTopOrder[par];
                    if (posPar >= startNode) {
                        ++predec[index - startNode];
                    }
                }
            }
        }

        for (VertexType nodePos = startNode; nodePos < endNode; nodePos++) {
            VertexType index = nodePos - startNode;
            if (predec[index] == 0) {
                if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                    ready.insert(nodePos);
                } else {
                    ready.insert(topOrder[nodePos]);
                }
            }
        }

        std::vector<std::vector<VertexType>> newAssignments(p);
        std::vector<std::vector<VertexType>> bestNewAssignments(p);

        const v_workw_t<Graph_t> minWeightParallelCheck = params.syncCostMultiplierParallelCheck * instance.synchronisationCosts();
        const v_workw_t<Graph_t> minSuperstepWeight = params.syncCostMultiplierMinSuperstepWeight * instance.synchronisationCosts();

        double desiredParallelism = static_cast<double>(p);

        VertexType totalAssigned = 0;
        supstep = 0;

        while (total_assigned < N) {
            VertexType limit = params.minSuperstepSize;
            double bestScore = 0;
            double bestParallelism = 0;

            typename std::set<VertexType>::iterator readyIter;
            typename std::set<VertexType>::iterator bestReadyIter;

            bool continueSuperstepAttempts = true;

            while (continueSuperstepAttempts) {
                for (unsigned p = 0; p < p; p++) {
                    newAssignments[p].clear();
                }
                futureReady.clear();

                for (unsigned p = 0; p < p; p++) {
                    procReady[p].clear();
                }

                readyIter = ready.begin();

                VertexType newTotalAssigned = 0;
                v_workw_t<Graph_t> weightLimit = 0;
                v_workw_t<Graph_t> totalWeightAssigned = 0;

                // Processor 0
                while (newAssignments[0].size() < limit) {
                    VertexType chosenNode = std::numeric_limits<VertexType>::max();
                    if (!procReady[0].empty()) {
                        chosenNode = *procReady[0].begin();
                        procReady[0].erase(procReady[0].begin());
                    } else if (readyIter != ready.end()) {
                        chosenNode = *readyIter;
                        readyIter++;
                    } else {
                        break;
                    }

                    newAssignments[0].push_back(chosen_node);
                    schedule.setAssignedProcessor(chosen_node, 0);
                    newTotalAssigned++;
                    weightLimit += graph.vertex_work_weight(chosen_node);

                    for (const VertexType &succ : graph.children(chosen_node)) {
                        if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                            if constexpr (has_children_in_vertex_order_v<Graph_t>) {
                                if (succ >= endNode) {
                                    break;
                                }
                            } else {
                                if (succ >= endNode) {
                                    continue;
                                }
                            }
                        } else {
                            if (posInTopOrder[succ] >= endNode) {
                                continue;
                            }
                        }

                        if (schedule.assignedProcessor(succ) == UINT_MAX) {
                            schedule.setAssignedProcessor(succ, 0);
                        } else if (schedule.assignedProcessor(succ) != 0) {
                            schedule.setAssignedProcessor(succ, P);
                        }

                        VertexType succIndex;
                        if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                            succIndex = succ - startNode;
                        } else {
                            succIndex = posInTopOrder[succ] - startNode;
                        }

                        --predec[succIndex];
                        if (predec[succIndex] == 0) {
                            if (schedule.assignedProcessor(succ) == 0) {
                                procReady[0].insert(succ);
                            } else {
                                futureReady.push_back(succ);
                            }
                        }
                    }
                }

                totalWeightAssigned += weight_limit;

                // Processors 1 through P-1
                for (unsigned proc = 1; proc < p; ++proc) {
                    v_workw_t<Graph_t> currentWeightAssigned = 0;
                    while (current_weight_assigned < weight_limit) {
                        VertexType chosenNode = std::numeric_limits<VertexType>::max();
                        if (!procReady[proc].empty()) {
                            chosenNode = *procReady[proc].begin();
                            procReady[proc].erase(procReady[proc].begin());
                        } else if (readyIter != ready.end()) {
                            chosenNode = *readyIter;
                            readyIter++;
                        } else {
                            break;
                        }

                        newAssignments[proc].push_back(chosen_node);
                        schedule.setAssignedProcessor(chosen_node, proc);
                        newTotalAssigned++;
                        currentWeightAssigned += graph.vertex_work_weight(chosen_node);

                        for (const VertexType &succ : graph.children(chosen_node)) {
                            if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                                if constexpr (has_children_in_vertex_order_v<Graph_t>) {
                                    if (succ >= endNode) {
                                        break;
                                    }
                                } else {
                                    if (succ >= endNode) {
                                        continue;
                                    }
                                }
                            } else {
                                if (posInTopOrder[succ] >= endNode) {
                                    continue;
                                }
                            }

                            if (schedule.assignedProcessor(succ) == UINT_MAX) {
                                schedule.setAssignedProcessor(succ, proc);
                            } else if (schedule.assignedProcessor(succ) != proc) {
                                schedule.setAssignedProcessor(succ, P);
                            }

                            VertexType succIndex;
                            if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                                succIndex = succ - startNode;
                            } else {
                                succIndex = posInTopOrder[succ] - startNode;
                            }

                            --predec[succIndex];
                            if (predec[succIndex] == 0) {
                                if (schedule.assignedProcessor(succ) == proc) {
                                    procReady[proc].insert(succ);
                                } else {
                                    futureReady.push_back(succ);
                                }
                            }
                        }
                    }

                    weightLimit = std::max(weight_limit, current_weight_assigned);
                    totalWeightAssigned += current_weight_assigned;
                }

                bool acceptStep = false;

                double score = static_cast<double>(total_weight_assigned)
                               / static_cast<double>(weight_limit + instance.synchronisationCosts());
                double parallelism = 0;
                if (weightLimit > 0) {
                    parallelism = static_cast<double>(total_weight_assigned) / static_cast<double>(weight_limit);
                }

                if (score > 0.97 * bestScore) {    // It is possible to make this less strict, i.e. score > 0.98 * best_score.
                                                   // The purpose of this would be to encourage larger supersteps.
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
                    if (totalAssigned + new_total_assigned == N) {
                        acceptStep = true;
                        continueSuperstepAttempts = false;
                    }
                }

                if (totalAssigned + new_total_assigned == N) {
                    continueSuperstepAttempts = false;
                }

                // undo proc assingments and predec increases in any case
                for (unsigned proc = 0; proc < p; ++proc) {
                    for (const VertexType &node : new_assignments[proc]) {
                        schedule.setAssignedProcessor(node, UINT_MAX);
                    }
                }

                for (unsigned proc = 0; proc < p; ++proc) {
                    for (const VertexType &node : new_assignments[proc]) {
                        for (const VertexType &succ : graph.children(node)) {
                            if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                                if constexpr (has_children_in_vertex_order_v<Graph_t>) {
                                    if (succ >= endNode) {
                                        break;
                                    }
                                } else {
                                    if (succ >= endNode) {
                                        continue;
                                    }
                                }
                            } else {
                                if (posInTopOrder[succ] >= endNode) {
                                    continue;
                                }
                            }

                            VertexType succIndex;
                            if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                                succIndex = succ - startNode;
                            } else {
                                succIndex = posInTopOrder[succ] - startNode;
                            }

                            ++predec[succIndex];
                        }
                    }
                }

                for (unsigned proc = 0; proc < p; ++proc) {
                    for (const VertexType &node : new_assignments[proc]) {
                        for (const VertexType &succ : graph.children(node)) {
                            if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                                if constexpr (has_children_in_vertex_order_v<Graph_t>) {
                                    if (succ >= endNode) {
                                        break;
                                    }
                                } else {
                                    if (succ >= endNode) {
                                        continue;
                                    }
                                }
                            } else {
                                if (posInTopOrder[succ] >= endNode) {
                                    continue;
                                }
                            }

                            schedule.setAssignedProcessor(succ, UINT_MAX);
                        }
                    }
                }

                if (acceptStep) {
                    bestNewAssignments.swap(new_assignments);
                    bestFutureReady.swap(futureReady);
                    bestProcReady.swap(procReady);
                    bestReadyIter = readyIter;
                }

                limit++;
                limit += (limit / 2);
            }

            // apply best iteration
            ready.erase(ready.begin(), bestReadyIter);
            ready.insert(best_futureReady.begin(), best_futureReady.end());
            for (unsigned proc = 0; proc < p; proc++) {
                ready.merge(best_procReady[proc]);
            }

            for (unsigned proc = 0; proc < p; ++proc) {
                for (const VertexType &node : best_new_assignments[proc]) {
                    schedule.setAssignedProcessor(node, proc);
                    schedule.setAssignedSuperstepNoUpdateNumSuperstep(node, supstep);
                    ++total_assigned;

                    for (const VertexType &succ : graph.children(node)) {
                        if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                            if constexpr (has_children_in_vertex_order_v<Graph_t>) {
                                if (succ >= endNode) {
                                    break;
                                }
                            } else {
                                if (succ >= endNode) {
                                    continue;
                                }
                            }
                        } else {
                            if (posInTopOrder[succ] >= endNode) {
                                continue;
                            }
                        }

                        VertexType succIndex;
                        if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                            succIndex = succ - startNode;
                        } else {
                            succIndex = posInTopOrder[succ] - startNode;
                        }

                        --predec[succIndex];
                    }
                }
            }

            desiredParallelism = (0.3 * desiredParallelism) + (0.6 * bestParallelism)
                                 + (0.1 * static_cast<double>(p));    // weights should sum up to one

            ++supstep;
        }

#ifdef TIME_THREADS_GROW_LOCAL_PARALLEL
        double endTime = omp_get_wtime();
        std::string padd = "";
        if (omp_get_thread_num() < 10) {
            padd = " ";
        }
        std::string outputString
            = "Thread: " + padd + std::to_string(omp_get_thread_num()) + "\t Time: " + std::to_string(endTime - startTime) + "\n";
        std::cout << outputString;
#endif
    }

    void IncrementScheduleSupersteps(BspSchedule<GraphT> &schedule,
                                     const VertexType startNode,
                                     const VertexType endNode,
                                     const unsigned incr) const {
        for (VertexType node = startNode; node < endNode; node++) {
            schedule.setAssignedSuperstepNoUpdateNumSuperstep(node, schedule.assignedSuperstep(node) + incr);
        }
    }

    void IncrementScheduleSuperstepsTopOrder(BspSchedule<GraphT> &schedule,
                                             const std::vector<VertexType> &topOrder,
                                             const VertexType startIndex,
                                             const VertexType endIndex,
                                             const unsigned incr) const {
        for (VertexType index = startIndex; index < endIndex; index++) {
            const VertexType node = topOrder[index];
            schedule.setAssignedSuperstepNoUpdateNumSuperstep(node, schedule.assignedSuperstep(node) + incr);
        }
    }

    RETURN_STATUS ComputeScheduleParallel(BspSchedule<GraphT> &schedule, unsigned int numThreads) const {
        const BspInstance<GraphT> &instance = schedule.getInstance();
        const GraphT &graph = instance.getComputationalDag();

        const VertexType n = instance.numberOfVertices();

        for (VertexType vert = 0; vert < N; ++vert) {
            schedule.setAssignedProcessor(vert, UINT_MAX);
        }

        VertexType numNodesPerThread = N / numThreads;
        std::vector<VertexType> startNodes;
        startNodes.reserve(numThreads + 1);
        VertexType startNode = 0;
        for (unsigned thr = 0; thr < numThreads; thr++) {
            startNodes.push_back(startNode);
            startNode += numNodesPerThread;
        }
        startNodes.push_back(N);

        static constexpr unsigned unsignedPadding = (cacheLineSize + sizeof(unsigned) - 1) / sizeof(unsigned);
        std::vector<unsigned> superstepsThread(numThreads * unsignedPadding, 0);
        std::vector<unsigned> supstepIncr(numThreads, 0);
        unsigned incr = 0;

        std::vector<VertexType> topOrder;
        if constexpr (not has_vertices_in_top_order_v<Graph_t>) {
            topOrder = GetTopOrder(graph);
        }

        std::vector<VertexType> posInTopOrder;
        if constexpr (not has_vertices_in_top_order_v<Graph_t>) {
            posInTopOrder = std::vector<VertexType>(graph.num_vertices());
            for (VertexType ind = 0; ind < static_cast<VertexType>(topOrder.size()); ++ind) {
                posInTopOrder[topOrder[ind]] = ind;
            }
        }

#pragma omp parallel num_threads(numThreads) default(none)                                                 \
    shared(schedule, topOrder, posInTopOrder, superstepsThread, supstepIncr, numThreads, startNodes, incr)
        {
#pragma omp for schedule(static, 1)
            for (unsigned thr = 0; thr < numThreads; thr++) {
                computePartialSchedule(
                    schedule, topOrder, posInTopOrder, startNodes[thr], startNodes[thr + 1], superstepsThread[thr * UnsignedPadding]);
            }

#pragma omp master
            {
                for (unsigned thr = 0; thr < numThreads; thr++) {
                    supstepIncr[thr] = incr;
                    incr += superstepsThread[thr * unsignedPadding];
                }
                // the value of incr is now the number of supersteps
            }

#pragma omp barrier

#pragma omp for schedule(static, 1)
            for (unsigned thr = 0; thr < numThreads; thr++) {
                if constexpr (has_vertices_in_top_order_v<Graph_t>) {
                    incrementScheduleSupersteps(schedule, startNodes[thr], startNodes[thr + 1], supstepIncr[thr]);
                } else {
                    incrementScheduleSupersteps_TopOrder(
                        schedule, topOrder, startNodes[thr], startNodes[thr + 1], supstepIncr[thr]);
                }
            }
        }

        schedule.setNumberOfSupersteps(incr);

        return RETURN_STATUS::OSP_SUCCESS;
    }

    /**
     * @brief Compute a schedule for the given BspInstance.
     *
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     *
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override {
        unsigned numThreads = params.numThreads;
        if (numThreads == 0) {
            // numThreads = static_cast<unsigned>(std::sqrt( static_cast<double>((schedule.getInstance().numberOfVertices() / 1000000)))) + 1;
            numThreads
                = static_cast<unsigned>(std::log2(static_cast<double>((schedule.getInstance().numberOfVertices() / 1000)))) + 1;
        }
        numThreads = std::min(numThreads, params.maxNumThreads);
        if (numThreads == 0) {
            numThreads = 1;
        }

        return computeScheduleParallel(schedule, numThreads);
    }

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "GrowLocalAutoCoresParallel" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "GrowLocalAutoCoresParallel"; }
};

}    // namespace osp
