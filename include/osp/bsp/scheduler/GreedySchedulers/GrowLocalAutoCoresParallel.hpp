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

static constexpr std::size_t CACHE_LINE_SIZE = 64;

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
 * This class inherits from the Scheduler class and implements the ComputeSchedule() and GetScheduleName() methods.
 * The ComputeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The GetScheduleName() method returns the name of the schedule, which is "GrowLocalAutoCoresParallel" in this
 * case.
 */
template <typename GraphT>
class GrowLocalAutoCoresParallel : public Scheduler<GraphT> {
    static_assert(isDirectedGraphV<GraphT>);
    static_assert(hasVertexWeightsV<GraphT>);

  private:
    using VertexType = VertexIdxT<GraphT>;

    GrowLocalAutoCoresParallelParams<VertexIdxT<GraphT>, VWorkwT<GraphT>> params_;

  public:
    /**
     * @brief Default constructor for GrowLocalAutoCoresParallel.
     */
    GrowLocalAutoCoresParallel(GrowLocalAutoCoresParallelParams<VertexIdxT<GraphT>, VWorkwT<GraphT>> params
                               = GrowLocalAutoCoresParallelParams<VertexIdxT<GraphT>, VWorkwT<GraphT>>())
        : params_(params) {}

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
        const BspInstance<GraphT> &instance = schedule.GetInstance();
        const GraphT &graph = instance.GetComputationalDag();

        const VertexType n = endNode - startNode;
        const unsigned p = instance.NumberOfProcessors();

        std::set<VertexType> ready;

        std::vector<VertexType> futureReady;
        std::vector<VertexType> bestFutureReady;

        std::vector<std::set<VertexType>> procReady(p);
        std::vector<std::set<VertexType>> bestProcReady(p);

        std::vector<VertexType> predec(n, 0);

        if constexpr (hasVerticesInTopOrderV<GraphT>) {
            if constexpr (hasChildrenInVertexOrderV<GraphT>) {
                for (VertexType vert = startNode; vert < endNode; ++vert) {
                    for (const VertexType &chld : graph.Children(vert)) {
                        if (chld >= endNode) {
                            break;
                        }
                        ++predec[chld - startNode];
                    }
                }
            } else {
                for (VertexType vert = startNode; vert < endNode; ++vert) {
                    for (const VertexType &chld : graph.Children(vert)) {
                        if (chld < endNode) {
                            ++predec[chld - startNode];
                        }
                    }
                }
            }
        } else {
            for (VertexType index = startNode; index < endNode; ++index) {
                VertexType vert = topOrder[index];
                for (const VertexType &par : graph.Parents(vert)) {
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
                if constexpr (hasVerticesInTopOrderV<GraphT>) {
                    ready.insert(nodePos);
                } else {
                    ready.insert(topOrder[nodePos]);
                }
            }
        }

        std::vector<std::vector<VertexType>> newAssignments(p);
        std::vector<std::vector<VertexType>> bestNewAssignments(p);

        const VWorkwT<GraphT> minWeightParallelCheck = params_.syncCostMultiplierParallelCheck_ * instance.SynchronisationCosts();
        const VWorkwT<GraphT> minSuperstepWeight = params_.syncCostMultiplierMinSuperstepWeight_ * instance.SynchronisationCosts();

        double desiredParallelism = static_cast<double>(p);

        VertexType totalAssigned = 0;
        supstep = 0;

        while (totalAssigned < n) {
            VertexType limit = params_.minSuperstepSize_;
            double bestScore = 0;
            double bestParallelism = 0;

            typename std::set<VertexType>::iterator readyIter;
            typename std::set<VertexType>::iterator bestReadyIter;

            bool continueSuperstepAttempts = true;

            while (continueSuperstepAttempts) {
                for (unsigned proc = 0; proc < p; proc++) {
                    newAssignments[proc].clear();
                }
                futureReady.clear();

                for (unsigned proc = 0; proc < p; proc++) {
                    procReady[proc].clear();
                }

                readyIter = ready.begin();

                VertexType newTotalAssigned = 0;
                VWorkwT<GraphT> weightLimit = 0;
                VWorkwT<GraphT> totalWeightAssigned = 0;

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

                    newAssignments[0].push_back(chosenNode);
                    schedule.SetAssignedProcessor(chosenNode, 0);
                    newTotalAssigned++;
                    weightLimit += graph.VertexWorkWeight(chosenNode);

                    for (const VertexType &succ : graph.Children(chosenNode)) {
                        if constexpr (hasVerticesInTopOrderV<GraphT>) {
                            if constexpr (hasChildrenInVertexOrderV<GraphT>) {
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

                        if (schedule.AssignedProcessor(succ) == UINT_MAX) {
                            schedule.SetAssignedProcessor(succ, 0);
                        } else if (schedule.AssignedProcessor(succ) != 0) {
                            schedule.SetAssignedProcessor(succ, p);
                        }

                        VertexType succIndex;
                        if constexpr (hasVerticesInTopOrderV<GraphT>) {
                            succIndex = succ - startNode;
                        } else {
                            succIndex = posInTopOrder[succ] - startNode;
                        }

                        --predec[succIndex];
                        if (predec[succIndex] == 0) {
                            if (schedule.AssignedProcessor(succ) == 0) {
                                procReady[0].insert(succ);
                            } else {
                                futureReady.push_back(succ);
                            }
                        }
                    }
                }

                totalWeightAssigned += weightLimit;

                // Processors 1 through P-1
                for (unsigned proc = 1; proc < p; ++proc) {
                    VWorkwT<GraphT> currentWeightAssigned = 0;
                    while (currentWeightAssigned < weightLimit) {
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

                        newAssignments[proc].push_back(chosenNode);
                        schedule.SetAssignedProcessor(chosenNode, proc);
                        newTotalAssigned++;
                        currentWeightAssigned += graph.VertexWorkWeight(chosenNode);

                        for (const VertexType &succ : graph.Children(chosenNode)) {
                            if constexpr (hasVerticesInTopOrderV<GraphT>) {
                                if constexpr (hasChildrenInVertexOrderV<GraphT>) {
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

                            if (schedule.AssignedProcessor(succ) == UINT_MAX) {
                                schedule.SetAssignedProcessor(succ, proc);
                            } else if (schedule.AssignedProcessor(succ) != proc) {
                                schedule.SetAssignedProcessor(succ, p);
                            }

                            VertexType succIndex;
                            if constexpr (hasVerticesInTopOrderV<GraphT>) {
                                succIndex = succ - startNode;
                            } else {
                                succIndex = posInTopOrder[succ] - startNode;
                            }

                            --predec[succIndex];
                            if (predec[succIndex] == 0) {
                                if (schedule.AssignedProcessor(succ) == proc) {
                                    procReady[proc].insert(succ);
                                } else {
                                    futureReady.push_back(succ);
                                }
                            }
                        }
                    }

                    weightLimit = std::max(weightLimit, currentWeightAssigned);
                    totalWeightAssigned += currentWeightAssigned;
                }

                bool acceptStep = false;

                double score = static_cast<double>(totalWeightAssigned)
                               / static_cast<double>(weightLimit + instance.SynchronisationCosts());
                double parallelism = 0;
                if (weightLimit > 0) {
                    parallelism = static_cast<double>(totalWeightAssigned) / static_cast<double>(weightLimit);
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
                    if (totalAssigned + newTotalAssigned == n) {
                        acceptStep = true;
                        continueSuperstepAttempts = false;
                    }
                }

                if (totalAssigned + newTotalAssigned == n) {
                    continueSuperstepAttempts = false;
                }

                // undo proc assingments and predec increases in any case
                for (unsigned proc = 0; proc < p; ++proc) {
                    for (const VertexType &node : newAssignments[proc]) {
                        schedule.SetAssignedProcessor(node, UINT_MAX);
                    }
                }

                for (unsigned proc = 0; proc < p; ++proc) {
                    for (const VertexType &node : newAssignments[proc]) {
                        for (const VertexType &succ : graph.Children(node)) {
                            if constexpr (hasVerticesInTopOrderV<GraphT>) {
                                if constexpr (hasChildrenInVertexOrderV<GraphT>) {
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
                            if constexpr (hasVerticesInTopOrderV<GraphT>) {
                                succIndex = succ - startNode;
                            } else {
                                succIndex = posInTopOrder[succ] - startNode;
                            }

                            ++predec[succIndex];
                        }
                    }
                }

                for (unsigned proc = 0; proc < p; ++proc) {
                    for (const VertexType &node : newAssignments[proc]) {
                        for (const VertexType &succ : graph.Children(node)) {
                            if constexpr (hasVerticesInTopOrderV<GraphT>) {
                                if constexpr (hasChildrenInVertexOrderV<GraphT>) {
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

                            schedule.SetAssignedProcessor(succ, UINT_MAX);
                        }
                    }
                }

                if (acceptStep) {
                    bestNewAssignments.swap(newAssignments);
                    bestFutureReady.swap(futureReady);
                    bestProcReady.swap(procReady);
                    bestReadyIter = readyIter;
                }

                limit++;
                limit += (limit / 2);
            }

            // apply best iteration
            ready.erase(ready.begin(), bestReadyIter);
            ready.insert(bestFutureReady.begin(), bestFutureReady.end());
            for (unsigned proc = 0; proc < p; proc++) {
                ready.merge(bestProcReady[proc]);
            }

            for (unsigned proc = 0; proc < p; ++proc) {
                for (const VertexType &node : bestNewAssignments[proc]) {
                    schedule.SetAssignedProcessor(node, proc);
                    schedule.SetAssignedSuperstepNoUpdateNumSuperstep(node, supstep);
                    ++totalAssigned;

                    for (const VertexType &succ : graph.Children(node)) {
                        if constexpr (hasVerticesInTopOrderV<GraphT>) {
                            if constexpr (hasChildrenInVertexOrderV<GraphT>) {
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
                        if constexpr (hasVerticesInTopOrderV<GraphT>) {
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
            schedule.SetAssignedSuperstepNoUpdateNumSuperstep(node, schedule.AssignedSuperstep(node) + incr);
        }
    }

    void IncrementScheduleSuperstepsTopOrder(BspSchedule<GraphT> &schedule,
                                             const std::vector<VertexType> &topOrder,
                                             const VertexType startIndex,
                                             const VertexType endIndex,
                                             const unsigned incr) const {
        for (VertexType index = startIndex; index < endIndex; index++) {
            const VertexType node = topOrder[index];
            schedule.SetAssignedSuperstepNoUpdateNumSuperstep(node, schedule.AssignedSuperstep(node) + incr);
        }
    }

    ReturnStatus ComputeScheduleParallel(BspSchedule<GraphT> &schedule, unsigned int numThreads) const {
        const BspInstance<GraphT> &instance = schedule.GetInstance();
        const GraphT &graph = instance.GetComputationalDag();

        const VertexType n = instance.NumberOfVertices();

        for (VertexType vert = 0; vert < n; ++vert) {
            schedule.SetAssignedProcessor(vert, UINT_MAX);
        }

        VertexType numNodesPerThread = n / numThreads;
        std::vector<VertexType> startNodes;
        startNodes.reserve(numThreads + 1);
        VertexType startNode = 0;
        for (unsigned thr = 0; thr < numThreads; thr++) {
            startNodes.push_back(startNode);
            startNode += numNodesPerThread;
        }
        startNodes.push_back(n);

        static constexpr unsigned unsignedPadding = (CACHE_LINE_SIZE + sizeof(unsigned) - 1) / sizeof(unsigned);
        std::vector<unsigned> superstepsThread(numThreads * unsignedPadding, 0);
        std::vector<unsigned> supstepIncr(numThreads, 0);
        unsigned incr = 0;

        std::vector<VertexType> topOrder;
        if constexpr (not hasVerticesInTopOrderV<GraphT>) {
            topOrder = GetTopOrder(graph);
        }

        std::vector<VertexType> posInTopOrder;
        if constexpr (not hasVerticesInTopOrderV<GraphT>) {
            posInTopOrder = std::vector<VertexType>(graph.NumVertices());
            for (VertexType ind = 0; ind < static_cast<VertexType>(topOrder.size()); ++ind) {
                posInTopOrder[topOrder[ind]] = ind;
            }
        }

#pragma omp parallel num_threads(numThreads) default(none)                                                 \
    shared(schedule, topOrder, posInTopOrder, superstepsThread, supstepIncr, numThreads, startNodes, incr)
        {
#pragma omp for schedule(static, 1)
            for (unsigned thr = 0; thr < numThreads; thr++) {
                ComputePartialSchedule(
                    schedule, topOrder, posInTopOrder, startNodes[thr], startNodes[thr + 1], superstepsThread[thr * unsignedPadding]);
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
                if constexpr (hasVerticesInTopOrderV<GraphT>) {
                    IncrementScheduleSupersteps(schedule, startNodes[thr], startNodes[thr + 1], supstepIncr[thr]);
                } else {
                    IncrementScheduleSuperstepsTopOrder(schedule, topOrder, startNodes[thr], startNodes[thr + 1], supstepIncr[thr]);
                }
            }
        }

        schedule.SetNumberOfSupersteps(incr);

        return ReturnStatus::OSP_SUCCESS;
    }

    /**
     * @brief Compute a schedule for the given BspInstance.
     *
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     *
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        unsigned numThreads = params_.numThreads_;
        if (numThreads == 0) {
            // numThreads = static_cast<unsigned>(std::sqrt( static_cast<double>((schedule.GetInstance().NumberOfVertices() / 1000000)))) + 1;
            numThreads
                = static_cast<unsigned>(std::log2(static_cast<double>((schedule.GetInstance().NumberOfVertices() / 1000)))) + 1;
        }
        numThreads = std::min(numThreads, params_.maxNumThreads_);
        if (numThreads == 0) {
            numThreads = 1;
        }

        return ComputeScheduleParallel(schedule, numThreads);
    }

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "GrowLocalAutoCoresParallel" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string GetScheduleName() const override { return "GrowLocalAutoCoresParallel"; }
};

}    // namespace osp
