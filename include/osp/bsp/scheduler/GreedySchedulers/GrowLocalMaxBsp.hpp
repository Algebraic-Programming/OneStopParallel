/*
Copyright 2026 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos Matzoros, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <algorithm>
#include <array>
#include <deque>
#include <iterator>
#include <limits>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "osp/bsp/scheduler/MaxBspScheduler.hpp"

namespace osp {

template <typename VertT, typename WeightT>
struct GrowLocalSSPParams {
    VertT minSuperstepSize_ = 10;
    WeightT syncCostMultiplierMinSuperstepWeight_ = 1;
    WeightT syncCostMultiplierParallelCheck_ = 4;
};

template <typename GraphT>
class GrowLocalSSP : public MaxBspScheduler<GraphT> {
    static_assert(isDirectedGraphV<GraphT>);
    static_assert(hasVertexWeightsV<GraphT>);

  private:
    using VertexType = VertexIdxT<GraphT>;

    static constexpr unsigned staleness{2U};
    GrowLocalSSPParams<VertexIdxT<GraphT>, VWorkwT<GraphT>> params_;

    typename std::deque<VertexType>::difference_type maxAllReadyUsage(const std::deque<VertexType> &currentlyReady,
                                                                      const std::deque<VertexType> &nextSuperstepReady) const;

  public:
    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override;
    ReturnStatus ComputeSchedule(MaxBspSchedule<GraphT> &schedule) override;

    std::string GetScheduleName() const override { return "GrowLocalSSP"; }
};

template <typename GraphT>
typename std::deque<VertexIdxT<GraphT>>::difference_type GrowLocalSSP<GraphT>::maxAllReadyUsage(
    const std::deque<VertexIdxT<GraphT>> &currentlyReady, const std::deque<VertexIdxT<GraphT>> &nextSuperstepReady) const {
    if constexpr (staleness == 1U) {
        return std::distance(currentlyReady.cbegin(), currentlyReady.cend());
    } else {
        typename std::deque<VertexType>::difference_type lengthCurrently
            = std::distance(currentlyReady.cbegin(), currentlyReady.cend());
        typename std::deque<VertexType>::difference_type lengthNext
            = std::distance(nextSuperstepReady.cbegin(), nextSuperstepReady.cend());

        typename std::deque<VertexType>::difference_type ans = ((lengthCurrently + lengthNext + 1) / 2);

        return ans;
    }
}

template <typename GraphT>
ReturnStatus GrowLocalSSP<GraphT>::ComputeSchedule(BspSchedule<GraphT> &schedule) {
    return MaxBspScheduler<GraphT>::ComputeSchedule(schedule);
}

template <typename GraphT>
ReturnStatus GrowLocalSSP<GraphT>::ComputeSchedule(MaxBspSchedule<GraphT> &schedule) {
    const BspInstance<GraphT> &instance = schedule.GetInstance();
    const GraphT &graph = instance.GetComputationalDag();
    const VertexType numVertices = graph.NumVertices();
    const unsigned numProcs = instance.NumberOfProcessors();

    std::deque<VertexType> currentlyReady;    // vertices ready in current superstep

    std::array<std::deque<VertexType>, staleness> futureReady;
    // For i = 1,2,..,staleness, the vertices in futureReady[(superstep + i) % staleness] becomes ready globally in superstep + i
    std::deque<VertexType> bestFutureReady;
    // vertices to be added to futureReady[superstep % staleness] which become ready globally in superstep + staleness

    std::vector<std::vector<std::pair<VertexType, unsigned>>> currentProcReadyHeaps(numProcs);
    std::vector<std::vector<std::pair<VertexType, unsigned>>> bestCurrentProcReadyHeaps(numProcs);

    std::array<std::vector<std::vector<std::pair<VertexType, unsigned>>>, staleness> procReady;
    // For i = 0,1,2,..,staleness-1 and p processor, the vertices in procReady[(superstep + i) % staleness][p] are ready locally
    // in superstep + i on processor p
    std::array<std::vector<std::vector<std::pair<VertexType, unsigned>>>, staleness> procReadyAdditions;
    std::array<std::vector<std::vector<std::pair<VertexType, unsigned>>>, staleness> bestProcReadyAdditions;

    for (auto &arrVal : procReady) {
        arrVal = std::vector<std::vector<std::pair<VertexType, unsigned>>>(numProcs);
    }
    for (auto &arrVal : procReadyAdditions) {
        arrVal = std::vector<std::vector<std::pair<VertexType, unsigned>>>(numProcs);
    }
    for (auto &arrVal : bestProcReadyAdditions) {
        arrVal = std::vector<std::vector<std::pair<VertexType, unsigned>>>(numProcs);
    }

    std::vector<VertexType> predec(numVertices);
    for (const auto vert : graph.Vertices()) {
        predec[vert] = graph.InDegree(vert);
        if (predec[vert] == 0U) {
            currentlyReady.emplace_back(vert);
        }
    }
    if constexpr (not hasVerticesInTopOrderV<GraphT>) {
        std::sort(currentlyReady.begin(), currentlyReady.end(), std::less<>{});
    }

    std::vector<std::vector<VertexType>> newAssignments(numProcs);
    std::vector<std::vector<VertexType>> bestNewAssignments(numProcs);

    const VWorkwT<GraphT> minWeightParallelCheck = params_.syncCostMultiplierParallelCheck_ * instance.SynchronisationCosts();
    const VWorkwT<GraphT> minSuperstepWeight = params_.syncCostMultiplierMinSuperstepWeight_ * instance.SynchronisationCosts();

    double desiredParallelism = static_cast<double>(numProcs);

    VertexType totalAssigned = 0;
    unsigned superStep = 0U;

    while (totalAssigned < numVertices) {
        const unsigned reducedSuperStep = superStep % staleness;

        std::deque<VertexType> &stepFutureReady = futureReady[reducedSuperStep];
        std::sort(stepFutureReady.begin(), stepFutureReady.end(), std::less<>{});
        const typename std::deque<VertexType>::difference_type lengthCurrentlyReady
            = std::distance(currentlyReady.begin(), currentlyReady.end());
        currentlyReady.insert(currentlyReady.end(), stepFutureReady.begin(), stepFutureReady.end());
        std::inplace_merge(currentlyReady.begin(), std::next(currentlyReady.begin(), lengthCurrentlyReady), currentlyReady.end(), std::less<>{});

        const typename std::deque<VertexType>::difference_type maxCurrentlyReadyUsage
            = std::max(static_cast<typename std::deque<VertexType>::difference_type>(
                           static_cast<double>(params_.minSuperstepSize_) * desiredParallelism),
                       maxAllReadyUsage(currentlyReady, futureReady[(superStep + 1U) % staleness]));

        std::vector<std::vector<std::pair<VertexType, unsigned>>> &stepProcReady = procReady[reducedSuperStep];
        for (auto &procHeap : stepProcReady) {
            std::make_heap(procHeap.begin(), procHeap.end(), std::greater<>{});    // min heap
        }

        VertexType limit = params_.minSuperstepSize_;
        double bestScore = std::numeric_limits<double>::lowest();
        double bestParallelism = 0.0;

        typename std::deque<VertexType>::const_iterator currentlyReadyIter;
        typename std::deque<VertexType>::const_iterator bestcurrentlyReadyIter;

        bool continueSuperstepAttemps = true;

        while (continueSuperstepAttemps) {
            for (auto &procAssignments : newAssignments) {
                procAssignments.clear();
            }
            stepFutureReady.clear();
            currentProcReadyHeaps = stepProcReady;

            currentlyReadyIter = currentlyReady.cbegin();

            for (auto &stepProcReadyAdditions : procReadyAdditions) {
                for (auto &localStepProcReadyAdditions : stepProcReadyAdditions) {
                    localStepProcReadyAdditions.clear();
                }
            }

            VertexType newTotalAssigned = 0;
            VWorkwT<GraphT> weightLimit = 0;
            VWorkwT<GraphT> totalWeightAssigned = 0;

            // Processor 0
            constexpr unsigned proc0{0U};
            while (newAssignments[proc0].size() < limit) {
                std::vector<std::pair<VertexType, unsigned>> &proc0Heap = currentProcReadyHeaps[proc0];
                VertexType chosenNode = std::numeric_limits<VertexType>::max();
                {
                    if (proc0Heap.size() != 0U) {
                        std::pop_heap(proc0Heap.begin(), proc0Heap.end(), std::greater<>{});
                        chosenNode = proc0Heap.back().first;
                        proc0Heap.pop_back();
                    } else if (currentlyReadyIter != currentlyReady.cend()) {
                        chosenNode = *currentlyReadyIter;
                        ++currentlyReadyIter;
                    } else {
                        break;
                    }
                }

                newAssignments[proc0].push_back(chosenNode);
                schedule.SetAssignedProcessor(chosenNode, proc0);
                schedule.SetAssignedSuperstepNoUpdateNumSuperstep(chosenNode, superStep);
                ++newTotalAssigned;
                weightLimit += graph.VertexWorkWeight(chosenNode);

                for (const VertexType &succ : graph.Children(chosenNode)) {
                    if (--predec[succ] == 0) {
                        unsigned earliest = superStep;
                        for (const VertexType &par : graph.Parents(succ)) {
                            const bool sameProc = (schedule.AssignedProcessor(par) == proc0);
                            const unsigned constraint = sameProc ? superStep : schedule.AssignedSuperstep(par) + staleness;
                            earliest = std::max(earliest, constraint);
                        }

                        if (earliest <= superStep) {
                            proc0Heap.emplace_back(succ, superStep + staleness);
                            std::push_heap(proc0Heap.begin(), proc0Heap.end(), std::greater<>{});
                        } else if (earliest < superStep + staleness) {
                            procReadyAdditions[earliest % staleness][proc0].emplace_back(succ, superStep + staleness);
                        } else {
                            stepFutureReady.emplace_back(succ);
                        }
                    }
                }
            }    // end while assigning

            totalWeightAssigned += weightLimit;

            // Processors 1 through P-1
            for (unsigned proc = 1U; proc < numProcs; ++proc) {
                VWorkwT<GraphT> currentWeightAssigned = 0;
                while (currentWeightAssigned < weightLimit) {
                    std::vector<std::pair<VertexType, unsigned>> &procHeap = currentProcReadyHeaps[proc];
                    VertexType chosenNode = std::numeric_limits<VertexType>::max();
                    {
                        if (procHeap.size() != 0U) {
                            std::pop_heap(procHeap.begin(), procHeap.end(), std::greater<>{});
                            chosenNode = procHeap.back().first;
                            procHeap.pop_back();
                        } else if (currentlyReadyIter != currentlyReady.cend()) {
                            chosenNode = *currentlyReadyIter;
                            ++currentlyReadyIter;
                        } else {
                            break;
                        }
                    }

                    newAssignments[proc].push_back(chosenNode);
                    schedule.SetAssignedProcessor(chosenNode, proc);
                    schedule.SetAssignedSuperstepNoUpdateNumSuperstep(chosenNode, superStep);
                    ++newTotalAssigned;
                    currentWeightAssigned += graph.VertexWorkWeight(chosenNode);

                    for (const VertexType &succ : graph.Children(chosenNode)) {
                        if (--predec[succ] == 0) {
                            unsigned earliest = superStep;
                            for (const VertexType &par : graph.Parents(succ)) {
                                const bool sameProc = (schedule.AssignedProcessor(par) == proc);
                                const unsigned constraint = sameProc ? superStep : schedule.AssignedSuperstep(par) + staleness;
                                earliest = std::max(earliest, constraint);
                            }

                            if (earliest <= superStep) {
                                procHeap.emplace_back(succ, superStep + staleness);
                                std::push_heap(procHeap.begin(), procHeap.end(), std::greater<>{});
                            } else if (earliest < superStep + staleness) {
                                procReadyAdditions[earliest % staleness][proc].emplace_back(succ, superStep + staleness);
                            } else {
                                stepFutureReady.emplace_back(succ);
                            }
                        }
                    }
                }    // end while assigning
                weightLimit = std::max(weightLimit, currentWeightAssigned);
                totalWeightAssigned += currentWeightAssigned;
            }    // end processor loops

            bool acceptStep = false;

            double score
                = static_cast<double>(totalWeightAssigned) / static_cast<double>(weightLimit + instance.SynchronisationCosts());
            double parallelism = 0.0;
            if (weightLimit > 0) {
                parallelism = static_cast<double>(totalWeightAssigned) / static_cast<double>(weightLimit);
            }

            if (score > 0.99 * bestScore) {    // It is possible to make this less strict, i.e. score > 0.98 * best_score.
                                               // The purpose of this would be to encourage larger supersteps.
                bestScore = std::max(bestScore, score);
                bestParallelism = parallelism;
                acceptStep = true;
            } else {
                continueSuperstepAttemps = false;
            }

            if (weightLimit >= minWeightParallelCheck) {
                if (parallelism < std::max(2.0, 0.8 * desiredParallelism)) {
                    continueSuperstepAttemps = false;
                }
            }

            if (weightLimit <= minSuperstepWeight) {
                continueSuperstepAttemps = true;
                if (totalAssigned + newTotalAssigned == numVertices) {
                    acceptStep = true;
                    continueSuperstepAttemps = false;
                }
            }

            if (currentlyReadyIter == currentlyReady.cend()) {
                continueSuperstepAttemps = false;
            }

            if (std::distance(currentlyReady.cbegin(), currentlyReadyIter) > maxCurrentlyReadyUsage) {
                continueSuperstepAttemps = false;
            }

            if (totalAssigned + newTotalAssigned == numVertices) {
                continueSuperstepAttemps = false;
            }

            // Undo predec decreases
            for (const auto &newLocalAssignments : newAssignments) {
                for (const VertexType &node : newLocalAssignments) {
                    for (const VertexType &succ : graph.Children(node)) {
                        ++predec[succ];
                    }
                }
            }

            if (acceptStep) {
                std::swap(bestFutureReady, stepFutureReady);
                std::swap(bestProcReadyAdditions, procReadyAdditions);
                std::swap(bestcurrentlyReadyIter, currentlyReadyIter);
                std::swap(bestNewAssignments, newAssignments);
                std::swap(bestCurrentProcReadyHeaps, currentProcReadyHeaps);
            }

            limit++;
            limit += (limit / 2);
        }

        // apply best iteration
        currentlyReady.erase(currentlyReady.begin(), bestcurrentlyReadyIter);
        std::swap(futureReady[reducedSuperStep], bestFutureReady);

        for (auto &localProcReady : procReady[reducedSuperStep]) {
            localProcReady.clear();
        }

        const unsigned nextSuperStep = superStep + 1U;
        for (unsigned proc = 0U; proc < numProcs; ++proc) {
            for (const auto &vertStepPair : bestCurrentProcReadyHeaps[proc]) {
                if (vertStepPair.second <= nextSuperStep) {
                    futureReady[nextSuperStep % staleness].emplace_back(vertStepPair.first);
                } else {
                    procReady[nextSuperStep % staleness][proc].emplace_back(vertStepPair);
                }
            }
        }

        for (std::size_t stepInd = 0U; stepInd < staleness; ++stepInd) {
            for (unsigned proc = 0U; proc < numProcs; ++proc) {
                procReady[stepInd][proc].insert(procReady[stepInd][proc].end(),
                                                bestProcReadyAdditions[stepInd][proc].begin(),
                                                bestProcReadyAdditions[stepInd][proc].end());
            }
        }

        for (unsigned proc = 0U; proc < numProcs; ++proc) {
            totalAssigned += bestNewAssignments[proc].size();
            for (const VertexType &node : bestNewAssignments[proc]) {
                schedule.SetAssignedProcessor(node, proc);

                for (const VertexType &succ : graph.Children(node)) {
                    --predec[succ];
                }
            }
        }

        ++superStep;
        desiredParallelism = (0.3 * desiredParallelism) + (0.6 * bestParallelism)
                             + (0.1 * static_cast<double>(numProcs));    // weights should sum up to one
    }

    schedule.SetNumberOfSupersteps(superStep);

    return ReturnStatus::OSP_SUCCESS;
}

}    // end namespace osp
