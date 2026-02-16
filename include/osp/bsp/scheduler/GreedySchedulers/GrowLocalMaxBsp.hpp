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

    /*! Vertices ready in current superstep */
    std::deque<VertexType> currentlyReady_;

    /*! For i = 1,2,..,staleness, the vertices in futureReady_[(superstep + i) % staleness] becomes ready globally in superstep + i */
    std::array<std::deque<VertexType>, staleness> futureReady_;
    /*! Vertices to be added to futureReady_[superstep % staleness] which become ready globally in superstep + staleness */
    std::deque<VertexType> bestFutureReady_;

    /*! Local to processor ready vertices in current superstep in a heap */
    std::vector<std::vector<std::pair<VertexType, unsigned>>> currentProcReadyHeaps_;
    /*! Leftover local to processor ready vertices in current superstep in a heap */
    std::vector<std::vector<std::pair<VertexType, unsigned>>> bestCurrentProcReadyHeaps_;

    /*! For i = 0,1,2,..,staleness-1 and p processor, the vertices in procReady_[(superstep + i) % staleness][p] are ready locally
     * in superstep + i on processor p */
    std::array<std::vector<std::vector<std::pair<VertexType, unsigned>>>, staleness> procReady_;
    /*! Additions to procReady_ in current superstep attempt */
    std::array<std::vector<std::vector<std::pair<VertexType, unsigned>>>, staleness> procReadyAdditions_;
    /*! Additions to procReady_ from best superstep attempt */
    std::array<std::vector<std::vector<std::pair<VertexType, unsigned>>>, staleness> bestProcReadyAdditions_;

    void Init(const unsigned numProcs);
    void ReleaseMemory();

    inline typename std::deque<VertexType>::difference_type MaxAllReadyUsage(const std::deque<VertexType> &currentlyReady,
                                                                             const std::deque<VertexType> &nextSuperstepReady) const;

    bool ChanceToFinish(const unsigned superStep) const;

  public:
    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override;
    ReturnStatus ComputeSchedule(MaxBspSchedule<GraphT> &schedule) override;

    inline GrowLocalSSPParams<VertexIdxT<GraphT>, VWorkwT<GraphT>> &GetParameters();
    inline const GrowLocalSSPParams<VertexIdxT<GraphT>, VWorkwT<GraphT>> &GetParameters() const;

    std::string GetScheduleName() const override { return "GrowLocalSSP"; }
};

template <typename GraphT>
inline GrowLocalSSPParams<VertexIdxT<GraphT>, VWorkwT<GraphT>> &GrowLocalSSP<GraphT>::GetParameters() {
    return params_;
}

template <typename GraphT>
inline const GrowLocalSSPParams<VertexIdxT<GraphT>, VWorkwT<GraphT>> &GrowLocalSSP<GraphT>::GetParameters() const {
    return params_;
}

template <typename GraphT>
void GrowLocalSSP<GraphT>::Init(const unsigned numProcs) {
    currentlyReady_.clear();

    for (auto &stepFutureReady : futureReady_) {
        stepFutureReady.clear();
    }

    bestFutureReady_.clear();

    currentProcReadyHeaps_ = std::vector<std::vector<std::pair<VertexType, unsigned>>>(numProcs);
    bestCurrentProcReadyHeaps_ = std::vector<std::vector<std::pair<VertexType, unsigned>>>(numProcs);

    for (auto &stepProcReady : procReady_) {
        stepProcReady = std::vector<std::vector<std::pair<VertexType, unsigned>>>(numProcs);
    }

    for (auto &stepProcReadyAdditions : procReadyAdditions_) {
        stepProcReadyAdditions = std::vector<std::vector<std::pair<VertexType, unsigned>>>(numProcs);
    }

    for (auto &stepBestProcReadyAdditions : bestProcReadyAdditions_) {
        stepBestProcReadyAdditions = std::vector<std::vector<std::pair<VertexType, unsigned>>>(numProcs);
    }
}

template <typename GraphT>
void GrowLocalSSP<GraphT>::ReleaseMemory() {
    currentlyReady_.clear();
    currentlyReady_.shrink_to_fit();

    for (auto &stepFutureReady : futureReady_) {
        stepFutureReady.clear();
        stepFutureReady.shrink_to_fit();
    }

    bestFutureReady_.clear();

    currentProcReadyHeaps_.clear();
    currentProcReadyHeaps_.shrink_to_fit();

    bestCurrentProcReadyHeaps_.clear();
    bestCurrentProcReadyHeaps_.shrink_to_fit();

    for (auto &stepProcReady : procReady_) {
        stepProcReady.clear();
        stepProcReady.shrink_to_fit();
    }

    for (auto &stepProcReadyAdditions : procReadyAdditions_) {
        stepProcReadyAdditions.clear();
        stepProcReadyAdditions.shrink_to_fit();
    }

    for (auto &stepBestProcReadyAdditions : bestProcReadyAdditions_) {
        stepBestProcReadyAdditions.clear();
        stepBestProcReadyAdditions.shrink_to_fit();
    }
}

template <typename GraphT>
inline typename std::deque<VertexIdxT<GraphT>>::difference_type GrowLocalSSP<GraphT>::MaxAllReadyUsage(
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
bool GrowLocalSSP<GraphT>::ChanceToFinish(const unsigned superStep) const {
    bool ans = std::all_of(futureReady_.cbegin(), futureReady_.cend(), [](const auto &deq) { return deq.empty(); });

    if (ans) {
        for (unsigned i = 1U; i < staleness; ++i) {
            const auto &stepProcReady = procReady_[(i + superStep) % staleness];
            ans = std::all_of(stepProcReady.cbegin(), stepProcReady.cend(), [](const auto &vec) { return vec.empty(); });
            if (not ans) {
                break;
            }
        }
    }

    if (ans) {
        for (unsigned i = 1U; i < staleness; ++i) {
            const auto &stepProcReadyAdditions = procReadyAdditions_[(i + superStep) % staleness];
            ans = std::all_of(
                stepProcReadyAdditions.cbegin(), stepProcReadyAdditions.cend(), [](const auto &vec) { return vec.empty(); });
            if (not ans) {
                break;
            }
        }
    }

    return ans;
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

    Init(numProcs);

    std::vector<VertexType> predec(numVertices);
    for (const auto vert : graph.Vertices()) {
        predec[vert] = graph.InDegree(vert);
        if (predec[vert] == 0U) {
            currentlyReady_.emplace_back(vert);
        }
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

        std::deque<VertexType> &stepFutureReady = futureReady_[reducedSuperStep];

        const typename std::deque<VertexType>::difference_type maxCurrentlyReadyUsage
            = std::max(static_cast<typename std::deque<VertexType>::difference_type>(
                           static_cast<double>(params_.minSuperstepSize_) * desiredParallelism),
                       MaxAllReadyUsage(currentlyReady_, futureReady_[(superStep + 1U) % staleness]));

        std::vector<std::vector<std::pair<VertexType, unsigned>>> &stepProcReady = procReady_[reducedSuperStep];
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
            currentProcReadyHeaps_ = stepProcReady;

            currentlyReadyIter = currentlyReady_.cbegin();

            for (auto &stepProcReadyAdditions : procReadyAdditions_) {
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
                std::vector<std::pair<VertexType, unsigned>> &proc0Heap = currentProcReadyHeaps_[proc0];
                VertexType chosenNode = std::numeric_limits<VertexType>::max();
                {
                    if (proc0Heap.size() != 0U) {
                        std::pop_heap(proc0Heap.begin(), proc0Heap.end(), std::greater<>{});
                        chosenNode = proc0Heap.back().first;
                        proc0Heap.pop_back();
                    } else if (currentlyReadyIter != currentlyReady_.cend()) {
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
                            procReadyAdditions_[earliest % staleness][proc0].emplace_back(succ, superStep + staleness);
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
                    std::vector<std::pair<VertexType, unsigned>> &procHeap = currentProcReadyHeaps_[proc];
                    VertexType chosenNode = std::numeric_limits<VertexType>::max();
                    {
                        if (procHeap.size() != 0U) {
                            std::pop_heap(procHeap.begin(), procHeap.end(), std::greater<>{});
                            chosenNode = procHeap.back().first;
                            procHeap.pop_back();
                        } else if (currentlyReadyIter != currentlyReady_.cend()) {
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
                                procReadyAdditions_[earliest % staleness][proc].emplace_back(succ, superStep + staleness);
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

            if (currentlyReadyIter == currentlyReady_.cend()) {
                continueSuperstepAttemps = false;
            }

            if (continueSuperstepAttemps) {
                if (std::distance(currentlyReady_.cbegin(), currentlyReadyIter) > maxCurrentlyReadyUsage) {
                    if (not((totalAssigned + newTotalAssigned >= (numVertices / 4) * 3) && ChanceToFinish(superStep))) {
                        continueSuperstepAttemps = false;
                    }
                }
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
                std::swap(bestFutureReady_, stepFutureReady);
                std::swap(bestProcReadyAdditions_, procReadyAdditions_);
                std::swap(bestcurrentlyReadyIter, currentlyReadyIter);
                std::swap(bestNewAssignments, newAssignments);
                std::swap(bestCurrentProcReadyHeaps_, currentProcReadyHeaps_);
            }

            limit++;
            limit += (limit / 2);
        }

        // apply best iteration
        currentlyReady_.erase(currentlyReady_.begin(), bestcurrentlyReadyIter);
        std::swap(futureReady_[reducedSuperStep], bestFutureReady_);

        for (auto &localProcReady : procReady_[reducedSuperStep]) {
            localProcReady.clear();
        }

        const unsigned nextSuperStep = superStep + 1U;
        for (unsigned proc = 0U; proc < numProcs; ++proc) {
            for (const auto &vertStepPair : bestCurrentProcReadyHeaps_[proc]) {
                if (vertStepPair.second <= nextSuperStep) {
                    futureReady_[nextSuperStep % staleness].emplace_back(vertStepPair.first);
                } else {
                    procReady_[nextSuperStep % staleness][proc].emplace_back(vertStepPair);
                }
            }
        }

        for (std::size_t stepInd = 0U; stepInd < staleness; ++stepInd) {
            for (unsigned proc = 0U; proc < numProcs; ++proc) {
                procReady_[stepInd][proc].insert(procReady_[stepInd][proc].end(),
                                                 bestProcReadyAdditions_[stepInd][proc].begin(),
                                                 bestProcReadyAdditions_[stepInd][proc].end());
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

        std::deque<VertexType> &nextStepFutureReady = futureReady_[nextSuperStep % staleness];
        std::sort(nextStepFutureReady.begin(), nextStepFutureReady.end(), std::less<>{});
        const typename std::deque<VertexType>::difference_type lengthCurrentlyReady
            = std::distance(currentlyReady_.begin(), currentlyReady_.end());
        currentlyReady_.insert(currentlyReady_.end(), nextStepFutureReady.begin(), nextStepFutureReady.end());
        std::inplace_merge(currentlyReady_.begin(),
                           std::next(currentlyReady_.begin(), lengthCurrentlyReady),
                           currentlyReady_.end(),
                           std::less<>{});
        nextStepFutureReady.clear();

        ++superStep;
        desiredParallelism = (0.3 * desiredParallelism) + (0.6 * bestParallelism)
                             + (0.1 * static_cast<double>(numProcs));    // weights should sum up to one
    }

    schedule.SetNumberOfSupersteps(superStep);
    ReleaseMemory();

    return ReturnStatus::OSP_SUCCESS;
}

}    // end namespace osp
