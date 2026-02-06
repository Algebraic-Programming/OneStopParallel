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
#include <limits>
#include <vector>

#include "osp/bsp/scheduler/MaxBspScheduler.hpp"

namespace osp {

template <typename VertT, typename WeightT>
struct GrowLocalSSPParams {
    VertT minSuperstepSize_ = 20;
    WeightT syncCostMultiplierMinSuperstepWeight_ = 1;
    WeightT syncCostMultiplierParallelCheck_ = 4;
};

template <typename GraphT>
class GrowLocalSSP : public MaxBspScheduler<GraphT> {
    static_assert(isDirectedGraphV<GraphT>);
    static_assert(hasVertexWeightsV<GraphT>);
    static_assert(hasVerticesInTopOrderV<GraphT>);
    static_assert(hasChildrenInVertexOrderV<GraphT>);

  private:
    using VertexType = VertexIdxT<GraphT>;

    constexpr std::size_t staleness{2U};
    GrowLocalSSPParams<VertexIdxT<GraphT>, VWorkwT<GraphT>> params_;

  public:
    ReturnStatus ComputeSchedule(MaxBspSchedule<GraphT> &schedule);

    std::string GetScheduleName() const override { return "GrowLocalSSP"; }
};

template <typename GraphT>
ReturnStatus GrowLocalSSP<GraphT>::ComputeSchedule(MaxBspSchedule<GraphT> &schedule) {
    const BspInstance<GraphT> &instance = schedule.GetInstance();
    const GraphT &graph = instance.GetComputationalDag();
    const VertexType numVertices = graph.NumVertices();
    const unsigned numProcs = instance.NumberOfProcessors();

    std::set<VertexType> currentlyReady;

    std::array<std::vector<VertexType>, staleness> futureReady;
    std::vector<VertexType> bestFutureReady;

    std::array<std::vector<std::set<VertexType>>, staleness> procReady(numProcs);
    std::vector<std::set<VertexType>> bestProcReady(numProcs);

    std::vector<VertexType> predec(numVertices);
    for (const auto vert : graph.Vertices()) {
        predec[vert] = graph.InDegree(vert);
        if (predec[vert] == 0U) {
            currentlyReady.insert(currentlyReady.end(), vert);
        }
    }

    std::vector<std::vector<VertexType>> newAssignments(numProcs);
    std::vector<std::vector<VertexType>> bestNewAssignments(numProcs);

    // const VWorkwT<GraphT> minWeightParallelCheck = params_.syncCostMultiplierParallelCheck_ * instance.SynchronisationCosts();
    // const VWorkwT<GraphT> minSuperstepWeight = params_.syncCostMultiplierMinSuperstepWeight_ * instance.SynchronisationCosts();
    // double desiredParallelism = static_cast<double>(numProcs);

    VertexType totalAssigned = 0;
    unsigned superStep = 0U;

    while (totalAssigned < numVertices) {
        unsigned reducedSuperStep = superStep % staleness;
        std::vector<std::set<VertexType>> &stepProcReady = procReady[reducedSuperStep];
        std::vector<VertexType> &stepFutureReady = futureReady[reducedSuperStep];

        VertexType limit = params_.minSuperstepSize_;
        double bestScore = 0;
        double bestParallelism = 0;

        typename std::set<VertexType>::const_iterator currentlyReadyIter;
        typename std::set<VertexType>::const_iterator bestcurrentlyReadyIter;

        bool continueSuperstepAttemps = true;

        while (continueSuperstepAttemps) {
            for (unsigned proc = 0; proc < p; proc++) {
                newAssignments[proc].clear();
            }
            stepFutureReady.clear();

            currentlyReadyIter = currentlyReady.cbegin();

            VertexType newTotalAssigned = 0;
            VWorkwT<GraphT> weightLimit = 0;
            VWorkwT<GraphT> totalWeightAssigned = 0;

            // Processor 0
            constexpr unsigned proc0{0U};
            while (newAssignments[proc0].size() < limit) {
                VertexType chosenNode = std::numeric_limits<VertexType>::max();
                {
                    const auto procReadyIt = stepProcReady[proc0].cbegin();
                    if (procReadyIt != stepProcReady[proc0].cend()) {
                        chosenNode = *procReadyIt;
                        stepProcReady[proc0].erase(procReadyIt);
                    } else if (currentlyReadyIter != currentlyReady.cend()) {
                        chosenNode = *currentlyReadyIter;
                        ++currentlyReadyIter;
                    } else {
                        break;
                    }
                }

                newAssignments[proc0].push_back(chosenNode);
                schedule.SetAssignedProcessor(chosenNode, proc0);
                ++newTotalAssigned;
                weightLimit += graph.VertexWorkWeight(chosenNode);

                for (const VertexType &succ : graph.Children(chosenNode)) {
                    if (--predec[succ] == 0) {
                        unsigned earliest = 0U;
                        bool differentProcParent = false;
                        for (const VertexType &par : graph.Parents(succ)) {
                            const bool differentProc = (schedule.AssignedProcessor(par) != proc0);
                            differentProcParent |= differentProc;
                            earliest = std::max(earliest, static_cast<unsigned>(differentProc) * schedule.AssignedSuperStep(par));
                        }
                        earliest += static_cast<unsigned>(differentProcParent) * staleness;

                        if (earliest <= superStep) {
                            stepProcReady[proc0].emplace(succ);
                        } else if (earliest < superStep + staleness) {
                            procReady[earliest % staleness][proc0].emplace(succ);
                        } else {
                            stepFutureReady.emplace_back(succ);
                        }
                    }
                }
            } // end while assigning

            totalWeightAssigned += weightLimit;

            // Processors 1 through P-1
            for (unsigned proc = 1U; proc < numProcs; ++proc) {
                VWorkwT<GraphT> currentWeightAssigned = 0;
                while (currentWeightAssigned < weightLimit) {
                    VertexType chosenNode = std::numeric_limits<VertexType>::max();
                    {
                        const auto procReadyIt = stepProcReady[proc].cbegin();
                        if (procReadyIt != stepProcReady[proc].cend()) {
                            chosenNode = *procReadyIt;
                            stepProcReady[proc].erase(procReadyIt);
                        } else if (currentlyReadyIter != currentlyReady.cend()) {
                            chosenNode = *currentlyReadyIter;
                            ++currentlyReadyIter;
                        } else {
                            break;
                        }
                    }

                    newAssignments[proc].push_back(chosenNode);
                    schedule.SetAssignedProcessor(chosenNode, proc);
                    ++newTotalAssigned;
                    currentWeightAssigned += graph.VertexWorkWeight(chosenNode);

                    for (const VertexType &succ : graph.Children(chosenNode)) {
                        if (--predec[succ] == 0) {
                            unsigned earliest = 0U;
                            bool differentProcParent = false;
                            for (const VertexType &par : graph.Parents(succ)) {
                                const bool differentProc = (schedule.AssignedProcessor(par) != proc);
                                differentProcParent |= differentProc;
                                earliest
                                    = std::max(earliest, static_cast<unsigned>(differentProc) * schedule.AssignedSuperStep(par));
                            }
                            earliest += static_cast<unsigned>(differentProcParent) * staleness;

                            if (earliest <= superStep) {
                                stepProcReady[proc].emplace(succ);
                            } else if (earliest < superStep + staleness) {
                                procReady[earliest % staleness][proc].emplace(succ);
                            } else {
                                stepFutureReady.emplace_back(succ);
                            }
                        }
                    }
                } // end while assigning
                weightLimit = std::max(weightLimit, currentWeightAssigned);
                totalWeightAssigned += currentWeightAssigned;
            } // end processor loops

            bool acceptStep = false;
        }
    }

    return ReturnStatus::OSP_SUCCESS;
}

}    // end namespace osp
