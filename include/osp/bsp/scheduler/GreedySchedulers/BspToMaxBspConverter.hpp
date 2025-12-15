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
    static_assert(IsComputationalDagV<GraphT>, "GreedyBspToMaxBspConverter can only be used with computational DAGs.");
    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "GreedyBspToMaxBspConverter requires work and comm. weights to have the same type.");

  protected:
    using vertexIdx = VertexIdxT<GraphT>;
    using costType = VWorkwT<GraphT>;
    using KeyTriple = std::tuple<VertexIdxT<GraphT>, unsigned int, unsigned int>;

    double latencyCoefficient_ = 1.25;
    double decayFactor_ = 0.5;

    std::vector<std::vector<std::deque<VertexIdxT<GraphT>>>> CreateSuperstepLists(const BspScheduleCS<GraphT> &schedule,
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
    const GraphT &dag = schedule.GetInstance().GetComputationalDag();

    // Initialize data structures
    std::vector<double> priorities;
    std::vector<std::vector<std::deque<vertexIdx>>> procList = CreateSuperstepLists(schedule, priorities);
    std::vector<std::vector<costType>> workRemainingProcSuperstep(schedule.GetInstance().NumberOfProcessors(),
                                                                   std::vector<costType>(schedule.NumberOfSupersteps(), 0));
    std::vector<vertexIdx> nodesRemainingSuperstep(schedule.NumberOfSupersteps(), 0);

    MaxBspScheduleCS<GraphT> scheduleMax(schedule.GetInstance());
    for (vertexIdx node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        workRemainingProcSuperstep[schedule.AssignedProcessor(node)][schedule.AssignedSuperstep(node)]
            += dag.VertexWorkWeight(node);
        ++nodesRemainingSuperstep[schedule.AssignedSuperstep(node)];
        scheduleMax.SetAssignedProcessor(node, schedule.AssignedProcessor(node));
    }

    std::vector<std::vector<costType>> sendCommRemainingProcSuperstep(schedule.GetInstance().NumberOfProcessors(),
                                                                       std::vector<costType>(schedule.NumberOfSupersteps(), 0));
    std::vector<std::vector<costType>> recCommRemainingProcSuperstep(schedule.GetInstance().NumberOfProcessors(),
                                                                      std::vector<costType>(schedule.NumberOfSupersteps(), 0));

    std::vector<std::set<std::pair<KeyTriple, unsigned>>> freeCommStepsForSuperstep(schedule.NumberOfSupersteps());
    std::vector<std::vector<std::pair<KeyTriple, unsigned>>> dependentCommStepsForNode(schedule.GetInstance().NumberOfVertices());
    for (auto const &[key, val] : schedule.GetCommunicationSchedule()) {
        if (schedule.AssignedSuperstep(std::get<0>(key)) == val) {
            dependentCommStepsForNode[std::get<0>(key)].emplace_back(key, val);

            costType commCost = dag.VertexCommWeight(std::get<0>(key))
                                 * schedule.GetInstance().GetArchitecture().SendCosts(std::get<1>(key), std::get<2>(key));
            sendCommRemainingProcSuperstep[std::get<1>(key)][val] += commCost;
            recCommRemainingProcSuperstep[std::get<2>(key)][val] += commCost;
        } else {
            freeCommStepsForSuperstep[val].emplace(key, val);
        }
    }

    // Iterate through supersteps
    unsigned currentStep = 0;
    for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
        std::vector<costType> workDoneOnProc(schedule.GetInstance().NumberOfProcessors(), 0);
        costType maxWorkDone = 0;
        std::vector<std::pair<KeyTriple, unsigned>> newlyFreedCommSteps;
        std::vector<costType> sendSumOfNewlyFreeOnProc(schedule.GetInstance().NumberOfProcessors(), 0),
            recSumOfNewlyFreeOnProc(schedule.GetInstance().NumberOfProcessors(), 0);

        std::vector<std::pair<KeyTriple, unsigned>> commInCurrentStep;

        std::vector<costType> sendOnProc(schedule.GetInstance().NumberOfProcessors(), 0),
            recOnProc(schedule.GetInstance().NumberOfProcessors(), 0);
        bool emptySuperstep = (nodesRemainingSuperstep[step] == 0);

        while (nodesRemainingSuperstep[step] > 0) {
            // I. Select the next node (from any proc) with highest priority
            unsigned chosenProc = schedule.GetInstance().NumberOfProcessors();
            double bestPrio = std::numeric_limits<double>::max();

            for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
                if (!procList[proc][step].empty()
                    && (chosenProc == schedule.GetInstance().NumberOfProcessors()
                        || priorities[procList[proc][step].front()] < bestPrio)) {
                    chosenProc = proc;
                    bestPrio = priorities[procList[proc][step].front()];
                }
            }
            if (chosenProc == schedule.GetInstance().NumberOfProcessors()) {
                break;
            }

            vertexIdx chosenNode = procList[chosenProc][step].front();
            procList[chosenProc][step].pop_front();
            workDoneOnProc[chosenProc] += dag.VertexWorkWeight(chosenNode);
            workRemainingProcSuperstep[chosenProc][step] -= dag.VertexWorkWeight(chosenNode);
            maxWorkDone = std::max(maxWorkDone, workDoneOnProc[chosenProc]);
            scheduleMax.SetAssignedSuperstep(chosenNode, currentStep);
            --nodesRemainingSuperstep[step];
            for (const std::pair<KeyTriple, unsigned> &entry : dependentCommStepsForNode[chosenNode]) {
                newlyFreedCommSteps.push_back(entry);
                costType commCost
                    = dag.VertexCommWeight(chosenNode)
                      * schedule.GetInstance().GetArchitecture().SendCosts(std::get<1>(entry.first), std::get<2>(entry.first));
                sendSumOfNewlyFreeOnProc[std::get<1>(entry.first)] += commCost;
                recSumOfNewlyFreeOnProc[std::get<2>(entry.first)] += commCost;
            }

            // II. Add nodes on all other processors if this doesn't increase work cost
            for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
                if (proc == chosenProc) {
                    continue;
                }
                while (!procList[proc][step].empty()
                       && workDoneOnProc[proc] + dag.VertexWorkWeight(procList[proc][step].front()) <= maxWorkDone) {
                    vertexIdx node = procList[proc][step].front();
                    procList[proc][step].pop_front();
                    workDoneOnProc[proc] += dag.VertexWorkWeight(node);
                    workRemainingProcSuperstep[proc][step] -= dag.VertexWorkWeight(node);
                    scheduleMax.SetAssignedSuperstep(node, currentStep);
                    --nodesRemainingSuperstep[step];
                    for (const std::pair<KeyTriple, unsigned> &entry : dependentCommStepsForNode[node]) {
                        newlyFreedCommSteps.push_back(entry);
                        costType commCost = dag.VertexCommWeight(node)
                                              * schedule.GetInstance().GetArchitecture().SendCosts(std::get<1>(entry.first),
                                                                                                   std::get<2>(entry.first));
                        sendSumOfNewlyFreeOnProc[std::get<1>(entry.first)] += commCost;
                        recSumOfNewlyFreeOnProc[std::get<2>(entry.first)] += commCost;
                    }
                }
            }

            // III. Add communication steps that are already available
            for (auto itr = freeCommStepsForSuperstep[step].begin(); itr != freeCommStepsForSuperstep[step].end();) {
                if (sendOnProc[std::get<1>(itr->first)] < maxWorkDone && recOnProc[std::get<2>(itr->first)] < maxWorkDone) {
                    costType commCost
                        = dag.VertexCommWeight(std::get<0>(itr->first))
                          * schedule.GetInstance().GetArchitecture().SendCosts(std::get<1>(itr->first), std::get<2>(itr->first))
                          * schedule.GetInstance().GetArchitecture().CommunicationCosts();
                    sendOnProc[std::get<1>(itr->first)] += commCost;
                    recOnProc[std::get<2>(itr->first)] += commCost;
                    if (currentStep - 1 >= scheduleMax.NumberOfSupersteps()) {
                        scheduleMax.SetNumberOfSupersteps(currentStep);
                    }
                    scheduleMax.addCommunicationScheduleEntry(itr->first, currentStep - 1);
                    commInCurrentStep.emplace_back(*itr);
                    freeCommStepsForSuperstep[step].erase(itr++);
                } else {
                    ++itr;
                }
            }

            // IV. Decide whether to split superstep here
            if (!freeCommStepsForSuperstep[step].empty() || nodesRemainingSuperstep[step] == 0) {
                continue;
            }

            costType maxWorkRemaining = 0, maxCommRemaining = 0, commAfterReduction = 0;
            for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
                maxWorkRemaining = std::max(maxWorkRemaining, workRemainingProcSuperstep[proc][step]);
                maxCommRemaining = std::max(maxCommRemaining, sendCommRemainingProcSuperstep[proc][step]);
                maxCommRemaining = std::max(maxCommRemaining, recCommRemainingProcSuperstep[proc][step]);
                commAfterReduction = std::max(
                    commAfterReduction, sendCommRemainingProcSuperstep[proc][step] - sendSumOfNewlyFreeOnProc[proc]);
                commAfterReduction = std::max(
                    commAfterReduction, recCommRemainingProcSuperstep[proc][step] - recSumOfNewlyFreeOnProc[proc]);
            }
            costType commReduction
                = (maxCommRemaining - commAfterReduction) * schedule.GetInstance().GetArchitecture().CommunicationCosts();

            costType gain = std::min(commReduction, maxWorkRemaining);
            if (gain > 0
                && static_cast<double>(gain) >= static_cast<double>(schedule.GetInstance().GetArchitecture().SynchronisationCosts())
                                                    * latencyCoefficient_) {
                // Split superstep
                for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
                    workDoneOnProc[proc] = 0;
                    sendOnProc[proc] = 0;
                    recOnProc[proc] = 0;
                    sendSumOfNewlyFreeOnProc[proc] = 0;
                    recSumOfNewlyFreeOnProc[proc] = 0;
                }
                maxWorkDone = 0;
                for (const std::pair<KeyTriple, unsigned> &entry : newlyFreedCommSteps) {
                    freeCommStepsForSuperstep[step].insert(entry);

                    costType commCost = dag.VertexCommWeight(std::get<0>(entry.first))
                                          * schedule.GetInstance().GetArchitecture().SendCosts(std::get<1>(entry.first),
                                                                                               std::get<2>(entry.first));
                    sendCommRemainingProcSuperstep[std::get<1>(entry.first)][step] -= commCost;
                    recCommRemainingProcSuperstep[std::get<2>(entry.first)][step] -= commCost;
                }
                newlyFreedCommSteps.clear();
                commInCurrentStep.clear();
                ++currentStep;
            }
        }

        if (!emptySuperstep) {
            ++currentStep;
        }

        for (const std::pair<KeyTriple, unsigned> &entry : newlyFreedCommSteps) {
            freeCommStepsForSuperstep[step].insert(entry);
        }

        if (freeCommStepsForSuperstep[step].empty()) {
            continue;
        }

        // Handle the remaining communication steps: creating a new superstep afterwards with no work
        costType maxCommCurrent = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            maxCommCurrent = std::max(maxCommCurrent, sendOnProc[proc]);
            maxCommCurrent = std::max(maxCommCurrent, recOnProc[proc]);
        }
        sendOnProc.clear();
        sendOnProc.resize(schedule.GetInstance().NumberOfProcessors(), 0);
        recOnProc.clear();
        recOnProc.resize(schedule.GetInstance().NumberOfProcessors(), 0);

        std::set<std::pair<vertexIdx, unsigned>> lateArrivingNodes;
        for (const std::pair<KeyTriple, unsigned> &entry : freeCommStepsForSuperstep[step]) {
            scheduleMax.addCommunicationScheduleEntry(entry.first, currentStep - 1);
            costType commCost
                = dag.VertexCommWeight(std::get<0>(entry.first))
                  * schedule.GetInstance().GetArchitecture().SendCosts(std::get<1>(entry.first), std::get<2>(entry.first))
                  * schedule.GetInstance().GetArchitecture().CommunicationCosts();
            sendOnProc[std::get<1>(entry.first)] += commCost;
            recOnProc[std::get<2>(entry.first)] += commCost;
            lateArrivingNodes.emplace(std::get<0>(entry.first), std::get<2>(entry.first));
        }

        // Edge case - check if it is worth moving all communications from the current superstep to the next one instead (thus
        // saving a sync cost) (for this we need to compute the h-relation-max in the current superstep, the next superstep, and
        // also their union)
        costType maxCommAfter = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            maxCommAfter = std::max(maxCommAfter, sendOnProc[proc]);
            maxCommAfter = std::max(maxCommAfter, recOnProc[proc]);
        }

        for (const std::pair<KeyTriple, unsigned> &entry : commInCurrentStep) {
            costType commCost
                = dag.VertexCommWeight(std::get<0>(entry.first))
                  * schedule.GetInstance().GetArchitecture().SendCosts(std::get<1>(entry.first), std::get<2>(entry.first))
                  * schedule.GetInstance().GetArchitecture().CommunicationCosts();
            sendOnProc[std::get<1>(entry.first)] += commCost;
            recOnProc[std::get<2>(entry.first)] += commCost;
        }
        costType maxCommTogether = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            maxCommTogether = std::max(maxCommTogether, sendOnProc[proc]);
            maxCommTogether = std::max(maxCommTogether, recOnProc[proc]);
        }

        costType workLimit = maxCommAfter;
        if (maxCommTogether + maxWorkDone <= maxCommAfter + std::max(maxWorkDone, maxCommCurrent)
                                                   + schedule.GetInstance().GetArchitecture().SynchronisationCosts()) {
            workLimit = maxCommTogether;
            for (const std::pair<KeyTriple, unsigned> &entry : commInCurrentStep) {
                if (currentStep - 1 >= scheduleMax.NumberOfSupersteps()) {
                    scheduleMax.SetNumberOfSupersteps(currentStep);
                }
                scheduleMax.addCommunicationScheduleEntry(entry.first, currentStep - 1);
                lateArrivingNodes.emplace(std::get<0>(entry.first), std::get<2>(entry.first));
            }
        }

        // Bring computation steps into the extra superstep from the next superstep, if possible,a s long as it does not increase cost
        if (step == schedule.NumberOfSupersteps() - 1) {
            continue;
        }

        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            costType workSoFar = 0;
            std::set<vertexIdx> broughtForward;
            for (vertexIdx node : procList[proc][step + 1]) {
                if (workSoFar + dag.VertexWorkWeight(node) > workLimit) {
                    continue;
                }

                bool hasDependency = false;

                for (const vertexIdx &parent : dag.Parents(node)) {
                    if (schedule.AssignedProcessor(node) != schedule.AssignedProcessor(parent)
                        && lateArrivingNodes.find(std::make_pair(parent, proc)) != lateArrivingNodes.end()) {
                        hasDependency = true;
                    }

                    if (schedule.AssignedProcessor(node) == schedule.AssignedProcessor(parent)
                        && schedule.AssignedSuperstep(parent) == step + 1
                        && broughtForward.find(parent) == broughtForward.end()) {
                        hasDependency = true;
                    }
                }

                if (hasDependency) {
                    continue;
                }

                broughtForward.insert(node);
                workSoFar += dag.VertexWorkWeight(node);
                scheduleMax.SetAssignedSuperstep(node, currentStep);
                workRemainingProcSuperstep[proc][step + 1] -= dag.VertexWorkWeight(node);
                --nodesRemainingSuperstep[step + 1];

                for (const std::pair<KeyTriple, unsigned> &entry : dependentCommStepsForNode[node]) {
                    freeCommStepsForSuperstep[step + 1].insert(entry);
                }
            }

            std::deque<vertexIdx> remaining;
            for (vertexIdx node : procList[proc][step + 1]) {
                if (broughtForward.find(node) == broughtForward.end()) {
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
std::vector<std::vector<std::deque<VertexIdxT<GraphT>>>> GreedyBspToMaxBspConverter<GraphT>::CreateSuperstepLists(
    const BspScheduleCS<GraphT> &schedule, std::vector<double> &priorities) const {
    const GraphT &dag = schedule.GetInstance().GetComputationalDag();
    std::vector<vertexIdx> topOrder = GetTopOrder(dag);
    priorities.clear();
    priorities.resize(dag.NumVertices());
    std::vector<vertexIdx> localInDegree(dag.NumVertices(), 0);

    // compute for each node the amount of dependent send cost in the same superstep
    std::vector<costType> commDependency(dag.NumVertices(), 0);
    for (auto const &[key, val] : schedule.GetCommunicationSchedule()) {
        if (schedule.AssignedSuperstep(std::get<0>(key)) == val) {
            commDependency[std::get<0>(key)]
                += dag.VertexCommWeight(std::get<0>(key))
                   * schedule.GetInstance().GetArchitecture().SendCosts(std::get<1>(key), std::get<2>(key));
        }
    }

    // assign priority to nodes - based on their own work/comm ratio, and that of its successors in the same proc/supstep
    for (auto itr = topOrder.rbegin(); itr != topOrder.rend(); ++itr) {
        vertexIdx node = *itr;
        double base = static_cast<double>(dag.VertexWorkWeight(node));
        if (commDependency[node] > 0) {
            base /= static_cast<double>(2 * commDependency[node]);
        }

        double successors = 0;
        unsigned numChildren = 0;
        for (const vertexIdx &child : dag.Children(node)) {
            if (schedule.AssignedProcessor(node) == schedule.AssignedProcessor(child)
                && schedule.AssignedSuperstep(node) == schedule.AssignedSuperstep(child)) {
                ++numChildren;
                successors += priorities[child];
                ++localInDegree[child];
            }
        }
        if (numChildren > 0) {
            successors = successors * decayFactor_ / static_cast<double>(numChildren);
        }
        priorities[node] = base + successors;
    }

    // create lists for each processor-superstep pair, in a topological order, sorted by priority
    std::vector<std::vector<std::deque<vertexIdx>>> superstepLists(
        schedule.GetInstance().NumberOfProcessors(), std::vector<std::deque<vertexIdx>>(schedule.NumberOfSupersteps()));

    std::set<std::pair<double, vertexIdx>> free;
    for (vertexIdx node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        if (localInDegree[node] == 0) {
            free.emplace(priorities[node], node);
        }
    }
    while (!free.empty()) {
        vertexIdx node = free.begin()->second;
        free.erase(free.begin());
        superstepLists[schedule.AssignedProcessor(node)][schedule.AssignedSuperstep(node)].push_back(node);
        for (const vertexIdx &child : dag.Children(node)) {
            if (schedule.AssignedProcessor(node) == schedule.AssignedProcessor(child)
                && schedule.AssignedSuperstep(node) == schedule.AssignedSuperstep(child)) {
                if (--localInDegree[child] == 0) {
                    free.emplace(priorities[child], child);
                }
            }
        }
    }

    return superstepLists;
}

}    // namespace osp
