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

#include "HeavyEdgePreProcess.hpp"
#include "VariancePartitioner.hpp"

namespace osp {

template <typename GraphT, typename InterpolationT, typename MemoryConstraintT = NoMemoryConstraint>
class LightEdgeVariancePartitioner : public VariancePartitioner<GraphT, InterpolationT, MemoryConstraintT> {
  private:
    using VertexType = VertexIdxT<GraphT>;

    struct VarianceCompare {
        bool operator()(const std::pair<VertexType, double> &lhs, const std::pair<VertexType, double> &rhs) const {
            return ((lhs.second > rhs.second) || ((lhs.second >= rhs.second) && (lhs.first < rhs.first)));
        }
    };

    /// @brief if an edge weights more than this multiple of the median, it is considered heavy
    double heavyIsXTimesMedian_;

    /// @brief the minimal percentage of components retained after heavy edge glueing
    double minPercentComponentsRetained_;

    /// @brief bound on the computational weight of any component as a percentage of average total work weight per core
    double boundComponentWeightPercent_;

  public:
    LightEdgeVariancePartitioner(double maxPercentIdleProcessors = 0.2,
                                 double variancePower = 2,
                                 double heavyIsXTimesMedian = 5.0,
                                 double minPercentComponentsRetained = 0.8,
                                 double boundComponentWeightPercent = 0.7,
                                 bool increaseParallelismInNewSuperstep = true,
                                 float maxPriorityDifferencePercent = 0.34f,
                                 float slack = 0.0f)
        : VariancePartitioner<GraphT, InterpolationT, MemoryConstraintT>(
              maxPercentIdleProcessors, variancePower, increaseParallelismInNewSuperstep, maxPriorityDifferencePercent, slack),
          heavyIsXTimesMedian_(heavyIsXTimesMedian),
          minPercentComponentsRetained_(minPercentComponentsRetained),
          boundComponentWeightPercent_(boundComponentWeightPercent) {};

    virtual ~LightEdgeVariancePartitioner() = default;

    std::string GetScheduleName() const override { return "LightEdgeVariancePartitioner"; };

    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        // DAGPartition output_partition(instance);

        using Base = VariancePartitioner<GraphT, InterpolationT, MemoryConstraintT>;

        const auto &instance = schedule.GetInstance();
        const auto &nVert = instance.NumberOfVertices();
        const unsigned &nProcessors = instance.NumberOfProcessors();
        const auto &graph = instance.GetComputationalDag();

        unsigned superstep = 0;

        if constexpr (isMemoryConstraintV<MemoryConstraintT>) {
            Base::memoryConstraint_.Initialize(instance);
        } else if constexpr (isMemoryConstraintScheduleV<MemoryConstraintT>) {
            Base::memoryConstraint_.Initialize(schedule, superstep);
        }

        std::vector<bool> hasVertexBeenAssigned(nVert, false);

        std::set<std::pair<VertexType, double>, VarianceCompare> ready;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(nProcessors);
        std::set<std::pair<VertexType, double>, VarianceCompare> allReady;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReadyPrior(nProcessors);

        std::vector<unsigned> whichProcReadyPrior(nVert, nProcessors);

        std::vector<double> variancePriorities = Base::ComputeWorkVariance(graph, Base::variancePower_);
        std::vector<VertexType> numUnallocatedParents(nVert, 0);

        VWorkwT<GraphT> totalWork = 0;
        for (const auto &v : graph.Vertices()) {
            schedule.SetAssignedProcessor(v, nProcessors);

            totalWork += graph.VertexWorkWeight(v);

            if (IsSource(v, graph)) {
                ready.insert(std::make_pair(v, variancePriorities[v]));
                allReady.insert(std::make_pair(v, variancePriorities[v]));

            } else {
                numUnallocatedParents[v] = graph.InDegree(v);
            }
        }

        std::vector<VWorkwT<GraphT>> totalPartitionWork(nProcessors, 0);
        std::vector<VWorkwT<GraphT>> superstepPartitionWork(nProcessors, 0);

        std::vector<std::vector<VertexType>> preprocessedPartition = HeavyEdgePreprocess(
            graph, heavyIsXTimesMedian_, minPercentComponentsRetained_, boundComponentWeightPercent_ / nProcessors);

        std::vector<size_t> whichPreprocessPartition(graph.NumVertices());
        for (size_t i = 0; i < preprocessedPartition.size(); i++) {
            for (const VertexType &vert : preprocessedPartition[i]) {
                whichPreprocessPartition[vert] = i;
            }
        }

        std::vector<VMemwT<GraphT>> memoryCostOfPreprocessedPartition(preprocessedPartition.size(), 0);
        for (size_t i = 0; i < preprocessedPartition.size(); i++) {
            for (const auto &vert : preprocessedPartition[i]) {
                memoryCostOfPreprocessedPartition[i] += graph.VertexMemWeight(vert);
            }
        }

        std::vector<VCommwT<GraphT>> transientCostOfPreprocessedPartition(preprocessedPartition.size(), 0);
        for (size_t i = 0; i < preprocessedPartition.size(); i++) {
            for (const auto &vert : preprocessedPartition[i]) {
                transientCostOfPreprocessedPartition[i]
                    = std::max(transientCostOfPreprocessedPartition[i], graph.VertexCommWeight(vert));
            }
        }

        std::set<unsigned> freeProcessors;

        bool endsuperstep = false;
        unsigned numUnableToPartitionNodeLoop = 0;

        while (!ready.empty()) {
            // Increase memory capacity if needed
            if (numUnableToPartitionNodeLoop == 1) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - unable to schedule.\n";
            } else {
                if constexpr (Base::useMemoryConstraint_) {
                    if (numUnableToPartitionNodeLoop >= 2) {
                        return ReturnStatus::ERROR;
                    }
                }
            }

            // Checking if new superstep is needed
            // std::cout << "freeprocessor " << freeProcessors.size() << " idle thresh " << maxPercentIdleProcessors_
            // * nProcessors << " ready size " << ready.size() << " small increase " << 1.2 * (nProcessors -
            // freeProcessors.size()) << " large increase " << nProcessors - freeProcessors.size() +  (0.5 *
            // freeProcessors.size()) << "\n";
            if (numUnableToPartitionNodeLoop == 0
                && static_cast<double>(freeProcessors.size()) > Base::maxPercentIdleProcessors_ * nProcessors
                && ((!Base::increaseParallelismInNewSuperstep_) || ready.size() >= nProcessors
                    || static_cast<double>(ready.size()) >= 1.2 * (nProcessors - static_cast<double>(freeProcessors.size()))
                    || static_cast<double>(ready.size()) >= nProcessors - static_cast<double>(freeProcessors.size())
                                                                + (0.5 * static_cast<double>(freeProcessors.size())))) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - parallelism.\n";
            }

            std::vector<float> processorPriorities
                = Base::ComputeProcessorPrioritiesInterpolation(superstepPartitionWork, totalPartitionWork, totalWork, instance);

            float minPriority = processorPriorities[0];
            float maxPriority = processorPriorities[0];
            for (const auto &prio : processorPriorities) {
                minPriority = std::min(minPriority, prio);
                maxPriority = std::max(maxPriority, prio);
            }
            if (numUnableToPartitionNodeLoop == 0
                && (maxPriority - minPriority)
                       > Base::maxPriorityDifferencePercent_ * static_cast<float>(totalWork) / static_cast<float>(nProcessors)) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - difference.\n";
            }

            // Introducing new superstep
            if (endsuperstep) {
                allReady = ready;
                for (unsigned proc = 0; proc < nProcessors; proc++) {
                    for (const auto &item : procReady[proc]) {
                        procReadyPrior[proc].insert(item);
                        whichProcReadyPrior[item.first] = proc;
                    }
                    procReady[proc].clear();

                    superstepPartitionWork[proc] = 0;
                }
                freeProcessors.clear();

                if constexpr (Base::useMemoryConstraint_) {
                    for (unsigned proc = 0; proc < nProcessors; proc++) {
                        Base::memoryConstraint_.Reset(proc);
                    }
                }

                superstep += 1;
                endsuperstep = false;
            }

            bool assignedANode = false;

            // Choosing next processor
            std::vector<unsigned> processorsInOrder
                = Base::ComputeProcessorPriority(superstepPartitionWork, totalPartitionWork, totalWork, instance, Base::slack_);

            for (unsigned &proc : processorsInOrder) {
                if ((freeProcessors.find(proc)) != freeProcessors.cend()) {
                    continue;
                }

                // Check for too many free processors - needed here because free processors may not have been detected
                // yet
                if (numUnableToPartitionNodeLoop == 0
                    && static_cast<double>(freeProcessors.size()) > this->maxPercentIdleProcessors_ * nProcessors
                    && ((!this->increaseParallelismInNewSuperstep_) || ready.size() >= nProcessors
                        || static_cast<double>(ready.size()) >= 1.2 * (nProcessors - static_cast<double>(freeProcessors.size()))
                        || static_cast<double>(ready.size()) >= nProcessors - static_cast<double>(freeProcessors.size())
                                                                    + (0.5 * static_cast<double>(freeProcessors.size())))) {
                    endsuperstep = true;
                    // std::cout << "\nCall for new superstep - parallelism.\n";
                    break;
                }

                assignedANode = false;

                // Choosing next node
                VertexType nextNode;
                for (auto vertexPriorPairIter = procReady[proc].begin(); vertexPriorPairIter != procReady[proc].end();
                     vertexPriorPairIter++) {
                    if (assignedANode) {
                        break;
                    }

                    const VertexType &vert = vertexPriorPairIter->first;
                    if constexpr (Base::useMemoryConstraint_) {
                        if (hasVertexBeenAssigned[vert]
                            || Base::memoryConstraint_.CanAdd(
                                proc,
                                memoryCostOfPreprocessedPartition[whichPreprocessPartition[vert]],
                                transientCostOfPreprocessedPartition[whichPreprocessPartition[vert]])) {
                            nextNode = vert;
                            assignedANode = true;
                        }
                    } else {
                        nextNode = vert;
                        assignedANode = true;
                    }
                }

                for (auto vertexPriorPairIter = procReadyPrior[proc].begin();
                     vertexPriorPairIter != procReadyPrior[proc].end();
                     vertexPriorPairIter++) {
                    if (assignedANode) {
                        break;
                    }

                    const VertexType &vert = vertexPriorPairIter->first;
                    if constexpr (Base::useMemoryConstraint_) {
                        if (hasVertexBeenAssigned[vert]
                            || Base::memoryConstraint_.CanAdd(
                                proc,
                                memoryCostOfPreprocessedPartition[whichPreprocessPartition[vert]],
                                transientCostOfPreprocessedPartition[whichPreprocessPartition[vert]])) {
                            nextNode = vert;
                            assignedANode = true;
                        }
                    } else {
                        nextNode = vert;
                        assignedANode = true;
                    }
                }
                for (auto vertexPriorPairIter = allReady.begin(); vertexPriorPairIter != allReady.cend();
                     vertexPriorPairIter++) {
                    if (assignedANode) {
                        break;
                    }

                    const VertexType &vert = vertexPriorPairIter->first;
                    if constexpr (Base::useMemoryConstraint_) {
                        if (hasVertexBeenAssigned[vert]
                            || Base::memoryConstraint_.CanAdd(
                                proc,
                                memoryCostOfPreprocessedPartition[whichPreprocessPartition[vert]],
                                transientCostOfPreprocessedPartition[whichPreprocessPartition[vert]])) {
                            nextNode = vert;
                            assignedANode = true;
                        }
                    } else {
                        nextNode = vert;
                        assignedANode = true;
                    }
                }

                if (!assignedANode) {
                    freeProcessors.insert(proc);
                } else {
                    // Assignments
                    if (hasVertexBeenAssigned[nextNode]) {
                        unsigned procAllocPrior = schedule.AssignedProcessor(nextNode);

                        // std::cout << "Allocated node " << nextNode << " to processor " << procAllocPrior << "
                        // previously.\n";

                        schedule.SetAssignedSuperstep(nextNode, superstep);

                        numUnableToPartitionNodeLoop = 0;

                        // Updating loads
                        superstepPartitionWork[procAllocPrior] += graph.VertexWorkWeight(nextNode);

                        // Deletion from Queues
                        std::pair<VertexType, double> pair = std::make_pair(nextNode, variancePriorities[nextNode]);
                        ready.erase(pair);
                        procReady[proc].erase(pair);
                        procReadyPrior[proc].erase(pair);
                        allReady.erase(pair);
                        if (whichProcReadyPrior[nextNode] != nProcessors) {
                            procReadyPrior[whichProcReadyPrior[nextNode]].erase(pair);
                        }

                        // Checking children
                        for (const auto &chld : graph.Children(nextNode)) {
                            numUnallocatedParents[chld] -= 1;
                            if (numUnallocatedParents[chld] == 0) {
                                // std::cout << "Inserting child " << chld << " into ready.\n";
                                ready.insert(std::make_pair(chld, variancePriorities[chld]));
                                bool isProcReady = true;
                                for (const auto &parent : graph.Parents(chld)) {
                                    if ((schedule.AssignedProcessor(parent) != procAllocPrior)
                                        && (schedule.AssignedSuperstep(parent) == superstep)) {
                                        isProcReady = false;
                                        break;
                                    }
                                }
                                if (isProcReady) {
                                    procReady[procAllocPrior].insert(std::make_pair(chld, variancePriorities[chld]));
                                    // std::cout << "Inserting child " << chld << " into procReady for processor " <<
                                    // procAllocPrior << ".\n";
                                }
                            }
                        }
                    } else {
                        schedule.SetAssignedProcessor(nextNode, proc);
                        hasVertexBeenAssigned[nextNode] = true;
                        // std::cout << "Allocated node " << nextNode << " to processor " << proc << ".\n";

                        schedule.SetAssignedSuperstep(nextNode, superstep);
                        numUnableToPartitionNodeLoop = 0;

                        // Updating loads
                        totalPartitionWork[proc] += graph.VertexWorkWeight(nextNode);
                        superstepPartitionWork[proc] += graph.VertexWorkWeight(nextNode);

                        if constexpr (Base::useMemoryConstraint_) {
                            Base::memoryConstraint_.Add(nextNode, proc);
                        }
                        // total_partition_memory[proc] += graph.VertexMemWeight(nextNode);
                        // transient_partition_memory[proc] =
                        //     std::max(transient_partition_memory[proc], graph.VertexCommWeight(nextNode));

                        // Deletion from Queues
                        std::pair<VertexType, double> pair = std::make_pair(nextNode, variancePriorities[nextNode]);
                        ready.erase(pair);
                        procReady[proc].erase(pair);
                        procReadyPrior[proc].erase(pair);
                        allReady.erase(pair);
                        if (whichProcReadyPrior[nextNode] != nProcessors) {
                            procReadyPrior[whichProcReadyPrior[nextNode]].erase(pair);
                        }

                        // Checking children
                        for (const auto &chld : graph.Children(nextNode)) {
                            numUnallocatedParents[chld] -= 1;
                            if (numUnallocatedParents[chld] == 0) {
                                // std::cout << "Inserting child " << chld << " into ready.\n";
                                ready.insert(std::make_pair(chld, variancePriorities[chld]));
                                bool isProcReady = true;
                                for (const auto &parent : graph.Parents(chld)) {
                                    if ((schedule.AssignedProcessor(parent) != proc)
                                        && (schedule.AssignedSuperstep(parent) == superstep)) {
                                        isProcReady = false;
                                        break;
                                    }
                                }
                                if (isProcReady) {
                                    procReady[proc].insert(std::make_pair(chld, variancePriorities[chld]));
                                    // std::cout << "Inserting child " << chld << " into procReady for processor " <<
                                    // proc << ".\n";
                                }
                            }
                        }

                        // Allocating all nodes in the same partition
                        for (VertexType nodeInSamePartition : preprocessedPartition[whichPreprocessPartition[nextNode]]) {
                            if (nodeInSamePartition == nextNode) {
                                continue;
                            }

                            // Allocation
                            schedule.SetAssignedProcessor(nodeInSamePartition, proc);
                            hasVertexBeenAssigned[nodeInSamePartition] = true;
                            // std::cout << "Allocated node " << nextNode << " to processor " << proc << ".\n";

                            // Update loads
                            totalPartitionWork[proc] += graph.VertexWorkWeight(nodeInSamePartition);

                            if constexpr (Base::useMemoryConstraint_) {
                                Base::memoryConstraint_.Add(nodeInSamePartition, proc);
                            }

                            // total_partition_memory[proc] += graph.VertexMemWeight(nodeInSamePartition);
                            // transient_partition_memory[proc] = std::max(
                            //     transient_partition_memory[proc], graph.VertexCommWeight(nodeInSamePartition));
                        }
                    }

                    break;
                }
            }
            if (!assignedANode) {
                numUnableToPartitionNodeLoop += 1;
            }
        }

        return ReturnStatus::OSP_SUCCESS;
    }
};

}    // namespace osp
