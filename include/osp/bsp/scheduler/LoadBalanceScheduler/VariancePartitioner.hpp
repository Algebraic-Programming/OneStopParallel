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

#include "LoadBalancerBase.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/MemoryConstraintModules.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

template <typename GraphT, typename InterpolationT, typename MemoryConstraintT = no_memory_constraint>
class VariancePartitioner : public LoadBalancerBase<GraphT, InterpolationT> {
    static_assert(IsComputationalDagV<GraphT>, "VariancePartitioner can only be used with computational DAGs.");

    using VertexType = VertexIdxT<GraphT>;

    struct VarianceCompare {
        bool operator()(const std::pair<VertexType, double> &lhs, const std::pair<VertexType, double> &rhs) const {
            return ((lhs.second > rhs.second) || ((lhs.second >= rhs.second) && (lhs.first < rhs.first)));
        }
    };

  protected:
    constexpr static bool useMemoryConstraint_ = IsMemoryConstraintV<MemoryConstraint_t>
                                                 or IsMemoryConstraintScheduleV<MemoryConstraint_t>;

    static_assert(not useMemoryConstraint_ or std::is_same_v<GraphT, typename MemoryConstraintT::Graph_impl_t>,
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

    MemoryConstraintT memoryConstraint_;

    /// @brief threshold percentage of idle processors as to when a new superstep should be introduced
    double maxPercentIdleProcessors_;

    /// @brief the power in the power mean average of the variance scheduler
    double variancePower_;

    /// @brief whether or not parallelism should be increased in the next superstep
    bool increaseParallelismInNewSuperstep_;

    /// @brief percentage of the average workload by which the processor priorities may diverge
    float maxPriorityDifferencePercent_;

    /// @brief how much to ignore the global work balance, value between 0 and 1
    float slack_;

    /// @brief Computes a power mean average of the bottom node distance
    /// @param graph graph
    /// @param power the power in the power mean average
    /// @return vector of the logarithm of power mean averaged bottom node distance
    std::vector<double> ComputeWorkVariance(const GraphT &graph, double power = 2) const {
        std::vector<double> workVariance(graph.NumVertices(), 0.0);

        const auto topOrder = GetTopOrder(graph);

        for (auto rIter = topOrder.rbegin(); rIter != topOrder.crend(); rIter++) {
            double temp = 0;
            double maxPriority = 0;
            for (const auto &child : graph.Children(*rIter)) {
                maxPriority = std::max(workVariance[child], maxPriority);
            }
            for (const auto &child : graph.Children(*rIter)) {
                temp += std::exp(power * (workVariance[child] - maxPriority));
            }
            temp = std::log(temp) / power + maxPriority;

            double nodeWeight = std::log(graph.VertexWorkWeight(*rIter));
            double largerVal = nodeWeight > temp ? nodeWeight : temp;

            workVariance[*rIter] = std::log(std::exp(nodeWeight - largerVal) + std::exp(temp - largerVal)) + largerVal;
        }

        return workVariance;
    }

  public:
    VariancePartitioner(double maxPercentIdleProcessors = 0.2,
                        double variancePower = 2.0,
                        bool increaseParallelismInNewSuperstep = true,
                        float maxPriorityDifferencePercent = 0.34f,
                        float slack = 0.0f)
        : maxPercentIdleProcessors_(maxPercentIdleProcessors),
          variancePower_(variancePower),
          increaseParallelismInNewSuperstep_(increaseParallelismInNewSuperstep),
          maxPriorityDifferencePercent_(maxPriorityDifferencePercent),
          slack_(slack) {};

    virtual ~VariancePartitioner() = default;

    virtual ReturnStatus computeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();
        const auto &nVert = instance.NumberOfVertices();
        const unsigned &nProcessors = instance.NumberOfProcessors();
        const auto &graph = instance.GetComputationalDag();

        unsigned superstep = 0;

        if constexpr (IsMemoryConstraintV<MemoryConstraint_t>) {
            memoryConstraint_.initialize(instance);
        } else if constexpr (IsMemoryConstraintScheduleV<MemoryConstraint_t>) {
            memoryConstraint_.initialize(schedule, superstep);
        }

        VWorkwT<GraphT> totalWork = 0;

        std::vector<VWorkwT<GraphT>> totalPartitionWork(nProcessors, 0);
        std::vector<VWorkwT<GraphT>> superstepPartitionWork(nProcessors, 0);

        std::vector<double> variancePriorities = ComputeWorkVariance(graph, variancePower_);
        std::vector<VertexType> numUnallocatedParents(nVert, 0);

        std::set<std::pair<VertexType, double>, VarianceCompare> ready;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(nProcessors);
        std::set<std::pair<VertexType, double>, VarianceCompare> allReady;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReadyPrior(nProcessors);

        std::vector<unsigned> whichProcReadyPrior(nVert, nProcessors);

        for (const auto &v : graph.vertices()) {
            schedule.SetAssignedProcessor(v, nProcessors);

            totalWork += graph.VertexWorkWeight(v);

            if (IsSource(v, graph)) {
                ready.insert(std::make_pair(v, variancePriorities[v]));
                allReady.insert(std::make_pair(v, variancePriorities[v]));

            } else {
                numUnallocatedParents[v] = graph.InDegree(v);
            }
        }

        std::set<unsigned> freeProcessors;

        bool endsuperstep = false;
        unsigned numUnableToPartitionNodeLoop = 0;
        // ReturnStatus status = ReturnStatus::OSP_SUCCESS;

        while (!ready.empty()) {
            // Increase memory capacity if needed
            if (numUnableToPartitionNodeLoop == 1) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - unable to schedule.\n";
            } else {
                if constexpr (useMemoryConstraint_) {
                    if (numUnableToPartitionNodeLoop >= 2) {
                        return ReturnStatus::ERROR;
                    }
                }
            }

            // Checking if new superstep is needed
            // std::cout << "freeprocessor " << free_processors.size() << " idle thresh " << max_percent_idle_processors
            // * n_processors << " ready size " << ready.size() << " small increase " << 1.2 * (n_processors -
            // free_processors.size()) << " large increase " << n_processors - free_processors.size() +  (0.5 *
            // free_processors.size()) << "\n";
            if (numUnableToPartitionNodeLoop == 0
                && static_cast<double>(freeProcessors.size()) > maxPercentIdleProcessors_ * nProcessors
                && ((!increaseParallelismInNewSuperstep_) || ready.size() >= nProcessors
                    || static_cast<double>(ready.size()) >= 1.2 * (nProcessors - static_cast<double>(freeProcessors.size()))
                    || static_cast<double>(ready.size()) >= nProcessors - static_cast<double>(freeProcessors.size())
                                                                + (0.5 * static_cast<double>(freeProcessors.size())))) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - parallelism.\n";
            }
            std::vector<float> processorPriorities
                = LoadBalancerBase<GraphT, InterpolationT>::computeProcessorPrioritiesInterpolation(
                    superstep_partition_work, total_partition_work, total_work, instance);
            float minPriority = processorPriorities[0];
            float maxPriority = processorPriorities[0];
            for (const auto &prio : processor_priorities) {
                min_priority = std::min(min_priority, prio);
                max_priority = std::max(max_priority, prio);
            }
            if (numUnableToPartitionNodeLoop == 0
                && (maxPriority - minPriority)
                       > maxPriorityDifferencePercent_ * static_cast<float>(total_work) / static_cast<float>(nProcessors)) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - difference.\n";
            }

            // Introducing new superstep
            if (endsuperstep) {
                allReady = ready;
                for (unsigned proc = 0; proc < nProcessors; proc++) {
                    for (const auto &item : procReady[proc]) {
                        procReadyPrior[proc].insert(item);
                        which_proc_ready_prior[item.first] = proc;
                    }
                    procReady[proc].clear();

                    superstepPartitionWork[proc] = 0;
                }
                freeProcessors.clear();

                if constexpr (useMemoryConstraint_) {
                    for (unsigned proc = 0; proc < nProcessors; proc++) {
                        memoryConstraint_.reset(proc);
                    }
                }

                superstep += 1;
                endsuperstep = false;
            }

            bool assignedANode = false;

            // Choosing next processor
            std::vector<unsigned> processorsInOrder = LoadBalancerBase<GraphT, InterpolationT>::computeProcessorPriority(
                superstep_partition_work, total_partition_work, total_work, instance, slack_);
            for (unsigned &proc : processors_in_order) {
                if ((free_processors.find(proc)) != free_processors.cend()) {
                    continue;
                }

                // Check for too many free processors - needed here because free processors may not have been detected
                // yet
                if (num_unable_to_partition_node_loop == 0
                    && static_cast<double>(free_processors.size()) > max_percent_idle_processors * n_processors
                    && ((!increase_parallelism_in_new_superstep) || ready.size() >= n_processors
                        || static_cast<double>(ready.size()) >= 1.2 * (n_processors - static_cast<double>(free_processors.size()))
                        || static_cast<double>(ready.size()) >= n_processors - static_cast<double>(free_processors.size())
                                                                    + (0.5 * static_cast<double>(free_processors.size())))) {
                    endsuperstep = true;
                    // std::cout << "\nCall for new superstep - parallelism.\n";
                    break;
                }

                assigned_a_node = false;

                // Choosing next node
                VertexType next_node;
                for (auto vertex_prior_pair_iter = procReady[proc].begin(); vertex_prior_pair_iter != procReady[proc].cend();
                     vertex_prior_pair_iter++) {
                    if (assigned_a_node) {
                        break;
                    }

                    if constexpr (use_memory_constraint) {
                        if (memory_constraint.can_add(vertex_prior_pair_iter->first, proc)) {
                            next_node = vertex_prior_pair_iter->first;
                            assigned_a_node = true;
                        }
                    } else {
                        next_node = vertex_prior_pair_iter->first;
                        assigned_a_node = true;
                    }
                }

                for (auto vertex_prior_pair_iter = procReadyPrior[proc].begin();
                     vertex_prior_pair_iter != procReadyPrior[proc].cend();
                     vertex_prior_pair_iter++) {
                    if (assigned_a_node) {
                        break;
                    }

                    if constexpr (use_memory_constraint) {
                        if (memory_constraint.can_add(vertex_prior_pair_iter->first, proc)) {
                            next_node = vertex_prior_pair_iter->first;
                            assigned_a_node = true;
                        }
                    } else {
                        next_node = vertex_prior_pair_iter->first;
                        assigned_a_node = true;
                    }
                }

                for (auto vertex_prior_pair_iter = allReady.begin(); vertex_prior_pair_iter != allReady.cend();
                     vertex_prior_pair_iter++) {
                    if (assigned_a_node) {
                        break;
                    }

                    if constexpr (use_memory_constraint) {
                        if (memory_constraint.can_add(vertex_prior_pair_iter->first, proc)) {
                            next_node = vertex_prior_pair_iter->first;
                            assigned_a_node = true;
                        }
                    } else {
                        next_node = vertex_prior_pair_iter->first;
                        assigned_a_node = true;
                    }
                }

                if (!assigned_a_node) {
                    free_processors.insert(proc);
                } else {
                    // Assignments
                    // std::cout << "Allocated node " << next_node << " to processor " << proc << ".\n";
                    schedule.SetAssignedProcessor(next_node, proc);
                    schedule.SetAssignedSuperstep(next_node, superstep);
                    num_unable_to_partition_node_loop = 0;

                    // Updating loads
                    total_partition_work[proc] += graph.VertexWorkWeight(next_node);
                    superstep_partition_work[proc] += graph.VertexWorkWeight(next_node);

                    if constexpr (use_memory_constraint) {
                        memory_constraint.add(next_node, proc);
                    }

                    // Deletion from Queues
                    std::pair<VertexType, double> pair = std::make_pair(next_node, variance_priorities[next_node]);
                    ready.erase(pair);
                    procReady[proc].erase(pair);
                    procReadyPrior[proc].erase(pair);
                    allReady.erase(pair);
                    if (which_proc_ready_prior[next_node] != n_processors) {
                        procReadyPrior[which_proc_ready_prior[next_node]].erase(pair);
                    }

                    // Checking children
                    for (const auto &chld : graph.Children(next_node)) {
                        num_unallocated_parents[chld] -= 1;
                        if (num_unallocated_parents[chld] == 0) {
                            // std::cout << "Inserting child " << chld << " into ready.\n";
                            ready.insert(std::make_pair(chld, variance_priorities[chld]));
                            bool is_proc_ready = true;
                            for (const auto &parent : graph.Parents(chld)) {
                                if ((schedule.AssignedProcessor(parent) != proc)
                                    && (schedule.AssignedSuperstep(parent) == superstep)) {
                                    is_proc_ready = false;
                                    break;
                                }
                            }
                            if (is_proc_ready) {
                                procReady[proc].insert(std::make_pair(chld, variance_priorities[chld]));
                                // std::cout << "Inserting child " << chld << " into procReady for processor " << proc
                                // << ".\n";
                            }
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

    std::string GetScheduleName() const override { return "VariancePartitioner"; };
};

}    // namespace osp
