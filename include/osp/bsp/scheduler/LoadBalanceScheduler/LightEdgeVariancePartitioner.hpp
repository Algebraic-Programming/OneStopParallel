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

template <typename GraphT, typename InterpolationT, typename MemoryConstraintT = no_memory_constraint>
class LightEdgeVariancePartitioner : public VariancePartitioner<GraphT, InterpolationT, MemoryConstraintT> {
  private:
    using VertexType = vertex_idx_t<Graph_t>;

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

    virtual RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override {
        // DAGPartition output_partition(instance);

        using Base = VariancePartitioner<GraphT, InterpolationT, MemoryConstraintT>;

        const auto &instance = schedule.GetInstance();
        const auto &nVert = instance.NumberOfVertices();
        const unsigned &nProcessors = instance.NumberOfProcessors();
        const auto &graph = instance.GetComputationalDag();

        unsigned superstep = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraint_t>) {
            Base::memory_constraint.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraint_t>) {
            Base::memory_constraint.initialize(schedule, superstep);
        }

        std::vector<bool> hasVertexBeenAssigned(nVert, false);

        std::set<std::pair<VertexType, double>, VarianceCompare> ready;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(nProcessors);
        std::set<std::pair<VertexType, double>, VarianceCompare> allReady;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReadyPrior(nProcessors);

        std::vector<unsigned> whichProcReadyPrior(nVert, nProcessors);

        std::vector<double> variancePriorities = Base::compute_work_variance(graph, Base::variance_power);
        std::vector<VertexType> numUnallocatedParents(nVert, 0);

        v_workw_t<Graph_t> totalWork = 0;
        for (const auto &v : graph.vertices()) {
            schedule.setAssignedProcessor(v, nProcessors);

            totalWork += graph.VertexWorkWeight(v);

            if (IsSource(v, graph)) {
                ready.insert(std::make_pair(v, variancePriorities[v]));
                allReady.insert(std::make_pair(v, variancePriorities[v]));

            } else {
                numUnallocatedParents[v] = graph.in_degree(v);
            }
        }

        std::vector<v_workw_t<Graph_t>> totalPartitionWork(nProcessors, 0);
        std::vector<v_workw_t<Graph_t>> superstepPartitionWork(nProcessors, 0);

        std::vector<std::vector<VertexType>> preprocessedPartition = heavy_edge_preprocess(
            graph, heavyIsXTimesMedian_, minPercentComponentsRetained_, boundComponentWeightPercent_ / nProcessors);

        std::vector<size_t> whichPreprocessPartition(graph.NumVertices());
        for (size_t i = 0; i < preprocessedPartition.size(); i++) {
            for (const VertexType &vert : preprocessed_partition[i]) {
                which_preprocess_partition[vert] = i;
            }
        }

        std::vector<v_memw_t<Graph_t>> memoryCostOfPreprocessedPartition(preprocessedPartition.size(), 0);
        for (size_t i = 0; i < preprocessedPartition.size(); i++) {
            for (const auto &vert : preprocessed_partition[i]) {
                memory_cost_of_preprocessed_partition[i] += graph.VertexMemWeight(vert);
            }
        }

        std::vector<v_commw_t<Graph_t>> transientCostOfPreprocessedPartition(preprocessedPartition.size(), 0);
        for (size_t i = 0; i < preprocessedPartition.size(); i++) {
            for (const auto &vert : preprocessed_partition[i]) {
                transient_cost_of_preprocessed_partition[i]
                    = std::max(transient_cost_of_preprocessed_partition[i], graph.VertexCommWeight(vert));
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
                if constexpr (Base::use_memory_constraint) {
                    if (numUnableToPartitionNodeLoop >= 2) {
                        return RETURN_STATUS::ERROR;
                    }
                }
            }

            // Checking if new superstep is needed
            // std::cout << "freeprocessor " << free_processors.size() << " idle thresh " << max_percent_idle_processors
            // * n_processors << " ready size " << ready.size() << " small increase " << 1.2 * (n_processors -
            // free_processors.size()) << " large increase " << n_processors - free_processors.size() +  (0.5 *
            // free_processors.size()) << "\n";
            if (numUnableToPartitionNodeLoop == 0
                && static_cast<double>(freeProcessors.size()) > Base::max_percent_idle_processors * nProcessors
                && ((!Base::increase_parallelism_in_new_superstep) || ready.size() >= nProcessors
                    || static_cast<double>(ready.size()) >= 1.2 * (nProcessors - static_cast<double>(freeProcessors.size()))
                    || static_cast<double>(ready.size()) >= nProcessors - static_cast<double>(freeProcessors.size())
                                                                + (0.5 * static_cast<double>(freeProcessors.size())))) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - parallelism.\n";
            }

            std::vector<float> processorPriorities = Base::computeProcessorPrioritiesInterpolation(
                superstep_partition_work, total_partition_work, total_work, instance);

            float minPriority = processorPriorities[0];
            float maxPriority = processorPriorities[0];
            for (const auto &prio : processor_priorities) {
                min_priority = std::min(min_priority, prio);
                max_priority = std::max(max_priority, prio);
            }
            if (numUnableToPartitionNodeLoop == 0
                && (maxPriority - minPriority) > Base::max_priority_difference_percent * static_cast<float>(total_work)
                                                     / static_cast<float>(nProcessors)) {
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

                if constexpr (Base::use_memory_constraint) {
                    for (unsigned proc = 0; proc < nProcessors; proc++) {
                        Base::memory_constraint.reset(proc);
                    }
                }

                superstep += 1;
                endsuperstep = false;
            }

            bool assignedANode = false;

            // Choosing next processor
            std::vector<unsigned> processorsInOrder = Base::computeProcessorPriority(
                superstep_partition_work, total_partition_work, total_work, instance, Base::slack);

            for (unsigned &proc : processors_in_order) {
                if ((free_processors.find(proc)) != free_processors.cend()) {
                    continue;
                }

                // Check for too many free processors - needed here because free processors may not have been detected
                // yet
                if (num_unable_to_partition_node_loop == 0
                    && static_cast<double>(free_processors.size()) > base::max_percent_idle_processors * n_processors
                    && ((!base::increase_parallelism_in_new_superstep) || ready.size() >= n_processors
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
                for (auto vertex_prior_pair_iter = procReady[proc].begin(); vertex_prior_pair_iter != procReady[proc].end();
                     vertex_prior_pair_iter++) {
                    if (assigned_a_node) {
                        break;
                    }

                    const VertexType &vert = vertex_prior_pair_iter->first;
                    if constexpr (base::use_memory_constraint) {
                        if (has_vertex_been_assigned[vert]
                            || base::memory_constraint.can_add(
                                proc,
                                memory_cost_of_preprocessed_partition[which_preprocess_partition[vert]],
                                transient_cost_of_preprocessed_partition[which_preprocess_partition[vert]])) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
                    } else {
                        next_node = vert;
                        assigned_a_node = true;
                    }
                }

                for (auto vertex_prior_pair_iter = procReadyPrior[proc].begin();
                     vertex_prior_pair_iter != procReadyPrior[proc].end();
                     vertex_prior_pair_iter++) {
                    if (assigned_a_node) {
                        break;
                    }

                    const VertexType &vert = vertex_prior_pair_iter->first;
                    if constexpr (base::use_memory_constraint) {
                        if (has_vertex_been_assigned[vert]
                            || base::memory_constraint.can_add(
                                proc,
                                memory_cost_of_preprocessed_partition[which_preprocess_partition[vert]],
                                transient_cost_of_preprocessed_partition[which_preprocess_partition[vert]])) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
                    } else {
                        next_node = vert;
                        assigned_a_node = true;
                    }
                }
                for (auto vertex_prior_pair_iter = allReady.begin(); vertex_prior_pair_iter != allReady.cend();
                     vertex_prior_pair_iter++) {
                    if (assigned_a_node) {
                        break;
                    }

                    const VertexType &vert = vertex_prior_pair_iter->first;
                    if constexpr (base::use_memory_constraint) {
                        if (has_vertex_been_assigned[vert]
                            || base::memory_constraint.can_add(
                                proc,
                                memory_cost_of_preprocessed_partition[which_preprocess_partition[vert]],
                                transient_cost_of_preprocessed_partition[which_preprocess_partition[vert]])) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
                    } else {
                        next_node = vert;
                        assigned_a_node = true;
                    }
                }

                if (!assigned_a_node) {
                    free_processors.insert(proc);
                } else {
                    // Assignments
                    if (has_vertex_been_assigned[next_node]) {
                        unsigned proc_alloc_prior = schedule.assignedProcessor(next_node);

                        // std::cout << "Allocated node " << next_node << " to processor " << proc_alloc_prior << "
                        // previously.\n";

                        schedule.setAssignedSuperstep(next_node, superstep);

                        num_unable_to_partition_node_loop = 0;

                        // Updating loads
                        superstep_partition_work[proc_alloc_prior] += graph.VertexWorkWeight(next_node);

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
                                    if ((schedule.assignedProcessor(parent) != proc_alloc_prior)
                                        && (schedule.assignedSuperstep(parent) == superstep)) {
                                        is_proc_ready = false;
                                        break;
                                    }
                                }
                                if (is_proc_ready) {
                                    procReady[proc_alloc_prior].insert(std::make_pair(chld, variance_priorities[chld]));
                                    // std::cout << "Inserting child " << chld << " into procReady for processor " <<
                                    // proc_alloc_prior << ".\n";
                                }
                            }
                        }
                    } else {
                        schedule.setAssignedProcessor(next_node, proc);
                        has_vertex_been_assigned[next_node] = true;
                        // std::cout << "Allocated node " << next_node << " to processor " << proc << ".\n";

                        schedule.setAssignedSuperstep(next_node, superstep);
                        num_unable_to_partition_node_loop = 0;

                        // Updating loads
                        total_partition_work[proc] += graph.VertexWorkWeight(next_node);
                        superstep_partition_work[proc] += graph.VertexWorkWeight(next_node);

                        if constexpr (base::use_memory_constraint) {
                            base::memory_constraint.add(next_node, proc);
                        }
                        // total_partition_memory[proc] += graph.VertexMemWeight(next_node);
                        // transient_partition_memory[proc] =
                        //     std::max(transient_partition_memory[proc], graph.VertexCommWeight(next_node));

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
                                    if ((schedule.assignedProcessor(parent) != proc)
                                        && (schedule.assignedSuperstep(parent) == superstep)) {
                                        is_proc_ready = false;
                                        break;
                                    }
                                }
                                if (is_proc_ready) {
                                    procReady[proc].insert(std::make_pair(chld, variance_priorities[chld]));
                                    // std::cout << "Inserting child " << chld << " into procReady for processor " <<
                                    // proc << ".\n";
                                }
                            }
                        }

                        // Allocating all nodes in the same partition
                        for (VertexType node_in_same_partition : preprocessed_partition[which_preprocess_partition[next_node]]) {
                            if (node_in_same_partition == next_node) {
                                continue;
                            }

                            // Allocation
                            schedule.setAssignedProcessor(node_in_same_partition, proc);
                            has_vertex_been_assigned[node_in_same_partition] = true;
                            // std::cout << "Allocated node " << next_node << " to processor " << proc << ".\n";

                            // Update loads
                            total_partition_work[proc] += graph.VertexWorkWeight(node_in_same_partition);

                            if constexpr (base::use_memory_constraint) {
                                base::memory_constraint.add(node_in_same_partition, proc);
                            }

                            // total_partition_memory[proc] += graph.VertexMemWeight(node_in_same_partition);
                            // transient_partition_memory[proc] = std::max(
                            //     transient_partition_memory[proc], graph.VertexCommWeight(node_in_same_partition));
                        }
                    }

                    break;
                }
            }
            if (!assignedANode) {
                numUnableToPartitionNodeLoop += 1;
            }
        }

        return RETURN_STATUS::OSP_SUCCESS;
    }
};

}    // namespace osp
