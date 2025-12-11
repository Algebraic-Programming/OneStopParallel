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

template <typename Graph_t, typename Interpolation_t, typename MemoryConstraint_t = no_memory_constraint>
class VariancePartitioner : public LoadBalancerBase<Graph_t, Interpolation_t> {
    static_assert(is_computational_dag_v<Graph_t>, "VariancePartitioner can only be used with computational DAGs.");

    using VertexType = vertex_idx_t<Graph_t>;

    struct VarianceCompare {
        bool operator()(const std::pair<VertexType, double> &lhs, const std::pair<VertexType, double> &rhs) const {
            return ((lhs.second > rhs.second) || ((lhs.second >= rhs.second) && (lhs.first < rhs.first)));
        }
    };

  protected:
    constexpr static bool use_memory_constraint = is_memory_constraint_v<MemoryConstraint_t>
                                                  or is_memory_constraint_schedule_v<MemoryConstraint_t>;

    static_assert(not use_memory_constraint or std::is_same_v<Graph_t, typename MemoryConstraint_t::Graph_impl_t>,
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

    MemoryConstraint_t memory_constraint;

    /// @brief threshold percentage of idle processors as to when a new superstep should be introduced
    double max_percent_idle_processors;

    /// @brief the power in the power mean average of the variance scheduler
    double variance_power;

    /// @brief whether or not parallelism should be increased in the next superstep
    bool increase_parallelism_in_new_superstep;

    /// @brief percentage of the average workload by which the processor priorities may diverge
    float max_priority_difference_percent;

    /// @brief how much to ignore the global work balance, value between 0 and 1
    float slack;

    /// @brief Computes a power mean average of the bottom node distance
    /// @param graph graph
    /// @param power the power in the power mean average
    /// @return vector of the logarithm of power mean averaged bottom node distance
    std::vector<double> compute_work_variance(const Graph_t &graph, double power = 2) const {
        std::vector<double> work_variance(graph.num_vertices(), 0.0);

        const auto top_order = GetTopOrder(graph);

        for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
            double temp = 0;
            double max_priority = 0;
            for (const auto &child : graph.children(*r_iter)) {
                max_priority = std::max(work_variance[child], max_priority);
            }
            for (const auto &child : graph.children(*r_iter)) {
                temp += std::exp(power * (work_variance[child] - max_priority));
            }
            temp = std::log(temp) / power + max_priority;

            double node_weight = std::log(graph.vertex_work_weight(*r_iter));
            double larger_val = node_weight > temp ? node_weight : temp;

            work_variance[*r_iter] = std::log(std::exp(node_weight - larger_val) + std::exp(temp - larger_val)) + larger_val;
        }

        return work_variance;
    }

  public:
    VariancePartitioner(double max_percent_idle_processors_ = 0.2,
                        double variance_power_ = 2.0,
                        bool increase_parallelism_in_new_superstep_ = true,
                        float max_priority_difference_percent_ = 0.34f,
                        float slack_ = 0.0f)
        : max_percent_idle_processors(max_percent_idle_processors_),
          variance_power(variance_power_),
          increase_parallelism_in_new_superstep(increase_parallelism_in_new_superstep_),
          max_priority_difference_percent(max_priority_difference_percent_),
          slack(slack_) {};

    virtual ~VariancePartitioner() = default;

    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {
        const auto &instance = schedule.getInstance();
        const auto &n_vert = instance.numberOfVertices();
        const unsigned &n_processors = instance.numberOfProcessors();
        const auto &graph = instance.getComputationalDag();

        unsigned superstep = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraint_t>) {
            memory_constraint.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraint_t>) {
            memory_constraint.initialize(schedule, superstep);
        }

        v_workw_t<Graph_t> total_work = 0;

        std::vector<v_workw_t<Graph_t>> total_partition_work(n_processors, 0);
        std::vector<v_workw_t<Graph_t>> superstep_partition_work(n_processors, 0);

        std::vector<double> variance_priorities = compute_work_variance(graph, variance_power);
        std::vector<VertexType> num_unallocated_parents(n_vert, 0);

        std::set<std::pair<VertexType, double>, VarianceCompare> ready;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(n_processors);
        std::set<std::pair<VertexType, double>, VarianceCompare> allReady;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReadyPrior(n_processors);

        std::vector<unsigned> which_proc_ready_prior(n_vert, n_processors);

        for (const auto &v : graph.vertices()) {
            schedule.setAssignedProcessor(v, n_processors);

            total_work += graph.vertex_work_weight(v);

            if (is_source(v, graph)) {
                ready.insert(std::make_pair(v, variance_priorities[v]));
                allReady.insert(std::make_pair(v, variance_priorities[v]));

            } else {
                num_unallocated_parents[v] = graph.in_degree(v);
            }
        }

        std::set<unsigned> free_processors;

        bool endsuperstep = false;
        unsigned num_unable_to_partition_node_loop = 0;
        // RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

        while (!ready.empty()) {
            // Increase memory capacity if needed
            if (num_unable_to_partition_node_loop == 1) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - unable to schedule.\n";
            } else {
                if constexpr (use_memory_constraint) {
                    if (num_unable_to_partition_node_loop >= 2) {
                        return RETURN_STATUS::ERROR;
                    }
                }
            }

            // Checking if new superstep is needed
            // std::cout << "freeprocessor " << free_processors.size() << " idle thresh " << max_percent_idle_processors
            // * n_processors << " ready size " << ready.size() << " small increase " << 1.2 * (n_processors -
            // free_processors.size()) << " large increase " << n_processors - free_processors.size() +  (0.5 *
            // free_processors.size()) << "\n";
            if (num_unable_to_partition_node_loop == 0
                && static_cast<double>(free_processors.size()) > max_percent_idle_processors * n_processors
                && ((!increase_parallelism_in_new_superstep) || ready.size() >= n_processors
                    || static_cast<double>(ready.size()) >= 1.2 * (n_processors - static_cast<double>(free_processors.size()))
                    || static_cast<double>(ready.size()) >= n_processors - static_cast<double>(free_processors.size())
                                                                + (0.5 * static_cast<double>(free_processors.size())))) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - parallelism.\n";
            }
            std::vector<float> processor_priorities
                = LoadBalancerBase<Graph_t, Interpolation_t>::computeProcessorPrioritiesInterpolation(
                    superstep_partition_work, total_partition_work, total_work, instance);
            float min_priority = processor_priorities[0];
            float max_priority = processor_priorities[0];
            for (const auto &prio : processor_priorities) {
                min_priority = std::min(min_priority, prio);
                max_priority = std::max(max_priority, prio);
            }
            if (num_unable_to_partition_node_loop == 0
                && (max_priority - min_priority)
                       > max_priority_difference_percent * static_cast<float>(total_work) / static_cast<float>(n_processors)) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - difference.\n";
            }

            // Introducing new superstep
            if (endsuperstep) {
                allReady = ready;
                for (unsigned proc = 0; proc < n_processors; proc++) {
                    for (const auto &item : procReady[proc]) {
                        procReadyPrior[proc].insert(item);
                        which_proc_ready_prior[item.first] = proc;
                    }
                    procReady[proc].clear();

                    superstep_partition_work[proc] = 0;
                }
                free_processors.clear();

                if constexpr (use_memory_constraint) {
                    for (unsigned proc = 0; proc < n_processors; proc++) {
                        memory_constraint.reset(proc);
                    }
                }

                superstep += 1;
                endsuperstep = false;
            }

            bool assigned_a_node = false;

            // Choosing next processor
            std::vector<unsigned> processors_in_order = LoadBalancerBase<Graph_t, Interpolation_t>::computeProcessorPriority(
                superstep_partition_work, total_partition_work, total_work, instance, slack);
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
                    schedule.setAssignedProcessor(next_node, proc);
                    schedule.setAssignedSuperstep(next_node, superstep);
                    num_unable_to_partition_node_loop = 0;

                    // Updating loads
                    total_partition_work[proc] += graph.vertex_work_weight(next_node);
                    superstep_partition_work[proc] += graph.vertex_work_weight(next_node);

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
                    for (const auto &chld : graph.children(next_node)) {
                        num_unallocated_parents[chld] -= 1;
                        if (num_unallocated_parents[chld] == 0) {
                            // std::cout << "Inserting child " << chld << " into ready.\n";
                            ready.insert(std::make_pair(chld, variance_priorities[chld]));
                            bool is_proc_ready = true;
                            for (const auto &parent : graph.parents(chld)) {
                                if ((schedule.assignedProcessor(parent) != proc)
                                    && (schedule.assignedSuperstep(parent) == superstep)) {
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
            if (!assigned_a_node) {
                num_unable_to_partition_node_loop += 1;
            }
        }

        return RETURN_STATUS::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return "VariancePartitioner"; };
};

}    // namespace osp
