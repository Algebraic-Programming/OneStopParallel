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

#include "dag_partitioners/LightEdgeVariancePartitioner.hpp"

std::vector<double> LightEdgeVariancePartitioner::compute_work_variance(const ComputationalDag &graph, double power) const {
    std::vector<double> work_variance(graph.numberOfVertices(), 0.0);

    const std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        double temp = 0;
        double max_priority = 0;
        for (const auto &child : graph.children(*r_iter)) {
            max_priority = std::max(work_variance[child], max_priority);
        }
        for (const auto &child : graph.children(*r_iter)) {
            temp += std::exp(power*(work_variance[child]-max_priority));
        }
        temp = std::log(temp) / power + max_priority;

        double node_weight = std::log((double)graph.nodeWorkWeight(*r_iter));
        double larger_val = node_weight > temp ? node_weight : temp;

        work_variance[*r_iter] = std::log( std::exp(node_weight-larger_val) + std::exp( temp - larger_val ) ) + larger_val;
    }

    return work_variance;
}

std::pair<RETURN_STATUS, DAGPartition> LightEdgeVariancePartitioner::computePartition(const BspInstance &instance) {
    DAGPartition output_partition(instance);

    const unsigned &n_vert = instance.numberOfVertices();
    const unsigned &n_processors = instance.numberOfProcessors();
    const auto &graph = instance.getComputationalDag();

    std::vector<bool> has_vertex_been_assigned(n_vert, false);

    long unsigned total_work = 0;
    for (unsigned v = 0; v < n_vert; v++) {
        total_work += graph.nodeWorkWeight(v);
    }
    std::vector<long unsigned> total_partition_work(n_processors, 0);
    std::vector<long unsigned> superstep_partition_work(n_processors, 0);

    std::vector<long unsigned> total_partition_memory(n_processors, 0);
    std::vector<long unsigned> transient_partition_memory(n_processors, 0);
    std::vector<double> memory_capacity(n_processors);
    for (unsigned proc = 0; proc < n_processors; proc++) {
        memory_capacity[proc] = instance.memoryBound(proc);
    }

    std::vector<double> variance_priorities = compute_work_variance( graph, variance_power );

    std::vector<unsigned> num_unallocated_parents(n_vert,0);
    for (const VertexType& v : graph.vertices()) {
        for (const VertexType chld : graph.children(v)) {
            num_unallocated_parents[chld] += 1;
        }
    }

    std::vector<std::vector<VertexType>> preprocessed_partition = heavy_edge_preprocess(graph, heavy_is_x_times_median, min_percent_components_retained, bound_component_weight_percent / n_processors);
    std::vector<size_t> which_preprocess_partition(graph.numberOfVertices());
    for (size_t i = 0; i < preprocessed_partition.size(); i++) {
        for (const VertexType &vert : preprocessed_partition[i]) {
            which_preprocess_partition[vert] = i;
        }
    }

    std::vector<int> memory_cost_of_preprocessed_partition(preprocessed_partition.size(), 0);
    for (size_t i = 0; i < preprocessed_partition.size(); i++) {
        for (const auto &vert : preprocessed_partition[i]) {
            memory_cost_of_preprocessed_partition[i] += graph.nodeMemoryWeight(vert);
        }
    }

    std::vector<int> transient_cost_of_preprocessed_partition(preprocessed_partition.size(), 0);
    for (size_t i = 0; i < preprocessed_partition.size(); i++) {
        for (const auto &vert : preprocessed_partition[i]) {
            transient_cost_of_preprocessed_partition[i] = std::max(transient_cost_of_preprocessed_partition[i], graph.nodeCommunicationWeight(vert));
        }
    }

    std::set<std::pair<VertexType, double>, VarianceCompare> ready;
    std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(n_processors);
    std::set<std::pair<VertexType, double>, VarianceCompare> allReady;
    std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReadyPrior(n_processors);
    std::vector<int> which_proc_ready_prior(n_vert, -1);

    for (const VertexType& v : graph.vertices()) {
        if (num_unallocated_parents[v] == 0) {
            ready.insert(std::make_pair(v, variance_priorities[v]));
            allReady.insert(std::make_pair(v, variance_priorities[v]));
        }
    }

    std::vector<unsigned> superstep_assignment(n_vert, 0);
    std::set<unsigned> free_processors;

    unsigned superstep = 0;
    bool endsuperstep = false;
    unsigned num_unable_to_partition_node_loop = 0;
    RETURN_STATUS status = SUCCESS;
    while (!ready.empty()) {
        // Increase memory capacity if needed
        if (num_unable_to_partition_node_loop == 1) {
            endsuperstep = true;
            // std::cout << "\nCall for new superstep - unable to schedule.\n";
        } else if (use_memory_constraint && num_unable_to_partition_node_loop >= 2) {
            for (unsigned proc = 0; proc < n_processors; proc++) {
                memory_capacity[proc] *= memory_capacity_increase;
            }
            std::cerr << "Memory capacity has been increased!" << std::endl;
            status = BEST_FOUND;

            // Memory increase can make processors viable again
            free_processors.clear();
        }

        // Checking if new superstep is needed
        // std::cout << "freeprocessor " << free_processors.size() << " idle thresh " << max_percent_idle_processors * n_processors << " ready size " << ready.size() << " small increase " << 1.2 * (n_processors - free_processors.size()) << " large increase " << n_processors - free_processors.size() +  (0.5 * free_processors.size()) << "\n";
        if (num_unable_to_partition_node_loop == 0 && free_processors.size() > max_percent_idle_processors * n_processors && ((!increase_parallelism_in_new_superstep) || ready.size() >= n_processors || ready.size() >= 1.2 * (n_processors - free_processors.size()) || ready.size() >=  n_processors - free_processors.size() +  (0.5 * free_processors.size()) )) {
            endsuperstep = true;
            // std::cout << "\nCall for new superstep - parallelism.\n";
        }
        std::vector<float> processor_priorities = computeProcessorPriorities(superstep_partition_work, total_partition_work, total_work, instance, slack);
        float min_priority = processor_priorities[0];
        float max_priority = processor_priorities[0];
        for (const auto& prio : processor_priorities) {
            min_priority = std::min(min_priority, prio);
            max_priority = std::max(max_priority, prio);
        }
        if ( num_unable_to_partition_node_loop == 0 && (max_priority - min_priority) >  max_priority_difference_percent * (float) total_work / (float) n_processors ) {
            endsuperstep = true;
            // std::cout << "\nCall for new superstep - difference.\n";
        }

        // Introducing new superstep
        if (endsuperstep) {
            allReady = ready;
            for (unsigned proc = 0; proc < n_processors; proc++) {
                for (const auto& item : procReady[proc]) {
                    procReadyPrior[proc].insert(item);
                    which_proc_ready_prior[item.first] = proc;
                }
                procReady[proc].clear();
                
                superstep_partition_work[proc] = 0;
            }
            free_processors.clear();

            if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {
                for (size_t proc = 0; proc < total_partition_memory.size(); proc++) {
                    total_partition_memory[proc] = 0;
                }
            }

            superstep += 1;
            endsuperstep = false;

        }

        bool assigned_a_node = false;

        // Choosing next processor
        std::vector<unsigned> processors_in_order = computeProcessorPriority(superstep_partition_work, total_partition_work, total_work, instance);
        for (unsigned &proc : processors_in_order) {
            if ((free_processors.find(proc)) != free_processors.cend()) continue;

            // Check for too many free processors - needed here because free processors may not have been detected yet
            if (num_unable_to_partition_node_loop == 0 && free_processors.size() > max_percent_idle_processors * n_processors && ((!increase_parallelism_in_new_superstep) || ready.size() >= n_processors || ready.size() >= 1.2 * (n_processors - free_processors.size()) || ready.size() >=  n_processors - free_processors.size() +  (0.5 * free_processors.size()) )) {
                endsuperstep = true;
                // std::cout << "\nCall for new superstep - parallelism.\n";
                break;
            }

            assigned_a_node = false;

            // Choosing next node
            VertexType next_node;
            for (auto vertex_prior_pair_iter = procReady[proc].begin(); vertex_prior_pair_iter != procReady[proc].cend(); vertex_prior_pair_iter++) {
                if (assigned_a_node) break;
                const VertexType& vert = vertex_prior_pair_iter->first;
                if (use_memory_constraint && instance.getArchitecture().getMemoryConstraintType() != NONE) {
                    if (instance.getArchitecture().getMemoryConstraintType() == LOCAL || instance.getArchitecture().getMemoryConstraintType() == GLOBAL) {
                        if (has_vertex_been_assigned[vert] || (total_partition_memory[proc] + memory_cost_of_preprocessed_partition[which_preprocess_partition[vert]] < memory_capacity[proc])) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
                    }
                    if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                        if (has_vertex_been_assigned[vert] || total_partition_memory[proc] + graph.nodeMemoryWeight(vert) + std::max(transient_partition_memory[proc], (long unsigned) transient_cost_of_preprocessed_partition[which_preprocess_partition[vert]]) < memory_capacity[proc]) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
                    }
                } else {
                    next_node = vert;
                    assigned_a_node = true;
                }
            }
            for (auto vertex_prior_pair_iter = procReadyPrior[proc].begin(); vertex_prior_pair_iter != procReadyPrior[proc].cend(); vertex_prior_pair_iter++) {
                if (assigned_a_node) break;
                const VertexType& vert = vertex_prior_pair_iter->first;
                if (use_memory_constraint && instance.getArchitecture().getMemoryConstraintType() != NONE) {
                    if (instance.getArchitecture().getMemoryConstraintType() == LOCAL || instance.getArchitecture().getMemoryConstraintType() == GLOBAL) {
                        if (has_vertex_been_assigned[vert] || (total_partition_memory[proc] + memory_cost_of_preprocessed_partition[which_preprocess_partition[vert]] < memory_capacity[proc])) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
                    }
                    if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                        if (has_vertex_been_assigned[vert] || total_partition_memory[proc] + graph.nodeMemoryWeight(vert) + std::max(transient_partition_memory[proc], (long unsigned) transient_cost_of_preprocessed_partition[which_preprocess_partition[vert]]) < memory_capacity[proc]) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
                    }
                } else {
                    next_node = vert;
                    assigned_a_node = true;
                }
            }
            for (auto vertex_prior_pair_iter = allReady.begin(); vertex_prior_pair_iter != allReady.cend(); vertex_prior_pair_iter++) {
                if (assigned_a_node) break;
                const VertexType& vert = vertex_prior_pair_iter->first;
                if (use_memory_constraint && instance.getArchitecture().getMemoryConstraintType() != NONE) {
                    if (instance.getArchitecture().getMemoryConstraintType() == LOCAL || instance.getArchitecture().getMemoryConstraintType() == GLOBAL) {
                        if (has_vertex_been_assigned[vert] || (total_partition_memory[proc] + memory_cost_of_preprocessed_partition[which_preprocess_partition[vert]] < memory_capacity[proc])) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
                    }
                    if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                        if (has_vertex_been_assigned[vert] || total_partition_memory[proc] + graph.nodeMemoryWeight(vert) + std::max(transient_partition_memory[proc], (long unsigned) transient_cost_of_preprocessed_partition[which_preprocess_partition[vert]]) < memory_capacity[proc]) {
                            next_node = vert;
                            assigned_a_node = true;
                        }
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
                    unsigned proc_alloc_prior = output_partition.assignedProcessor(next_node);
                    // std::cout << "Allocated node " << next_node << " to processor " << proc_alloc_prior << " previously.\n";

                    superstep_assignment[next_node] = superstep;
                    num_unable_to_partition_node_loop = 0;

                    // Updating loads
                    superstep_partition_work[proc_alloc_prior] += graph.nodeWorkWeight(next_node);

                    // Deletion from Queues
                    std::pair<VertexType, double> pair = std::make_pair(next_node, variance_priorities[next_node]);
                    ready.erase(pair);
                    procReady[proc].erase(pair);
                    procReadyPrior[proc].erase(pair);
                    allReady.erase(pair);
                    if (which_proc_ready_prior[next_node] != -1) {
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
                                if ((output_partition.assignedProcessor(parent) != proc_alloc_prior) && (superstep_assignment[parent] == superstep) ) {
                                    is_proc_ready = false;
                                    break;
                                }
                            }
                            if (is_proc_ready) {
                                procReady[proc_alloc_prior].insert(std::make_pair(chld, variance_priorities[chld]));
                                // std::cout << "Inserting child " << chld << " into procReady for processor " << proc_alloc_prior << ".\n";
                            }
                        }
                    }
                } else {
                    output_partition.setAssignedProcessor(next_node, proc);
                    has_vertex_been_assigned[next_node] = true;
                    // std::cout << "Allocated node " << next_node << " to processor " << proc << ".\n";

                    superstep_assignment[next_node] = superstep;
                    num_unable_to_partition_node_loop = 0;

                    // Updating loads
                    total_partition_work[proc] += graph.nodeWorkWeight(next_node);
                    superstep_partition_work[proc] += graph.nodeWorkWeight(next_node);
                    total_partition_memory[proc] += graph.nodeMemoryWeight(next_node);
                    transient_partition_memory[proc] = std::max(transient_partition_memory[proc], (long unsigned) graph.nodeCommunicationWeight(next_node));

                    // Deletion from Queues
                    std::pair<VertexType, double> pair = std::make_pair(next_node, variance_priorities[next_node]);
                    ready.erase(pair);
                    procReady[proc].erase(pair);
                    procReadyPrior[proc].erase(pair);
                    allReady.erase(pair);
                    if (which_proc_ready_prior[next_node] != -1) {
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
                                if ((output_partition.assignedProcessor(parent) != proc) && (superstep_assignment[parent] == superstep) ) {
                                    is_proc_ready = false;
                                    break;
                                }
                            }
                            if (is_proc_ready) {
                                procReady[proc].insert(std::make_pair(chld, variance_priorities[chld]));
                                // std::cout << "Inserting child " << chld << " into procReady for processor " << proc << ".\n";
                            }
                        }
                    }

                    // Allocating all nodes in the same partition
                    for (VertexType node_in_same_partition : preprocessed_partition[which_preprocess_partition[next_node]]) {
                        if (node_in_same_partition == next_node) continue;

                        // Allocation
                        output_partition.setAssignedProcessor(node_in_same_partition, proc);
                        has_vertex_been_assigned[node_in_same_partition] = true;
                        // std::cout << "Allocated node " << next_node << " to processor " << proc << ".\n";

                        // Update loads
                        total_partition_work[proc] += graph.nodeWorkWeight(node_in_same_partition);
                        total_partition_memory[proc] += graph.nodeMemoryWeight(node_in_same_partition);
                        transient_partition_memory[proc] = std::max(transient_partition_memory[proc], (long unsigned) graph.nodeCommunicationWeight(node_in_same_partition));
                    }
                }



                break;
            }
        }
        if (!assigned_a_node) {
            num_unable_to_partition_node_loop += 1;
        }
    }
    


    return {status, output_partition};
}