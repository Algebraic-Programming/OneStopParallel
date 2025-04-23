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

#include "scheduler/GreedySchedulers/GreedyVarianceScheduler.hpp"

std::vector<double> GreedyVarianceScheduler::compute_work_variance(const ComputationalDag &graph) const {
    std::vector<double> work_variance(graph.numberOfVertices(), 0.0);

    const std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        double temp = 0;
        double max_priority = 0;
        for (const auto &child : graph.children(*r_iter)) {
            max_priority = std::max(work_variance[child], max_priority);
        }
        for (const auto &child : graph.children(*r_iter)) {
            temp += std::exp(2 * (work_variance[child] - max_priority));
        }
        temp = std::log(temp) / 2 + max_priority;

        double node_weight = std::log((double)std::max(graph.nodeWorkWeight(*r_iter), 1));
        double larger_val = node_weight > temp ? node_weight : temp;

        work_variance[*r_iter] =
            std::log(std::exp(node_weight - larger_val) + std::exp(temp - larger_val)) + larger_val;
    }

    return work_variance;
}

std::pair<RETURN_STATUS, BspSchedule> GreedyVarianceScheduler::computeSchedule(const BspInstance &instance) {

    if (use_memory_constraint) {

        switch (instance.getArchitecture().getMemoryConstraintType()) {

        case LOCAL:
            current_proc_persistent_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            break;

        case PERSISTENT_AND_TRANSIENT:
            current_proc_persistent_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            current_proc_transient_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            break;

        case GLOBAL:
            throw std::invalid_argument("Global memory constraint not supported");

        case NONE:
            use_memory_constraint = false;
            std::cerr << "Warning: Memory constraint type set to NONE, ignoring memory constraint" << std::endl;
            break;

        default:
            break;
        }
    }

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    const std::vector<double> work_variances = compute_work_variance(G);

    // std::cout << "Max priority is: " << *std::max_element(work_variances.begin(), work_variances.end()) << std::endl;

    std::set<std::pair<VertexType, double>, VarianceCompare> ready;

    // auto variance_compare = [work_variances] (const VertexType& l, const VertexType& r) { return ((work_variances[l]
    // < work_variances[r]) || ((work_variances[l] == work_variances[r]) && (l > r))); };
    std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(params_p);
    std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> allReady(instance.getArchitecture().getNumberOfProcessorTypes());

    std::vector<std::vector<unsigned>> procTypesCompatibleWithNodeType = instance.getProcTypesCompatibleWithNodeType();

    std::vector<unsigned> nrPredecDone(N, 0);
    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::vector<unsigned> nr_ready_nodes_per_type(G.getNumberOfNodeTypes(), 0);
    std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
    for (unsigned proc = 0; proc < params_p; ++proc)
        ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];

    std::set<std::pair<size_t, VertexType>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    for (const auto &v : G.sourceVertices()) {
        ready.insert(std::make_pair(v, work_variances[v]));
        ++nr_ready_nodes_per_type[G.nodeType(v)];
        for(unsigned procType : procTypesCompatibleWithNodeType[G.nodeType(v)])
            allReady[procType].insert(std::make_pair(v, work_variances[v]));
    }

    unsigned supstepIdx = 0;
    bool endSupStep = false;
    while (!ready.empty() || !finishTimes.empty()) {
        if (finishTimes.empty() && endSupStep) {
            for (unsigned i = 0; i < params_p; ++i)
                procReady[i].clear();

            for(int procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes(); ++procType)
                allReady[procType].clear();

            for(const auto &nodeAndValuePair : ready)
            {
                const unsigned node = nodeAndValuePair.first;
                for(unsigned procType : procTypesCompatibleWithNodeType[G.nodeType(node)])
                    allReady[procType].insert(allReady[procType].end(), nodeAndValuePair);
            }

            ++supstepIdx;

            if (use_memory_constraint && instance.getArchitecture().getMemoryConstraintType() == LOCAL) {
                for (unsigned proc = 0; proc < params_p; proc++) {
                    current_proc_persistent_memory[proc] = 0;
                }
            }

            endSupStep = false;

            finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
        }

        const size_t time = finishTimes.begin()->first;

        // Find new ready jobs
        while (!finishTimes.empty() && finishTimes.begin()->first == time) {
            const VertexType node = finishTimes.begin()->second;
            finishTimes.erase(finishTimes.begin());
            if (node != std::numeric_limits<VertexType>::max()) {
                for (const auto &succ : G.children(node)) {
                    ++nrPredecDone[succ];
                    if (nrPredecDone[succ] == G.numberOfParents(succ)) {
                        ready.insert(std::make_pair(succ, work_variances[succ]));
                        ++nr_ready_nodes_per_type[G.nodeType(succ)];

                        bool canAdd = true;
                        for (const auto &pred : G.parents(succ)) {
                            if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                schedule.assignedSuperstep(pred) == supstepIdx)
                                canAdd = false;
                        }

                        if (use_memory_constraint && canAdd) {

                            if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                                if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) >
                                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                                    canAdd = false;
                                }

                            } else if (instance.getArchitecture().getMemoryConstraintType() ==
                                       PERSISTENT_AND_TRANSIENT) {

                                if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) +
                                        std::max(current_proc_transient_memory[schedule.assignedProcessor(node)],
                                                 instance.getComputationalDag().nodeCommunicationWeight(succ)) >
                                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                                    canAdd = false;
                                }
                            }
                        }

                        if (!instance.isCompatible(succ, schedule.assignedProcessor(node)))
                            canAdd = false;

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].insert(
                                std::make_pair(succ, work_variances[succ]));
                        }
                    }
                }
                procFree[schedule.assignedProcessor(node)] = true;
                ++free;
            }
        }

        if (endSupStep)
            continue;

        // Assign new jobs to processors
        if (!CanChooseNode(instance, allReady, procReady, procFree)) {
            endSupStep = true;
        }
        while (CanChooseNode(instance, allReady, procReady, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = params_p;
            Choose(instance, work_variances, allReady, procReady, procFree, nextNode, nextProc);

            if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == params_p) {
                endSupStep = true;
                break;
            }

            if (procReady[nextProc].find(std::make_pair(nextNode, work_variances[nextNode])) !=
                procReady[nextProc].end()) {
                procReady[nextProc].erase(std::make_pair(nextNode, work_variances[nextNode]));
            } else {
                for(unsigned procType : procTypesCompatibleWithNodeType[G.nodeType(nextNode)])
                    allReady[procType].erase(std::make_pair(nextNode, work_variances[nextNode]));
            }

            ready.erase(std::make_pair(nextNode, work_variances[nextNode]));
            --nr_ready_nodes_per_type[G.nodeType(nextNode)];
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            if (use_memory_constraint) {

                std::vector<std::pair<VertexType, double>> toErase;
                if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                    current_proc_persistent_memory[nextProc] +=
                        instance.getComputationalDag().nodeMemoryWeight(nextNode);

                    for (const auto &node_pair : procReady[nextProc]) {
                        if (current_proc_persistent_memory[nextProc] +
                                instance.getComputationalDag().nodeMemoryWeight(node_pair.first) >
                            instance.getArchitecture().memoryBound(nextProc)) {
                            toErase.push_back(node_pair);
                        }
                    }

                } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                    current_proc_persistent_memory[nextProc] +=
                        instance.getComputationalDag().nodeMemoryWeight(nextNode);
                    current_proc_transient_memory[nextProc] =
                        std::max(current_proc_transient_memory[nextProc],
                                 instance.getComputationalDag().nodeCommunicationWeight(nextNode));

                    for (const auto &node_pair : procReady[nextProc]) {
                        if (current_proc_persistent_memory[nextProc] +
                                instance.getComputationalDag().nodeMemoryWeight(node_pair.first) +
                                std::max(current_proc_transient_memory[nextProc],
                                         instance.getComputationalDag().nodeCommunicationWeight(node_pair.first)) >
                            instance.getArchitecture().memoryBound(nextProc)) {
                            toErase.push_back(node_pair);
                        }
                    }
                }

                for (const auto &node : toErase) {
                    procReady[nextProc].erase(node);
                }
            }

            finishTimes.emplace(time + G.nodeWorkWeight(nextNode), nextNode);
            procFree[nextProc] = false;
            --free;
        }

        if (use_memory_constraint && not check_mem_feasibility(instance, allReady, procReady)) {

            return {ERROR, schedule};
        }

        if (free > params_p * max_percent_idle_processors &&
            ((!increase_parallelism_in_new_superstep) ||
             get_nr_parallelizable_nodes(instance, nr_ready_nodes_per_type, nr_procs_per_type) >= std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                                      params_p - free + ((unsigned)(0.5 * free)))))
            endSupStep = true;
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyVarianceScheduler::CanChooseNode(
    const BspInstance &instance, const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
    const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
    const std::vector<bool> &procFree) const {
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
        if (procFree[i] && !procReady[i].empty())
            return true;

    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
        if (procFree[i] && !allReady[instance.getArchitecture().processorType(i)].empty())
            return true;

    return false;
};

void GreedyVarianceScheduler::Choose(
    const BspInstance &instance, const std::vector<double> &work_variance,
    const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
    const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
    const std::vector<bool> &procFree, VertexType &node, unsigned &p) const {

    double maxScore = -1;
    bool found_allocation = false;
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
        if (procFree[i] && !procReady[i].empty()) {
            // select node
            const std::pair<VertexType, double> &node_pair = *procReady[i].begin();
            const double &score = node_pair.second;

            // for (auto it = procReady[i].begin(); it != procReady[i].cend(); it++) {
            //     std::cout << "Node: " << it->first << " Score: " << it->second << ",  ";
            // }
            // std::cout << std::endl;

            if (score > maxScore) {
                maxScore = score;
                node = node_pair.first;
                p = i;
                found_allocation = true;
            }
        }
    }

    if (found_allocation)
        return;

    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
        if (procFree[i] && !allReady[instance.getArchitecture().processorType(i)].empty()) {
            // select node
            for (auto it = allReady[instance.getArchitecture().processorType(i)].begin(); it != allReady[instance.getArchitecture().processorType(i)].end();) {

                const double &score = it->second;

                if (score > maxScore) {
                    if (use_memory_constraint) {

                        if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                            if (current_proc_persistent_memory[i] +
                                    instance.getComputationalDag().nodeMemoryWeight(it->first) <=
                                instance.getArchitecture().memoryBound(i)) {

                                node = it->first;
                                p = i;
                                return;
                            }

                        } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                            if (current_proc_persistent_memory[i] +
                                    instance.getComputationalDag().nodeMemoryWeight(it->first) +
                                    std::max(current_proc_transient_memory[i],
                                            instance.getComputationalDag().nodeCommunicationWeight(it->first)) <=
                                instance.getArchitecture().memoryBound(i)) {

                                node = it->first;
                                p = i;
                                return;
                            }
                        }

                    } else {
                        node = it->first;
                        p = i;
                        return;
                    }
                }
                it++;
            }

        }
    }
};

bool GreedyVarianceScheduler::check_mem_feasibility(
    const BspInstance &instance, const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
    const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady) const {

    if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {
        return true;

    } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (!procReady[i].empty()) {

                const std::pair<VertexType, double> &node_pair = *procReady[i].begin();
                VertexType top_node = node_pair.first;

                if (current_proc_persistent_memory[i] + instance.getComputationalDag().nodeMemoryWeight(top_node) +
                        std::max(current_proc_transient_memory[i],
                                 instance.getComputationalDag().nodeCommunicationWeight(top_node)) <=
                    instance.getArchitecture().memoryBound(i)) {
                    return true;
                }
            }
        }

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {

            if(allReady[instance.getArchitecture().processorType(i)].empty())
                continue;

            const std::pair<VertexType, double> &node_pair = *allReady[instance.getArchitecture().processorType(i)].begin();
            VertexType top_node = node_pair.first;

            if (current_proc_persistent_memory[i] + instance.getComputationalDag().nodeMemoryWeight(top_node) +
                    std::max(current_proc_transient_memory[i],
                                instance.getComputationalDag().nodeCommunicationWeight(top_node)) <=
                instance.getArchitecture().memoryBound(i)) {
                return true;
            }
        }

        return false;
    }
};

// get number of ready nodes that can be run in parallel, to check whether more parallelism is available
// (currently OK for triangular compatibility matrix, otherwise just heuristic - can be changed to matching-based solution later)
unsigned GreedyVarianceScheduler::get_nr_parallelizable_nodes(const BspInstance &instance,
                                            const std::vector<unsigned>& nr_ready_nodes_per_type,
                                            const std::vector<unsigned>& nr_procs_per_type) const {
    unsigned nr_nodes = 0;

    std::vector<unsigned> ready_nodes_per_type = nr_ready_nodes_per_type;
    std::vector<unsigned> procs_per_type = nr_procs_per_type;
    for(unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type)
        for(unsigned node_type = 0; node_type < instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type)
            if(instance.isCompatibleType(node_type, proc_type))
            {
                unsigned matched = std::min(ready_nodes_per_type[node_type], procs_per_type[proc_type]);
                nr_nodes += matched;
                ready_nodes_per_type[node_type] -= matched;
                procs_per_type[proc_type] -= matched;
            }

    return nr_nodes;
}
