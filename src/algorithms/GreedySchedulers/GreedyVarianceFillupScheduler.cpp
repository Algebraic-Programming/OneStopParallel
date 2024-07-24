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

#include "algorithms/GreedySchedulers/GreedyVarianceFillupScheduler.hpp"

std::vector<double> GreedyVarianceFillupScheduler::compute_work_variance(const ComputationalDag &graph) const {
    std::vector<double> work_variance(graph.numberOfVertices(), 0.0);

    const std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        double temp = 0;
        double max_priority = 0;
        for (const auto &child : graph.children(*r_iter)) {
            max_priority = std::max(work_variance[child], max_priority);
        }
        for (const auto &child : graph.children(*r_iter)) {
            temp += std::exp(2*(work_variance[child]-max_priority));
        }
        temp = std::log(temp) / 2 + max_priority;

        double node_weight = std::log((double)graph.nodeWorkWeight(*r_iter));
        double larger_val = node_weight > temp ? node_weight : temp;

        work_variance[*r_iter] = std::log( std::exp(node_weight-larger_val) + std::exp( temp - larger_val ) ) + larger_val;
    }

    return work_variance;
}

std::pair<RETURN_STATUS, BspSchedule> GreedyVarianceFillupScheduler::computeSchedule(const BspInstance &instance) {

    if (use_memory_constraint) {
        current_proc_memory = std::vector<unsigned>(instance.numberOfProcessors(), 0);
    }

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    const std::vector<double> work_variances = compute_work_variance(G);

    std::set<std::pair<VertexType, double>, VarianceCompare> ready;
    std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(params_p);
    std::set<std::pair<VertexType, double>, VarianceCompare> allReady;

    std::vector<unsigned> nrPredecRemain(N);
    for (VertexType node = 0; node < N; node++) {
        const unsigned num_parents = G.numberOfParents(node);
        nrPredecRemain[node] = num_parents;
        if (num_parents == 0) {
            ready.insert(std::make_pair(node, work_variances[node]));
            allReady.insert(std::make_pair(node, work_variances[node]));
        }
        
    }

    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::set<std::pair<size_t, VertexType>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    unsigned supstepIdx = 0;
    bool endSupStep = false;
    while (!ready.empty() || !finishTimes.empty()) {
        if (finishTimes.empty() && endSupStep) {
            for (unsigned i = 0; i < params_p; ++i) {
                procReady[i].clear();
            }

            allReady = ready;

            if (use_memory_constraint) {
                for (unsigned proc = 0; proc < params_p; proc++) {
                    current_proc_memory[proc] = 0;
                }
            }

            ++supstepIdx;
            endSupStep = false;

            finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
        }

        const size_t time = finishTimes.begin()->first;
        const size_t max_finish_time = finishTimes.rbegin()->first;

        // Find new ready jobs
        while (!finishTimes.empty() && finishTimes.begin()->first == time) {
            const VertexType node = finishTimes.begin()->second;
            finishTimes.erase(finishTimes.begin());
            if (node != std::numeric_limits<VertexType>::max()) {
                for (const auto &succ : G.children(node)) {
                    nrPredecRemain[succ]--;
                    if (nrPredecRemain[succ] == 0) {
                        ready.emplace(succ, work_variances[succ]);

                        bool canAdd = true;
                        for (const auto &pred : G.parents(succ)) {
                            if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                schedule.assignedSuperstep(pred) == supstepIdx)
                                canAdd = false;
                        }

                        if (use_memory_constraint && canAdd) {
                            if (current_proc_memory[schedule.assignedProcessor(node)]
                                    + instance.getComputationalDag().nodeMemoryWeight(succ)
                                        > instance.getArchitecture().memoryBound()) {
                                canAdd = false;
                            }
                        } 

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].emplace(succ, work_variances[succ]);
                        }   
                    }
                }
                procFree[schedule.assignedProcessor(node)] = true;
                ++free;
            }
        }

        // Assign new jobs to processors
        if (! CanChooseNode(instance, allReady, procReady, procFree) ) {
            endSupStep = true;
        }
        while (CanChooseNode(instance, allReady, procReady, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = params_p;
            Choose(instance, work_variances, allReady, procReady, procFree, nextNode, nextProc, endSupStep, max_finish_time - time);

            if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == params_p) {
                endSupStep = true;
                break;
            }

            if (procReady[nextProc].find(std::make_pair(nextNode, work_variances[nextNode])) != procReady[nextProc].end()) {
                procReady[nextProc].erase(std::make_pair(nextNode, work_variances[nextNode]));
            } else {
                allReady.erase(std::make_pair(nextNode, work_variances[nextNode]));
            }

            ready.erase(std::make_pair(nextNode, work_variances[nextNode]));
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            if (use_memory_constraint) {
                current_proc_memory[nextProc] += instance.getComputationalDag().nodeMemoryWeight(nextNode);

                std::vector<std::pair<VertexType, double>> toErase;
                for (const auto &node_pair : procReady[nextProc]) {
                    if (current_proc_memory[nextProc] + instance.getComputationalDag().nodeMemoryWeight(node_pair.first) >
                        instance.getArchitecture().memoryBound()) {
                        toErase.push_back(node_pair);
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
        if (allReady.empty() && free > params_p * max_percent_idle_processors && ready.size() >= std::min(std::min(params_p, (unsigned) (1.2 * (params_p - free))), params_p - free + ((unsigned) (0.5 * free)) ))
            endSupStep = true;
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyVarianceFillupScheduler::CanChooseNode(const BspInstance &instance, const std::set<std::pair<VertexType, double>, VarianceCompare> &allReady,
                                            const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
                                            const std::vector<bool> &procFree) const {
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
        if (procFree[i] && !procReady[i].empty())
            return true;

    if (!allReady.empty())
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            if (procFree[i])
                return true;

    return false;
};

void GreedyVarianceFillupScheduler::Choose(const BspInstance &instance, const std::vector<double> &work_variance,
                                     std::set<std::pair<VertexType, double>, VarianceCompare> &allReady, std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
                                     const std::vector<bool> &procFree, VertexType &node, unsigned &p,
                                     const bool endSupStep, const size_t remaining_time) const {

    double maxScore = -1;
    bool found_allocation = false;
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
        if (procFree[i] && !procReady[i].empty()) {
            // select node
            for (auto node_pair_it = procReady[i].begin(); node_pair_it != procReady[i].end(); ) {
                if (endSupStep && (remaining_time < instance.getComputationalDag().nodeWorkWeight(node_pair_it->first))) {
                    node_pair_it = procReady[i].erase(node_pair_it);
                    continue;
                }
                
                const double& score = node_pair_it->second;

                if (score > maxScore) {
                    maxScore = score;
                    node = node_pair_it->first;
                    p = i;
                    found_allocation = true;
                    break;
                }
                node_pair_it++;
            }
        }
    }

    if (found_allocation) return;

    for (auto it = allReady.begin(); it != allReady.end(); ) {
        if (endSupStep && (remaining_time < instance.getComputationalDag().nodeWorkWeight(it->first))) {
            it = allReady.erase(it);
            continue;
        }
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (!procFree[i]) continue;

            double score = it->second;

            if (score > maxScore) {
                if (use_memory_constraint) {

                    if (current_proc_memory[i] + instance.getComputationalDag().nodeMemoryWeight(it->first) <=
                        instance.getArchitecture().memoryBound()) {

                        node = it->first;
                        p = i;
                        return;
                    }
                } else {

                    node = it->first;
                    p = i;
                    return;
                }
            }
        }
        it++;
    }
};