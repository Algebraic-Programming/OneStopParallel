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

#include "scheduler/GreedySchedulers/Variance_csr.hpp"

std::vector<double> Variance_csr::compute_work_variance(const ComputationalDag &graph) const {
    std::vector<double> work_variance(graph.numberOfVertices(), 0.0);

    const std::vector<VertexType> top_order = graph.dfs_reverse_topoOrder();

    auto iter_end = top_order.end();
    for (auto r_iter = top_order.begin(); r_iter != iter_end; r_iter++) {
        // const std::vector<VertexType> top_order = graph.GetTopOrder();

        // for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        double temp = 0;
        double max_priority = 0;
        for (const auto &child : graph.children(*r_iter)) {
            max_priority = std::max(work_variance[child], max_priority);
        }
        for (const auto &child : graph.children(*r_iter)) {
            temp += std::exp(2 * (work_variance[child] - max_priority));
        }
        temp = std::log(temp) / 2 + max_priority;

        double node_weight = std::log((double)graph.nodeWorkWeight(*r_iter));
        double larger_val = node_weight > temp ? node_weight : temp;

        work_variance[*r_iter] =
            std::log(std::exp(node_weight - larger_val) + std::exp(temp - larger_val)) + larger_val;
    }

    return work_variance;
}

std::vector<double> Variance_csr::compute_work_variance_csr(const csr_graph &graph) const {

    std::vector<double> work_variance(boost::num_vertices(graph), 0.0);

    std::vector<VertexType> top_order;

    std::back_insert_iterator result(top_order);
    typedef boost::topo_sort_visitor<decltype(result)> TopoVisitor;
    boost::depth_first_search(graph, boost::visitor(TopoVisitor(result)));

    auto iter_end = top_order.end();
    for (auto r_iter = top_order.begin(); r_iter != iter_end; r_iter++) {
        // const std::vector<VertexType> top_order = graph.GetTopOrder();

        // for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        double temp = 0;
        double max_priority = 0;

        for (const auto &edge : boost::make_iterator_range(boost::in_edges(*r_iter, graph))) {
            const VertexType child = boost::source(edge, graph);
            max_priority = std::max(work_variance[child], max_priority);
        }
       for (const auto &edge : boost::make_iterator_range(boost::in_edges(*r_iter, graph))) {
            const VertexType child = boost::source(edge, graph);
            temp += std::exp(2 * (work_variance[child] - max_priority));
        }

        //                                                 }
        // for (const auto &child : graph.children(*r_iter)) {
        //     max_priority = std::max(work_variance[child], max_priority);
        // }
        // for (const auto &child : graph.children(*r_iter)) {
        //     temp += std::exp(2 * (work_variance[child] - max_priority));
        // }
        temp = std::log(temp) / 2 + max_priority;

        double node_weight = std::log((double) graph[*r_iter].workWeight);
        double larger_val = node_weight > temp ? node_weight : temp;

        work_variance[*r_iter] =
            std::log(std::exp(node_weight - larger_val) + std::exp(temp - larger_val)) + larger_val;
    }

    return work_variance;
}

void Variance_csr::computeSchedule(const BspInstance_csr &instance) {

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    std::vector<unsigned> node_proc(N, -1);
    std::vector<unsigned> node_superstep(N, 0);

    // approx_transitive_edge_reduction filter(G);

    // boost::filtered_graph<csr_graph , approx_transitive_edge_reduction>  filtered_G(G, filter);

    const std::vector<double> work_variances = compute_work_variance_csr(G);

    procReady = std::vector<std::vector<heap_node>>(params_p);

    allReady.reserve(2 * N / params_p);

    for (unsigned i = 0; i < params_p; ++i) {
        procReady[i].reserve(500);
    }

    std::vector<unsigned> nrPredecRemain(N);
    for (VertexType node = 0; node < N; node++) {
        const unsigned num_parents = boost::in_degree(node, G);

        nrPredecRemain[node] = num_parents;
        if (num_parents == 0) {

            ready.emplace(node);

            allReady.emplace_back(node, work_variances[node]);
            std::push_heap(allReady.begin(), allReady.end());

            // allReady.emplace(node, work_variances[node]);

            // ready.insert(std::make_pair(node, work_variances[node]));
            // allReady.insert(std::make_pair(node, work_variances[node]));
        }
    }

    // std::make_heap(allReady.begin(), allReady.end());

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

            allReady.clear();
            for (const auto &rdy_node : ready) {
                allReady.emplace_back(rdy_node, work_variances[rdy_node]);
                std::push_heap(allReady.begin(), allReady.end());
            }
            // std::make_heap(allReady.begin(), allReady.end());

            // allReady = ready;
            // allReady_handles = ready_handles;

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

                for (const auto &out_edge : boost::make_iterator_range(boost::out_edges(node, G))) {
                    const VertexType succ = boost::target(out_edge, G);

                    // for (const auto &succ : G.children(node)) {
                    nrPredecRemain[succ]--;
                    if (nrPredecRemain[succ] == 0) {

                        ready.emplace(succ);
                        // ready.emplace(succ, work_variances[succ]);

                        bool canAdd = true;

                        for (const auto &edge : boost::make_iterator_range(boost::in_edges(succ, G))) {
                            const VertexType pred = boost::source(edge, G);

                            // for (const auto &pred : G.parents(succ)) {
                            if (node_proc[pred] != node_proc[node] && node_superstep[pred] == supstepIdx)
                                canAdd = false;
                        }

                        if (canAdd) {
                            procReady[node_proc[node]].emplace_back(succ, work_variances[succ]);
                            std::push_heap(procReady[node_proc[node]].begin(), procReady[node_proc[node]].end());
                            // procReady[schedule.assignedProcessor(node)].emplace(succ, work_variances[succ]);
                        }
                    }
                }
                procFree[node_proc[node]] = true;
                ++free;
            }
        }

        // Assign new jobs to processors
        if (!CanChooseNode(params_p, procFree)) {
            endSupStep = true;
        }
        while (CanChooseNode(params_p, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = params_p;
            Choose_csr(params_p, G, work_variances, procFree, nextNode, nextProc, endSupStep, max_finish_time - time);

            if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == params_p) {
                endSupStep = true;
                break;
            }

            ready.erase(nextNode);

            node_proc[nextNode] = nextProc;
            node_superstep[nextNode] = supstepIdx;

            finishTimes.emplace(time + G[nextNode].workWeight, nextNode);
            procFree[nextProc] = false;
            --free;
        }
        if (allReady.empty() && free > params_p * max_percent_idle_processors && ((!increase_parallelism_in_new_superstep) ||
            ready.size() >= std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                                     params_p - free + ((unsigned)(0.5 * free)))))
            endSupStep = true;
    }
};

void Variance_csr::Choose_csr(unsigned num_p, const csr_graph &G, const std::vector<double> &work_variance,
                              const std::vector<bool> &procFree, VertexType &node, unsigned &p, const bool endSupStep,
                              const size_t remaining_time) {

    double maxScore = -1;
    bool found_allocation = false;

    for (unsigned i = 0; i < num_p; ++i) {

        if (procFree[i]) {
            while (!procReady[i].empty()) {

                if (endSupStep && (remaining_time < G[procReady[i][0].node].workWeight)) {
                    std::pop_heap(procReady[i].begin(), procReady[i].end());
                    procReady[i].pop_back();
                    continue;
                }

                const double &score = procReady[i][0].score;
                if (score > maxScore) {
                    maxScore = score;
                    p = i;
                    found_allocation = true;
                }
                break;
            }
        }
    }

    if (found_allocation) {
        std::pop_heap(procReady[p].begin(), procReady[p].end());
        node = procReady[p].back().node;
        procReady[p].pop_back();
        return;
    }

    while (!allReady.empty()) {

        if (endSupStep && (remaining_time < G[allReady[0].node].workWeight)) {
            std::pop_heap(allReady.begin(), allReady.end());
            allReady.pop_back();
            continue;
        }

        for (unsigned i = 0; i < num_p; ++i) {
            if (procFree[i]) {

                if (allReady[0].score > maxScore) {
                    std::pop_heap(allReady.begin(), allReady.end());
                    node = allReady.back().node;
                    allReady.pop_back();
                    p = i;
                    return;
                }
            }
        }
        break;
    }
};

std::pair<RETURN_STATUS, BspSchedule> Variance_csr::computeSchedule(const BspInstance &instance) {

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    // var_dfs_visitor vis;
    // vis.work_var = std::vector<double>(N, 0.0);

    // boost::depth_first_search(G, boost::visitor(vis));

    const std::vector<double> work_variances = compute_work_variance(G);

    // std::cout << "Variance: " << vis.work_var[0] << " " << work_variances[0] << std::endl;
    // std::cout << "Variance: " << vis.work_var[1] << " " << work_variances[1] << std::endl;

    // procReady = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);
    procReady = std::vector<std::vector<heap_node>>(params_p);

    allReady.reserve(2 * N / params_p);

    for (unsigned i = 0; i < params_p; ++i) {
        procReady[i].reserve(500);
    }

    std::vector<unsigned> nrPredecRemain(N);
    for (VertexType node = 0; node < N; node++) {
        const unsigned num_parents = G.numberOfParents(node);
        nrPredecRemain[node] = num_parents;
        if (num_parents == 0) {

            ready.emplace(node);

            allReady.emplace_back(node, work_variances[node]);
            std::push_heap(allReady.begin(), allReady.end());

            // allReady.emplace(node, work_variances[node]);

            // ready.insert(std::make_pair(node, work_variances[node]));
            // allReady.insert(std::make_pair(node, work_variances[node]));
        }
    }

    // std::make_heap(allReady.begin(), allReady.end());

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

            allReady.clear();
            for (const auto &rdy_node : ready) {
                allReady.emplace_back(rdy_node, work_variances[rdy_node]);
                std::push_heap(allReady.begin(), allReady.end());
            }
            // std::make_heap(allReady.begin(), allReady.end());

            // allReady = ready;
            // allReady_handles = ready_handles;

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

                        ready.emplace(succ);
                        // ready.emplace(succ, work_variances[succ]);

                        bool canAdd = true;
                        for (const auto &pred : G.parents(succ)) {
                            if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                schedule.assignedSuperstep(pred) == supstepIdx)
                                canAdd = false;
                        }

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].emplace_back(succ, work_variances[succ]);
                            std::push_heap(procReady[schedule.assignedProcessor(node)].begin(),
                                           procReady[schedule.assignedProcessor(node)].end());
                            // procReady[schedule.assignedProcessor(node)].emplace(succ, work_variances[succ]);
                        }
                    }
                }
                procFree[schedule.assignedProcessor(node)] = true;
                ++free;
            }
        }

        // Assign new jobs to processors
        if (!CanChooseNode(params_p, procFree)) {
            endSupStep = true;
        }
        while (CanChooseNode(params_p, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = params_p;
            Choose(instance, work_variances, procFree, nextNode, nextProc, endSupStep, max_finish_time - time);

            if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == params_p) {
                endSupStep = true;
                break;
            }

            ready.erase(nextNode);

            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            finishTimes.emplace(time + G.nodeWorkWeight(nextNode), nextNode);
            procFree[nextProc] = false;
            --free;
        }
        if (allReady.empty() && free > params_p * max_percent_idle_processors &&
            ready.size() >= std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                                     params_p - free + ((unsigned)(0.5 * free))))
            endSupStep = true;
    }

    //   assert(schedule.satisfiesPrecedenceConstraints());

    //  schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

void Variance_csr::Choose(const BspInstance &instance, const std::vector<double> &work_variance,
                          const std::vector<bool> &procFree, VertexType &node, unsigned &p, const bool endSupStep,
                          const size_t remaining_time) {

    double maxScore = -1;
    bool found_allocation = false;

    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {

        if (procFree[i]) {
            while (!procReady[i].empty()) {

                if (endSupStep &&
                    (remaining_time < instance.getComputationalDag().nodeWorkWeight(procReady[i][0].node))) {
                    std::pop_heap(procReady[i].begin(), procReady[i].end());
                    procReady[i].pop_back();
                    continue;
                }

                const double &score = procReady[i][0].score;
                if (score > maxScore) {
                    maxScore = score;
                    p = i;
                    found_allocation = true;
                }
                break;
            }
        }
    }

    if (found_allocation) {
        std::pop_heap(procReady[p].begin(), procReady[p].end());
        node = procReady[p].back().node;
        procReady[p].pop_back();
        return;
    }

    while (!allReady.empty()) {

        if (endSupStep && (remaining_time < instance.getComputationalDag().nodeWorkWeight(allReady[0].node))) {
            std::pop_heap(allReady.begin(), allReady.end());
            allReady.pop_back();
            continue;
        }

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (procFree[i]) {

                if (allReady[0].score > maxScore) {
                    std::pop_heap(allReady.begin(), allReady.end());
                    node = allReady.back().node;
                    allReady.pop_back();
                    p = i;
                    return;
                }
            }
        }
        break;
    }
};