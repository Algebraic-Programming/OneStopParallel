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


#include "scheduler/PebblingILP/AuxiliaryForPartialILP/SubproblemMultiScheduling.hpp"
#include <stdexcept>

// currently duplicated from BSP locking scheduler's code
std::vector<int> SubproblemMultiScheduling::get_longest_path(const ComputationalDag &graph) {
    std::vector<int> longest_path(graph.numberOfVertices(), 0);

    const std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        longest_path[*r_iter] = graph.nodeWorkWeight(*r_iter);
        if (graph.numberOfChildren(*r_iter) > 0) {
            int max = 0;
            for (const auto &child : graph.children(*r_iter)) {
                if (max <= longest_path[child])
                    max = longest_path[child];
            }
            longest_path[*r_iter] += max;
        }
    }

    return longest_path;
}


std::pair<RETURN_STATUS, std::vector<std::set<unsigned> > > SubproblemMultiScheduling::computeMultiSchedule(const BspInstance &instance)
{
    const unsigned &N = instance.numberOfVertices();
    const unsigned &P = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    processors_to_nodes.clear();
    processors_to_nodes.resize(N);

    proc_task_lists.clear();
    proc_task_lists.resize(P);

    last_node_on_proc.clear();
    last_node_on_proc.resize(P, UINT_MAX);

    longest_outgoing_path = get_longest_path(G);

    std::set<std::pair<unsigned, unsigned> > readySet;

    std::vector<unsigned> nrPredecRemain(N);
    for (unsigned node = 0; node < N; node++) {
        nrPredecRemain[node] = G.numberOfParents(node);
        if (G.numberOfParents(node) == 0) {
            readySet.emplace(-longest_outgoing_path[node], node);
        }
    }

    std::set<unsigned> free_procs;
    for(unsigned proc = 0; proc < P; ++proc)
        free_procs.insert(proc);

    std::vector<double> node_finish_time(N, 0);

    std::set<std::pair<double, unsigned>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<unsigned>::max());

    unsigned supstepIdx = 0;
    while (!readySet.empty() || !finishTimes.empty()) {

        const double time = finishTimes.begin()->first;

        // Find new ready jobs
        while (!finishTimes.empty() && abs(finishTimes.begin()->first - time) < 0.0001) {

            const unsigned node = finishTimes.begin()->second;
            finishTimes.erase(finishTimes.begin());

            if (node != std::numeric_limits<unsigned>::max())
            {
                for (const auto &succ : G.children(node))
                {
                    nrPredecRemain[succ]--;
                    if (nrPredecRemain[succ] == 0)
                        readySet.emplace(-longest_outgoing_path[succ], succ);
                }
                for(unsigned proc : processors_to_nodes[node])
                    free_procs.insert(proc);
            }
        }

        // Assign new jobs to idle processors

        // first assign free processors to ready nodes
        std::vector<std::pair<unsigned, unsigned> > new_assingments = makeAssignment(instance, readySet, free_procs);

        for(auto entry : new_assingments)
        {
            unsigned node = entry.first;
            unsigned proc = entry.second;

            processors_to_nodes[node].insert(proc);
            proc_task_lists[proc].push_back(node);
            finishTimes.emplace(time + (double) G.nodeWorkWeight(node), node);
            node_finish_time[node] = time + (double) G.nodeWorkWeight(node);
            last_node_on_proc[proc] = node;
            free_procs.erase(proc);
            readySet.erase({-longest_outgoing_path[node], node});
        }

        // assign remaining free processors to already started nodes, if it helps
        std::set<std::pair<double, unsigned>>::reverse_iterator itr = finishTimes.rbegin();
        while(!free_procs.empty() && itr != finishTimes.rend())
        {
            double last_finish_time = itr->first;

            std::set<std::pair<double, unsigned>>::reverse_iterator itr_latest = itr;
            std::set<std::pair<unsigned, unsigned> > possible_nodes;
            while(itr_latest !=finishTimes.rend() && itr_latest->first + 0.0001 > last_finish_time)
            {
                unsigned node = itr_latest->second;
                double new_finish_time = time + (double) G.nodeWorkWeight(node) / ((double) processors_to_nodes[node].size() + 1);
                if(new_finish_time + 0.0001 < itr_latest->first)
                    possible_nodes.emplace(-longest_outgoing_path[node], node);
                
                ++itr_latest;
            }
            std::vector<std::pair<unsigned, unsigned> > new_assingments = makeAssignment(instance, possible_nodes, free_procs);
            for(auto entry : new_assingments)
            {
                unsigned node = entry.first;
                unsigned proc = entry.second;

                processors_to_nodes[node].insert(proc);
                proc_task_lists[proc].push_back(node);
                finishTimes.erase({node_finish_time[node], node});
                double new_finish_time = time + (double) G.nodeWorkWeight(node) / ((double) processors_to_nodes[node].size());
                finishTimes.emplace(new_finish_time, node);
                node_finish_time[node] = new_finish_time;
                last_node_on_proc[proc] = node;
                free_procs.erase(proc);
            }
            if(new_assingments.empty())
                itr = itr_latest;
        }

    }

    return {SUCCESS, processors_to_nodes};
}

std::vector<std::pair<unsigned, unsigned> > SubproblemMultiScheduling::makeAssignment(const BspInstance &instance,
                                                    const std::set<std::pair<unsigned, unsigned> > &nodes_available,
                                                    const std::set<unsigned> &procs_available) const
{
    std::vector<std::pair<unsigned, unsigned> > assignments;
    if(nodes_available.empty() || procs_available.empty())
        return assignments;

    std::set<unsigned> assigned_nodes;
    std::vector<bool> assigned_procs(instance.numberOfProcessors(), false);

    for(unsigned proc : procs_available)
    {
        if(last_node_on_proc[proc] == UINT_MAX)
            continue;

        for (const auto &succ : instance.getComputationalDag().children(last_node_on_proc[proc]))
            if(nodes_available.find({-longest_outgoing_path[succ], succ}) != nodes_available.end() && instance.isCompatible(succ, proc)
                && assigned_nodes.find(succ) == assigned_nodes.end())
            {
                assignments.emplace_back(succ, proc);
                assigned_nodes.insert(succ);
                assigned_procs[proc] = true;
                break;
            }
    }
        
    for(unsigned proc : procs_available)
        if(!assigned_procs[proc])
            for(auto itr = nodes_available.begin(); itr != nodes_available.end(); ++itr)
            {
                unsigned node = itr->second;
                if(instance.isCompatible(node, proc) && assigned_nodes.find(node) == assigned_nodes.end())
                {
                    assignments.emplace_back(node, proc);
                    assigned_nodes.insert(node);
                    break;
                }
            }

    return assignments;
}

std::pair<RETURN_STATUS, BspSchedule> SubproblemMultiScheduling::computeSchedule(const BspInstance &instance) {
    return {ERROR, BspSchedule()};
}
