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

#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp{

template<typename Graph_t>
class SubproblemMultiScheduling : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "PebblingSchedule can only be used with computational DAGs."); 

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using commweight_type = v_commw_t<Graph_t>;
    using workweight_type = v_workw_t<Graph_t>;

    std::vector<vertex_idx> last_node_on_proc;
    std::vector<std::vector<vertex_idx> > proc_task_lists;
    std::vector<workweight_type> longest_outgoing_path;

  public:
    SubproblemMultiScheduling() {}

    virtual ~SubproblemMultiScheduling() = default;

    RETURN_STATUS computeMultiSchedule(const BspInstance<Graph_t> &instance, std::vector<std::set<unsigned> >& processors_to_node);

    std::vector<std::pair<vertex_idx, unsigned> > makeAssignment(const BspInstance<Graph_t> &instance,
                                                    const std::set<std::pair<unsigned, vertex_idx> > &nodes_available,
                                                    const std::set<unsigned> &procs_available) const;

    std::vector<workweight_type> static get_longest_path(const Graph_t &graph);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override;
  
    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "SubproblemMultiScheduling"; }

    inline const std::vector<std::vector<unsigned> >& getProcTaskLists() const { return proc_task_lists; }

};

// currently duplicated from BSP locking scheduler's code
template<typename Graph_t>
std::vector<v_workw_t<Graph_t> > SubproblemMultiScheduling<Graph_t>::get_longest_path(const Graph_t &graph) {
    std::vector<workweight_type> longest_path(graph.num_vertices(), 0);

    std::vector<vertex_idx> top_order = GetTopOrder(graph);

    for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        longest_path[*r_iter] = graph.vertex_work_weight(*r_iter);
        if (graph.out_degree(*r_iter) > 0) {
            workweight_type max = 0;
            for (const auto &child : graph.children(*r_iter)) {
                if (max <= longest_path[child])
                    max = longest_path[child];
            }
            longest_path[*r_iter] += max;
        }
    }

    return longest_path;
}

template<typename Graph_t>
RETURN_STATUS SubproblemMultiScheduling<Graph_t>::computeMultiSchedule(const BspInstance<Graph_t> &instance, std::vector<std::set<unsigned> >& processors_to_node)
{
    const unsigned &N = static_cast<unsigned>(instance.numberOfVertices());
    const unsigned &P = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    processors_to_node.clear();
    processors_to_node.resize(N);

    proc_task_lists.clear();
    proc_task_lists.resize(P);

    last_node_on_proc.clear();
    last_node_on_proc.resize(P, UINT_MAX);

    longest_outgoing_path = get_longest_path(G);

    std::set<std::pair<unsigned, vertex_idx> > readySet;

    std::vector<unsigned> nrPredecRemain(N);
    for (vertex_idx node = 0; node < N; node++) {
        nrPredecRemain[node] = static_cast<unsigned>(G.in_degree(node));
        if (G.in_degree(node) == 0) {
            readySet.emplace(-longest_outgoing_path[node], node);
        }
    }

    std::set<unsigned> free_procs;
    for(unsigned proc = 0; proc < P; ++proc)
        free_procs.insert(proc);

    std::vector<double> node_finish_time(N, 0);

    std::set<std::pair<double, vertex_idx>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<unsigned>::max());

    while (!readySet.empty() || !finishTimes.empty()) {

        const double time = finishTimes.begin()->first;

        // Find new ready jobs
        while (!finishTimes.empty() && fabs(finishTimes.begin()->first - time) < 0.0001 ) {

            const vertex_idx node = finishTimes.begin()->second;
            finishTimes.erase(finishTimes.begin());

            if (node != std::numeric_limits<unsigned>::max())
            {
                for (const vertex_idx &succ : G.children(node))
                {
                    nrPredecRemain[succ]--;
                    if (nrPredecRemain[succ] == 0)
                        readySet.emplace(-longest_outgoing_path[succ], succ);
                }
                for(unsigned proc : processors_to_node[node])
                    free_procs.insert(proc);
            }
        }

        // Assign new jobs to idle processors

        // first assign free processors to ready nodes
        std::vector<std::pair<vertex_idx, unsigned> > new_assingments = makeAssignment(instance, readySet, free_procs);

        for(auto entry : new_assingments)
        {
            vertex_idx node = entry.first;
            unsigned proc = entry.second;

            processors_to_node[node].insert(proc);
            proc_task_lists[proc].push_back(node);
            finishTimes.emplace(time + G.vertex_work_weight(node), node);
            node_finish_time[node] = time + G.vertex_work_weight(node);
            last_node_on_proc[proc] = node;
            free_procs.erase(proc);
            readySet.erase({-longest_outgoing_path[node], node});
        }

        // assign remaining free processors to already started nodes, if it helps
        decltype(finishTimes.rbegin()) itr = finishTimes.rbegin();
        while(!free_procs.empty() && itr != finishTimes.rend())
        {
            double last_finish_time = itr->first;

            decltype(finishTimes.rbegin()) itr_latest = itr;
            std::set<std::pair<workweight_type, vertex_idx> > possible_nodes;
            while(itr_latest !=finishTimes.rend() && itr_latest->first + 0.0001 > last_finish_time)
            {
                vertex_idx node = itr_latest->second;
                double new_finish_time = time + static_cast<double>(G.vertex_work_weight(node)) / (static_cast<double>(processors_to_node[node].size()) + 1);
                if(new_finish_time + 0.0001 < itr_latest->first)
                    possible_nodes.emplace(-longest_outgoing_path[node], node);
                
                ++itr_latest;
            }
            std::vector<std::pair<vertex_idx, unsigned> > new_assingments = makeAssignment(instance, possible_nodes, free_procs);
            for(auto entry : new_assingments)
            {
                vertex_idx node = entry.first;
                unsigned proc = entry.second;

                processors_to_node[node].insert(proc);
                proc_task_lists[proc].push_back(node);
                finishTimes.erase({node_finish_time[node], node});
                double new_finish_time = time + static_cast<double>(G.vertex_work_weight(node)) / (static_cast<double>(processors_to_node[node].size()));
                finishTimes.emplace(new_finish_time, node);
                node_finish_time[node] = new_finish_time;
                last_node_on_proc[proc] = node;
                free_procs.erase(proc);
            }
            if(new_assingments.empty())
                itr = itr_latest;
        }

    }

    return RETURN_STATUS::OSP_SUCCESS;
}

template<typename Graph_t>
std::vector<std::pair<vertex_idx_t<Graph_t>, unsigned> > SubproblemMultiScheduling<Graph_t>::makeAssignment(const BspInstance<Graph_t> &instance,
                                                    const std::set<std::pair<unsigned, vertex_idx> > &nodes_available,
                                                    const std::set<unsigned> &procs_available) const
{
    std::vector<std::pair<vertex_idx, unsigned> > assignments;
    if(nodes_available.empty() || procs_available.empty())
        return assignments;

    std::set<vertex_idx> assigned_nodes;
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
                vertex_idx node = itr->second;
                if(instance.isCompatible(node, proc) && assigned_nodes.find(node) == assigned_nodes.end())
                {
                    assignments.emplace_back(node, proc);
                    assigned_nodes.insert(node);
                    break;
                }
            }

    return assignments;
}

template<typename Graph_t>
RETURN_STATUS SubproblemMultiScheduling<Graph_t>::computeSchedule(BspSchedule<Graph_t> &) {
    return RETURN_STATUS::ERROR;
}

}