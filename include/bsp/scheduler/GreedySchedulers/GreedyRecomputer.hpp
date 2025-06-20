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

#include "bsp/model/BspScheduleRecomp.hpp"

namespace osp {
/**
 * @brief The GreedyReccomputer class applies a greedy algorithm to remove some of the communciation steps in
 * a BspSchedule by recomputation steps if this decreases the cost.
 */
template<typename Graph_t>
class GreedyRecomputer {

    static_assert(is_computational_dag_v<Graph_t>, "GreedyRecomputer can only be used with computational DAGs.");
    
private:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = v_workw_t<Graph_t>;
    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;

    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>, "GreedyRecomputer requires work and comm. weights to have the same type.");


public:

    /**
     * @brief Default destructor for GreedyRecomputer.
     */
    virtual ~GreedyRecomputer() = default;

    RETURN_STATUS computeRecompSchedule(BspScheduleCS<Graph_t> &initial_schedule, BspScheduleRecomp<Graph_t>& out_schedule) const;

};

template<typename Graph_t>
RETURN_STATUS GreedyRecomputer<Graph_t>::computeRecompSchedule(BspScheduleCS<Graph_t> &initial_schedule, BspScheduleRecomp<Graph_t>& out_schedule) const
{
    const vertex_idx& N = initial_schedule.getInstance().numberOfVertices();
    const unsigned& P = initial_schedule.getInstance().numberOfProcessors();
    const unsigned& S = initial_schedule.numberOfSupersteps();
    const Graph_t& G = initial_schedule.getInstance().getComputationalDag();

    out_schedule = BspScheduleRecomp<Graph_t>(initial_schedule.getInstance());
    out_schedule.setNumberOfSupersteps(initial_schedule.numberOfSupersteps());

    // Initialize required data structures
    std::vector<std::vector<cost_type>> work_cost(P, std::vector<cost_type>(S, 0)),
                                        send_cost(P, std::vector<cost_type>(S, 0)),
                                        rec_cost(P, std::vector<cost_type>(S, 0));

    std::vector<std::vector<unsigned>> first_computable(N, std::vector<unsigned>(P, 0U)),
                                        first_present(N, std::vector<unsigned>(P, std::numeric_limits<unsigned>::max()));
    
    std::vector<std::vector<std::multiset<unsigned> > > needed_on_proc(N, std::vector<std::multiset<unsigned> >(P, {S}));
    
    std::vector<cost_type> max_work(S, 0), max_comm(S, 0);

    std::vector<std::set<KeyTriple> > comm_steps(S);                              

    for(vertex_idx node = 0; node < N; ++node)
    {
      const unsigned& proc = initial_schedule.assignedProcessor(node);
      const unsigned& step = initial_schedule.assignedSuperstep(node);

      work_cost[proc][step] += G.vertex_work_weight(node);
      first_present[node][proc] = std::min(first_present[node][proc], step);
      for(vertex_idx pred : G.parents(node))
        needed_on_proc[pred][proc].insert(step);
      
      out_schedule.assignments(node).emplace_back(proc, step);
    }
    for(const std::pair<KeyTriple, unsigned> item : initial_schedule.getCommunicationSchedule())
    {
      const vertex_idx& node = std::get<0>(item.first);
      const unsigned& from_proc = std::get<1>(item.first);
      const unsigned& to_proc = std::get<2>(item.first);
      const unsigned& step = item.second;
      send_cost[from_proc][step] += G.vertex_comm_weight(node) * 
                                      initial_schedule.getInstance().getArchitecture().communicationCosts(from_proc, to_proc);
      rec_cost[to_proc][step] += G.vertex_comm_weight(node) * 
                                      initial_schedule.getInstance().getArchitecture().communicationCosts(from_proc, to_proc);

      comm_steps[step].emplace(item.first);
      needed_on_proc[node][from_proc].insert(step);
      first_present[node][to_proc] = std::min(first_present[node][to_proc], step+1);
    }
    for(unsigned step = 0; step < S; ++step)
      for(unsigned proc = 0; proc < P; ++proc)
      {
        max_work[step] =std::max(max_work[step], work_cost[proc][step]);
        max_comm[step] =std::max(max_comm[step], send_cost[proc][step]);
        max_comm[step] =std::max(max_comm[step], rec_cost[proc][step]);
      }

    for(vertex_idx node = 0; node < N; ++node)
      for(const vertex_idx& pred : G.parents(node))
        for(unsigned proc = 0; proc < P; ++proc)
          first_computable[node][proc] = std::max(first_computable[node][proc], first_present[pred][proc]);
    
    // Find improvement steps
    bool still_improved = true;
    while(still_improved)
    {
      still_improved = false;

      for(unsigned step = 0; step < S; ++step)
      {
        std::vector<KeyTriple> to_erase;
        for(const KeyTriple& entry : comm_steps[step])
        {
          const vertex_idx& node = std::get<0>(entry);
          const unsigned& from_proc = std::get<1>(entry);
          const unsigned& to_proc = std::get<2>(entry);

          // check how much comm cost we save by removing comm schedule entry
          cost_type comm_induced = G.vertex_comm_weight(node) * 
                                      initial_schedule.getInstance().getArchitecture().communicationCosts(from_proc, to_proc);

          cost_type new_max_comm = 0;
          for(unsigned proc = 0; proc < P; ++proc)
          {
            if(proc == from_proc)
              new_max_comm = std::max(new_max_comm, send_cost[proc][step]-comm_induced);
            else
              new_max_comm = std::max(new_max_comm, send_cost[proc][step]);
            if(proc == to_proc)
              new_max_comm = std::max(new_max_comm, rec_cost[proc][step]-comm_induced);
            else
              new_max_comm = std::max(new_max_comm, rec_cost[proc][step]);
          }
          if(new_max_comm == max_comm[step])
            continue;

          if(!initial_schedule.getInstance().isCompatible(node, to_proc))
            continue;

          cost_type decrease = max_comm[step] - new_max_comm;
          if(max_comm[step] > 0 && new_max_comm == 0)
            decrease += initial_schedule.getInstance().getArchitecture().synchronisationCosts();

          // check how much it would increase the work cost instead
          unsigned best_step = S; 
          cost_type smallest_increase = std::numeric_limits<cost_type>::max();
          for(unsigned comp_step = first_computable[node][to_proc]; comp_step <= *needed_on_proc[node][to_proc].begin(); ++comp_step)
          {
            cost_type increase = work_cost[to_proc][comp_step] + G.vertex_work_weight(node) > max_work[comp_step] ?
                                work_cost[to_proc][comp_step] + G.vertex_work_weight(node) - max_work[comp_step] : 0 ;
            
            if(increase < smallest_increase)
            {
              best_step = comp_step;
              smallest_increase = increase;
            }
          }

          // check if this modification is beneficial
          if(best_step == S || smallest_increase > decrease)
            continue;

          // execute the modification
          to_erase.emplace_back(entry);
          out_schedule.assignments(node).emplace_back(to_proc, best_step);

          send_cost[from_proc][step] -= comm_induced;
          rec_cost[to_proc][step] -= comm_induced;
          max_comm[step] = new_max_comm;

          work_cost[to_proc][best_step] += G.vertex_work_weight(node);
          max_work[best_step] += smallest_increase;

          // update movability bounds
          for(const vertex_idx& pred : G.parents(node))
            needed_on_proc[pred][to_proc].insert(best_step);

          needed_on_proc[node][from_proc].erase(needed_on_proc[node][from_proc].lower_bound(step));

          first_present[node][to_proc] = best_step;
          for(const vertex_idx& succ : G.children(node))
          {
            for(const vertex_idx& pred : G.parents(node))
              first_computable[succ][to_proc] = std::max(first_computable[succ][to_proc], first_present[pred][to_proc]);
          }

          still_improved = true;

        }
        for(const KeyTriple& entry : to_erase)
          comm_steps[step].erase(entry);
      }
    }

    for(unsigned step = 0; step < S; ++step)
      for(const KeyTriple& entry : comm_steps[step])
        out_schedule.getCommunicationSchedule().emplace(entry, step);

    out_schedule.mergeSupersteps();

    return SUCCESS;
}

} // namespace osp