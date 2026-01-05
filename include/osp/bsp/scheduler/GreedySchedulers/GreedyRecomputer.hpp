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

#include "osp/bsp/model/BspScheduleRecomp.hpp"

namespace osp {

/**
 * @brief The GreedyReccomputer class applies a greedy algorithm to replace some of the communciation steps in
 * a BspSchedule by recomputation steps if this decreases the cost.
 */
template <typename GraphT>
class GreedyRecomputer {
    static_assert(isComputationalDagV<GraphT>, "GreedyRecomputer can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;
    using CostType = VWorkwT<GraphT>;
    using KeyTriple = std::tuple<VertexIdxT<GraphT>, unsigned int, unsigned int>;

    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "GreedyRecomputer requires work and comm. weights to have the same type.");

    // auxiliary data to handle schedule efficiently
    std::vector<std::vector<cost_type>> work_cost, send_cost, rec_cost;
    std::vector<std::vector<unsigned>> first_present;
    std::vector<std::vector<std::multiset<unsigned> > > needed_on_proc;
    std::vector<std::vector<std::vector<vertex_idx> > > nodes_per_proc_and_step;
    std::vector<cost_type> max_work, max_comm;
    std::vector<std::set<KeyTriple> > comm_steps;

    void RefreshAuxData(const BspScheduleRecomp<GraphT> &schedule);

    // elementary operations to edit schedule - add/remove step, and update data structures
    void AddCommStep(const BspScheduleRecomp<GraphT> &schedule, const KeyTriple &new_comm, const unsigned step);
    void RemoveCommStep(const BspScheduleRecomp<GraphT> &schedule, const KeyTriple &removed_comm, const unsigned step);
    void AddRecomputeStep(BspScheduleRecomp<GraphT> &schedule, const vertex_idx node, const unsigned proc, const unsigned step);

    // DIFFERENT TECHNIQUES TO IMPROVE SCHEDULE BY INTRODUCING RECOMPUTATION
    // (return values show whether there were any succesful improvement steps)

    // Replace single comm. steps by recomp, if it is better
    bool GreedyImprove(BspScheduleRecomp<GraphT> &schedule);

    // Merge consecutive supersteps using recomp, if it is better
    bool MergeEntireSupersteps(BspScheduleRecomp<GraphT> &schedule);

    // Copy all the (necessary) nodes from one processor to another in a superstep, if it is better
    bool RecomputeEntireSupersteps(BspScheduleRecomp<GraphT> &schedule);

    // Remove multiple comm steps from the same superstep at once, attempting to escape local minima
    bool BatchRemoveSteps(BspScheduleRecomp<GraphT> &schedule);

  public:
    /**
     * @brief Default destructor for GreedyRecomputer.
     */
    virtual ~GreedyRecomputer() = default;

    RETURN_STATUS ComputeRecompScheduleBasic(BspScheduleCS<GraphT> &initial_schedule, BspScheduleRecomp<GraphT> &recomp_schedule);

    RETURN_STATUS ComputeRecompScheduleAdvanced(BspScheduleCS<GraphT> &initial_schedule, BspScheduleRecomp<GraphT> &recomp_schedule);
};

template <typename GraphT>
RETURN_STATUS GreedyRecomputer<GraphT>::computeRecompScheduleBasic(BspScheduleCS<GraphT> &initial_schedule, BspScheduleRecomp<GraphT> &recomp_schedule)
{
    recomp_schedule = BspScheduleRecomp<Graph_t>(initial_schedule);
    GreedyImprove(recomp_schedule);
    recomp_schedule.MergeSupersteps();
    return RETURN_STATUS::OSP_SUCCESS;
}

template <typename GraphT>
RETURN_STATUS GreedyRecomputer<GraphT>::ComputeRecompScheduleAdvanced(BspScheduleCS<GraphT> &initial_schedule, BspScheduleRecomp<GraphT> &recomp_schedule)
{
    recomp_schedule = BspScheduleRecomp<GraphT>(initial_schedule);
    bool keeps_improving = true;
    while (keeps_improving)
    {
      keeps_improving = BatchRemoveSteps(recomp_schedule); // no need for greedyImprove if we use this more general one
      recomp_schedule.MergeSupersteps();

      keeps_improving = MergeEntireSupersteps(recomp_schedule) || keeps_improving;
      recomp_schedule.CleanSchedule();
      recomp_schedule.MergeSupersteps();

      keeps_improving = RecomputeEntireSupersteps(recomp_schedule) || keeps_improving;
      recomp_schedule.MergeSupersteps();

      // add further methods, if desired
    }
    
    return RETURN_STATUS::OSP_SUCCESS;
}

template <typename GraphT>
bool GreedyRecomputer<GraphT>::GreedyImprove(BspScheduleRecomp<GraphT> &schedule)
{
    const vertex_idx N = schedule.GetInstance().NumberOfVertices();
    const unsigned P = schedule.GetInstance().NumberOfProcessors();
    const unsigned S = schedule.NumberOfSupersteps();
    const GraphT &G = schedule.GetInstance().GetComputationalDag();

    bool improved = false;

    // Initialize required data structures
    RefreshAuxData(schedule);

    std::vector<std::vector<unsigned>> first_computable(N, std::vector<unsigned>(P, 0U));
    for (vertex_idx node = 0; node < N; ++node) {
      for (const vertex_idx &pred : G.Parents(node)) {
        for (unsigned proc = 0; proc < P; ++proc) {
          first_computable[node][proc] = std::max(first_computable[node][proc], first_present[pred][proc]);
        }
      }
    }

    // Find improvement steps
    bool still_improved = true;
    while (still_improved) {
      still_improved = false;

      for (unsigned step = 0; step < S; ++step) {
        std::vector<KeyTriple> to_erase;
        for (const KeyTriple &entry : comm_steps[step]) {
          const vertex_idx &node = std::get<0>(entry);
          const unsigned &from_proc = std::get<1>(entry);
          const unsigned &to_proc = std::get<2>(entry);

          // check how much comm cost we save by removing comm schedule entry
          cost_type comm_induced = G.VertexCommWeight(node)
                                   * schedule.GetInstance().GetArchitecture().CommunicationCosts(from_proc, to_proc);

          cost_type new_max_comm = 0;
          for (unsigned proc = 0; proc < P; ++proc) {
            if (proc == from_proc) {
              new_max_comm = std::max(new_max_comm, send_cost[proc][step]-comm_induced);
            } else {
              new_max_comm = std::max(new_max_comm, send_cost[proc][step]);
            }
            if (proc == to_proc) {
              new_max_comm = std::max(new_max_comm, rec_cost[proc][step]-comm_induced);
            } else {
              new_max_comm = std::max(new_max_comm, rec_cost[proc][step]);
            }
          }
          if (new_max_comm == max_comm[step]) {
            continue;
          }

          if (!schedule.GetInstance().IsCompatible(node, to_proc)) {
            continue;
          }

          cost_type decrease = max_comm[step] - new_max_comm;
          if (max_comm[step] > 0 && new_max_comm == 0) {
            decrease += schedule.GetInstance().GetArchitecture().SynchronisationCosts();
          }

          // check how much it would increase the work cost instead
          unsigned best_step = S;
          cost_type smallest_increase = std::numeric_limits<cost_type>::max();
          for (unsigned comp_step = first_computable[node][to_proc]; comp_step <= *needed_on_proc[node][to_proc].begin(); ++comp_step) {
            cost_type increase = work_cost[to_proc][comp_step] + G.VertexWorkWeight(node) > max_work[comp_step]
                                  ? work_cost[to_proc][comp_step] + G.VertexWorkWeight(node) - max_work[comp_step]
                                  : 0;
            
            if (increase < smallest_increase) {
              best_step = comp_step;
              smallest_increase = increase;
            }
          }

          // check if this modification is beneficial
          if (best_step == S || smallest_increase > decrease) {
            continue;
          }

          // execute the modification
          to_erase.emplace_back(entry);
          AddRecomputeStep(schedule, node, to_proc, best_step);
          improved = true;

          send_cost[from_proc][step] -= comm_induced;
          rec_cost[to_proc][step] -= comm_induced;
          max_comm[step] = new_max_comm;

          max_work[best_step] += smallest_increase;

          // update movability bounds
          needed_on_proc[node][from_proc].erase(needed_on_proc[node][from_proc].lower_bound(step));

          first_present[node][to_proc] = best_step;
          for (const vertex_idx &succ : G.Children(node)) {
            first_computable[succ][to_proc] = 0U;
            for (const vertex_idx &pred : G.parents(succ)) {
              first_computable[succ][to_proc] = std::max(first_computable[succ][to_proc], first_present[pred][to_proc]);
            }
          }

          still_improved = true;
        }
        for (const KeyTriple &entry : to_erase) {
          comm_steps[step].erase(entry);
        }
      }
    }

    schedule.GetCommunicationSchedule().clear();
    for (unsigned step = 0; step < S; ++step) {
      for (const KeyTriple &entry : comm_steps[step]) {
        schedule.AddCommunicationScheduleEntry(entry, step);
      }
    }

    return improved;
}

template <typename GraphT>
bool GreedyRecomputer<GraphT>::MergeEntireSupersteps(BspScheduleRecomp<GraphT> &schedule)
{
  bool improved = false;
  RefreshAuxData(schedule);
  std::vector<bool> step_removed(schedule.NumberOfSupersteps(), false);

  const GraphT &G = schedule.GetInstance().GetComputationalDag();
  
  unsigned previous_step = 0;
  for (unsigned step = 0; step < schedule.NumberOfSupersteps() - 1; ++step) {
    if (step_removed[step]) {
      continue;
    }

    for (unsigned next_step = step + 1; next_step < schedule.NumberOfSupersteps(); ++next_step) {
      
      // TRY TO MERGE step AND next_step
      std::set<KeyTriple> new_comm_steps_before, new_comm_steps_after;
      std::set<std::pair<vertex_idx, unsigned> > new_work_steps;

      std::vector<std::set<vertex_idx> > must_replicate(schedule.GetInstance().NumberOfProcessors());

      for (const KeyTriple &entry : comm_steps[step]) {
        const vertex_idx &node = std::get<0>(entry);
        const unsigned &from_proc = std::get<1>(entry);
        const unsigned &to_proc = std::get<2>(entry);

        bool used = false;
        if (needed_on_proc[node][to_proc].empty() || *needed_on_proc[node][to_proc].begin() > next_step) {
          new_comm_steps_after.insert(entry);
          continue;
        }
        
        if (step > 0 && first_present[node][from_proc] <= previous_step) {
          new_comm_steps_before.insert(entry);
        } else {
          must_replicate[to_proc].insert(node);
        }
      }
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        for (const vertex_idx node : nodes_per_proc_and_step[proc][next_step]) {
          new_work_steps.emplace(node, proc);
        }

        while (!must_replicate[proc].empty()) {
          const vertex_idx node = *must_replicate[proc].begin();
          must_replicate[proc].erase(must_replicate[proc].begin());
          if (new_work_steps.find(std::make_pair(node, proc)) != new_work_steps.end()) {
            continue;
          }
          new_work_steps.emplace(node, proc);
          for (const vertex_idx &pred : G.Parents(node)) {
            if (first_present[pred][proc] <= step) {
              continue;
            }
            
            unsigned send_from_proc_before = std::numeric_limits<unsigned>::max();
            for (unsigned proc_offset = 0; proc_offset < schedule.GetInstance().NumberOfProcessors(); ++proc_offset) {
              unsigned from_proc = (proc + proc_offset) % schedule.GetInstance().NumberOfProcessors();
              if (step > 0 && first_present[pred][from_proc] <= previous_step) {
                send_from_proc_before = from_proc;
                break;
              }
            }
            if (send_from_proc_before < std::numeric_limits<unsigned>::max()) {
              new_comm_steps_before.emplace(pred, send_from_proc_before, proc);
            } else {
              must_replicate[proc].insert(pred);
            }
          }
        }
      }

      // now that new_work_steps is finalized, check types
      bool types_incompatible = false;
      for (const std::pair<vertex_idx, unsigned> &node_and_proc : new_work_steps) {
        if (!schedule.GetInstance().IsCompatible(node_and_proc.first, node_and_proc.second)) {
            types_incompatible = true;
            break;
        }
      }
      if (types_incompatible) {
        break;
      }

      // EVALUATE COST  
      int cost_change = 0;
      
      // work cost in merged step
      std::vector<cost_type> new_work_cost(schedule.GetInstance().NumberOfProcessors());
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        new_work_cost[proc] = work_cost[proc][step];
      }

      for (const std::pair<vertex_idx, unsigned> &new_compute : new_work_steps) {
        new_work_cost[new_compute.second] += G.VertexWorkWeight(new_compute.first);
      }

      cost_type new_max = 0;
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        new_max = std::max(new_max, new_work_cost[proc]);
      }
      
      cost_change += static_cast<int>(new_max) - static_cast<int>(max_work[step] + max_work[next_step]);

      // comm cost before merged step
      std::vector<cost_type> new_send_cost(schedule.GetInstance().NumberOfProcessors()), new_rec_cost(schedule.GetInstance().NumberOfProcessors());
      if (step > 0) {
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          new_send_cost[proc] = send_cost[proc][previous_step];
          new_rec_cost[proc] = rec_cost[proc][previous_step];
        }
        for (const KeyTriple &new_comm : new_comm_steps_before) {
          cost_type comm_cost = G.VertexCommWeight(std::get<0>(new_comm)) * 
                                      schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(new_comm), std::get<2>(new_comm));
          new_send_cost[std::get<1>(new_comm)] += comm_cost;
          new_rec_cost[std::get<2>(new_comm)] += comm_cost;
        }
        
        new_max = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          new_max = std::max(new_max, new_send_cost[proc]);
          new_max = std::max(new_max, new_rec_cost[proc]);
        }
        cost_change += static_cast<int>(new_max) - static_cast<int>(max_comm[previous_step]);

        cost_type old_sync = (max_comm[previous_step] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;
        cost_type new_sync = (new_max > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        cost_change += static_cast<int>(new_sync) - static_cast<int>(old_sync);
      }

      // comm cost after merged step
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        new_send_cost[proc] = send_cost[proc][next_step];
        new_rec_cost[proc] = rec_cost[proc][next_step];
      }
      for (const KeyTriple &new_comm : new_comm_steps_after) {
        cost_type comm_cost = G.VertexCommWeight(std::get<0>(new_comm))
                              * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(new_comm), std::get<2>(new_comm));
        new_send_cost[std::get<1>(new_comm)] += comm_cost;
        new_rec_cost[std::get<2>(new_comm)] += comm_cost;
      }
      
      new_max = 0;
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        new_max = std::max(new_max, new_send_cost[proc]);
        new_max = std::max(new_max, new_rec_cost[proc]);
      }
      cost_change += static_cast<int>(new_max) - static_cast<int>(max_comm[step] + max_comm[next_step]);

      cost_type old_sync = ((max_comm[step] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0)
                            + ((max_comm[next_step] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0);
      cost_type new_sync = (new_max > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;

      cost_change += static_cast<int>(new_sync) - static_cast<int>(old_sync);

      if (cost_change < 0)
      {
        // MERGE STEPS - change schedule and update data structures

        // update assignments and compute data
        for (const std::pair<vertex_idx, unsigned> &node_and_proc : new_work_steps) {
          AddRecomputeStep(schedule, node_and_proc.first, node_and_proc.second, step);
        }
        max_work[step] = 0;
        max_work[next_step] = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          max_work[step] = std::max(max_work[step], work_cost[proc][step]);
          work_cost[proc][next_step] = 0;
          for (const vertex_idx node : nodes_per_proc_and_step[proc][next_step]) {
            auto &assignments = schedule.assignments(node);
            for (auto itr = assignments.begin(); itr != assignments.end(); ++itr) {
              if (*itr == std::make_pair(proc, next_step)) {
                assignments.erase(itr);
                break;
              }
            }
            for (const vertex_idx &pred : G.Parents(node)) {
              needed_on_proc[pred][proc].erase(needed_on_proc[pred][proc].lower_bound(next_step));
            }
          }
          nodes_per_proc_and_step[proc][next_step].clear();
        }

        // update comm and its data in step (imported mostly from next_step)
        for (const KeyTriple &entry : comm_steps[step]) {
          needed_on_proc[std::get<0>(entry)][std::get<1>(entry)].erase(needed_on_proc[std::get<0>(entry)][std::get<1>(entry)].lower_bound(step));
        }
        
        for (const KeyTriple &entry : comm_steps[next_step]) {
          needed_on_proc[std::get<0>(entry)][std::get<1>(entry)].erase(needed_on_proc[std::get<0>(entry)][std::get<1>(entry)].lower_bound(next_step));
        }
        
        comm_steps[step].clear();
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          send_cost[proc][step] = 0;
          rec_cost[proc][step] = 0;
          send_cost[proc][next_step] = 0;
          rec_cost[proc][next_step] = 0;
        }
        std::set<KeyTriple> comm_next_step = comm_steps[next_step];
        comm_steps[next_step].clear();
        for (const KeyTriple &new_comm : comm_next_step) {
          AddCommStep(schedule, new_comm, step);
        }

        for (const KeyTriple &new_comm : new_comm_steps_after) {
          AddCommStep(schedule, new_comm, step);
        }

        max_comm[next_step] = 0;
          
        max_comm[step] = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          max_comm[step] = std::max(max_comm[step], send_cost[proc][step]);
          max_comm[step] = std::max(max_comm[step], rec_cost[proc][step]);
        }

        // update comm and its data in step-1
        if (step > 0) {
          for (const KeyTriple &new_comm : new_comm_steps_before) {
            AddCommStep(schedule, new_comm, previous_step);
          }
          
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            max_comm[previous_step] = std::max(max_comm[previous_step], send_cost[proc][previous_step]);
            max_comm[previous_step] = std::max(max_comm[previous_step], rec_cost[proc][previous_step]);
          }
        }

        step_removed[next_step] = true;
        improved = true;
      } else {
        break;
      }
    }
    previous_step = step;
  }

  schedule.GetCommunicationSchedule().clear();
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (const KeyTriple &entry : comm_steps[step]) {
      schedule.AddCommunicationScheduleEntry(entry, step);
    }
  }

  return improved;
}

template <typename GraphT>
bool GreedyRecomputer<GraphT>::RecomputeEntireSupersteps(BspScheduleRecomp<GraphT> &schedule)
{
  bool improved = false;
  RefreshAuxData(schedule);

  const GraphT &G = schedule.getInstance().GetComputationalDag();

  std::map<std::pair<vertex_idx, unsigned>, std::vector<std::pair<unsigned, unsigned>>> comm_step_per_node_and_receiver;
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (const KeyTriple &entry : comm_steps[step]) {
      comm_step_per_node_and_receiver[std::make_pair(std::get<0>(entry), std::get<2>(entry))].emplace_back(std::get<1>(entry), step);
    }
  }
  
  for (unsigned step = 0; step < schedule.numberOfSupersteps(); ++step) {
    for (unsigned from_proc = 0; from_proc < schedule.GetInstance().NumberOfProcessors(); ++from_proc) {
      for (unsigned to_proc = 0; to_proc < schedule.GetInstance().NumberOfProcessors(); ++to_proc) {
        if (from_proc == to_proc) {
          continue;
        }

        // ATTEMPT TO REPLICATE all the necessary nodes of (from_proc, step) on (to_proc, step)

        // collect the nodes that would be useful to replicate (not present before, not unnecessary)
        std::set<KeyTriple> new_comm_steps_before, removed_comm_steps_after;
        std::set<vertex_idx> must_replicate;

        for (const vertex_idx node : nodes_per_proc_and_step[from_proc][step])
        {
          if (first_present[node][to_proc] <= step) {
            continue;
          }
          must_replicate.insert(node);
        }

        std::map<vertex_idx, unsigned> internal_out_degree;
        for (const vertex_idx node : must_replicate) {
          internal_out_degree[node] = 0;
        }
        for (const vertex_idx node : must_replicate) {
          for (const vertex_idx &pred : G.Parents(node)) {
            if (must_replicate.find(pred) == must_replicate.end()) {
              continue;
            }
            internal_out_degree[pred] += 1;
          }
        }
        
        std::set<vertex_idx> check_if_disposable;
        for (const vertex_idx node : must_replicate) {
          if (internal_out_degree.at(node) == 0) {
            check_if_disposable.insert(node);
          }
        }

        while (!check_if_disposable.empty()) {
          const vertex_idx node = *check_if_disposable.begin();
          check_if_disposable.erase(check_if_disposable.begin());
          if (needed_on_proc[node][to_proc].empty()) {
            must_replicate.erase(node);
            for (const vertex_idx &pred : G.parents(node)) {
              if (must_replicate.find(pred) == must_replicate.end()) {
                continue;
              }
              if ((--internal_out_degree[pred]) == 0) {
                check_if_disposable.insert(pred);
              }
            }
          }
        }

        // now that must_replicate is finalized, check types
        bool types_incompatible = false;
        for (const vertex_idx node : must_replicate) {
          if (!schedule.GetInstance().IsCompatible(node, to_proc)) {
              types_incompatible = true;
              break;
            }
        }
        if (types_incompatible) {
          continue;
        }

        // collect new comm steps - before
        for (const vertex_idx node : must_replicate) {
          for (const vertex_idx &pred : G.Parents(node)) {
            if (first_present[pred][to_proc] <= step || must_replicate.find(pred) != must_replicate.end()) {
              continue;
            }

            unsigned send_from_proc_before = from_proc;
            for (unsigned proc_offset = 0; proc_offset < schedule.getInstance().numberOfProcessors(); ++proc_offset) {
              unsigned send_from_candidate = (from_proc + proc_offset) % schedule.GetInstance().NumberOfProcessors();
              if (step > 0 && first_present[pred][send_from_candidate] <= step - 1) {
                send_from_proc_before = send_from_candidate;
                break;
              }
            }
            if (send_from_proc_before < std::numeric_limits<unsigned>::max()) {
              new_comm_steps_before.emplace(pred, send_from_proc_before, to_proc);
            } else {
              std::cout<<"ERROR: parent of replicated node not present anywhere."<<std::endl;
            }
          }
        }

        // collect comm steps to remove afterwards
        for (const vertex_idx node : must_replicate) {
          for (const std::pair<unsigned, unsigned> &entry : comm_step_per_node_and_receiver[std::make_pair(node, to_proc)]) {
            removed_comm_steps_after.emplace(node, entry.first, entry.second);
          }
        }

        // EVALUATE COST

        int cost_change = 0;
      
        // work cost
        cost_type new_work_cost = work_cost[to_proc][step];
        for (const vertex_idx node : must_replicate) {
          new_work_cost += G.VertexWorkWeight(node);
        }
        cost_type new_max = std::max(max_work[step], new_work_cost);
        
        cost_change += static_cast<int>(new_max) - static_cast<int>(max_work[step]);

        // comm cost before merged step
        if (step > 0) {
          std::vector<cost_type> new_send_cost(schedule.GetInstance().NumberOfProcessors());
          cost_type new_rec_cost = rec_cost[to_proc][step-1];
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            new_send_cost[proc] = send_cost[proc][step-1];
          }
          for (const KeyTriple &new_comm : new_comm_steps_before)
          {
            cost_type comm_cost = G.VertexCommWeight(std::get<0>(new_comm))
                                  * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(new_comm), std::get<2>(new_comm));
            new_send_cost[std::get<1>(new_comm)] += comm_cost;
            new_rec_cost += comm_cost;
          }
          
          new_max = std::max(max_comm[step - 1], new_rec_cost);
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            new_max = std::max(new_max, new_send_cost[proc]);
          }
          cost_change += static_cast<int>(new_max) - static_cast<int>(max_comm[step - 1]);

          cost_type old_sync = (max_comm[step - 1] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;
          cost_type new_sync = (new_max > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;

          cost_change += static_cast<int>(new_sync) - static_cast<int>(old_sync);
        }

        // comm cost after merged step
        std::map<unsigned, std::map<unsigned, cost_type>> changed_steps_sent;
        std::map<unsigned, cost_type> changed_steps_rec;
        for (const KeyTriple &new_comm : removed_comm_steps_after) {
          cost_type comm_cost = G.VertexCommWeight(std::get<0>(new_comm))
                                * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(new_comm), to_proc);
          if (changed_steps_sent[std::get<2>(new_comm)].find(std::get<1>(new_comm)) == changed_steps_sent[std::get<2>(new_comm)].end()) {
            changed_steps_sent[std::get<2>(new_comm)][std::get<1>(new_comm)] = comm_cost;
          } else {
            changed_steps_sent[std::get<2>(new_comm)][std::get<1>(new_comm)] += comm_cost;
          }
          if (changed_steps_rec.find(std::get<2>(new_comm)) == changed_steps_rec.end()) {
            changed_steps_rec[std::get<2>(new_comm)] = comm_cost;
          } else {
            changed_steps_rec[std::get<2>(new_comm)] += comm_cost;
          }
        }
        for (const auto &changing_step : changed_steps_rec) {
          unsigned step_changed = changing_step.first;
          
          std::vector<cost_type> new_send_cost(schedule.GetInstance().NumberOfProcessors());
          cost_type new_rec_cost = rec_cost[to_proc][step_changed] - changing_step.second;
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            new_send_cost[proc] = send_cost[proc][step_changed];
          }
          for (const auto &proc_and_change : changed_steps_sent[step_changed]) {
            new_send_cost[proc_and_change.first] -= proc_and_change.second;
          }
          
          new_max = 0;
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            new_max = std::max(new_max, new_send_cost[proc]);
          }
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            if (proc == to_proc) {
              new_max = std::max(new_max, new_rec_cost);
            } else {
              new_max = std::max(new_max, rec_cost[proc][step_changed]);
            }
          }
          cost_change += static_cast<int>(new_max) - static_cast<int>(max_comm[step_changed]);

          cost_type old_sync = (max_comm[step_changed] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;
          cost_type new_sync = (new_max > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;

          cost_change += static_cast<int>(new_sync) - static_cast<int>(old_sync);
        }

        if (cost_change < 0) {
          // REPLICATE STEP IF BENEFICIAL - change schedule and update data structures

          // update assignments and compute data
          for (const vertex_idx node : must_replicate) {
            AddRecomputeStep(schedule, node, to_proc, step);
            auto itr = comm_step_per_node_and_receiver.find(std::make_pair(node, to_proc));
            if (itr != comm_step_per_node_and_receiver.end()) {
              comm_step_per_node_and_receiver.erase(itr);
            }
          }
          max_work[step] = std::max(max_work[step], work_cost[to_proc][step]);

          // update comm and its data in step-1
          if (step > 0) {
            for (const KeyTriple &new_comm : new_comm_steps_before) {
              addCommStep(schedule, new_comm, step - 1);
            }
            
            for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
              max_comm[step - 1] = std::max(max_comm[step - 1], send_cost[proc][step - 1]);
              max_comm[step - 1] = std::max(max_comm[step - 1], rec_cost[proc][step - 1]);
            }
          }

          // update comm and its data in later steps
          for (const KeyTriple &new_comm : removed_comm_steps_after) {
            unsigned changing_step = std::get<2>(new_comm);
            RemoveCommStep(schedule, KeyTriple(std::get<0>(new_comm), std::get<1>(new_comm), to_proc), changing_step);
          }
          for (const auto &step_and_change : changed_steps_rec) {
            unsigned changing_step = step_and_change.first;
            max_comm[changing_step] = 0;
            for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
              max_comm[changing_step] = std::max(max_comm[changing_step], send_cost[proc][changing_step]);
              max_comm[changing_step] = std::max(max_comm[changing_step], rec_cost[proc][changing_step]);
            }
          }

          improved = true;
        }
      }
    }
  }

  schedule.GetCommunicationSchedule().clear();
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (const KeyTriple &entry : comm_steps[step]) {      
      schedule.AddCommunicationScheduleEntry(entry, step);
    }
  }

  return improved;
}

template <typename GraphT>
bool GreedyRecomputer<GraphT>::BatchRemoveSteps(BspScheduleRecomp<GraphT> &schedule)
{
  bool improved = false;
  const GraphT &G = schedule.GetInstance().GetComputationalDag();

  // Initialize required data structures
  RefreshAuxData(schedule);

  std::vector<std::vector<unsigned>> first_computable(schedule.GetInstance().NumberOfVertices(), std::vector<unsigned>(schedule.GetInstance().NumberOfProcessors(), 0U));
  for (vertex_idx node = 0; node < schedule.GetInstance().NumberOfVertices(); ++node) {
    for (const vertex_idx &pred : G.Parents(node)) {
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        first_computable[node][proc] = std::max(first_computable[node][proc], first_present[pred][proc]);
      }
    }
  }
  
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    
    bool can_reduce = (max_comm[step] > 0);
    while (can_reduce) {

      // find processors where send/rec costs equals the maximum (so we want to remove comm steps)
      can_reduce = false;
      std::vector<bool> send_saturated(schedule.GetInstance().NumberOfProcessors(), false), rec_saturated(schedule.GetInstance().NumberOfProcessors(), false);
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        if (send_cost[proc][step] == max_comm[step]) {
          send_saturated[proc] = true;
        }
        if (rec_cost[proc][step] == max_comm[step]) {
          rec_saturated[proc] = true;
        }
      }

      // initialize required variables
      std::map<std::pair<unsigned, unsigned>, cost_type> work_increased;
      std::set<KeyTriple> removed_comm_steps, added_compute_steps;
      std::vector<std::set<KeyTriple> > send_comm_steps(schedule.GetInstance().NumberOfProcessors()),
                                        rec_comm_steps(schedule.GetInstance().NumberOfProcessors());
      for (const KeyTriple &comm_step : comm_steps[step]) {
        send_comm_steps[std::get<1>(comm_step)].insert(comm_step);
        rec_comm_steps[std::get<2>(comm_step)].insert(comm_step);
      }
      bool skip_step = false;
      cost_type work_increase = 0;
      cost_type comm_decrease = std::numeric_limits<cost_type>::max();

      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        for (unsigned send_or_rec = 0; send_or_rec < 2; ++send_or_rec) {
         
          std::set<KeyTriple> *current_comm_steps; 
          if (send_or_rec == 0) {
            if (!send_saturated[proc]) {
              continue;
            }
            current_comm_steps = &send_comm_steps[proc];
          } else {
            if (!rec_saturated[proc]) {
              continue;
            }
            current_comm_steps = &rec_comm_steps[proc];
          }

          KeyTriple best_comm_step;
          unsigned best_step_target = std::numeric_limits<unsigned>::max();
          cost_type smallest_increase = std::numeric_limits<cost_type>::max();
          for (const KeyTriple &comm_step : *current_comm_steps) {
            const vertex_idx node = std::get<0>(comm_step);
            const unsigned from_proc = std::get<1>(comm_step);
            const unsigned to_proc = std::get<2>(comm_step);
            if (G.VertexCommWeight(node) == 0) {
              continue;
            }
            if (!schedule.GetInstance().IsCompatible(node, to_proc)) {
              continue;
            }

            for (unsigned comp_step = first_computable[node][to_proc]; comp_step <= *needed_on_proc[node][to_proc].begin(); ++comp_step) {
              auto itr = work_increased.find(std::make_pair(to_proc, comp_step));
              cost_type assigned_extra = (itr != work_increased.end()) ? itr->second : 0;
              cost_type increase = 0;
              if (work_cost[to_proc][comp_step] + assigned_extra + G.VertexWorkWeight(node) > max_work[comp_step]) {
                increase = work_cost[to_proc][comp_step] + assigned_extra + G.VertexWorkWeight(node) - max_work[comp_step];
              }
              if (increase < smallest_increase) {
                smallest_increase = increase;
                best_step_target = comp_step;
                best_comm_step = comm_step;
              }
            }
          }

          // save this if this is the cheapest way to move away a comm step
          if (smallest_increase < std::numeric_limits<cost_type>::max()) {
            const vertex_idx node = std::get<0>(best_comm_step);
            const unsigned from_proc = std::get<1>(best_comm_step);
            const unsigned to_proc = std::get<2>(best_comm_step);
            added_compute_steps.emplace(node, to_proc, best_step_target);
            auto itr = work_increased.find(std::make_pair(to_proc, best_step_target));
            if (itr == work_increased.end()) {
              work_increased[std::make_pair(to_proc, best_step_target)] = G.VertexWorkWeight(node);
            } else {
              itr->second += G.VertexWorkWeight(node);
            }

            send_saturated[from_proc] = false;
            rec_saturated[to_proc] = false;

            removed_comm_steps.insert(best_comm_step);
            work_increase += smallest_increase;
            cost_type comm_cost = schedule.GetInstance().GetComputationalDag().VertexCommWeight(node)
                        * schedule.GetInstance().GetArchitecture().CommunicationCosts(from_proc, to_proc);
            comm_decrease = std::min(comm_decrease, comm_cost);

          } else {
            skip_step = true;
          }
        }
        if (skip_step) {
          // weird edge case if all comm steps have weight 0 (can be removed?)
          break;
        }              
      }
      if (skip_step) {
        continue;
      }                               

      if (max_comm[step] > 0 && comm_steps[step].size() == removed_comm_steps.size()) {
        comm_decrease += schedule.GetInstance().GetArchitecture().SynchronisationCosts();
      }
      // execute step if total work cost increase < total comm cost decrease
      if (comm_decrease > work_increase) {
        for (const KeyTriple &new_comp : added_compute_steps) {
          const vertex_idx node = std::get<0>(new_comp);
          const unsigned proc = std::get<1>(new_comp);
          const unsigned new_step = std::get<2>(new_comp);
          AddRecomputeStep(schedule, node, proc, new_step);
          first_present[node][proc] = new_step;
          max_work[new_step] = std::max(max_work[new_step], work_cost[proc][new_step]);

          for (const vertex_idx &succ : G.Children(node)) {
            first_computable[succ][proc] = 0U;
            for (const vertex_idx &pred : G.Parents(succ)) {
              first_computable[succ][proc] = std::max(first_computable[succ][proc], first_present[pred][proc]);
            }
          }
        }
        for (const KeyTriple &removed_comm : removed_comm_steps) {
          RemoveCommStep(schedule, removed_comm, step);
        }
        max_comm[step] = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          max_comm[step] = std::max(max_comm[step], send_cost[proc][step]);
          max_comm[step] = std::max(max_comm[step], rec_cost[proc][step]);
        }
        
        can_reduce = true;
        improved = true;
      }
    }
  }

  schedule.GetCommunicationSchedule().clear();
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (const KeyTriple &entry : comm_steps[step]) {
      schedule.AddCommunicationScheduleEntry(entry, step);
    }
  }

  return improved;
}

template <typename GraphT>
void GreedyRecomputer<GraphT>::RefreshAuxData(const BspScheduleRecomp<GraphT> &schedule)
{
    const vertex_idx N = schedule.GetInstance().NumberOfVertices();
    const unsigned P = schedule.GetInstance().NumberOfProcessors();
    const unsigned S = schedule.NumberOfSupersteps();
    const GraphT &G = schedule.GetInstance().GetComputationalDag();

    work_cost.clear();
    send_cost.clear();
    rec_cost.clear();

    work_cost.resize(P, std::vector<cost_type>(S, 0));
    send_cost.resize(P, std::vector<cost_type>(S, 0)),
    rec_cost.resize(P, std::vector<cost_type>(S, 0));

    first_present.clear();
    first_present.resize(N, std::vector<unsigned>(P, std::numeric_limits<unsigned>::max()));

    nodes_per_proc_and_step.clear();
    nodes_per_proc_and_step.resize(P, std::vector<std::vector<vertex_idx> >(S));
    
    needed_on_proc.clear();
    needed_on_proc.resize(N, std::vector<std::multiset<unsigned> >(P, {S}));
    
    max_work.clear();
    max_comm.clear();
    max_work.resize(S, 0);
    max_comm.resize(S, 0);

    comm_steps.clear();
    comm_steps.resize(S);                              

    for (vertex_idx node = 0; node < N; ++node) {
      for (const std::pair<unsigned, unsigned> &proc_and_step : schedule.assignments(node)) {
        const unsigned &proc = proc_and_step.first;
        const unsigned &step = proc_and_step.second;
        nodes_per_proc_and_step[proc][step].push_back(node);
        work_cost[proc][step] += G.VertexWorkWeight(node);
        first_present[node][proc] = std::min(first_present[node][proc], step);
        for (vertex_idx pred : G.Parents(node)) {
          needed_on_proc[pred][proc].insert(step);
        }
      }
    }
    for (const std::pair<KeyTriple, unsigned> item : schedule.GetCommunicationSchedule()) {
      const vertex_idx &node = std::get<0>(item.first);
      const unsigned &from_proc = std::get<1>(item.first);
      const unsigned &to_proc = std::get<2>(item.first);
      const unsigned &step = item.second;
      send_cost[from_proc][step] += G.VertexCommWeight(node)
                                    * schedule.GetInstance().GetArchitecture().CommunicationCosts(from_proc, to_proc);
      rec_cost[to_proc][step] += G.vertex_comm_weight(node)
                                    * schedule.GetInstance().GetArchitecture().CommunicationCosts(from_proc, to_proc);

      comm_steps[step].emplace(item.first);
      needed_on_proc[node][from_proc].insert(step);
      first_present[node][to_proc] = std::min(first_present[node][to_proc], step + 1);
    }
    for (unsigned step = 0; step < S; ++step) {
      for (unsigned proc = 0; proc < P; ++proc) {
        max_work[step] =std::max(max_work[step], work_cost[proc][step]);
        max_comm[step] =std::max(max_comm[step], send_cost[proc][step]);
        max_comm[step] =std::max(max_comm[step], rec_cost[proc][step]);
      }
    }
}

template <typename GraphT>
void GreedyRecomputer<GraphT>::AddRecomputeStep(BspScheduleRecomp<GraphT> &schedule, const vertex_idx node, const unsigned proc, const unsigned step)
{
  schedule.assignments(node).emplace_back(proc, step);
  nodes_per_proc_and_step[proc][step].push_back(node);
  work_cost[proc][step] += schedule.getInstance().GetComputationalDag().VertexWorkWeight(node);
  first_present[node][proc] = std::min(first_present[node][proc], step);
  for (const vertex_idx &pred : schedule.GetInstance().GetComputationalDag().parents(node)) {
    needed_on_proc[pred][proc].insert(step);
  }
}

template <typename GraphT>
void GreedyRecomputer<GraphT>::AddCommStep(const BspScheduleRecomp<GraphT> &schedule, const KeyTriple &new_comm, const unsigned step)
{
  comm_steps[step].insert(new_comm);
  cost_type comm_cost = schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(new_comm))
                        * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(new_comm), std::get<2>(new_comm));
  send_cost[std::get<1>(new_comm)][step] += comm_cost;
  rec_cost[std::get<2>(new_comm)][step] += comm_cost;
  needed_on_proc[std::get<0>(new_comm)][std::get<1>(new_comm)].insert(step);
  unsigned &first_pres = first_present[std::get<0>(new_comm)][std::get<2>(new_comm)];
  if (first_pres > step + 1 && first_pres <= schedule.NumberOfSupersteps()) {
    auto itr = comm_steps[first_pres - 1].find(new_comm);
    if (itr != comm_steps[first_pres - 1].end()) {
      comm_steps[first_pres - 1].erase(itr);
      send_cost[std::get<1>(new_comm)][first_pres - 1] -= comm_cost;
      rec_cost[std::get<2>(new_comm)][first_pres - 1] -= comm_cost;
      needed_on_proc[std::get<0>(new_comm)][std::get<1>(new_comm)].erase(needed_on_proc[std::get<0>(new_comm)][std::get<1>(new_comm)].lower_bound(first_pres - 1));
    }
  }
  first_pres = std::min(first_pres, step + 1);
}

template <typename GraphT>
void GreedyRecomputer<GraphT>::RemoveCommStep(const BspScheduleRecomp<GraphT> &schedule, const KeyTriple &removed_comm, const unsigned step)
{
  needed_on_proc[std::get<0>(removed_comm)][std::get<1>(removed_comm)].erase(needed_on_proc[std::get<0>(removed_comm)][std::get<1>(removed_comm)].lower_bound(step));

  cost_type comm_cost = schedule.getInstance().GetComputationalDag().VertexCommWeight(std::get<0>(removed_comm))
                        * schedule.getInstance().GetArchitecture().CommunicationCosts(std::get<1>(removed_comm), std::get<2>(removed_comm));

  auto itr = comm_steps[step].find(removed_comm);
  if (itr != comm_steps[step].end()) {
    comm_steps[step].erase(itr);
  }
  send_cost[std::get<1>(removed_comm)][step] -= comm_cost;
  rec_cost[std::get<2>(removed_comm)][step] -= comm_cost;
}

} // namespace osp