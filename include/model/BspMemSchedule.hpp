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

#include <list>
#include <map>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iostream>

#include "BspSchedule.hpp"


typedef std::tuple<unsigned int, unsigned int, unsigned int> KeyTriple;

/**
 * @class BspMemSchedule
 * @brief Represents a schedule for the Bulk Synchronous Parallel (BSP) model with mem constraints.
 *
 * The `BspSchedule` class is responsible for managing the assignment of nodes to processors and supersteps in the BSP
 * model. It stores information such as the number of supersteps, the assignment of nodes to processors and supersteps,
 * and the communication schedule.
 *
 * The class provides methods for setting and retrieving the assigned superstep and processor for a given node, as well
 * as methods for checking the validity of the communication schedule and computing the costs of the schedule. It also
 * provides methods for setting the assigned supersteps and processors based on external assignments, and for updating
 * the number of supersteps.
 *
 * The `BspSchedule` class is designed to work with a `BspInstance` object, which represents the instance of the BSP
 * problem being solved.
 *
 * @see BspInstance
 */
class BspMemSchedule {

  private:
    const BspInstance *instance;

    unsigned int number_of_supersteps;

    bool need_to_load_inputs = true;

    struct compute_step
    {
      unsigned node;
      std::vector<unsigned> nodes_evicted_after;

      compute_step() {}
      compute_step(unsigned node_) : node(node_) {}
      compute_step(unsigned node_, const std::vector<unsigned>& evicted_) : node(node_), nodes_evicted_after(evicted_) {}
    };

    // executed nodes in order in a computation phase, for processor p and superstep s
    std::vector<std::vector<std::vector<compute_step> > > compute_steps_for_proc_superstep;

    // nodes evicted from cache in a given superstep's comm phase
    std::vector<std::vector<std::vector<unsigned> > > nodes_evicted_in_comm;

    // nodes sent down to processor p in superstep s
    std::vector<std::vector<std::vector<unsigned> > > nodes_sent_down;

    // nodes sent up from processor p in superstep s
    std::vector<std::vector<std::vector<unsigned> > > nodes_sent_up;

    // set of nodes that need to have blue pebble at end, sinks by default, and
    // set of nodes on each processor that begin with red pebble, nothing by default
    // (TODO: maybe move to problem definition classes instead?)
    std::set<unsigned> needs_blue_at_end;
    std::vector<std::set<unsigned> > has_red_in_beginning;

    // nodes that are from a previous part of a larger DAG, handled differently in conversion
    std::set<unsigned> external_sources;

  public:

    enum CACHE_EVICTION_STRATEGY
    {
        FORESIGHT,
        LEAST_RECENTLY_USED,
        LARGEST_ID
    };

    /**
     * @brief Default constructor for the BspMemSchedule class.
     */
    BspMemSchedule() : instance(nullptr), number_of_supersteps(0) {}

    BspMemSchedule(const BspInstance &inst) : instance(&inst)
    {
      BspSchedule schedule(inst, std::vector<unsigned int>(inst.numberOfVertices(), 0), std::vector<unsigned int>(inst.numberOfVertices(), 0));
      ConvertFromBsp(schedule);
    }

    BspMemSchedule(const BspInstance &inst, const std::vector<unsigned>& processor_assignment_,
                  const std::vector<unsigned>& superstep_assignment_) : instance(&inst)
    {
      BspSchedule schedule(inst, processor_assignment_, superstep_assignment_);
      ConvertFromBsp(schedule);
    }

    BspMemSchedule(const BspInstance &inst,
                   const std::vector<std::vector<std::vector<unsigned> > >& compute_steps,
                   const std::vector<std::vector<std::vector<std::vector<unsigned> > > >& nodes_evicted_after_compute,
                   const std::vector<std::vector<std::vector<unsigned> > >& nodes_sent_up_,
                   const std::vector<std::vector<std::vector<unsigned> > >& nodes_sent_down_,
                   const std::vector<std::vector<std::vector<unsigned> > >& nodes_evicted_in_comm_,
                   const std::set<unsigned>& needs_blue_at_end_ = std::set<unsigned>(),
                   const std::vector<std::set<unsigned> >& has_red_in_beginning_ = std::vector<std::set<unsigned> >(),
                   const bool need_to_load_inputs_ = false) :
                   instance(&inst), number_of_supersteps(0),
                   needs_blue_at_end(needs_blue_at_end_), need_to_load_inputs (need_to_load_inputs_),
                   has_red_in_beginning(has_red_in_beginning_),
                   nodes_sent_up(nodes_sent_up_), nodes_sent_down(nodes_sent_down_), nodes_evicted_in_comm(nodes_evicted_in_comm_)

    {
      compute_steps_for_proc_superstep.resize(compute_steps.size(), std::vector<std::vector<compute_step> >(compute_steps[0].size()));
      for(unsigned proc = 0; proc < compute_steps.size(); ++proc)
      {
        number_of_supersteps = std::max(number_of_supersteps, (unsigned)compute_steps[proc].size());
        for(unsigned supstep = 0; supstep < compute_steps[proc].size(); ++supstep)
          for(unsigned step_index = 0; step_index < compute_steps[proc][supstep].size(); ++step_index)
            compute_steps_for_proc_superstep[proc][supstep].emplace_back(compute_steps[proc][supstep][step_index],
                                                                          nodes_evicted_after_compute[proc][supstep][step_index]);
      }
    }

    BspMemSchedule(const BspSchedule &schedule, CACHE_EVICTION_STRATEGY evict_rule = LARGEST_ID)
    : instance(&schedule.getInstance()) { ConvertFromBsp(schedule, evict_rule); }

    virtual ~BspMemSchedule() = default;

    // cost computation
    unsigned computeCost() const;
    unsigned computeAsynchronousCost() const;

    // remove unnecessary steps (e.g. from ILP solution)
    void cleanSchedule();

    // convert from unconstrained schedule
    void ConvertFromBsp(const BspSchedule &schedule, CACHE_EVICTION_STRATEGY evict_rule = LARGEST_ID);

    //auxiliary for conversion
    std::vector<std::vector<std::vector<unsigned> > > computeTopOrdersDFS(const BspSchedule &schedule) const;
    static bool hasValidSolution(const BspInstance &instance, const std::set<unsigned>& external_sources = std::set<unsigned>());
    void SplitSupersteps(const BspSchedule &schedule);
    void SetMemoryMovement(CACHE_EVICTION_STRATEGY evict_rule = LARGEST_ID);

    // other basic operations
    bool isValid() const;
    static std::vector<unsigned> minimumMemoryRequiredPerNodeType(const BspInstance& instance, const std::set<unsigned>& external_sources = std::set<unsigned>());

    // convert to BSP (ignores vertical I/O and recomputation)
    BspSchedule ConvertToBsp() const;

    /**
     * @brief Returns a reference to the BspInstance for the schedule.
     *
     * @return A reference to the BspInstance for the schedule.
     */
    const BspInstance &getInstance() const { return *instance; }

  
    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    unsigned numberOfSupersteps() const { return number_of_supersteps; }

    void updateNumberOfSupersteps(unsigned new_number_of_supersteps);

    inline bool needsToLoadInputs() const { return need_to_load_inputs; }
    inline void setNeedToLoadInputs(const bool load_inputs_) { need_to_load_inputs = load_inputs_;}

    void getDataForMultiprocessorPebbling(std::vector<std::vector<std::vector<unsigned> > >& computeSteps,
                                          std::vector<std::vector<std::vector<unsigned> > >& sendUpSteps,
                                          std::vector<std::vector<std::vector<unsigned> > >& sendDownSteps,
                                          std::vector<std::vector<std::vector<unsigned> > >& nodesEvictedAfterStep) const;


    // utility for partial ILPs
    std::vector<std::set<unsigned> > getMemContentAtEnd() const;
    void removeEvictStepsFromEnd();

    void CreateFromPartialPebblings(const std::vector<BspMemSchedule>& pebblings,
                                    const std::vector<std::set<unsigned> >& processors_to_parts,
                                    const std::vector<std::map<unsigned, unsigned> >& original_node_id,
                                    const std::vector<std::map<unsigned, unsigned> >& original_proc_id,
                                    const std::vector<std::vector<std::set<unsigned> > >& has_reds_in_beginning);

    const std::vector<compute_step>& GetComputeStepsForProcSuperstep(unsigned proc, unsigned supstep) const {return compute_steps_for_proc_superstep[proc][supstep];}
    const std::vector<unsigned>& GetNodesEvictedInComm(unsigned proc, unsigned supstep) const {return nodes_evicted_in_comm[proc][supstep];}
    const std::vector<unsigned>& GetNodesSentDown(unsigned proc, unsigned supstep) const {return nodes_sent_down[proc][supstep];}
    const std::vector<unsigned>& GetNodesSentUp(unsigned proc, unsigned supstep) const {return nodes_sent_up[proc][supstep];}

    void SetNeedsBlueAtEnd(const std::set<unsigned>& nodes_) {needs_blue_at_end = nodes_;}
    void SetExternalSources(const std::set<unsigned>& nodes_) {external_sources = nodes_;}
    void SetHasRedInBeginning(const std::vector<std::set<unsigned> >& nodes_) {has_red_in_beginning = nodes_;}

};


