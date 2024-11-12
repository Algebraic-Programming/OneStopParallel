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

    // memory limit
    unsigned memory_limit;

    // nodes evicted from cache in a given superstep's comm phase
    std::vector<std::vector<std::vector<unsigned> > > nodes_evicted_in_comm;

    // nodes sent down to processor p in superstep s
    std::vector<std::vector<std::vector<unsigned> > > nodes_sent_down;

    // nodes sent up from processor p in superstep s
    std::vector<std::vector<std::vector<unsigned> > > nodes_sent_up;

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
    BspMemSchedule() : instance(nullptr), number_of_supersteps(0), memory_limit(0) {}

    BspMemSchedule(const BspInstance &inst) : instance(&inst)
    {
      BspSchedule schedule(inst, std::vector<unsigned int>(inst.numberOfVertices(), 0), std::vector<unsigned int>(inst.numberOfVertices(), 0));
      memory_limit = minimumMemoryRequired(inst);
      ConvertFromBsp(schedule);
    }

    BspMemSchedule(const BspInstance &inst, const std::vector<unsigned>& processor_assignment_,
                  const std::vector<unsigned>& superstep_assignment_) : instance(&inst)
    {
      BspSchedule schedule(inst, processor_assignment_, superstep_assignment_);
      memory_limit = minimumMemoryRequired(inst);
      ConvertFromBsp(schedule);
    }

    BspMemSchedule(const BspInstance &inst, unsigned Mem_limit,
                   const std::vector<std::vector<std::vector<unsigned> > >& compute_steps,
                   const std::vector<std::vector<std::vector<std::vector<unsigned> > > >& nodes_evicted_after_compute,
                   const std::vector<std::vector<std::vector<unsigned> > >& nodes_sent_up_,
                   const std::vector<std::vector<std::vector<unsigned> > >& nodes_sent_down_,
                   const std::vector<std::vector<std::vector<unsigned> > >& nodes_evicted_in_comm_) :
                   instance(&inst), memory_limit(Mem_limit), number_of_supersteps(0),
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

    BspMemSchedule(const BspSchedule &schedule, unsigned Mem_limit, CACHE_EVICTION_STRATEGY evict_rule = LARGEST_ID)
    : instance(&schedule.getInstance()), memory_limit(Mem_limit) { ConvertFromBsp(schedule, evict_rule); }

    virtual ~BspMemSchedule() = default;

    // cost computation
    unsigned computeCost() const;
    unsigned computeAsynchronousCost() const;

    // remove unnecessary steps (e.g. from ILP solution)
    void cleanSchedule();

    // convert from unconstrained schedule
    void ConvertFromBsp(const BspSchedule &schedule, CACHE_EVICTION_STRATEGY evict_rule = LARGEST_ID);

    //auxiliary for conversion
    static std::vector<std::vector<std::vector<unsigned> > > computeTopOrdersDFS(const BspSchedule &schedule);
    void SplitSupersteps(const BspSchedule &schedule);
    void SetMemoryMovement(CACHE_EVICTION_STRATEGY evict_rule = LARGEST_ID);

    // other basic operations
    bool isValid();
    static unsigned minimumMemoryRequired(const BspInstance& instance);

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

};


