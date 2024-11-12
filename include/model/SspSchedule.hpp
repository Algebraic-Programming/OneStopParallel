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
#include <limits.h>

#include "model/IBspSchedule.hpp"
#include "model/BspSchedule.hpp"


typedef std::tuple<unsigned int, unsigned int, unsigned int> KeyTriple;

/**
 * @class SspSchedule
 * @brief Represents a schedule for the Stale Synchronous Parallel (SSP) model.
 *
 * The `SspSchedule` class is responsible for managing the assignment of nodes to processors and supersteps in the SSP
 * model. It stores information such as the number of supersteps, the assignment of nodes to processors and supersteps,
 * and the communication schedule.
 *
 * The class provides methods for setting and retrieving the assigned superstep and processor for a given node, as well
 * as methods for checking the validity of the communication schedule and computing the costs of the schedule. It also
 * provides methods for setting the assigned supersteps and processors based on external assignments, and for updating
 * the number of supersteps.
 *
 * The `SspSchedule` class is designed to work with a `BspInstance` object, which represents the instance of the SSP
 * problem being solved.
 *
 * @see IBspSchedule
 * @see BspSchedule
 * @see BspInstance
 */
class SspSchedule : public BspSchedule {

  private:
    unsigned staleness;

  public:

    /**
     * @brief Default constructor for the SspSchedule class.
     */
    SspSchedule() : BspSchedule(), staleness(1) {}

    /**
     * @brief Constructs a SspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    SspSchedule(const BspInstance &inst) : BspSchedule(inst), staleness(1) {}

    /**
     * @brief Constructs a SspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    SspSchedule(const BspInstance &inst, unsigned staleness_) : BspSchedule(inst), staleness(staleness_) {}

    /**
     * @brief Constructs a SspSchedule object with the specified BspInstance, processor assignment, and superstep assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */ 
    SspSchedule(const BspInstance &inst, const std::vector<unsigned> &processor_assignment_,
                const std::vector<unsigned> &superstep_assignment_) : BspSchedule(inst, processor_assignment_, superstep_assignment_), staleness(1) {
        updateNumberOfSupersteps();
  
    }

    /**
     * @brief Constructs a SspSchedule object with the specified BspInstance, processor assignment, and superstep assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */ 
    SspSchedule(const BspInstance &inst, const std::vector<unsigned> &processor_assignment_,
                const std::vector<unsigned> &superstep_assignment_, unsigned staleness_) : BspSchedule(inst, processor_assignment_, superstep_assignment_), staleness(staleness_) {
        updateNumberOfSupersteps();
  
    }

    SspSchedule(const BspSchedule bspsched) : BspSchedule(bspsched), staleness(1) {
      updateNumberOfSupersteps();
    }

    SspSchedule(const BspSchedule bspsched, unsigned staleness_) : BspSchedule(bspsched), staleness(staleness_) {
      updateNumberOfSupersteps();
    }
    
    virtual ~SspSchedule() = default;

    /**
     * @brief Get the Staleness of the SspSchedule
     * 
     * @return unsigned 
     */
    inline unsigned getStaleness() const { return staleness; }

    /**
     * @brief Gets the maximimum staleness of the SspSchedule
     * 
     */
    unsigned getMaxStaleness() const;

    /**
     * @brief Sets the staleness of the SspSchedule
     * 
     */
    void setStaleness(unsigned stale) { staleness = stale; };

    /**
     * @brief Sets the maximimum staleness of the SspSchedule
     * 
     */
    void setMaxStaleness() { staleness = getMaxStaleness(); };


    unsigned computeWorkCosts() const;

    unsigned computeSspCosts(unsigned stale) const;
    unsigned computeSspCosts() const;

    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u and v are assigned to different processors, the
     * superstep assigned to node u is less than the superstep assigned to node v.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    bool satisfiesPrecedenceConstraints(unsigned stale) const;

    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u and v are assigned to different processors, the
     * superstep assigned to node u is less than the superstep assigned to node v.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    bool satisfiesPrecedenceConstraints() const override;

};


