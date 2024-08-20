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

#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <vector>

#include "BspInstance.hpp"
#include "BspSchedule.hpp"

class BspScheduleRecomp {

  private:
    const BspInstance *instance;

    unsigned int number_of_supersteps;

   

  public:

    std::vector<std::vector<unsigned>> node_processor_assignment;
    std::vector<std::vector<unsigned>> node_superstep_assignment;

    std::map<KeyTriple, unsigned> commSchedule;

    BspScheduleRecomp() = default;

    BspScheduleRecomp(const BspInstance &inst, unsigned number_of_supersteps_)
        : instance(&inst), number_of_supersteps(number_of_supersteps_), node_processor_assignment(inst.numberOfVertices(), std::vector<unsigned>()),
          node_superstep_assignment(inst.numberOfVertices(), std::vector<unsigned>()) {}

    virtual ~BspScheduleRecomp() = default;

    const BspInstance &getInstance() const { return *instance; }

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    unsigned numberOfSupersteps() const { return number_of_supersteps; }
    void setNumberOfSupersteps(unsigned number_of_supersteps_) { number_of_supersteps = number_of_supersteps_; }

    std::vector<unsigned>& assignedProcessors(unsigned node) {
        return node_processor_assignment[node];
    }

    const std::vector<unsigned>& assignedProcessors(unsigned node) const {
        return node_processor_assignment[node];
    }

    std::vector<unsigned>& assignedSupersteps(unsigned node) {
        return node_superstep_assignment[node];
    }

    const std::vector<unsigned>& assignedSupersteps(unsigned node) const {
        return node_superstep_assignment[node];
    }


    /**
     * @brief Sets the communication schedule for the schedule.
     *
     * @param cs The communication schedule to set.
     */
    void setCommunicationSchedule(const std::map<KeyTriple, unsigned int> &cs);

    /**
     * @brief Adds an entry to the communication schedule.
     *
     * @param key The key for the communication schedule entry.
     * @param step The superstep for the communication schedule entry.
     */
    void addCommunicationScheduleEntry(KeyTriple key, unsigned step);

    /**
     * @brief Adds an entry to the communication schedule.
     *
     * @param node The node resp. its data which is sent.
     * @param from_proc The processor from which the data is sent.
     * @param to_proc The processor to which the data is sent.
     * @param step The superstep in which the data is sent.
     */
    void addCommunicationScheduleEntry(unsigned node, unsigned from_proc, unsigned to_proc, unsigned step);

    /**
     * @brief Returns the communication schedule for the schedule.
     *
     * @return The communication schedule for the schedule.
     */
    const std::map<KeyTriple, unsigned int> &getCommunicationSchedule() const { return commSchedule; }

    std::map<KeyTriple, unsigned int> &getCommunicationSchedule() { return commSchedule; }

    bool checkCommScheduleValidity(const std::map<KeyTriple, unsigned int> &cs) const;

    bool hasValidCommSchedule() const;

    unsigned computeWorkCosts() const;

    unsigned computeCosts() const;

    unsigned computeCostsBufferedSending() const;

    double computeCostsTotalCommunication() const;

    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u
     * and v are assigned to different processors, the superstep assigned to node u is less than the superstep assigned
     * to node v.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    bool satisfiesPrecedenceConstraints() const;
   
   
    bool satisfiesMemoryConstraints() const;

    unsigned total_node_assignments() const {
        unsigned total = 0;
        for (unsigned i = 0; i < node_processor_assignment.size(); i++) {
            total += node_processor_assignment[i].size();
        }
        return total;
    }
   

};
