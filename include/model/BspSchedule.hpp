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

#include "IBspSchedule.hpp"


typedef std::tuple<unsigned int, unsigned int, unsigned int> KeyTriple;

/**
 * @class BspSchedule
 * @brief Represents a schedule for the Bulk Synchronous Parallel (BSP) model.
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
 * @see IBspSchedule
 * @see BspInstance
 */
class BspSchedule : public IBspSchedule {

  private:
    const BspInstance *instance;

    unsigned int number_of_supersteps;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<unsigned> node_to_superstep_assignment;

    // contains entries: (vertex, from_proc, to_proc ) : step
    std::map<KeyTriple, unsigned> commSchedule;

  public:

    /**
     * @brief Default constructor for the BspSchedule class.
     */
    BspSchedule() : instance(nullptr), number_of_supersteps(0) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    BspSchedule(const BspInstance &inst)
        : instance(&inst), number_of_supersteps(1),
          node_to_processor_assignment(std::vector<unsigned int>(inst.numberOfVertices(), 0)),
          node_to_superstep_assignment(std::vector<unsigned int>(inst.numberOfVertices(), 0)) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */ 
    BspSchedule(const BspInstance &inst, const std::vector<unsigned> &processor_assignment_,
                const std::vector<unsigned> &superstep_assignment_)
        : instance(&inst), node_to_processor_assignment(processor_assignment_),
          node_to_superstep_assignment(superstep_assignment_) {
        updateNumberOfSupersteps();
    }

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, superstep assignment,
     * and communication schedule.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     * @param comm_ The communication schedule for the nodes.
     */
    BspSchedule(const BspInstance &inst, const std::vector<unsigned int>& processor_assignment_,
                const std::vector<unsigned int>& superstep_assignment_, const std::map<KeyTriple, unsigned int>& comm_)
        : instance(&inst), node_to_processor_assignment(processor_assignment_),
          node_to_superstep_assignment(superstep_assignment_), commSchedule(comm_) { updateNumberOfSupersteps(); }


    
    virtual ~BspSchedule() = default;

    /**
     * @brief Returns a reference to the BspInstance for the schedule.
     *
     * @return A reference to the BspInstance for the schedule.
     */
    const BspInstance &getInstance() const override { return *instance; }

  

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    unsigned numberOfSupersteps() const override { return number_of_supersteps; }

    /**
     * @brief Returns the number of processors in the schedule.
     *
     * @return The number of processors in the schedule.
     */
    void updateNumberOfSupersteps();

    /**
     * @brief Returns the superstep assigned to the specified node.
     *
     * @param node The node for which to return the assigned superstep.
     * @return The superstep assigned to the specified node.
     */
    inline unsigned assignedSuperstep(unsigned node) const override { return node_to_superstep_assignment[node]; }
    
    /**
     * @brief Returns the processor assigned to the specified node.
     *
     * @param node The node for which to return the assigned processor.
     * @return The processor assigned to the specified node.
     */
    inline unsigned assignedProcessor(unsigned node) const override { return node_to_processor_assignment[node]; }

    /**
     * @brief Returns the superstep assignment for the schedule.
     *
     * @return The superstep assignment for the schedule.
     */
    inline const std::vector<unsigned> &assignedSupersteps() const { return node_to_superstep_assignment; }
    
    /**
     * @brief Returns the processor assignment for the schedule.
     *
     * @return The processor assignment for the schedule.
     */
    inline const std::vector<unsigned> &assignedProcessors() const { return node_to_processor_assignment; }

    /**
     * @brief Sets the superstep assigned to the specified node.
     *
     * @param node The node for which to set the assigned superstep.
     * @param superstep The superstep to assign to the node.
     */
    void setAssignedSuperstep(unsigned node, unsigned superstep) override;
    
    /**
     * @brief Sets the processor assigned to the specified node.
     *
     * @param node The node for which to set the assigned processor.
     * @param processor The processor to assign to the node.
     */
    void setAssignedProcessor(unsigned node, unsigned processor) override;

    /**
     * @brief Sets the superstep assignment for the schedule.
     *
     * @param vec The superstep assignment to set.
     */
    void setAssignedSupersteps(const std::vector<unsigned int> &vec);


    /**
     * @brief Sets the processor assignment for the schedule.
     *
     * @param vec The processor assignment to set.
     */
    void setAssignedProcessors(const std::vector<unsigned int> &vec);


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

    unsigned computeBaseCommCost() const;

    unsigned computeCostsBufferedSending() const;

    unsigned computeBaseCommCostsBufferedSending() const;
  
    double computeCostsTotalCommunication() const;

    double computeBaseCommCostsTotalCommunication() const;

    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u and v are assigned to different processors, the
     * superstep assigned to node u is less than the superstep assigned to node v.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    bool satisfiesPrecedenceConstraints() const;
    bool satisfiesNodeTypeConstraints() const;
    bool satisfiesMemoryConstraints() const;

    void setAutoCommunicationSchedule();
    void setImprovedLazyCommunicationSchedule();
    void setLazyCommunicationSchedule();
    void setEagerCommunicationSchedule();

    std::vector<unsigned int> getAssignedNodeVector(unsigned int processor) const;
    std::vector<unsigned int> getAssignedNodeVector(unsigned int processor, unsigned int superstep) const;

    void setNumberOfSupersteps(unsigned int number_of_supersteps_) { number_of_supersteps = number_of_supersteps_; }

    unsigned num_assigned_nodes(unsigned processor) const;
    std::vector<unsigned> num_assigned_nodes_per_processor() const;

    std::vector<std::vector<unsigned>> num_assigned_nodes_per_superstep_processor() const;

};


