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

#include "model/BspInstance.hpp"


/**
 * @class DAGPartition
 * @brief Represents a partition of the BspInstance.
 *
 * The `DAGPartition` class is responsible for managing the assignment of nodes to processors.
 * It stores information such as the assignment of nodes to processors.
 *
 * The class provides methods for setting and retrieving the assigned processor for a given node, as well
 * as methods for checking the validity of the partition and computing the costs of the partition. It also
 * provides methods for setting the assigned processors based on external assignments.
 *
 * The `DAGPartition` class is designed to work with a `BspInstance` object, which represents the instance of the BSP
 * problem being solved.
 *
 * @see BspInstance
 */
class DAGPartition {

  private:
    const BspInstance *instance;

    std::vector<unsigned> node_to_processor_assignment;

  public:

    /**
     * @brief Default constructor for the DAGPartition class.
     */
    DAGPartition() : instance(nullptr) {}

    /**
     * @brief Constructs a DAGPartition object with the specified BspInstance.
     *
     * @param inst The BspInstance for the partition.
     */
    DAGPartition(const BspInstance &inst)
        : instance(&inst),
          node_to_processor_assignment(std::vector<unsigned int>(inst.numberOfVertices(), 0)) { }

    /**
     * @brief Constructs a DAGPartition object with the specified BspInstance, processor assignment.
     *
     * @param inst The BspInstance for the partition.
     * @param processor_assignment_ The processor assignment for the nodes.
     */ 
    DAGPartition(const BspInstance &inst, std::vector<unsigned> processor_assignment_)
        : instance(&inst), node_to_processor_assignment(processor_assignment_) { }

    
    virtual ~DAGPartition() = default;

    /**
     * @brief Returns a reference to the BspInstance for the partition.
     *
     * @return A reference to the BspInstance for the partition.
     */
    const BspInstance &getInstance() const { return *instance; }
    
    /**
     * @brief Returns the processor assigned to the specified node.
     *
     * @param node The node for which to return the assigned processor.
     * @return The processor assigned to the specified node.
     */
    inline unsigned assignedProcessor(unsigned node) const { return node_to_processor_assignment[node]; }
    
    /**
     * @brief Returns the processor assignment for the partition.
     *
     * @return The processor assignment for the partition.
     */
    inline const std::vector<unsigned> &assignedProcessors() const { return node_to_processor_assignment; }
    
    /**
     * @brief Sets the processor assigned to the specified node.
     *
     * @param node The node for which to set the assigned processor.
     * @param processor The processor to assign to the node.
     */
    void setAssignedProcessor(unsigned node, unsigned processor);


    /**
     * @brief Sets the processor assignment for the partition.
     *
     * @param vec The processor assignment to set.
     */
    void setAssignedProcessors(const std::vector<unsigned int> &vec);
    
    /**
     * @brief Computes the work loads of all processors
     */
    std::vector<unsigned> computeAllWorkCosts() const;

    /**
     * @brief Computes the work load of a given processor
     * 
     * @param processor processor
     */
    unsigned computeWorkCosts(unsigned processor) const;

    /**
     * @brief Computes the maximum work load of a processor
     */
    unsigned computeMaxWorkCosts() const;

    /**
     * @brief Computes MaxWorkLoad / AvgWorkLoad 
     */
    float computeWorkImbalance() const;

    /**
     * @brief Computes the memory load of all processors
     */
    std::vector<unsigned> computeAllMemoryCosts() const;

    /**
     * @brief Computes the memory load of a given processor
     * 
     * @param processor processor
     */
    unsigned computeMemoryCosts(unsigned processor) const;

    /**
     * @brief Computes the maximum memory load
     */
    unsigned computeMaxMemoryCosts() const;

    /**
     * @brief Checks whether the current partition satisfies all memory constraints
     */
    bool satisfiesMemoryConstraints() const;

    /**
     * @brief Returns a vector with all nodes allocated to a given processor
     * 
     * @param processor processor 
     */
    std::vector<unsigned int> getAssignedNodeVector(unsigned int processor) const;

    /**
     * @brief Returns the number of nodes allocated to a given processor
     * 
     * @param processor processor 
     */
    unsigned num_assigned_nodes(unsigned processor) const;

    /**
     * @brief Returns the number of nodes allocated to all processors 
     */
    std::vector<unsigned> num_assigned_nodes() const;

    /**
     * @brief Computes all to all interprocessor communication costs
     */
    std::vector<std::vector<unsigned>> computeAlltoAllCommunication() const;

    /**
     * @brief Computes the total amount of communication
     */
    unsigned computeTotalCommunication() const;

    /**
     * @brief Computes the relative amount of communication
     */
    float computeCommunicationRatio() const;

    /**
     * @brief Computes the number of cut edges
     */
    unsigned computeCutEdges() const;

    /**
     * @brief Computes the relative number of cut edges
     */
    float computeCutEdgesRatio() const;

};


