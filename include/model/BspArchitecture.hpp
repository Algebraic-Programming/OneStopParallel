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
#include <numeric>
#include <stdexcept>
#include <vector>
#include <iomanip>
#include <iostream>

enum MEMORY_CONSTRAINT_TYPE { NONE, LOCAL, GLOBAL, PERSISTENT_AND_TRANSIENT };

/**
 * @class BspArchitecture
 * @brief Represents the architecture of a BSP (Bulk Synchronous Parallel) system.
 *
 * The BspArchitecture class stores information about the number of processors, communication costs,
 * synchronization costs, and send costs between processors in a BSP system. It provides methods to
 * set and retrieve these values.
 */
class BspArchitecture {

  private:
    unsigned number_processors;
    unsigned number_of_processor_types;
    unsigned communication_costs;
    unsigned synchronisation_costs;

    std::vector<unsigned> memory_bound;

    bool isNuma;

    std::vector<unsigned> processor_type;

    std::vector<std::vector<unsigned int>> send_costs;

    MEMORY_CONSTRAINT_TYPE memory_const_type = NONE;

    bool are_send_cost_numa();

  public:
    BspArchitecture()
        : number_processors(2), number_of_processor_types(1), communication_costs(1), synchronisation_costs(2),
          memory_bound(std::vector<unsigned>(number_processors, 100)), 
          isNuma(false),
          processor_type(std::vector<unsigned int>(number_processors, 0)),
          send_costs(std::vector<std::vector<unsigned int>>(number_processors,
                                                            std::vector<unsigned int>(number_processors, 1))) {
        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }
    }

    /**
     * @brief Constructs a BspArchitecture object with the specified number of processors, communication cost, and
     * synchronization cost.
     *
     * @param processors The number of processors in the architecture.
     * @param comm_cost The communication cost between processors.
     * @param synch_cost The synchronization cost between processors.
     */
    BspArchitecture(unsigned processors, unsigned comm_cost, unsigned synch_cost, unsigned memory_bound_ = 100) 
        : number_processors(processors), number_of_processor_types(1), communication_costs(comm_cost),
          synchronisation_costs(synch_cost), 
          memory_bound(std::vector<unsigned>(number_processors, memory_bound_)),
          isNuma(false), 
          processor_type(std::vector<unsigned int>(number_processors, 0)),
          send_costs(std::vector<std::vector<unsigned int>>(number_processors,
                                                            std::vector<unsigned int>(number_processors, 1))) {

        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }
    }

    /**
     * @brief Constructs a BspArchitecture object with the specified number of processors, communication cost, and
     * synchronization cost.
     *
     * @param processors The number of processors in the architecture.
     * @param comm_cost The communication cost between processors.
     * @param synch_cost The synchronization cost between processors.
     */
    BspArchitecture(unsigned int processors, unsigned int comm_cost, unsigned int synch_cost,
                    std::vector<std::vector<unsigned>> send_costs_)
        : number_processors(processors), number_of_processor_types(1), communication_costs(comm_cost),
          synchronisation_costs(synch_cost), 
          memory_bound(std::vector<unsigned>(number_processors, 100)),          
          processor_type(std::vector<unsigned int>(number_processors, 0)),
          send_costs(send_costs_) 
 {

        if (number_processors != send_costs.size()) {
            throw std::invalid_argument("send_costs_ needs to be a processors x processors matrix.\n");
        }
        if (std::any_of(send_costs.begin(), send_costs.end(),
                        [processors](const auto &thing) { return thing.size() != processors; })) {
            throw std::invalid_argument("send_costs_ needs to be a processors x processors matrix.\n");
        }
        
        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }

        isNuma = are_send_cost_numa();
    }

    /**
     * @brief Constructs a BspArchitecture object with the specified number of processors, communication cost, and
     * synchronization cost.
     *
     * @param processors The number of processors in the architecture.
     * @param comm_cost The communication cost between processors.
     * @param synch_cost The synchronization cost between processors.
     */
    BspArchitecture(unsigned int processors, unsigned int comm_cost, unsigned int synch_cost, unsigned memory_bound_,
                    std::vector<std::vector<unsigned>> send_costs_)
        : number_processors(processors), number_of_processor_types(1), communication_costs(comm_cost),
          synchronisation_costs(synch_cost), 
          memory_bound(std::vector<unsigned>(number_processors, memory_bound_)),
          processor_type(std::vector<unsigned int>(number_processors, 0)),
          send_costs(send_costs_) {

        if (number_processors != send_costs.size()) {
            throw std::invalid_argument("send_costs_ needs to be a processors x processors matrix.\n");
        }
        if (std::any_of(send_costs.begin(), send_costs.end(),
                        [processors](const auto &thing) { return thing.size() != processors; })) {
            throw std::invalid_argument("send_costs_ needs to be a processors x processors matrix.\n");
        }
        
        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }

        isNuma = are_send_cost_numa();
    }

    /**
     * Sets the uniform send cost for each pair of processors in the BSP architecture.
     * The send cost is set to 0 if the processors are the same, and 1 otherwise.
     * This function assumes that the number of processors has already been set.
     */
    void SetUniformSendCost();

    /**
     * @brief Sets the exponential send cost for the BspArchitecture.
     *
     * This function calculates and sets the exponential send cost for each pair of processors in the BspArchitecture.
     * The send cost is determined based on the base value and the position of the processors in the architecture.
     *
     * @param base The base value used to calculate the send cost.
     */
    void SetExpSendCost(unsigned int base);

    /**
     * @brief Computes the average communication cost of the BspArchitecture.
     *
     * This function computes the average communication cost of the BspArchitecture object.
     * The average communication cost is calculated as the sum of the send costs between processors divided by the
     * number of processors.
     *
     * @return The average communication cost as an unsigned integer.
     */
    unsigned int computeCommAverage() const;

    /**
     * Sets the send costs for the BspArchitecture.
     *
     * @param vec A 2D vector representing the send costs between processors.
     *            The size of the vector must be equal to the number of processors.
     *            Each inner vector must also have a size equal to the number of processors.
     * @throws std::invalid_argument if the size of the vector or inner vectors is invalid.
     */
    void setSendCosts(const std::vector<std::vector<unsigned int>> &vec);

    /**
     * Sets the send costs between two processors.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     * @param cost The cost of sending data between the processors.
     *
     * @remarks If the two processors are the same, the send cost is not set.
     *          If the cost is not equal to 1, the architecture is considered NUMA.
     */
    void setSendCosts(unsigned p1, unsigned p2, unsigned cost) {

        if (p1 < 0 || p1 >= number_processors || p2 < 0 || p2 > number_processors)
            throw std::invalid_argument("Invalid Argument");

        if (p1 != p2) {
            send_costs[p1][p2] = cost;

            isNuma = are_send_cost_numa();
        }
    }

    /**
     * Sets the memory bound for all processors of the BspArchitecture.
     *
     * @param memory_bound_ The new memory bound for all processors.
     */
    inline void setMemoryBound(unsigned memory_bound_) {
        memory_bound = std::vector<unsigned>(number_processors, memory_bound_);
    }

    inline void setMemoryBound(const std::vector<unsigned> &memory_bound_) { memory_bound = memory_bound_; }

    inline void setMemoryBound(unsigned memory_bound_, unsigned proc) { memory_bound[proc] = memory_bound_; }

    /**
     * @brief Sets the synchronization costs for the BspArchitecture.
     *
     * This function sets the synchronization costs for the BspArchitecture object.
     * The synchronization costs represent the costs of establishing communication between processors.
     *
     * @param synch_cost The synchronization costs to be set.
     */
    inline void setSynchronisationCosts(unsigned synch_cost) { synchronisation_costs = synch_cost; }

    /**
     * @brief Sets the communication costs for the BspArchitecture.
     *
     * This function sets the communication costs for the BspArchitecture object.
     * The communication costs represent the costs of sending messages between processors.
     *
     * @param comm_cost The communication costs to be set.
     */
    inline void setCommunicationCosts(unsigned comm_cost) { communication_costs = comm_cost; }

    /**
     * @brief Sets the number of processors in the BSP architecture.
     *
     * This function sets the number of processors in the BSP architecture and sets the send costs between processors
     * to 1. The send_costs matrix represents the costs of sending messages between processors. The diagonal elements of
     * the matrix are set to 0, indicating that there is no cost to send a message from a processor to itself.
     *
     * @param num_proc The number of processors in the BSP architecture.
     */
    void setNumberOfProcessors(unsigned num_proc);

    /**
     * @brief Sets the number of processors and their types in the BSP architecture.
     *
     * This function sets the number of processors in the BSP architecture and sets the send costs between processors
     * to 1. The send_costs matrix represents the costs of sending messages between processors. The diagonal elements of
     * the matrix are set to 0, indicating that there is no cost to send a message from a processor to itself.
     *
     * @param processor_types_ The type of the respective processors.
     */
    void setProcessorsWithTypes(const std::vector<unsigned> &processor_types_);

    /**
     * Returns whether the architecture is NUMA.
     *
     * @return True if the architecture is NUMA, false otherwise.
     */
    inline bool isNumaArchitecture() const { return isNuma; }


    void set_processors_consequ_types(const std::vector<unsigned> &processor_type_count_, const std::vector<unsigned> &processor_type_memory_);

    /**
     * Returns the memory bound of the BspArchitecture.
     *
     * @return The memory bound as an unsigned integer.
     */
    inline const std::vector<unsigned> &memoryBound() const { return memory_bound; }

    inline unsigned memoryBound(unsigned proc) const { return memory_bound[proc]; }

    unsigned minMemoryBound() const { return *(std::min_element(memory_bound.begin(), memory_bound.end())); }
    unsigned maxMemoryBound() const { return *(std::max_element(memory_bound.begin(), memory_bound.end())); }
    unsigned sumMemoryBound() const { return std::accumulate(memory_bound.begin(), memory_bound.end(), 0); }

    unsigned maxMemoryBoundProcType(unsigned procType) const;
    
    /**
     * Returns the number of processors in the architecture.
     *
     * @return The number of processors.
     */
    inline unsigned numberOfProcessors() const { return number_processors; }

    /**
     * Returns the communication costs of the BSP architecture.
     *
     * @return The communication costs as an unsigned integer.
     */
    inline unsigned communicationCosts() const { return communication_costs; }

    /**
     * Returns the synchronization costs of the BspArchitecture.
     *
     * @return The synchronization costs as an unsigned integer.
     */
    inline unsigned synchronisationCosts() const { return synchronisation_costs; }

    /**
     * Returns a copy of the send costs matrix.
     *
     * @return A copy of the send costs matrix.
     */
    inline std::vector<std::vector<unsigned int>> sendCostMatrixCopy() const { return send_costs; }

    /**
     * Returns a reference to the send costs matrix.
     *
     * @return A reference to the send costs matrix.
     */
    inline const std::vector<std::vector<unsigned int>> &sendCostMatrix() const { return send_costs; }

    // the type indeces of the processor (e.g. CPU, vector/tensor core)
    inline const std::vector<unsigned int> &processorTypes() const { return processor_type; }
    
    /**
     * Returns the communication costs between two processors. The communication costs are the send costs multiplied by
     * the communication costs.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     *
     * @return The send costs between the two processors.
     */
    inline unsigned communicationCosts(unsigned p1, unsigned p2) const {
        return communication_costs * send_costs[p1][p2];
    }

    /**
     * Returns the send costs between two processors.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     *
     * @return The send costs between the two processors.
     */
    inline unsigned sendCosts(unsigned p1, unsigned p2) const { return send_costs[p1][p2]; }

    // the type index of the processor (e.g. CPU, vector/tensor core)
    inline unsigned processorType(unsigned p1) const { return processor_type[p1]; }

    void setProcessorType(unsigned p1, unsigned type) {

        if (p1 < 0 || p1 >= number_processors || type < 0)
            throw std::invalid_argument("Invalid Argument");

        processor_type[p1] = type;
        number_of_processor_types = std::max(number_of_processor_types, type + 1);
    }

    std::vector<unsigned> getProcessorTypeCount() const {

        std::vector<unsigned> type_count(number_of_processor_types, 0);
        for (unsigned p = 0; p < number_processors; p++) {
            type_count[processor_type[p]]++;
        }
        return type_count;

    }

    void print_architecture(std::ostream& os) const;

    void updateNumberOfProcessorTypes();

    inline unsigned getNumberOfProcessorTypes() const { return number_of_processor_types; };

    inline MEMORY_CONSTRAINT_TYPE getMemoryConstraintType() const { return memory_const_type; }
    inline void setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE memory_const_type_) {
        memory_const_type = memory_const_type_;
    }
};
