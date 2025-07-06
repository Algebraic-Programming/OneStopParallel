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
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "auxiliary/misc.hpp"
#include "concepts/computational_dag_concept.hpp"
#include "concepts/graph_traits.hpp"

namespace osp {

static constexpr unsigned CacheLineSize = 64;

enum class MEMORY_CONSTRAINT_TYPE {
    NONE,
    LOCAL,
    GLOBAL,
    PERSISTENT_AND_TRANSIENT,
    LOCAL_IN_OUT,
    LOCAL_INC_EDGES,
    LOCAL_SOURCES_INC_EDGES
};

std::ostream &operator<<(std::ostream &os, MEMORY_CONSTRAINT_TYPE type) {
    switch (type) {
    case MEMORY_CONSTRAINT_TYPE::NONE:
        os << "NONE";
        break;
    case MEMORY_CONSTRAINT_TYPE::LOCAL:
        os << "LOCAL";
        break;
    case MEMORY_CONSTRAINT_TYPE::GLOBAL:
        os << "GLOBAL";
        break;
    case MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT:
        os << "PERSISTENT_AND_TRANSIENT";
        break;
    case MEMORY_CONSTRAINT_TYPE::LOCAL_IN_OUT:
        os << "LOCAL_IN_OUT";
        break;
    case MEMORY_CONSTRAINT_TYPE::LOCAL_INC_EDGES:
        os << "LOCAL_INC_EDGES";
        break;
    case MEMORY_CONSTRAINT_TYPE::LOCAL_SOURCES_INC_EDGES:
        os << "LOCAL_SOURCES_INC_EDGES";
        break;
    default:
        os << "UNKNOWN";
        break;
    }
    return os;
}

/**
 * @class BspArchitecture
 * @brief Represents the architecture of a BSP (Bulk Synchronous Parallel) system.
 *
 * The BspArchitecture class stores information about the number of processors, communication costs,
 * synchronization costs, and send costs between processors in a BSP system. It provides methods to
 * set and retrieve these values.
 */
template<typename Graph_t>
class BspArchitecture {

    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");

  private:
    unsigned number_processors;
    unsigned number_of_processor_types;

    v_commw_t<Graph_t> communication_costs;
    v_commw_t<Graph_t> synchronisation_costs;

    std::vector<v_memw_t<Graph_t>> memory_bound;

    bool isNuma;

    std::vector<unsigned> processor_type;

    std::vector<std::vector<v_commw_t<Graph_t>>> send_costs;

    MEMORY_CONSTRAINT_TYPE memory_const_type = MEMORY_CONSTRAINT_TYPE::NONE;

    bool are_send_cost_numa() {
        if (number_processors == 1)
            return false;

        v_commw_t<Graph_t> val = send_costs[0][1];
        for (unsigned p1 = 0; p1 < number_processors; p1++) {
            for (unsigned p2 = 0; p2 < number_processors; p2++) {
                if (p1 == p2)
                    continue;
                if (send_costs[p1][p2] != val)
                    return true;
            }
        }
        return false;
    }

  public:
    BspArchitecture()
        : number_processors(2), number_of_processor_types(1), communication_costs(1), synchronisation_costs(2),
          memory_bound(std::vector<v_memw_t<Graph_t>>(number_processors, 100)), isNuma(false),
          processor_type(std::vector<unsigned>(number_processors, 0)),
          send_costs(std::vector<std::vector<v_commw_t<Graph_t>>>(
              number_processors, std::vector<v_commw_t<Graph_t>>(number_processors, 1))) {
        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }
    }

    BspArchitecture(const BspArchitecture &other) = default;
    BspArchitecture(BspArchitecture &&other) = default;
    BspArchitecture &operator=(const BspArchitecture &other) = default;
    BspArchitecture &operator=(BspArchitecture &&other) = default;
    ~BspArchitecture() = default;

    /**
     * @brief Constructs a BspArchitecture object with the specified number of processors, communication cost, and
     * synchronization cost.
     *
     * @param processors The number of processors in the architecture.
     * @param comm_cost The communication cost between processors.
     * @param synch_cost The synchronization cost between processors.
     */
    BspArchitecture(unsigned processors, v_commw_t<Graph_t> comm_cost, v_commw_t<Graph_t> synch_cost,
                    v_memw_t<Graph_t> memory_bound_ = 100)
        : number_processors(processors), number_of_processor_types(1), communication_costs(comm_cost),
          synchronisation_costs(synch_cost),
          memory_bound(std::vector<v_memw_t<Graph_t>>(number_processors, memory_bound_)), isNuma(false),
          processor_type(std::vector<unsigned>(number_processors, 0)),
          send_costs(std::vector<std::vector<v_commw_t<Graph_t>>>(
              number_processors, std::vector<v_commw_t<Graph_t>>(number_processors, 1))) {

        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }
    }

    template<typename Graph_t_other>
    BspArchitecture(const BspArchitecture<Graph_t_other> &other)
        : number_processors(other.numberOfProcessors()), number_of_processor_types(other.getNumberOfProcessorTypes()),
          communication_costs(other.communicationCosts()), synchronisation_costs(other.synchronisationCosts()),
          memory_bound(other.memoryBound()), isNuma(other.isNumaArchitecture()), processor_type(other.processorTypes()),
          send_costs(other.sendCosts()) {

        static_assert(std::is_same_v<v_memw_t<Graph_t>, v_memw_t<Graph_t_other>>,
                      "BspArchitecture: Graph_t and Graph_t_other have the same memory weight type.");

        static_assert(std::is_same_v<v_commw_t<Graph_t>, v_commw_t<Graph_t_other>>,
                      "BspArchitecture: Graph_t and Graph_t_other have the same communication weight type.");

        static_assert(std::is_same_v<v_type_t<Graph_t>, v_type_t<Graph_t_other>>,
                      "BspArchitecture: Graph_t and Graph_t_other have the same processor type.");
    }

    /**
     * @brief Constructs a BspArchitecture object with the specified number of processors, communication cost, and
     * synchronization cost.
     *
     * @param processors The number of processors in the architecture.
     * @param comm_cost The communication cost between processors.
     * @param synch_cost The synchronization cost between processors.
     */
    BspArchitecture(unsigned int processors, v_commw_t<Graph_t> comm_cost, v_commw_t<Graph_t> synch_cost,
                    std::vector<std::vector<v_commw_t<Graph_t>>> send_costs_)
        : number_processors(processors), number_of_processor_types(1), communication_costs(comm_cost),
          synchronisation_costs(synch_cost), memory_bound(std::vector<v_memw_t<Graph_t>>(number_processors, 100)),
          processor_type(std::vector<unsigned>(number_processors, 0)), send_costs(send_costs_) {

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
    BspArchitecture(unsigned int processors, v_commw_t<Graph_t> comm_cost, v_commw_t<Graph_t> synch_cost,
                    v_memw_t<Graph_t> memory_bound_, std::vector<std::vector<v_commw_t<Graph_t>>> send_costs_)
        : number_processors(processors), number_of_processor_types(1), communication_costs(comm_cost),
          synchronisation_costs(synch_cost),
          memory_bound(std::vector<v_memw_t<Graph_t>>(number_processors, memory_bound_)),
          processor_type(std::vector<unsigned>(number_processors, 0)), send_costs(send_costs_) {

        if (number_processors != send_costs.size()) {
            throw std::invalid_argument("send_costs_ needs to be a processors x processors matrix.\n");
        }
        if (std::any_of(send_costs.begin(), send_costs.end(),
                        [processors](const auto &thing) { return thing.size() != processors; })) {
            throw std::invalid_argument("send_costs_ needs to be a processors x processors matrix.\n");
        }

        for (unsigned i = 0u; i < number_processors; i++) {
            send_costs[i][i] = 0u;
        }

        isNuma = are_send_cost_numa();
    }

    /**
     * Sets the uniform send cost for each pair of processors in the BSP architecture.
     * The send cost is set to 0 if the processors are the same, and 1 otherwise.
     * This function assumes that the number of processors has already been set.
     */
    void SetUniformSendCost() {

        for (unsigned i = 0; i < number_processors; i++) {
            for (unsigned j = 0; j < number_processors; j++) {
                if (i == j) {
                    send_costs[i][j] = 0;
                } else {
                    send_costs[i][j] = 1;
                }
            }
        }
        isNuma = false;
    }

    /**
     * @brief Sets the exponential send cost for the BspArchitecture.
     *
     * This function calculates and sets the exponential send cost for each pair of processors in the BspArchitecture.
     * The send cost is determined based on the base value and the position of the processors in the architecture.
     *
     * @param base The base value used to calculate the send cost.
     */
    void SetExpSendCost(unsigned int base) {

        isNuma = true;

        unsigned maxPos = 1;
        const unsigned two = 2;
        for (; uintpow(two, maxPos + 1) <= number_processors - 1; ++maxPos) {
        }
        for (unsigned i = 0; i < number_processors; ++i)
            for (unsigned j = i + 1; j < number_processors; ++j)
                for (unsigned pos = maxPos; pos <= maxPos; --pos)
                    if (((1 << pos) & i) != ((1 << pos) & j)) {
                        send_costs[i][j] = send_costs[j][i] = intpow(base, pos);
                        break;
                    }
    }

    /**
     * @brief Computes the average communication cost of the BspArchitecture.
     *
     * This function computes the average communication cost of the BspArchitecture object.
     * The average communication cost is calculated as the sum of the send costs between processors divided by the
     * number of processors.
     *
     * @return The average communication cost as an unsigned integer.
     */
    v_commw_t<Graph_t> computeCommAverage() const {

        double avg = 0;
        for (unsigned i = 0; i < number_processors; ++i)
            for (unsigned j = 0; j < number_processors; ++j)
                avg += static_cast<double>(send_costs[i][j]);
        avg = avg * (double)communication_costs / (double)number_processors / (double)number_processors;

        if (avg > static_cast<double>(std::numeric_limits<unsigned>::max())) {
            throw std::invalid_argument("avg comm exceeds the limit (something is very wrong)");
        }

        return static_cast<v_commw_t<Graph_t>>(std::round(avg));
    }

    /**
     * Sets the send costs for the BspArchitecture.
     *
     * @param vec A 2D vector representing the send costs between processors.
     *            The size of the vector must be equal to the number of processors.
     *            Each inner vector must also have a size equal to the number of processors.
     * @throws std::invalid_argument if the size of the vector or inner vectors is invalid.
     */
    void setSendCosts(const std::vector<std::vector<v_commw_t<Graph_t>>> &vec) {

        if (vec.size() != number_processors) {
            throw std::invalid_argument("Invalid Argument");
        }

        isNuma = false;
        for (unsigned i = 0; i < number_processors; i++) {

            if (vec[i].size() != number_processors) {
                throw std::invalid_argument("Invalid Argument");
            }

            for (unsigned j = 0; j < number_processors; j++) {

                if (i == j) {
                    if (vec[i][j] != 0)
                        throw std::invalid_argument("Invalid Argument, Diagonal elements should be 0");
                } else {
                    send_costs[i][j] = vec[i][j];

                    if (number_processors > 1 && vec[i][j] != vec[0][1]) {
                        isNuma = true;
                    }
                }
            }
        }
    }

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
    void setSendCosts(unsigned p1, unsigned p2, v_commw_t<Graph_t> cost) {

        if (p1 >= number_processors || p2 > number_processors)
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
    inline void setMemoryBound(v_memw_t<Graph_t> memory_bound_) {
        memory_bound = std::vector<v_memw_t<Graph_t>>(number_processors, memory_bound_);
    }

    inline void setMemoryBound(const std::vector<v_memw_t<Graph_t>> &memory_bound_) { memory_bound = memory_bound_; }

    inline void setMemoryBound(v_memw_t<Graph_t> memory_bound_, unsigned proc) {

        if (proc >= number_processors) {
            throw std::invalid_argument("Invalid Argument setMemoryBound");
        }

        memory_bound[proc] = memory_bound_;
    }

    /**
     * @brief Sets the synchronization costs for the BspArchitecture.
     *
     * This function sets the synchronization costs for the BspArchitecture object.
     * The synchronization costs represent the costs of establishing communication between processors.
     *
     * @param synch_cost The synchronization costs to be set.
     */
    inline void setSynchronisationCosts(v_commw_t<Graph_t> synch_cost) { synchronisation_costs = synch_cost; }

    /**
     * @brief Sets the communication costs for the BspArchitecture.
     *
     * This function sets the communication costs for the BspArchitecture object.
     * The communication costs represent the costs of sending messages between processors.
     *
     * @param comm_cost The communication costs to be set.
     */
    inline void setCommunicationCosts(v_commw_t<Graph_t> comm_cost) { communication_costs = comm_cost; }

    /**
     * @brief Sets the number of processors in the BSP architecture.
     *
     * This function sets the number of processors in the BSP architecture and sets the send costs between processors
     * to 1. The send_costs matrix represents the costs of sending messages between processors. The diagonal elements of
     * the matrix are set to 0, indicating that there is no cost to send a message from a processor to itself.
     *
     * @param num_proc The number of processors in the BSP architecture.
     */
    void setNumberOfProcessors(unsigned num_proc) {

        number_processors = num_proc;
        number_of_processor_types = 1;
        processor_type = std::vector<unsigned>(number_processors, 0);
        send_costs = std::vector<std::vector<v_commw_t<Graph_t>>>(
            number_processors, std::vector<v_commw_t<Graph_t>>(number_processors, 1));
        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }
        memory_bound.resize(num_proc, memory_bound.back());

        isNuma = false;
    }

    /**
     * @brief Sets the number of processors and their types in the BSP architecture.
     *
     * This function sets the number of processors in the BSP architecture and sets the send costs between processors
     * to 1. The send_costs matrix represents the costs of sending messages between processors. The diagonal elements of
     * the matrix are set to 0, indicating that there is no cost to send a message from a processor to itself.
     *
     * @param processor_types_ The type of the respective processors.
     */
    void setProcessorsWithTypes(const std::vector<v_type_t<Graph_t>> &processor_types_) {

        if (processor_types_.size() > std::numeric_limits<unsigned>::max()) {
            throw std::invalid_argument("Invalid Argument, number of processors exceeds the limit");
        }

        number_processors = static_cast<unsigned>(processor_types_.size());

        number_of_processor_types = 0;
        processor_type = processor_types_;
        send_costs = std::vector<std::vector<v_commw_t<Graph_t>>>(
            number_processors, std::vector<v_commw_t<Graph_t>>(number_processors, 1));
        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }
        memory_bound.resize(number_processors, memory_bound.back());

        isNuma = false;
        updateNumberOfProcessorTypes();
    }

    /**
     * Returns whether the architecture is NUMA.
     *
     * @return True if the architecture is NUMA, false otherwise.
     */
    inline bool isNumaArchitecture() const { return isNuma; }

    void set_processors_consequ_types(const std::vector<v_type_t<Graph_t>> &processor_type_count_,
                                      const std::vector<v_memw_t<Graph_t>> &processor_type_memory_) {

        if (processor_type_count_.size() != processor_type_memory_.size()) {
            throw std::invalid_argument(
                "Invalid Argument, processor_type_count_ and processor_type_memory_ must have the same size");
        }

        if (processor_type_count_.size() > std::numeric_limits<unsigned>::max()) {
            throw std::invalid_argument("Invalid Argument, number of processors exceeds the limit");
        }

        number_of_processor_types = static_cast<unsigned>(processor_type_count_.size());
        number_processors = std::accumulate(processor_type_count_.begin(), processor_type_count_.end(), 0u);

        processor_type = std::vector<v_type_t<Graph_t>>(number_processors, 0);
        memory_bound = std::vector<v_memw_t<Graph_t>>(number_processors, 0);

        unsigned offset = 0;
        for (unsigned i = 0; i < processor_type_count_.size(); i++) {

            for (unsigned j = 0; j < processor_type_count_[i]; j++) {
                processor_type[offset + j] = i;
                memory_bound[offset + j] = processor_type_memory_[i];
            }
            offset += processor_type_count_[i];
        }

        send_costs = std::vector<std::vector<v_commw_t<Graph_t>>>(
            number_processors, std::vector<v_commw_t<Graph_t>>(number_processors, 1));
        for (unsigned i = 0; i < number_processors; i++) {
            send_costs[i][i] = 0;
        }
        isNuma = false;
    }

    /**
     * Returns the memory bound of the BspArchitecture.
     *
     * @return The memory bound as an unsigned integer.
     */
    inline const std::vector<v_memw_t<Graph_t>> &memoryBound() const { return memory_bound; }

    inline v_memw_t<Graph_t> memoryBound(unsigned proc) const { return memory_bound[proc]; }

    v_memw_t<Graph_t> minMemoryBound() const { return *(std::min_element(memory_bound.begin(), memory_bound.end())); }
    v_memw_t<Graph_t> maxMemoryBound() const { return *(std::max_element(memory_bound.begin(), memory_bound.end())); }
    v_memw_t<Graph_t> sumMemoryBound() const { return std::accumulate(memory_bound.begin(), memory_bound.end(), 0); }

    v_memw_t<Graph_t> maxMemoryBoundProcType(v_type_t<Graph_t> procType) const {
        v_memw_t<Graph_t> max_mem = 0;
        for (unsigned proc = 0; proc < number_processors; proc++) {
            if (processor_type[proc] == procType) {
                max_mem = std::max(max_mem, memory_bound[proc]);
            }
        }
        return max_mem;
    }

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
    inline v_commw_t<Graph_t> communicationCosts() const { return communication_costs; }

    /**
     * Returns the synchronization costs of the BspArchitecture.
     *
     * @return The synchronization costs as an unsigned integer.
     */
    inline v_commw_t<Graph_t> synchronisationCosts() const { return synchronisation_costs; }

    /**
     * Returns a copy of the send costs matrix.
     *
     * @return A copy of the send costs matrix.
     */
    inline std::vector<std::vector<v_commw_t<Graph_t>>> sendCostMatrixCopy() const { return send_costs; }

    /**
     * Returns a reference to the send costs matrix.
     *
     * @return A reference to the send costs matrix.
     */
    inline const std::vector<std::vector<v_commw_t<Graph_t>>> &sendCostMatrix() const { return send_costs; }

    // the type indeces of the processor (e.g. CPU, vector/tensor core)
    inline const std::vector<unsigned> &processorTypes() const { return processor_type; }

    /**
     * Returns the communication costs between two processors. The communication costs are the send costs multiplied by
     * the communication costs.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     *
     * @return The send costs between the two processors.
     */
    inline v_commw_t<Graph_t> communicationCosts(unsigned p1, unsigned p2) const {
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
    inline v_commw_t<Graph_t> sendCosts(unsigned p1, unsigned p2) const { return send_costs[p1][p2]; }

    inline auto sendCosts() const { return send_costs; }

    // the type index of the processor (e.g. CPU, vector/tensor core)
    inline v_type_t<Graph_t> processorType(unsigned p1) const { return processor_type[p1]; }

    void setProcessorType(unsigned p1, v_type_t<Graph_t> type) {

        if (p1 >= number_processors)
            throw std::invalid_argument("Invalid Argument");

        processor_type[p1] = type;
        number_of_processor_types = std::max(number_of_processor_types, type + 1u);
    }

    std::vector<unsigned> getProcessorTypeCount() const {

        std::vector<unsigned> type_count(number_of_processor_types, 0u);
        for (unsigned p = 0u; p < number_processors; p++) {
            type_count[processor_type[p]]++;
        }
        return type_count;
    }

    void print_architecture(std::ostream &os) const {

        os << "Architectur info:  number of processors: " << number_processors
           << ", Number of processor types: " << number_of_processor_types
           << ", Communication costs: " << communication_costs << ", Synchronization costs: " << synchronisation_costs
           << std::endl;
        os << std::setw(17) << " Processor: ";
        for (unsigned i = 0; i < number_processors; i++) {
            os << std::right << std::setw(5) << i << " ";
        }
        os << std::endl;
        os << std::setw(17) << "Processor type: ";
        for (unsigned i = 0; i < number_processors; i++) {
            os << std::right << std::setw(5) << processor_type[i] << " ";
        }
        os << std::endl;
        os << std::setw(17) << "Memory bound: ";
        for (unsigned i = 0; i < number_processors; i++) {
            os << std::right << std::setw(5) << memory_bound[i] << " ";
        }
        os << std::endl;
    }

    void updateNumberOfProcessorTypes() {
        number_of_processor_types = 0;
        for (unsigned p = 0; p < number_processors; p++) {
            if (processor_type[p] >= number_of_processor_types) {
                number_of_processor_types = processor_type[p] + 1;
            }
        }
    }

    inline unsigned getNumberOfProcessorTypes() const { return number_of_processor_types; };

    inline MEMORY_CONSTRAINT_TYPE getMemoryConstraintType() const { return memory_const_type; }
    inline void setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE memory_const_type_) {
        memory_const_type = memory_const_type_;
    }
};

} // namespace osp