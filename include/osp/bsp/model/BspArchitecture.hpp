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

#include "osp/auxiliary/misc.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/graph_traits.hpp"
#include "osp/graph_implementations/integral_range.hpp"

namespace osp {

/**
 * @enum MEMORY_CONSTRAINT_TYPE
 * @brief Enumerates the different types of memory constraints.
 * Memory bounds are set per processor and apply to aggregated memory weights of nodes according to the different types of memory constraints.
 */
enum class MEMORY_CONSTRAINT_TYPE {
    NONE,                     /** No memory constraints. */
    LOCAL,                    /** The memory bounds apply to the sum of memory weights of nodes assigned to the same processor and superstep. */
    GLOBAL,                   /** The memory bounds apply to the sum of memory weights of the nodes assigned to the same processor. */
    PERSISTENT_AND_TRANSIENT, /** Memory bounds apply to the sum of memory weights of nodes assigned to the same processor plus the maximum communication weight of a node assigned to a processor. */
    LOCAL_IN_OUT,             /** Memory constraints are local in-out. Experimental. */
    LOCAL_INC_EDGES,          /** Memory constraints are local incident edges. Experimental. */
    LOCAL_SOURCES_INC_EDGES   /** Memory constraints are local source incident edges. Experimental. */
};

inline std::ostream &operator<<(std::ostream &os, MEMORY_CONSTRAINT_TYPE type) {
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
    /**
     * @brief The number of processors in the architecture. Must be at least 1.
     *
     */
    unsigned numberOfProcessors_;

    /**
     * @brief The number of processor types in the architecture. See processorTypes_ for more details.
     *
     */
    unsigned numberOfProcessorTypes_;

    /**
     * @brief The communication costs, typically denoted 'g' for the BSP model.
     */
    v_commw_t<Graph_t> communicationCosts_;

    /**
     * @brief The synchronisation costs, typically denoted 'L' for the BSP model.
     */
    v_commw_t<Graph_t> synchronisationCosts_;

    /**
     * @brief The architecture allows to specify memory bounds per processor.
     */
    std::vector<v_memw_t<Graph_t>> memoryBound_;

    /**
     * @brief Flag to indicate whether the architecture is NUMA , i.e., whether the send costs are different for different pairs of processors.
     */
    bool isNuma_;

    /**
     * @brief The architecture allows to specify processor types. Processor types are used to express compatabilities, which can be specified in the BspInstance, regarding node types.
     */
    std::vector<unsigned> processorTypes_;

    /**
     * @brief A  p x p matrix of send costs. Diagonal entries should be zero.
     */
    std::vector<std::vector<v_commw_t<Graph_t>>> sendCosts_;

    /**
     * @brief The memory constraint type.
     */
    MEMORY_CONSTRAINT_TYPE memoryConstraintType_ = MEMORY_CONSTRAINT_TYPE::NONE;

    bool AreSendCostsNuma() {
        if (numberOfProcessors_ == 1)
            return false;

        v_commw_t<Graph_t> val = sendCosts_[0][1];
        for (unsigned p1 = 0; p1 < numberOfProcessors_; p1++) {
            for (unsigned p2 = 0; p2 < numberOfProcessors_; p2++) {
                if (p1 == p2)
                    continue;
                if (sendCosts_[p1][p2] != val)
                    return true;
            }
        }
        return false;
    }

  public:
    /**
     * @brief Default constructor.
     * Initializes a BSP architecture with 2 processors, 1 processor type,
     * communication costs of 1, synchronisation costs of 2, memory bounds of 100,
     * and send costs of 1 between all processors.
     */
    BspArchitecture()
        : numberOfProcessors_(2), numberOfProcessorTypes_(1), communicationCosts_(1), synchronisationCosts_(2),
          memoryBound_(std::vector<v_memw_t<Graph_t>>(numberOfProcessors_, 100)), isNuma_(false),
          processorTypes_(std::vector<unsigned>(numberOfProcessors_, 0)),
          sendCosts_(std::vector<std::vector<v_commw_t<Graph_t>>>(
              numberOfProcessors_, std::vector<v_commw_t<Graph_t>>(numberOfProcessors_, 1))) {
        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            sendCosts_[i][i] = 0;
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
     * @param NumberOfProcessors The number of processors in the architecture.
     * @param CommunicationCost The communication cost between processors.
     * @param SynchronisationCost The synchronization cost between processors.
     * @param MemoryBound The memory bound for each processor (default: 100).
     */
    BspArchitecture(unsigned NumberOfProcessors, v_commw_t<Graph_t> CommunicationCost, v_commw_t<Graph_t> SynchronisationCost,
                    v_memw_t<Graph_t> MemoryBound = 100)
        : numberOfProcessors_(NumberOfProcessors), numberOfProcessorTypes_(1), communicationCosts_(CommunicationCost),
          synchronisationCosts_(SynchronisationCost),
          memoryBound_(std::vector<v_memw_t<Graph_t>>(NumberOfProcessors, MemoryBound)), isNuma_(false),
          processorTypes_(std::vector<unsigned>(NumberOfProcessors, 0)),
          sendCosts_(std::vector<std::vector<v_commw_t<Graph_t>>>(
              numberOfProcessors_, std::vector<v_commw_t<Graph_t>>(numberOfProcessors_, 1))) {
        if (NumberOfProcessors == 0) {
            throw std::runtime_error("BspArchitecture: Number of processors must be greater than 0.");
        }

        for (unsigned i = 0; i < NumberOfProcessors; i++) {
            sendCosts_[i][i] = 0;
        }
    }

    /**
     * @brief Copy constructor from a BspArchitecture with a different graph type.
     *
     * @tparam Graph_t_other The graph type of the other BspArchitecture.
     * @param other The other BspArchitecture object.
     */
    template<typename Graph_t_other>
    BspArchitecture(const BspArchitecture<Graph_t_other> &other)
        : numberOfProcessors_(other.numberOfProcessors()), numberOfProcessorTypes_(other.getNumberOfProcessorTypes()),
          communicationCosts_(other.communicationCosts()), synchronisationCosts_(other.synchronisationCosts()),
          memoryBound_(other.memoryBound()), isNuma_(other.isNumaArchitecture()), processorTypes_(other.processorTypes()),
          sendCosts_(other.sendCosts()) {

        static_assert(std::is_same_v<v_memw_t<Graph_t>, v_memw_t<Graph_t_other>>,
                      "BspArchitecture: Graph_t and Graph_t_other have the same memory weight type.");

        static_assert(std::is_same_v<v_commw_t<Graph_t>, v_commw_t<Graph_t_other>>,
                      "BspArchitecture: Graph_t and Graph_t_other have the same communication weight type.");

        static_assert(std::is_same_v<v_type_t<Graph_t>, v_type_t<Graph_t_other>>,
                      "BspArchitecture: Graph_t and Graph_t_other have the same processor type.");
    }

    /**
     * @brief Constructs a BspArchitecture object with custom send costs.
     *
     * @param NumberOfProcessors The number of processors.
     * @param CommunicationCost The communication cost.
     * @param SynchronisationCost The synchronization cost.
     * @param SendCosts The matrix of send costs between processors.
     */
    BspArchitecture(unsigned NumberOfProcessors, v_commw_t<Graph_t> CommunicationCost, v_commw_t<Graph_t> SynchronisationCost,
                    std::vector<std::vector<v_commw_t<Graph_t>>> SendCosts)
        : numberOfProcessors_(NumberOfProcessors), numberOfProcessorTypes_(1), communicationCosts_(CommunicationCost),
          synchronisationCosts_(SynchronisationCost), memoryBound_(std::vector<v_memw_t<Graph_t>>(NumberOfProcessors, 100)),
          processorTypes_(std::vector<unsigned>(NumberOfProcessors, 0)), sendCosts_(SendCosts) {
        if (numberOfProcessors_ != sendCosts_.size()) {
            throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
        }
        if (std::any_of(sendCosts_.begin(), sendCosts_.end(),
                        [NumberOfProcessors](const auto &thing) { return thing.size() != NumberOfProcessors; })) {
            throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
        }

        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            sendCosts_[i][i] = 0;
        }

        isNuma_ = AreSendCostsNuma();
    }

    /**
     * @brief Constructs a BspArchitecture object with custom send costs and memory bound.
     *
     * @param NumberOfProcessors The number of processors.
     * @param CommunicationCost The communication cost.
     * @param SynchronisationCost The synchronization cost.
     * @param MemoryBound The memory bound for each processor.
     * @param SendCosts The matrix of send costs between processors.
     */
    BspArchitecture(unsigned NumberOfProcessors, v_commw_t<Graph_t> CommunicationCost, v_commw_t<Graph_t> SynchronisationCost,
                    v_memw_t<Graph_t> MemoryBound, std::vector<std::vector<v_commw_t<Graph_t>>> SendCosts)
        : numberOfProcessors_(NumberOfProcessors), numberOfProcessorTypes_(1), communicationCosts_(CommunicationCost),
          synchronisationCosts_(SynchronisationCost),
          memoryBound_(std::vector<v_memw_t<Graph_t>>(NumberOfProcessors, MemoryBound)),
          processorTypes_(std::vector<unsigned>(NumberOfProcessors, 0)), sendCosts_(SendCosts) {
        if (numberOfProcessors_ != sendCosts_.size()) {
            throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
        }
        if (std::any_of(sendCosts_.begin(), sendCosts_.end(),
                        [NumberOfProcessors](const auto &thing) { return thing.size() != NumberOfProcessors; })) {
            throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
        }

        for (unsigned i = 0u; i < numberOfProcessors_; i++) {
            sendCosts_[i][i] = 0u;
        }

        isNuma_ = AreSendCostsNuma();
    }

    /**
     * @brief Sets the uniform send cost for each pair of processors.
     * The send cost is set to 0 if the processors are the same, and 1 otherwise.
     */
    void SetUniformSendCost() {
        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            for (unsigned j = 0; j < numberOfProcessors_; j++) {
                if (i == j) {
                    sendCosts_[i][j] = 0;
                } else {
                    sendCosts_[i][j] = 1;
                }
            }
        }
        isNuma_ = false;
    }

    /**
     * @brief Sets the exponential send cost for the BspArchitecture.
     *
     * This function calculates and sets the exponential send cost for each pair of processors in the BspArchitecture.
     * The send cost is determined based on the base value and the position of the processors in the architecture.
     *
     * @param base The base value used to calculate the send cost.
     */
    void SetExpSendCost(v_commw_t<Graph_t> base) {
        isNuma_ = true;

        unsigned maxPos = 1;
        constexpr unsigned two = 2;
        for (; intpow(two, maxPos + 1) <= numberOfProcessors_ - 1; ++maxPos) {
        }

        for (unsigned i = 0; i < numberOfProcessors_; ++i) {
            for (unsigned j = i + 1; j < numberOfProcessors_; ++j) {
                // Corrected loop to avoid underflow issues with unsigned
                for (int pos = static_cast<int>(maxPos); pos >= 0; --pos) {
                    if (((1 << pos) & i) != ((1 << pos) & j)) {
                        sendCosts_[i][j] = sendCosts_[j][i] = intpow(base, static_cast<unsigned>(pos));
                        break;
                    }
                }
            }
        }
    }

    /**
     * @brief Returns a view of processor indices from 0 to numberOfProcessors_ - 1.
     * @return An integral view of processor indices.
     */
    inline auto processors() const { return integral_range<unsigned>(numberOfProcessors_); }

    /**
     * @brief Sets the send costs for the BspArchitecture.
     *
     * @param vec A 2D vector representing the send costs between processors.
     * @throws std::invalid_argument if the size of the vector is invalid or diagonal elements are not 0.
     */
    void SetSendCosts(const std::vector<std::vector<v_commw_t<Graph_t>>> &vec) {
        if (vec.size() != numberOfProcessors_) {
            throw std::invalid_argument("Invalid Argument: Vector size mismatch.");
        }

        isNuma_ = false;
        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            if (vec[i].size() != numberOfProcessors_) {
                throw std::invalid_argument("Invalid Argument: Inner vector size mismatch.");
            }

            for (unsigned j = 0; j < numberOfProcessors_; j++) {
                if (i == j) {
                    if (vec[i][j] != 0)
                        throw std::invalid_argument("Invalid Argument: Diagonal elements should be 0.");
                } else {
                    sendCosts_[i][j] = vec[i][j];

                    if (numberOfProcessors_ > 1 && vec[i][j] != vec[0][1]) {
                        isNuma_ = true;
                    }
                }
            }
        }
    }

    /**
     * @brief Sets the send costs between two processors.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     * @param cost The cost of sending data between the processors.
     * @throws std::invalid_argument if the processor indices are out of bounds.
     */
    void SetSendCosts(unsigned p1, unsigned p2, v_commw_t<Graph_t> cost) {
        if (p1 >= numberOfProcessors_ || p2 >= numberOfProcessors_) // Fixed condition: p2 >= number_processors
            throw std::invalid_argument("Invalid Argument: Processor index out of bounds.");

        if (p1 != p2) {
            sendCosts_[p1][p2] = cost;
            isNuma_ = AreSendCostsNuma();
        }
    }

    /**
     * @brief Sets the memory bound for all processors.
     * @param MemoryBound The new memory bound.
     */
    inline void setMemoryBound(v_memw_t<Graph_t> MemoryBound) {
        memoryBound_ = std::vector<v_memw_t<Graph_t>>(numberOfProcessors_, MemoryBound);
    }

    /**
     * @brief Sets the memory bound for all processors using a vector.
     * @param MemoryBound The vector of memory bounds.
     * @throws std::invalid_argument if the size of the vector is invalid.
     */
    inline void setMemoryBound(const std::vector<v_memw_t<Graph_t>> &MemoryBound) {
        if (MemoryBound.size() != numberOfProcessors_) {
            throw std::invalid_argument("Invalid Argument: Memory bound vector size does not match number of processors.");
        }
        memoryBound_ = MemoryBound;
    }

    /**
     * @brief Sets the memory bound for a specific processor.
     * @param MemoryBound The new memory bound.
     * @param proc The processor index.
     * @throws std::invalid_argument if the processor index is out of bounds.
     */
    inline void setMemoryBound(v_memw_t<Graph_t> MemoryBound, unsigned proc) {
        if (proc >= numberOfProcessors_) {
            throw std::invalid_argument("Invalid Argument: Processor index out of bounds in setMemoryBound.");
        }
        memoryBound_[proc] = MemoryBound;
    }

    /**
     * @brief Sets the synchronization costs.
     * @param SynchCost The synchronization costs.
     */
    inline void setSynchronisationCosts(v_commw_t<Graph_t> SynchCost) { synchronisationCosts_ = SynchCost; }

    /**
     * @brief Sets the communication costs.
     * @param CommCost The communication costs.
     */
    inline void setCommunicationCosts(v_commw_t<Graph_t> CommCost) { communicationCosts_ = CommCost; }

    /**
     * @brief Sets the number of processors. Processor type is set to 0 for all processors.
     * Resets send costs to uniform (1) and diagonal to 0.
     * @param num_proc The number of processors.
     * @throws std::invalid_argument if the number of processors is 0.
     */
    void setNumberOfProcessors(unsigned num_proc) {
        numberOfProcessors_ = num_proc;
        numberOfProcessorTypes_ = 1;
        processor_types = std::vector<unsigned>(numberOfProcessors_, 0);
        sendCosts_ = std::vector<std::vector<v_commw_t<Graph_t>>>(numberOfProcessors_, std::vector<v_commw_t<Graph_t>>(numberOfProcessors_, 1));

        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            sendCosts_[i][i] = 0;
        }
        memoryBound_.resize(num_proc, memoryBound_.back());

        isNuma_ = false;
    }

    /**
     * @brief Sets the number of processors and their types.
     * Resets send costs to uniform (1).
     * @param processor_types_ The types of the respective processors.
     */
    void setProcessorsWithTypes(const std::vector<v_type_t<Graph_t>> &processor_types_) {
        if (processor_types_.size() > std::numeric_limits<unsigned>::max()) {
            throw std::invalid_argument("Invalid Argument: Number of processors exceeds the limit.");
        }
        numberOfProcessors_ = static_cast<unsigned>(processor_types_.size());
        numberOfProcessorTypes_ = 0;
        processor_types = processor_types_;
        sendCosts_ = std::vector<std::vector<v_commw_t<Graph_t>>>(numberOfProcessors_, std::vector<v_commw_t<Graph_t>>(numberOfProcessors_, 1));

        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            sendCosts_[i][i] = 0;
        }
        memoryBound_.resize(numberOfProcessors_, memoryBound_.back());
        isNuma_ = false;
        updateNumberOfProcessorTypes();
    }

    /**
     * @brief Checks if the architecture is NUMA.
     * @return True if NUMA, false otherwise.
     */
    [[nodiscard]] inline bool isNumaArchitecture() const { return isNuma_; }

    /**
     * @brief Sets processors based on counts of consecutive types.
     * @param processor_type_count_ Vector where index is type and value is count of processors of that type.
     * @param processor_type_memory_ Vector where index is type and value is memory bound for that type.
     */
    void SetProcessorsConsequTypes(const std::vector<v_type_t<Graph_t>> &processor_type_count_,
                                   const std::vector<v_memw_t<Graph_t>> &processor_type_memory_) {
        if (processor_type_count_.size() != processor_type_memory_.size()) {
            throw std::invalid_argument(
                "Invalid Argument: processor_type_count_ and processor_type_memory_ must have the same size.");
        }

        if (processor_type_count_.size() > std::numeric_limits<unsigned>::max()) {
            throw std::invalid_argument("Invalid Argument: Number of processors exceeds the limit.");
        }

        numberOfProcessorTypes_ = static_cast<unsigned>(processor_type_count_.size());
        numberOfProcessors_ = std::accumulate(processor_type_count_.begin(), processor_type_count_.end(), 0u);

        processor_types = std::vector<v_type_t<Graph_t>>(numberOfProcessors_, 0);
        memoryBound_ = std::vector<v_memw_t<Graph_t>>(numberOfProcessors_, 0);

        unsigned offset = 0;
        for (unsigned i = 0; i < processor_type_count_.size(); i++) {
            for (unsigned j = 0; j < processor_type_count_[i]; j++) {
                processor_types[offset + j] = i;
                memoryBound_[offset + j] = processor_type_memory_[i];
            }
            offset += processor_type_count_[i];
        }

        sendCosts_ = std::vector<std::vector<v_commw_t<Graph_t>>>(
            numberOfProcessors_, std::vector<v_commw_t<Graph_t>>(numberOfProcessors_, 1));
        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            sendCosts_[i][i] = 0;
        }
        isNuma_ = false;
    }

    /**
     * @brief Returns the memory bounds of all processors.
     * @return Vector of memory bounds.
     */
    [[nodiscard]] inline const std::vector<v_memw_t<Graph_t>> &memoryBound() const { return memoryBound_; }

    /**
     * @brief Returns the memory bound of a specific processor.
     * @param proc The processor index.
     * @return The memory bound.
     */
    [[nodiscard]] inline v_memw_t<Graph_t> memoryBound(unsigned proc) const { return memoryBound_.at(proc); }

    [[nodiscard]] v_memw_t<Graph_t> minMemoryBound() const { return *(std::min_element(memoryBound_.begin(), memoryBound_.end())); }
    [[nodiscard]] v_memw_t<Graph_t> maxMemoryBound() const { return *(std::max_element(memoryBound_.begin(), memoryBound_.end())); }
    [[nodiscard]] v_memw_t<Graph_t> sumMemoryBound() const { return std::accumulate(memoryBound_.begin(), memoryBound_.end(), 0); }

    [[nodiscard]] v_memw_t<Graph_t> maxMemoryBoundProcType(v_type_t<Graph_t> procType) const {
        v_memw_t<Graph_t> max_mem = 0;
        for (unsigned proc = 0; proc < numberOfProcessors_; proc++) {
            if (processor_types.at(proc) == procType) {
                max_mem = std::max(max_mem, memoryBound_.at(proc));
            }
        }
        return max_mem;
    }

    /**
     * @brief Returns the number of processors.
     * @return The number of processors.
     */
    [[nodiscard]] inline unsigned numberOfProcessors() const { return numberOfProcessors_; }

    /**
     * @brief Returns the communication costs.
     * @return The communication costs.
     */
    [[nodiscard]] inline v_commw_t<Graph_t> communicationCosts() const { return communication_costs; }

    /**
     * @brief Returns the synchronization costs.
     * @return The synchronization costs.
     */
    [[nodiscard]] inline v_commw_t<Graph_t> synchronisationCosts() const { return synchronisation_costs; }

    /**
     * @brief Returns a copy of the send costs matrix.
     * @return A copy of the send costs matrix.
     */
    [[nodiscard]] inline std::vector<std::vector<v_commw_t<Graph_t>>> sendCostMatrixCopy() const { return sendCosts_; }

    /**
     * @brief Returns a reference to the send costs matrix.
     * @return A reference to the send costs matrix.
     */
    [[nodiscard]] inline const std::vector<std::vector<v_commw_t<Graph_t>>> &sendCostMatrix() const { return sendCosts_; }

    /**
     * @brief Returns the processor types.
     * @return Vector of processor types.
     */
    [[nodiscard]] inline const std::vector<unsigned> &processorTypes() const { return processor_types; }

    /**
     * @brief Returns the communication costs between two processors.
     * The communication costs are the send costs multiplied by the communication costs factor.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     * @return The communication costs between the two processors.
     */
    [[nodiscard]] inline v_commw_t<Graph_t> communicationCosts(unsigned p1, unsigned p2) const {
        return communication_costs * sendCosts_[p1][p2];
    }

    /**
     * @brief Returns the send costs between two processors.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     * @return The send costs between the two processors.
     */
    [[nodiscard]] inline v_commw_t<Graph_t> sendCosts(unsigned p1, unsigned p2) const { return sendCosts_[p1][p2]; }

    /**
     * @brief Returns the type of a specific processor.
     * @param p1 The processor index.
     * @return The processor type.
     */
    [[nodiscard]] inline v_type_t<Graph_t> processorType(unsigned p1) const { return processor_types[p1]; }

    /**
     * @brief Sets the type of a specific processor.
     * @param p1 The processor index.
     * @param type The new processor type.
     */
    void setProcessorType(unsigned p1, v_type_t<Graph_t> type) {
        if (p1 >= numberOfProcessors_)
            throw std::invalid_argument("Invalid Argument: Processor index out of bounds.");

        processor_types[p1] = type;
        numberOfProcessorTypes_ = std::max(numberOfProcessorTypes_, type + 1u);
    }

    /**
     * @brief Returns the count of processors for each type.
     * @return Vector where index is type and value is count.
     */
    [[nodiscard]] std::vector<unsigned> getProcessorTypeCount() const {
        std::vector<unsigned> type_count(numberOfProcessorTypes_, 0u);
        for (unsigned p = 0u; p < numberOfProcessors_; p++) {
            type_count[processor_types[p]]++;
        }
        return type_count;
    }

    [[nodiscard]] unsigned getMinProcessorTypeCount() const {
        const auto &type_count = getProcessorTypeCount();
        if (type_count.empty()) {
            return 0;
        }
        return *std::min_element(type_count.begin(), type_count.end());
    }

    /**
     * @brief Prints the architecture details to the output stream.
     * @param os The output stream.
     */
    void print(std::ostream &os) const {
        os << "Architecture info:  number of processors: " << numberOfProcessors_
           << ", Number of processor types: " << numberOfProcessorTypes_
           << ", Communication costs: " << communication_costs << ", Synchronization costs: " << synchronisation_costs
           << "\n";
        os << std::setw(17) << " Processor: ";
        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            os << std::right << std::setw(5) << i << " ";
        }
        os << "\n";
        os << std::setw(17) << "Processor type: ";
        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            os << std::right << std::setw(5) << processor_types[i] << " ";
        }
        os << "\n";
        os << std::setw(17) << "Memory bound: ";
        for (unsigned i = 0; i < numberOfProcessors_; i++) {
            os << std::right << std::setw(5) << memory_bound[i] << " ";
        }
        os << "\n";
    }

    void updateNumberOfProcessorTypes() {
        numberOfProcessorTypes_ = 0;
        for (unsigned p = 0; p < numberOfProcessors_; p++) {
            if (processor_types[p] >= numberOfProcessorTypes_) {
                numberOfProcessorTypes_ = processor_types[p] + 1;
            }
        }
    }

    [[nodiscard]] std::vector<std::vector<unsigned>> getProcessorIdsByType() const {
        std::vector<std::vector<unsigned>> processor_ids_by_type(numberOfProcessorTypes_);
        for (unsigned i = 0; i < numberOfProcessors(); ++i) {
            processor_ids_by_type[processorType(i)].push_back(i);
        }
        return processor_ids_by_type;
    }

    [[nodiscard]] inline unsigned getNumberOfProcessorTypes() const { return numberOfProcessorTypes_; };

    [[nodiscard]] inline MEMORY_CONSTRAINT_TYPE getMemoryConstraintType() const { return memory_constraint_type; }
    inline void setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE memory_constraint_type_) {
        memory_constraint_type = memory_constraint_type_;
    }
};

} // namespace osp