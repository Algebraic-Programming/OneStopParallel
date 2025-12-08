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
     * @brief A flattened p x p matrix of send costs.
     * Access via index [i * numberOfProcessors_ + j].
     */
    std::vector<v_commw_t<Graph_t>> sendCosts_;

    /**
     * @brief The memory constraint type.
     */
    MEMORY_CONSTRAINT_TYPE memoryConstraintType_ = MEMORY_CONSTRAINT_TYPE::NONE;

    std::size_t FlatIndex(const unsigned row, const unsigned col) const {
        return static_cast<std::size_t>(row) * numberOfProcessors_ + col;
    }

    bool AreSendCostsNuma() {
        if (numberOfProcessors_ == 1U)
            return false;

        const v_commw_t<Graph_t> val = sendCosts_.at(1U);
        for (unsigned p1 = 0U; p1 < numberOfProcessors_; p1++) {
            for (unsigned p2 = 0U; p2 < numberOfProcessors_; p2++) {
                if (p1 == p2)
                    continue;
                if (sendCosts_.at(FlatIndex(p1, p2)) != val)
                    return true;
            }
        }
        return false;
    }

    void UpdateNumberOfProcessorTypes() {
        numberOfProcessorTypes_ = 0U;
        for (unsigned p = 0U; p < numberOfProcessors_; p++) {
            if (processorTypes_.at(p) >= numberOfProcessorTypes_) {
                numberOfProcessorTypes_ = processorTypes_.at(p) + 1U;
            }
        }
    }

    void SetSendCostDiagonalToZero() {
        for (unsigned i = 0U; i < numberOfProcessors_; i++) {
            sendCosts_.at(FlatIndex(i, i)) = 0U;
        }
    }

    void InitializeUniformSendCosts() {
        sendCosts_.resize(numberOfProcessors_ * numberOfProcessors_, 1U);
        SetSendCostDiagonalToZero();
        isNuma_ = false;
    }

  public:
    /**
     * @brief Default constructor.
     * Initializes a BSP architecture with 2 processors, 1 processor type,
     * communication costs of 1, synchronisation costs of 2, memory bounds of 100,
     * and send costs of 1 between all processors.
     */
    BspArchitecture()
        : numberOfProcessors_(2U), numberOfProcessorTypes_(1U), communicationCosts_(1U), synchronisationCosts_(2U),
          memoryBound_(std::vector<v_memw_t<Graph_t>>(numberOfProcessors_, 100U)), isNuma_(false),
          processorTypes_(std::vector<unsigned>(numberOfProcessors_, 0U)), sendCosts_(numberOfProcessors_ * numberOfProcessors_, 1U) {
        SetSendCostDiagonalToZero();
    }

    BspArchitecture(const BspArchitecture &other) = default;
    BspArchitecture(BspArchitecture &&other) noexcept = default;
    BspArchitecture &operator=(const BspArchitecture &other) = default;
    BspArchitecture &operator=(BspArchitecture &&other) noexcept = default;
    virtual ~BspArchitecture() = default;

    /**
     * @brief Constructs a BspArchitecture object with the specified number of processors, communication cost, and
     * synchronization cost.
     *
     * @param NumberOfProcessors The number of processors in the architecture. Must be greater than 0.
     * @param CommunicationCost The communication cost between processors.
     * @param SynchronisationCost The synchronization cost between processors.
     * @param MemoryBound The memory bound for each processor (default: 100).
     */
    BspArchitecture(const unsigned NumberOfProcessors, const v_commw_t<Graph_t> CommunicationCost, const v_commw_t<Graph_t> SynchronisationCost,
                    const v_memw_t<Graph_t> MemoryBound = 100U)
        : numberOfProcessors_(NumberOfProcessors), numberOfProcessorTypes_(1U), communicationCosts_(CommunicationCost),
          synchronisationCosts_(SynchronisationCost),
          memoryBound_(std::vector<v_memw_t<Graph_t>>(NumberOfProcessors, MemoryBound)), isNuma_(false),
          processorTypes_(std::vector<unsigned>(NumberOfProcessors, 0U)), sendCosts_(numberOfProcessors_ * numberOfProcessors_, 1U) {
        if (NumberOfProcessors == 0U) {
            throw std::runtime_error("BspArchitecture: Number of processors must be greater than 0.");
        }
        SetSendCostDiagonalToZero();
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
          sendCosts_(other.sendCostsVector()) {

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
     * @param NumberOfProcessors The number of processors. Must be greater than 0.
     * @param CommunicationCost The communication cost.
     * @param SynchronisationCost The synchronization cost.
     * @param SendCosts The matrix of send costs between processors. Needs to be a processors x processors matrix. Diagonal entries are forced to zero.
     */
    BspArchitecture(const unsigned NumberOfProcessors, const v_commw_t<Graph_t> CommunicationCost, const v_commw_t<Graph_t> SynchronisationCost,
                    const std::vector<std::vector<v_commw_t<Graph_t>>> &SendCosts)
        : numberOfProcessors_(NumberOfProcessors), numberOfProcessorTypes_(1U), communicationCosts_(CommunicationCost),
          synchronisationCosts_(SynchronisationCost), memoryBound_(std::vector<v_memw_t<Graph_t>>(NumberOfProcessors, 100U)),
          processorTypes_(std::vector<unsigned>(NumberOfProcessors, 0U)) {
        if (NumberOfProcessors == 0U) {
            throw std::runtime_error("BspArchitecture: Number of processors must be greater than 0.");
        }
        if (NumberOfProcessors != SendCosts.size()) {
            throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
        }
        if (std::any_of(SendCosts.begin(), SendCosts.end(),
                        [NumberOfProcessors](const auto &thing) { return thing.size() != NumberOfProcessors; })) {
            throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
        }

        sendCosts_.reserve(NumberOfProcessors * NumberOfProcessors);
        for (const auto &row : SendCosts) {
            sendCosts_.insert(sendCosts_.end(), row.begin(), row.end());
        }

        SetSendCostDiagonalToZero();
        isNuma_ = AreSendCostsNuma();
    }

    /**
     * @brief Constructs a BspArchitecture object with custom send costs and memory bound.
     *
     * @param NumberOfProcessors The number of processors. Must be greater than 0.
     * @param CommunicationCost The communication cost.
     * @param SynchronisationCost The synchronization cost.
     * @param MemoryBound The memory bound for each processor.
     * @param SendCosts The matrix of send costs between processors. Needs to be a processors x processors matrix. Diagonal entries are forced to zero.
     */
    BspArchitecture(const unsigned NumberOfProcessors, const v_commw_t<Graph_t> CommunicationCost, const v_commw_t<Graph_t> SynchronisationCost,
                    const v_memw_t<Graph_t> MemoryBound, const std::vector<std::vector<v_commw_t<Graph_t>>> &SendCosts)
        : numberOfProcessors_(NumberOfProcessors), numberOfProcessorTypes_(1U), communicationCosts_(CommunicationCost),
          synchronisationCosts_(SynchronisationCost),
          memoryBound_(std::vector<v_memw_t<Graph_t>>(NumberOfProcessors, MemoryBound)),
          processorTypes_(std::vector<unsigned>(NumberOfProcessors, 0U)) {
        if (NumberOfProcessors == 0U) {
            throw std::runtime_error("BspArchitecture: Number of processors must be greater than 0.");
        }
        if (NumberOfProcessors != SendCosts.size()) {
            throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
        }
        if (std::any_of(SendCosts.begin(), SendCosts.end(),
                        [NumberOfProcessors](const auto &thing) { return thing.size() != NumberOfProcessors; })) {
            throw std::invalid_argument("sendCosts_ needs to be a processors x processors matrix.\n");
        }

        sendCosts_.reserve(NumberOfProcessors * NumberOfProcessors);
        for (const auto &row : SendCosts) {
            sendCosts_.insert(sendCosts_.end(), row.begin(), row.end());
        }

        SetSendCostDiagonalToZero();
        isNuma_ = AreSendCostsNuma();
    }

    /**
     * @brief Sets the uniform send cost for each pair of processors.
     * The send cost is set to 0 if the processors are the same, and 1 otherwise.
     */
    void SetUniformSendCost() {
        for (unsigned i = 0U; i < numberOfProcessors_; i++) {
            for (unsigned j = 0U; j < numberOfProcessors_; j++) {
                if (i == j) {
                    sendCosts_.at(FlatIndex(i, j)) = 0U;
                } else {
                    sendCosts_.at(FlatIndex(i, j)) = 1U;
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
    void SetExpSendCost(const v_commw_t<Graph_t> base) {
        isNuma_ = true;

        unsigned maxPos = 1;
        constexpr unsigned two = 2;
        for (; intpow(two, maxPos + 1) <= numberOfProcessors_ - 1; ++maxPos) {
        }

        for (unsigned i = 0U; i < numberOfProcessors_; ++i) {
            for (unsigned j = i + 1U; j < numberOfProcessors_; ++j) {
                // Corrected loop to avoid underflow issues with unsigned
                for (int pos = static_cast<int>(maxPos); pos >= 0; --pos) {
                    if (((1U << pos) & i) != ((1U << pos) & j)) {
                        sendCosts_.at(FlatIndex(i, j)) = sendCosts_.at(FlatIndex(j, i)) = intpow(base, static_cast<unsigned>(pos));
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
    [[nodiscard]] auto processors() const { return integral_range<unsigned>(numberOfProcessors_); }

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
        for (unsigned i = 0U; i < numberOfProcessors_; i++) {
            if (vec.at(i).size() != numberOfProcessors_) {
                throw std::invalid_argument("Invalid Argument: Inner vector size mismatch.");
            }

            for (unsigned j = 0U; j < numberOfProcessors_; j++) {
                if (i == j) {
                    if (vec.at(i).at(j) != 0U)
                        throw std::invalid_argument("Invalid Argument: Diagonal elements should be 0.");
                } else {
                    sendCosts_.at(FlatIndex(i, j)) = vec.at(i).at(j);

                    if (numberOfProcessors_ > 1U && vec.at(i).at(j) != vec.at(0U).at(1U)) {
                        isNuma_ = true;
                    }
                }
            }
        }
    }

    /**
     * @brief Sets the send costs between two processors.
     *
     * @param p1 The index of the first processor. Must be less than numberOfProcessors_.
     * @param p2 The index of the second processor. Must be less than numberOfProcessors_.
     * @param cost The cost of sending data between the processors.
     * @throws std::invalid_argument if the processor indices are out of bounds.
     */
    void SetSendCosts(const unsigned p1, const unsigned p2, const v_commw_t<Graph_t> cost) {
        if (p1 >= numberOfProcessors_ || p2 >= numberOfProcessors_) // Fixed condition: p2 >= number_processors
            throw std::invalid_argument("Invalid Argument: Processor index out of bounds.");

        if (p1 != p2) {
            sendCosts_.at(FlatIndex(p1, p2)) = cost;
            isNuma_ = AreSendCostsNuma();
        }
    }

    /**
     * @brief Sets the memory bound for all processors.
     * @param MemoryBound The new memory bound for all processors.
     */
    void setMemoryBound(const v_memw_t<Graph_t> MemoryBound) {
        memoryBound_ = std::vector<v_memw_t<Graph_t>>(numberOfProcessors_, MemoryBound);
    }

    /**
     * @brief Sets the memory bound for all processors using a vector.
     * @param MemoryBound The vector of memory bounds.
     * @throws std::invalid_argument if the size of the vector is invalid.
     */
    void setMemoryBound(const std::vector<v_memw_t<Graph_t>> &MemoryBound) {
        if (MemoryBound.size() != numberOfProcessors_) {
            throw std::invalid_argument("Invalid Argument: Memory bound vector size does not match number of processors.");
        }
        memoryBound_ = MemoryBound;
    }

    /**
     * @brief Sets the memory bound for a specific processor.
     * @param MemoryBound The new memory bound for the processor.
     * @param processorIndex The processor index. Must be less than numberOfProcessors_.
     * @throws std::invalid_argument if the processor index is out of bounds.
     */
    void setMemoryBound(const v_memw_t<Graph_t> MemoryBound, const unsigned processorIndex) {
        if (processorIndex >= numberOfProcessors_) {
            throw std::invalid_argument("Invalid Argument: Processor index out of bounds in setMemoryBound.");
        }
        memoryBound_.at(processorIndex) = MemoryBound;
    }

    /**
     * @brief Sets the synchronization costs.
     * @param SynchCost The new synchronization costs.
     */
    void setSynchronisationCosts(const v_commw_t<Graph_t> SynchCost) { synchronisationCosts_ = SynchCost; }

    /**
     * @brief Sets the communication costs.
     * @param CommCost The new communication costs.
     */
    void setCommunicationCosts(const v_commw_t<Graph_t> CommCost) { communicationCosts_ = CommCost; }

    /**
     * @brief Checks if the architecture is NUMA.
     * @return True if NUMA, false otherwise.
     */
    [[nodiscard]] bool isNumaArchitecture() const { return isNuma_; }

    /**
     * @brief Sets the number of processors. Processor type is set to 0 for all processors.
     * Resets send costs to uniform (1) and diagonal to 0. The memory bound is set to 100 for all processors.
     * @param numberOfProcessors The number of processors. Must be greater than 0.
     * @throws std::invalid_argument if the number of processors is 0.
     */
    void setNumberOfProcessors(const unsigned numberOfProcessors) {
        if (numberOfProcessors == 0) {
            throw std::invalid_argument("Invalid Argument: Number of processors must be greater than 0.");
        }
        numberOfProcessors_ = numberOfProcessors;
        numberOfProcessorTypes_ = 1U;
        processorTypes_ = std::vector<unsigned>(numberOfProcessors_, 0U);

        InitializeUniformSendCosts();

        // initialize memory bound to 100 for all processors
        memoryBound_.resize(numberOfProcessors_, 100U);
    }

    /**
     * @brief Sets the number of processors and their types. Number of processors is set to the size of the processor types vector.
     * Resets send costs to uniform (1). Resets memory bound to 100 for all processors.
     * @param processorTypes The types of the respective processors.
     */
    void setProcessorsWithTypes(const std::vector<v_type_t<Graph_t>> &processorTypes) {
        if (processorTypes.empty()) {
            throw std::invalid_argument("Invalid Argument: Processor types vector is empty.");
        }
        if (processorTypes.size() > std::numeric_limits<unsigned>::max()) {
            throw std::invalid_argument("Invalid Argument: Number of processors exceeds the limit.");
        }
        numberOfProcessors_ = static_cast<unsigned>(processorTypes.size());
        processorTypes_ = processorTypes;

        InitializeUniformSendCosts();

        // initialize memory bound to 100 for all processors
        memoryBound_.resize(numberOfProcessors_, 100U);
        UpdateNumberOfProcessorTypes();
    }

    /**
     * @brief Sets processors based on counts of consecutive types.
     * The architecture will have processorTypeCount[0] processors of type 0, processorTypeCount[1] processors of type 1, etc.
     * The memory bound for each processor of type i is set to processorTypeMemory[i].
     * The send costs are set to uniform (1).
     * @param processorTypeCount Vector where index is type and value is count of processors of that type.
     * @param processorTypeMemory Vector where index is type and value is memory bound for that type.
     */
    void SetProcessorsConsequTypes(const std::vector<v_type_t<Graph_t>> &processorTypeCount,
                                   const std::vector<v_memw_t<Graph_t>> &processorTypeMemory) {
        if (processorTypeCount.size() != processorTypeMemory.size()) {
            throw std::invalid_argument("Invalid Argument: processorTypeCount and processorTypeMemory must have the same size.");
        }

        if (processorTypeCount.size() > std::numeric_limits<unsigned>::max()) {
            throw std::invalid_argument("Invalid Argument: Number of processors exceeds the limit.");
        }

        numberOfProcessorTypes_ = static_cast<unsigned>(processorTypeCount.size());
        numberOfProcessors_ = std::accumulate(processorTypeCount.begin(), processorTypeCount.end(), 0U);

        // initialize processor types and memory bound
        processorTypes_ = std::vector<v_type_t<Graph_t>>(numberOfProcessors_, 0U);
        memoryBound_ = std::vector<v_memw_t<Graph_t>>(numberOfProcessors_, 0U);

        unsigned offset = 0U;
        for (unsigned i = 0U; i < processorTypeCount.size(); i++) {
            for (unsigned j = 0U; j < processorTypeCount.at(i); j++) {
                processorTypes_.at(offset + j) = i;
                memoryBound_.at(offset + j) = processorTypeMemory.at(i);
            }
            offset += processorTypeCount.at(i);
        }

        InitializeUniformSendCosts();
    }

    /**
     * @brief Returns the memory bounds of all processors.
     * @return Vector of memory bounds.
     */
    [[nodiscard]] const std::vector<v_memw_t<Graph_t>> &memoryBound() const { return memoryBound_; }

    /**
     * @brief Returns the memory bound of a specific processor.
     * @param proc The processor index.
     * @return The memory bound.
     */
    [[nodiscard]] v_memw_t<Graph_t> memoryBound(const unsigned proc) const { return memoryBound_.at(proc); }

    /**
     * @brief Returns the maximum memory bound over all processors.
     * @return The maximum memory bound.
     */
    [[nodiscard]] v_memw_t<Graph_t> maxMemoryBound() const { return *(std::max_element(memoryBound_.begin(), memoryBound_.end())); }

    /**
     * @brief Returns the maximum memory bound over all processors of a specific type.
     *
     * @param procType The processor type.
     * @return The maximum memory bound.
     */
    [[nodiscard]] v_memw_t<Graph_t> maxMemoryBoundProcType(const v_type_t<Graph_t> procType) const {
        v_memw_t<Graph_t> max_mem = 0U;
        for (unsigned proc = 0U; proc < numberOfProcessors_; proc++) {
            if (processorTypes_.at(proc) == procType) {
                max_mem = std::max(max_mem, memoryBound_.at(proc));
            }
        }
        return max_mem;
    }

    /**
     * @brief Returns the number of processors.
     * @return The number of processors.
     */
    [[nodiscard]] unsigned numberOfProcessors() const { return numberOfProcessors_; }

    /**
     * @brief Returns the communication costs.
     * @return The communication costs.
     */
    [[nodiscard]] v_commw_t<Graph_t> communicationCosts() const { return communicationCosts_; }

    /**
     * @brief Returns the synchronization costs.
     * @return The synchronization costs.
     */
    [[nodiscard]] v_commw_t<Graph_t> synchronisationCosts() const { return synchronisationCosts_; }

    /**
     * @brief Returns the send costs matrix.
     * @return The send costs matrix.
     */
    [[nodiscard]] std::vector<std::vector<v_commw_t<Graph_t>>> sendCostMatrix() const {
        std::vector<std::vector<v_commw_t<Graph_t>>> matrix(numberOfProcessors_, std::vector<v_commw_t<Graph_t>>(numberOfProcessors_));
        for (unsigned i = 0; i < numberOfProcessors_; ++i) {
            for (unsigned j = 0; j < numberOfProcessors_; ++j) {
                matrix[i][j] = sendCosts_[FlatIndex(i, j)];
            }
        }
        return matrix;
    }

    /**
     * @brief Returns the flattened send costs vector.
     * @return The send costs vector.
     */
    [[nodiscard]] const std::vector<v_commw_t<Graph_t>> &sendCostsVector() const { return sendCosts_; }

    /**
     * @brief Returns the processor types.
     * @return Vector of processor types.
     */
    [[nodiscard]] const std::vector<unsigned> &processorTypes() const { return processorTypes_; }

    /**
     * @brief Returns the communication costs between two processors.
     * The communication costs are the send costs multiplied by the communication costs factor.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     * @return The communication costs between the two processors.
     */
    [[nodiscard]] v_commw_t<Graph_t> communicationCosts(const unsigned p1, const unsigned p2) const {
        return communicationCosts_ * sendCosts_.at(FlatIndex(p1, p2));
    }

    /**
     * @brief Returns the send costs between two processors.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     * @return The send costs between the two processors.
     */
    [[nodiscard]] v_commw_t<Graph_t> sendCosts(const unsigned p1, const unsigned p2) const { return sendCosts_.at(FlatIndex(p1, p2)); }

    /**
     * @brief Returns the send costs matrix.
     * @return The send costs matrix.
     */
    [[nodiscard]] std::vector<std::vector<v_commw_t<Graph_t>>> sendCosts() const { return sendCostMatrix(); }

    /**
     * @brief Returns the type of a specific processor.
     * @param p1 The processor index.
     * @return The processor type.
     */
    [[nodiscard]] v_type_t<Graph_t> processorType(const unsigned p1) const { return processorTypes_.at(p1); }

    /**
     * @brief Sets the type of a specific processor.
     * @param p1 The processor index.
     * @param type The new processor type.
     */
    void setProcessorType(const unsigned p1, const v_type_t<Graph_t> type) {
        if (p1 >= numberOfProcessors_)
            throw std::invalid_argument("Invalid Argument: Processor index out of bounds.");

        processorTypes_.at(p1) = type;
        numberOfProcessorTypes_ = std::max(numberOfProcessorTypes_, type + 1U);
    }

    /**
     * @brief Returns the count of processors for each type.
     * @return Vector where index is type and value is count.
     */
    [[nodiscard]] std::vector<unsigned> getProcessorTypeCount() const {
        std::vector<unsigned> type_count(numberOfProcessorTypes_, 0U);
        for (unsigned p = 0U; p < numberOfProcessors_; p++) {
            type_count[processorTypes_.at(p)]++;
        }
        return type_count;
    }

    /**
     * @brief Prints the architecture details to the output stream.
     * @param os The output stream.
     */
    void print(std::ostream &os) const {
        os << "Architecture info:  number of processors: " << numberOfProcessors_
           << ", Number of processor types: " << numberOfProcessorTypes_
           << ", Communication costs: " << communicationCosts_ << ", Synchronization costs: " << synchronisationCosts_
           << "\n";
        os << std::setw(17) << " Processor: ";
        for (unsigned i = 0U; i < numberOfProcessors_; i++) {
            os << std::right << std::setw(5) << i << " ";
        }
        os << "\n";
        os << std::setw(17) << "Processor type: ";
        for (unsigned i = 0U; i < numberOfProcessors_; i++) {
            os << std::right << std::setw(5) << processorTypes_.at(i) << " ";
        }
        os << "\n";
        os << std::setw(17) << "Memory bound: ";
        for (unsigned i = 0U; i < numberOfProcessors_; i++) {
            os << std::right << std::setw(5) << memoryBound_.at(i) << " ";
        }
        os << "\n";
    }

    [[nodiscard]] unsigned getNumberOfProcessorTypes() const { return numberOfProcessorTypes_; };

    [[nodiscard]] MEMORY_CONSTRAINT_TYPE getMemoryConstraintType() const { return memoryConstraintType_; }
    void setMemoryConstraintType(const MEMORY_CONSTRAINT_TYPE memoryConstraintType) {
        memoryConstraintType_ = memoryConstraintType;
    }
};

} // namespace osp