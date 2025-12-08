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

#include <iostream>

#include "BspArchitecture.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"

namespace osp {

/**
 * @class BspInstance
 * @brief Represents an instance of the BSP (Bulk Synchronous Parallel) model.
 *
 * The BspInstance class encapsulates the computational DAG (Directed Acyclic Graph) and the BSP architecture
 * for a specific instance of the BSP model. It provides methods to access and modify the architecture and DAG,
 * as well as retrieve information about the instance such as the number of vertices and processors.
 *
 * The instance specifies the compatibility between node types and processor types.
 *
 * @tparam Graph_t The type of the computational DAG.
 */
template<typename Graph_t>
class BspInstance {
    static_assert(is_computational_dag_v<Graph_t>, "BspInstance can only be used with computational DAGs.");

  private:
    /**
     * @brief  The computational DAG of the instance. Holds the graph structure and the node types, work, memory, communication weights.
     */
    Graph_t cdag;
    /**
     * @brief The BSP architecture of the instance. Holds the processor types and the memory bounds. Communication and synchronization cost. And the send cost between processors.
     */
    BspArchitecture<Graph_t> architecture;

    /**
     * @brief Stores the compatibility between node types and processor types.
     *
     * The architecture defines a type for each processor, and the dag defines a type for each node.
     * This matrix stores for each node type and processor type whether they are compatible, i.e.,
     * if a node of the can be assigned to a processor of the given type in a schedule.
     * @note The outer vector is indexed by node type, the inner vector is indexed by processor type.
     */
    std::vector<std::vector<bool>> nodeProcessorCompatibility = std::vector<std::vector<bool>>({{true}});

  public:
    /**
     * @brief Default constructor for the BspInstance class.
     */
    BspInstance() = default;

    /**
     * @brief Constructs a BspInstance object with the specified computational DAG and BSP architecture.
     * Computational DAG and BSP architecture are copied!
     *
     * @param cdag The computational DAG for the instance.
     * @param architecture The BSP architecture for the instance.
     */
    BspInstance(const Graph_t &cdag_, const BspArchitecture<Graph_t> &architecture_,
                std::vector<std::vector<bool>> nodeProcessorCompatibility_ = std::vector<std::vector<bool>>({{true}}))
        : cdag(cdag_), architecture(architecture_), nodeProcessorCompatibility(nodeProcessorCompatibility_) {}

    /**
     * @brief Constructs a BspInstance object with the specified computational DAG and BSP architecture.
     * Computational DAG and BSP architecture are moved!
     *
     * @param cdag The computational DAG for the instance.
     * @param architecture The BSP architecture for the instance.
     */
    BspInstance(Graph_t &&cdag_, BspArchitecture<Graph_t> &&architecture_,
                std::vector<std::vector<bool>> nodeProcessorCompatibility_ = std::vector<std::vector<bool>>({{true}}))
        : cdag(std::move(cdag_)), architecture(std::move(architecture_)), nodeProcessorCompatibility(nodeProcessorCompatibility_) {
    }

    template<typename Graph_t_other>
    explicit BspInstance(const BspInstance<Graph_t_other> &other)
        : architecture(other.getArchitecture()),
          nodeProcessorCompatibility(other.getNodeProcessorCompatibilityMatrix()) {
        constructComputationalDag(other.getComputationalDag(), cdag);
    }

    BspInstance(const BspInstance<Graph_t> &other) = default;
    BspInstance(BspInstance<Graph_t> &&other) noexcept = default;

    BspInstance<Graph_t> &operator=(const BspInstance<Graph_t> &other) = default;
    BspInstance<Graph_t> &operator=(BspInstance<Graph_t> &&other) noexcept = default;

    /**
     * @brief Returns a reference to the BSP architecture of the instance.
     */
    [[nodiscard]] const BspArchitecture<Graph_t> &getArchitecture() const { return architecture; }
    [[nodiscard]] BspArchitecture<Graph_t> &getArchitecture() { return architecture; }

    /**
     * @brief Sets the BSP architecture for the instance.
     *
     * @param architecture_ The BSP architecture for the instance.
     */
    void setArchitecture(const BspArchitecture<Graph_t> &architechture_) { architecture = architechture_; }

    /**
     * @brief Returns a reference to the computational DAG of the instance.
     */
    [[nodiscard]] const Graph_t &getComputationalDag() const { return cdag; }
    [[nodiscard]] Graph_t &getComputationalDag() { return cdag; }

    /**
     * @brief Returns the number of vertices in the computational DAG.
     */
    [[nodiscard]] vertex_idx_t<Graph_t> numberOfVertices() const { return cdag.num_vertices(); }

    /**
     * @brief Returns a view over the vertex indices of the computational DAG.
     */
    [[nodiscard]] auto vertices() const { return cdag.vertices(); }

    /**
     * @brief Returns a view over the processor indices of the BSP architecture.
     */
    [[nodiscard]] auto processors() const { return architecture.processors(); }

    /**
     * @brief Returns the number of processors in the BSP architecture.
     */
    [[nodiscard]] unsigned numberOfProcessors() const { return architecture.numberOfProcessors(); }

    /**
     * @brief Returns the communication costs between two processors. Does not perform bounds checking.
     * The communication costs are the send costs multiplied by the communication costs.
     *
     * @param p_send The index of the sending processor.
     * @param p_receive The index of the receiving processor.
     */
    [[nodiscard]] v_commw_t<Graph_t> communicationCosts(const unsigned p_send, const unsigned p_receive) const {
        return architecture.communicationCosts(p_send, p_receive);
    }

    /**
     * @brief Returns the send costs between two processors. Does not perform bounds checking.
     * Does not the communication costs into account.
     *
     * @param p_send The index of the sending processor.
     * @param p_receive The index of the receiving processor.
     */
    [[nodiscard]] v_commw_t<Graph_t> sendCosts(const unsigned p_send, const unsigned p_receive) const {
        return architecture.sendCosts(p_send, p_receive);
    }

    /**
     * @brief Returns a copy of the send costs matrix.
     */
    [[nodiscard]] std::vector<std::vector<v_commw_t<Graph_t>>> sendCosts() const { return architecture.sendCosts(); }

    /**
     * @brief Returns the flattened send costs vector.
     */
    [[nodiscard]] const std::vector<v_commw_t<Graph_t>> &sendCostsVector() const {
        return architecture.sendCostsVector();
    }

    /**
     * @brief Returns the communication costs of the BSP architecture.
     */
    [[nodiscard]] v_commw_t<Graph_t> communicationCosts() const { return architecture.communicationCosts(); }

    /**
     * @brief Returns the synchronization costs of the BSP architecture.
     */
    [[nodiscard]] v_commw_t<Graph_t> synchronisationCosts() const { return architecture.synchronisationCosts(); }

    /**
     * @brief Returns the memory bound for a specific processor.
     *
     * @param proc The processor index.
     */
    [[nodiscard]] v_memw_t<Graph_t> memoryBound(const unsigned proc) const { return architecture.memoryBound(proc); }

    /**
     * @brief Sets the communication costs of the BSP architecture.
     * @param cost The communication costs to set.
     */
    void setCommunicationCosts(const v_commw_t<Graph_t> cost) { architecture.setCommunicationCosts(cost); }

    /**
     * @brief Sets the synchronisation costs of the BSP architecture.
     * @param cost The synchronisation costs to set.
     */
    void setSynchronisationCosts(const v_commw_t<Graph_t> cost) { architecture.setSynchronisationCosts(cost); }

    /**
     * @brief Sets the number of processors. Processor type is set to 0 for all processors.
     * Resets send costs to uniform (1) and diagonal to 0. The memory bound is set to 100 for all processors.
     * @param numberOfProcessors The number of processors. Must be greater than 0.
     * @throws std::invalid_argument if the number of processors is 0.
     */
    void setNumberOfProcessors(const unsigned num) { architecture.setNumberOfProcessors(num); }

    /**
     * @brief Returns false if there is a node whose weight does not fit on any of its compatible processors.
     * @return True if the memory constraints are feasible, false otherwise.
     */
    [[nodiscard]] bool CheckMemoryConstraintsFeasibility() const {
        std::vector<v_memw_t<Graph_t>> max_memory_per_proc_type(architecture.getNumberOfProcessorTypes(), 0);
        for (unsigned proc = 0; proc < architecture.numberOfProcessors(); proc++) {
            max_memory_per_proc_type[architecture.processorType(proc)] =
                std::max(max_memory_per_proc_type[architecture.processorType(proc)], architecture.memoryBound(proc));
        }

        for (unsigned vertType = 0; vertType < cdag.num_vertex_types(); vertType++) {
            v_memw_t<Graph_t> max_memory_of_type = max_memory_weight(vertType, cdag);
            bool fits = false;

            for (unsigned proc_type = 0; proc_type < architecture.getNumberOfProcessorTypes(); proc_type++) {
                if (isCompatibleType(vertType, proc_type)) {
                    fits = fits | (max_memory_of_type <= max_memory_per_proc_type[proc_type]);
                    if (fits)
                        break;
                }
            }

            if (!fits)
                return false;
        }

        return true;
    }

    /**
     * @brief Returns the processor type for a given processor index. Does not perform bounds checking.
     * @param proc The processor index.
     */
    [[nodiscard]] v_type_t<Graph_t> processorType(const unsigned proc) const { return architecture.processorType(proc); }

    /**
     * @brief Checks if a node is compatible with a processor. Does not perform bounds checking.
     *
     * @param node The node index.
     * @param processor_id The processor index.
     * @return True if the node is compatible with the processor, false otherwise.
     */
    [[nodiscard]] bool isCompatible(const vertex_idx_t<Graph_t> &node, const unsigned processor_id) const {
        return isCompatibleType(cdag.vertex_type(node), architecture.processorType(processor_id));
    }

    /**
     * @brief Checks if a node type is compatible with a processor type. Does not perform bounds checking.
     *
     * @param nodeType The node type.
     * @param processorType The processor type.
     * @return True if the node type is compatible with the processor type, false otherwise.
     */
    [[nodiscard]] bool isCompatibleType(const v_type_t<Graph_t> nodeType, const v_type_t<Graph_t> processorType) const {
        return nodeProcessorCompatibility[nodeType][processorType];
    }

    /**
     * @brief Sets the node-processor compatibility matrix. The matrix is copied.
     * @param compatibility_ The compatibility matrix.
     * @throw std::runtime_error if the compatibility matrix size does not match the number of node types and processor types.
     */
    void setNodeProcessorCompatibility(const std::vector<std::vector<bool>> &compatibility_) {
        if (compatibility_.size() < cdag.num_vertex_types() || compatibility_[0].size() < architecture.getNumberOfProcessorTypes()) {
            throw std::runtime_error("Compatibility matrix size does not match the number of node types and processor types.");
        }
        nodeProcessorCompatibility = compatibility_;
    }

    /**
     * @brief Returns the node type - processor type compatibility matrix.
     */
    [[nodiscard]] const std::vector<std::vector<bool>> &getProcessorCompatibilityMatrix() const { return nodeProcessorCompatibility; }

    /**
     * @brief Sets the compatibility matrix to be diagonal. This implies that node type `i` is only compatible with processor type `i`.
     * @param number_of_types The number of types.
     */
    void setDiagonalCompatibilityMatrix(const unsigned number_of_types) {
        nodeProcessorCompatibility.assign(number_of_types, std::vector<bool>(number_of_types, false));
        for (unsigned i = 0; i < number_of_types; ++i)
            nodeProcessorCompatibility[i][i] = true;
    }

    /**
     * @brief Sets the compatibility matrix to all ones. This implies that all node types are compatible with all processor types.
     */
    void setAllOnesCompatibilityMatrix() {
        nodeProcessorCompatibility.assign(cdag.num_vertex_types(), std::vector<bool>(architecture.getNumberOfProcessorTypes(), true));
    }

    /**
     * @brief Returns a list of compatible processor types for each node type.
     *
     * @return A vector where the index is the node type and the value is a vector of compatible processor types.
     */
    [[nodiscard]] std::vector<std::vector<unsigned>> getProcTypesCompatibleWithNodeType() const {
        unsigned numberOfNodeTypes = cdag.num_vertex_types();
        unsigned numberOfProcTypes = architecture.getNumberOfProcessorTypes();
        std::vector<std::vector<unsigned>> compatibleProcTypes(numberOfNodeTypes);

        for (unsigned nodeType = 0; nodeType < numberOfNodeTypes; ++nodeType)
            for (unsigned processorType = 0; processorType < numberOfProcTypes; ++processorType)
                if (isCompatibleType(nodeType, processorType))
                    compatibleProcTypes[nodeType].push_back(processorType);

        return compatibleProcTypes;
    }

    /**
     * @brief Returns the node-processor compatibility matrix.
     */
    [[nodiscard]] const std::vector<std::vector<bool>> &getNodeProcessorCompatibilityMatrix() const {
        return nodeProcessorCompatibility;
    }
};

} // namespace osp