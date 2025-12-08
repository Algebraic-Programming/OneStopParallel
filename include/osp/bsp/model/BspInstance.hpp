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

enum class RETURN_STATUS { OSP_SUCCESS,
                           BEST_FOUND,
                           TIMEOUT,
                           ERROR };

/**
 * @brief Converts the enum to a string literal.
 * Returns const char* to avoid std::string allocation overhead.
 */
inline const char *to_string(const RETURN_STATUS status) {
    switch (status) {
    case RETURN_STATUS::OSP_SUCCESS:
        return "SUCCESS";
    case RETURN_STATUS::BEST_FOUND:
        return "BEST FOUND";
    case RETURN_STATUS::TIMEOUT:
        return "TIMEOUT";
    case RETURN_STATUS::ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Stream operator overload using the helper function.
 */
inline std::ostream &operator<<(std::ostream &os, RETURN_STATUS status) {
    return os << to_string(status);
}

/**
 * @class BspInstance
 * @brief Represents an instance of the BSP (Bulk Synchronous Parallel) model.
 *
 * The BspInstance class encapsulates the computational DAG (Directed Acyclic Graph) and the BSP architecture
 * for a specific instance of the BSP model. It provides methods to access and modify the architecture and DAG,
 * as well as retrieve information about the instance such as the number of vertices and processors.
 */
template<typename Graph_t>
class BspInstance {

    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");

  private:
    Graph_t cdag;
    BspArchitecture<Graph_t> architecture;

    // for problem instances with heterogeneity
    std::vector<std::vector<bool>> nodeProcessorCompatibility = std::vector<std::vector<bool>>({{true}});

    /**
     * @brief Calculates the maximum memory bound for each processor type.
     *
     * @return A vector where the index corresponds to the processor type and the value is the maximum memory bound for that type.
     */
    std::vector<v_memw_t<Graph_t>> calculateMaxMemoryPerProcessorType() const {
        std::vector<v_memw_t<Graph_t>> max_memory_per_proc_type(architecture.getNumberOfProcessorTypes(), 0);
        for (unsigned proc = 0; proc < architecture.numberOfProcessors(); proc++) {
            max_memory_per_proc_type[architecture.processorType(proc)] =
                std::max(max_memory_per_proc_type[architecture.processorType(proc)], architecture.memoryBound(proc));
        }
        return max_memory_per_proc_type;
    }

  public:
    /**
     * @brief Default constructor for the BspInstance class.
     */
    BspInstance() = default;

    /**
     * @brief Constructs a BspInstance object with the specified computational DAG and BSP architecture.
     *
     * @param cdag The computational DAG for the instance.
     * @param architecture The BSP architecture for the instance.
     */
    BspInstance(const Graph_t &cdag_, const BspArchitecture<Graph_t> &architecture_,
                std::vector<std::vector<bool>> nodeProcessorCompatibility_ = std::vector<std::vector<bool>>({{true}}))
        : cdag(cdag_), architecture(architecture_), nodeProcessorCompatibility(nodeProcessorCompatibility_) {}

    /**
     * @brief Constructs a BspInstance object with the specified computational DAG and BSP architecture.
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
     * @brief Returns a reference to the BSP architecture for the instance.
     *
     * @return A reference to the BSP architecture for the instance.
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
     * @brief Returns a reference to the computational DAG for the instance.
     *
     * @return A reference to the computational DAG for the instance.
     */
    [[nodiscard]] const Graph_t &getComputationalDag() const { return cdag; }
    [[nodiscard]] Graph_t &getComputationalDag() { return cdag; }

    /**
     * @brief Returns the number of vertices in the computational DAG.
     *
     * @return The number of vertices.
     */
    [[nodiscard]] vertex_idx_t<Graph_t> numberOfVertices() const { return cdag.num_vertices(); }

    /**
     * @brief Returns a view over the vertex indices of the computational DAG.
     * @return A view over the vertex indices.
     */
    [[nodiscard]] auto vertices() const { return cdag.vertices(); }

    /**
     * @brief Returns a view over the processor indices of the BSP architecture.
     * @return A view over the processor indices.
     */
    [[nodiscard]] auto processors() const { return architecture.processors(); }

    /**
     * @brief Returns the number of processors in the BSP architecture.
     * @return The number of processors in the BSP architecture.
     */
    [[nodiscard]] unsigned numberOfProcessors() const { return architecture.numberOfProcessors(); }

    /**
     * @brief Returns the communication costs between two processors.
     * The communication costs are the send costs multiplied by the communication costs.
     *
     * @param p_send The index of the sending processor.
     * @param p_receive The index of the receiving processor.
     *
     * @return The communication costs between the two processors.
     */
    [[nodiscard]] v_commw_t<Graph_t> communicationCosts(const unsigned p_send, const unsigned p_receive) const {
        return architecture.communicationCosts(p_send, p_receive);
    }

    /**
     * @brief Returns the send costs between two processors.
     *
     * @param p_send The index of the sending processor.
     * @param p_receive The index of the receiving processor.
     *
     * @return The send costs between the two processors.
     */
    [[nodiscard]] v_commw_t<Graph_t> sendCosts(const unsigned p_send, const unsigned p_receive) const {
        return architecture.sendCosts(p_send, p_receive);
    }

    /**
     * @brief Returns a copy of the send costs matrix.
     * @return A copy of the send costs matrix.
     */
    [[nodiscard]] std::vector<std::vector<v_commw_t<Graph_t>>> sendCosts() const { return architecture.sendCosts(); }

    /**
     * @brief Returns the flattened send costs vector.
     *
     * @return The flattened send costs vector.
     */
    [[nodiscard]] const std::vector<v_commw_t<Graph_t>> &sendCostsVector() const {
        return architecture.sendCostsVector();
    }

    /**
     * @brief Returns the communication costs of the BSP architecture.
     *
     * @return The communication costs as an unsigned integer.
     */
    [[nodiscard]] v_commw_t<Graph_t> communicationCosts() const { return architecture.communicationCosts(); }

    /**
     * @brief Returns the synchronization costs of the BSP architecture.
     *
     * @return The synchronization costs as an unsigned integer.
     */
    [[nodiscard]] v_commw_t<Graph_t> synchronisationCosts() const { return architecture.synchronisationCosts(); }

    /**
     * @brief Returns whether the architecture is NUMA.
     *
     * @return True if the architecture is NUMA, false otherwise.
     */
    [[nodiscard]] bool isNumaInstance() const { return architecture.isNumaArchitecture(); }

    /**
     * @brief Returns the memory bound for a specific processor.
     *
     * @param proc The processor index.
     * @return The memory bound for the processor.
     */
    [[nodiscard]] v_memw_t<Graph_t> memoryBound(const unsigned proc) const { return architecture.memoryBound(proc); }

    /**
     * @brief Returns the maximum memory bound for a specific processor type.
     *
     * @param procType The processor type.
     * @return The maximum memory bound for the processor type.
     */
    [[nodiscard]] v_memw_t<Graph_t> maxMemoryBoundProcType(const unsigned procType) const {
        return architecture.maxMemoryBoundProcType(procType);
    }

    /**
     * @brief Returns the maximum memory bound for a specific node type.
     *
     * This considers all compatible processor types for the given node type.
     *
     * @param nodeType The node type.
     * @return The maximum memory bound for the node type.
     */
    [[nodiscard]] v_memw_t<Graph_t> maxMemoryBoundNodeType(const unsigned nodeType) const {
        int max_mem = 0;
        for (unsigned proc = 0; proc < architecture.getNumberOfProcessorTypes(); proc++) {
            if (isCompatibleType(nodeType, architecture.processorType(proc))) {
                max_mem = std::max(max_mem, architecture.memoryBound(proc));
            }
        }
        return max_mem;
    }

    /**
     * @brief Sets the communication costs of the BSP architecture.
     *
     * @param cost The communication costs to set.
     */
    void setCommunicationCosts(const v_commw_t<Graph_t> cost) { architecture.setCommunicationCosts(cost); }

    /**
     * @brief Sets the synchronisation costs of the BSP architecture.
     *
     * @param cost The synchronisation costs to set.
     */
    void setSynchronisationCosts(const v_commw_t<Graph_t> cost) { architecture.setSynchronisationCosts(cost); }

    /**
     * @brief Sets the number of processors in the BSP architecture.
     *
     * @param num The number of processors to set.
     */
    void setNumberOfProcessors(const unsigned num) { architecture.setNumberOfProcessors(num); }

    /**
     * @brief Checks if the memory constraints are feasible for the given instance.
     *
     * @return True if the memory constraints are feasible, false otherwise.
     */
    [[nodiscard]] bool CheckMemoryConstraintsFeasibility() const {
        const auto max_memory_per_proc_type = calculateMaxMemoryPerProcessorType();

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
     * @brief Adjusts the memory constraints of the architecture to ensure feasibility.
     *
     * If a node type requires more memory than available on any compatible processor type,
     * the memory bound of compatible processors is increased.
     */
    void adjust_memory_constraints() {
        const auto max_memory_per_proc_type = calculateMaxMemoryPerProcessorType();

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

            if (!fits) {
                std::cout << "Warning: Computational DAG memory weight exceeds architecture memory bound." << std::endl;
                std::cout << "VertexType " << vertType << " has memory "
                          << " and exceeds compatible processor types memory limit." << std::endl;

                for (unsigned proc = 0; proc < architecture.numberOfProcessors(); proc++) {
                    if (isCompatibleType(vertType, architecture.processorType(proc))) {
                        std::cout << "Increasing memory of processor " << proc << " of type "
                                  << architecture.processorType(proc) << " to " << max_memory_of_type << "."
                                  << std::endl;
                        architecture.setMemoryBound(max_memory_of_type, proc);
                    }
                }
            }
        }
    }

    /**
     * @brief Returns the processor type for a given processor index.
     *
     * @param proc The processor index.
     * @return The processor type.
     */
    [[nodiscard]] v_type_t<Graph_t> processorType(const unsigned proc) const { return architecture.processorType(proc); }

    /**
     * @brief Checks if a node is compatible with a processor.
     *
     * @param node The node index.
     * @param processor_id The processor index.
     * @return True if the node is compatible with the processor, false otherwise.
     */
    [[nodiscard]] bool isCompatible(const vertex_idx_t<Graph_t> &node, const unsigned processor_id) const {
        return isCompatibleType(cdag.vertex_type(node), architecture.processorType(processor_id));
    }

    /**
     * @brief Checks if a node type is compatible with a processor type.
     *
     * @param nodeType The node type.
     * @param processorType The processor type.
     * @return True if the node type is compatible with the processor type, false otherwise.
     */
    [[nodiscard]] bool isCompatibleType(const v_type_t<Graph_t> nodeType, const v_type_t<Graph_t> processorType) const {
        return nodeProcessorCompatibility[nodeType][processorType];
    }

    /**
     * @brief Sets the node-processor compatibility matrix.
     *
     * @param compatibility_ The compatibility matrix.
     */
    void setNodeProcessorCompatibility(const std::vector<std::vector<bool>> &compatibility_) {
        nodeProcessorCompatibility = compatibility_;
    }

    /**
     * @brief Returns the node-processor compatibility matrix.
     *
     * @return The node-processor compatibility matrix.
     */
    [[nodiscard]] const std::vector<std::vector<bool>> &getProcessorCompatibilityMatrix() const { return nodeProcessorCompatibility; }

    /**
     * @brief Sets the compatibility matrix to be diagonal.
     *
     * This implies that node type `i` is only compatible with processor type `i`.
     *
     * @param number_of_types The number of types.
     */
    void setDiagonalCompatibilityMatrix(const unsigned number_of_types) {
        nodeProcessorCompatibility =
            std::vector<std::vector<bool>>(number_of_types, std::vector<bool>(number_of_types, false));
        for (unsigned i = 0; i < number_of_types; ++i)
            nodeProcessorCompatibility[i][i] = true;
    }

    /**
     * @brief Sets the compatibility matrix to all ones.
     *
     * This implies that all node types are compatible with all processor types.
     */
    void setAllOnesCompatibilityMatrix() {
        unsigned number_of_node_types = cdag.num_vertex_types();
        unsigned number_of_proc_types = architecture.getNumberOfProcessorTypes();

        nodeProcessorCompatibility =
            std::vector<std::vector<bool>>(number_of_node_types, std::vector<bool>(number_of_proc_types, true));
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
     * @brief Returns a compatibility matrix between node types.
     *
     * Two node types are compatible if they share at least one compatible processor type.
     *
     * @return A matrix where `[i][j]` is true if node type `i` and node type `j` are compatible.
     */
    [[nodiscard]] std::vector<std::vector<bool>> getNodeNodeCompatabilityMatrix() const {
        std::vector<std::vector<bool>> compMat(cdag.num_vertex_types(),
                                               std::vector<bool>(cdag.num_vertex_types(), false));
        for (unsigned nodeType1 = 0; nodeType1 < cdag.num_vertex_types(); nodeType1++) {
            for (unsigned nodeType2 = 0; nodeType2 < cdag.num_vertex_types(); nodeType2++) {
                for (unsigned procType = 0; procType < architecture.getNumberOfProcessorTypes(); procType++) {
                    if (isCompatibleType(nodeType1, procType) && isCompatibleType(nodeType2, procType)) {
                        compMat[nodeType1][nodeType2] = true;
                        break;
                    }
                }
            }
        }
        return compMat;
    }

    /**
     * @brief Returns the node-processor compatibility matrix.
     *
     * @return The node-processor compatibility matrix.
     */
    [[nodiscard]] const std::vector<std::vector<bool>> &getNodeProcessorCompatibilityMatrix() const {
        return nodeProcessorCompatibility;
    }
};

} // namespace osp