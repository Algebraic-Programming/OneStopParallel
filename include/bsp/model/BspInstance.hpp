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
#include "concepts/computational_dag_concept.hpp"
#include "graph_algorithms/computational_dag_util.hpp"

namespace osp {

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
    BspInstance(Graph_t cdag, BspArchitecture<Graph_t> architecture_,
                std::vector<std::vector<bool>> nodeProcessorCompatibility_ = std::vector<std::vector<bool>>({{true}}))
        : cdag(cdag), architecture(architecture_), nodeProcessorCompatibility(nodeProcessorCompatibility_) {}

    /**
     * @brief Returns a reference to the BSP architecture for the instance.
     *
     * @return A reference to the BSP architecture for the instance.
     */
    inline const BspArchitecture<Graph_t> &getArchitecture() const { return architecture; }

    /**
     * @brief Returns a reference to the BSP architecture for the instance.
     *
     * @return A reference to the BSP architecture for the instance.
     */
    inline BspArchitecture<Graph_t> &getArchitecture() { return architecture; }

    /**
     * @brief Sets the BSP architecture for the instance.
     *
     * @param architecture_ The BSP architecture for the instance.
     */
    inline void setArchitecture(const BspArchitecture<Graph_t> &architechture_) { architecture = architechture_; }

    /**
     * @brief Returns a reference to the computational DAG for the instance.
     *
     * @return A reference to the computational DAG for the instance.
     */
    inline const Graph_t &getComputationalDag() const { return cdag; }

    /**
     * @brief Returns a reference to the computational DAG for the instance.
     *
     * @return A reference to the computational DAG for the instance.
     */
    inline Graph_t &getComputationalDag() { return cdag; }

    inline std::size_t numberOfVertices() const { return cdag.num_vertices(); }

    inline auto vertices() const { return cdag.vertices(); }

    /**
     * @brief Returns the number of processors in the BSP architecture.
     *
     * @return The number of processors in the BSP architecture.
     */
    inline unsigned numberOfProcessors() const { return architecture.numberOfProcessors(); }

    /**
     * @brief Returns the communication costs between two processors.
     *
     * The communication costs are the send costs multiplied by the communication costs.
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     *
     * @return The communication costs between the two processors.
     */
    inline v_commw_t<Graph_t> communicationCosts(unsigned int p1, unsigned int p2) const {
        return architecture.communicationCosts(p1, p2);
    }

    /**
     * @brief Returns the send costs between two processors.
     *
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     *
     * @return The send costs between the two processors.
     */
    inline v_commw_t<Graph_t> sendCosts(unsigned int p1, unsigned int p2) const {
        return architecture.sendCosts(p1, p2);
    }

    /**
     * @brief Returns a copy of the send costs matrix.
     *
     * @return A copy of the send costs matrix.
     */
    inline const std::vector<std::vector<v_commw_t<Graph_t>>> &sendCostMatrix() const {
        return architecture.sendCostMatrix();
    }

    /**
     * @brief Returns the communication costs of the BSP architecture.
     *
     * @return The communication costs as an unsigned integer.
     */
    inline v_commw_t<Graph_t> communicationCosts() const { return architecture.communicationCosts(); }

    /**
     * @brief Returns the synchronization costs of the BSP architecture.
     *
     * @return The synchronization costs as an unsigned integer.
     */
    inline v_commw_t<Graph_t> synchronisationCosts() const { return architecture.synchronisationCosts(); }

    /**
     * @brief Returns whether the architecture is NUMA.
     *
     * @return True if the architecture is NUMA, false otherwise.
     */
    inline bool isNumaInstance() const { return architecture.isNumaArchitecture(); }

    inline v_memw_t<Graph_t> memoryBound(unsigned proc) const { return architecture.memoryBound(proc); }

    v_memw_t<Graph_t> maxMemoryBoundProcType(unsigned procType) const {
        return architecture.maxMemoryBoundProcType(procType);
    }

    v_memw_t<Graph_t> maxMemoryBoundNodeType(unsigned nodeType) const {
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
    inline void setCommunicationCosts(const v_commw_t<Graph_t> cost) { architecture.setCommunicationCosts(cost); }

    /**
     * @brief Sets the synchronisation costs of the BSP architecture.
     *
     * @param cost The synchronisation costs to set.
     */
    inline void setSynchronisationCosts(const v_commw_t<Graph_t> cost) { architecture.setSynchronisationCosts(cost); }

    /**
     * @brief Sets the number of processors in the BSP architecture.
     *
     * @param num The number of processors to set.
     */
    inline void setNumberOfProcessors(const unsigned num) { architecture.setNumberOfProcessors(num); }

    bool check_memory_constraints_feasibility() const {

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

    void adjust_memory_constraints() {

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

    bool isCompatible(const vertex_idx_t<Graph_t> &node, unsigned processor_id) const {
        return isCompatibleType(cdag.vertex_type(node), architecture.processorType(processor_id));
    }

    bool isCompatibleType(v_type_t<Graph_t> nodeType, v_type_t<Graph_t> processorType) const {

        return nodeProcessorCompatibility[nodeType][processorType];
    }

    void setNodeProcessorCompatibility(const std::vector<std::vector<bool>> &compatibility_) {

        nodeProcessorCompatibility = compatibility_;
    }

    const std::vector<std::vector<bool>> &getProcessorCompatibilityMatrix() const { return nodeProcessorCompatibility; }

    void setDiagonalCompatibilityMatrix(unsigned number_of_types) {

        nodeProcessorCompatibility =
            std::vector<std::vector<bool>>(number_of_types, std::vector<bool>(number_of_types, false));
        for (unsigned i = 0; i < number_of_types; ++i)
            nodeProcessorCompatibility[i][i] = true;
    }

    void setAllOnesCompatibilityMatrix() {

        unsigned number_of_node_types = cdag.num_vertex_types();
        unsigned number_of_proc_types = architecture.getNumberOfProcessorTypes();

        nodeProcessorCompatibility =
            std::vector<std::vector<bool>>(number_of_node_types, std::vector<bool>(number_of_proc_types, true));
    }

    std::vector<std::vector<unsigned>> getProcTypesCompatibleWithNodeType() const {
        unsigned numberOfNodeTypes = cdag.num_vertex_types();
        unsigned numberOfProcTypes = architecture.getNumberOfProcessorTypes();
        std::vector<std::vector<unsigned>> compatibleProcTypes(numberOfNodeTypes);

        for (unsigned nodeType = 0; nodeType < numberOfNodeTypes; ++nodeType)
            for (unsigned processorType = 0; processorType < numberOfProcTypes; ++processorType)
                if (isCompatibleType(nodeType, processorType))
                    compatibleProcTypes[nodeType].push_back(processorType);

        return compatibleProcTypes;
    }

    std::vector<std::vector<bool>> getNodeNodeCompatabilityMatrix() const {
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

    inline const std::vector<std::vector<bool>> &getNodeProcessorCompatibilityMatrix() const {
        return nodeProcessorCompatibility;
    }
};

} // namespace osp