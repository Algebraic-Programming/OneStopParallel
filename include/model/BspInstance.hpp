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

#include "BspArchitecture.hpp"
#include "ComputationalDag.hpp"

/**
 * @class BspInstance
 * @brief Represents an instance of the BSP (Bulk Synchronous Parallel) model.
 * 
 * The BspInstance class encapsulates the computational DAG (Directed Acyclic Graph) and the BSP architecture
 * for a specific instance of the BSP model. It provides methods to access and modify the architecture and DAG,
 * as well as retrieve information about the instance such as the number of vertices and processors.
 */
class BspInstance {

  private:
    ComputationalDag cdag;
    BspArchitecture architecture;


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
    BspInstance(ComputationalDag cdag, BspArchitecture architecture)
        : cdag(cdag), architecture(architecture) {}

    /**
     * @brief Returns a reference to the BSP architecture for the instance.
     *
     * @return A reference to the BSP architecture for the instance.
     */    
    inline const BspArchitecture &getArchitecture() const { return architecture; }

    /**
     * @brief Returns a reference to the BSP architecture for the instance.
     *
     * @return A reference to the BSP architecture for the instance.
     */
    inline BspArchitecture &getArchitecture() { return architecture; }
    
    
    /**
     * @brief Sets the BSP architecture for the instance.
     *
     * @param architecture_ The BSP architecture for the instance.
     */
    inline void setArchitecture(const BspArchitecture& architechture_) { architecture = architechture_; }

    /**
     * @brief Returns a reference to the computational DAG for the instance.
     *
     * @return A reference to the computational DAG for the instance.
     */
    inline const ComputationalDag &getComputationalDag() const { return cdag; }
    
    /**
     * @brief Returns a reference to the computational DAG for the instance.
     *
     * @return A reference to the computational DAG for the instance.
     */
    inline ComputationalDag &getComputationalDag() { return cdag; }

    
    inline unsigned int numberOfVertices() const { return cdag.numberOfVertices(); }

    /**
     * @brief Returns the number of processors in the BSP architecture.
     *
     * @return The number of processors in the BSP architecture.
     */
    inline unsigned int numberOfProcessors() const { return architecture.numberOfProcessors(); }

    
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
    inline unsigned int communicationCosts(unsigned int p1, unsigned int p2) const { return architecture.communicationCosts(p1, p2); }
    
    
    /**
     * @brief Returns the send costs between two processors.
     *
     *
     * @param p1 The index of the first processor.
     * @param p2 The index of the second processor.
     *
     * @return The send costs between the two processors.
     */
    inline unsigned int sendCosts(unsigned int p1, unsigned int p2) const { return architecture.sendCosts(p1, p2); }
    
    /**
     * @brief Returns a copy of the send costs matrix.
     *
     * @return A copy of the send costs matrix.
     */
    inline const std::vector<std::vector<unsigned int>>& sendCostMatrix() const { return architecture.sendCostMatrix(); }
    
    
    /**
     * @brief Returns the communication costs of the BSP architecture.
     *
     * @return The communication costs as an unsigned integer.
     */
    inline unsigned int communicationCosts() const { return architecture.communicationCosts(); }
    
    /**
     * @brief Returns the synchronization costs of the BSP architecture.
     *
     * @return The synchronization costs as an unsigned integer.
     */
    inline unsigned int synchronisationCosts() const { return architecture.synchronisationCosts(); }
    
    
    /**
     * @brief Returns whether the architecture is NUMA.
     *
     * @return True if the architecture is NUMA, false otherwise.
     */
    inline bool isNumaInstance() const { return architecture.isNumaArchitecture(); }

    inline unsigned memoryBound() const { return architecture.memoryBound(); }
    inline unsigned memoryBound(unsigned proc) const { return architecture.memoryBound(proc); }

    /**
     * @brief Sets the communication costs of the BSP architecture.
     *
     * @param cost The communication costs to set.
     */
    inline void setCommunicationCosts(const unsigned int cost) { architecture.setCommunicationCosts(cost); }
    
    /**
     * @brief Sets the synchronisation costs of the BSP architecture.
     *
     * @param cost The synchronisation costs to set.
     */
    inline void setSynchronisationCosts(const unsigned int cost) { architecture.setSynchronisationCosts(cost); }
    
    /**
     * @brief Sets the number of processors in the BSP architecture.
     *
     * @param num The number of processors to set.
     */       
    inline void setNumberOfProcessors(const unsigned int num) { architecture.setNumberOfProcessors(num); }


};
