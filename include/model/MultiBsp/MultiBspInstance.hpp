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

#include "model/BspArchitecture.hpp"
#include "model/ComputationalDag.hpp"

class MultiBspArchitecture {

  private:
    std::vector<BspArchitecture> architectures;

  public:
    MultiBspArchitecture() = default;
    virtual ~MultiBspArchitecture() = default;

    const std::vector<BspArchitecture> &getArchitectures() const { return architectures; }

    std::vector<BspArchitecture> &getArchitectures() { return architectures; }

    unsigned numberOfProcessors() const {

        if (architectures.empty()) {
            return 0;
        }

        unsigned num = architectures[0].numberOfProcessors();

        for (unsigned i = 1; i < architectures.size(); i++) {
            num *= architectures[i].numberOfProcessors();
        }

        return num;
    }

    inline unsigned communicationCosts(unsigned p1, unsigned p2) const {

        unsigned num_proc = numberOfProcessors();

        unsigned comm_costs = 0;

        for (unsigned i = 0; i < architectures.size(); i++) {

            const unsigned proc_0 = architectures[i].numberOfProcessors();

            const unsigned num_buckets = num_proc / proc_0;

            bool same_bucket = false;

            for (unsigned j = 0; j < num_buckets; j++) {
                if (p1 < (j + 1) * proc_0 && p1 >= j * proc_0 && p2 < (j + 1) * proc_0 && p2 >= j * proc_0) {
                    same_bucket = true;
                    break;
                }
            }

            if (not same_bucket) {
                return architectures[i].communicationCosts(p1 % proc_0, p2 % proc_0);
            }

            num_proc = architectures[i + 1].numberOfProcessors();
            for (unsigned j = 2; j < architectures.size(); j++) {
                num_proc *= architectures[i + j].numberOfProcessors();
            }
        }

        throw std::runtime_error("Not implemented yet");
        return 1;
    }
};

/**
 * @class MultiBspInstance
 * @brief Represents an instance of the Multi BSP (Bulk Synchronous Parallel) model.
 *
 * The MultiBspInstance class encapsulates the computational DAG (Directed Acyclic Graph) and the BSP architecture
 * for a specific instance of the BSP model. It provides methods to access and modify the architecture and DAG,
 * as well as retrieve information about the instance such as the number of vertices and processors.
 */
class MultiBspInstance {

  private:
    ComputationalDag cdag;
    MultiBspArchitecture architecture;

  public:
    /**
     * @brief Default constructor for the BspInstance class.
     */
    MultiBspInstance() = default;

    /**
     * @brief Constructs a MultiBspInstance object with the specified computational DAG and BSP architecture.
     *
     * @param cdag The computational DAG for the instance.
     * @param architecture The multi BSP architecture for the instance.
     */
    MultiBspInstance(ComputationalDag cdag, MultiBspArchitecture architecture)
        : cdag(cdag), architecture(architecture) {}

    /**
     * @brief Returns a reference to the BSP architecture for the instance.
     *
     * @return A reference to the BSP architecture for the instance.
     */
    inline const MultiBspArchitecture &getArchitecture() const { return architecture; }

    /**
     * @brief Returns a reference to the BSP architecture for the instance.
     *
     * @return A reference to the BSP architecture for the instance.
     */
    inline MultiBspArchitecture &getArchitecture() { return architecture; }

    /**
     * @brief Sets the BSP architecture for the instance.
     *
     * @param architecture_ The BSP architecture for the instance.
     */
    inline void setArchitecture(const MultiBspArchitecture &architechture_) { architecture = architechture_; }

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
    inline unsigned int communicationCosts(unsigned int p1, unsigned int p2) const {
        return architecture.communicationCosts(p1, p2);
    }
};
