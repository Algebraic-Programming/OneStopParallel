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

#include "model/DAGPartition.hpp"

void DAGPartition::setAssignedProcessor(unsigned node, unsigned processor) {

    if (node < instance->numberOfVertices() && processor < instance->numberOfProcessors()) {
        node_to_processor_assignment[node] = processor;
    } else {
        // std::cout << "node " << node << " num nodes " << instance->numberOfVertices() << "  processor " << processor
        //          << " num proc " << instance->numberOfProcessors() << std::endl;
        throw std::invalid_argument("Invalid Argument while assigning node to processor");
    }
}

void DAGPartition::setAssignedProcessors(const std::vector<unsigned> &vec) {

    if (vec.size() == instance->numberOfVertices()) {
        for (unsigned i = 0; i < instance->numberOfVertices(); ++i) {

            if (vec[i] >= instance->numberOfProcessors()) {
                throw std::invalid_argument(
                    "Invalid Argument while assigning processors: processor index out of range.");
            }

            node_to_processor_assignment[i] = vec[i];
        }
    } else {
        throw std::invalid_argument(
            "Invalid Argument while assigning processors: size does not match number of nodes.");
    }
}

std::vector<unsigned> DAGPartition::getAssignedNodeVector(unsigned processor) const {

    std::vector<unsigned> vec;

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {

        if (node_to_processor_assignment[i] == processor) {
            vec.push_back(i);
        }
    }

    return vec;
}

std::vector<unsigned> DAGPartition::computeAllMemoryCosts() const {
    std::vector<unsigned> memory(instance->numberOfProcessors(), 0);
    for (unsigned v = 0; v < instance->numberOfVertices(); v++) {
        memory[node_to_processor_assignment[v]] += instance->getComputationalDag().nodeMemoryWeight(v);
    }

    return memory;
};

unsigned DAGPartition::computeMemoryCosts(unsigned processor) const {
    unsigned memory = 0;
    for (unsigned v = 0; v < instance->numberOfVertices(); v++) {
        if (node_to_processor_assignment[v] == processor) {
            memory += instance->getComputationalDag().nodeMemoryWeight(v);
        }
    }

    return memory;
};

unsigned DAGPartition::computeMaxMemoryCosts() const {
    std::vector<unsigned> memory = computeAllMemoryCosts();
    unsigned max_memory = 0;
    for (unsigned weight : memory) {
        if (weight > max_memory) {
            max_memory = weight;
        }
    }
    return max_memory;
};


bool DAGPartition::satisfiesMemoryConstraints() const {

    std::vector<unsigned> memory = computeAllMemoryCosts();
    for (unsigned p = 0; p < instance->numberOfProcessors(); p++) {
        if (memory[p] > instance->getArchitecture().memoryBound(p)) {
            return false;
        }
    }
    return true;
};


std::vector<unsigned> DAGPartition::computeAllWorkCosts() const {
    std::vector<unsigned> work_cost(instance->numberOfProcessors(), 0);
    for (unsigned v = 0; v < instance->numberOfVertices(); v++) {
        work_cost[node_to_processor_assignment[v]] += instance->getComputationalDag().nodeWorkWeight(v);
    }

    return work_cost;
};

unsigned DAGPartition::computeWorkCosts(unsigned processor) const {
    unsigned work_cost = 0;
    for (unsigned v = 0; v < instance->numberOfVertices(); v++) {
        if (node_to_processor_assignment[v] == processor) {
            work_cost += instance->getComputationalDag().nodeWorkWeight(v);
        }
    }

    return work_cost;
};

unsigned DAGPartition::computeMaxWorkCosts() const {
    std::vector<unsigned> work_cost = computeAllWorkCosts();
    unsigned max_work = 0;
    for (unsigned weight : work_cost) {
        if (weight > max_work) {
            max_work = weight;
        }
    }
    return max_work;
};

float DAGPartition::computeWorkImbalance() const {
    unsigned max_work_cost = computeMaxWorkCosts();
    long unsigned total_work = 0;
    for (unsigned v = 0; v < instance->numberOfVertices(); v++) {
        total_work += instance->getComputationalDag().nodeWorkWeight(v);
    }
    return (float) max_work_cost / ( (float) total_work / (float) instance->numberOfProcessors() );
};


unsigned DAGPartition::num_assigned_nodes(unsigned processor) const {

    unsigned num = 0;

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        if (node_to_processor_assignment[i] == processor) {
            num++;
        }
    }

    return num;
};

std::vector<unsigned> DAGPartition::num_assigned_nodes() const {

    std::vector<unsigned> num(instance->numberOfProcessors(), 0);

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        num[node_to_processor_assignment[i]]++;
    }

    return num;
};

std::vector<std::vector<unsigned>> DAGPartition::computeAlltoAllCommunication() const {
    std::vector<std::vector<unsigned>> alltoall(instance->numberOfProcessors(), std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (const auto& edge : instance->getComputationalDag().edges()) {
        if ( node_to_processor_assignment[ edge.m_source ] != node_to_processor_assignment[ edge.m_target ] ) {
            alltoall[node_to_processor_assignment[ edge.m_source ]][node_to_processor_assignment[ edge.m_target ]] += instance->getComputationalDag().nodeCommunicationWeight(edge.m_source);
        }
    }

    return alltoall;
};

unsigned DAGPartition::computeTotalCommunication() const {
    unsigned total_comm = 0;

    for (const auto& edge : instance->getComputationalDag().edges()) {
        if ( node_to_processor_assignment[ edge.m_source ] != node_to_processor_assignment[ edge.m_target ] ) {
            total_comm += instance->getComputationalDag().nodeCommunicationWeight(edge.m_source);
        }
    }

    return total_comm;
};

float DAGPartition::computeCommunicationRatio() const {
    unsigned comm = 0;
    unsigned total_comm = 0;

    for (const auto& edge : instance->getComputationalDag().edges()) {
        total_comm += instance->getComputationalDag().nodeCommunicationWeight(edge.m_source);
        if ( node_to_processor_assignment[ edge.m_source ] != node_to_processor_assignment[ edge.m_target ] ) {
            comm += instance->getComputationalDag().nodeCommunicationWeight(edge.m_source);
        }
    }

    total_comm = std::max(total_comm, 1U);

    return (float) comm / (float) total_comm;
};

unsigned DAGPartition::computeCutEdges() const {
    unsigned total_cut_edges = 0;

    for (const auto& edge : instance->getComputationalDag().edges()) {
        if ( node_to_processor_assignment[ edge.m_source ] != node_to_processor_assignment[ edge.m_target ] ) {
            total_cut_edges += 1;
        }
    }

    return total_cut_edges;
};

float DAGPartition::computeCutEdgesRatio() const {
    return (float) computeCutEdges() / (float) std::max(instance->getComputationalDag().numberOfEdges(),1U);
};