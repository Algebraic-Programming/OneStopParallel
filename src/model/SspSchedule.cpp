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

#include "model/SspSchedule.hpp"

unsigned SspSchedule::getMaxStaleness() const {

    unsigned stale = UINT_MAX;
    for (const auto& edge : instance->getComputationalDag().edges()) {
        const VertexType& source = edge.m_source;
        const VertexType& target = edge.m_target;
        if ( node_to_processor_assignment[source] != node_to_processor_assignment[target] ) {
            if (node_to_superstep_assignment[target] > node_to_superstep_assignment[source]) {
                stale = std::min(stale, node_to_superstep_assignment[target] - node_to_superstep_assignment[source]);
            } else {
                throw std::logic_error("SspSchedule does not satisfy BSP constraints");
            }
        } else if (node_to_superstep_assignment[target] < node_to_superstep_assignment[source]) {
            throw std::logic_error("SspSchedule does not satisfy BSP constraints");
        }
    }
    return stale;
}

bool SspSchedule::satisfiesPrecedenceConstraints() const {
    return satisfiesPrecedenceConstraints(staleness);
}

bool SspSchedule::satisfiesPrecedenceConstraints(unsigned stale) const {

    if (node_to_processor_assignment.size() != instance->numberOfVertices() ||
        node_to_superstep_assignment.size() != instance->numberOfVertices()) {
        return false;
    }

    for (const auto &ep : instance->getComputationalDag().edges()) {
        const unsigned &source = instance->getComputationalDag().source(ep);
        const unsigned &target = instance->getComputationalDag().target(ep);

        const int different_processors =
            (node_to_processor_assignment[source] == node_to_processor_assignment[target]) ? 0 : stale;

        if (node_to_superstep_assignment[source] + different_processors > node_to_superstep_assignment[target]) {
            return false;
        }
    }

    return true;
};

unsigned SspSchedule::computeSspCosts() const {
    return computeSspCosts(staleness);
}

unsigned SspSchedule::computeSspCosts(unsigned stale) const {
    assert(satisfiesPrecedenceConstraints(stale));

    std::vector<std::vector<unsigned>> work = std::vector<std::vector<unsigned>>(
        number_of_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
        work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]] +=
            instance->getComputationalDag().nodeWorkWeight(node);
    }

    std::vector<std::vector<unsigned>> rec(number_of_supersteps,
                                           std::vector<unsigned>(instance->numberOfProcessors(), 0));
    std::vector<std::vector<unsigned>> send(number_of_supersteps,
                                            std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (const auto& edge : instance->getComputationalDag().edges()) {
        const VertexType& source = edge.m_source;
        const VertexType& target = edge.m_target;

        unsigned comm_cost = instance->sendCosts(node_to_processor_assignment[source], node_to_processor_assignment[target])
                                * instance->getComputationalDag().nodeCommunicationWeight(source);

        send[node_to_superstep_assignment[source]][node_to_processor_assignment[source]] += comm_cost;
        rec[node_to_superstep_assignment[source]][node_to_processor_assignment[target]] += comm_cost;
    }

    std::vector<unsigned> start_comm(number_of_supersteps, 0);
    std::vector<unsigned> end_comm(number_of_supersteps, 0);
    std::vector<unsigned> start_work(number_of_supersteps, 0);
    std::vector<unsigned> end_work(number_of_supersteps, 0);

    for (size_t step = 0; step < number_of_supersteps; step++) {
        unsigned earliest_work_start = 0;
        if (step > 0) {
            earliest_work_start = std::max(earliest_work_start, end_work[step - 1]);
        }
        if (step >= stale) {
            earliest_work_start = std::max(earliest_work_start, end_comm[step - stale]);
        }
        start_work[step] = earliest_work_start;

        unsigned earliest_comm_start = 0;
        if (step > 0) {
            earliest_comm_start = std::max(end_comm[step - 1], end_work[step - 1]);
        }
        start_comm[step] = earliest_comm_start;

        unsigned work_max = 0;
        for (const auto& work_proc : work[step]) {
            work_max = std::max(work_max, work_proc);
        }
        end_work[step] = start_work[step] + work_max;
        
        unsigned comm_max = 0;
        for (const auto& send_proc : send[step]) {
            comm_max = std::max(comm_max, send_proc);
        }
        for (const auto& rec_proc : rec[step]) {
            comm_max = std::max(comm_max, rec_proc);
        }
        end_comm[step] = start_comm[step] + comm_max + instance->synchronisationCosts();
    }

    return end_work.back();
}

