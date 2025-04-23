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

@author Christos Matzoros, Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
*/

#include "model/SmSchedule.hpp"
#include "model/SetSchedule.hpp"

void SmSchedule::updateNumberOfSupersteps() {

    number_of_supersteps = 0;

    for (unsigned i = 0; i < instance->numberOfVertices(); ++i) {

        if (node_to_superstep_assignment[i] >= number_of_supersteps) {
            number_of_supersteps = node_to_superstep_assignment[i] + 1;
        }
    }
}

void SmSchedule::setAssignedSuperstep(unsigned node, unsigned superstep) {

    if (node < instance->numberOfVertices()) {
        node_to_superstep_assignment[node] = superstep;

        if (superstep >= number_of_supersteps) {
            number_of_supersteps = superstep + 1;
        }

    } else {
        throw std::invalid_argument("Invalid Argument while assigning node to superstep: index out of range.");
    }
}

void SmSchedule::setAssignedProcessor(unsigned node, unsigned processor) {

    if (node < instance->numberOfVertices() && processor < instance->numberOfProcessors()) {
        node_to_processor_assignment[node] = processor;
    } else {
        // std::cout << "node " << node << " num nodes " << instance->numberOfVertices() << "  processor " << processor
        //          << " num proc " << instance->numberOfProcessors() << std::endl;
        throw std::invalid_argument("Invalid Argument while assigning node to processor");
    }
}

void SmSchedule::addCommunicationScheduleEntry(unsigned node, unsigned from_proc, unsigned to_proc, unsigned step) {
    addCommunicationScheduleEntry(std::make_tuple(node, from_proc, to_proc), step);
}

void SmSchedule::addCommunicationScheduleEntry(KeyTriple key, unsigned step) {

    if (step >= number_of_supersteps)
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: step out of range.");

    if (get<0>(key) >= instance->numberOfVertices())
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: node out of range.");

    if (get<1>(key) >= instance->numberOfProcessors())
        throw std::invalid_argument(
            "Invalid Argument while adding communication schedule entry: from processor out of range.");

    if (get<2>(key) >= instance->numberOfProcessors())
        throw std::invalid_argument(
            "Invalid Argument while adding communication schedule entry: to processor out of range.");

    commSchedule[key] = step;
}

void SmSchedule::setAssignedSupersteps(const std::vector<unsigned> &vec) {

    if (vec.size() == instance->numberOfVertices()) {
        for (unsigned i = 0; i < instance->numberOfVertices(); ++i) {

            if (vec[i] >= number_of_supersteps) {
                number_of_supersteps = vec[i] + 1;
            }

            node_to_superstep_assignment[i] = vec[i];
        }
    } else {
        throw std::invalid_argument(
            "Invalid Argument while assigning supersteps: size does not match number of nodes.");
    }
}

void SmSchedule::setAssignedProcessors(const std::vector<unsigned> &vec) {

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

std::vector<unsigned> SmSchedule::getAssignedNodeVector(unsigned processor) const {

    std::vector<unsigned> vec;

    for (unsigned i = 0; i < (*(instance->getMatrix().getCSR())).rows(); i++) {

        if (node_to_processor_assignment[i] == processor) {
            vec.push_back(i);
        }
    }

    return vec;
}

std::vector<unsigned int> SmSchedule::getAssignedNodeVector(unsigned processor, unsigned superstep) const {
    std::vector<unsigned int> vec;

    for (unsigned int i = 0; i < instance->numberOfVertices(); i++) {

        if (node_to_processor_assignment[i] == processor && node_to_superstep_assignment[i] == superstep) {
            vec.push_back(i);
        }
    }

    return vec;
}


bool SmSchedule::satisfiesPrecedenceConstraints() const {

    if (node_to_processor_assignment.size() != instance->numberOfVertices() ||
        node_to_superstep_assignment.size() != instance->numberOfVertices()) {
        return false;
    }

    for (size_t vert = 0; vert < instance->getMatrix().numberOfVertices(); vert++) {
        SM_csc::InnerIterator c_it(*(instance->getMatrix().getCSC()), vert);
        ++c_it;

        const unsigned int number_of_children = instance->getMatrix().numberOfChildren(vert);
        auto succ = static_cast<VertexType>(c_it.index());

        for (unsigned i = 0; i < number_of_children; ++i){
            succ = static_cast<VertexType>(c_it.index());
    
            const int different_processors =
                (node_to_processor_assignment[vert] == node_to_processor_assignment[succ]) ? 0U : 1U;

            if ((node_to_superstep_assignment[vert] == UINT_MAX) && (different_processors == 1U)) {
                return false;
            }

            if (node_to_superstep_assignment[vert] + different_processors > node_to_superstep_assignment[succ]) {
                return false;
            }
        }
    }

    return true;
};


bool SmSchedule::satisfiesNodeTypeConstraints() const {

    if (node_to_processor_assignment.size() != instance->numberOfVertices())
        return false;

    for (unsigned int node = 0; node < instance->numberOfVertices(); node++) {
        if (!instance->isCompatible(node, node_to_processor_assignment[node]))
            return false;
    }

    return true;
};



unsigned SmSchedule::computeWorkCosts() const {

    assert(satisfiesPrecedenceConstraints());

    std::vector<std::vector<unsigned>> work = std::vector<std::vector<unsigned>>(
        number_of_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
        work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]] +=
            instance->getMatrix().nodeWorkWeight(node);
    }

    unsigned total_costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        unsigned max_work = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            max_work = std::max(max_work, work[step][proc]);
        }

        total_costs += max_work;
    }

    return total_costs;
}

unsigned SmSchedule::computeCosts() const {

    assert(satisfiesPrecedenceConstraints());

    std::set<unsigned> used_supersteps;
    for (unsigned vert = 0; vert < instance->numberOfVertices(); vert++) {
        used_supersteps.emplace( node_to_superstep_assignment[vert] );
    }

    return computeWorkCosts() + (instance->getArchitecture().communicationCosts() * computeBaseCommCost()) + (instance->getArchitecture().synchronisationCosts() * (used_supersteps.size() - 1));
}


unsigned SmSchedule::computeBaseCommCost() const {
    std::vector<std::vector<unsigned>> rec(number_of_supersteps,
                                           std::vector<unsigned>(instance->numberOfProcessors(), 0));
    std::vector<std::vector<unsigned>> send(number_of_supersteps,
                                            std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (auto const &[key, val] : commSchedule) {

        send[val][get<1>(key)] += instance->getMatrix().nodeCommunicationWeight(get<0>(key));
        rec[val][get<2>(key)] += instance->getMatrix().nodeCommunicationWeight(get<0>(key));
    }

    unsigned base_comm_cost = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        unsigned max_comm = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            if (max_comm < send[step][proc])
                max_comm = send[step][proc];
            if (max_comm < rec[step][proc])
                max_comm = rec[step][proc];
        }

        base_comm_cost += max_comm;
    }
    return base_comm_cost;
}

unsigned SmSchedule::num_assigned_nodes(unsigned processor) const {

    unsigned num = 0;

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        if (node_to_processor_assignment[i] == processor) {
            num++;
        }
    }

    return num;
}

std::vector<unsigned> SmSchedule::num_assigned_nodes_per_processor() const {

    std::vector<unsigned> num(instance->numberOfProcessors(), 0);

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        num[node_to_processor_assignment[i]]++;
    }

    return num;
}

std::vector<std::vector<unsigned>> SmSchedule::num_assigned_nodes_per_superstep_processor() const {

    std::vector<std::vector<unsigned>> num(number_of_supersteps,
                                           std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        num[node_to_superstep_assignment[i]][node_to_processor_assignment[i]] += 1;
    }

    return num;
}

bool SmSchedule::noOutOfBounds() const {
    for (unsigned vert = 0; vert < instance->numberOfVertices(); vert++) {
        if (node_to_processor_assignment[vert] >= instance->numberOfProcessors()) return false;
    }

    unsigned actualSteps = 0;
    for (unsigned vert = 0; vert < instance->numberOfVertices(); vert++) {
        if (node_to_superstep_assignment[vert] >= number_of_supersteps) return false;
        actualSteps = std::max(actualSteps, node_to_superstep_assignment[vert]);
    }
    if (actualSteps + 1U != number_of_supersteps) return false;

    return true;
}

unsigned SmSchedule::computeBaseCommCostsBufferedSending() const {

    assert(satisfiesPrecedenceConstraints());

    // std::vector<unsigned> comm = std::vector<unsigned>(number_of_supersteps, 0);
    std::vector<std::vector<unsigned>> rec(instance->numberOfProcessors(),
                                           std::vector<unsigned>(number_of_supersteps, 0));
    std::vector<std::vector<unsigned>> send(instance->numberOfProcessors(),
                                            std::vector<unsigned>(number_of_supersteps, 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {

        std::vector<unsigned> step_needed(instance->numberOfProcessors(), number_of_supersteps);
        for (const auto &target : instance->getMatrix().children(node)) {

            if (node_to_processor_assignment[node] != node_to_processor_assignment[target]) {
                step_needed[node_to_processor_assignment[target]] =
                    std::min(step_needed[node_to_processor_assignment[target]], node_to_superstep_assignment[target]);
            }
        }

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (step_needed[proc] < number_of_supersteps) {
                send[node_to_processor_assignment[node]][node_to_superstep_assignment[node]] +=
                    instance->sendCosts(node_to_processor_assignment[node], proc) *
                    instance->getMatrix().nodeCommunicationWeight(node);

                rec[proc][step_needed[proc] - 1] += instance->sendCosts(node_to_processor_assignment[node], proc) *
                                                    instance->getMatrix().nodeCommunicationWeight(node);
            }
        }
    }

    unsigned costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {
        unsigned max_send = 0;
        unsigned max_rec = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            if (max_send < send[proc][step])
                max_send = send[proc][step];
            if (max_rec < rec[proc][step])
                max_rec = rec[proc][step];
        }

        costs += std::max(max_send, max_rec);
    }

    return costs;
}

double SmSchedule::computeBaseCommCostsTotalCommunication() const {

    assert(satisfiesPrecedenceConstraints());

    double total_communication = 0;

    for (unsigned src = 0; src < instance->numberOfVertices(); src++) {
        for (unsigned tgt_ind = instance->getMatrix().getCSC()->outerIndexPtr()[src] + 1; tgt_ind < instance->getMatrix().getCSC()->outerIndexPtr()[src + 1]; tgt_ind++) {
            unsigned tgt = instance->getMatrix().getCSC()->innerIndexPtr()[tgt_ind];

            if (node_to_processor_assignment[src] != node_to_processor_assignment[tgt]) {
                total_communication +=
                    instance->sendCosts(node_to_processor_assignment[src], node_to_processor_assignment[tgt]) *
                    instance->getMatrix().nodeCommunicationWeight(src);
            }
        }
    }

    return total_communication * (1.0 / instance->numberOfProcessors());
}