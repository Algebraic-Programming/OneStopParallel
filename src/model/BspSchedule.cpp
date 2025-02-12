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

#include "model/BspSchedule.hpp"
#include "model/SetSchedule.hpp"

void BspSchedule::updateNumberOfSupersteps() {

    number_of_supersteps = 0;

    for (unsigned i = 0; i < instance->numberOfVertices(); ++i) {

        if (node_to_superstep_assignment[i] >= number_of_supersteps) {
            number_of_supersteps = node_to_superstep_assignment[i] + 1;
        }
    }
}

void BspSchedule::setAssignedSuperstep(unsigned node, unsigned superstep) {

    if (node < instance->numberOfVertices()) {
        node_to_superstep_assignment[node] = superstep;

        if (superstep >= number_of_supersteps) {
            number_of_supersteps = superstep + 1;
        }

    } else {
        throw std::invalid_argument("Invalid Argument while assigning node to superstep: index out of range.");
    }
}

void BspSchedule::setAssignedProcessor(unsigned node, unsigned processor) {

    if (node < instance->numberOfVertices() && processor < instance->numberOfProcessors()) {
        node_to_processor_assignment[node] = processor;
    } else {
        // std::cout << "node " << node << " num nodes " << instance->numberOfVertices() << "  processor " << processor
        //          << " num proc " << instance->numberOfProcessors() << std::endl;
        throw std::invalid_argument("Invalid Argument while assigning node to processor");
    }
}

void BspSchedule::addCommunicationScheduleEntry(unsigned node, unsigned from_proc, unsigned to_proc, unsigned step) {
    addCommunicationScheduleEntry(std::make_tuple(node, from_proc, to_proc), step);
}

void BspSchedule::addCommunicationScheduleEntry(KeyTriple key, unsigned step) {

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

void BspSchedule::setAssignedSupersteps(const std::vector<unsigned> &vec) {

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

void BspSchedule::setAssignedProcessors(const std::vector<unsigned> &vec) {

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

std::vector<unsigned> BspSchedule::getAssignedNodeVector(unsigned processor) const {

    std::vector<unsigned> vec;

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {

        if (node_to_processor_assignment[i] == processor) {
            vec.push_back(i);
        }
    }

    return vec;
}

std::vector<unsigned int> BspSchedule::getAssignedNodeVector(unsigned processor, unsigned superstep) const {
    std::vector<unsigned int> vec;

    for (unsigned int i = 0; i < instance->numberOfVertices(); i++) {

        if (node_to_processor_assignment[i] == processor && node_to_superstep_assignment[i] == superstep) {
            vec.push_back(i);
        }
    }

    return vec;
}

bool BspSchedule::satisfiesPrecedenceConstraints() const {

    if (node_to_processor_assignment.size() != instance->numberOfVertices() ||
        node_to_superstep_assignment.size() != instance->numberOfVertices()) {
        return false;
    }

    // bool comm_edge_found = false;

    for (const auto &ep : instance->getComputationalDag().edges()) {
        const unsigned &source = instance->getComputationalDag().source(ep);
        const unsigned &target = instance->getComputationalDag().target(ep);

        const int different_processors =
            (node_to_processor_assignment[source] == node_to_processor_assignment[target]) ? 0 : 1;

        if (node_to_superstep_assignment[source] + different_processors > node_to_superstep_assignment[target]) {
            // std::cout << "This is not a valid scheduling (problems with nodes " << source << " and " << target <<
            // ")."
            //           << std::endl; // todo should be removed
            return false;
        }
    }

    return true;
};

bool BspSchedule::satisfiesNodeTypeConstraints() const {

    if (node_to_processor_assignment.size() != instance->numberOfVertices())
        return false;

    for (unsigned int node = 0; node < instance->numberOfVertices(); node++) {
        if (!instance->isCompatible(node, node_to_processor_assignment[node]))
            return false;
    }

    return true;
};

bool BspSchedule::satisfiesMemoryConstraints() const {

    switch (instance->getArchitecture().getMemoryConstraintType()) {

    case LOCAL: {

        SetSchedule set_schedule = SetSchedule(*this);

        for (unsigned step = 0; step < number_of_supersteps; step++) {
            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                unsigned memory = 0;
                for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                    memory += instance->getComputationalDag().nodeMemoryWeight(node);
                }

                if (memory > instance->getArchitecture().memoryBound(proc)) {
                    return false;
                }
            }
        }

        break;
    }

    case PERSISTENT_AND_TRANSIENT: {
        std::vector<int> current_proc_persistent_memory(instance->numberOfProcessors(), 0);
        std::vector<int> current_proc_transient_memory(instance->numberOfProcessors(), 0);

        for (VertexType node = 0; node < instance->numberOfVertices(); node++) {

            const unsigned proc = node_to_processor_assignment[node];
            current_proc_persistent_memory[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
            current_proc_transient_memory[proc] = std::max(
                current_proc_transient_memory[proc], instance->getComputationalDag().nodeCommunicationWeight(node));

            if (current_proc_persistent_memory[proc] + current_proc_transient_memory[proc] >
                instance->getArchitecture().memoryBound(proc)) {
                return false;
            }
        }
        break;
    }

    case GLOBAL: {
        std::vector<unsigned> current_proc_memory(instance->numberOfProcessors(), 0);

        for (VertexType node = 0; node < instance->numberOfVertices(); node++) {

            const unsigned proc = node_to_processor_assignment[node];
            current_proc_memory[proc] += instance->getComputationalDag().nodeMemoryWeight(node);

            if (current_proc_memory[proc] > instance->getArchitecture().memoryBound(proc)) {
                return false;
            }
        }
        break;
    }

    case LOCAL_IN_OUT: {

        SetSchedule set_schedule = SetSchedule(*this);

        for (unsigned step = 0; step < number_of_supersteps; step++) {
            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                long int memory = 0;
                for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                    memory += instance->getComputationalDag().nodeMemoryWeight(node) +
                              instance->getComputationalDag().nodeCommunicationWeight(node);

                    for (const auto &parent : instance->getComputationalDag().parents(node)) {

                        if (node_to_processor_assignment[parent] == proc &&
                            node_to_superstep_assignment[parent] == step) {
                            memory -= instance->getComputationalDag().nodeCommunicationWeight(parent);
                        }
                    }
                }

                if (memory > static_cast<long int>(instance->getArchitecture().memoryBound(proc))) {
                    return false;
                }
            }
        }

        break;
    }

    case LOCAL_INC_EDGES: {

        SetSchedule set_schedule = SetSchedule(*this);

        for (unsigned step = 0; step < number_of_supersteps; step++) {
            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                std::unordered_set<unsigned> nodes_with_incoming_edges;

                int memory = 0;
                for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                    memory += instance->getComputationalDag().nodeCommunicationWeight(node);

                    for (const auto &parent : instance->getComputationalDag().parents(node)) {

                        if (node_to_superstep_assignment[parent] != step) {
                            nodes_with_incoming_edges.insert(parent);
                        }
                    }
                }

                for (const auto &node : nodes_with_incoming_edges) {
                    memory += instance->getComputationalDag().nodeCommunicationWeight(node);
                }

                if (memory > instance->getArchitecture().memoryBound(proc)) {
                    return false;
                }
            }
        }
        break;
    }

    case NONE: {
        break;
    }

    default: {
        throw std::invalid_argument("Unknown memory constraint type.");
        break;
    }
    }

    return true;
};

bool BspSchedule::hasValidCommSchedule() const { return checkCommScheduleValidity(commSchedule); }

bool BspSchedule::checkCommScheduleValidity(const std::map<KeyTriple, unsigned int> &cs) const {

    std::vector<std::vector<unsigned>> first_at = std::vector<std::vector<unsigned>>(
        instance->numberOfVertices(), std::vector<unsigned>(instance->numberOfProcessors(), number_of_supersteps));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
        first_at[node][node_to_processor_assignment[node]] = node_to_superstep_assignment[node];
    }

    for (auto const &[key, val] : cs) {

        if (val >= number_of_supersteps)
            return false;

        if (get<0>(key) >= instance->numberOfVertices())
            return false;

        if (get<1>(key) >= instance->numberOfProcessors())
            return false;

        if (get<2>(key) >= instance->numberOfProcessors())
            return false;

        first_at[get<0>(key)][get<2>(key)] = std::min(first_at[get<0>(key)][get<2>(key)], val + 1);
    }

    for (auto const &[key, val] : cs) {

        if (val < first_at[get<0>(key)][get<1>(key)]) {
            return false;
        }
    }

    for (const auto &ep : instance->getComputationalDag().edges()) {
        const unsigned int source = instance->getComputationalDag().source(ep);
        const unsigned int target = instance->getComputationalDag().target(ep);

        if (first_at[source][node_to_processor_assignment[target]] > node_to_superstep_assignment[target]) {
            return false;
        }
    }
    return true;
}

void BspSchedule::setLazyCommunicationSchedule() {
    commSchedule.clear();

    for (const auto &ep : instance->getComputationalDag().edges()) {
        const unsigned int source = instance->getComputationalDag().source(ep);
        const unsigned int target = instance->getComputationalDag().target(ep);

        if (node_to_processor_assignment[source] != node_to_processor_assignment[target]) {

            const auto tmp =
                std::make_tuple(source, node_to_processor_assignment[source], node_to_processor_assignment[target]);
            if (commSchedule.find(tmp) == commSchedule.end()) {
                commSchedule[tmp] = node_to_superstep_assignment[target] - 1;

            } else {
                commSchedule[tmp] = std::min(node_to_superstep_assignment[target] - 1, commSchedule[tmp]);
            }
        }
    }
}

void BspSchedule::setImprovedLazyCommunicationSchedule() {
    commSchedule.clear();
    if (instance->getComputationalDag().numberOfVertices() <= 1 || number_of_supersteps <= 1)
        return;

    std::vector<std::vector<std::vector<size_t>>> step_proc_node_list(
        number_of_supersteps, std::vector<std::vector<size_t>>(instance->numberOfProcessors(), std::vector<size_t>()));
    std::vector<std::vector<bool>> node_to_proc_been_sent(instance->numberOfVertices(),
                                                          std::vector<bool>(instance->numberOfProcessors(), false));

    for (size_t node = 0; node < instance->numberOfVertices(); node++) {
        step_proc_node_list[node_to_superstep_assignment[node]][node_to_processor_assignment[node]].push_back(node);
        node_to_proc_been_sent[node][node_to_processor_assignment[node]] = true;
    }

    // processor, ordered list of (cost, node, to_processor)
    std::vector<std::set<std::vector<size_t>, std::greater<>>> require_sending(instance->numberOfProcessors());
    for (size_t proc = 0; proc < instance->numberOfProcessors(); proc++) {
        for (const auto &node : step_proc_node_list[0][proc]) {
            for (const auto &out_edge : instance->getComputationalDag().out_edges(node))
                if (proc != assignedProcessor(out_edge.m_target)) {
                    require_sending[proc].insert({instance->getComputationalDag().nodeCommunicationWeight(node) *
                                                      instance->getArchitecture().sendCosts(
                                                          proc, node_to_processor_assignment[out_edge.m_target]),
                                                  node, node_to_processor_assignment[out_edge.m_target]});
                }
        }
    }

    for (size_t step = 1; step < number_of_supersteps; step++) {
        std::vector<unsigned> send_cost(instance->numberOfProcessors(), 0);
        std::vector<unsigned> receive_cost(instance->numberOfProcessors(), 0);

        // must send in superstep step-1
        for (size_t proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto &node : step_proc_node_list[step][proc]) {
                for (const auto &in_edge : instance->getComputationalDag().in_edges(node)) {
                    if (!node_to_proc_been_sent[in_edge.m_source][proc]) {
                        assert(node_to_superstep_assignment[in_edge.m_source] < step);
                        commSchedule.emplace(
                            std::make_tuple(in_edge.m_source, node_to_processor_assignment[in_edge.m_source], proc),
                            step - 1);
                        node_to_proc_been_sent[in_edge.m_source][proc] = true;
                        unsigned comm_cost =
                            instance->getComputationalDag().nodeCommunicationWeight(in_edge.m_source) *
                            instance->getArchitecture().sendCosts(node_to_processor_assignment[in_edge.m_source], proc);
                        require_sending[assignedProcessor(in_edge.m_source)].erase({comm_cost, in_edge.m_source, proc});
                        send_cost[node_to_processor_assignment[in_edge.m_source]] += comm_cost;
                        receive_cost[proc] += comm_cost;
                    }
                }
            }
        }

        // getting max costs
        unsigned max_comm_cost = 0;
        for (size_t proc = 0; proc < instance->numberOfProcessors(); proc++) {
            max_comm_cost = std::max(max_comm_cost, send_cost[proc]);
            max_comm_cost = std::max(max_comm_cost, receive_cost[proc]);
        }

        // extra sends
        // TODO: permute the order of processors
        for (size_t proc = 0; proc < instance->numberOfProcessors(); proc++) {
            if (require_sending[proc].empty() ||
                (*(require_sending[proc].rbegin()))[0] + send_cost[proc] > max_comm_cost)
                continue;
            auto iter = require_sending[proc].begin();
            while (iter != require_sending[proc].cend()) {
                if ((*iter)[0] + send_cost[proc] > max_comm_cost ||
                    (*iter)[0] + receive_cost[(*iter)[2]] > max_comm_cost) {
                    iter++;
                } else {
                    commSchedule.emplace(std::make_tuple((*iter)[1], proc, (*iter)[2]), step - 1);
                    node_to_proc_been_sent[(*iter)[1]][(*iter)[2]] = true;
                    send_cost[proc] += (*iter)[0];
                    receive_cost[(*iter)[2]] += (*iter)[0];
                    iter = require_sending[proc].erase(iter);
                    if (require_sending[proc].empty() ||
                        (*(require_sending[proc].rbegin()))[0] + send_cost[proc] > max_comm_cost)
                        break;
                }
            }
        }

        // updating require_sending
        for (size_t proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto &node : step_proc_node_list[step][proc]) {
                for (const auto &out_edge : instance->getComputationalDag().out_edges(node))
                    if (proc != assignedProcessor(out_edge.m_target)) {
                        require_sending[proc].insert({instance->getComputationalDag().nodeCommunicationWeight(node) *
                                                          instance->getArchitecture().sendCosts(
                                                              proc, node_to_processor_assignment[out_edge.m_target]),
                                                      node, node_to_processor_assignment[out_edge.m_target]});
                    }
            }
        }
    }
}

void BspSchedule::setEagerCommunicationSchedule() {
    commSchedule.clear();

    for (const auto &ep : instance->getComputationalDag().edges()) {
        const unsigned int source = instance->getComputationalDag().source(ep);
        const unsigned int target = instance->getComputationalDag().target(ep);

        if (node_to_processor_assignment[source] != node_to_processor_assignment[target]) {

            commSchedule[std::make_tuple(source, node_to_processor_assignment[source],
                                         node_to_processor_assignment[target])] = node_to_superstep_assignment[source];
        }
    }
}

void BspSchedule::setAutoCommunicationSchedule() {
    std::map<KeyTriple, unsigned> best_comm_schedule;
    unsigned best_comm_cost = UINT_MAX;

    if (hasValidCommSchedule()) {
        unsigned costs_com = computeCosts();
        if (costs_com < best_comm_cost) {
            best_comm_schedule = commSchedule;
            best_comm_cost = costs_com;
        }
    }

    setImprovedLazyCommunicationSchedule();
    unsigned costs_com = computeCosts();
    // std::cout << "Improved Lazy: " << costs_com << std::endl;
    if (costs_com < best_comm_cost) {
        best_comm_schedule = commSchedule;
        best_comm_cost = costs_com;
    }

    setLazyCommunicationSchedule();
    costs_com = computeCosts();
    // std::cout << "Lazy: " << costs_com << std::endl;
    if (costs_com < best_comm_cost) {
        best_comm_schedule = commSchedule;
        best_comm_cost = costs_com;
    }

    setEagerCommunicationSchedule();
    costs_com = computeCosts();
    // std::cout << "Eager: " << costs_com << std::endl;
    if (costs_com < best_comm_cost) {
        best_comm_schedule = commSchedule;
        best_comm_cost = costs_com;
    }

    commSchedule = best_comm_schedule;
}

void BspSchedule::setCommunicationSchedule(const std::map<KeyTriple, unsigned int> &cs) {
    if (checkCommScheduleValidity(cs)) {
        commSchedule.clear();
        commSchedule = std::map<KeyTriple, unsigned int>(cs);

    } else {
        throw std::invalid_argument("Given communication schedule is not valid for instance");
    }
}

unsigned BspSchedule::computeWorkCosts() const {

    assert(satisfiesPrecedenceConstraints());

    std::vector<std::vector<unsigned>> work = std::vector<std::vector<unsigned>>(
        number_of_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
        work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]] +=
            instance->getComputationalDag().nodeWorkWeight(node);
    }

    unsigned total_costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        unsigned max_work = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (max_work < work[step][proc]) {
                max_work = work[step][proc];
            }
        }

        total_costs += max_work;
    }

    return total_costs;
}

unsigned BspSchedule::computeCosts() const {

    assert(satisfiesPrecedenceConstraints());
    assert(hasValidCommSchedule());

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

    for (auto const &[key, val] : commSchedule) {

        send[val][get<1>(key)] += instance->sendCosts(get<1>(key), get<2>(key)) *
                                  instance->getComputationalDag().nodeCommunicationWeight(get<0>(key));
        rec[val][get<2>(key)] += instance->sendCosts(get<1>(key), get<2>(key)) *
                                 instance->getComputationalDag().nodeCommunicationWeight(get<0>(key));
    }

    unsigned total_costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        unsigned max_comm = 0;
        unsigned max_work = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            if (max_comm < send[step][proc])
                max_comm = send[step][proc];
            if (max_comm < rec[step][proc])
                max_comm = rec[step][proc];

            if (max_work < work[step][proc]) {
                max_work = work[step][proc];
            }
        }

        total_costs += max_work;
        if (max_comm > 0) {
            total_costs += instance->synchronisationCosts() + max_comm * instance->communicationCosts();
        }
    }

    return total_costs;
}

unsigned BspSchedule::computeCostsBufferedSending() const {

    assert(satisfiesPrecedenceConstraints());

    std::vector<unsigned> comm = std::vector<unsigned>(number_of_supersteps, 0);
    std::vector<std::vector<unsigned>> rec(instance->numberOfProcessors(),
                                           std::vector<unsigned>(number_of_supersteps, 0));
    std::vector<std::vector<unsigned>> send(instance->numberOfProcessors(),
                                            std::vector<unsigned>(number_of_supersteps, 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {

        std::vector<unsigned> step_needed(instance->numberOfProcessors(), number_of_supersteps);
        for (const auto &target : instance->getComputationalDag().children(node)) {

            if (node_to_processor_assignment[node] != node_to_processor_assignment[target]) {
                step_needed[node_to_processor_assignment[target]] =
                    std::min(step_needed[node_to_processor_assignment[target]], node_to_superstep_assignment[target]);
            }
        }

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (step_needed[proc] < number_of_supersteps) {
                send[node_to_processor_assignment[node]][node_to_superstep_assignment[node]] +=
                    instance->sendCosts(node_to_processor_assignment[node], proc) *
                    instance->getComputationalDag().nodeCommunicationWeight(node);

                rec[proc][step_needed[proc] - 1] += instance->sendCosts(node_to_processor_assignment[node], proc) *
                                                    instance->getComputationalDag().nodeCommunicationWeight(node);
            }
        }
    }

    for (unsigned step = 0; step < number_of_supersteps; step++) {
        unsigned max_send = 0;
        unsigned max_rec = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            if (max_send < send[proc][step])
                max_send = send[proc][step];
            if (max_rec < rec[proc][step])
                max_rec = rec[proc][step];
        }
        comm[step] = std::max(max_send, max_rec);
    }

    std::vector<unsigned> sync = std::vector<unsigned>(number_of_supersteps, 0);
    for (unsigned step = 0; step < number_of_supersteps; step++) {
        if (comm[step] > 0)
            sync[step] = 1;
    }

    std::vector<std::vector<unsigned>> work = std::vector<std::vector<unsigned>>(
        instance->numberOfProcessors(), std::vector<unsigned>(number_of_supersteps, 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
        work[node_to_processor_assignment[node]][node_to_superstep_assignment[node]] +=
            instance->getComputationalDag().nodeWorkWeight(node);
    }

    std::vector<unsigned> work_step = std::vector<unsigned>(number_of_supersteps, 0);
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        unsigned max_work = 0;
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (max_work < work[proc][step]) {
                max_work = work[proc][step];
            }
        }
        work_step[step] = max_work;
    }

    unsigned costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {
        costs += work_step[step] + comm[step] * instance->communicationCosts() +
                 sync[step] * instance->synchronisationCosts();
    }

    return costs;
}

unsigned BspSchedule::computeBaseCommCostsBufferedSending() const {

    assert(satisfiesPrecedenceConstraints());

    // std::vector<unsigned> comm = std::vector<unsigned>(number_of_supersteps, 0);
    std::vector<std::vector<unsigned>> rec(instance->numberOfProcessors(),
                                           std::vector<unsigned>(number_of_supersteps, 0));
    std::vector<std::vector<unsigned>> send(instance->numberOfProcessors(),
                                            std::vector<unsigned>(number_of_supersteps, 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {

        std::vector<unsigned> step_needed(instance->numberOfProcessors(), number_of_supersteps);
        for (const auto &target : instance->getComputationalDag().children(node)) {

            if (node_to_processor_assignment[node] != node_to_processor_assignment[target]) {
                step_needed[node_to_processor_assignment[target]] =
                    std::min(step_needed[node_to_processor_assignment[target]], node_to_superstep_assignment[target]);
            }
        }

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (step_needed[proc] < number_of_supersteps) {
                send[node_to_processor_assignment[node]][node_to_superstep_assignment[node]] +=
                    instance->sendCosts(node_to_processor_assignment[node], proc) *
                    instance->getComputationalDag().nodeCommunicationWeight(node);

                rec[proc][step_needed[proc] - 1] += instance->sendCosts(node_to_processor_assignment[node], proc) *
                                                    instance->getComputationalDag().nodeCommunicationWeight(node);
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

double BspSchedule::computeCostsTotalCommunication() const {

    assert(satisfiesPrecedenceConstraints());

    double total_communication = 0;

    for (const auto &edge : instance->getComputationalDag().edges()) {

        const unsigned &source = instance->getComputationalDag().source(edge);
        const unsigned &target = instance->getComputationalDag().target(edge);

        if (node_to_processor_assignment[source] != node_to_processor_assignment[target]) {
            total_communication +=
                instance->sendCosts(node_to_processor_assignment[source], node_to_processor_assignment[target]) *
                instance->getComputationalDag().nodeCommunicationWeight(source);
        }
    }

    std::vector<std::vector<unsigned>> work = std::vector<std::vector<unsigned>>(
        number_of_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
        work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]] +=
            instance->getComputationalDag().nodeWorkWeight(node);
    }

    unsigned total_work = 0;
    std::vector<unsigned> work_step = std::vector<unsigned>(number_of_supersteps, 0);
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        unsigned max_work = 0;
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (max_work < work[step][proc]) {
                max_work = work[step][proc];
            }
        }
        total_work += max_work;
    }

    return total_work + total_communication * instance->communicationCosts() * (1.0 / instance->numberOfProcessors()) +
           (number_of_supersteps - 1) * instance->synchronisationCosts();
}

double BspSchedule::computeBaseCommCostsTotalCommunication() const {

    assert(satisfiesPrecedenceConstraints());

    double total_communication = 0;

    for (const auto &edge : instance->getComputationalDag().edges()) {

        const unsigned &source = instance->getComputationalDag().source(edge);
        const unsigned &target = instance->getComputationalDag().target(edge);

        if (node_to_processor_assignment[source] != node_to_processor_assignment[target]) {
            total_communication +=
                instance->sendCosts(node_to_processor_assignment[source], node_to_processor_assignment[target]) *
                instance->getComputationalDag().nodeCommunicationWeight(source);
        }
    }

    return total_communication * (1.0 / instance->numberOfProcessors());
}

unsigned BspSchedule::computeBaseCommCost() const {
    std::vector<std::vector<unsigned>> rec(number_of_supersteps,
                                           std::vector<unsigned>(instance->numberOfProcessors(), 0));
    std::vector<std::vector<unsigned>> send(number_of_supersteps,
                                            std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (auto const &[key, val] : commSchedule) {

        send[val][get<1>(key)] += instance->getComputationalDag().nodeCommunicationWeight(get<0>(key));
        rec[val][get<2>(key)] += instance->getComputationalDag().nodeCommunicationWeight(get<0>(key));
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

unsigned BspSchedule::num_assigned_nodes(unsigned processor) const {

    unsigned num = 0;

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        if (node_to_processor_assignment[i] == processor) {
            num++;
        }
    }

    return num;
}

std::vector<unsigned> BspSchedule::num_assigned_nodes_per_processor() const {

    std::vector<unsigned> num(instance->numberOfProcessors(), 0);

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        num[node_to_processor_assignment[i]]++;
    }

    return num;
}

std::vector<std::vector<unsigned>> BspSchedule::num_assigned_nodes_per_superstep_processor() const {

    std::vector<std::vector<unsigned>> num(number_of_supersteps,
                                           std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        num[node_to_superstep_assignment[i]][node_to_processor_assignment[i]] += 1;
    }

    return num;
}