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

#include "model/BspScheduleRecomp.hpp"

void BspScheduleRecomp::addCommunicationScheduleEntry(unsigned node, unsigned from_proc, unsigned to_proc,
                                                      unsigned step) {
    addCommunicationScheduleEntry(std::make_tuple(node, from_proc, to_proc), step);
}

void BspScheduleRecomp::addCommunicationScheduleEntry(KeyTriple key, unsigned step) {

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

bool BspScheduleRecomp::satisfiesPrecedenceConstraints() const {

 //   throw std::runtime_error("Not implemented yet.");
    /*
        std::vector<unsigned> node_superstep = std::vector<unsigned>(instance->numberOfVertices(),
       number_of_supersteps); std::vector<std::vector<unsigned>> node_to_processor_assignment =
            std::vector<std::vector<unsigned>>(instance->numberOfVertices(), std::vector<unsigned>());

        for (unsigned step = 0; step < number_of_supersteps; step++) {
            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                for (const auto &node : step_processor_vertices[step][proc]) {

                    if (node_superstep[node] < step)
                        node_superstep[node] = step;

                    node_to_processor_assignment[node].push_back(proc);
                }
            }
        }

        for (const auto &ep : instance->getComputationalDag().edges()) {
            const unsigned &source = instance->getComputationalDag().source(ep);
            const unsigned &target = instance->getComputationalDag().target(ep);

            for (const auto &t_proc : node_to_processor_assignment[target]) {

                int different_processors = 1;
                for (const auto &s_proc : node_to_processor_assignment[source]) {

                    if (s_proc == t_proc) {
                        different_processors = 0;
                    }
                }
            }

            const int different_processors = 1;

            if (node_to_superstep_assignment[source] + different_processors > node_to_superstep_assignment[target]) {
                // std::cout << "This is not a valid scheduling (problems with nodes " << source << " and " << target <<
                // ")."
                //           << std::endl; // todo should be removed
                return false;
            }
        }
    */
    return true;
};

bool BspScheduleRecomp::satisfiesMemoryConstraints() const {

    std::vector<std::vector<unsigned>> step_proc_memory(number_of_supersteps,
                                                        std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {

        for (size_t i = 0; i < node_processor_assignment[node].size(); i++) {
            step_proc_memory[node_superstep_assignment[node][i]][node_processor_assignment[node][i]] +=
                instance->getComputationalDag().nodeMemoryWeight(node);
        }
    }

    for (unsigned step = 0; step < number_of_supersteps; step++) {

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (step_proc_memory[step][proc] > instance->getArchitecture().memoryBound(proc)) {
                return false;
            }
        }
    }

    return true;
};

bool BspScheduleRecomp::hasValidCommSchedule() const { return checkCommScheduleValidity(commSchedule); }

bool BspScheduleRecomp::checkCommScheduleValidity(const std::map<KeyTriple, unsigned int> &cs) const {

    //throw std::runtime_error("Not implemented yet.");

    /*
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
    */
    return true;
}

unsigned BspScheduleRecomp::computeWorkCosts() const {

    assert(satisfiesPrecedenceConstraints());

    std::vector<std::vector<unsigned>> step_proc_work(number_of_supersteps,
                                                      std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {

        for (size_t i = 0; i < node_processor_assignment[node].size(); i++) {
            step_proc_work[node_superstep_assignment[node][i]][node_processor_assignment[node][i]] +=
                instance->getComputationalDag().nodeWorkWeight(node);
        }
    }

    unsigned total_costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        unsigned max_work = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (max_work < step_proc_work[step][proc]) {
                max_work = step_proc_work[step][proc];
            }
        }

        total_costs += max_work;
    }

    return total_costs;
}

unsigned BspScheduleRecomp::computeCosts() const {
    
    assert(satisfiesPrecedenceConstraints());
    assert(hasValidCommSchedule());

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

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            if (max_comm < send[step][proc])
                max_comm = send[step][proc];
            if (max_comm < rec[step][proc])
                max_comm = rec[step][proc];

        }

        if (max_comm > 0) {
            total_costs += instance->synchronisationCosts() + max_comm * instance->communicationCosts();
        }
    }

    total_costs += computeWorkCosts();

    return total_costs;
        
}

unsigned BspScheduleRecomp::computeCostsBufferedSending() const {

    throw std::runtime_error("Not implemented yet.");
    /*
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
                        std::min(step_needed[node_to_processor_assignment[target]],
       node_to_superstep_assignment[target]);
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
        */
}

double BspScheduleRecomp::computeCostsTotalCommunication() const {

    throw std::runtime_error("Not implemented yet.");
    /*
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

        return total_work + total_communication * instance->communicationCosts() * (1.0 /
       instance->numberOfProcessors()) + (number_of_supersteps - 1) * instance->synchronisationCosts();
               */
}