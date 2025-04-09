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

#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <vector>

#include "SetSchedule.hpp"

namespace osp {

    

/**
 * @class BspSchedule
 * @brief Represents a schedule for the Bulk Synchronous Parallel (BSP) model.
 *
 * The `BspSchedule` class is responsible for managing the assignment of nodes to processors and supersteps in the BSP
 * model. It stores information such as the number of supersteps, the assignment of nodes to processors and supersteps,
 * and the communication schedule.
 *
 * The class provides methods for setting and retrieving the assigned superstep and processor for a given node, as well
 * as methods for checking the validity of the communication schedule and computing the costs of the schedule. It also
 * provides methods for setting the assigned supersteps and processors based on external assignments, and for updating
 * the number of supersteps.
 *
 * The `BspSchedule` class is designed to work with a `BspInstance` object, which represents the instance of the BSP
 * problem being solved.
 *
 * @see BspInstance
 */
template<typename Graph_t>
class BspSchedule : IBspSchedule<Graph_t> {
public:

    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;
    
private:
    using vertex_idx = vertex_idx_t<Graph_t>;

    const BspInstance<Graph_t> *instance;

    unsigned number_of_supersteps;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<unsigned> node_to_superstep_assignment;

    // contains entries: (vertex, from_proc, to_proc ) : step
    std::map<KeyTriple, unsigned> commSchedule;

public:



    /**
     * @brief Default constructor for the BspSchedule class.
     */
    BspSchedule() : instance(nullptr), number_of_supersteps(0) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    BspSchedule(const BspInstance<Graph_t> &inst)
        : instance(&inst), number_of_supersteps(1),
          node_to_processor_assignment(std::vector<unsigned>(inst.numberOfVertices(), 0)),
          node_to_superstep_assignment(std::vector<unsigned>(inst.numberOfVertices(), 0)) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep
     * assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */
    BspSchedule(const BspInstance<Graph_t> &inst, const std::vector<unsigned> &processor_assignment_,
                const std::vector<unsigned> &superstep_assignment_)
        : instance(&inst), node_to_processor_assignment(processor_assignment_),
          node_to_superstep_assignment(superstep_assignment_) {
        updateNumberOfSupersteps();
    }

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, superstep
     * assignment, and communication schedule.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     * @param comm_ The communication schedule for the nodes.
     */
    BspSchedule(const BspInstance<Graph_t> &inst, const std::vector<unsigned int> &processor_assignment_,
                const std::vector<unsigned int> &superstep_assignment_, const std::map<KeyTriple, unsigned int> &comm_)
        : instance(&inst), node_to_processor_assignment(processor_assignment_),
          node_to_superstep_assignment(superstep_assignment_), commSchedule(comm_) {

        updateNumberOfSupersteps();
    }

    virtual ~BspSchedule() = default;

    /**
     * @brief Returns a reference to the BspInstance for the schedule.
     *
     * @return A reference to the BspInstance for the schedule.
     */
    inline const BspInstance<Graph_t> &getInstance() const override { return *instance; }

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    inline unsigned numberOfSupersteps() const override { return number_of_supersteps; }

    /**
     * @brief Returns the number of processors in the schedule.
     *
     * @return The number of processors in the schedule.
     */
    void updateNumberOfSupersteps() {

        number_of_supersteps = 0;

        for (unsigned i = 0; i < instance->numberOfVertices(); ++i) {

            if (node_to_superstep_assignment[i] >= number_of_supersteps) {
                number_of_supersteps = node_to_superstep_assignment[i] + 1;
            }
        }
    }

    /**
     * @brief Returns the superstep assigned to the specified node.
     *
     * @param node The node for which to return the assigned superstep.
     * @return The superstep assigned to the specified node.
     */
    inline unsigned assignedSuperstep(vertex_idx node) const override { return node_to_superstep_assignment[node]; }

    /**
     * @brief Returns the processor assigned to the specified node.
     *
     * @param node The node for which to return the assigned processor.
     * @return The processor assigned to the specified node.
     */
    inline unsigned assignedProcessor(vertex_idx node) const override { return node_to_processor_assignment[node]; }

    /**
     * @brief Returns the superstep assignment for the schedule.
     *
     * @return The superstep assignment for the schedule.
     */
    inline const std::vector<unsigned> &assignedSupersteps() const { return node_to_superstep_assignment; }

    /**
     * @brief Returns the processor assignment for the schedule.
     *
     * @return The processor assignment for the schedule.
     */
    inline const std::vector<unsigned> &assignedProcessors() const { return node_to_processor_assignment; }

    /**
     * @brief Sets the superstep assigned to the specified node.
     *
     * @param node The node for which to set the assigned superstep.
     * @param superstep The superstep to assign to the node.
     */
    void setAssignedSuperstep(vertex_idx node, unsigned superstep) {

        if (node < instance->numberOfVertices()) {
            node_to_superstep_assignment[node] = superstep;

            if (superstep >= number_of_supersteps) {
                number_of_supersteps = superstep + 1;
            }

        } else {
            throw std::invalid_argument("Invalid Argument while assigning node to superstep: index out of range.");
        }
    }

    /**
     * @brief Sets the processor assigned to the specified node.
     *
     * @param node The node for which to set the assigned processor.
     * @param processor The processor to assign to the node.
     */
    void setAssignedProcessor(vertex_idx node, unsigned processor) {

        if (node < instance->numberOfVertices() && processor < instance->numberOfProcessors()) {
            node_to_processor_assignment[node] = processor;
        } else {
            // std::cout << "node " << node << " num nodes " << instance->numberOfVertices() << "  processor " <<
            // processor
            //          << " num proc " << instance->numberOfProcessors() << std::endl;
            throw std::invalid_argument("Invalid Argument while assigning node to processor");
        }
    }

    /**
     * @brief Sets the superstep assignment for the schedule.
     *
     * @param vec The superstep assignment to set.
     */
    void setAssignedSupersteps(const std::vector<unsigned> &vec) {

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

    /**
     * @brief Sets the processor assignment for the schedule.
     *
     * @param vec The processor assignment to set.
     */
    void setAssignedProcessors(const std::vector<unsigned> &vec) {

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

    v_workw_t<Graph_t> computeWorkCosts() const {

        std::vector<std::vector<v_workw_t<Graph_t>>> work = std::vector<std::vector<v_workw_t<Graph_t>>>(
            number_of_supersteps, std::vector<v_workw_t<Graph_t>>(instance->numberOfProcessors(), 0));

        for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
            work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]] +=
                instance->getComputationalDag().vertex_work_weight(node);
        }

        v_workw_t<Graph_t> total_costs = 0;
        for (unsigned step = 0; step < number_of_supersteps; step++) {

            v_workw_t<Graph_t> max_work = 0;

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                if (max_work < work[step][proc]) {
                    max_work = work[step][proc];
                }
            }

            total_costs += max_work;
        }

        return total_costs;
    }

    v_workw_t<Graph_t> computeCosts() const {

        std::vector<std::vector<v_workw_t<Graph_t>>> work = std::vector<std::vector<v_workw_t<Graph_t>>>(
            number_of_supersteps, std::vector<v_workw_t<Graph_t>>(instance->numberOfProcessors(), 0));

        for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
            work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]] +=
                instance->getComputationalDag().vertex_work_weight(node);
        }

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(
            number_of_supersteps, std::vector<v_commw_t<Graph_t>>(instance->numberOfProcessors(), 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(
            number_of_supersteps, std::vector<v_commw_t<Graph_t>>(instance->numberOfProcessors(), 0));

        for (auto const &[key, val] : commSchedule) {

            send[val][std::get<1>(key)] += instance->sendCosts(std::get<1>(key), std::get<2>(key)) *
                                           instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
            rec[val][std::get<2>(key)] += instance->sendCosts(std::get<1>(key), std::get<2>(key)) *
                                          instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
        }

        v_workw_t<Graph_t> total_costs = 0;
        for (unsigned step = 0; step < number_of_supersteps; step++) {

            v_commw_t<Graph_t> max_comm = 0;
            v_workw_t<Graph_t> max_work = 0;

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

    v_commw_t<Graph_t> computeBaseCommCost() const {

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(
            number_of_supersteps, std::vector<v_commw_t<Graph_t>>(instance->numberOfProcessors(), 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(
            number_of_supersteps, std::vector<v_commw_t<Graph_t>>(instance->numberOfProcessors(), 0));

        for (auto const &[key, val] : commSchedule) {

            send[val][std::get<1>(key)] += instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
            rec[val][std::get<2>(key)] += instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
        }

        v_commw_t<Graph_t> base_comm_cost = 0;
        for (unsigned step = 0; step < number_of_supersteps; step++) {

            v_commw_t<Graph_t> max_comm = 0;

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

    v_workw_t<Graph_t> computeCostsBufferedSending() const {

        std::vector<v_commw_t<Graph_t>> comm = std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0);

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(instance->numberOfProcessors(),
                                                         std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(instance->numberOfProcessors(),
                                                          std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));

        for (unsigned node = 0; node < instance->numberOfVertices(); node++) {

            std::vector<unsigned> step_needed(instance->numberOfProcessors(), number_of_supersteps);
            for (const auto &target : instance->getComputationalDag().children(node)) {

                if (node_to_processor_assignment[node] != node_to_processor_assignment[target]) {
                    step_needed[node_to_processor_assignment[target]] = std::min(
                        step_needed[node_to_processor_assignment[target]], node_to_superstep_assignment[target]);
                }
            }

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                if (step_needed[proc] < number_of_supersteps) {
                    send[node_to_processor_assignment[node]][node_to_superstep_assignment[node]] +=
                        instance->sendCosts(node_to_processor_assignment[node], proc) *
                        instance->getComputationalDag().vertex_comm_weight(node);

                    rec[proc][step_needed[proc] - 1] += instance->sendCosts(node_to_processor_assignment[node], proc) *
                                                        instance->getComputationalDag().vertex_comm_weight(node);
                }
            }
        }

        for (unsigned step = 0; step < number_of_supersteps; step++) {
            v_commw_t<Graph_t> max_send = 0;
            v_commw_t<Graph_t> max_rec = 0;

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                if (max_send < send[proc][step])
                    max_send = send[proc][step];
                if (max_rec < rec[proc][step])
                    max_rec = rec[proc][step];
            }
            comm[step] = std::max(max_send, max_rec);
        }

        v_commw_t<Graph_t> sync = 0;
        for (unsigned step = 0; step < number_of_supersteps; step++) {
            if (comm[step] > 0)
                sync += instance->synchronisationCosts();
        }

        std::vector<std::vector<v_workw_t<Graph_t>>> work = std::vector<std::vector<v_workw_t<Graph_t>>>(
            instance->numberOfProcessors(), std::vector<v_workw_t<Graph_t>>(number_of_supersteps, 0));

        for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
            work[node_to_processor_assignment[node]][node_to_superstep_assignment[node]] +=
                instance->getComputationalDag().vertex_work_weight(node);
        }

        std::vector<v_workw_t<Graph_t>> work_step = std::vector<v_workw_t<Graph_t>>(number_of_supersteps, 0);
        for (unsigned step = 0; step < number_of_supersteps; step++) {

            v_workw_t<Graph_t> max_work = 0;
            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                if (max_work < work[proc][step]) {
                    max_work = work[proc][step];
                }
            }
            work_step[step] = max_work;
        }

        v_workw_t<Graph_t> costs = 0;
        for (unsigned step = 0; step < number_of_supersteps; step++) {
            costs += work_step[step] + comm[step] * instance->communicationCosts();
        }

        return costs + sync;
    }

    v_commw_t<Graph_t> computeBaseCommCostsBufferedSending() const {

        // std::vector<unsigned> comm = std::vector<unsigned>(number_of_supersteps, 0);
        std::vector<std::vector<v_commw_t<Graph_t>>> rec(instance->numberOfProcessors(),
                                                         std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(instance->numberOfProcessors(),
                                                          std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));

        for (vertex_idx node = 0; node < instance->numberOfVertices(); node++) {

            std::vector<unsigned> step_needed(instance->numberOfProcessors(), number_of_supersteps);
            for (const auto &target : instance->getComputationalDag().children(node)) {

                if (node_to_processor_assignment[node] != node_to_processor_assignment[target]) {
                    step_needed[node_to_processor_assignment[target]] = std::min(
                        step_needed[node_to_processor_assignment[target]], node_to_superstep_assignment[target]);
                }
            }

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                if (step_needed[proc] < number_of_supersteps) {
                    send[node_to_processor_assignment[node]][node_to_superstep_assignment[node]] +=
                        instance->sendCosts(node_to_processor_assignment[node], proc) *
                        instance->getComputationalDag().vertex_comm_weight(node);

                    rec[proc][step_needed[proc] - 1] += instance->sendCosts(node_to_processor_assignment[node], proc) *
                                                        instance->getComputationalDag().vertex_comm_weight(node);
                }
            }
        }

        v_commw_t<Graph_t> costs = 0;
        for (unsigned step = 0; step < number_of_supersteps; step++) {
            v_commw_t<Graph_t> max_send = 0;
            v_commw_t<Graph_t> max_rec = 0;

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

    double computeCostsTotalCommunication() const {

        assert(satisfiesPrecedenceConstraints());

        double total_communication = 0;

        for (const auto &v : instance->getComputationalDag().vertices()) {
            for (const auto &target : instance->getComputationalDag().children(v)) {

                if (node_to_processor_assignment[v] != node_to_processor_assignment[target]) {
                    total_communication += (double)(instance->sendCosts(node_to_processor_assignment[v],
                                                                        node_to_processor_assignment[target]) *
                                                    instance->getComputationalDag().vertex_comm_weight(v));
                }
            }
        }

        std::vector<std::vector<v_workw_t<Graph_t>>> work = std::vector<std::vector<v_workw_t<Graph_t>>>(
            number_of_supersteps, std::vector<v_workw_t<Graph_t>>(instance->numberOfProcessors(), 0));

        for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
            work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]] +=
                instance->getComputationalDag().vertex_work_weight(node);
        }

        v_workw_t<Graph_t> total_work = 0;
        std::vector<v_workw_t<Graph_t>> work_step = std::vector<v_workw_t<Graph_t>>(number_of_supersteps, 0);
        for (unsigned step = 0; step < number_of_supersteps; step++) {

            v_workw_t<Graph_t> max_work = 0;
            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                if (max_work < work[step][proc]) {
                    max_work = work[step][proc];
                }
            }
            total_work += max_work;
        }

        v_commw_t<Graph_t> sync_cost = 0;

        if (number_of_supersteps >= 1)
            sync_cost = instance->synchronisationCosts() * static_cast<v_commw_t<Graph_t>>(number_of_supersteps - 1);

        return (double) total_work +
               total_communication * instance->communicationCosts() * (1.0 / instance->numberOfProcessors()) + sync_cost;
    }

    double computeBaseCommCostsTotalCommunication() const {

        assert(satisfiesPrecedenceConstraints());

        double total_communication = 0;

        for (const auto &v : instance->getComputationalDag().vertices()) {
            for (const auto &target : instance->getComputationalDag().children(v)) {

                if (node_to_processor_assignment[v] != node_to_processor_assignment[target]) {
                    total_communication += instance->getComputationalDag().vertex_comm_weight(v);
                }
            }
        }

        return total_communication * (1.0 / instance->numberOfProcessors());
    }

    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u
     * and v are assigned to different processors, the superstep assigned to node u is less than the superstep assigned
     * to node v.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    bool satisfiesPrecedenceConstraints() const {

        if (node_to_processor_assignment.size() != instance->numberOfVertices() ||
            node_to_superstep_assignment.size() != instance->numberOfVertices()) {
            return false;
        }

        for (const auto &v : instance->getComputationalDag().vertices()) {
            for (const auto &target : instance->getComputationalDag().children(v)) {

                const unsigned different_processors =
                    (node_to_processor_assignment[v] == node_to_processor_assignment[target]) ? 0u : 1u;

                if (node_to_superstep_assignment[v] + different_processors > node_to_superstep_assignment[target]) {
                    // std::cout << "This is not a valid scheduling (problems with nodes " << v << " and " << target <<
                    // ")."
                    //           << std::endl; // todo should be removed
                    return false;
                }
            }
        }

        return true;
    };

    bool satisfiesNodeTypeConstraints() const {

        if (node_to_processor_assignment.size() != instance->numberOfVertices())
            return false;

        for (unsigned int node = 0; node < instance->numberOfVertices(); node++) {
            if (!instance->isCompatible(node, node_to_processor_assignment[node]))
                return false;
        }

        return true;
    };

    bool satisfiesMemoryConstraints() const {

        switch (instance->getArchitecture().getMemoryConstraintType()) {

        case LOCAL: {

            SetSchedule set_schedule = SetSchedule(*this);

            for (unsigned step = 0; step < number_of_supersteps; step++) {
                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                    v_memw_t<Graph_t> memory = 0;
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
            std::vector<v_memw_t<Graph_t>> current_proc_persistent_memory(instance->numberOfProcessors(), 0);
            std::vector<v_memw_t<Graph_t>> current_proc_transient_memory(instance->numberOfProcessors(), 0);

            for (vertex_idx node = 0; node < instance->numberOfVertices(); node++) {

                const unsigned proc = node_to_processor_assignment[node];
                current_proc_persistent_memory[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
                current_proc_transient_memory[proc] = std::max(
                    current_proc_transient_memory[proc], instance->getComputationalDag().vertex_comm_weight(node));

                if (current_proc_persistent_memory[proc] + current_proc_transient_memory[proc] >
                    instance->getArchitecture().memoryBound(proc)) {
                    return false;
                }
            }
            break;
        }

        case GLOBAL: {
            std::vector<v_memw_t<Graph_t>> current_proc_memory(instance->numberOfProcessors(), 0);

            for (vertex_idx node = 0; node < instance->numberOfVertices(); node++) {

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

                    v_memw_t<Graph_t> memory = 0;
                    for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                        memory += instance->getComputationalDag().nodeMemoryWeight(node) +
                                  instance->getComputationalDag().vertex_comm_weight(node);

                        for (const auto &parent : instance->getComputationalDag().parents(node)) {

                            if (node_to_processor_assignment[parent] == proc &&
                                node_to_superstep_assignment[parent] == step) {
                                memory -= instance->getComputationalDag().vertex_comm_weight(parent);
                            }
                        }
                    }

                    if (memory > instance->getArchitecture().memoryBound(proc)) {
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

                    std::unordered_set<vertex_idx_t<Graph_t>> nodes_with_incoming_edges;

                    v_memw_t<Graph_t> memory = 0;
                    for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                        memory += instance->getComputationalDag().vertex_comm_weight(node);

                        for (const auto &parent : instance->getComputationalDag().parents(node)) {

                            if (node_to_superstep_assignment[parent] != step) {
                                nodes_with_incoming_edges.insert(parent);
                            }
                        }
                    }

                    for (const auto &node : nodes_with_incoming_edges) {
                        memory += instance->getComputationalDag().vertex_comm_weight(node);
                    }

                    if (memory > instance->getArchitecture().memoryBound(proc)) {
                        return false;
                    }
                }
            }
            break;
        }

        case LOCAL_INC_EDGES_2: {

            SetSchedule set_schedule = SetSchedule(*this);

            for (unsigned step = 0; step < number_of_supersteps; step++) {
                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                    std::unordered_set<vertex_idx_t<Graph_t>> nodes_with_incoming_edges;

                    v_memw_t<Graph_t> memory = 0;
                    for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {

                        if (instance->getComputationalDag().isSource(node)) {
                            memory += instance->getComputationalDag().nodeMemoryWeight(node);
                        }

                        for (const auto &parent : instance->getComputationalDag().parents(node)) {

                            if (node_to_superstep_assignment[parent] != step) {
                                nodes_with_incoming_edges.insert(parent);
                            }
                        }
                    }

                    for (const auto &node : nodes_with_incoming_edges) {
                        memory += instance->getComputationalDag().vertex_comm_weight(node);
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

    std::vector<vertex_idx_t<Graph_t>> getAssignedNodeVector(unsigned int processor) const {

        std::vector<vertex_idx_t<Graph_t>> vec;

        for (vertex_idx_t<Graph_t> i = 0; i < instance->numberOfVertices(); i++) {

            if (node_to_processor_assignment[i] == processor) {
                vec.push_back(i);
            }
        }

        return vec;
    }

    std::vector<vertex_idx_t<Graph_t>> getAssignedNodeVector(unsigned int processor, unsigned int superstep) const {
        std::vector<vertex_idx_t<Graph_t>> vec;

        for (vertex_idx_t<Graph_t> i = 0; i < instance->numberOfVertices(); i++) {

            if (node_to_processor_assignment[i] == processor && node_to_superstep_assignment[i] == superstep) {
                vec.push_back(i);
            }
        }

        return vec;
    }

    inline void setNumberOfSupersteps(unsigned int number_of_supersteps_) {
        number_of_supersteps = number_of_supersteps_;
    }

    unsigned num_assigned_nodes(unsigned processor) const {

        unsigned num = 0;

        for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
            if (node_to_processor_assignment[i] == processor) {
                num++;
            }
        }

        return num;
    }

    std::vector<unsigned> num_assigned_nodes_per_processor() const {

        std::vector<unsigned> num(instance->numberOfProcessors(), 0);

        for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
            num[node_to_processor_assignment[i]]++;
        }

        return num;
    }

    std::vector<std::vector<unsigned>> num_assigned_nodes_per_superstep_processor() const {

        std::vector<std::vector<unsigned>> num(number_of_supersteps,
                                               std::vector<unsigned>(instance->numberOfProcessors(), 0));

        for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
            num[node_to_superstep_assignment[i]][node_to_processor_assignment[i]] += 1;
        }

        return num;
    }

    inline const std::map<KeyTriple, unsigned> &getCommunicationSchedule() const { return commSchedule; }
    inline std::map<KeyTriple, unsigned> &getCommunicationSchedule() { return commSchedule; }

    void addCommunicationScheduleEntry(KeyTriple key, unsigned step) {

        if (step >= number_of_supersteps)
            throw std::invalid_argument(
                "Invalid Argument while adding communication schedule entry: step out of range.");

        if (std::get<0>(key) >= instance->numberOfVertices())
            throw std::invalid_argument(
                "Invalid Argument while adding communication schedule entry: node out of range.");

        if (std::get<1>(key) >= instance->numberOfProcessors())
            throw std::invalid_argument(
                "Invalid Argument while adding communication schedule entry: from processor out of range.");

        if (std::get<2>(key) >= instance->numberOfProcessors())
            throw std::invalid_argument(
                "Invalid Argument while adding communication schedule entry: to processor out of range.");

        commSchedule[key] = step;
    }

    /**
     * @brief Adds an entry to the communication schedule.
     *
     * @param node The node resp. its data which is sent.
     * @param from_proc The processor from which the data is sent.
     * @param to_proc The processor to which the data is sent.
     * @param step The superstep in which the data is sent.
     */
    void addCommunicationScheduleEntry(unsigned node, unsigned from_proc, unsigned to_proc, unsigned step) {
        addCommunicationScheduleEntry(std::make_tuple(node, from_proc, to_proc), step);
    }

    /**
     * @brief Sets the communication schedule for the schedule.
     *
     * @param cs The communication schedule to set.
     */
    void setCommunicationSchedule(const std::map<KeyTriple, unsigned int> &cs) {

        if (checkCommScheduleValidity(cs)) {
            commSchedule = cs;
        } else {
            throw std::invalid_argument("Given communication schedule is not valid for instance");
        }
    }

    bool checkCommScheduleValidity(const std::map<KeyTriple, unsigned int> &cs) const {

        std::vector<std::vector<unsigned>> first_at = std::vector<std::vector<unsigned>>(
            instance->numberOfVertices(), std::vector<unsigned>(instance->numberOfProcessors(), number_of_supersteps));

        for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
            first_at[node][node_to_processor_assignment[node]] = node_to_superstep_assignment[node];
        }

        for (auto const &[key, val] : cs) {

            if (val >= number_of_supersteps)
                return false;

            if (std::get<0>(key) >= instance->numberOfVertices())
                return false;

            if (std::get<1>(key) >= instance->numberOfProcessors())
                return false;

            if (std::get<2>(key) >= instance->numberOfProcessors())
                return false;

            first_at[std::get<0>(key)][std::get<2>(key)] =
                std::min(first_at[std::get<0>(key)][std::get<2>(key)], val + 1);
        }

        for (auto const &[key, val] : cs) {

            if (val < first_at[std::get<0>(key)][std::get<1>(key)]) {
                return false;
            }
        }

        for (const auto &v : instance->getComputationalDag().vertices()) {
            for (const auto &target : instance->getComputationalDag().children(v)) {

                if (node_to_processor_assignment[v] != node_to_processor_assignment[target]) {
                    if (first_at[v][node_to_processor_assignment[target]] > node_to_superstep_assignment[target]) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    inline bool hasValidCommSchedule() const { return checkCommScheduleValidity(commSchedule); }

    void setAutoCommunicationSchedule() {
        std::map<KeyTriple, unsigned> best_comm_schedule;
        v_workw_t<Graph_t> best_comm_cost =
            std::numeric_limits<v_workw_t<Graph_t>>::max(); // computeCosts retunrs v_workw_t<Graph_t>

        if (hasValidCommSchedule()) {
            v_workw_t<Graph_t> costs_com = computeCosts();
            if (costs_com < best_comm_cost) {
                best_comm_schedule = commSchedule;
                best_comm_cost = costs_com;
            }
        }

        setImprovedLazyCommunicationSchedule();
        v_workw_t<Graph_t> costs_com = computeCosts();
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

    void setImprovedLazyCommunicationSchedule() {
        commSchedule.clear();
        if (instance->getComputationalDag().num_vertices() <= 1 || number_of_supersteps <= 1)
            return;

        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> step_proc_node_list(
            number_of_supersteps, std::vector<std::vector<vertex_idx_t<Graph_t>>>(
                                      instance->numberOfProcessors(), std::vector<vertex_idx_t<Graph_t>>()));
        std::vector<std::vector<bool>> node_to_proc_been_sent(instance->numberOfVertices(),
                                                              std::vector<bool>(instance->numberOfProcessors(), false));

        for (vertex_idx_t<Graph_t> node = 0; node < instance->numberOfVertices(); node++) {
            step_proc_node_list[node_to_superstep_assignment[node]][node_to_processor_assignment[node]].push_back(node);
            node_to_proc_been_sent[node][node_to_processor_assignment[node]] = true;
        }

        // processor, ordered list of (cost, node, to_processor)
        std::vector<std::set<std::vector<vertex_idx_t<Graph_t>>, std::greater<>>> require_sending(
            instance->numberOfProcessors());
        // TODO the datastructure seems to be wrong. the vectors added to the set have elements of different types.
        // it should really be std::vector<std::set<std::tuple<v_commw_t<Graph_t>, vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>>>> 
        // added many static_cast below as tmp fix

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto &node : step_proc_node_list[0][proc]) {

                for (const auto &target : instance->getComputationalDag().children(node)) {
                    if (proc != assignedProcessor(target)) {
                        require_sending[proc].insert(
                            {static_cast<vertex_idx_t<Graph_t>>(
                                 instance->getComputationalDag().vertex_comm_weight(node) *
                                 instance->getArchitecture().sendCosts(proc, node_to_processor_assignment[target])),
                             node, node_to_processor_assignment[target]});
                    }
                }
            }
        }

        for (unsigned step = 1; step < number_of_supersteps; step++) {
            std::vector<v_commw_t<Graph_t>> send_cost(instance->numberOfProcessors(), 0);
            std::vector<v_commw_t<Graph_t>> receive_cost(instance->numberOfProcessors(), 0);

            // must send in superstep step-1
            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                for (const auto &node : step_proc_node_list[step][proc]) {
                    for (const auto &source : instance->getComputationalDag().parents(node)) {

                        if (!node_to_proc_been_sent[source][proc]) {
                            assert(node_to_superstep_assignment[source] < step);
                            commSchedule.emplace(std::make_tuple(source, node_to_processor_assignment[source], proc),
                                                 step - 1);
                            node_to_proc_been_sent[source][proc] = true;
                            v_commw_t<Graph_t> comm_cost =
                                instance->getComputationalDag().vertex_comm_weight(source) *
                                instance->getArchitecture().sendCosts(node_to_processor_assignment[source], proc);
                            require_sending[assignedProcessor(source)].erase(
                                {static_cast<vertex_idx_t<Graph_t>>(comm_cost), source, proc});
                            send_cost[node_to_processor_assignment[source]] += comm_cost;
                            receive_cost[proc] += comm_cost;
                        }
                    }
                }
            }

            // getting max costs
            v_commw_t<Graph_t> max_comm_cost = 0;
            for (size_t proc = 0; proc < instance->numberOfProcessors(); proc++) {
                max_comm_cost = std::max(max_comm_cost, send_cost[proc]);
                max_comm_cost = std::max(max_comm_cost, receive_cost[proc]);
            }

            // extra sends
            // TODO: permute the order of processors
            for (size_t proc = 0; proc < instance->numberOfProcessors(); proc++) {
                if (require_sending[proc].empty() ||
                    static_cast<v_commw_t<Graph_t>>((*(require_sending[proc].rbegin()))[0]) + send_cost[proc] >
                        max_comm_cost)
                    continue;
                auto iter = require_sending[proc].begin();
                while (iter != require_sending[proc].cend()) {
                    if (static_cast<v_commw_t<Graph_t>>((*iter)[0]) + send_cost[proc] > max_comm_cost ||
                        static_cast<v_commw_t<Graph_t>>((*iter)[0]) + receive_cost[(*iter)[2]] > max_comm_cost) {
                        iter++;
                    } else {
                        commSchedule.emplace(std::make_tuple((*iter)[1], proc, (*iter)[2]), step - 1);
                        node_to_proc_been_sent[(*iter)[1]][(*iter)[2]] = true;
                        send_cost[proc] += static_cast<v_commw_t<Graph_t>>((*iter)[0]);
                        receive_cost[(*iter)[2]] += static_cast<v_commw_t<Graph_t>>((*iter)[0]);
                        iter = require_sending[proc].erase(iter);
                        if (require_sending[proc].empty() ||
                            static_cast<v_commw_t<Graph_t>>((*(require_sending[proc].rbegin()))[0]) + send_cost[proc] >
                                max_comm_cost)
                            break;
                    }
                }
            }

            // updating require_sending
            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                for (const auto &node : step_proc_node_list[step][proc]) {

                    for (const auto &target : instance->getComputationalDag().children(node))
                        if (proc != assignedProcessor(target)) {
                            require_sending[proc].insert(
                                {static_cast<vertex_idx_t<Graph_t>>(
                                     instance->getComputationalDag().vertex_comm_weight(node) *
                                     instance->getArchitecture().sendCosts(proc, node_to_processor_assignment[target])),
                                 node, node_to_processor_assignment[target]});
                        }
                }
            }
        }
    }

    void setLazyCommunicationSchedule() {
        commSchedule.clear();

        for (const auto &source : instance->getComputationalDag().vertices()) {
            for (const auto &target : instance->getComputationalDag().children(source)) {

                if (node_to_processor_assignment[source] != node_to_processor_assignment[target]) {

                    const auto tmp = std::make_tuple(source, node_to_processor_assignment[source],
                                                     node_to_processor_assignment[target]);
                    if (commSchedule.find(tmp) == commSchedule.end()) {
                        commSchedule[tmp] = node_to_superstep_assignment[target] - 1;

                    } else {
                        commSchedule[tmp] = std::min(node_to_superstep_assignment[target] - 1, commSchedule[tmp]);
                    }
                }
            }
        }
    }
    void setEagerCommunicationSchedule() {
        commSchedule.clear();

        for (const auto &source : instance->getComputationalDag().vertices()) {
            for (const auto &target : instance->getComputationalDag().children(source)) {

                if (node_to_processor_assignment[source] != node_to_processor_assignment[target]) {

                    commSchedule[std::make_tuple(source, node_to_processor_assignment[source],
                                                 node_to_processor_assignment[target])] =
                        node_to_superstep_assignment[source];
                }
            }
        }
    }
};

} // namespace osp