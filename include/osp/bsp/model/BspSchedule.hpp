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
#include <unordered_set>
#include <vector>

#include "IBspScheduleEval.hpp"
#include "IBspSchedule.hpp"
#include "SetSchedule.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

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
class BspSchedule : public IBspSchedule<Graph_t>, public IBspScheduleEval<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t> >, "BspSchedule requires work and comm. weights to have the same type.");

  protected:
    using vertex_idx = vertex_idx_t<Graph_t>;

    const BspInstance<Graph_t> *instance;

    unsigned number_of_supersteps;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<unsigned> node_to_superstep_assignment;

  public:
  
    BspSchedule() = delete;

    /**
     * @brief Constructs a BspSchedule object with the specified Bspinstance->
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

    BspSchedule(const IBspSchedule<Graph_t> &schedule)
        : instance(&schedule.getInstance()), number_of_supersteps(schedule.numberOfSupersteps()),
          node_to_processor_assignment(schedule.getInstance().numberOfVertices()),
          node_to_superstep_assignment(schedule.getInstance().numberOfVertices()) {

        for (const auto &v : schedule.getInstance().getComputationalDag().vertices()) {

            node_to_processor_assignment[v] = schedule.assignedProcessor(v);
            node_to_superstep_assignment[v] = schedule.assignedSuperstep(v);
        }
    }

    BspSchedule(const BspSchedule<Graph_t> &schedule)
        : instance(schedule.instance), number_of_supersteps(schedule.number_of_supersteps),
          node_to_processor_assignment(schedule.node_to_processor_assignment),
          node_to_superstep_assignment(schedule.node_to_superstep_assignment) {}

    BspSchedule<Graph_t> operator=(const BspSchedule<Graph_t> &schedule) {
        if (this != &schedule) {
            instance = schedule.instance;
            number_of_supersteps = schedule.number_of_supersteps;
            node_to_processor_assignment = schedule.node_to_processor_assignment;
            node_to_superstep_assignment = schedule.node_to_superstep_assignment;
        }
        return *this;
    }

    BspSchedule(BspSchedule<Graph_t> &&schedule)
        : instance(schedule.instance), number_of_supersteps(schedule.number_of_supersteps),
          node_to_processor_assignment(std::move(schedule.node_to_processor_assignment)),
          node_to_superstep_assignment(std::move(schedule.node_to_superstep_assignment)) {}

    BspSchedule<Graph_t> &operator=(BspSchedule<Graph_t> &&schedule) {
        if (this != &schedule) {
            instance = schedule.instance;
            number_of_supersteps = schedule.number_of_supersteps;
            node_to_processor_assignment = std::move(schedule.node_to_processor_assignment);
            node_to_superstep_assignment = std::move(schedule.node_to_superstep_assignment);
        }
        return *this;
    }

    template<typename Graph_t_other>
    BspSchedule(const BspInstance<Graph_t> &instance_, const BspSchedule<Graph_t_other> &schedule)
        : instance(&instance_), number_of_supersteps(schedule.numberOfSupersteps()),
          node_to_processor_assignment(schedule.assignedProcessors()),
          node_to_superstep_assignment(schedule.assignedSupersteps()) {}

    /**
     * @brief Destructor for the BspSchedule class.
     */
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
    inline std::vector<unsigned> &assignedSupersteps() { return node_to_superstep_assignment; }

    /**
     * @brief Returns the processor assignment for the schedule.
     *
     * @return The processor assignment for the schedule.
     */
    inline const std::vector<unsigned> &assignedProcessors() const { return node_to_processor_assignment; }
    inline std::vector<unsigned> &assignedProcessors() { return node_to_processor_assignment; }

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
     * @brief Sets the superstep assigned to the specified node.
     *
     * @param node The node for which to set the assigned superstep.
     * @param superstep The superstep to assign to the node.
     */
    inline void setAssignedSuperstep_noUpdateNumSuperstep(vertex_idx node, unsigned superstep) {
        node_to_superstep_assignment.at(node) = superstep;
    }

    /**
     * @brief Sets the processor assigned to the specified node.
     *
     * @param node The node for which to set the assigned processor.
     * @param processor The processor to assign to the node.
     */
    inline void setAssignedProcessor(vertex_idx node, unsigned processor) {
        node_to_processor_assignment.at(node) = processor;
    }

    /**
     * @brief Sets the superstep assignment for the schedule.
     *
     * @param vec The superstep assignment to set.
     */
    void setAssignedSupersteps(const std::vector<unsigned> &vec) {

        if (vec.size() == static_cast<std::size_t>( instance->numberOfVertices() )) {

            number_of_supersteps = 0;

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
     * @brief Sets the superstep assignment for the schedule.
     *
     * @param vec The superstep assignment to set.
     */
    void setAssignedSupersteps(std::vector<unsigned> &&vec) {

        if (vec.size() == static_cast<std::size_t>( instance->numberOfVertices() )) {
            node_to_superstep_assignment = std::move(vec);
        } else {
            throw std::invalid_argument(
                "Invalid Argument while assigning supersteps: size does not match number of nodes.");
        }
        
        updateNumberOfSupersteps();
    }

    /**
     * @brief Sets the processor assignment for the schedule.
     *
     * @param vec The processor assignment to set.
     */
    void setAssignedProcessors(const std::vector<unsigned> &vec) {

        if (vec.size() == static_cast<std::size_t>( instance->numberOfVertices() )) {
            node_to_processor_assignment = vec;
        } else {
            throw std::invalid_argument(
                "Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    /**
     * @brief Sets the processor assignment for the schedule.
     *
     * @param vec The processor assignment to set.
     */
    void setAssignedProcessors(std::vector<unsigned> &&vec) {

        if (vec.size() == static_cast<std::size_t>( instance->numberOfVertices() )) {
            node_to_processor_assignment = std::move(vec);
        } else {
            throw std::invalid_argument(
                "Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    virtual v_workw_t<Graph_t> computeWorkCosts() const override {

        std::vector<std::vector<v_workw_t<Graph_t>>> work = std::vector<std::vector<v_workw_t<Graph_t>>>(
            number_of_supersteps, std::vector<v_workw_t<Graph_t>>(instance->numberOfProcessors(), 0));

        for (const auto &node : instance->vertices()) {
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

    double compute_total_communication_costs() const {

        assert(satisfiesPrecedenceConstraints());

        double total_communication = 0;

        for (const auto &v : instance->vertices()) {
            for (const auto &target : instance->getComputationalDag().children(v)) {

                if (node_to_processor_assignment[v] != node_to_processor_assignment[target]) {
                    total_communication +=
                        instance->sendCosts(node_to_processor_assignment[v], node_to_processor_assignment[target]) *
                        instance->getComputationalDag().vertex_comm_weight(v);
                }
            }
        }

        return total_communication * static_cast<double>(instance->communicationCosts()) / static_cast<double>(instance->numberOfProcessors());
    }

    double computeTotalCosts() const {

        assert(satisfiesPrecedenceConstraints());

        const v_commw_t<Graph_t> sync_cost =
            number_of_supersteps >= 1
                ? instance->synchronisationCosts() * static_cast<v_commw_t<Graph_t>>(number_of_supersteps - 1)
                : 0;

        return static_cast<double>(computeWorkCosts()) + compute_total_communication_costs() + sync_cost;
    }

    double compute_total_lambda_communication_cost() const {

        assert(satisfiesPrecedenceConstraints());

        double comm_costs = 0;
        const double comm_multiplier = 1.0 / instance->numberOfProcessors();

        for (const auto &v : instance->vertices()) {
            if (instance->getComputationalDag().out_degree(v) == 0)
                continue;

            std::unordered_set<unsigned> target_procs;
            for (const auto &target : instance->getComputationalDag().children(v)) {
                target_procs.insert(node_to_processor_assignment[target]);
            }

            const unsigned source_proc = node_to_processor_assignment[v];
            const auto v_comm_cost = instance->getComputationalDag().vertex_comm_weight(v);

            for (const auto& target_proc : target_procs) {
                comm_costs += v_comm_cost * instance->sendCosts(source_proc, target_proc);
            }
        }

        return comm_costs * comm_multiplier * static_cast<double>(instance->communicationCosts());
    }
    
    double computeTotalLambdaCosts() const {
        assert(satisfiesPrecedenceConstraints());

        const v_commw_t<Graph_t> sync_cost =
            number_of_supersteps >= 1
                ? instance->synchronisationCosts() * static_cast<v_commw_t<Graph_t>>(number_of_supersteps - 1)
                : 0;

        return static_cast<double>(computeWorkCosts()) + compute_total_lambda_communication_cost() + sync_cost;
    }

    v_commw_t<Graph_t> compute_buffered_sending_communication_costs() const {

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

            const auto step_comm_cost = std::max(max_send, max_rec) * instance->communicationCosts();

            costs += step_comm_cost;

            if (step_comm_cost > 0) {
                costs += instance->synchronisationCosts();
            }
        }

        return costs;
    }

    v_workw_t<Graph_t> computeBufferedSendingCosts() const {

        return compute_buffered_sending_communication_costs() + computeWorkCosts();
    }

    v_commw_t<Graph_t> compute_lazy_communication_costs() const {

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(instance->numberOfProcessors(),
                                                         std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));

        std::vector<std::vector<v_commw_t<Graph_t>>> send(instance->numberOfProcessors(),
                                                          std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));

        for (const auto &node : instance->vertices()) {

            std::vector<unsigned> step_needed(instance->numberOfProcessors(), number_of_supersteps);
            for (const auto &target : instance->getComputationalDag().children(node)) {

                if (node_to_processor_assignment[node] != node_to_processor_assignment[target]) {
                    step_needed[node_to_processor_assignment[target]] = std::min(
                        step_needed[node_to_processor_assignment[target]], node_to_superstep_assignment[target]);
                }
            }

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                if (step_needed[proc] < number_of_supersteps) {

                    send[node_to_processor_assignment[node]][step_needed[proc] - 1] +=
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

            const auto step_comm_cost = std::max(max_send, max_rec) * instance->communicationCosts();

            costs += step_comm_cost;

            if (step_comm_cost > 0) {
                costs += instance->synchronisationCosts();
            }
        }

        return costs;
    }

    virtual v_workw_t<Graph_t> computeCosts() const override { return compute_lazy_communication_costs() + computeWorkCosts(); }

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

        for (const auto &v : instance->vertices()) {

            if (node_to_superstep_assignment[v] >= number_of_supersteps) {
                return false;
            }

            if (node_to_processor_assignment[v] >= instance->numberOfProcessors()) {
                return false;
            }

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

        for (const auto &node : instance->vertices()) {
            if (!instance->isCompatible(node, node_to_processor_assignment[node]))
                return false;
        }

        return true;
    };

    bool satisfiesMemoryConstraints() const {

        switch (instance->getArchitecture().getMemoryConstraintType()) {

        case MEMORY_CONSTRAINT_TYPE::LOCAL: {

            SetSchedule set_schedule = SetSchedule(*this);

            for (unsigned step = 0; step < number_of_supersteps; step++) {
                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                    v_memw_t<Graph_t> memory = 0;
                    for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                        memory += instance->getComputationalDag().vertex_mem_weight(node);
                    }

                    if (memory > instance->getArchitecture().memoryBound(proc)) {
                        return false;
                    }
                }
            }

            break;
        }

        case MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT: {
            std::vector<v_memw_t<Graph_t>> current_proc_persistent_memory(instance->numberOfProcessors(), 0);
            std::vector<v_memw_t<Graph_t>> current_proc_transient_memory(instance->numberOfProcessors(), 0);

            for (const auto &node : instance->vertices()) {

                const unsigned proc = node_to_processor_assignment[node];
                current_proc_persistent_memory[proc] += instance->getComputationalDag().vertex_mem_weight(node);
                current_proc_transient_memory[proc] = std::max(
                    current_proc_transient_memory[proc], instance->getComputationalDag().vertex_comm_weight(node));

                if (current_proc_persistent_memory[proc] + current_proc_transient_memory[proc] >
                    instance->getArchitecture().memoryBound(proc)) {
                    return false;
                }
            }
            break;
        }

        case MEMORY_CONSTRAINT_TYPE::GLOBAL: {
            std::vector<v_memw_t<Graph_t>> current_proc_memory(instance->numberOfProcessors(), 0);

            for (const auto &node : instance->vertices()) {

                const unsigned proc = node_to_processor_assignment[node];
                current_proc_memory[proc] += instance->getComputationalDag().vertex_mem_weight(node);

                if (current_proc_memory[proc] > instance->getArchitecture().memoryBound(proc)) {
                    return false;
                }
            }
            break;
        }

        case MEMORY_CONSTRAINT_TYPE::LOCAL_IN_OUT: {

            SetSchedule set_schedule = SetSchedule(*this);

            for (unsigned step = 0; step < number_of_supersteps; step++) {
                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                    v_memw_t<Graph_t> memory = 0;
                    for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                        memory += instance->getComputationalDag().vertex_mem_weight(node) +
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

        case MEMORY_CONSTRAINT_TYPE::LOCAL_INC_EDGES: {

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

        case MEMORY_CONSTRAINT_TYPE::LOCAL_SOURCES_INC_EDGES: {

            SetSchedule set_schedule = SetSchedule(*this);

            for (unsigned step = 0; step < number_of_supersteps; step++) {
                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                    std::unordered_set<vertex_idx_t<Graph_t>> nodes_with_incoming_edges;

                    v_memw_t<Graph_t> memory = 0;
                    for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {

                        if (is_source(node, instance->getComputationalDag())) {
                            memory += instance->getComputationalDag().vertex_mem_weight(node);
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

        case MEMORY_CONSTRAINT_TYPE::NONE: {
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

        for (const auto &node : instance->vertices()) {

            if (node_to_processor_assignment[node] == processor) {
                vec.push_back(node);
            }
        }

        return vec;
    }

    std::vector<vertex_idx_t<Graph_t>> getAssignedNodeVector(unsigned int processor, unsigned int superstep) const {
        std::vector<vertex_idx_t<Graph_t>> vec;

        for (const auto &node : instance->vertices()) {

            if (node_to_processor_assignment[node] == processor && node_to_superstep_assignment[node] == superstep) {
                vec.push_back(node);
            }
        }

        return vec;
    }

    inline void setNumberOfSupersteps(unsigned int number_of_supersteps_) {
        number_of_supersteps = number_of_supersteps_;
    }

    unsigned num_assigned_nodes(unsigned processor) const {

        unsigned num = 0;

        for (const auto &node : instance->vertices()) {
            if (node_to_processor_assignment[node] == processor) {
                num++;
            }
        }

        return num;
    }

    std::vector<unsigned> num_assigned_nodes_per_processor() const {

        std::vector<unsigned> num(instance->numberOfProcessors(), 0);

        for (const auto &node : instance->vertices()) {
            num[node_to_processor_assignment[node]]++;
        }

        return num;
    }

    std::vector<std::vector<unsigned>> num_assigned_nodes_per_superstep_processor() const {

        std::vector<std::vector<unsigned>> num(number_of_supersteps,
                                               std::vector<unsigned>(instance->numberOfProcessors(), 0));

        for (const auto &v : instance->vertices()) {
            num[node_to_superstep_assignment[v]][node_to_processor_assignment[v]] += 1;
        }

        return num;
    }
};

} // namespace osp