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
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "IBspSchedule.hpp"
#include "IBspScheduleEval.hpp"
#include "SetSchedule.hpp"
#include "osp/bsp/model/cost/LazyCommunicationCost.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

/**
 * @class BspSchedule
 * @brief Represents a schedule for the Bulk Synchronous Parallel (BSP) model.
 *
 * The `BspSchedule` class manages the assignment of nodes to processors and supersteps within the BSP model.
 * It serves as a core component for scheduling algorithms, providing mechanisms to:
 * - Store and retrieve node-to-processor and node-to-superstep assignments.
 * - Validate schedules against precedence, memory, and node type constraints.
 * - Compute costs associated with the schedule.
 * - Manipulate the schedule, including updating assignments and merging supersteps.
 *
 * This class is templated on `Graph_t`, which must satisfy the `computational_dag_concept`.
 * Moreover, the work and communication weights of the nodes must be of the same type in order to properly compute the cost.
 *
 * It interacts closely with `BspInstance` to access problem-specific data and constraints. In fact, a `BspSchedule` object is tied to a `BspInstance` object.
 *
 * @tparam Graph_t The type of the computational DAG, which must satisfy `is_computational_dag_v`.
 * @see BspInstance
 * @see IBspSchedule
 * @see IBspScheduleEval
 */
template<typename Graph_t>
class BspSchedule : public IBspSchedule<Graph_t>, public IBspScheduleEval<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>, "BspSchedule requires work and comm. weights to have the same type.");

  protected:
    using vertex_idx = vertex_idx_t<Graph_t>;

    const BspInstance<Graph_t> *instance;

    unsigned number_of_supersteps;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<unsigned> node_to_superstep_assignment;

  public:
    BspSchedule() = delete;

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    explicit BspSchedule(const BspInstance<Graph_t> &inst)
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
     * @brief Copy constructor from an IBspSchedule.
     *
     * @param schedule The schedule to copy.
     */
    explicit BspSchedule(const IBspSchedule<Graph_t> &schedule)
        : instance(&schedule.getInstance()), number_of_supersteps(schedule.numberOfSupersteps()),
          node_to_processor_assignment(schedule.getInstance().numberOfVertices()),
          node_to_superstep_assignment(schedule.getInstance().numberOfVertices()) {
        for (const auto &v : schedule.getInstance().getComputationalDag().vertices()) {
            node_to_processor_assignment[v] = schedule.assignedProcessor(v);
            node_to_superstep_assignment[v] = schedule.assignedSuperstep(v);
        }
    }

    /**
     * @brief Copy constructor.
     *
     * @param schedule The schedule to copy.
     */
    BspSchedule(const BspSchedule<Graph_t> &schedule)
        : instance(schedule.instance), number_of_supersteps(schedule.number_of_supersteps),
          node_to_processor_assignment(schedule.node_to_processor_assignment),
          node_to_superstep_assignment(schedule.node_to_superstep_assignment) {}

    /**
     * @brief Copy assignment operator.
     *
     * @param schedule The schedule to copy.
     * @return A reference to this schedule.
     */
    BspSchedule<Graph_t> &operator=(const BspSchedule<Graph_t> &schedule) {
        if (this != &schedule) {
            instance = schedule.instance;
            number_of_supersteps = schedule.number_of_supersteps;
            node_to_processor_assignment = schedule.node_to_processor_assignment;
            node_to_superstep_assignment = schedule.node_to_superstep_assignment;
        }
        return *this;
    }

    /**
     * @brief Move constructor.
     *
     * @param schedule The schedule to move.
     */
    BspSchedule(BspSchedule<Graph_t> &&schedule) noexcept
        : instance(schedule.instance), number_of_supersteps(schedule.number_of_supersteps),
          node_to_processor_assignment(std::move(schedule.node_to_processor_assignment)),
          node_to_superstep_assignment(std::move(schedule.node_to_superstep_assignment)) {}

    /**
     * @brief Move assignment operator.
     *
     * @param schedule The schedule to move.
     * @return A reference to this schedule.
     */
    BspSchedule<Graph_t> &operator=(BspSchedule<Graph_t> &&schedule) noexcept {
        if (this != &schedule) {
            instance = schedule.instance;
            number_of_supersteps = schedule.number_of_supersteps;
            node_to_processor_assignment = std::move(schedule.node_to_processor_assignment);
            node_to_superstep_assignment = std::move(schedule.node_to_superstep_assignment);
        }
        return *this;
    }

    /**
     * @brief Constructs a BspSchedule object from another schedule with a different graph type.
     *
     * @tparam Graph_t_other The graph type of the other schedule.
     * @param instance_ The BspInstance for the new schedule.
     * @param schedule The other schedule to copy from.
     */
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
    [[nodiscard]] const BspInstance<Graph_t> &getInstance() const override { return *instance; }

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    [[nodiscard]] unsigned numberOfSupersteps() const override { return number_of_supersteps; }

    /**
     * @brief Updates the number of supersteps based on the current assignment.
     */
    void updateNumberOfSupersteps() {
        number_of_supersteps = 0;
        for (vertex_idx_t<Graph_t> i = 0; i < static_cast<vertex_idx_t<Graph_t>>(instance->numberOfVertices()); ++i) {
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
    [[nodiscard]] unsigned assignedSuperstep(vertex_idx node) const override { return node_to_superstep_assignment[node]; }

    /**
     * @brief Returns the processor assigned to the specified node.
     *
     * @param node The node for which to return the assigned processor.
     * @return The processor assigned to the specified node.
     */
    [[nodiscard]] unsigned assignedProcessor(vertex_idx node) const override { return node_to_processor_assignment[node]; }

    /**
     * @brief Returns the superstep assignment for the schedule.
     *
     * @return The superstep assignment for the schedule.
     */
    [[nodiscard]] const std::vector<unsigned> &assignedSupersteps() const { return node_to_superstep_assignment; }
    [[nodiscard]] std::vector<unsigned> &assignedSupersteps() { return node_to_superstep_assignment; }

    /**
     * @brief Returns the processor assignment for the schedule.
     *
     * @return The processor assignment for the schedule.
     */
    [[nodiscard]] const std::vector<unsigned> &assignedProcessors() const { return node_to_processor_assignment; }
    [[nodiscard]] std::vector<unsigned> &assignedProcessors() { return node_to_processor_assignment; }

    /**
     * @brief Returns the staleness of the schedule.
     * The staleness determines the minimum number of supersteps that must elapse between the assignment of a node to a processor and the assignment of one of its neighbors to a different processor.
     * The staleness for the BspSchedule is always 1.
     *
     * @return The staleness of the schedule.
     */
    [[nodiscard]] virtual unsigned getStaleness() const { return 1; }

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
     * @brief Sets the superstep assigned to the specified node without updating the number of supersteps.
     *
     * @param node The node for which to set the assigned superstep.
     * @param superstep The superstep to assign to the node.
     */
    void setAssignedSuperstepNoUpdateNumSuperstep(vertex_idx node, unsigned superstep) {
        node_to_superstep_assignment.at(node) = superstep;
    }

    /**
     * @brief Sets the processor assigned to the specified node.
     *
     * @param node The node for which to set the assigned processor.
     * @param processor The processor to assign to the node.
     */
    void setAssignedProcessor(vertex_idx node, unsigned processor) {
        node_to_processor_assignment.at(node) = processor;
    }

    /**
     * @brief Sets the superstep assignment for the schedule.
     *
     * @param vec The superstep assignment to set.
     */
    void setAssignedSupersteps(const std::vector<unsigned> &vec) {
        if (vec.size() == static_cast<std::size_t>(instance->numberOfVertices())) {
            number_of_supersteps = 0;

            for (vertex_idx_t<Graph_t> i = 0; i < instance->numberOfVertices(); ++i) {
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
        if (vec.size() == static_cast<std::size_t>(instance->numberOfVertices())) {
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
        if (vec.size() == static_cast<std::size_t>(instance->numberOfVertices())) {
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
        if (vec.size() == static_cast<std::size_t>(instance->numberOfVertices())) {
            node_to_processor_assignment = std::move(vec);
        } else {
            throw std::invalid_argument(
                "Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    /**
     * @brief Computes the work costs of the schedule.
     * The workload of a processor in a superstep is the sum of the workloads of all nodes assigned to that processor in that superstep.
     * The workload in a superstep is the maximum workload of any processor in that superstep.
     * The work cost of the schedule is the sum of the workloads of all supersteps.
     *
     * @return The work costs of the schedule.
     */
    virtual v_workw_t<Graph_t> computeWorkCosts() const override { return cost_helpers::compute_work_costs(*this); }

    /**
     * @brief Computes the costs of the schedule accoring to lazy communication cost evaluation.
     *
     * @return The costs of the schedule.
     */
    virtual v_workw_t<Graph_t> computeCosts() const override { return LazyCommunicationCost<Graph_t>()(*this); }

    /**
     * @brief Checks if the schedule is valid.
     *
     * A schedule is valid if it satisfies all precedence, memory, and node type constraints.
     *
     * @return True if the schedule is valid, false otherwise.
     */
    [[nodiscard]] bool isValid() const { return satisfiesPrecedenceConstraints() && satisfiesMemoryConstraints() && satisfiesNodeTypeConstraints(); }

    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u
     * and v are assigned to different processors, the difference between the superstep assigned to node u and the
     * superstep assigned to node v is less than the staleness of the schedule. For the BspSchedule staleness is 1.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    [[nodiscard]] bool satisfiesPrecedenceConstraints() const {
        if (static_cast<vertex_idx_t<Graph_t>>(node_to_processor_assignment.size()) != instance->numberOfVertices() ||
            static_cast<vertex_idx_t<Graph_t>>(node_to_superstep_assignment.size()) != instance->numberOfVertices()) {
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
                const unsigned different_processors = (node_to_processor_assignment[v] == node_to_processor_assignment[target]) ? 0u : getStaleness();
                if (node_to_superstep_assignment[v] + different_processors > node_to_superstep_assignment[target]) {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * @brief Checks if the schedule satisfies node type constraints.
     *
     * Node type constraints are checked based on the compatibility of nodes with their assigned processors.
     *
     * @return True if node type constraints are satisfied, false otherwise.
     */
    [[nodiscard]] bool satisfiesNodeTypeConstraints() const {
        if (node_to_processor_assignment.size() != instance->numberOfVertices()) {
            return false;
        }

        for (const auto &node : instance->vertices()) {
            if (!instance->isCompatible(node, node_to_processor_assignment[node])) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Checks if the schedule satisfies memory constraints.
     *
     * Memory constraints are checked based on the type of memory constraint specified in the architecture.
     *
     * @return True if memory constraints are satisfied, false otherwise.
     */
    [[nodiscard]] bool satisfiesMemoryConstraints() const {

        switch (instance->getArchitecture().getMemoryConstraintType()) {

        case MEMORY_CONSTRAINT_TYPE::LOCAL:
            return satisfiesLocalMemoryConstraints();

        case MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT:
            return satisfiesPersistentAndTransientMemoryConstraints();

        case MEMORY_CONSTRAINT_TYPE::GLOBAL:
            return satisfiesGlobalMemoryConstraints();

        case MEMORY_CONSTRAINT_TYPE::LOCAL_IN_OUT:
            return satisfiesLocalInOutMemoryConstraints();

        case MEMORY_CONSTRAINT_TYPE::LOCAL_INC_EDGES:
            return satisfiesLocalIncEdgesMemoryConstraints();

        case MEMORY_CONSTRAINT_TYPE::LOCAL_SOURCES_INC_EDGES:
            return satisfiesLocalSourcesIncEdgesMemoryConstraints();

        case MEMORY_CONSTRAINT_TYPE::NONE:
            return true;

        default:
            throw std::invalid_argument("Unknown memory constraint type.");
        }
    }

    /**
     * @brief Returns a vector of nodes assigned to the specified processor.
     *
     * @param processor The processor index.
     * @return A vector of nodes assigned to the specified processor.
     */
    [[nodiscard]] std::vector<vertex_idx_t<Graph_t>> getAssignedNodeVector(unsigned int processor) const {
        std::vector<vertex_idx_t<Graph_t>> vec;

        for (const auto &node : instance->vertices()) {
            if (node_to_processor_assignment[node] == processor) {
                vec.push_back(node);
            }
        }

        return vec;
    }

    /**
     * @brief Returns a vector of nodes assigned to the specified processor and superstep.
     *
     * @param processor The processor index.
     * @param superstep The superstep index.
     * @return A vector of nodes assigned to the specified processor and superstep.
     */
    [[nodiscard]] std::vector<vertex_idx_t<Graph_t>> getAssignedNodeVector(unsigned int processor, unsigned int superstep) const {
        std::vector<vertex_idx_t<Graph_t>> vec;

        for (const auto &node : instance->vertices()) {
            if (node_to_processor_assignment[node] == processor && node_to_superstep_assignment[node] == superstep) {
                vec.push_back(node);
            }
        }

        return vec;
    }

    /**
     * @brief Sets the number of supersteps in the schedule.
     *
     * @param number_of_supersteps_ The number of supersteps.
     */
    void setNumberOfSupersteps(unsigned int number_of_supersteps_) {
        number_of_supersteps = number_of_supersteps_;
    }

    /**
     * @brief Returns the number of nodes assigned to the specified processor.
     *
     * @param processor The processor index.
     * @return The number of nodes assigned to the specified processor.
     */
    [[nodiscard]] unsigned numAssignedNodes(unsigned processor) const {
        unsigned num = 0;

        for (const auto &node : instance->vertices()) {
            if (node_to_processor_assignment[node] == processor) {
                num++;
            }
        }

        return num;
    }

    /**
     * @brief Returns a vector containing the number of nodes assigned to each processor.
     *
     * @return A vector containing the number of nodes assigned to each processor.
     */
    [[nodiscard]] std::vector<unsigned> numAssignedNodesPerProcessor() const {
        std::vector<unsigned> num(instance->numberOfProcessors(), 0);

        for (const auto &node : instance->vertices()) {
            num[node_to_processor_assignment[node]]++;
        }

        return num;
    }

    /**
     * @brief Returns a 2D vector containing the number of nodes assigned to each processor in each superstep.
     *
     * @return A 2D vector containing the number of nodes assigned to each processor in each superstep.
     */
    [[nodiscard]] std::vector<std::vector<unsigned>> numAssignedNodesPerSuperstepProcessor() const {
        std::vector<std::vector<unsigned>> num(number_of_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

        for (const auto &v : instance->vertices()) {
            num[node_to_superstep_assignment[v]][node_to_processor_assignment[v]] += 1;
        }

        return num;
    }

    /**
     * @brief Shrinks the schedule by merging supersteps where no communication occurs.
     */
    virtual void shrinkByMergingSupersteps() {
        std::vector<bool> comm_phase_empty(number_of_supersteps, true);
        for (const auto &node : instance->vertices()) {
            for (const auto &child : instance->getComputationalDag().children(node)) {
                if (node_to_processor_assignment[node] != node_to_processor_assignment[child]) {
                    for (unsigned offset = 1; offset <= getStaleness(); ++offset)
                        comm_phase_empty[node_to_superstep_assignment[child] - offset] = false;
                }
            }
        }

        std::vector<unsigned> new_step_index(number_of_supersteps);
        unsigned current_index = 0;
        for (unsigned step = 0; step < number_of_supersteps; ++step) {
            new_step_index[step] = current_index;
            if (!comm_phase_empty[step])
                current_index++;
        }
        for (const auto &node : instance->vertices()) {
            node_to_superstep_assignment[node] = new_step_index[node_to_superstep_assignment[node]];
        }
        setNumberOfSupersteps(current_index);
    }

  private:
    /**
     * @brief Checks if the schedule satisfies local memory constraints.
     *
     * In this model, the memory usage of a processor in a superstep is the sum of the memory weights of all nodes
     * assigned to it in that superstep.
     *
     * @return True if local memory constraints are satisfied, false otherwise.
     */
    bool satisfiesLocalMemoryConstraints() const {
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
        return true;
    }

    /**
     * @brief Checks if the schedule satisfies persistent and transient memory constraints.
     *
     * This model distinguishes between persistent memory (node memory weight) and transient memory (max communication
     * weight). The total memory usage on a processor is the sum of persistent memory of all assigned nodes plus the
     * maximum transient memory required by any single node assigned to it.
     *
     * @return True if persistent and transient memory constraints are satisfied, false otherwise.
     */
    bool satisfiesPersistentAndTransientMemoryConstraints() const {
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
        return true;
    }

    /**
     * @brief Checks if the schedule satisfies global memory constraints.
     *
     * In this model, the memory usage of a processor is the sum of the memory weights of all nodes assigned to it,
     * regardless of the superstep.
     *
     * @return True if global memory constraints are satisfied, false otherwise.
     */
    bool satisfiesGlobalMemoryConstraints() const {
        std::vector<v_memw_t<Graph_t>> current_proc_memory(instance->numberOfProcessors(), 0);

        for (const auto &node : instance->vertices()) {
            const unsigned proc = node_to_processor_assignment[node];
            current_proc_memory[proc] += instance->getComputationalDag().vertex_mem_weight(node);

            if (current_proc_memory[proc] > instance->getArchitecture().memoryBound(proc)) {
                return false;
            }
        }
        return true;
    }

    bool satisfiesLocalInOutMemoryConstraints() const {
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

        return true;
    }

    bool satisfiesLocalIncEdgesMemoryConstraints() const {
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
        return true;
    }

    bool satisfiesLocalSourcesIncEdgesMemoryConstraints() const {
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
        return true;
    }
};

} // namespace osp