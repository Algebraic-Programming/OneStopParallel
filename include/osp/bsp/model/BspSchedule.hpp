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
#include "osp/bsp/model/cost/LazyCommunicationCost.hpp"
#include "osp/bsp/model/util/SetSchedule.hpp"
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
 * This class is templated on `GraphT`, which must satisfy the `computational_dag_concept`.
 * Moreover, the work and communication weights of the nodes must be of the same type in order to properly compute the cost.
 *
 * It interacts closely with `BspInstance` to access problem-specific data and constraints. In fact, a `BspSchedule` object is
 * tied to a `BspInstance` object.
 *
 * @tparam GraphT The type of the computational DAG, which must satisfy `is_computational_dag_v`.
 * @see BspInstance
 * @see IBspSchedule
 * @see IBspScheduleEval
 */
template <typename GraphT>
class BspSchedule : public IBspSchedule<GraphT>, public IBspScheduleEval<GraphT> {
    static_assert(isComputationalDagV<GraphT>, "BspSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "BspSchedule requires work and comm. weights to have the same type.");

  protected:
    using VertexIdx = VertexIdxT<GraphT>;

    const BspInstance<GraphT> *instance_;

    unsigned numberOfSupersteps_;

    std::vector<unsigned> nodeToProcessorAssignment_;
    std::vector<unsigned> nodeToSuperstepAssignment_;

  public:
    BspSchedule() = delete;

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    explicit BspSchedule(const BspInstance<GraphT> &inst)
        : instance_(&inst),
          numberOfSupersteps_(1),
          nodeToProcessorAssignment_(std::vector<unsigned>(inst.NumberOfVertices(), 0)),
          nodeToSuperstepAssignment_(std::vector<unsigned>(inst.NumberOfVertices(), 0)) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep
     * assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */
    BspSchedule(const BspInstance<GraphT> &inst,
                const std::vector<unsigned> &processorAssignment,
                const std::vector<unsigned> &superstepAssignment)
        : instance_(&inst), nodeToProcessorAssignment_(processorAssignment), nodeToSuperstepAssignment_(superstepAssignment) {
        UpdateNumberOfSupersteps();
    }

    /**
     * @brief Copy constructor from an IBspSchedule.
     *
     * @param schedule The schedule to copy.
     */
    explicit BspSchedule(const IBspSchedule<GraphT> &schedule)
        : instance_(&schedule.GetInstance()),
          numberOfSupersteps_(schedule.NumberOfSupersteps()),
          nodeToProcessorAssignment_(schedule.GetInstance().NumberOfVertices()),
          nodeToSuperstepAssignment_(schedule.GetInstance().NumberOfVertices()) {
        for (const auto &v : schedule.GetInstance().GetComputationalDag().Vertices()) {
            nodeToProcessorAssignment_[v] = schedule.AssignedProcessor(v);
            nodeToSuperstepAssignment_[v] = schedule.AssignedSuperstep(v);
        }
    }

    /**
     * @brief Copy constructor.
     *
     * @param schedule The schedule to copy.
     */
    BspSchedule(const BspSchedule<GraphT> &schedule)
        : instance_(schedule.instance_),
          numberOfSupersteps_(schedule.numberOfSupersteps_),
          nodeToProcessorAssignment_(schedule.nodeToProcessorAssignment_),
          nodeToSuperstepAssignment_(schedule.nodeToSuperstepAssignment_) {}

    /**
     * @brief Copy assignment operator.
     *
     * @param schedule The schedule to copy.
     * @return A reference to this schedule.
     */
    BspSchedule<GraphT> &operator=(const BspSchedule<GraphT> &schedule) {
        if (this != &schedule) {
            instance_ = schedule.instance_;
            numberOfSupersteps_ = schedule.numberOfSupersteps_;
            nodeToProcessorAssignment_ = schedule.nodeToProcessorAssignment_;
            nodeToSuperstepAssignment_ = schedule.nodeToSuperstepAssignment_;
        }
        return *this;
    }

    /**
     * @brief Move constructor.
     *
     * @param schedule The schedule to move.
     */
    BspSchedule(BspSchedule<GraphT> &&schedule) noexcept
        : instance_(schedule.instance_),
          numberOfSupersteps_(schedule.numberOfSupersteps_),
          nodeToProcessorAssignment_(std::move(schedule.nodeToProcessorAssignment_)),
          nodeToSuperstepAssignment_(std::move(schedule.nodeToSuperstepAssignment_)) {}

    /**
     * @brief Move assignment operator.
     *
     * @param schedule The schedule to move.
     * @return A reference to this schedule.
     */
    BspSchedule<GraphT> &operator=(BspSchedule<GraphT> &&schedule) noexcept {
        if (this != &schedule) {
            instance_ = schedule.instance_;
            numberOfSupersteps_ = schedule.numberOfSupersteps_;
            nodeToProcessorAssignment_ = std::move(schedule.nodeToProcessorAssignment_);
            nodeToSuperstepAssignment_ = std::move(schedule.nodeToSuperstepAssignment_);
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
    template <typename GraphTOther>
    BspSchedule(const BspInstance<GraphT> &instance, const BspSchedule<GraphTOther> &schedule)
        : instance_(&instance),
          numberOfSupersteps_(schedule.NumberOfSupersteps()),
          nodeToProcessorAssignment_(schedule.AssignedProcessors()),
          nodeToSuperstepAssignment_(schedule.AssignedSupersteps()) {}

    /**
     * @brief Destructor for the BspSchedule class.
     */
    virtual ~BspSchedule() = default;

    /**
     * @brief Returns a reference to the BspInstance for the schedule.
     *
     * @return A reference to the BspInstance for the schedule.
     */
    [[nodiscard]] const BspInstance<GraphT> &GetInstance() const override { return *instance_; }

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    [[nodiscard]] unsigned NumberOfSupersteps() const override { return numberOfSupersteps_; }

    /**
     * @brief Updates the number of supersteps based on the current assignment.
     */
    void UpdateNumberOfSupersteps() {
        numberOfSupersteps_ = 0;
        for (VertexIdxT<GraphT> i = 0; i < static_cast<VertexIdxT<GraphT>>(instance_->NumberOfVertices()); ++i) {
            if (nodeToSuperstepAssignment_[i] >= numberOfSupersteps_) {
                numberOfSupersteps_ = nodeToSuperstepAssignment_[i] + 1;
            }
        }
    }

    /**
     * @brief Returns the superstep assigned to the specified node.
     *
     * @param node The node for which to return the assigned superstep.
     * @return The superstep assigned to the specified node.
     */
    [[nodiscard]] unsigned AssignedSuperstep(const VertexIdx node) const override { return nodeToSuperstepAssignment_[node]; }

    /**
     * @brief Returns the processor assigned to the specified node.
     *
     * @param node The node for which to return the assigned processor.
     * @return The processor assigned to the specified node.
     */
    [[nodiscard]] unsigned AssignedProcessor(const VertexIdx node) const override { return nodeToProcessorAssignment_[node]; }

    /**
     * @brief Returns the superstep assignment for the schedule.
     *
     * @return The superstep assignment for the schedule.
     */
    [[nodiscard]] const std::vector<unsigned> &AssignedSupersteps() const { return nodeToSuperstepAssignment_; }

    [[nodiscard]] std::vector<unsigned> &AssignedSupersteps() { return nodeToSuperstepAssignment_; }

    /**
     * @brief Returns the processor assignment for the schedule.
     *
     * @return The processor assignment for the schedule.
     */
    [[nodiscard]] const std::vector<unsigned> &AssignedProcessors() const { return nodeToProcessorAssignment_; }

    [[nodiscard]] std::vector<unsigned> &AssignedProcessors() { return nodeToProcessorAssignment_; }

    /**
     * @brief Returns the staleness of the schedule.
     * The staleness determines the minimum number of supersteps that must elapse between the assignment of a node to a processor
     * and the assignment of one of its neighbors to a different processor. The staleness for the BspSchedule is always 1.
     *
     * @return The staleness of the schedule.
     */
    [[nodiscard]] virtual unsigned GetStaleness() const { return 1; }

    /**
     * @brief Sets the superstep assigned to the specified node.
     *
     * @param node The node for which to set the assigned superstep.
     * @param superstep The superstep to assign to the node.
     */
    void SetAssignedSuperstep(const VertexIdx node, const unsigned superstep) {
        if (node < instance_->NumberOfVertices()) {
            nodeToSuperstepAssignment_[node] = superstep;

            if (superstep >= numberOfSupersteps_) {
                numberOfSupersteps_ = superstep + 1;
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
    void SetAssignedSuperstepNoUpdateNumSuperstep(const VertexIdx node, const unsigned superstep) {
        nodeToSuperstepAssignment_.at(node) = superstep;
    }

    /**
     * @brief Sets the processor assigned to the specified node.
     *
     * @param node The node for which to set the assigned processor.
     * @param processor The processor to assign to the node.
     */
    void SetAssignedProcessor(const VertexIdx node, const unsigned processor) { nodeToProcessorAssignment_.at(node) = processor; }

    /**
     * @brief Sets the superstep assignment for the schedule.
     *
     * @param vec The superstep assignment to set.
     */
    void SetAssignedSupersteps(const std::vector<unsigned> &vec) {
        if (vec.size() == static_cast<std::size_t>(instance_->NumberOfVertices())) {
            numberOfSupersteps_ = 0;

            for (VertexIdxT<GraphT> i = 0; i < instance_->NumberOfVertices(); ++i) {
                if (vec[i] >= numberOfSupersteps_) {
                    numberOfSupersteps_ = vec[i] + 1;
                }

                nodeToSuperstepAssignment_[i] = vec[i];
            }
        } else {
            throw std::invalid_argument("Invalid Argument while assigning supersteps: size does not match number of nodes.");
        }
    }

    /**
     * @brief Sets the superstep assignment for the schedule.
     *
     * @param vec The superstep assignment to set.
     */
    void SetAssignedSupersteps(std::vector<unsigned> &&vec) {
        if (vec.size() == static_cast<std::size_t>(instance_->NumberOfVertices())) {
            nodeToSuperstepAssignment_ = std::move(vec);
        } else {
            throw std::invalid_argument("Invalid Argument while assigning supersteps: size does not match number of nodes.");
        }

        UpdateNumberOfSupersteps();
    }

    /**
     * @brief Sets the processor assignment for the schedule.
     *
     * @param vec The processor assignment to set.
     */
    void SetAssignedProcessors(const std::vector<unsigned> &vec) {
        if (vec.size() == static_cast<std::size_t>(instance_->NumberOfVertices())) {
            nodeToProcessorAssignment_ = vec;
        } else {
            throw std::invalid_argument("Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    /**
     * @brief Sets the processor assignment for the schedule.
     *
     * @param vec The processor assignment to set.
     */
    void SetAssignedProcessors(std::vector<unsigned> &&vec) {
        if (vec.size() == static_cast<std::size_t>(instance_->NumberOfVertices())) {
            nodeToProcessorAssignment_ = std::move(vec);
        } else {
            throw std::invalid_argument("Invalid Argument while assigning processors: size does not match number of nodes.");
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
    virtual VWorkwT<GraphT> ComputeWorkCosts() const override { return cost_helpers::ComputeWorkCosts(*this); }

    /**
     * @brief Computes the costs of the schedule accoring to lazy communication cost evaluation.
     *
     * @return The costs of the schedule.
     */
    virtual VWorkwT<GraphT> ComputeCosts() const override { return LazyCommunicationCost<GraphT>()(*this); }

    /**
     * @brief Checks if the schedule is valid.
     *
     * A schedule is valid if it satisfies all precedence, memory, and node type constraints.
     *
     * @return True if the schedule is valid, false otherwise.
     */
    [[nodiscard]] bool IsValid() const {
        return SatisfiesPrecedenceConstraints() && SatisfiesMemoryConstraints() && SatisfiesNodeTypeConstraints();
    }

    /**
     * @brief Returns true if the schedule satisfies the precedence constraints of the computational DAG.
     *
     * The precedence constraints of the computational DAG are satisfied if, for each directed edge (u, v) such that u
     * and v are assigned to different processors, the difference between the superstep assigned to node u and the
     * superstep assigned to node v is less than the staleness of the schedule. For the BspSchedule staleness is 1.
     *
     * @return True if the schedule satisfies the precedence constraints of the computational DAG, false otherwise.
     */
    [[nodiscard]] bool SatisfiesPrecedenceConstraints() const {
        if (static_cast<VertexIdxT<GraphT>>(nodeToProcessorAssignment_.size()) != instance_->NumberOfVertices()
            || static_cast<VertexIdxT<GraphT>>(nodeToSuperstepAssignment_.size()) != instance_->NumberOfVertices()) {
            return false;
        }

        for (const auto &v : instance_->Vertices()) {
            if (nodeToSuperstepAssignment_[v] >= numberOfSupersteps_) {
                return false;
            }
            if (nodeToProcessorAssignment_[v] >= instance_->NumberOfProcessors()) {
                return false;
            }

            for (const auto &target : instance_->GetComputationalDag().Children(v)) {
                const unsigned differentProcessors
                    = (nodeToProcessorAssignment_[v] == nodeToProcessorAssignment_[target]) ? 0u : GetStaleness();
                if (nodeToSuperstepAssignment_[v] + differentProcessors > nodeToSuperstepAssignment_[target]) {
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
    [[nodiscard]] bool SatisfiesNodeTypeConstraints() const {
        if (nodeToProcessorAssignment_.size() != instance_->NumberOfVertices()) {
            return false;
        }

        for (const auto &node : instance_->Vertices()) {
            if (!instance_->IsCompatible(node, nodeToProcessorAssignment_[node])) {
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
    [[nodiscard]] bool SatisfiesMemoryConstraints() const {
        switch (instance_->GetArchitecture().GetMemoryConstraintType()) {
            case MemoryConstraintType::LOCAL:
                return SatisfiesLocalMemoryConstraints();

            case MemoryConstraintType::PERSISTENT_AND_TRANSIENT:
                return SatisfiesPersistentAndTransientMemoryConstraints();

            case MemoryConstraintType::GLOBAL:
                return SatisfiesGlobalMemoryConstraints();

            case MemoryConstraintType::LOCAL_IN_OUT:
                return SatisfiesLocalInOutMemoryConstraints();

            case MemoryConstraintType::LOCAL_INC_EDGES:
                return SatisfiesLocalIncEdgesMemoryConstraints();

            case MemoryConstraintType::LOCAL_SOURCES_INC_EDGES:
                return SatisfiesLocalSourcesIncEdgesMemoryConstraints();

            case MemoryConstraintType::NONE:
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
    [[nodiscard]] std::vector<VertexIdxT<GraphT>> GetAssignedNodeVector(const unsigned processor) const {
        std::vector<VertexIdxT<GraphT>> vec;

        for (const auto &node : instance_->Vertices()) {
            if (nodeToProcessorAssignment_[node] == processor) {
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
    [[nodiscard]] std::vector<VertexIdxT<GraphT>> GetAssignedNodeVector(const unsigned processor, const unsigned superstep) const {
        std::vector<VertexIdxT<GraphT>> vec;

        for (const auto &node : instance_->Vertices()) {
            if (nodeToProcessorAssignment_[node] == processor && nodeToSuperstepAssignment_[node] == superstep) {
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
    void SetNumberOfSupersteps(const unsigned numberOfSupersteps) { numberOfSupersteps_ = numberOfSupersteps; }

    /**
     * @brief Returns the number of nodes assigned to the specified processor.
     *
     * @param processor The processor index.
     * @return The number of nodes assigned to the specified processor.
     */
    [[nodiscard]] unsigned NumAssignedNodes(const unsigned processor) const {
        unsigned num = 0;

        for (const auto &node : instance_->Vertices()) {
            if (nodeToProcessorAssignment_[node] == processor) {
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
    [[nodiscard]] std::vector<unsigned> NumAssignedNodesPerProcessor() const {
        std::vector<unsigned> num(instance_->NumberOfProcessors(), 0);

        for (const auto &node : instance_->Vertices()) {
            num[nodeToProcessorAssignment_[node]]++;
        }

        return num;
    }

    /**
     * @brief Returns a 2D vector containing the number of nodes assigned to each processor in each superstep.
     *
     * @return A 2D vector containing the number of nodes assigned to each processor in each superstep.
     */
    [[nodiscard]] std::vector<std::vector<unsigned>> NumAssignedNodesPerSuperstepProcessor() const {
        std::vector<std::vector<unsigned>> num(numberOfSupersteps_, std::vector<unsigned>(instance_->NumberOfProcessors(), 0));

        for (const auto &v : instance_->Vertices()) {
            num[nodeToSuperstepAssignment_[v]][nodeToProcessorAssignment_[v]] += 1;
        }

        return num;
    }

    /**
     * @brief Shrinks the schedule by merging supersteps where no communication occurs.
     */
    virtual void ShrinkByMergingSupersteps() {
        std::vector<bool> commPhaseEmpty(numberOfSupersteps_, true);
        for (const auto &node : instance_->Vertices()) {
            for (const auto &child : instance_->GetComputationalDag().Children(node)) {
                if (nodeToProcessorAssignment_[node] != nodeToProcessorAssignment_[child]) {
                    for (unsigned offset = 1; offset <= GetStaleness(); ++offset) {
                        commPhaseEmpty[nodeToSuperstepAssignment_[child] - offset] = false;
                    }
                }
            }
        }

        std::vector<unsigned> newStepIndex(numberOfSupersteps_);
        unsigned currentIndex = 0;
        for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
            newStepIndex[step] = currentIndex;
            if (!commPhaseEmpty[step]) {
                currentIndex++;
            }
        }
        for (const auto &node : instance_->Vertices()) {
            nodeToSuperstepAssignment_[node] = newStepIndex[nodeToSuperstepAssignment_[node]];
        }
        SetNumberOfSupersteps(currentIndex);
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
    bool SatisfiesLocalMemoryConstraints() const {
        SetSchedule setSchedule = SetSchedule(*this);

        for (unsigned step = 0; step < numberOfSupersteps_; step++) {
            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                VMemwT<GraphT> memory = 0;
                for (const auto &node : setSchedule.stepProcessorVertices_[step][proc]) {
                    memory += instance_->GetComputationalDag().VertexMemWeight(node);
                }

                if (memory > instance_->GetArchitecture().MemoryBound(proc)) {
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
    bool SatisfiesPersistentAndTransientMemoryConstraints() const {
        std::vector<VMemwT<GraphT>> currentProcPersistentMemory(instance_->NumberOfProcessors(), 0);
        std::vector<VMemwT<GraphT>> currentProcTransientMemory(instance_->NumberOfProcessors(), 0);

        for (const auto &node : instance_->Vertices()) {
            const unsigned proc = nodeToProcessorAssignment_[node];
            currentProcPersistentMemory[proc] += instance_->GetComputationalDag().VertexMemWeight(node);
            currentProcTransientMemory[proc]
                = std::max(currentProcTransientMemory[proc], instance_->GetComputationalDag().VertexCommWeight(node));

            if (currentProcPersistentMemory[proc] + currentProcTransientMemory[proc]
                > instance_->GetArchitecture().MemoryBound(proc)) {
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
    bool SatisfiesGlobalMemoryConstraints() const {
        std::vector<VMemwT<GraphT>> currentProcMemory(instance_->NumberOfProcessors(), 0);

        for (const auto &node : instance_->Vertices()) {
            const unsigned proc = nodeToProcessorAssignment_[node];
            currentProcMemory[proc] += instance_->GetComputationalDag().VertexMemWeight(node);

            if (currentProcMemory[proc] > instance_->GetArchitecture().MemoryBound(proc)) {
                return false;
            }
        }
        return true;
    }

    bool SatisfiesLocalInOutMemoryConstraints() const {
        SetSchedule setSchedule = SetSchedule(*this);

        for (unsigned step = 0; step < numberOfSupersteps_; step++) {
            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                VMemwT<GraphT> memory = 0;
                for (const auto &node : setSchedule.stepProcessorVertices_[step][proc]) {
                    memory += instance_->GetComputationalDag().VertexMemWeight(node)
                              + instance_->GetComputationalDag().VertexCommWeight(node);

                    for (const auto &parent : instance_->GetComputationalDag().Parents(node)) {
                        if (nodeToProcessorAssignment_[parent] == proc && nodeToSuperstepAssignment_[parent] == step) {
                            memory -= instance_->GetComputationalDag().VertexCommWeight(parent);
                        }
                    }
                }

                if (memory > instance_->GetArchitecture().MemoryBound(proc)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool SatisfiesLocalIncEdgesMemoryConstraints() const {
        SetSchedule setSchedule = SetSchedule(*this);

        for (unsigned step = 0; step < numberOfSupersteps_; step++) {
            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                std::unordered_set<VertexIdxT<GraphT>> nodesWithIncomingEdges;

                VMemwT<GraphT> memory = 0;
                for (const auto &node : setSchedule.stepProcessorVertices_[step][proc]) {
                    memory += instance_->GetComputationalDag().VertexCommWeight(node);

                    for (const auto &parent : instance_->GetComputationalDag().Parents(node)) {
                        if (nodeToSuperstepAssignment_[parent] != step) {
                            nodesWithIncomingEdges.insert(parent);
                        }
                    }
                }

                for (const auto &node : nodesWithIncomingEdges) {
                    memory += instance_->GetComputationalDag().VertexCommWeight(node);
                }

                if (memory > instance_->GetArchitecture().MemoryBound(proc)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool SatisfiesLocalSourcesIncEdgesMemoryConstraints() const {
        SetSchedule setSchedule = SetSchedule(*this);

        for (unsigned step = 0; step < numberOfSupersteps_; step++) {
            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                std::unordered_set<VertexIdxT<GraphT>> nodesWithIncomingEdges;

                VMemwT<GraphT> memory = 0;
                for (const auto &node : setSchedule.stepProcessorVertices_[step][proc]) {
                    if (IsSource(node, instance_->GetComputationalDag())) {
                        memory += instance_->GetComputationalDag().VertexMemWeight(node);
                    }

                    for (const auto &parent : instance_->GetComputationalDag().Parents(node)) {
                        if (nodeToSuperstepAssignment_[parent] != step) {
                            nodesWithIncomingEdges.insert(parent);
                        }
                    }
                }

                for (const auto &node : nodesWithIncomingEdges) {
                    memory += instance_->GetComputationalDag().VertexCommWeight(node);
                }

                if (memory > instance_->GetArchitecture().MemoryBound(proc)) {
                    return false;
                }
            }
        }
        return true;
    }
};

}    // namespace osp
