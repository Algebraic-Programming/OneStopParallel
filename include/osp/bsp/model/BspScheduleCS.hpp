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

#include "BspSchedule.hpp"
#include "IBspScheduleEval.hpp"

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
template <typename GraphT>
class BspScheduleCS : public BspSchedule<GraphT> {
    static_assert(is_computational_dag_v<GraphT>, "BspScheduleCS can only be used with computational DAGs.");

  public:
    using KeyTriple = std::tuple<VertexIdxT<GraphT>, unsigned int, unsigned int>;

  private:
    using VertexIdx = VertexIdxT<GraphT>;

    // contains entries: (vertex, from_proc, to_proc ) : step
    std::map<KeyTriple, unsigned> commSchedule_;

  protected:
    void ComputeCsCommunicationCostsHelper(std::vector<std::vector<VCommwT<GraphT>>> &rec,
                                           std::vector<std::vector<VCommwT<GraphT>>> &send) const {
        for (auto const &[key, val] : commSchedule_) {
            send[std::get<1>(key)][val]
                += BspSchedule<GraphT>::instance->sendCosts(std::get<1>(key), std::get<2>(key))
                   * BspSchedule<GraphT>::instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
            rec[std::get<2>(key)][val]
                += BspSchedule<GraphT>::instance->sendCosts(std::get<1>(key), std::get<2>(key))
                   * BspSchedule<GraphT>::instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
        }
    }

  public:
    BspScheduleCS() = delete;

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    BspScheduleCS(const BspInstance<GraphT> &inst) : BspSchedule<GraphT>(inst) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep
     * assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */
    BspScheduleCS(const BspInstance<GraphT> &inst,
                  const std::vector<unsigned> &processorAssignment,
                  const std::vector<unsigned> &superstepAssignment)
        : BspSchedule<GraphT>(inst, processorAssignment, superstepAssignment) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, superstep
     * assignment, and communication schedule.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     * @param comm_ The communication schedule for the nodes.
     */
    BspScheduleCS(const BspInstance<GraphT> &inst,
                  const std::vector<unsigned int> &processorAssignment,
                  const std::vector<unsigned int> &superstepAssignment,
                  const std::map<KeyTriple, unsigned int> &comm)
        : BspSchedule<GraphT>(inst, processorAssignment, superstepAssignment), commSchedule_(comm) {}

    explicit BspScheduleCS(BspSchedule<GraphT> &&schedule) : BspSchedule<GraphT>(std::move(schedule)) {
        SetAutoCommunicationSchedule();
    }

    BspScheduleCS(BspSchedule<GraphT> &&schedule, const std::map<KeyTriple, unsigned int> &comm)
        : BspSchedule<GraphT>(std::move(schedule)), commSchedule_(comm) {}

    BspScheduleCS(BspSchedule<GraphT> &&schedule, std::map<KeyTriple, unsigned int> &&comm)
        : BspSchedule<GraphT>(std::move(schedule)), commSchedule_(std::move(comm)) {
        comm.clear();
    }

    explicit BspScheduleCS(const BspSchedule<GraphT> &schedule) : BspSchedule<GraphT>(schedule) {
        SetAutoCommunicationSchedule();
    }

    BspScheduleCS(const BspSchedule<GraphT> &schedule, const std::map<KeyTriple, unsigned int> &comm)
        : BspSchedule<GraphT>(schedule), commSchedule_(comm) {}

    BspScheduleCS(const BspScheduleCS &other) = default;
    BspScheduleCS(BspScheduleCS &&other) = default;
    BspScheduleCS &operator=(const BspScheduleCS &other) = default;
    BspScheduleCS &operator=(BspScheduleCS &&other) = default;
    virtual ~BspScheduleCS() = default;

    inline const std::map<KeyTriple, unsigned> &GetCommunicationSchedule() const { return commSchedule_; }

    inline std::map<KeyTriple, unsigned> &GetCommunicationSchedule() { return commSchedule_; }

    inline bool HasValidCommSchedule() const { return CheckCommScheduleValidity(commSchedule_); }

    void AddCommunicationScheduleEntry(KeyTriple key, unsigned step) {
        if (step >= BspSchedule<GraphT>::number_of_supersteps) {
            throw std::invalid_argument("Invalid Argument while adding communication schedule entry: step out of range.");
        }

        if (std::get<0>(key) >= BspSchedule<GraphT>::instance->numberOfVertices()) {
            throw std::invalid_argument("Invalid Argument while adding communication schedule entry: node out of range.");
        }

        if (std::get<1>(key) >= BspSchedule<GraphT>::instance->numberOfProcessors()) {
            throw std::invalid_argument(
                "Invalid Argument while adding communication schedule entry: from processor out of range.");
        }

        if (std::get<2>(key) >= BspSchedule<GraphT>::instance->numberOfProcessors()) {
            throw std::invalid_argument("Invalid Argument while adding communication schedule entry: to processor out of range.");
        }

        commSchedule_[key] = step;
    }

    /**
     * @brief Adds an entry to the communication schedule.
     *
     * @param node The node resp. its data which is sent.
     * @param from_proc The processor from which the data is sent.
     * @param to_proc The processor to which the data is sent.
     * @param step The superstep in which the data is sent.
     */
    void AddCommunicationScheduleEntry(VertexIdx node, unsigned fromProc, unsigned toProc, unsigned step) {
        addCommunicationScheduleEntry(std::make_tuple(node, fromProc, toProc), step);
    }

    /**
     * @brief Sets the communication schedule for the schedule.
     *
     * @param cs The communication schedule to set.
     */
    void SetCommunicationSchedule(const std::map<KeyTriple, unsigned int> &cs) {
        if (CheckCommScheduleValidity(cs)) {
            commSchedule_ = cs;
        } else {
            throw std::invalid_argument("Given communication schedule is not valid for instance");
        }
    }

    bool CheckCommScheduleValidity(const std::map<KeyTriple, unsigned int> &cs) const {
        std::vector<std::vector<unsigned>> firstAt = std::vector<std::vector<unsigned>>(
            BspSchedule<GraphT>::instance->numberOfVertices(),
            std::vector<unsigned>(BspSchedule<GraphT>::instance->numberOfProcessors(), BspSchedule<GraphT>::number_of_supersteps));

        for (const auto &node : BspSchedule<GraphT>::instance->vertices()) {
            firstAt[node][BspSchedule<GraphT>::node_to_processor_assignment[node]]
                = BspSchedule<GraphT>::node_to_superstep_assignment[node];
        }

        for (auto const &[key, val] : cs) {
            if (val >= BspSchedule<GraphT>::number_of_supersteps) {
                return false;
            }

            if (std::get<0>(key) >= BspSchedule<GraphT>::instance->numberOfVertices()) {
                return false;
            }

            if (std::get<1>(key) >= BspSchedule<GraphT>::instance->numberOfProcessors()) {
                return false;
            }

            if (std::get<2>(key) >= BspSchedule<GraphT>::instance->numberOfProcessors()) {
                return false;
            }

            firstAt[std::get<0>(key)][std::get<2>(key)]
                = std::min(firstAt[std::get<0>(key)][std::get<2>(key)], val + this->GetStaleness());
        }

        for (auto const &[key, val] : cs) {
            if (val < firstAt[std::get<0>(key)][std::get<1>(key)]) {
                return false;
            }
        }

        for (const auto &v : BspSchedule<GraphT>::instance->getComputationalDag().vertices()) {
            for (const auto &target : BspSchedule<GraphT>::instance->getComputationalDag().children(v)) {
                if (BspSchedule<GraphT>::node_to_processor_assignment[v]
                    != BspSchedule<GraphT>::node_to_processor_assignment[target]) {
                    if (firstAt[v][BspSchedule<GraphT>::node_to_processor_assignment[target]]
                        > BspSchedule<GraphT>::node_to_superstep_assignment[target]) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    VCommwT<GraphT> ComputeCsCommunicationCosts() const {
        std::vector<std::vector<VCommwT<GraphT>>> rec(this->instance_->numberOfProcessors(),
                                                      std::vector<VCommwT<GraphT>>(this->numberOfSupersteps_, 0));
        std::vector<std::vector<VCommwT<GraphT>>> send(this->instance_->numberOfProcessors(),
                                                       std::vector<VCommwT<GraphT>>(this->numberOfSupersteps_, 0));

        ComputeCsCommunicationCostsHelper(rec, send);
        const std::vector<VCommwT<GraphT>> maxCommPerStep = cost_helpers::compute_max_comm_per_step(*this, rec, send);

        VCommwT<GraphT> costs = 0;
        for (unsigned step = 0; step < this->numberOfSupersteps_; step++) {
            const auto stepCommCost = maxCommPerStep[step];
            costs += stepCommCost;

            if (stepCommCost > 0) {
                costs += this->instance_->synchronisationCosts();
            }
        }
        return costs;
    }

    virtual VWorkwT<GraphT> computeCosts() const override { return ComputeCsCommunicationCosts() + this->computeWorkCosts(); }

    void SetAutoCommunicationSchedule() {
        std::map<KeyTriple, unsigned> bestCommSchedule;
        VWorkwT<GraphT> bestCommCost = std::numeric_limits<VWorkwT<GraphT>>::max();    // computeCosts retunrs v_workw_t<Graph_t>

        if (HasValidCommSchedule()) {
            VWorkwT<GraphT> costsCom = BspSchedule<GraphT>::computeCosts();
            if (costsCom < bestCommCost) {
                bestCommSchedule = commSchedule_;
                bestCommCost = costsCom;
            }
        }

        SetImprovedLazyCommunicationSchedule();
        VWorkwT<GraphT> costsCom = BspSchedule<GraphT>::computeCosts();
        // std::cout << "Improved Lazy: " << costs_com << std::endl;
        if (costsCom < bestCommCost) {
            bestCommSchedule = commSchedule_;
            bestCommCost = costsCom;
        }

        SetLazyCommunicationSchedule();
        costsCom = BspSchedule<GraphT>::computeCosts();
        // std::cout << "Lazy: " << costs_com << std::endl;
        if (costsCom < bestCommCost) {
            bestCommSchedule = commSchedule_;
            bestCommCost = costsCom;
        }

        SetEagerCommunicationSchedule();
        costsCom = BspSchedule<GraphT>::computeCosts();
        // std::cout << "Eager: " << costs_com << std::endl;
        if (costsCom < bestCommCost) {
            bestCommSchedule = commSchedule_;
            bestCommCost = costsCom;
        }

        commSchedule_ = bestCommSchedule;
    }

    void SetImprovedLazyCommunicationSchedule() {
        commSchedule_.clear();
        if (BspSchedule<GraphT>::instance->getComputationalDag().num_vertices() <= 1
            || BspSchedule<GraphT>::number_of_supersteps <= 1) {
            return;
        }

        std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> stepProcNodeList(
            BspSchedule<GraphT>::number_of_supersteps,
            std::vector<std::vector<VertexIdxT<GraphT>>>(BspSchedule<GraphT>::instance->numberOfProcessors(),
                                                         std::vector<VertexIdxT<GraphT>>()));
        std::vector<std::vector<bool>> nodeToProcBeenSent(
            BspSchedule<GraphT>::instance->numberOfVertices(),
            std::vector<bool>(BspSchedule<GraphT>::instance->numberOfProcessors(), false));

        for (VertexIdxT<GraphT> node = 0; node < BspSchedule<GraphT>::instance->numberOfVertices(); node++) {
            stepProcNodeList[BspSchedule<GraphT>::node_to_superstep_assignment[node]]
                            [BspSchedule<GraphT>::node_to_processor_assignment[node]]
                                .push_back(node);
            nodeToProcBeenSent[node][BspSchedule<GraphT>::node_to_processor_assignment[node]] = true;
        }

        // The data structure stores for each processor a set of tuples representing required sends.
        // Each tuple is (communication_cost, source_node, destination_processor).
        std::vector<std::set<std::tuple<VCommwT<GraphT>, VertexIdxT<GraphT>, unsigned>, std::greater<>>> requireSending(
            BspSchedule<GraphT>::instance->numberOfProcessors());

        for (unsigned proc = 0; proc < BspSchedule<GraphT>::instance->numberOfProcessors(); proc++) {
            for (const auto &node : stepProcNodeList[0][proc]) {
                for (const auto &target : BspSchedule<GraphT>::instance->getComputationalDag().children(node)) {
                    if (proc != BspSchedule<GraphT>::assignedProcessor(target)) {
                        requireSending[proc].insert({BspSchedule<GraphT>::instance->getComputationalDag().vertex_comm_weight(node)
                                                         * BspSchedule<GraphT>::instance->getArchitecture().sendCosts(
                                                             proc, BspSchedule<GraphT>::node_to_processor_assignment[target]),
                                                     node,
                                                     BspSchedule<GraphT>::node_to_processor_assignment[target]});
                    }
                }
            }
        }

        for (unsigned step = 1; step < BspSchedule<GraphT>::number_of_supersteps; step++) {
            std::vector<VCommwT<GraphT>> sendCost(BspSchedule<GraphT>::instance->numberOfProcessors(), 0);
            std::vector<VCommwT<GraphT>> receiveCost(BspSchedule<GraphT>::instance->numberOfProcessors(), 0);

            // must send in superstep step-1
            for (unsigned proc = 0; proc < BspSchedule<GraphT>::instance->numberOfProcessors(); proc++) {
                for (const auto &node : stepProcNodeList[step][proc]) {
                    for (const auto &source : BspSchedule<GraphT>::instance->getComputationalDag().parents(node)) {
                        if (!nodeToProcBeenSent[source][proc]) {
                            assert(BspSchedule<GraphT>::node_to_superstep_assignment[source] < step + 1 - this->GetStaleness());
                            commSchedule_.emplace(
                                std::make_tuple(source, BspSchedule<GraphT>::node_to_processor_assignment[source], proc),
                                step - this->GetStaleness());
                            nodeToProcBeenSent[source][proc] = true;
                            VCommwT<GraphT> commCost
                                = BspSchedule<GraphT>::instance->getComputationalDag().vertex_comm_weight(source)
                                  * BspSchedule<GraphT>::instance->getArchitecture().sendCosts(
                                      BspSchedule<GraphT>::node_to_processor_assignment[source], proc);
                            requireSending[BspSchedule<GraphT>::node_to_processor_assignment[source]].erase(
                                {commCost, source, proc});
                            sendCost[BspSchedule<GraphT>::node_to_processor_assignment[source]] += commCost;
                            receiveCost[proc] += commCost;
                        }
                    }
                }
            }

            // getting max costs
            VCommwT<GraphT> maxCommCost = 0;
            for (size_t proc = 0; proc < BspSchedule<GraphT>::instance->numberOfProcessors(); proc++) {
                maxCommCost = std::max(maxCommCost, sendCost[proc]);
                maxCommCost = std::max(maxCommCost, receiveCost[proc]);
            }

            // extra sends
            // TODO: permute the order of processors
            for (size_t proc = 0; proc < BspSchedule<GraphT>::instance->numberOfProcessors(); proc++) {
                if (requireSending[proc].empty() || std::get<0>(*requireSending[proc].rbegin()) + sendCost[proc] > maxCommCost) {
                    continue;
                }
                auto iter = requireSending[proc].begin();
                while (iter != requireSending[proc].end()) {
                    const auto &[comm_cost, node_to_send, dest_proc] = *iter;
                    if (comm_cost + sendCost[proc] > maxCommCost || comm_cost + receiveCost[dest_proc] > maxCommCost) {
                        iter++;
                    } else {
                        commSchedule_.emplace(std::make_tuple(node_to_send, proc, dest_proc), step - this->GetStaleness());
                        nodeToProcBeenSent[node_to_send][dest_proc] = true;
                        sendCost[proc] += comm_cost;
                        receiveCost[dest_proc] += comm_cost;
                        iter = requireSending[proc].erase(iter);
                        if (requireSending[proc].empty()
                            || std::get<0>(*requireSending[proc].rbegin()) + sendCost[proc] > maxCommCost) {
                            break;    // Exit if no more sends can possibly fit.
                        }
                    }
                }
            }

            // updating require_sending
            for (unsigned proc = 0; proc < BspSchedule<GraphT>::instance->numberOfProcessors(); proc++) {
                for (const auto &node : stepProcNodeList[step][proc]) {
                    for (const auto &target : BspSchedule<GraphT>::instance->getComputationalDag().children(node)) {
                        if (proc != BspSchedule<GraphT>::assignedProcessor(target)) {
                            requireSending[proc].insert(
                                {BspSchedule<GraphT>::instance->getComputationalDag().vertex_comm_weight(node)
                                     * BspSchedule<GraphT>::instance->getArchitecture().sendCosts(
                                         proc, BspSchedule<GraphT>::node_to_processor_assignment[target]),
                                 node,
                                 BspSchedule<GraphT>::node_to_processor_assignment[target]});
                        }
                    }
                }
            }
        }
    }

    void SetLazyCommunicationSchedule() {
        commSchedule_.clear();

        for (const auto &source : BspSchedule<GraphT>::instance->getComputationalDag().vertices()) {
            for (const auto &target : BspSchedule<GraphT>::instance->getComputationalDag().children(source)) {
                if (BspSchedule<GraphT>::node_to_processor_assignment[source]
                    != BspSchedule<GraphT>::node_to_processor_assignment[target]) {
                    const auto tmp = std::make_tuple(source,
                                                     BspSchedule<GraphT>::node_to_processor_assignment[source],
                                                     BspSchedule<GraphT>::node_to_processor_assignment[target]);
                    if (commSchedule_.find(tmp) == commSchedule_.end()) {
                        commSchedule_[tmp] = BspSchedule<GraphT>::node_to_superstep_assignment[target] - this->GetStaleness();

                    } else {
                        commSchedule_[tmp] = std::min(
                            BspSchedule<GraphT>::node_to_superstep_assignment[target] - this->GetStaleness(), commSchedule_[tmp]);
                    }
                }
            }
        }
    }

    void SetEagerCommunicationSchedule() {
        commSchedule_.clear();

        for (const auto &source : BspSchedule<GraphT>::instance->getComputationalDag().vertices()) {
            for (const auto &target : BspSchedule<GraphT>::instance->getComputationalDag().children(source)) {
                if (BspSchedule<GraphT>::node_to_processor_assignment[source]
                    != BspSchedule<GraphT>::node_to_processor_assignment[target]) {
                    commSchedule_[std::make_tuple(source,
                                                  BspSchedule<GraphT>::node_to_processor_assignment[source],
                                                  BspSchedule<GraphT>::node_to_processor_assignment[target])]
                        = BspSchedule<GraphT>::node_to_superstep_assignment[source];
                }
            }
        }
    }

    virtual void shrinkByMergingSupersteps() override {
        std::vector<unsigned> superstepLatestDependency(this->numberOfSupersteps_, 0);
        std::vector<std::vector<unsigned>> firstAt = GetFirstPresence();

        for (auto const &[key, val] : commSchedule_) {
            if (this->assignedProcessor(std::get<0>(key)) != std::get<1>(key)) {
                superstepLatestDependency[val]
                    = std::max(superstepLatestDependency[val], firstAt[std::get<0>(key)][std::get<1>(key)]);
            }
        }

        for (const auto &node : BspSchedule<GraphT>::instance->getComputationalDag().vertices()) {
            for (const auto &child : BspSchedule<GraphT>::instance->getComputationalDag().children(node)) {
                if (this->assignedProcessor(node) != this->assignedProcessor(child)) {
                    superstepLatestDependency[this->assignedSuperstep(child)] = std::max(
                        superstepLatestDependency[this->assignedSuperstep(child)], firstAt[node][this->assignedProcessor(child)]);
                }
            }
        }

        std::vector<bool> mergeWithPrevious(this->numberOfSupersteps_, false);
        for (unsigned step = this->numberOfSupersteps_ - 1; step < this->numberOfSupersteps_; --step) {
            unsigned limit = 0;
            while (step > limit) {
                limit = std::max(limit, superstepLatestDependency[step]);
                if (step > limit) {
                    mergeWithPrevious[step] = true;
                    --step;
                }
            }
        }

        std::vector<unsigned> newStepIndex(this->numberOfSupersteps_);
        unsigned currentIndex = std::numeric_limits<unsigned>::max();
        for (unsigned step = 0; step < this->numberOfSupersteps_; ++step) {
            if (!mergeWithPrevious[step]) {
                currentIndex++;
            }

            newStepIndex[step] = currentIndex;
        }
        for (const auto &node : this->instance_->vertices()) {
            this->nodeToSuperstepAssignment_[node] = newStepIndex[this->nodeToSuperstepAssignment_[node]];
        }
        for (auto &[key, val] : commSchedule_) {
            val = newStepIndex[val];
        }

        this->SetNumberOfSupersteps(currentIndex + 1);
    }

    // for each vertex v and processor p, find the first superstep where v is present on p by the end of the compute phase
    std::vector<std::vector<unsigned>> GetFirstPresence() const {
        std::vector<std::vector<unsigned>> firstAt(
            BspSchedule<GraphT>::instance->numberOfVertices(),
            std::vector<unsigned>(BspSchedule<GraphT>::instance->numberOfProcessors(), std::numeric_limits<unsigned>::max()));

        for (const auto &node : BspSchedule<GraphT>::instance->getComputationalDag().vertices()) {
            firstAt[node][this->assignedProcessor(node)] = this->assignedSuperstep(node);
        }

        for (auto const &[key, val] : commSchedule_) {
            firstAt[std::get<0>(key)][std::get<2>(key)]
                = std::min(firstAt[std::get<0>(key)][std::get<2>(key)], val + 1);    // TODO: replace by staleness after merge
        }

        return firstAt;
    }

    // remove unneeded comm. schedule entries - these can happen in ILPs, partial ILPs, etc.
    void CleanCommSchedule() {
        // data that is already present before it arrives
        std::vector<std::vector<std::multiset<unsigned>>> arrivesAt(
            BspSchedule<GraphT>::instance->numberOfVertices(),
            std::vector<std::multiset<unsigned>>(BspSchedule<GraphT>::instance->numberOfProcessors()));
        for (const auto &node : BspSchedule<GraphT>::instance->getComputationalDag().vertices()) {
            arrivesAt[node][this->assignedProcessor(node)].insert(this->assignedSuperstep(node));
        }

        for (auto const &[key, val] : commSchedule_) {
            arrivesAt[std::get<0>(key)][std::get<2>(key)].insert(val);
        }

        std::vector<KeyTriple> toErase;
        for (auto const &[key, val] : commSchedule_) {
            auto itr = arrivesAt[std::get<0>(key)][std::get<2>(key)].begin();
            if (*itr < val) {
                toErase.push_back(key);
            } else if (*itr == val && ++itr != arrivesAt[std::get<0>(key)][std::get<2>(key)].end() && *itr == val) {
                toErase.push_back(key);
                arrivesAt[std::get<0>(key)][std::get<2>(key)].erase(itr);
            }
        }

        for (const KeyTriple &key : toErase) {
            commSchedule_.erase(key);
        }

        // data that is not used after being sent
        std::vector<std::vector<std::multiset<unsigned>>> usedAt(
            BspSchedule<GraphT>::instance->numberOfVertices(),
            std::vector<std::multiset<unsigned>>(BspSchedule<GraphT>::instance->numberOfProcessors()));
        for (const auto &node : BspSchedule<GraphT>::instance->getComputationalDag().vertices()) {
            for (const auto &child : BspSchedule<GraphT>::instance->getComputationalDag().children(node)) {
                usedAt[node][this->assignedProcessor(child)].insert(this->assignedSuperstep(child));
            }
        }

        for (auto const &[key, val] : commSchedule_) {
            usedAt[std::get<0>(key)][std::get<1>(key)].insert(val);
        }

        // (need to visit cs entries in reverse superstep order here)
        std::vector<std::vector<KeyTriple>> entries(this->numberOfSupersteps_);
        for (auto const &[key, val] : commSchedule_) {
            entries[val].push_back(key);
        }

        toErase.clear();
        for (unsigned step = this->numberOfSupersteps_ - 1; step < this->numberOfSupersteps_; --step) {
            for (const KeyTriple &key : entries[step]) {
                if (usedAt[std::get<0>(key)][std::get<2>(key)].empty()
                    || *usedAt[std::get<0>(key)][std::get<2>(key)].rbegin() <= step) {
                    toErase.push_back(key);
                    auto itr = usedAt[std::get<0>(key)][std::get<1>(key)].find(step);
                    usedAt[std::get<0>(key)][std::get<1>(key)].erase(itr);
                }
            }
        }

        for (const KeyTriple &key : toErase) {
            commSchedule_.erase(key);
        }
    }
};

}    // namespace osp
