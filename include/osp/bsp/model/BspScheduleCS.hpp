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


#include "IBspScheduleEval.hpp"
#include "BspSchedule.hpp"

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
class BspScheduleCS : public BspSchedule<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "BspScheduleCS can only be used with computational DAGs.");

  public:
    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;

    // contains entries: (vertex, from_proc, to_proc ) : step
    std::map<KeyTriple, unsigned> commSchedule;

  protected:

    void compute_cs_communication_costs_helper(std::vector<std::vector<v_commw_t<Graph_t>>> & rec, std::vector<std::vector<v_commw_t<Graph_t>>> & send) const {        
        for (auto const &[key, val] : commSchedule) {
            send[std::get<1>(key)][val] +=
                BspSchedule<Graph_t>::instance->sendCosts(std::get<1>(key), std::get<2>(key)) *
                BspSchedule<Graph_t>::instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
            rec[std::get<2>(key)][val] +=
                BspSchedule<Graph_t>::instance->sendCosts(std::get<1>(key), std::get<2>(key)) *
                BspSchedule<Graph_t>::instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
        }       
    }

  public:
    BspScheduleCS() = delete;

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    BspScheduleCS(const BspInstance<Graph_t> &inst) : BspSchedule<Graph_t>(inst) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, and superstep
     * assignment.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     */
    BspScheduleCS(const BspInstance<Graph_t> &inst, const std::vector<unsigned> &processor_assignment_,
                  const std::vector<unsigned> &superstep_assignment_)
        : BspSchedule<Graph_t>(inst, processor_assignment_, superstep_assignment_) {}

    /**
     * @brief Constructs a BspSchedule object with the specified BspInstance, processor assignment, superstep
     * assignment, and communication schedule.
     *
     * @param inst The BspInstance for the schedule.
     * @param processor_assignment_ The processor assignment for the nodes.
     * @param superstep_assignment_ The superstep assignment for the nodes.
     * @param comm_ The communication schedule for the nodes.
     */
    BspScheduleCS(const BspInstance<Graph_t> &inst, const std::vector<unsigned int> &processor_assignment_,
                  const std::vector<unsigned int> &superstep_assignment_,
                  const std::map<KeyTriple, unsigned int> &comm_)
        : BspSchedule<Graph_t>(inst, processor_assignment_, superstep_assignment_), commSchedule(comm_) {}

    BspScheduleCS(BspSchedule<Graph_t> &&schedule) : BspSchedule<Graph_t>(std::move(schedule)) {
        setAutoCommunicationSchedule();
    }

    BspScheduleCS(BspSchedule<Graph_t> &&schedule, const std::map<KeyTriple, unsigned int> &comm_)
        : BspSchedule<Graph_t>(std::move(schedule)), commSchedule(comm_) {}

    BspScheduleCS(BspSchedule<Graph_t> &&schedule, std::map<KeyTriple, unsigned int> &&comm_)
        : BspSchedule<Graph_t>(std::move(schedule)), commSchedule(std::move(comm_)) {
        comm_.clear();
    }

    BspScheduleCS(const BspSchedule<Graph_t> &schedule) : BspSchedule<Graph_t>(schedule) {
        setAutoCommunicationSchedule();
    }

    BspScheduleCS(const BspSchedule<Graph_t> &schedule, const std::map<KeyTriple, unsigned int> &comm_)
        : BspSchedule<Graph_t>(schedule), commSchedule(comm_) {}

    BspScheduleCS(const BspScheduleCS &other) = default;
    BspScheduleCS(BspScheduleCS &&other) = default;
    BspScheduleCS &operator=(const BspScheduleCS &other) = default;
    BspScheduleCS &operator=(BspScheduleCS &&other) = default;
    virtual ~BspScheduleCS() = default;

    inline const std::map<KeyTriple, unsigned> &getCommunicationSchedule() const { return commSchedule; }
    inline std::map<KeyTriple, unsigned> &getCommunicationSchedule() { return commSchedule; }

    inline bool hasValidCommSchedule() const { return checkCommScheduleValidity(commSchedule); }

    void addCommunicationScheduleEntry(KeyTriple key, unsigned step) {

        if (step >= BspSchedule<Graph_t>::number_of_supersteps)
            throw std::invalid_argument(
                "Invalid Argument while adding communication schedule entry: step out of range.");

        if (std::get<0>(key) >= BspSchedule<Graph_t>::instance->numberOfVertices())
            throw std::invalid_argument(
                "Invalid Argument while adding communication schedule entry: node out of range.");

        if (std::get<1>(key) >= BspSchedule<Graph_t>::instance->numberOfProcessors())
            throw std::invalid_argument(
                "Invalid Argument while adding communication schedule entry: from processor out of range.");

        if (std::get<2>(key) >= BspSchedule<Graph_t>::instance->numberOfProcessors())
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
    void addCommunicationScheduleEntry(vertex_idx node, unsigned from_proc, unsigned to_proc, unsigned step) {
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
            BspSchedule<Graph_t>::instance->numberOfVertices(),
            std::vector<unsigned>(BspSchedule<Graph_t>::instance->numberOfProcessors(),
                                  BspSchedule<Graph_t>::number_of_supersteps));

        for (const auto &node : BspSchedule<Graph_t>::instance->vertices()) {
            first_at[node][BspSchedule<Graph_t>::node_to_processor_assignment[node]] =
                BspSchedule<Graph_t>::node_to_superstep_assignment[node];
        }

        for (auto const &[key, val] : cs) {

            if (val >= BspSchedule<Graph_t>::number_of_supersteps)
                return false;

            if (std::get<0>(key) >= BspSchedule<Graph_t>::instance->numberOfVertices())
                return false;

            if (std::get<1>(key) >= BspSchedule<Graph_t>::instance->numberOfProcessors())
                return false;

            if (std::get<2>(key) >= BspSchedule<Graph_t>::instance->numberOfProcessors())
                return false;

            first_at[std::get<0>(key)][std::get<2>(key)] =
                std::min(first_at[std::get<0>(key)][std::get<2>(key)], val + 1);
        }

        for (auto const &[key, val] : cs) {

            if (val < first_at[std::get<0>(key)][std::get<1>(key)]) {
                return false;
            }
        }

        for (const auto &v : BspSchedule<Graph_t>::instance->getComputationalDag().vertices()) {
            for (const auto &target : BspSchedule<Graph_t>::instance->getComputationalDag().children(v)) {

                if (BspSchedule<Graph_t>::node_to_processor_assignment[v] !=
                    BspSchedule<Graph_t>::node_to_processor_assignment[target]) {
                    if (first_at[v][BspSchedule<Graph_t>::node_to_processor_assignment[target]] >
                        BspSchedule<Graph_t>::node_to_superstep_assignment[target]) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    v_commw_t<Graph_t> compute_cs_communication_costs() const {

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(
            BspSchedule<Graph_t>::instance->numberOfProcessors(),
            std::vector<v_commw_t<Graph_t>>(BspSchedule<Graph_t>::number_of_supersteps, 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(
            BspSchedule<Graph_t>::instance->numberOfProcessors(),
            std::vector<v_commw_t<Graph_t>>(BspSchedule<Graph_t>::number_of_supersteps, 0));

        compute_cs_communication_costs_helper(rec, send);
        const std::vector<v_commw_t<Graph_t>> max_comm_per_step = this->compute_max_comm_per_step_helper(rec, send);

        v_commw_t<Graph_t> costs = 0;
        for (unsigned step = 0; step < this->number_of_supersteps; step++) {
            const auto step_comm_cost = max_comm_per_step[step];
            costs += step_comm_cost;

            if (step_comm_cost > 0) {
                costs += this->instance->synchronisationCosts();
            }
        }
        return costs;
    }

    virtual v_workw_t<Graph_t> computeCosts() const override {
        return compute_cs_communication_costs() + this->computeWorkCosts();
    }

    void setAutoCommunicationSchedule() {
        std::map<KeyTriple, unsigned> best_comm_schedule;
        v_workw_t<Graph_t> best_comm_cost =
            std::numeric_limits<v_workw_t<Graph_t>>::max(); // computeCosts retunrs v_workw_t<Graph_t>

        if (hasValidCommSchedule()) {
            v_workw_t<Graph_t> costs_com = BspSchedule<Graph_t>::computeCosts();
            if (costs_com < best_comm_cost) {
                best_comm_schedule = commSchedule;
                best_comm_cost = costs_com;
            }
        }

        setImprovedLazyCommunicationSchedule();
        v_workw_t<Graph_t> costs_com = BspSchedule<Graph_t>::computeCosts();
        // std::cout << "Improved Lazy: " << costs_com << std::endl;
        if (costs_com < best_comm_cost) {
            best_comm_schedule = commSchedule;
            best_comm_cost = costs_com;
        }

        setLazyCommunicationSchedule();
        costs_com = BspSchedule<Graph_t>::computeCosts();
        // std::cout << "Lazy: " << costs_com << std::endl;
        if (costs_com < best_comm_cost) {
            best_comm_schedule = commSchedule;
            best_comm_cost = costs_com;
        }

        setEagerCommunicationSchedule();
        costs_com = BspSchedule<Graph_t>::computeCosts();
        // std::cout << "Eager: " << costs_com << std::endl;
        if (costs_com < best_comm_cost) {
            best_comm_schedule = commSchedule;
            best_comm_cost = costs_com;
        }

        commSchedule = best_comm_schedule;
    }

    void setImprovedLazyCommunicationSchedule() {
        commSchedule.clear();
        if (BspSchedule<Graph_t>::instance->getComputationalDag().num_vertices() <= 1 ||
            BspSchedule<Graph_t>::number_of_supersteps <= 1)
            return;

        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> step_proc_node_list(
            BspSchedule<Graph_t>::number_of_supersteps,
            std::vector<std::vector<vertex_idx_t<Graph_t>>>(BspSchedule<Graph_t>::instance->numberOfProcessors(),
                                                            std::vector<vertex_idx_t<Graph_t>>()));
        std::vector<std::vector<bool>> node_to_proc_been_sent(
            BspSchedule<Graph_t>::instance->numberOfVertices(),
            std::vector<bool>(BspSchedule<Graph_t>::instance->numberOfProcessors(), false));

        for (vertex_idx_t<Graph_t> node = 0; node < BspSchedule<Graph_t>::instance->numberOfVertices(); node++) {
            step_proc_node_list[BspSchedule<Graph_t>::node_to_superstep_assignment[node]]
                               [BspSchedule<Graph_t>::node_to_processor_assignment[node]]
                                   .push_back(node);
            node_to_proc_been_sent[node][BspSchedule<Graph_t>::node_to_processor_assignment[node]] = true;
        }

        // processor, ordered list of (cost, node, to_processor)
        std::vector<std::set<std::vector<vertex_idx_t<Graph_t>>, std::greater<>>> require_sending(
            BspSchedule<Graph_t>::instance->numberOfProcessors());
        // TODO the datastructure seems to be wrong. the vectors added to the set have elements of different types.
        // it should really be std::vector<std::set<std::tuple<v_commw_t<Graph_t>, vertex_idx_t<Graph_t>,
        // vertex_idx_t<Graph_t>>>> added many static_cast below as tmp fix

        for (unsigned proc = 0; proc < BspSchedule<Graph_t>::instance->numberOfProcessors(); proc++) {
            for (const auto &node : step_proc_node_list[0][proc]) {

                for (const auto &target : BspSchedule<Graph_t>::instance->getComputationalDag().children(node)) {
                    if (proc != BspSchedule<Graph_t>::assignedProcessor(target)) {
                        require_sending[proc].insert(
                            {static_cast<vertex_idx_t<Graph_t>>(
                                 BspSchedule<Graph_t>::instance->getComputationalDag().vertex_comm_weight(node) *
                                 BspSchedule<Graph_t>::instance->getArchitecture().sendCosts(
                                     proc, BspSchedule<Graph_t>::node_to_processor_assignment[target])),
                             node, BspSchedule<Graph_t>::node_to_processor_assignment[target]});
                    }
                }
            }
        }

        for (unsigned step = 1; step < BspSchedule<Graph_t>::number_of_supersteps; step++) {
            std::vector<v_commw_t<Graph_t>> send_cost(BspSchedule<Graph_t>::instance->numberOfProcessors(), 0);
            std::vector<v_commw_t<Graph_t>> receive_cost(BspSchedule<Graph_t>::instance->numberOfProcessors(), 0);

            // must send in superstep step-1
            for (unsigned proc = 0; proc < BspSchedule<Graph_t>::instance->numberOfProcessors(); proc++) {
                for (const auto &node : step_proc_node_list[step][proc]) {
                    for (const auto &source : BspSchedule<Graph_t>::instance->getComputationalDag().parents(node)) {

                        if (!node_to_proc_been_sent[source][proc]) {
                            assert(BspSchedule<Graph_t>::node_to_superstep_assignment[source] < step);
                            commSchedule.emplace(
                                std::make_tuple(source, BspSchedule<Graph_t>::node_to_processor_assignment[source],
                                                proc),
                                step - 1);
                            node_to_proc_been_sent[source][proc] = true;
                            v_commw_t<Graph_t> comm_cost =
                                BspSchedule<Graph_t>::instance->getComputationalDag().vertex_comm_weight(source) *
                                BspSchedule<Graph_t>::instance->getArchitecture().sendCosts(
                                    BspSchedule<Graph_t>::node_to_processor_assignment[source], proc);
                            require_sending[BspSchedule<Graph_t>::assignedProcessor(source)].erase(
                                {static_cast<vertex_idx_t<Graph_t>>(comm_cost), source, proc});
                            send_cost[BspSchedule<Graph_t>::node_to_processor_assignment[source]] += comm_cost;
                            receive_cost[proc] += comm_cost;
                        }
                    }
                }
            }

            // getting max costs
            v_commw_t<Graph_t> max_comm_cost = 0;
            for (size_t proc = 0; proc < BspSchedule<Graph_t>::instance->numberOfProcessors(); proc++) {
                max_comm_cost = std::max(max_comm_cost, send_cost[proc]);
                max_comm_cost = std::max(max_comm_cost, receive_cost[proc]);
            }

            // extra sends
            // TODO: permute the order of processors
            for (size_t proc = 0; proc < BspSchedule<Graph_t>::instance->numberOfProcessors(); proc++) {
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
            for (unsigned proc = 0; proc < BspSchedule<Graph_t>::instance->numberOfProcessors(); proc++) {
                for (const auto &node : step_proc_node_list[step][proc]) {

                    for (const auto &target : BspSchedule<Graph_t>::instance->getComputationalDag().children(node))
                        if (proc != BspSchedule<Graph_t>::assignedProcessor(target)) {
                            require_sending[proc].insert(
                                {static_cast<vertex_idx_t<Graph_t>>(
                                     BspSchedule<Graph_t>::instance->getComputationalDag().vertex_comm_weight(node) *
                                     BspSchedule<Graph_t>::instance->getArchitecture().sendCosts(
                                         proc, BspSchedule<Graph_t>::node_to_processor_assignment[target])),
                                 node, BspSchedule<Graph_t>::node_to_processor_assignment[target]});
                        }
                }
            }
        }
    }

    void setLazyCommunicationSchedule() {
        commSchedule.clear();

        for (const auto &source : BspSchedule<Graph_t>::instance->getComputationalDag().vertices()) {
            for (const auto &target : BspSchedule<Graph_t>::instance->getComputationalDag().children(source)) {

                if (BspSchedule<Graph_t>::node_to_processor_assignment[source] !=
                    BspSchedule<Graph_t>::node_to_processor_assignment[target]) {

                    const auto tmp = std::make_tuple(source, BspSchedule<Graph_t>::node_to_processor_assignment[source],
                                                     BspSchedule<Graph_t>::node_to_processor_assignment[target]);
                    if (commSchedule.find(tmp) == commSchedule.end()) {
                        commSchedule[tmp] = BspSchedule<Graph_t>::node_to_superstep_assignment[target] - 1;

                    } else {
                        commSchedule[tmp] =
                            std::min(BspSchedule<Graph_t>::node_to_superstep_assignment[target] - 1, commSchedule[tmp]);
                    }
                }
            }
        }
    }
    void setEagerCommunicationSchedule() {
        commSchedule.clear();

        for (const auto &source : BspSchedule<Graph_t>::instance->getComputationalDag().vertices()) {
            for (const auto &target : BspSchedule<Graph_t>::instance->getComputationalDag().children(source)) {

                if (BspSchedule<Graph_t>::node_to_processor_assignment[source] !=
                    BspSchedule<Graph_t>::node_to_processor_assignment[target]) {

                    commSchedule[std::make_tuple(source, BspSchedule<Graph_t>::node_to_processor_assignment[source],
                                                 BspSchedule<Graph_t>::node_to_processor_assignment[target])] =
                        BspSchedule<Graph_t>::node_to_superstep_assignment[source];
                }
            }
        }
    }

    virtual void shrinkSchedule() override {

        std::vector<bool> comm_phase_empty(this->number_of_supersteps, true);
        for (auto const &[key, val] : commSchedule)
            comm_phase_empty[val] = false;

        std::vector<unsigned> new_step_index(this->number_of_supersteps);
        unsigned current_index = 0;
        for(unsigned step = 0; step < this->number_of_supersteps; ++step)
        {
            new_step_index[step] = current_index;
            if(!comm_phase_empty[step])
                current_index++;
        }
        for (const auto& node : this->instance->vertices())
            this->node_to_superstep_assignment[node] = new_step_index[this->node_to_superstep_assignment[node]];
        for (auto &[key, val] : commSchedule)
            val = new_step_index[val];

        this->setNumberOfSupersteps(current_index);
    }
};

} // namespace osp