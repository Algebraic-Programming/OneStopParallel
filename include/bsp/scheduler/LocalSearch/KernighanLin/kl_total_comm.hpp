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
#include <chrono>
#include <climits>
#include <string>
#include <vector>

#include "kl_total.hpp"

namespace osp {
template<typename Graph_t>
class kl_total_comm : public kl_total<Graph_t> {

  protected:
    virtual void compute_comm_gain(vertex_idx_t<Graph_t> node, unsigned current_step, unsigned current_proc,
                                   unsigned new_proc) override {

        if (current_schedule.use_node_communication_costs) {

            if (current_proc == new_proc) {

                for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

                    if ((current_step + 1 == current_schedule.vector_schedule.assignedSuperstep(target) &&
                         current_proc != current_schedule.vector_schedule.assignedProcessor(target)) ||
                        (current_step == current_schedule.vector_schedule.assignedSuperstep(target) &&
                         current_proc == current_schedule.vector_schedule.assignedProcessor(target))) {
                        node_gains[node][current_proc][2] -= penalty;

                    } else if ((current_step == current_schedule.vector_schedule.assignedSuperstep(target) &&
                                current_proc != current_schedule.vector_schedule.assignedProcessor(target)) ||
                               (current_step - 1 == current_schedule.vector_schedule.assignedSuperstep(target) &&
                                current_proc == current_schedule.vector_schedule.assignedProcessor(target))) {

                        node_gains[node][current_proc][0] +=
                            (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(node) + reward;
                    }
                }

                for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

                    if ((current_step - 1 == current_schedule.vector_schedule.assignedSuperstep(source) &&
                         current_proc != current_schedule.vector_schedule.assignedProcessor(source)) ||
                        (current_step == current_schedule.vector_schedule.assignedSuperstep(source) &&
                         current_proc == current_schedule.vector_schedule.assignedProcessor(source))) {
                        node_gains[node][current_proc][0] -= penalty;

                    } else if ((current_step == current_schedule.vector_schedule.assignedSuperstep(source) &&
                                current_proc != current_schedule.vector_schedule.assignedProcessor(source)) ||
                               (current_step + 1 == current_schedule.vector_schedule.assignedSuperstep(source) &&
                                current_proc == current_schedule.vector_schedule.assignedProcessor(source))) {

                        node_gains[node][current_proc][2] +=
                            (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(source) +
                            reward;
                    }
                }
            } else {

                // current_proc != new_proc

                for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

                    const unsigned &target_proc = current_schedule.vector_schedule.assignedProcessor(target);
                    if (target_proc == current_proc) {

                        const double loss =
                            (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(node) *
                            current_schedule.instance->communicationCosts(new_proc, target_proc) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] -= loss;
                        node_gains[node][new_proc][1] -= loss;
                        node_gains[node][new_proc][2] -= loss;

                        node_change_in_costs[node][new_proc][0] += loss;
                        node_change_in_costs[node][new_proc][1] += loss;
                        node_change_in_costs[node][new_proc][2] += loss;

                        if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                            node_gains[node][new_proc][1] -= penalty;
                            node_gains[node][new_proc][2] -= penalty;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step + 1) {

                            node_gains[node][new_proc][2] -= penalty;
                        }

                    } else if (target_proc == new_proc) {

                        const double gain =
                            (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(node) *
                            current_schedule.instance->communicationCosts(current_proc, target_proc) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] += gain;
                        node_gains[node][new_proc][1] += gain;
                        node_gains[node][new_proc][2] += gain;

                        node_change_in_costs[node][new_proc][0] -= gain;
                        node_change_in_costs[node][new_proc][1] -= gain;
                        node_change_in_costs[node][new_proc][2] -= gain;

                        if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                            node_gains[node][new_proc][1] +=
                                (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(node) +
                                reward;

                            node_gains[node][new_proc][0] +=
                                (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(node) +
                                reward;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(target) < current_step) {

                            node_gains[node][new_proc][0] +=
                                (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(node) +
                                reward;
                        }

                    } else {

                        assert(target_proc != current_proc && target_proc != new_proc);

                        const double gain =
                            (double)(current_schedule.instance->communicationCosts(new_proc, target_proc) -
                                     current_schedule.instance->communicationCosts(current_proc, target_proc)) *
                            current_schedule.instance->getComputationalDag().vertex_comm_weight(node) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] += gain;
                        node_gains[node][new_proc][1] += gain;
                        node_gains[node][new_proc][2] += gain;

                        node_change_in_costs[node][new_proc][0] -= gain;
                        node_change_in_costs[node][new_proc][1] -= gain;
                        node_change_in_costs[node][new_proc][2] -= gain;

                        if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step + 1) {

                            node_gains[node][new_proc][2] -= penalty;
                        } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                            node_gains[node][new_proc][0] +=
                                (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(node) +
                                reward;
                        }
                    }
                }

                for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

                    const unsigned &source_proc = current_schedule.vector_schedule.assignedProcessor(source);
                    if (source_proc == current_proc) {

                        const double loss =
                            (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(source) *
                            current_schedule.instance->communicationCosts(current_proc, new_proc) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] -= loss;
                        node_gains[node][new_proc][1] -= loss;
                        node_gains[node][new_proc][2] -= loss;

                        node_change_in_costs[node][new_proc][0] += loss;
                        node_change_in_costs[node][new_proc][1] += loss;
                        node_change_in_costs[node][new_proc][2] += loss;

                        if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                            node_gains[node][new_proc][0] -= penalty;
                            node_gains[node][new_proc][1] -= penalty;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step - 1) {

                            node_gains[node][new_proc][0] -= penalty;
                        }

                    } else if (source_proc == new_proc) {

                        assert(source_proc != current_proc);
                        const double gain =
                            (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(source) *
                            current_schedule.instance->communicationCosts(current_proc, new_proc) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] += gain;
                        node_gains[node][new_proc][1] += gain;
                        node_gains[node][new_proc][2] += gain;

                        node_change_in_costs[node][new_proc][0] -= gain;
                        node_change_in_costs[node][new_proc][1] -= gain;
                        node_change_in_costs[node][new_proc][2] -= gain;

                        if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                            node_gains[node][new_proc][1] +=
                                (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(source) +
                                reward;

                            node_gains[node][new_proc][2] +=
                                (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(source) +
                                reward;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step + 1) {

                            node_gains[node][new_proc][2] +=
                                (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(source) +
                                reward;
                        }

                    } else {

                        assert(source_proc != current_proc && source_proc != new_proc);
                        const double gain =
                            (double)(current_schedule.instance->communicationCosts(new_proc, source_proc) -
                                     current_schedule.instance->communicationCosts(current_proc, source_proc)) *
                            current_schedule.instance->getComputationalDag().vertex_comm_weight(source) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] += gain;
                        node_gains[node][new_proc][1] += gain;
                        node_gains[node][new_proc][2] += gain;

                        node_change_in_costs[node][new_proc][0] -= gain;
                        node_change_in_costs[node][new_proc][1] -= gain;
                        node_change_in_costs[node][new_proc][2] -= gain;

                        if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step - 1) {

                            node_gains[node][new_proc][0] -= penalty;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                            node_gains[node][new_proc][2] +=
                                (double)current_schedule.instance->getComputationalDag().vertex_comm_weight(source) +
                                reward;
                        }
                    }
                }
            }
        } else {

            if (current_proc == new_proc) {

                for (const auto &out_edge : current_schedule.instance->getComputationalDag().out_edges(node)) {
                    const auto &target = target(out_edge, current_schedule.instance->getComputationalDag());
                    // for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

                    if ((current_step + 1 == current_schedule.vector_schedule.assignedSuperstep(target) &&
                         current_proc != current_schedule.vector_schedule.assignedProcessor(target)) ||
                        (current_step == current_schedule.vector_schedule.assignedSuperstep(target) &&
                         current_proc == current_schedule.vector_schedule.assignedProcessor(target))) {

                        node_gains[node][current_proc][2] -= penalty;

                    } else if ((current_step == current_schedule.vector_schedule.assignedSuperstep(target) &&
                                current_proc != current_schedule.vector_schedule.assignedProcessor(target)) ||
                               (current_step - 1 == current_schedule.vector_schedule.assignedSuperstep(target) &&
                                current_proc == current_schedule.vector_schedule.assignedProcessor(target))) {

                        node_gains[node][current_proc][0] +=
                            (double)current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) +
                            reward;
                    }
                }

                for (const auto &in_edge : current_schedule.instance->getComputationalDag().in_edges(node)) {

                    const auto &source = source(in_edge, current_schedule.instance->getComputationalDag());
                    // for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

                    if ((current_step - 1 == current_schedule.vector_schedule.assignedSuperstep(source) &&
                         current_proc != current_schedule.vector_schedule.assignedProcessor(source)) ||
                        (current_step == current_schedule.vector_schedule.assignedSuperstep(source) &&
                         current_proc == current_schedule.vector_schedule.assignedProcessor(source))) {

                        node_gains[node][current_proc][0] -= penalty;

                    } else if ((current_step == current_schedule.vector_schedule.assignedSuperstep(source) &&
                                current_proc != current_schedule.vector_schedule.assignedProcessor(source)) ||
                               (current_step + 1 == current_schedule.vector_schedule.assignedSuperstep(source) &&
                                current_proc == current_schedule.vector_schedule.assignedProcessor(source))) {

                        node_gains[node][current_proc][2] +=
                            (double)current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) + reward;
                    }
                }
            } else {

                // current_proc != new_proc

                for (const auto &out_edge : current_schedule.instance->getComputationalDag().out_edges(node)) {

                    const auto &target = target(out_edge, current_schedule.instance->getComputationalDag());
                    const unsigned &target_proc = current_schedule.vector_schedule.assignedProcessor(target);

                    if (target_proc == current_proc) {

                        const double loss =
                            (double)current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) *
                            current_schedule.instance->communicationCosts(new_proc, target_proc) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] -= loss;
                        node_gains[node][new_proc][1] -= loss;
                        node_gains[node][new_proc][2] -= loss;

                        node_change_in_costs[node][new_proc][0] += loss;
                        node_change_in_costs[node][new_proc][1] += loss;
                        node_change_in_costs[node][new_proc][2] += loss;

                        if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                            node_gains[node][new_proc][1] -= penalty;
                            node_gains[node][new_proc][2] -= penalty;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step + 1) {

                            node_gains[node][new_proc][2] -= penalty;
                        }

                    } else if (target_proc == new_proc) {

                        const double gain =
                            (double)current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) *
                            current_schedule.instance->communicationCosts(current_proc, target_proc) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] += gain;
                        node_gains[node][new_proc][1] += gain;
                        node_gains[node][new_proc][2] += gain;

                        node_change_in_costs[node][new_proc][0] -= gain;
                        node_change_in_costs[node][new_proc][1] -= gain;
                        node_change_in_costs[node][new_proc][2] -= gain;

                        if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                            node_gains[node][new_proc][1] +=
                                (double)current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) +
                                reward;
                            node_gains[node][new_proc][0] +=
                                (double)current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) +
                                reward;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step - 1) {

                            node_gains[node][new_proc][0] +=
                                (double)current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) +
                                reward;
                        }

                    } else {

                        assert(target_proc != current_proc && target_proc != new_proc);

                        const double gain =
                            (double)(current_schedule.instance->communicationCosts(new_proc, target_proc) -
                                     current_schedule.instance->communicationCosts(current_proc, target_proc)) *
                            current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] += gain;
                        node_gains[node][new_proc][1] += gain;
                        node_gains[node][new_proc][2] += gain;

                        node_change_in_costs[node][new_proc][0] -= gain;
                        node_change_in_costs[node][new_proc][1] -= gain;
                        node_change_in_costs[node][new_proc][2] -= gain;

                        if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step + 1) {

                            node_gains[node][new_proc][2] -= penalty;
                        } else if (current_schedule.vector_schedule.assignedSuperstep(target) == current_step) {

                            node_gains[node][new_proc][0] +=
                                (double)current_schedule.instance->getComputationalDag().edge_comm_weight(out_edge) +
                                reward;
                        }
                    }
                }

                for (const auto &in_edge : current_schedule.instance->getComputationalDag().in_edges(node)) {
                    const auto &source = source(in_edge, current_schedule.instance->getComputationalDag());

                    const unsigned &source_proc = current_schedule.vector_schedule.assignedProcessor(source);
                    if (source_proc == current_proc) {

                        const double loss =
                            (double)current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) *
                            current_schedule.instance->communicationCosts(current_proc, new_proc) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] -= loss;
                        node_gains[node][new_proc][1] -= loss;
                        node_gains[node][new_proc][2] -= loss;

                        node_change_in_costs[node][new_proc][0] += loss;
                        node_change_in_costs[node][new_proc][1] += loss;
                        node_change_in_costs[node][new_proc][2] += loss;

                        if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                            node_gains[node][new_proc][0] -= penalty;
                            node_gains[node][new_proc][1] -= penalty;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step - 1) {

                            node_gains[node][new_proc][0] -= penalty;
                        }
                    } else if (source_proc == new_proc) {

                        assert(source_proc != current_proc);
                        const double gain =
                            (double)current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) *
                            current_schedule.instance->communicationCosts(current_proc, new_proc) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] += gain;
                        node_gains[node][new_proc][1] += gain;
                        node_gains[node][new_proc][2] += gain;

                        node_change_in_costs[node][new_proc][0] -= gain;
                        node_change_in_costs[node][new_proc][1] -= gain;
                        node_change_in_costs[node][new_proc][2] -= gain;

                        if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                            node_gains[node][new_proc][1] +=
                                (double)current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) +
                                reward;

                            node_gains[node][new_proc][2] +=
                                (double)current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) +
                                reward;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step + 1) {

                            node_gains[node][new_proc][2] +=
                                (double)current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) +
                                reward;
                        }

                    } else {

                        assert(source_proc != current_proc && source_proc != new_proc);
                        const double gain =
                            (double)(current_schedule.instance->communicationCosts(new_proc, source_proc) -
                                     current_schedule.instance->communicationCosts(current_proc, source_proc)) *
                            current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) *
                            current_schedule.comm_multiplier;

                        node_gains[node][new_proc][0] += gain;
                        node_gains[node][new_proc][1] += gain;
                        node_gains[node][new_proc][2] += gain;

                        node_change_in_costs[node][new_proc][0] -= gain;
                        node_change_in_costs[node][new_proc][1] -= gain;
                        node_change_in_costs[node][new_proc][2] -= gain;

                        if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step - 1) {

                            node_gains[node][new_proc][0] -= penalty;

                        } else if (current_schedule.vector_schedule.assignedSuperstep(source) == current_step) {

                            node_gains[node][new_proc][2] +=
                                (double)current_schedule.instance->getComputationalDag().edge_comm_weight(in_edge) +
                                reward;
                        }
                    }
                }
            }
        }
    }

    virtual double compute_current_costs() override {

        double work_costs = 0;
        for (unsigned step = 0; step < current_schedule.num_steps(); step++) {
            work_costs += current_schedule.step_max_work[step];
        }

        double comm_costs = 0;
        for (const auto &edge : current_schedule.instance->getComputationalDag().edges()) {

            const unsigned &source = current_schedule.instance->getComputationalDag().source(edge);
            const unsigned &source_proc = current_schedule.vector_schedule.assignedProcessor(source);
            const unsigned &target_proc = current_schedule.vector_schedule.assignedProcessor(
                target(edge, current_schedule.instance->getComputationalDag()));

            if (source_proc != target_proc) {

                if (current_schedule.use_node_communication_costs) {
                    comm_costs += current_schedule.instance->getComputationalDag().vertex_comm_weight(source) *
                                  current_schedule.instance->communicationCosts(source_proc, target_proc);
                } else {
                    comm_costs += current_schedule.instance->getComputationalDag().edge_comm_weight(edge) *
                                  current_schedule.instance->communicationCosts(source_proc, target_proc);
                }
            }
        }

        current_schedule.current_cost =
            work_costs + comm_costs * current_schedule.comm_multiplier +
            (current_schedule.num_steps() - 1) * current_schedule.instance->synchronisationCosts();

        return current_schedule.current_cost;
    }

  public:
    kl_total_comm(bool use_node_communication_costs_ = false) : kl_total(use_node_communication_costs_) {}

    virtual ~kl_total_comm() = default;

    virtual std::string getScheduleName() const override { return "KLTotalComm"; }
};

template<typename Graph_t>
class kl_total_comm_test : public kl_total_comm<Graph_t> {

  public:
    kl_total_comm_test() : kl_total_comm() {}

    virtual ~kl_total_comm_test() = default;

    virtual std::string getScheduleName() const override { return "KLBaseTest"; }

    kl_current_schedule_total &get_current_schedule() { return current_schedule; }

    auto &get_node_gains() { return node_gains; }
    auto &get_node_change_in_costs() { return node_change_in_costs; }
    auto &get_max_gain_heap() { return max_gain_heap; }

    void initialize_gain_heap_test(const std::unordered_set<VertexType> &nodes, double reward_ = 0.0,
                                   double penalty_ = 0.0) {
        reward = reward_;
        penalty = penalty_;

        initialize_gain_heap(nodes);
    }

    void test_setup_schedule(BspSchedule<Graph_t> &schedule) {

        current_schedule.instance = &schedule.getInstance();

        best_schedule = &schedule;

        num_nodes = current_schedule.instance->numberOfVertices();
        num_procs = current_schedule.instance->numberOfProcessors();

        set_parameters();
        initialize_datastructures();
    }

    RETURN_STATUS improve_schedule_test_1(BspSchedule<Graph_t> &schedule) {

        current_schedule.instance = &schedule.getInstance();

        best_schedule = &schedule;
        num_nodes = current_schedule.instance->numberOfVertices();
        num_procs = current_schedule.instance->numberOfProcessors();

        set_parameters();
        initialize_datastructures();

        bool improvement_found = run_local_search_simple();

        assert(best_schedule->satisfiesPrecedenceConstraints());

        schedule.setImprovedLazyCommunicationSchedule();

        if (improvement_found)
            return SUCCESS;
        else
            return BEST_FOUND;
    }

    RETURN_STATUS improve_schedule_test_2(BspSchedule<Graph_t> &schedule) {

        current_schedule.instance = &schedule.getInstance();

        best_schedule = &schedule;
        num_nodes = current_schedule.instance->numberOfVertices();
        num_procs = current_schedule.instance->numberOfProcessors();

        set_parameters();
        initialize_datastructures();

        bool improvement_found = run_local_search_unlock_delay();

        assert(best_schedule->satisfiesPrecedenceConstraints());

        schedule.setImprovedLazyCommunicationSchedule();

        if (improvement_found)
            return SUCCESS;
        else
            return BEST_FOUND;
    }
};

} // namespace osp