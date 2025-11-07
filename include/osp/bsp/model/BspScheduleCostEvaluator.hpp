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

#include "BspSchedule.hpp"

namespace osp {

/**
 * @class BspScheduleCostEvaluator
 * @brief A class to compute various cost functions for a BspSchedule.
 *
 * This class wraps a BspSchedule by reference to avoid unnecessary copies
 * while providing an interface to compute different cost models.
 */
template<typename Graph_t>
class BspScheduleCostEvaluator {

    static_assert(is_computational_dag_v<Graph_t>, "BspScheduleCostEvaluator can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>,
                  "BspScheduleCostEvaluator requires work and comm. weights to have the same type.");

  protected:
    const BspSchedule<Graph_t>& schedule;
    const BspInstance<Graph_t>& instance;

    void compute_lazy_communication_costs_helper(std::vector<std::vector<v_commw_t<Graph_t>>> & rec, std::vector<std::vector<v_commw_t<Graph_t>>> & send) const {
        const unsigned number_of_supersteps = schedule.numberOfSupersteps();
        for (const auto &node : instance.vertices()) {

            std::vector<unsigned> step_needed(instance.numberOfProcessors(), number_of_supersteps);
            for (const auto &target : instance.getComputationalDag().children(node)) {

                if (schedule.assignedProcessor(node) != schedule.assignedProcessor(target)) {
                    step_needed[schedule.assignedProcessor(target)] = std::min(
                        step_needed[schedule.assignedProcessor(target)], schedule.assignedSuperstep(target));
                }
            }

            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {

                if (step_needed[proc] < number_of_supersteps) {

                    send[schedule.assignedProcessor(node)][step_needed[proc] - 1] +=
                        instance.sendCosts(schedule.assignedProcessor(node), proc) *
                        instance.getComputationalDag().vertex_comm_weight(node);

                    rec[proc][step_needed[proc] - 1] += instance.sendCosts(schedule.assignedProcessor(node), proc) *
                                                        instance.getComputationalDag().vertex_comm_weight(node);
                }
            }
        }
    }

    std::vector<v_commw_t<Graph_t>> compute_max_comm_per_step_helper(const std::vector<std::vector<v_commw_t<Graph_t>>> & rec, const std::vector<std::vector<v_commw_t<Graph_t>>> & send) const {
        const unsigned number_of_supersteps = schedule.numberOfSupersteps();
        std::vector<v_commw_t<Graph_t>> max_comm_per_step(number_of_supersteps, 0);
        for (unsigned step = 0; step < number_of_supersteps; step++) {
            v_commw_t<Graph_t> max_send = 0;
            v_commw_t<Graph_t> max_rec = 0;

            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                if (max_send < send[proc][step])
                    max_send = send[proc][step];
                if (max_rec < rec[proc][step])
                    max_rec = rec[proc][step];
            }
            max_comm_per_step[step] = std::max(max_send, max_rec) * instance.communicationCosts();
        }
        return max_comm_per_step;
    }

  public:
    /**
     * @brief Construct a new Bsp Schedule Cost Evaluator object.
     *
     * @param sched The BspSchedule to evaluate.
     */
    BspScheduleCostEvaluator(const BspSchedule<Graph_t>& sched) : schedule(sched), instance(sched.getInstance()) {}

    /**
     * @brief Computes the communication costs using the lazy sending model.
     *
     * In the lazy sending model, data is sent in the superstep immediately
     * preceding the superstep where it is first needed.
     *
     * @return The lazy communication costs.
     */
    v_commw_t<Graph_t> compute_lazy_communication_costs() const {

        const unsigned number_of_supersteps = schedule.numberOfSupersteps();

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(instance.numberOfProcessors(),
                                                         std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(instance.numberOfProcessors(),
                                                          std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));

        compute_lazy_communication_costs_helper(rec, send);
        const std::vector<v_commw_t<Graph_t>> max_comm_per_step = compute_max_comm_per_step_helper(rec, send);

        v_commw_t<Graph_t> costs = 0;
        for (unsigned step = 0; step < number_of_supersteps; step++) {
            const auto step_comm_cost = max_comm_per_step[step];
            costs += step_comm_cost;
            
            costs += instance.synchronisationCosts();
            
        }

        return costs;
    }

    /**
     * @brief Computes the work costs for each superstep.
     *
     * @return The work cost per superstep.
     */
    std::vector<v_workw_t<Graph_t>> compute_max_work_per_step_helper() const {
        const unsigned number_of_supersteps = schedule.numberOfSupersteps();
        std::vector<std::vector<v_workw_t<Graph_t>>> work = std::vector<std::vector<v_workw_t<Graph_t>>>(
            number_of_supersteps, std::vector<v_workw_t<Graph_t>>(instance.numberOfProcessors(), 0));
        for (const auto &node : instance.vertices()) {
            work[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)] +=
                instance.getComputationalDag().vertex_work_weight(node);
        }

        std::vector<v_workw_t<Graph_t>> max_work_per_step(number_of_supersteps, 0);
        for (unsigned step = 0; step < number_of_supersteps; step++) {
            v_workw_t<Graph_t> max_work = 0;
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                if (max_work < work[step][proc]) {
                    max_work = work[step][proc];
                }
            }

            max_work_per_step[step] = max_work;
        }

        return max_work_per_step;
    }

    /**
     * @brief Computes the total work costs of the schedule.
     *
     * The work cost is the sum of the maximum work done in each superstep
     * across all processors.
     *
     * @return The total work costs.
     */
    v_workw_t<Graph_t> computeWorkCosts() const {
        const std::vector<v_workw_t<Graph_t>> work_per_step = compute_max_work_per_step_helper();
        return std::accumulate(work_per_step.begin(), work_per_step.end(), static_cast<v_workw_t<Graph_t>>(0));
    }

    /**
     * @brief Computes the total costs of the schedule using the lazy communication model.
     *
     * @return The total costs.
     */
    v_workw_t<Graph_t> computeCosts() const { return compute_lazy_communication_costs() + computeWorkCosts(); }
};

} // namespace osp
