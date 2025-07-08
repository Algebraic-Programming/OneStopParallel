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

#include <deque>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "ClassicSchedule.hpp"
#include "MemoryConstraintModules.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

enum CilkMode { CILK, SJF };

/**
 * @class GreedyCilkScheduler
 * @brief A class that represents a greedy scheduler for Cilk-based BSP scheduling scheduler.
 *
 * The GreedyCilkScheduler class is a concrete implementation of the Scheduler base class. It provides
 * a greedy scheduling algorithm for Cilk-based BSP (Bulk Synchronous Parallel) systems. The scheduler
 * selects the next node and processor to execute a task based on a greedy strategy.
 */
template<typename Graph_t>
class CilkScheduler : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "CilkScheduler can only be used with computational DAGs.");

  private:
    using tv_pair = std::pair<v_workw_t<Graph_t>, vertex_idx_t<Graph_t>>;

    CilkMode mode; /**< The mode of the Cilk scheduler. */

    // constexpr static bool use_memory_constraint = is_memory_constraint_v<MemoryConstraint_t>;

    // static_assert(not use_memory_constraint ||
    //                   std::is_same_v<MemoryConstraint_t, persistent_transient_memory_constraint<Graph_t>>,
    //               "CilkScheduler implements only persistent_transient_memory_constraint.");

    // MemoryConstraint_t memory_constraint;

    std::mt19937 gen;

    void Choose(const BspInstance<Graph_t> &instance, std::vector<std::deque<vertex_idx_t<Graph_t>>> &procQueue,
                const std::set<vertex_idx_t<Graph_t>> &readyNodes, const std::vector<bool> &procFree,
                vertex_idx_t<Graph_t> &node, unsigned &p) {
        if (mode == SJF) {

            node = *readyNodes.begin();
            for (auto &r : readyNodes)
                if (instance.getComputationalDag().vertex_work_weight(r) <
                    instance.getComputationalDag().vertex_work_weight(node))
                    node = r;

            p = 0;
            for (; p < instance.numberOfProcessors(); ++p)
                if (procFree[p])
                    break;

        } else if (mode == CILK) {
            for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
                if (procFree[i] && !procQueue[i].empty()) {
                    p = i;
                    node = procQueue[i].back();
                    procQueue[i].pop_back();
                    return;
                }

            // Time to steal
            for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
                if (procFree[i]) {
                    p = i;
                    break;
                }

            std::vector<unsigned> canStealFrom;
            for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
                if (!procQueue[i].empty())
                    canStealFrom.push_back(i);

            if (canStealFrom.empty()) {
                node = std::numeric_limits<vertex_idx_t<Graph_t>>::max();
                return;
            }

            // Pick a random queue to steal from
            std::uniform_int_distribution<unsigned> dis(0, static_cast<unsigned>(canStealFrom.size() - 1));
            const unsigned chosenIndex = dis(gen);
            const unsigned chosenQueue = canStealFrom[chosenIndex];
            node = procQueue[chosenQueue].front();
            procQueue[chosenQueue].pop_front();
        }
    }

  public:
    /**
     * @brief Constructs a GreedyCilkScheduler object with the specified Cilk mode.
     *
     * This constructor initializes a GreedyCilkScheduler object with the specified Cilk mode.
     *
     * @param mode_ The Cilk mode for the scheduler.
     */
    CilkScheduler(CilkMode mode_ = CILK) : Scheduler<Graph_t>(), mode(mode_), gen(std::random_device{}()) {}

    /**
     * @brief Destroys the GreedyCilkScheduler object.
     *
     * This destructor destroys the GreedyCilkScheduler object.
     */
    virtual ~CilkScheduler() = default;

    /**
     * @brief Computes the schedule for the given BSP instance using the greedy scheduling algorithm.
     *
     * This member function computes the schedule for the given BSP instance using the greedy scheduling algorithm.
     * It overrides the computeSchedule() function of the base Scheduler class.
     *
     * @param instance The BSP instance to compute the schedule for.
     * @return A pair containing the return status and the computed BSP schedule.
     */
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &bsp_schedule) override {

        // if constexpr (use_memory_constraint) {
        //     memory_constraint.initialize(instance);
        // }

        const auto &instance = bsp_schedule.getInstance();

        CSchedule<Graph_t> schedule(instance.numberOfVertices());

        std::set<vertex_idx_t<Graph_t>> ready;

        std::vector<unsigned> nrPredecDone(instance.numberOfVertices(), 0);

        std::vector<bool> procFree(instance.numberOfProcessors(), true);

        unsigned nrProcFree = instance.numberOfProcessors();

        std::vector<std::deque<vertex_idx_t<Graph_t>>> procQueue(instance.numberOfProcessors());
        std::vector<std::deque<vertex_idx_t<Graph_t>>> greedyProcLists(instance.numberOfProcessors());

        std::set<tv_pair> finishTimes;
        const tv_pair start(0, std::numeric_limits<vertex_idx_t<Graph_t>>::max());

        finishTimes.insert(start);

        for (const auto &v : source_vertices_view(instance.getComputationalDag())) {
            ready.insert(v);
            if (mode == CILK)
                procQueue[0].push_front(v);
        }

        while (!finishTimes.empty()) {
            const v_workw_t<Graph_t> time = finishTimes.begin()->first;

            // Find new ready jobs
            while (!finishTimes.empty() && finishTimes.begin()->first == time) {
                const tv_pair &currentPair = *finishTimes.begin();
                finishTimes.erase(finishTimes.begin());
                const vertex_idx_t<Graph_t> &node = currentPair.second;
                if (node != std::numeric_limits<vertex_idx_t<Graph_t>>::max()) {

                    for (const auto &succ : instance.getComputationalDag().children(node)) {

                        ++nrPredecDone[succ];
                        if (nrPredecDone[succ] == instance.getComputationalDag().in_degree(succ)) {

                            ready.insert(succ);
                            if (mode == CILK)
                                procQueue[schedule.proc[node]].push_back(succ);
                        }
                    }
                    procFree[schedule.proc[node]] = true;
                    ++nrProcFree;
                }
            }

            // Assign new jobs to processors
            while (nrProcFree > 0 && !ready.empty()) {

                unsigned nextProc = instance.numberOfProcessors();
                vertex_idx_t<Graph_t> nextNode = std::numeric_limits<vertex_idx_t<Graph_t>>::max();

                Choose(instance, procQueue, ready, procFree, nextNode, nextProc);

                ready.erase(nextNode);
                schedule.proc[nextNode] = nextProc;
                schedule.time[nextNode] = time;

                // if constexpr (use_memory_constraint) {
                //     memory_constraint.add(nextNode, nextProc);
                // }

                finishTimes.insert({time + instance.getComputationalDag().vertex_work_weight(nextNode), nextNode});
                procFree[nextProc] = false;

                if (nrProcFree > 0)
                    --nrProcFree;

                greedyProcLists[nextProc].push_back(nextNode);
            }
        }

        schedule.convertToBspSchedule(instance, greedyProcLists, bsp_schedule);

        return RETURN_STATUS::OSP_SUCCESS;
    }

    /**
     * @brief Sets the Cilk mode for the scheduler.
     *
     * This member function sets the Cilk mode for the scheduler.
     *
     * @param mode_ The Cilk mode to set.
     */
    inline void setMode(CilkMode mode_) { mode = mode_; }

    /**
     * @brief Gets the Cilk mode of the scheduler.
     *
     * This member function gets the Cilk mode of the scheduler.
     *
     * @return The Cilk mode of the scheduler.
     */
    inline CilkMode getMode() const { return mode; }

    /**
     * @brief Gets the name of the schedule.
     *
     * This member function gets the name of the schedule based on the Cilk mode.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override {

        switch (mode) {
        case CILK:
            return "CilkGreedy";
            break;

        case SJF:
            return "SJFGreedy";

        default:
            return "UnknownModeGreedy";
        }
    }
};

} // namespace osp