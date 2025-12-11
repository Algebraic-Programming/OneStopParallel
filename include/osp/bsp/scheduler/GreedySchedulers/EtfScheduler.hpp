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

#include <chrono>
#include <climits>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ClassicSchedule.hpp"
#include "MemoryConstraintModules.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

enum EtfMode { ETF, BL_EST };

/**
 * @brief The EtfScheduler class is a subclass of the Scheduler class and implements a greedy scheduling
 * algorithm.
 *
 * This class provides methods to compute a schedule using the greedy ETF (Earliest Task First) algorithm.
 * It calculates the bottom level of each task and uses it to determine the earliest start time (EST) for each task on
 * each processor. The algorithm selects the task with the earliest EST and assigns it to the processor with the
 * earliest available start time. The process is repeated until all tasks are scheduled.
 */
template <typename Graph_t, typename MemoryConstraint_t = no_memory_constraint>
class EtfScheduler : public Scheduler<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>, "EtfScheduler can only be used with computational DAGs.");

    static_assert(std::is_convertible_v<v_commw_t<Graph_t>, v_workw_t<Graph_t>>,
                  "EtfScheduler requires that work and communication weights are convertible.");

    static_assert(not has_edge_weights_v<Graph_t> || std::is_convertible_v<e_commw_t<Graph_t>, v_workw_t<Graph_t>>,
                  "EtfScheduler requires that work and communication weights are convertible.");

  private:
    using tv_pair = std::pair<v_workw_t<Graph_t>, vertex_idx_t<Graph_t>>;

    EtfMode mode;     // The mode of the scheduler (ETF or BL_EST)
    bool use_numa;    // Flag indicating whether to use NUMA-aware scheduling

    constexpr static bool use_memory_constraint = is_memory_constraint_v<MemoryConstraint_t>;

    static_assert(not use_memory_constraint || std::is_same_v<MemoryConstraint_t, persistent_transient_memory_constraint<Graph_t>>,
                  "EtfScheduler implements only persistent_transient_memory_constraint.");

    MemoryConstraint_t memory_constraint;

    /**
     * @brief Computes the bottom level of each task.
     *
     * @param instance The BspInstance object representing the BSP instance.
     * @param avg_ The average execution time of the tasks.
     * @return A vector containing the bottom level of each task.
     */
    std::vector<v_workw_t<Graph_t>> ComputeBottomLevel(const BspInstance<Graph_t> &instance) const {
        std::vector<v_workw_t<Graph_t>> BL(instance.numberOfVertices(), 0);

        const std::vector<vertex_idx_t<Graph_t>> topOrder = GetTopOrder(instance.getComputationalDag());
        auto r_iter = topOrder.rbegin();

        for (; r_iter != topOrder.rend(); ++r_iter) {
            const auto node = *r_iter;

            v_workw_t<Graph_t> maxval = 0;

            if constexpr (has_edge_weights_v<Graph_t>) {
                for (const auto &out_edge : out_edges(node, instance.getComputationalDag())) {
                    const v_workw_t<Graph_t> tmp_val = BL[target(out_edge, instance.getComputationalDag())]
                                                       + instance.getComputationalDag().edge_comm_weight(out_edge);

                    if (tmp_val > maxval) {
                        maxval = tmp_val;
                    }
                }

            } else {
                for (const auto &child : instance.getComputationalDag().children(node)) {
                    const v_workw_t<Graph_t> tmp_val = BL[child] + instance.getComputationalDag().vertex_comm_weight(child);

                    if (tmp_val > maxval) {
                        maxval = tmp_val;
                    }
                }
            }

            BL[node] = maxval + instance.getComputationalDag().vertex_work_weight(node);
        }
        return BL;
    }

    bool check_mem_feasibility(const BspInstance<Graph_t> &instance, const std::set<tv_pair> &ready) const {
        if (instance.getArchitecture().getMemoryConstraintType() == MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT) {
            if (ready.empty()) {
                return true;
            }

            for (const auto &node_pair : ready) {
                for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
                    const auto node = node_pair.second;

                    if constexpr (use_memory_constraint) {
                        if (memory_constraint.can_add(node, i)) {
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        return true;
    };

    /**
     * @brief Calculates the earliest start time (EST) for a task on a processor.
     *
     * @param instance The BspInstance object representing the BSP instance.
     * @param schedule The current schedule.
     * @param node The node (processor) on which the task is to be scheduled.
     * @param proc The processor index.
     * @param procAvailableFrom The earliest available start time for each processor.
     * @param send The send buffer sizes for each node.
     * @param rec The receive buffer sizes for each node.
     * @param avg_ The average execution time of the tasks.
     * @return The earliest start time (EST) for the task on the processor.
     */
    v_workw_t<Graph_t> GetESTforProc(const BspInstance<Graph_t> &instance,
                                     CSchedule<Graph_t> &schedule,
                                     vertex_idx_t<Graph_t> node,
                                     unsigned proc,
                                     const v_workw_t<Graph_t> procAvailableFrom,
                                     std::vector<v_workw_t<Graph_t>> &send,
                                     std::vector<v_workw_t<Graph_t>> &rec) const {
        std::vector<tv_pair> predec;
        for (const auto &pred : instance.getComputationalDag().parents(node)) {
            predec.emplace_back(schedule.time[pred] + instance.getComputationalDag().vertex_work_weight(pred), pred);
        }

        std::sort(predec.begin(), predec.end());

        v_workw_t<Graph_t> EST = procAvailableFrom;
        for (const auto &next : predec) {
            v_workw_t<Graph_t> t = schedule.time[next.second] + instance.getComputationalDag().vertex_work_weight(next.second);
            if (schedule.proc[next.second] != proc) {
                t = std::max(t, send[schedule.proc[next.second]]);
                t = std::max(t, rec[proc]);

                if constexpr (has_edge_weights_v<Graph_t>) {
                    t += instance.getComputationalDag().edge_comm_weight(
                             edge_desc(next.second, node, instance.getComputationalDag()).first)
                         * instance.sendCosts(schedule.proc[next.second], proc);

                } else {
                    t += instance.getComputationalDag().vertex_comm_weight(next.second)
                         * instance.sendCosts(schedule.proc[next.second], proc);
                }

                send[schedule.proc[next.second]] = t;
                rec[proc] = t;
            }
            EST = std::max(EST, t);
        }
        return EST;
    };

    /**
     * @brief Finds the best EST for a set of nodes.
     *
     * @param instance The BspInstance object representing the BSP instance.
     * @param schedule The current schedule.
     * @param nodeList The list of nodes to consider.
     * @param procAvailableFrom The earliest available start time for each processor.
     * @param send The send buffer sizes for each node.
     * @param rec The receive buffer sizes for each node.
     * @param avg_ The average execution time of the tasks.
     * @return A triple containing the best EST, the node index, and the processor index.
     */
    tv_pair GetBestESTforNodes(const BspInstance<Graph_t> &instance,
                               CSchedule<Graph_t> &schedule,
                               const std::vector<vertex_idx_t<Graph_t>> &nodeList,
                               const std::vector<v_workw_t<Graph_t>> &procAvailableFrom,
                               std::vector<v_workw_t<Graph_t>> &send,
                               std::vector<v_workw_t<Graph_t>> &rec,
                               unsigned &bestProc) const {
        v_workw_t<Graph_t> bestEST = std::numeric_limits<v_workw_t<Graph_t>>::max();
        vertex_idx_t<Graph_t> bestNode = 0;
        std::vector<v_workw_t<Graph_t>> bestSend, bestRec;
        for (const auto &node : nodeList) {
            for (unsigned j = 0; j < instance.numberOfProcessors(); ++j) {
                if constexpr (use_memory_constraint) {
                    if (not memory_constraint.can_add(node, j)) {
                        continue;
                    }
                }

                std::vector<v_workw_t<Graph_t>> newSend = send;
                std::vector<v_workw_t<Graph_t>> newRec = rec;
                v_workw_t<Graph_t> EST = GetESTforProc(instance, schedule, node, j, procAvailableFrom[j], newSend, newRec);
                if (EST < bestEST) {
                    bestEST = EST;
                    bestProc = j;
                    bestNode = node;
                    bestSend = newSend;
                    bestRec = newRec;
                }
            }
        }

        send = bestSend;
        rec = bestRec;

        return {bestEST, bestNode};
    };

  public:
    /**
     * @brief Constructs a EtfScheduler object with the specified mode.
     *
     * @param mode_ The mode of the scheduler (ETF or BL_EST).
     */
    EtfScheduler(EtfMode mode_ = ETF) : Scheduler<Graph_t>(), mode(mode_), use_numa(true) {}

    /**
     * @brief Default destructor for the EtfScheduler class.
     */
    virtual ~EtfScheduler() = default;

    /**
     * @brief Computes a schedule for the given BSP instance using the greedy ETF algorithm.
     *
     * @param instance The BspInstance object representing the BSP instance.
     * @return A pair containing the return status and the computed BspSchedule object.
     */
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &bsp_schedule) override {
        const auto &instance = bsp_schedule.getInstance();

        if constexpr (use_memory_constraint) {
            memory_constraint.initialize(instance);
        }

        CSchedule<Graph_t> schedule(instance.numberOfVertices());

        std::vector<std::deque<vertex_idx_t<Graph_t>>> greedyProcLists(instance.numberOfProcessors());

        std::vector<vertex_idx_t<Graph_t>> predecProcessed(instance.numberOfVertices(), 0);

        std::vector<v_workw_t<Graph_t>> finishTimes(instance.numberOfProcessors(), 0), send(instance.numberOfProcessors(), 0),
            rec(instance.numberOfProcessors(), 0);

        std::vector<v_workw_t<Graph_t>> BL;
        if (mode == BL_EST) {
            BL = ComputeBottomLevel(instance);
        } else {
            BL = std::vector<v_workw_t<Graph_t>>(instance.numberOfVertices(), 0);
        }

        std::set<tv_pair> ready;

        for (const auto &v : source_vertices_view(instance.getComputationalDag())) {
            ready.insert({BL[v], v});
        }

        while (!ready.empty()) {
            tv_pair best_tv(0, 0);
            unsigned best_proc = 0;

            if (mode == BL_EST) {
                std::vector<vertex_idx_t<Graph_t>> nodeList{ready.begin()->second};
                ready.erase(ready.begin());
                best_tv = GetBestESTforNodes(instance, schedule, nodeList, finishTimes, send, rec, best_proc);
            }

            if (mode == ETF) {
                std::vector<vertex_idx_t<Graph_t>> nodeList;
                for (const auto &next : ready) {
                    nodeList.push_back(next.second);
                }
                best_tv = GetBestESTforNodes(instance, schedule, nodeList, finishTimes, send, rec, best_proc);
                ready.erase(tv_pair({0, best_tv.second}));
            }
            const auto node = best_tv.second;

            schedule.proc[node] = best_proc;
            greedyProcLists[best_proc].push_back(node);

            schedule.time[node] = best_tv.first;
            finishTimes[best_proc] = schedule.time[node] + instance.getComputationalDag().vertex_work_weight(node);

            if constexpr (use_memory_constraint) {
                memory_constraint.add(node, best_proc);
            }

            for (const auto &succ : instance.getComputationalDag().children(node)) {
                ++predecProcessed[succ];
                if (predecProcessed[succ] == instance.getComputationalDag().in_degree(succ)) {
                    ready.insert({BL[succ], succ});
                }
            }

            if constexpr (use_memory_constraint) {
                if (not check_mem_feasibility(instance, ready)) {
                    return RETURN_STATUS::ERROR;
                }
            }
        }

        schedule.convertToBspSchedule(instance, greedyProcLists, bsp_schedule);

        return RETURN_STATUS::OSP_SUCCESS;
    }

    /**
     * @brief Sets the mode of the scheduler.
     *
     * @param mode_ The mode of the scheduler (ETF or BL_EST).
     */
    inline void setMode(EtfMode mode_) { mode = mode_; }

    /**
     * @brief Gets the mode of the scheduler.
     *
     * @return The mode of the scheduler (ETF or BL_EST).
     */
    inline EtfMode getMode() const { return mode; }

    /**
     * @brief Sets whether to use NUMA-aware scheduling.
     *
     * @param numa Flag indicating whether to use NUMA-aware scheduling.
     */
    inline void setUseNuma(bool numa) { use_numa = numa; }

    /**
     * @brief Checks if NUMA-aware scheduling is enabled.
     *
     * @return True if NUMA-aware scheduling is enabled, false otherwise.
     */
    inline bool useNuma() const { return use_numa; }

    /**
     * @brief Gets the name of the schedule.
     *
     * @return The name of the schedule based on the mode.
     */
    virtual std::string getScheduleName() const override {
        switch (mode) {
            case ETF:
                return "ETFGreedy";

            case BL_EST:
                return "BL-ESTGreedy";

            default:
                return "UnknownModeGreedy";
        }
    }
};

}    // namespace osp
