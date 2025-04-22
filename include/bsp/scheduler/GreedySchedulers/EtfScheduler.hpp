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
#include "auxiliary/misc.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/directed_graph_top_sort.hpp"
#include "graph_algorithms/directed_graph_edge_desc_util.hpp"

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
template<typename Graph_t>
class EtfScheduler : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "EtfScheduler can only be used with computational DAGs.");

    static_assert(std::is_convertible_v<v_commw_t<Graph_t>, v_workw_t<Graph_t>>,
                  "EtfScheduler requires that work and communication weights are convertible.");

    static_assert(not is_computational_dag_edge_desc_v<Graph_t> ||
                      std::is_convertible_v<e_commw_t<Graph_t>, v_workw_t<Graph_t>>,
                  "EtfScheduler requires that work and communication weights are convertible.");

  private:
    using tv_pair = std::pair<v_workw_t<Graph_t>, vertex_idx_t<Graph_t>>;

    EtfMode mode;  // The mode of the scheduler (ETF or BL_EST)
    bool use_numa; // Flag indicating whether to use NUMA-aware scheduling

    bool use_memory_constraint = false;

    std::vector<v_memw_t<Graph_t>> current_proc_persistent_memory;
    std::vector<v_commw_t<Graph_t>> current_proc_transient_memory;

    /**
     * @brief Computes the bottom level of each task.
     *
     * @param instance The BspInstance object representing the BSP instance.
     * @param avg_ The average execution time of the tasks.
     * @return A vector containing the bottom level of each task.
     */
    std::vector<v_workw_t<Graph_t>> ComputeBottomLevel(const BspInstance<Graph_t> &instance) const {

        std::vector<v_workw_t<Graph_t>> BL(instance.numberOfVertices(), 0);

        const std::vector<vertex_idx_t<Graph_t>> topOrder = GetTopOrder(AS_IT_COMES, instance.getComputationalDag());
        auto r_iter = topOrder.rbegin();

        for (; r_iter != topOrder.rend(); ++r_iter) {
            const auto node = *r_iter;

            v_workw_t<Graph_t> maxval = 0;

            if constexpr (is_computational_dag_edge_desc_v<Graph_t>) {

                for (const auto &out_edge : instance.getComputationalDag().out_edges(node)) {

                    const v_workw_t<Graph_t> tmp_val = BL[target(out_edge, instance.getComputationalDag())] +
                                                       instance.getComputationalDag().edge_comm_weight(out_edge);

                    if (tmp_val > maxval) {
                        maxval = tmp_val;
                    }
                }

            } else {

                for (const auto &child : instance.getComputationalDag().children(node)) {

                    const v_workw_t<Graph_t> tmp_val =
                        BL[child] + instance.getComputationalDag().vertex_comm_weight(child);

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

        if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

            if (ready.empty()) {
                return true;
            }

            for (const auto &node_pair : ready) {
                for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {

                    const auto node = node_pair.second;

                    if (current_proc_persistent_memory[i] + instance.getComputationalDag().vertex_mem_weight(node) +
                            std::max(current_proc_transient_memory[i],
                                     instance.getComputationalDag().vertex_comm_weight(node)) <=
                        instance.getArchitecture().memoryBound(i)) {
                        return true;
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
    v_workw_t<Graph_t> GetESTforProc(const BspInstance<Graph_t> &instance, CSchedule<Graph_t> &schedule,
                                     vertex_idx_t<Graph_t> node, unsigned proc,
                                     const v_workw_t<Graph_t> procAvailableFrom, std::vector<v_workw_t<Graph_t>> &send,
                                     std::vector<v_workw_t<Graph_t>> &rec) const {

        std::vector<tv_pair> predec;
        for (const auto &pred : instance.getComputationalDag().parents(node)) {
            predec.emplace_back(schedule.time[pred] + instance.getComputationalDag().vertex_work_weight(pred), pred);
        }

        std::sort(predec.begin(), predec.end());

        v_workw_t<Graph_t> EST = procAvailableFrom;
        for (const auto &next : predec) {
            v_workw_t<Graph_t> t =
                schedule.time[next.second] + instance.getComputationalDag().vertex_work_weight(next.second);
            if (schedule.proc[next.second] != proc) {
                t = std::max(t, send[schedule.proc[next.second]]);
                t = std::max(t, rec[proc]);

                if constexpr (is_computational_dag_edge_desc_v<Graph_t>) {

                    t += instance.getComputationalDag().edge_comm_weight(
                             edge_desc(next.second, node, instance.getComputationalDag()).first) *
                         instance.sendCosts(schedule.proc[next.second], proc);

                } else {

                    t += instance.getComputationalDag().vertex_comm_weight(next.second) *
                         instance.sendCosts(schedule.proc[next.second], proc);
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
    tv_pair GetBestESTforNodes(const BspInstance<Graph_t> &instance, CSchedule<Graph_t> &schedule,
                               const std::vector<vertex_idx_t<Graph_t>> &nodeList,
                               const std::vector<v_workw_t<Graph_t>> &procAvailableFrom,
                               std::vector<v_workw_t<Graph_t>> &send, std::vector<v_workw_t<Graph_t>> &rec,
                               unsigned &bestProc) const {

        v_workw_t<Graph_t> bestEST = std::numeric_limits<v_workw_t<Graph_t>>::max();
        vertex_idx_t<Graph_t> bestNode = 0;
        std::vector<v_workw_t<Graph_t>> bestSend, bestRec;
        for (const auto &node : nodeList)
            for (unsigned j = 0; j < instance.numberOfProcessors(); ++j) {

                if (use_memory_constraint) {

                    if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                        if (current_proc_persistent_memory[j] + instance.getComputationalDag().vertex_mem_weight(node) >
                            instance.getArchitecture().memoryBound(j)) {
                            continue;
                        }

                    } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                        if (current_proc_persistent_memory[j] + instance.getComputationalDag().vertex_mem_weight(node) +
                                std::max(current_proc_transient_memory[j],
                                         instance.getComputationalDag().vertex_comm_weight(node)) >
                            instance.getArchitecture().memoryBound(j)) {
                            continue;
                        }
                    }
                }

                std::vector<v_workw_t<Graph_t>> newSend = send;
                std::vector<v_workw_t<Graph_t>> newRec = rec;
                v_workw_t<Graph_t> EST =
                    GetESTforProc(instance, schedule, node, j, procAvailableFrom[j], newSend, newRec);
                if (EST < bestEST) {
                    bestEST = EST;
                    bestProc = j;
                    bestNode = node;
                    bestSend = newSend;
                    bestRec = newRec;
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
    EtfScheduler(EtfMode mode_ = ETF) : Scheduler<Graph_t>(), mode(mode_), use_numa(false) {}

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
    virtual std::pair<RETURN_STATUS, BspSchedule<Graph_t>>
    computeSchedule(const BspInstance<Graph_t> &instance) override {

        if (use_memory_constraint) {

            switch (instance.getArchitecture().getMemoryConstraintType()) {

            case LOCAL:
                throw std::invalid_argument("Local memory constraint not supported");

            case PERSISTENT_AND_TRANSIENT:
                current_proc_persistent_memory = std::vector<v_memw_t<Graph_t>>(instance.numberOfProcessors(), 0);
                current_proc_transient_memory = std::vector<v_commw_t<Graph_t>>(instance.numberOfProcessors(), 0);
                break;

            case GLOBAL:
                throw std::invalid_argument("Global memory constraint not supported");

            case NONE:
                use_memory_constraint = false;
                std::cerr << "Warning: Memory constraint type set to NONE, ignoring memory constraint" << std::endl;
                break;

            default:
                break;
            }
        }

        CSchedule<Graph_t> schedule(instance.numberOfVertices());

        std::vector<std::deque<vertex_idx_t<Graph_t>>> greedyProcLists(instance.numberOfProcessors());

        std::vector<vertex_idx_t<Graph_t>> predecProcessed(instance.numberOfVertices(), 0);

        std::vector<v_workw_t<Graph_t>> finishTimes(instance.numberOfProcessors(), 0),
            send(instance.numberOfProcessors(), 0), rec(instance.numberOfProcessors(), 0);

        std::vector<v_workw_t<Graph_t>> BL;
        if (mode == BL_EST)
            BL = ComputeBottomLevel(instance);
        else
            BL = std::vector<v_workw_t<Graph_t>>(instance.numberOfVertices(), 0);

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
                for (const auto &next : ready)
                    nodeList.push_back(next.second);
                best_tv = GetBestESTforNodes(instance, schedule, nodeList, finishTimes, send, rec, best_proc);
                ready.erase(tv_pair({0, best_tv.second}));
            }
            const auto node = best_tv.second;

            schedule.proc[node] = best_proc;
            greedyProcLists[best_proc].push_back(node);

            schedule.time[node] = best_tv.first;
            finishTimes[best_proc] = schedule.time[node] + instance.getComputationalDag().vertex_work_weight(node);

            if (use_memory_constraint) {

                if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                    current_proc_persistent_memory[best_proc] += instance.getComputationalDag().vertex_mem_weight(node);
                    current_proc_transient_memory[best_proc] =
                        std::max(current_proc_transient_memory[best_proc],
                                 instance.getComputationalDag().vertex_mem_weight(node));
                }
            }

            for (const auto &succ : instance.getComputationalDag().children(node)) {
                ++predecProcessed[succ];
                if (predecProcessed[succ] == instance.getComputationalDag().in_degree(succ))
                    ready.insert({BL[succ], succ});
            }

            if (use_memory_constraint && not check_mem_feasibility(instance, ready)) {

                return {ERROR, schedule.convertToBspSchedule(instance, greedyProcLists)};
            }
        }

        return {SUCCESS, schedule.convertToBspSchedule(instance, greedyProcLists)};
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

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override {
        use_memory_constraint = use_memory_constraint_;
    }

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

} // namespace osp