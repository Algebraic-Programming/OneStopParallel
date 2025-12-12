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
template <typename GraphT, typename MemoryConstraintT = NoMemoryConstraint>
class EtfScheduler : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<GraphT>, "EtfScheduler can only be used with computational DAGs.");

    static_assert(std::is_convertible_v<VCommwT<GraphT>, VWorkwT<GraphT>>,
                  "EtfScheduler requires that work and communication weights are convertible.");

    static_assert(not HasEdgeWeightsV<GraphT> || std::is_convertible_v<ECommwT<GraphT>, VWorkwT<GraphT>>,
                  "EtfScheduler requires that work and communication weights are convertible.");

  private:
    using tv_pair = std::pair<VWorkwT<GraphT>, VertexIdxT<GraphT>>;

    EtfMode mode_;    // The mode of the scheduler (ETF or BL_EST)
    bool useNuma_;    // Flag indicating whether to use NUMA-aware scheduling

    constexpr static bool useMemoryConstraint_ = IsMemoryConstraintV<MemoryConstraintT>;

    static_assert(not useMemoryConstraint_ || std::is_same_v<MemoryConstraintT, PersistentTransientMemoryConstraint<GraphT>>,
                  "EtfScheduler implements only persistent_transient_memory_constraint.");

    MemoryConstraintT memoryConstraint_;

    /**
     * @brief Computes the bottom level of each task.
     *
     * @param instance The BspInstance object representing the BSP instance.
     * @param avg_ The average execution time of the tasks.
     * @return A vector containing the bottom level of each task.
     */
    std::vector<VWorkwT<GraphT>> ComputeBottomLevel(const BspInstance<GraphT> &instance) const {
        std::vector<VWorkwT<GraphT>> bl(instance.NumberOfVertices(), 0);

        const std::vector<VertexIdxT<GraphT>> topOrder = GetTopOrder(instance.GetComputationalDag());
        auto rIter = topOrder.rbegin();

        for (; rIter != topOrder.rend(); ++rIter) {
            const auto node = *rIter;

            VWorkwT<GraphT> maxval = 0;

            if constexpr (HasEdgeWeightsV<GraphT>) {
                for (const auto &outEdge : OutEdges(node, instance.GetComputationalDag())) {
                    const VWorkwT<GraphT> tmpVal = bl[Traget(outEdge, instance.GetComputationalDag())]
                                                   + instance.GetComputationalDag().EdgeCommWeight(outEdge);

                    if (tmpVal > maxval) {
                        maxval = tmpVal;
                    }
                }

            } else {
                for (const auto &child : instance.GetComputationalDag().Children(node)) {
                    const VWorkwT<GraphT> tmpVal = bl[child] + instance.GetComputationalDag().VertexCommWeight(child);

                    if (tmpVal > maxval) {
                        maxval = tmpVal;
                    }
                }
            }

            bl[node] = maxval + instance.GetComputationalDag().VertexWorkWeight(node);
        }
        return bl;
    }

    bool CheckMemFeasibility(const BspInstance<GraphT> &instance, const std::set<tv_pair> &ready) const {
        if (instance.GetArchitecture().GetMemoryConstraintType() == MemoryConstraintType::PERSISTENT_AND_TRANSIENT) {
            if (ready.empty()) {
                return true;
            }

            for (const auto &nodePair : ready) {
                for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                    const auto node = nodePair.second;

                    if constexpr (useMemoryConstraint_) {
                        if (memoryConstraint_.CanAdd(node, i)) {
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
    VWorkwT<GraphT> GetESTforProc(const BspInstance<GraphT> &instance,
                                  CSchedule<GraphT> &schedule,
                                  VertexIdxT<GraphT> node,
                                  unsigned proc,
                                  const VWorkwT<GraphT> procAvailableFrom,
                                  std::vector<VWorkwT<GraphT>> &send,
                                  std::vector<VWorkwT<GraphT>> &rec) const {
        std::vector<tv_pair> predec;
        for (const auto &pred : instance.GetComputationalDag().Parents(node)) {
            predec.emplace_back(schedule.time[pred] + instance.GetComputationalDag().VertexWorkWeight(pred), pred);
        }

        std::sort(predec.begin(), predec.end());

        VWorkwT<GraphT> est = procAvailableFrom;
        for (const auto &next : predec) {
            VWorkwT<GraphT> t = schedule.time[next.second] + instance.GetComputationalDag().VertexWorkWeight(next.second);
            if (schedule.proc[next.second] != proc) {
                t = std::max(t, send[schedule.proc[next.second]]);
                t = std::max(t, rec[proc]);

                if constexpr (HasEdgeWeightsV<GraphT>) {
                    t += instance.GetComputationalDag().EdgeCommWeight(
                             EdgeDesc(next.second, node, instance.GetComputationalDag()).first)
                         * instance.SendCosts(schedule.proc[next.second], proc);

                } else {
                    t += instance.GetComputationalDag().VertexCommWeight(next.second)
                         * instance.SendCosts(schedule.proc[next.second], proc);
                }

                send[schedule.proc[next.second]] = t;
                rec[proc] = t;
            }
            est = std::max(est, t);
        }
        return est;
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
    tv_pair GetBestESTforNodes(const BspInstance<GraphT> &instance,
                               CSchedule<GraphT> &schedule,
                               const std::vector<VertexIdxT<GraphT>> &nodeList,
                               const std::vector<VWorkwT<GraphT>> &procAvailableFrom,
                               std::vector<VWorkwT<GraphT>> &send,
                               std::vector<VWorkwT<GraphT>> &rec,
                               unsigned &bestProc) const {
        VWorkwT<GraphT> bestEST = std::numeric_limits<VWorkwT<GraphT>>::max();
        VertexIdxT<GraphT> bestNode = 0;
        std::vector<VWorkwT<GraphT>> bestSend, bestRec;
        for (const auto &node : nodeList) {
            for (unsigned j = 0; j < instance.NumberOfProcessors(); ++j) {
                if constexpr (useMemoryConstraint_) {
                    if (not memoryConstraint_.CanAdd(node, j)) {
                        continue;
                    }
                }

                std::vector<VWorkwT<GraphT>> newSend = send;
                std::vector<VWorkwT<GraphT>> newRec = rec;
                VWorkwT<GraphT> est = GetESTforProc(instance, schedule, node, j, procAvailableFrom[j], newSend, newRec);
                if (est < bestEST) {
                    bestEST = est;
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
    EtfScheduler(EtfMode mode = ETF) : Scheduler<GraphT>(), mode_(mode), useNuma_(true) {}

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
    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &bspSchedule) override {
        const auto &instance = bspSchedule.GetInstance();

        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.Initialize(instance);
        }

        CSchedule<GraphT> schedule(instance.NumberOfVertices());

        std::vector<std::deque<VertexIdxT<GraphT>>> greedyProcLists(instance.NumberOfProcessors());

        std::vector<VertexIdxT<GraphT>> predecProcessed(instance.NumberOfVertices(), 0);

        std::vector<VWorkwT<GraphT>> finishTimes(instance.NumberOfProcessors(), 0), send(instance.NumberOfProcessors(), 0),
            rec(instance.NumberOfProcessors(), 0);

        std::vector<VWorkwT<GraphT>> bl;
        if (mode_ == BL_EST) {
            bl = ComputeBottomLevel(instance);
        } else {
            bl = std::vector<VWorkwT<GraphT>>(instance.NumberOfVertices(), 0);
        }

        std::set<tv_pair> ready;

        for (const auto &v : SourceVerticesView(instance.GetComputationalDag())) {
            ready.insert({bl[v], v});
        }

        while (!ready.empty()) {
            tv_pair bestTv(0, 0);
            unsigned bestProc = 0;

            if (mode_ == BL_EST) {
                std::vector<VertexIdxT<GraphT>> nodeList{ready.begin()->second};
                ready.erase(ready.begin());
                bestTv = GetBestESTforNodes(instance, schedule, nodeList, finishTimes, send, rec, bestProc);
            }

            if (mode_ == ETF) {
                std::vector<VertexIdxT<GraphT>> nodeList;
                for (const auto &next : ready) {
                    nodeList.push_back(next.second);
                }
                bestTv = GetBestESTforNodes(instance, schedule, nodeList, finishTimes, send, rec, bestProc);
                ready.erase(tv_pair({0, bestTv.second}));
            }
            const auto node = bestTv.second;

            schedule.proc[node] = bestProc;
            greedyProcLists[bestProc].push_back(node);

            schedule.time[node] = bestTv.first;
            finishTimes[bestProc] = schedule.time[node] + instance.GetComputationalDag().VertexWorkWeight(node);

            if constexpr (useMemoryConstraint_) {
                memoryConstraint_.Add(node, bestProc);
            }

            for (const auto &succ : instance.GetComputationalDag().Children(node)) {
                ++predecProcessed[succ];
                if (predecProcessed[succ] == instance.GetComputationalDag().InDegree(succ)) {
                    ready.insert({bl[succ], succ});
                }
            }

            if constexpr (useMemoryConstraint_) {
                if (not CheckMemoryFeasibility(instance, ready)) {
                    return ReturnStatus::ERROR;
                }
            }
        }

        schedule.ConvertToBspSchedule(instance, greedyProcLists, bspSchedule);

        return ReturnStatus::OSP_SUCCESS;
    }

    /**
     * @brief Sets the mode of the scheduler.
     *
     * @param mode_ The mode of the scheduler (ETF or BL_EST).
     */
    inline void SetMode(EtfMode mode) { mode_ = mode; }

    /**
     * @brief Gets the mode of the scheduler.
     *
     * @return The mode of the scheduler (ETF or BL_EST).
     */
    inline EtfMode GetMode() const { return mode_; }

    /**
     * @brief Sets whether to use NUMA-aware scheduling.
     *
     * @param numa Flag indicating whether to use NUMA-aware scheduling.
     */
    inline void SetUseNuma(bool numa) { useNuma_ = numa; }

    /**
     * @brief Checks if NUMA-aware scheduling is enabled.
     *
     * @return True if NUMA-aware scheduling is enabled, false otherwise.
     */
    inline bool UseNuma() const { return useNuma_; }

    /**
     * @brief Gets the name of the schedule.
     *
     * @return The name of the schedule based on the mode.
     */
    virtual std::string GetScheduleName() const override {
        switch (mode_) {
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
