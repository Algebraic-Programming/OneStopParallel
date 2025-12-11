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
#include <boost/heap/fibonacci_heap.hpp>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "kl_current_schedule.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

// #define KL_PRINT_SCHEDULE

#ifdef KL_PRINT_SCHEDULE
#    include "file_interactions/DotFileWriter.hpp"
#endif

namespace osp {

struct KlBaseParameter {
    double maxDivBestSolBasePercent = 1.05;
    double maxDivBestSolRatePercent = 0.002;

    unsigned maxNumUnlocks = 1;
    unsigned maxNumFailedBranches = 5;

    unsigned maxInnerIterations = 150;
    unsigned maxOuterIterations = 100;

    unsigned maxNoImprovementIterations = 75;

    std::size_t selectionThreshold;
    bool selectAllNodes = false;

    double initialPenalty = 0.0;

    double gainThreshold = -10.0;
    double changeInCostThreshold = 0.0;

    bool quickPass = false;

    unsigned maxStepSelectionEpochs = 4;
    unsigned resetEpochCounterThreshold = 10;

    unsigned violationsThreshold = 0;
};

template <typename GraphT, typename MemoryConstraintT>
class KlBase : public ImprovementScheduler<GraphT>, public IklCostFunction {
    static_assert(isDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(hasHashableEdgeDescV<GraphT>, "Graph_t must satisfy the has_hashable_edge_desc concept");
    static_assert(isComputationalDagV<GraphT>, "Graph_t must satisfy the computational_dag concept");

  private:
    using MemwT = VMemwT<GraphT>;
    using CommwT = VCommwT<GraphT>;
    using WorkwT = VWorkwT<GraphT>;

  protected:
    using VertexType = VertexIdxT<GraphT>;

    KlBaseParameter parameters_;

    std::mt19937 gen_;

    VertexType numNodes_;
    unsigned numProcs_;

    double penalty_ = 0.0;
    double reward_ = 0.0;

    virtual void UpdateRewardPenalty() = 0;
    virtual void SetInitialRewardPenalty() = 0;

    boost::heap::fibonacci_heap<KlMove<GraphT>> maxGainHeap_;
    using HeapHandle = typename boost::heap::fibonacci_heap<KlMove<GraphT>>::handle_type;

    std::unordered_map<VertexType, HeapHandle> nodeHeapHandles_;

    std::vector<std::vector<std::vector<double>>> nodeGains_;
    std::vector<std::vector<std::vector<double>>> nodeChangeInCosts_;

    KlCurrentSchedule<GraphT, MemoryConstraintT> &currentSchedule_;

    BspSchedule<GraphT> *bestSchedule_;
    double bestScheduleCosts_;

    std::unordered_set<VertexType> lockedNodes_;
    std::unordered_set<VertexType> superLockedNodes_;
    std::vector<unsigned> unlock_;

    bool UnlockNode(VertexType node) {
        if (superLockedNodes_.find(node) == superLockedNodes_.end()) {
            if (lockedNodes_.find(node) == lockedNodes_.end()) {
                return true;
            } else if (lockedNodes_.find(node) != lockedNodes_.end() && unlock_[node] > 0) {
                unlock_[node]--;

                lockedNodes_.erase(node);

                return true;
            }
        }
        return false;
    }

    bool CheckNodeUnlocked(VertexType node) {
        if (superLockedNodes_.find(node) == superLockedNodes_.end() && lockedNodes_.find(node) == lockedNodes_.end()) {
            return true;
        }
        return false;
    };

    void ResetLockedNodes() {
        for (const auto &i : lockedNodes_) {
            unlock_[i] = parameters_.maxNumUnlocks;
        }

        lockedNodes_.clear();
    }

    bool CheckViolationLocked() {
        if (currentSchedule_.current_violations.empty()) {
            return false;
        }

        for (auto &edge : currentSchedule_.current_violations) {
            const auto &sourceV = source(edge, currentSchedule_.instance->getComputationalDag());
            const auto &targetV = target(edge, currentSchedule_.instance->getComputationalDag());

            if (lockedNodes_.find(sourceV) == lockedNodes_.end() || lockedNodes_.find(targetV) == lockedNodes_.end()) {
                return false;
            }

            bool abort = false;
            if (lockedNodes_.find(sourceV) != lockedNodes_.end()) {
                if (UnlockNode(sourceV)) {
                    nodesToUpdate_.insert(sourceV);
                    nodeSelection_.insert(sourceV);
                } else {
                    abort = true;
                }
            }

            if (lockedNodes_.find(targetV) != lockedNodes_.end()) {
                if (UnlockNode(targetV)) {
                    nodesToUpdate_.insert(targetV);
                    nodeSelection_.insert(targetV);
                    abort = false;
                }
            }

            if (abort) {
                return true;
            }
        }

        return false;
    }

    void ResetGainHeap() {
        maxGainHeap_.clear();
        nodeHeapHandles_.clear();
    }

    virtual void InitializeDatastructures() {
#ifdef KL_DEBUG
        std::cout << "KLBase initialize datastructures" << std::endl;
#endif

        nodeGains_ = std::vector<std::vector<std::vector<double>>>(
            numNodes_, std::vector<std::vector<double>>(numProcs_, std::vector<double>(3, 0)));

        nodeChangeInCosts_ = std::vector<std::vector<std::vector<double>>>(
            numNodes_, std::vector<std::vector<double>>(numProcs_, std::vector<double>(3, 0)));

        unlock_ = std::vector<unsigned>(numNodes_, parameters_.maxNumUnlocks);

        currentSchedule_.initialize_current_schedule(*bestSchedule_);
        bestScheduleCosts_ = currentSchedule_.current_cost;
    }

    std::unordered_set<VertexType> nodesToUpdate_;

    void ComputeNodesToUpdate(KlMove<GraphT> move) {
        nodesToUpdate_.clear();

        for (const auto &target : currentSchedule_.instance->getComputationalDag().Children(move.node)) {
            if (nodeSelection_.find(target) != nodeSelection_.end() && lockedNodes_.find(target) == lockedNodes_.end()
                && superLockedNodes_.find(target) == superLockedNodes_.end()) {
                nodesToUpdate_.insert(target);
            }
        }

        for (const auto &source : currentSchedule_.instance->getComputationalDag().Parents(move.node)) {
            if (nodeSelection_.find(source) != nodeSelection_.end() && lockedNodes_.find(source) == lockedNodes_.end()
                && superLockedNodes_.find(source) == superLockedNodes_.end()) {
                nodesToUpdate_.insert(source);
            }
        }

        const unsigned startStep = std::min(move.from_step, move.to_step) == 0 ? 0 : std::min(move.from_step, move.to_step) - 1;
        const unsigned endStep = std::min(currentSchedule_.num_steps(), std::max(move.from_step, move.to_step) + 2);

#ifdef KL_DEBUG
        std::cout << "updating from step " << start_step << " to step " << end_step << std::endl;
#endif

        for (unsigned step = startStep; step < endStep; step++) {
            for (unsigned proc = 0; proc < numProcs_; proc++) {
                for (const auto &node : currentSchedule_.set_schedule.step_processor_vertices[step][proc]) {
                    if (nodeSelection_.find(node) != nodeSelection_.end() && lockedNodes_.find(node) == lockedNodes_.end()
                        && superLockedNodes_.find(node) == superLockedNodes_.end()) {
                        nodesToUpdate_.insert(node);
                    }
                }
            }
        }
    }

    void InitializeGainHeap(const std::unordered_set<VertexType> &nodes) {
        ResetGainHeap();

        for (const auto &node : nodes) {
            ComputeNodeGain(node);
            ComputeMaxGainInsertOrUpdateHeap(node);
        }
    }

    void InitializeGainHeapUnlockedNodes(const std::unordered_set<VertexType> &nodes) {
        ResetGainHeap();

        for (const auto &node : nodes) {
            if (lockedNodes_.find(node) == lockedNodes_.end() && superLockedNodes_.find(node) == superLockedNodes_.end()) {
                ComputeNodeGain(node);
                ComputeMaxGainInsertOrUpdateHeap(node);
            }
        }
    }

    void ComputeNodeGain(VertexType node) {
        const unsigned &currentProc = currentSchedule_.vector_schedule.assignedProcessor(node);
        const unsigned &currentStep = currentSchedule_.vector_schedule.assignedSuperstep(node);

        for (unsigned newProc = 0; newProc < numProcs_; newProc++) {
            if (currentSchedule_.instance->isCompatible(node, newProc)) {
                nodeGains_[node][newProc][0] = 0.0;
                nodeGains_[node][newProc][1] = 0.0;
                nodeGains_[node][newProc][2] = 0.0;

                nodeChangeInCosts_[node][newProc][0] = 0;
                nodeChangeInCosts_[node][newProc][1] = 0;
                nodeChangeInCosts_[node][newProc][2] = 0;

                ComputeCommGain(node, currentStep, currentProc, newProc);
                ComputeWorkGain(node, currentStep, currentProc, newProc);

                if constexpr (currentSchedule_.use_memory_constraint) {
                    if (not currentSchedule_.memory_constraint.can_move(
                            node, newProc, currentSchedule_.vector_schedule.assignedSuperstep(node))) {
                        nodeGains_[node][newProc][1] = std::numeric_limits<double>::lowest();
                    }

                    if (currentSchedule_.vector_schedule.assignedSuperstep(node) > 0) {
                        if (not currentSchedule_.memory_constraint.can_move(
                                node, newProc, currentSchedule_.vector_schedule.assignedSuperstep(node) - 1)) {
                            nodeGains_[node][newProc][0] = std::numeric_limits<double>::lowest();
                        }
                    }
                    if (currentSchedule_.vector_schedule.assignedSuperstep(node) < currentSchedule_.num_steps() - 1) {
                        if (not currentSchedule_.memory_constraint.can_move(
                                node, newProc, currentSchedule_.vector_schedule.assignedSuperstep(node) + 1)) {
                            nodeGains_[node][newProc][2] = std::numeric_limits<double>::lowest();
                        }
                    }
                }

            } else {
                nodeGains_[node][newProc][0] = std::numeric_limits<double>::lowest();
                nodeGains_[node][newProc][1] = std::numeric_limits<double>::lowest();
                nodeGains_[node][newProc][2] = std::numeric_limits<double>::lowest();
            }
        }
    }

    double ComputeMaxGainInsertOrUpdateHeap(VertexType node) {
        double nodeMaxGain = std::numeric_limits<double>::lowest();
        double nodeChangeInCost = 0;
        unsigned nodeBestStep = 0;
        unsigned nodeBestProc = 0;

        double procChangeInCost = 0;
        double procMax = 0;
        unsigned bestStep = 0;

        for (unsigned proc = 0; proc < numProcs_; proc++) {
            int randCount = 0;

            if (currentSchedule_.vector_schedule.assignedSuperstep(node) > 0
                && currentSchedule_.vector_schedule.assignedSuperstep(node) < currentSchedule_.num_steps() - 1) {
                if (nodeGains_[node][proc][0] > nodeGains_[node][proc][1]) {
                    if (nodeGains_[node][proc][0] > nodeGains_[node][proc][2]) {
                        procMax = nodeGains_[node][proc][0];
                        procChangeInCost = nodeChangeInCosts_[node][proc][0];
                        bestStep = 0;

                    } else {
                        procMax = nodeGains_[node][proc][2];
                        procChangeInCost = nodeChangeInCosts_[node][proc][2];
                        bestStep = 2;
                    }

                } else {
                    if (nodeGains_[node][proc][1] > nodeGains_[node][proc][2]) {
                        procMax = nodeGains_[node][proc][1];
                        procChangeInCost = nodeChangeInCosts_[node][proc][1];
                        bestStep = 1;
                    } else {
                        procMax = nodeGains_[node][proc][2];
                        procChangeInCost = nodeChangeInCosts_[node][proc][2];
                        bestStep = 2;
                    }
                }

            } else if (currentSchedule_.vector_schedule.assignedSuperstep(node) == 0
                       && currentSchedule_.vector_schedule.assignedSuperstep(node) < currentSchedule_.num_steps() - 1) {
                if (nodeGains_[node][proc][2] > nodeGains_[node][proc][1]) {
                    procMax = nodeGains_[node][proc][2];
                    procChangeInCost = nodeChangeInCosts_[node][proc][2];
                    bestStep = 2;
                } else {
                    procMax = nodeGains_[node][proc][1];
                    procChangeInCost = nodeChangeInCosts_[node][proc][1];
                    bestStep = 1;
                }

            } else if (currentSchedule_.vector_schedule.assignedSuperstep(node) > 0
                       && currentSchedule_.vector_schedule.assignedSuperstep(node) == currentSchedule_.num_steps() - 1) {
                if (nodeGains_[node][proc][1] > nodeGains_[node][proc][0]) {
                    procMax = nodeGains_[node][proc][1];
                    procChangeInCost = nodeChangeInCosts_[node][proc][1];
                    bestStep = 1;
                } else {
                    procMax = nodeGains_[node][proc][0];
                    procChangeInCost = nodeChangeInCosts_[node][proc][0];
                    bestStep = 0;
                }
            } else {
                procMax = nodeGains_[node][proc][1];
                procChangeInCost = nodeChangeInCosts_[node][proc][1];
                bestStep = 1;
            }

            if (nodeMaxGain < procMax) {
                nodeMaxGain = procMax;
                nodeChangeInCost = procChangeInCost;
                nodeBestStep = currentSchedule_.vector_schedule.assignedSuperstep(node) + bestStep - 1;
                nodeBestProc = proc;
                randCount = 0;

            } else if (nodeMaxGain <= procMax) {    // only ==

                if (rand() % (2 + randCount) == 0) {
                    nodeMaxGain = procMax;
                    nodeChangeInCost = procChangeInCost;
                    nodeBestStep = currentSchedule_.vector_schedule.assignedSuperstep(node) + bestStep - 1;
                    nodeBestProc = proc;
                    randCount++;
                }
            }
        }

        if (nodeHeapHandles_.find(node) != nodeHeapHandles_.end()) {
            (*nodeHeapHandles_[node]).to_proc = nodeBestProc;
            (*nodeHeapHandles_[node]).to_step = nodeBestStep;
            (*nodeHeapHandles_[node]).change_in_cost = nodeChangeInCost;

            if ((*nodeHeapHandles_[node]).gain >= nodeMaxGain) {
                (*nodeHeapHandles_[node]).gain = nodeMaxGain;
                maxGainHeap_.update(nodeHeapHandles_[node]);
            }

        } else {
            // if (node_max_gain < parameters.gain_threshold && node_change_in_cost >
            // parameters.change_in_cost_threshold)
            //     return node_max_gain;

            KlMove<GraphT> move(node,
                                nodeMaxGain,
                                nodeChangeInCost,
                                currentSchedule_.vector_schedule.assignedProcessor(node),
                                currentSchedule_.vector_schedule.assignedSuperstep(node),
                                nodeBestProc,
                                nodeBestStep);
            nodeHeapHandles_[node] = maxGainHeap_.push(move);
        }

        return nodeMaxGain;
    }

    void ComputeWorkGain(VertexType node, unsigned currentStep, unsigned currentProc, unsigned newProc) {
        if (currentProc == newProc) {
            nodeGains_[node][currentProc][1] = std::numeric_limits<double>::lowest();

        } else {
            if (currentSchedule_.step_max_work[currentStep] == currentSchedule_.step_processor_work[currentStep][currentProc]
                && currentSchedule_.step_processor_work[currentStep][currentProc]
                       > currentSchedule_.step_second_max_work[currentStep]) {
                // new max
                const double newMaxWork = std::max(currentSchedule_.step_processor_work[currentStep][currentProc]
                                                       - currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node),
                                                   currentSchedule_.step_second_max_work[currentStep]);

                if (currentSchedule_.step_processor_work[currentStep][newProc]
                        + currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node)
                    > newMaxWork) {
                    const double gain
                        = static_cast<double>(currentSchedule_.step_max_work[currentStep])
                          - (static_cast<double>(currentSchedule_.step_processor_work[currentStep][newProc])
                             + static_cast<double>(currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node)));

                    nodeGains_[node][newProc][1] += gain;
                    nodeChangeInCosts_[node][newProc][1] -= gain;

                } else {
                    const double gain
                        = static_cast<double>(currentSchedule_.step_max_work[currentStep]) - static_cast<double>(newMaxWork);

                    nodeGains_[node][newProc][1] += gain;
                    nodeChangeInCosts_[node][newProc][1] -= gain;
                }

            } else {
                if (currentSchedule_.step_max_work[currentStep]
                    < currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node)
                          + currentSchedule_.step_processor_work[currentStep][newProc]) {
                    const double gain
                        = (static_cast<double>(currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node))
                           + static_cast<double>(currentSchedule_.step_processor_work[currentStep][newProc])
                           - static_cast<double>(currentSchedule_.step_max_work[currentStep]));

                    nodeGains_[node][newProc][1] -= gain;
                    nodeChangeInCosts_[node][newProc][1] += gain;
                }
            }
        }

        if (currentStep > 0) {
            if (currentSchedule_.step_max_work[currentStep - 1]
                < currentSchedule_.step_processor_work[currentStep - 1][newProc]
                      + currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node)) {
                const double gain = static_cast<double>(currentSchedule_.step_processor_work[currentStep - 1][newProc])
                                    + static_cast<double>(currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node))
                                    - static_cast<double>(currentSchedule_.step_max_work[currentStep - 1]);

                nodeGains_[node][newProc][0] -= gain;

                nodeChangeInCosts_[node][newProc][0] += gain;
            }

            if (currentSchedule_.step_max_work[currentStep] == currentSchedule_.step_processor_work[currentStep][currentProc]
                && currentSchedule_.step_processor_work[currentStep][currentProc]
                       > currentSchedule_.step_second_max_work[currentStep]) {
                if (currentSchedule_.step_max_work[currentStep]
                        - currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node)
                    > currentSchedule_.step_second_max_work[currentStep]) {
                    const double gain = currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node);
                    nodeGains_[node][newProc][0] += gain;
                    nodeChangeInCosts_[node][newProc][0] -= gain;

                } else {
                    const double gain
                        = currentSchedule_.step_max_work[currentStep] - currentSchedule_.step_second_max_work[currentStep];

                    nodeGains_[node][newProc][0] += gain;
                    nodeChangeInCosts_[node][newProc][0] -= gain;
                }
            }

        } else {
            nodeGains_[node][newProc][0] = std::numeric_limits<double>::lowest();
        }

        if (currentStep < currentSchedule_.num_steps() - 1) {
            if (currentSchedule_.step_max_work[currentStep + 1]
                < currentSchedule_.step_processor_work[currentStep + 1][newProc]
                      + currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node)) {
                const double gain = static_cast<double>(currentSchedule_.step_processor_work[currentStep + 1][newProc])
                                    + static_cast<double>(currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node))
                                    - static_cast<double>(currentSchedule_.step_max_work[currentStep + 1]);

                nodeGains_[node][newProc][2] -= gain;
                nodeChangeInCosts_[node][newProc][2] += gain;
            }

            if (currentSchedule_.step_max_work[currentStep] == currentSchedule_.step_processor_work[currentStep][currentProc]
                && currentSchedule_.step_processor_work[currentStep][currentProc]
                       > currentSchedule_.step_second_max_work[currentStep]) {
                if ((currentSchedule_.step_max_work[currentStep]
                     - currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node))
                    > currentSchedule_.step_second_max_work[currentStep]) {
                    const double gain = currentSchedule_.instance->getComputationalDag().VertexWorkWeight(node);

                    nodeGains_[node][newProc][2] += gain;
                    nodeChangeInCosts_[node][newProc][2] -= gain;

                } else {
                    const double gain
                        = currentSchedule_.step_max_work[currentStep] - currentSchedule_.step_second_max_work[currentStep];

                    nodeGains_[node][newProc][2] += gain;
                    nodeChangeInCosts_[node][newProc][2] -= gain;
                }
            }
        } else {
            nodeGains_[node][newProc][2] = std::numeric_limits<double>::lowest();
        }
    }

    virtual void ComputeCommGain(VertexIdxT<GraphT> node, unsigned currentStep, unsigned currentProc, unsigned newProc) = 0;

    void UpdateNodeGains(const std::unordered_set<VertexType> &nodes) {
        for (const auto &node : nodes) {
            ComputeNodeGain(node);
            ComputeMaxGainInsertOrUpdateHeap(node);
        }
    };

    KlMove<GraphT> FindBestMove() {
        const unsigned localMax = 50;
        std::vector<VertexType> maxNodes(localMax);
        unsigned count = 0;
        for (auto iter = maxGainHeap_.ordered_begin(); iter != maxGainHeap_.ordered_end(); ++iter) {
            if (iter->gain >= maxGainHeap_.top().gain && count < localMax) {
                maxNodes[count] = (iter->node);
                count++;

            } else {
                break;
            }
        }

        std::uniform_int_distribution<unsigned> dis(0, count - 1);
        unsigned i = dis(gen_);

        KlMove<GraphT> bestMove = KlMove<GraphT>((*nodeHeapHandles_[maxNodes[i]]));

        maxGainHeap_.erase(nodeHeapHandles_[maxNodes[i]]);
        nodeHeapHandles_.erase(maxNodes[i]);

        return bestMove;
    }

    KlMove<GraphT> ComputeBestMove(VertexType node) {
        double nodeMaxGain = std::numeric_limits<double>::lowest();
        double nodeChangeInCost = 0;
        unsigned nodeBestStep = 0;
        unsigned nodeBestProc = 0;

        double procChangeInCost = 0;
        double procMax = 0;
        unsigned bestStep = 0;
        for (unsigned proc = 0; proc < numProcs_; proc++) {
            unsigned randCount = 0;

            if (currentSchedule_.vector_schedule.assignedSuperstep(node) > 0
                && currentSchedule_.vector_schedule.assignedSuperstep(node) < currentSchedule_.num_steps() - 1) {
                if (nodeGains_[node][proc][0] > nodeGains_[node][proc][1]) {
                    if (nodeGains_[node][proc][0] > nodeGains_[node][proc][2]) {
                        procMax = nodeGains_[node][proc][0];
                        procChangeInCost = nodeChangeInCosts_[node][proc][0];
                        bestStep = 0;

                    } else {
                        procMax = nodeGains_[node][proc][2];
                        procChangeInCost = nodeChangeInCosts_[node][proc][2];
                        bestStep = 2;
                    }

                } else {
                    if (nodeGains_[node][proc][1] > nodeGains_[node][proc][2]) {
                        procMax = nodeGains_[node][proc][1];
                        procChangeInCost = nodeChangeInCosts_[node][proc][1];
                        bestStep = 1;
                    } else {
                        procMax = nodeGains_[node][proc][2];
                        procChangeInCost = nodeChangeInCosts_[node][proc][2];
                        bestStep = 2;
                    }
                }

            } else if (currentSchedule_.vector_schedule.assignedSuperstep(node) == 0
                       && currentSchedule_.vector_schedule.assignedSuperstep(node) < currentSchedule_.num_steps() - 1) {
                if (nodeGains_[node][proc][2] > nodeGains_[node][proc][1]) {
                    procMax = nodeGains_[node][proc][2];
                    procChangeInCost = nodeChangeInCosts_[node][proc][2];
                    bestStep = 2;
                } else {
                    procMax = nodeGains_[node][proc][1];
                    procChangeInCost = nodeChangeInCosts_[node][proc][1];
                    bestStep = 1;
                }

            } else if (currentSchedule_.vector_schedule.assignedSuperstep(node) > 0
                       && currentSchedule_.vector_schedule.assignedSuperstep(node) == currentSchedule_.num_steps() - 1) {
                if (nodeGains_[node][proc][1] > nodeGains_[node][proc][0]) {
                    procMax = nodeGains_[node][proc][1];
                    procChangeInCost = nodeChangeInCosts_[node][proc][1];
                    bestStep = 1;
                } else {
                    procMax = nodeGains_[node][proc][0];
                    procChangeInCost = nodeChangeInCosts_[node][proc][0];
                    bestStep = 0;
                }
            } else {
                procMax = nodeGains_[node][proc][1];
                procChangeInCost = nodeChangeInCosts_[node][proc][1];
                bestStep = 1;
            }

            if (nodeMaxGain < procMax) {
                nodeMaxGain = procMax;
                nodeChangeInCost = procChangeInCost;
                nodeBestStep = currentSchedule_.vector_schedule.assignedSuperstep(node) + bestStep - 1;
                nodeBestProc = proc;
                randCount = 0;

            } else if (nodeMaxGain <= procMax) {
                if (rand() % (2 + randCount) == 0) {
                    nodeMaxGain = procMax;
                    nodeChangeInCost = procChangeInCost;
                    nodeBestStep = currentSchedule_.vector_schedule.assignedSuperstep(node) + bestStep - 1;
                    nodeBestProc = proc;
                    randCount++;
                }
            }
        }

        return KlMove<GraphT>(node,
                              nodeMaxGain,
                              nodeChangeInCost,
                              currentSchedule_.vector_schedule.assignedProcessor(node),
                              currentSchedule_.vector_schedule.assignedSuperstep(node),
                              nodeBestProc,
                              nodeBestStep);
    }

    KlMove<GraphT> BestMoveChangeSuperstep(VertexType node) {
        double nodeMaxGain = std::numeric_limits<double>::lowest();
        double nodeChangeInCost = 0;
        unsigned nodeBestStep = 0;
        unsigned nodeBestProc = 0;

        double procChangeInCost = 0;
        double procMax = 0;
        unsigned bestStep = 0;
        for (unsigned proc = 0; proc < numProcs_; proc++) {
            if (currentSchedule_.vector_schedule.assignedSuperstep(node) > 0
                && currentSchedule_.vector_schedule.assignedSuperstep(node) < currentSchedule_.num_steps() - 1) {
                if (nodeGains_[node][proc][0] > nodeGains_[node][proc][2]) {
                    procMax = nodeGains_[node][proc][0];
                    procChangeInCost = nodeChangeInCosts_[node][proc][0];
                    bestStep = 0;

                } else {
                    procMax = nodeGains_[node][proc][2];
                    procChangeInCost = nodeChangeInCosts_[node][proc][2];
                    bestStep = 2;
                }

            } else if (currentSchedule_.vector_schedule.assignedSuperstep(node) == 0
                       && currentSchedule_.vector_schedule.assignedSuperstep(node) < currentSchedule_.num_steps() - 1) {
                procMax = nodeGains_[node][proc][2];
                procChangeInCost = nodeChangeInCosts_[node][proc][2];
                bestStep = 2;

            } else if (currentSchedule_.vector_schedule.assignedSuperstep(node) > 0
                       && currentSchedule_.vector_schedule.assignedSuperstep(node) == currentSchedule_.num_steps() - 1) {
                procMax = nodeGains_[node][proc][0];
                procChangeInCost = nodeChangeInCosts_[node][proc][0];
                bestStep = 0;

            } else {
                throw std::invalid_argument("error lk base best_move_change_superstep");
            }

            if (nodeMaxGain < procMax) {
                nodeMaxGain = procMax;
                nodeChangeInCost = procChangeInCost;
                nodeBestStep = currentSchedule_.vector_schedule.assignedSuperstep(node) + bestStep - 1;
                nodeBestProc = proc;
            }
        }

        return KlMove<GraphT>(node,
                              nodeMaxGain,
                              nodeChangeInCost,
                              currentSchedule_.vector_schedule.assignedProcessor(node),
                              currentSchedule_.vector_schedule.assignedSuperstep(node),
                              nodeBestProc,
                              nodeBestStep);
    }

    void SaveBestSchedule(const IBspSchedule<GraphT> &schedule) {
        for (const auto &node : currentSchedule_.instance->vertices()) {
            bestSchedule_->setAssignedProcessor(node, schedule.assignedProcessor(node));
            bestSchedule_->setAssignedSuperstep(node, schedule.assignedSuperstep(node));
        }
        bestSchedule_->updateNumberOfSupersteps();
    }

    void ReverseMoveBestSchedule(KlMove<GraphT> move) {
        bestSchedule_->setAssignedProcessor(move.node, move.from_proc);
        bestSchedule_->setAssignedSuperstep(move.node, move.from_step);
    }

    std::unordered_set<VertexType> nodeSelection_;

    void SelectNodes() {
        if (parameters_.selectAllNodes) {
            for (const auto &node : currentSchedule_.instance->vertices()) {
                if (superLockedNodes_.find(node) == superLockedNodes_.end()) {
                    nodeSelection_.insert(node);
                }
            }

        } else {
            SelectNodesThreshold(parameters_.selectionThreshold - superLockedNodes_.size());
        }
    }

    virtual void SelectNodesComm() {
        for (const auto &node : currentSchedule_.instance->vertices()) {
            if (superLockedNodes_.find(node) != superLockedNodes_.end()) {
                continue;
            }

            for (const auto &source : currentSchedule_.instance->getComputationalDag().Parents(node)) {
                if (currentSchedule_.vector_schedule.assignedProcessor(node)
                    != currentSchedule_.vector_schedule.assignedProcessor(source)) {
                    nodeSelection_.insert(node);
                    break;
                }
            }

            for (const auto &target : currentSchedule_.instance->getComputationalDag().Children(node)) {
                if (currentSchedule_.vector_schedule.assignedProcessor(node)
                    != currentSchedule_.vector_schedule.assignedProcessor(target)) {
                    nodeSelection_.insert(node);
                    break;
                }
            }
        }
    }

    void SelectNodesThreshold(std::size_t threshold) {
        std::uniform_int_distribution<VertexIdxT<GraphT>> dis(0, numNodes_ - 1);

        while (nodeSelection_.size() < threshold) {
            auto node = dis(gen_);

            if (superLockedNodes_.find(node) == superLockedNodes_.end()) {
                nodeSelection_.insert(node);
            }
        }
    }

    void SelectNodesPermutationThreshold(std::size_t threshold) {
        std::vector<VertexType> permutation(numNodes_);
        std::iota(std::begin(permutation), std::end(permutation), 0);

        std::shuffle(permutation.begin(), permutation.end(), gen_);

        for (std::size_t i = 0; i < threshold; i++) {
            if (superLockedNodes_.find(permutation[i]) == superLockedNodes_.end()) {
                nodeSelection_.insert(permutation[i]);
            }
        }
    }

    void SelectNodesViolations() {
        if (currentSchedule_.current_violations.empty()) {
            SelectNodes();
            return;
        }

        for (const auto &edge : currentSchedule_.current_violations) {
            const auto &sourceV = source(edge, currentSchedule_.instance->getComputationalDag());
            const auto &targetV = target(edge, currentSchedule_.instance->getComputationalDag());

            nodeSelection_.insert(sourceV);
            nodeSelection_.insert(targetV);

            for (const auto &child : currentSchedule_.instance->getComputationalDag().Children(sourceV)) {
                if (child != targetV) {
                    nodeSelection_.insert(child);
                }
            }

            for (const auto &parent : currentSchedule_.instance->getComputationalDag().Parents(sourceV)) {
                if (parent != targetV) {
                    nodeSelection_.insert(parent);
                }
            }

            for (const auto &child : currentSchedule_.instance->getComputationalDag().Children(targetV)) {
                if (child != sourceV) {
                    nodeSelection_.insert(child);
                }
            }

            for (const auto &parent : currentSchedule_.instance->getComputationalDag().Parents(targetV)) {
                if (parent != sourceV) {
                    nodeSelection_.insert(parent);
                }
            }
        }
    }

    void SelectNodesConsequeMaxWork(bool doNotSelectSuperLockedNodes = false) {
        if (stepSelectionEpochCounter_ > parameters_.maxStepSelectionEpochs) {
#ifdef KL_DEBUG
            std::cout << "step selection epoch counter exceeded. conseque max work" << std::endl;
#endif

            SelectNodes();
            return;
        }

        unsigned maxWorkStep = 0;
        unsigned maxStep = 0;
        unsigned secondMaxWorkStep = 0;
        unsigned secondMaxStep = 0;

        for (unsigned proc = 0; proc < numProcs_; proc++) {
            if (currentSchedule_.step_processor_work[stepSelectionCounter_][proc] > maxWorkStep) {
                secondMaxWorkStep = maxWorkStep;
                secondMaxStep = maxStep;
                maxWorkStep = currentSchedule_.step_processor_work[stepSelectionCounter_][proc];
                maxStep = proc;

            } else if (currentSchedule_.step_processor_work[stepSelectionCounter_][proc] > secondMaxWorkStep) {
                secondMaxWorkStep = currentSchedule_.step_processor_work[stepSelectionCounter_][proc];
                secondMaxStep = proc;
            }
        }

        if (currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][maxStep].size()
            < parameters_.selectionThreshold * .66) {
            nodeSelection_.insert(currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][maxStep].begin(),
                                  currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][maxStep].end());

        } else {
            std::sample(currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][maxStep].begin(),
                        currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][maxStep].end(),
                        std::inserter(nodeSelection_, nodeSelection_.end()),
                        static_cast<unsigned>(std::round(parameters_.selectionThreshold * .66)),
                        gen_);
        }

        if (currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][secondMaxStep].size()
            < parameters_.selectionThreshold * .33) {
            nodeSelection_.insert(
                currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][secondMaxStep].begin(),
                currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][secondMaxStep].end());

        } else {
            std::sample(currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][secondMaxStep].begin(),
                        currentSchedule_.set_schedule.step_processor_vertices[stepSelectionCounter_][secondMaxStep].end(),
                        std::inserter(nodeSelection_, nodeSelection_.end()),
                        static_cast<unsigned>(std::round(parameters_.selectionThreshold * .33)),
                        gen_);
        }

        if (doNotSelectSuperLockedNodes) {
            for (const auto &node : superLockedNodes_) {
                nodeSelection_.erase(node);
            }
        }

#ifdef KL_DEBUG
        std::cout << "step selection conseque max work, node selection size " << node_selection.size()
                  << " ... selected nodes assigend to superstep " << step_selection_counter << " and procs " << max_step
                  << " and " << second_max_step << std::endl;
#endif

        stepSelectionCounter_++;
        if (stepSelectionCounter_ >= currentSchedule_.num_steps()) {
            stepSelectionCounter_ = 0;
            stepSelectionEpochCounter_++;
        }
    }

    void SelectNodesCheckRemoveSuperstep() {
        if (stepSelectionEpochCounter_ > parameters_.maxStepSelectionEpochs) {
#ifdef KL_DEBUG
            std::cout << "step selection epoch counter exceeded, remove supersteps" << std::endl;
#endif

            SelectNodes();
            return;
        }

        for (unsigned stepToRemove = stepSelectionCounter_; stepToRemove < currentSchedule_.num_steps(); stepToRemove++) {
#ifdef KL_DEBUG
            std::cout << "checking step to remove " << step_to_remove << " / " << current_schedule.num_steps() << std::endl;
#endif

            if (CheckRemoveSuperstep(stepToRemove)) {
#ifdef KL_DEBUG
                std::cout << "trying to remove superstep " << step_to_remove << std::endl;
#endif

                if (ScatterNodesRemoveSuperstep(stepToRemove)) {
                    for (unsigned proc = 0; proc < numProcs_; proc++) {
                        if (stepToRemove < currentSchedule_.num_steps()) {
                            nodeSelection_.insert(
                                currentSchedule_.set_schedule.step_processor_vertices[stepToRemove][proc].begin(),
                                currentSchedule_.set_schedule.step_processor_vertices[stepToRemove][proc].end());
                        }

                        if (stepToRemove > 0) {
                            nodeSelection_.insert(
                                currentSchedule_.set_schedule.step_processor_vertices[stepToRemove - 1][proc].begin(),
                                currentSchedule_.set_schedule.step_processor_vertices[stepToRemove - 1][proc].end());
                        }
                    }

                    stepSelectionCounter_ = stepToRemove + 1;

                    if (stepSelectionCounter_ >= currentSchedule_.num_steps()) {
                        stepSelectionCounter_ = 0;
                        stepSelectionEpochCounter_++;
                    }

                    parameters_.violationsThreshold = 0;
                    superLockedNodes_.clear();
#ifdef KL_DEBUG
                    std::cout << "---- reset super locked nodes" << std::endl;
#endif

                    return;
                }
            }
        }

#ifdef KL_DEBUG
        std::cout << "no superstep to remove" << std::endl;
#endif

        stepSelectionEpochCounter_++;
        SelectNodes();
        return;
    }

    unsigned stepSelectionCounter_ = 0;
    unsigned stepSelectionEpochCounter_ = 0;

    bool autoAlternate_ = false;
    bool alternateResetRemoveSuperstep_ = false;
    bool resetSuperstep_ = false;

    virtual bool CheckRemoveSuperstep(unsigned step) {
        if (currentSchedule_.num_steps() <= 2) {
            return false;
        }

        VWorkwT<GraphT> totalWork = 0;

        for (unsigned proc = 0; proc < numProcs_; proc++) {
            totalWork += currentSchedule_.step_processor_work[step][proc];
        }

        if (totalWork < 2.0 * currentSchedule_.instance->synchronisationCosts()) {
            return true;
        }
        return false;
    }

    bool ScatterNodesRemoveSuperstep(unsigned step) {
        assert(step < currentSchedule_.num_steps());

        std::vector<KlMove<GraphT>> moves;

        bool abort = false;

        for (unsigned proc = 0; proc < numProcs_; proc++) {
            for (const auto &node : currentSchedule_.set_schedule.step_processor_vertices[step][proc]) {
                ComputeNodeGain(node);
                moves.push_back(BestMoveChangeSuperstep(node));

                if (moves.back().gain <= std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                if constexpr (currentSchedule_.use_memory_constraint) {
                    currentSchedule_.memory_constraint.apply_move(node, proc, step, moves.back().to_proc, moves.back().to_step);
                }
            }

            if (abort) {
                break;
            }
        }

        if (abort) {
            currentSchedule_.recompute_neighboring_supersteps(step);

#ifdef KL_DEBUG
            BspSchedule<Graph_t> tmp_schedule(current_schedule.set_schedule);
            if (not tmp_schedule.satisfiesMemoryConstraints()) {
                std::cout << "Mem const violated" << std::endl;
            }
#endif

            return false;
        }

        for (unsigned proc = 0; proc < numProcs_; proc++) {
            currentSchedule_.set_schedule.step_processor_vertices[step][proc].clear();
        }

        for (const auto &move : moves) {
#ifdef KL_DEBUG
            std::cout << "scatter node " << move.node << " to proc " << move.to_proc << " to step " << move.to_step << std::endl;
#endif

            currentSchedule_.vector_schedule.setAssignedSuperstep(move.node, move.to_step);
            currentSchedule_.vector_schedule.setAssignedProcessor(move.node, move.to_proc);
            currentSchedule_.set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
        }

        currentSchedule_.remove_superstep(step);

#ifdef KL_DEBUG
        BspSchedule<Graph_t> tmp_schedule(current_schedule.set_schedule);
        if (not tmp_schedule.satisfiesMemoryConstraints()) {
            std::cout << "Mem const violated" << std::endl;
        }
#endif

        return true;
    }

    void SelectNodesCheckResetSuperstep() {
        if (stepSelectionEpochCounter_ > parameters_.maxStepSelectionEpochs) {
#ifdef KL_DEBUG
            std::cout << "step selection epoch counter exceeded, reset supersteps" << std::endl;
#endif

            SelectNodes();
            return;
        }

        for (unsigned stepToRemove = stepSelectionCounter_; stepToRemove < currentSchedule_.num_steps(); stepToRemove++) {
#ifdef KL_DEBUG
            std::cout << "checking step to reset " << step_to_remove << " / " << current_schedule.num_steps() << std::endl;
#endif

            if (CheckResetSuperstep(stepToRemove)) {
#ifdef KL_DEBUG
                std::cout << "trying to reset superstep " << step_to_remove << std::endl;
#endif

                if (ScatterNodesResetSuperstep(stepToRemove)) {
                    for (unsigned proc = 0; proc < numProcs_; proc++) {
                        if (stepToRemove < currentSchedule_.num_steps() - 1) {
                            nodeSelection_.insert(
                                currentSchedule_.set_schedule.step_processor_vertices[stepToRemove + 1][proc].begin(),
                                currentSchedule_.set_schedule.step_processor_vertices[stepToRemove + 1][proc].end());
                        }

                        if (stepToRemove > 0) {
                            nodeSelection_.insert(
                                currentSchedule_.set_schedule.step_processor_vertices[stepToRemove - 1][proc].begin(),
                                currentSchedule_.set_schedule.step_processor_vertices[stepToRemove - 1][proc].end());
                        }
                    }

                    stepSelectionCounter_ = stepToRemove + 1;

                    if (stepSelectionCounter_ >= currentSchedule_.num_steps()) {
                        stepSelectionCounter_ = 0;
                        stepSelectionEpochCounter_++;
                    }

                    parameters_.violationsThreshold = 0;
                    superLockedNodes_.clear();
#ifdef KL_DEBUG
                    std::cout << "---- reset super locked nodes" << std::endl;
#endif

                    return;
                }
            }
        }

#ifdef KL_DEBUG
        std::cout << "no superstep to reset" << std::endl;
#endif

        stepSelectionEpochCounter_++;
        SelectNodes();
        return;
    }

    virtual bool CheckResetSuperstep(unsigned step) {
        if (currentSchedule_.num_steps() <= 2) {
            return false;
        }

        VWorkwT<GraphT> totalWork = 0;
        VWorkwT<GraphT> maxTotalWork = 0;
        VWorkwT<GraphT> minTotalWork = std::numeric_limits<VWorkwT<GraphT>>::max();

        for (unsigned proc = 0; proc < numProcs_; proc++) {
            totalWork += currentSchedule_.step_processor_work[step][proc];
            maxTotalWork = std::max(maxTotalWork, currentSchedule_.step_processor_work[step][proc]);
            minTotalWork = std::min(minTotalWork, currentSchedule_.step_processor_work[step][proc]);
        }

#ifdef KL_DEBUG

        std::cout << " avg "
                  << static_cast<double>(total_work) / static_cast<double>(current_schedule.instance->numberOfProcessors())
                  << " max " << max_total_work << " min " << min_total_work << std::endl;
#endif

        if (static_cast<double>(totalWork) / static_cast<double>(currentSchedule_.instance->numberOfProcessors())
                - static_cast<double>(minTotalWork)
            > 0.1 * static_cast<double>(minTotalWork)) {
            return true;
        }

        return false;
    }

    bool ScatterNodesResetSuperstep(unsigned step) {
        assert(step < currentSchedule_.num_steps());

        std::vector<KlMove<GraphT>> moves;

        bool abort = false;

        for (unsigned proc = 0; proc < numProcs_; proc++) {
            for (const auto &node : currentSchedule_.set_schedule.step_processor_vertices[step][proc]) {
                ComputeNodeGain(node);
                moves.push_back(BestMoveChangeSuperstep(node));

                if (moves.back().gain == std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                if constexpr (currentSchedule_.use_memory_constraint) {
                    currentSchedule_.memory_constraint.apply_forward_move(
                        node, proc, step, moves.back().to_proc, moves.back().to_step);
                }
            }

            if (abort) {
                break;
            }
        }

        if (abort) {
            currentSchedule_.recompute_neighboring_supersteps(step);
            return false;
        }

        for (unsigned proc = 0; proc < numProcs_; proc++) {
            currentSchedule_.set_schedule.step_processor_vertices[step][proc].clear();
        }

        for (const auto &move : moves) {
#ifdef KL_DEBUG
            std::cout << "scatter node " << move.node << " to proc " << move.to_proc << " to step " << move.to_step << std::endl;
#endif

            currentSchedule_.vector_schedule.setAssignedSuperstep(move.node, move.to_step);
            currentSchedule_.vector_schedule.setAssignedProcessor(move.node, move.to_proc);
            currentSchedule_.set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
        }

        currentSchedule_.reset_superstep(step);

        return true;
    }

    void SelectUnlockNeighbors(VertexType node) {
        for (const auto &target : currentSchedule_.instance->getComputationalDag().Children(node)) {
            if (CheckNodeUnlocked(target)) {
                nodeSelection_.insert(target);
                nodesToUpdate_.insert(target);
            }
        }

        for (const auto &source : currentSchedule_.instance->getComputationalDag().Parents(node)) {
            if (CheckNodeUnlocked(source)) {
                nodeSelection_.insert(source);
                nodesToUpdate_.insert(source);
            }
        }
    }

    void SetParameters() {
        if (numNodes_ < 250) {
            parameters_.maxOuterIterations = 300;

            parameters_.selectAllNodes = true;
            parameters_.selectionThreshold = numNodes_;

        } else if (numNodes_ < 1000) {
            parameters_.maxOuterIterations = static_cast<unsigned>(numNodes_ / 2);

            parameters_.selectAllNodes = true;
            parameters_.selectionThreshold = numNodes_;

        } else if (numNodes_ < 5000) {
            parameters_.maxOuterIterations = 4 * static_cast<unsigned>(std::sqrt(numNodes_));

            parameters_.selectionThreshold = numNodes_ / 3;

        } else if (numNodes_ < 10000) {
            parameters_.maxOuterIterations = 3 * static_cast<unsigned>(std::sqrt(numNodes_));

            parameters_.selectionThreshold = numNodes_ / 3;

        } else if (numNodes_ < 50000) {
            parameters_.maxOuterIterations = static_cast<unsigned>(std::sqrt(numNodes_));

            parameters_.selectionThreshold = numNodes_ / 5;

        } else if (numNodes_ < 100000) {
            parameters_.maxOuterIterations = 2 * static_cast<unsigned>(std::log(numNodes_));

            parameters_.selectionThreshold = numNodes_ / 10;

        } else {
            parameters_.maxOuterIterations = static_cast<unsigned>(std::min(10000.0, std::log(numNodes_)));

            parameters_.selectionThreshold = numNodes_ / 10;
        }

        if (parameters_.quickPass) {
            parameters_.maxOuterIterations = 50;
            parameters_.maxNoImprovementIterations = 25;
        }

        if (autoAlternate_ && currentSchedule_.instance->getArchitecture().synchronisationCosts() > 10000.0) {
#ifdef KL_DEBUG
            std::cout << "KLBase set parameters, large synchchost: only remove supersets" << std::endl;
#endif
            resetSuperstep_ = false;
            alternateResetRemoveSuperstep_ = false;
        }

#ifdef KL_DEBUG
        if (parameters.select_all_nodes) {
            std::cout << "KLBase set parameters, select all nodes" << std::endl;
        } else {
            std::cout << "KLBase set parameters, selection threshold: " << parameters.selection_threshold << std::endl;
        }
#endif
    }

    virtual void CleanupDatastructures() {
        nodeChangeInCosts_.clear();
        nodeGains_.clear();

        unlock_.clear();

        maxGainHeap_.clear();
        nodeHeapHandles_.clear();

        currentSchedule_.cleanup_superstep_datastructures();
    }

    void ResetRunDatastructures() {
        nodeSelection_.clear();
        nodesToUpdate_.clear();
        lockedNodes_.clear();
        superLockedNodes_.clear();
    }

    bool RunLocalSearchWithoutViolations() {
        penalty_ = std::numeric_limits<double>::max() * .24;

        double initialCosts = currentSchedule_.current_cost;

        auto startTime = std::chrono::high_resolution_clock::now();

        SelectNodesThreshold(parameters_.selectionThreshold);

        InitializeGainHeap(nodeSelection_);

#ifdef KL_DEBUG
        std::cout << "Initial costs " << initial_costs << std::endl;
#endif

        for (unsigned outerCounter = 0; outerCounter < parameters_.maxOuterIterations; outerCounter++) {
#ifdef KL_DEBUG
            std::cout << "outer iteration " << outer_counter << std::endl;
#endif
            unsigned failedBranches = 0;
            // double best_iter_costs = current_schedule.current_cost;

            unsigned innerCounter = 0;

            while (failedBranches < 3 && innerCounter < parameters_.maxInnerIterations && maxGainHeap_.size() > 0) {
                innerCounter++;

                const double iterCosts = currentSchedule_.current_cost;

                KlMove<GraphT> bestMove = FindBestMove();    // O(log n)

                if (bestMove.gain < -std::numeric_limits<double>::max() * .25) {
                    continue;
                }

                currentSchedule_.apply_move(bestMove);    // O(p + log n)

                lockedNodes_.insert(bestMove.node);

#ifdef KL_DEBUG
                double tmp_costs = current_schedule.current_cost;
                if (tmp_costs != compute_current_costs()) {
                    std::cout << "current costs: " << current_schedule.current_cost << " best move gain: " << best_move.gain
                              << " best move costs: " << best_move.change_in_cost << " tmp cost: " << tmp_costs << std::endl;

                    std::cout << "! costs not equal " << std::endl;
                }
#endif

                if (bestMove.change_in_cost > 0 && currentSchedule_.current_feasible) {
                    if (bestScheduleCosts_ > iterCosts) {
#ifdef KL_DEBUG
                        std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                        bestScheduleCosts_ = iterCosts;
                        SaveBestSchedule(currentSchedule_.vector_schedule);    // O(n)
                        ReverseMoveBestSchedule(bestMove);
                    }
                }

                ComputeNodesToUpdate(bestMove);

                SelectUnlockNeighbors(bestMove.node);

                UpdateNodeGains(nodesToUpdate_);

#ifdef KL_DEBUG
                std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                          << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                          << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                          << " violations: " << current_schedule.current_violations.size() << " cost "
                          << current_schedule.current_cost << std::endl;
#endif

                // if (not current_schedule.current_feasible) {

                if (currentSchedule_.current_cost > (1.04 + outerCounter * 0.002) * bestScheduleCosts_) {
#ifdef KL_DEBUG
                    std::cout << "current cost " << current_schedule.current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs << " rollback to best schedule"
                              << std::endl;
#endif

                    currentSchedule_.set_current_schedule(*bestSchedule_);

                    // set_initial_reward_penalty();
                    InitializeGainHeapUnlockedNodes(nodeSelection_);

                    failedBranches++;
                }
                //}

            }    // while

#ifdef KL_DEBUG
            std::cout << "end inner loop current cost: " << current_schedule.current_cost << " with "
                      << current_schedule.current_violations.size() << " violation, best sol cost: " << best_schedule_costs
                      << " with " << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                      << parameters.max_outer_iterations << std::endl;
#endif

            if (currentSchedule_.current_feasible) {
                if (currentSchedule_.current_cost <= bestScheduleCosts_) {
                    SaveBestSchedule(currentSchedule_.vector_schedule);
                    bestScheduleCosts_ = currentSchedule_.current_cost;
                } else {
                    currentSchedule_.set_current_schedule(*bestSchedule_);
                }
            } else {
                currentSchedule_.set_current_schedule(*bestSchedule_);
            }

            ResetLockedNodes();
            nodeSelection_.clear();
            SelectNodesThreshold(parameters_.selectionThreshold);

            InitializeGainHeap(nodeSelection_);

#ifdef KL_DEBUG
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

            if (computeWithTimeLimit_) {
                auto finishTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();

                if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds) {
                    break;
                }
            }

        }    // for

        CleanupDatastructures();

        if (initialCosts > currentSchedule_.current_cost) {
            return true;
        } else {
            return false;
        }
    }

    bool RunLocalSearchSimple() {
        SetInitialRewardPenalty();

        const double initialCosts = currentSchedule_.current_cost;

        unsigned improvementCounter = 0;

        auto startTime = std::chrono::high_resolution_clock::now();

        SelectNodes();

        InitializeGainHeap(nodeSelection_);

#ifdef KL_DEBUG
        std::cout << "Initial costs " << initial_costs << std::endl;
#endif

        for (unsigned outerCounter = 0; outerCounter < parameters_.maxOuterIterations; outerCounter++) {
#ifdef KL_DEBUG
            std::cout << "outer iteration " << outer_counter << std::endl;
            if (max_gain_heap.size() == 0) {
                std::cout << "max gain heap empty" << std::endl;
            }
#endif
            unsigned failedBranches = 0;
            double bestIterCosts = currentSchedule_.current_cost;

            VertexType nodeCausingFirstViolation = 0;

            unsigned innerCounter = 0;

            while (failedBranches < parameters_.maxNumFailedBranches && innerCounter < parameters_.maxInnerIterations
                   && maxGainHeap_.size() > 0) {
                innerCounter++;

                const bool iterFeasible = currentSchedule_.current_feasible;
                const double iterCosts = currentSchedule_.current_cost;

                KlMove<GraphT> bestMove = FindBestMove();    // O(log n)

                if (bestMove.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                    std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                          << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                          << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                          << " violations: " << current_schedule.current_violations.size() << " cost "
                          << current_schedule.current_cost << std::endl;
#endif

                currentSchedule_.apply_move(bestMove);    // O(p + log n)

                UpdateRewardPenalty();
                lockedNodes_.insert(bestMove.node);

#ifdef KL_DEBUG
                double tmp_costs = current_schedule.current_cost;
                if (tmp_costs != compute_current_costs()) {
                    std::cout << "current costs: " << current_schedule.current_cost << " best move gain: " << best_move.gain
                              << " best move costs: " << best_move.change_in_cost << " tmp cost: " << tmp_costs << std::endl;

                    std::cout << "! costs not equal " << std::endl;
                }
#endif

                if (iterFeasible != currentSchedule_.current_feasible) {
                    if (iterFeasible) {
#ifdef KL_DEBUG
                        std::cout << "===> current schedule changed from feasible to infeasible" << std::endl;
#endif

                        nodeCausingFirstViolation = bestMove.node;

                        if (iterCosts < bestScheduleCosts_) {
#ifdef KL_DEBUG
                            std::cout << "save best schedule with costs " << iter_costs << std::endl;
#endif
                            bestScheduleCosts_ = iterCosts;
                            SaveBestSchedule(currentSchedule_.vector_schedule);    // O(n)
                            ReverseMoveBestSchedule(bestMove);
                        }

                    } else {
#ifdef KL_DEBUG
                        std::cout << "===> current schedule changed from infeasible to feasible" << std::endl;
#endif
                    }
                } else if (bestMove.change_in_cost > 0 && currentSchedule_.current_feasible) {
                    if (iterCosts < bestScheduleCosts_) {
#ifdef KL_DEBUG
                        std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                        bestScheduleCosts_ = iterCosts;
                        SaveBestSchedule(currentSchedule_.vector_schedule);    // O(n)
                        ReverseMoveBestSchedule(bestMove);
                    }
                }

                ComputeNodesToUpdate(bestMove);

                SelectUnlockNeighbors(bestMove.node);

                if (CheckViolationLocked()) {
                    if (iterFeasible != currentSchedule_.current_feasible && iterFeasible) {
                        nodeCausingFirstViolation = bestMove.node;
                    }
                    superLockedNodes_.insert(nodeCausingFirstViolation);
#ifdef KL_DEBUG
                    std::cout << "abort iteration on locked violation, super locking node " << node_causing_first_violation
                              << std::endl;
#endif
                    break;
                }

                UpdateNodeGains(nodesToUpdate_);

                if (currentSchedule_.current_cost
                    > (parameters_.maxDivBestSolBasePercent + outerCounter * parameters_.maxDivBestSolRatePercent)
                          * bestScheduleCosts_) {
#ifdef KL_DEBUG
                    std::cout << "current cost " << current_schedule.current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs << " rollback to best schedule"
                              << std::endl;
#endif

                    currentSchedule_.set_current_schedule(*bestSchedule_);

                    SetInitialRewardPenalty();
                    InitializeGainHeapUnlockedNodes(nodeSelection_);

                    failedBranches++;
                }

            }    // while

#ifdef KL_DEBUG
            std::cout << "end inner loop current cost: " << current_schedule.current_cost << " with "
                      << current_schedule.current_violations.size() << " violation, best sol cost: " << best_schedule_costs
                      << " with " << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                      << parameters.max_outer_iterations << std::endl;
#endif

            if (currentSchedule_.current_feasible) {
                if (currentSchedule_.current_cost <= bestScheduleCosts_) {
                    SaveBestSchedule(currentSchedule_.vector_schedule);
                    bestScheduleCosts_ = currentSchedule_.current_cost;
                } else {
                    currentSchedule_.set_current_schedule(*bestSchedule_);
                }
            } else {
                currentSchedule_.set_current_schedule(*bestSchedule_);
            }

            if (outerCounter == 20) {
                parameters_.initialPenalty = 0.0;
#ifdef KL_DEBUG
                std::cout << "---- reset initial penalty" << std::endl;
#endif
            }
            if (outerCounter > 0 && outerCounter % 30 == 0) {
                superLockedNodes_.clear();
#ifdef KL_DEBUG
                std::cout << "---- reset super locked nodes" << std::endl;
#endif
            }

            ResetLockedNodes();

            nodeSelection_.clear();
            SelectNodes();

            SetInitialRewardPenalty();

            InitializeGainHeap(nodeSelection_);

#ifdef KL_DEBUG
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

            if (computeWithTimeLimit_) {
                auto finishTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();
                if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds) {
                    break;
                }
            }

            if (bestIterCosts <= currentSchedule_.current_cost) {
                if (improvementCounter++ >= parameters_.maxNoImprovementIterations) {
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << parameters.max_no_improvement_iterations
                              << " iterations, end local search" << std::endl;
#endif
                    break;
                }
            } else {
                improvementCounter = 0;
            }

        }    // for

        CleanupDatastructures();

#ifdef KL_DEBUG
        std::cout << "kl done, current cost " << best_schedule_costs << " vs " << initial_costs << " initial costs" << std::endl;
        assert(best_schedule->satisfiesPrecedenceConstraints());
#endif

        if (initialCosts > currentSchedule_.current_cost) {
            return true;
        } else {
            return false;
        }
    }

    bool RunLocalSearchRemoveSupersteps() {
        const double initialCosts = currentSchedule_.current_cost;

#ifdef KL_DEBUG
        std::cout << "Initial costs " << initial_costs << std::endl;
#endif

        unsigned noImprovementIterCounter = 0;

        auto startTime = std::chrono::high_resolution_clock::now();

        SelectNodesCheckRemoveSuperstep();

        UpdateRewardPenalty();

        InitializeGainHeap(nodeSelection_);

        for (unsigned outerCounter = 0; outerCounter < parameters_.maxOuterIterations; outerCounter++) {
#ifdef KL_DEBUG
            std::cout << "outer iteration " << outer_counter << " current costs: " << current_schedule.current_cost << std::endl;
            if (max_gain_heap.size() == 0) {
                std::cout << "max gain heap empty" << std::endl;
            }
#endif

            unsigned conseqNoGainMovesCounter = 0;

            unsigned failedBranches = 0;
            double bestIterCosts = currentSchedule_.current_cost;

            VertexType nodeCausingFirstViolation = 0;

            unsigned innerCounter = 0;

            while (failedBranches < parameters_.maxNumFailedBranches && innerCounter < parameters_.maxInnerIterations
                   && maxGainHeap_.size() > 0) {
                innerCounter++;

                const bool iterFeasible = currentSchedule_.current_feasible;
                const double iterCosts = currentSchedule_.current_cost;

                KlMove<GraphT> bestMove = FindBestMove();    // O(log n)

                if (bestMove.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                    std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                          << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                          << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                          << " violations: " << current_schedule.current_violations.size() << " old cost "
                          << current_schedule.current_cost << " new cost "
                          << current_schedule.current_cost + best_move.change_in_cost << std::endl;
#endif

                currentSchedule_.apply_move(bestMove);    // O(p + log n)

                UpdateRewardPenalty();
                lockedNodes_.insert(bestMove.node);

#ifdef KL_DEBUG
                double tmp_costs = current_schedule.current_cost;
                if (tmp_costs != compute_current_costs()) {
                    std::cout << "current costs: " << current_schedule.current_cost << " best move gain: " << best_move.gain
                              << " best move costs: " << best_move.change_in_cost << " tmp cost: " << tmp_costs << std::endl;

                    std::cout << "! costs not equal " << std::endl;
                }
#endif

                if (iterFeasible != currentSchedule_.current_feasible) {
                    if (iterFeasible) {
#ifdef KL_DEBUG
                        std::cout << "===> current schedule changed from feasible to infeasible" << std::endl;
#endif

                        nodeCausingFirstViolation = bestMove.node;

                        if (iterCosts < bestScheduleCosts_) {
#ifdef KL_DEBUG
                            std::cout << "save best schedule with costs " << iter_costs << std::endl;
#endif
                            bestScheduleCosts_ = iterCosts;
                            SaveBestSchedule(currentSchedule_.vector_schedule);    // O(n)
                            ReverseMoveBestSchedule(bestMove);
#ifdef KL_DEBUG
                            std::cout << "KLBase save best schedule with (source node comm) cost "
                                      << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                                      << best_schedule->numberOfSupersteps() << std::endl;
#endif
                        }

                    } else {
#ifdef KL_DEBUG
                        std::cout << "===> current schedule changed from infeasible to feasible" << std::endl;
#endif
                    }
                } else if (bestMove.change_in_cost > 0 && currentSchedule_.current_feasible) {
                    if (iterCosts < bestScheduleCosts_) {
#ifdef KL_DEBUG
                        std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                        bestScheduleCosts_ = iterCosts;
                        SaveBestSchedule(currentSchedule_.vector_schedule);    // O(n)
                        ReverseMoveBestSchedule(bestMove);
#ifdef KL_DEBUG
                        std::cout << "KLBase save best schedule with (source node comm) cost "
                                  << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                                  << best_schedule->numberOfSupersteps() << std::endl;
#endif
                    }
                }

                ComputeNodesToUpdate(bestMove);

                SelectUnlockNeighbors(bestMove.node);

                if (CheckViolationLocked()) {
                    if (iterFeasible != currentSchedule_.current_feasible && iterFeasible) {
                        nodeCausingFirstViolation = bestMove.node;
                    }
                    superLockedNodes_.insert(nodeCausingFirstViolation);
#ifdef KL_DEBUG
                    std::cout << "abort iteration on locked violation, super locking node " << node_causing_first_violation
                              << std::endl;
#endif
                    break;
                }

                UpdateNodeGains(nodesToUpdate_);

                if (currentSchedule_.current_cost
                    > (parameters_.maxDivBestSolBasePercent + outerCounter * parameters_.maxDivBestSolRatePercent)
                          * bestScheduleCosts_) {
#ifdef KL_DEBUG
                    std::cout << "current cost " << current_schedule.current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs << " rollback to best schedule"
                              << std::endl;
#endif

                    currentSchedule_.set_current_schedule(*bestSchedule_);

                    SetInitialRewardPenalty();
                    InitializeGainHeapUnlockedNodes(nodeSelection_);

#ifdef KL_DEBUG
                    std::cout << "new current cost " << current_schedule.current_cost << std::endl;
#endif

                    failedBranches++;
                }

            }    // while

#ifdef KL_DEBUG
            std::cout << std::setprecision(12) << "end inner loop current cost: " << current_schedule.current_cost << " with "
                      << current_schedule.current_violations.size() << " violation, best sol cost: " << best_schedule_costs
                      << " with " << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                      << parameters.max_outer_iterations << std::endl;
#endif

            if (currentSchedule_.current_feasible) {
                if (currentSchedule_.current_cost <= bestScheduleCosts_) {
                    SaveBestSchedule(currentSchedule_.vector_schedule);
                    bestScheduleCosts_ = currentSchedule_.current_cost;
#ifdef KL_DEBUG
                    std::cout << "KLBase save best schedule with (source node comm) cost "
                              << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                              << best_schedule->numberOfSupersteps() << std::endl;
#endif
                } else {
                    currentSchedule_.set_current_schedule(*bestSchedule_);
                }
            } else {
                currentSchedule_.set_current_schedule(*bestSchedule_);
            }

            if (outerCounter > 0 && outerCounter % 30 == 0) {
                superLockedNodes_.clear();
#ifdef KL_DEBUG
                std::cout << "---- reset super locked nodes" << std::endl;
#endif
            }

            ResetLockedNodes();

            nodeSelection_.clear();
            SelectNodesCheckRemoveSuperstep();

            UpdateRewardPenalty();

            InitializeGainHeap(nodeSelection_);

#ifdef KL_DEBUG
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

            if (computeWithTimeLimit_) {
                auto finishTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();
                if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds) {
                    break;
                }
            }

            if (bestIterCosts <= currentSchedule_.current_cost) {
                noImprovementIterCounter++;

                if (noImprovementIterCounter > parameters_.resetEpochCounterThreshold) {
                    stepSelectionEpochCounter_ = 0;
                    parameters_.resetEpochCounterThreshold += currentSchedule_.num_steps();
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << no_improvement_iter_counter
                              << " iterations, reset epoc counter. Increase reset threshold to "
                              << parameters.reset_epoch_counter_threshold << std::endl;
#endif
                }

                if (noImprovementIterCounter > 10) {
                    parameters_.initialPenalty = 0.0;
                    parameters_.violationsThreshold = 3;
#ifdef KL_DEBUG
                    std::cout << "---- reset initial penalty " << parameters.initial_penalty << " violations threshold "
                              << parameters.violations_threshold << std::endl;
#endif
                }

                if (noImprovementIterCounter == 35) {
                    parameters_.maxDivBestSolBasePercent *= 1.02;
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << no_improvement_iter_counter
                              << " iterations, increase max_div_best_sol_base_percent to "
                              << parameters.max_div_best_sol_base_percent << std::endl;
#endif
                }

                if (noImprovementIterCounter >= parameters_.maxNoImprovementIterations) {
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << parameters.max_no_improvement_iterations
                              << " iterations, end local search" << std::endl;
#endif
                    break;
                }
            } else {
                noImprovementIterCounter = 0;
            }

        }    // for

        CleanupDatastructures();

#ifdef KL_DEBUG
        std::cout << "kl done, current cost " << best_schedule_costs << " vs " << initial_costs << " initial costs" << std::endl;
        assert(best_schedule->satisfiesPrecedenceConstraints());
#endif

        if (initialCosts > currentSchedule_.current_cost) {
            return true;
        } else {
            return false;
        }
    }

    bool RunLocalSearchUnlockDelay() {
        const double initialCosts = currentSchedule_.current_cost;

#ifdef KL_DEBUG_1
        std::cout << "Initial costs " << initial_costs << " with " << best_schedule->numberOfSupersteps() << " supersteps."
                  << std::endl;
#endif

#ifdef KL_PRINT_SCHEDULE
        print_best_schedule(0);
#endif

        unsigned noImprovementIterCounter = 0;

        auto startTime = std::chrono::high_resolution_clock::now();

        SelectNodesCheckRemoveSuperstep();

        UpdateRewardPenalty();

        InitializeGainHeap(nodeSelection_);

        for (unsigned outerCounter = 0; outerCounter < parameters_.maxOuterIterations; outerCounter++) {
#ifdef KL_DEBUG
            std::cout << "outer iteration " << outer_counter << " current costs: " << current_schedule.current_cost << std::endl;
            if (max_gain_heap.size() == 0) {
                std::cout << "max gain heap empty" << std::endl;
            }
#endif

            // unsigned conseq_no_gain_moves_counter = 0;

            unsigned failedBranches = 0;
            double bestIterCosts = currentSchedule_.current_cost;

            VertexType nodeCausingFirstViolation = 0;

            unsigned innerCounter = 0;

            while (failedBranches < parameters_.maxNumFailedBranches && innerCounter < parameters_.maxInnerIterations
                   && maxGainHeap_.size() > 0) {
                innerCounter++;

                const bool iterFeasible = currentSchedule_.current_feasible;
                const double iterCosts = currentSchedule_.current_cost;
#ifdef KL_DEBUG
                print_heap();
#endif
                KlMove<GraphT> bestMove = FindBestMove();    // O(log n)

                if (bestMove.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                    std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                          << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                          << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                          << " violations: " << current_schedule.current_violations.size() << " old cost "
                          << current_schedule.current_cost << " new cost "
                          << current_schedule.current_cost + best_move.change_in_cost << std::endl;

                if constexpr (current_schedule.use_memory_constraint) {
                    std::cout << "memory to step/proc "
                              << current_schedule.memory_constraint.step_processor_memory[best_move.to_step][best_move.to_proc]
                              << std::endl;
                }

                printSetScheduleWorkMemNodesGrid(std::cout, current_schedule.set_schedule, true);
#endif

                currentSchedule_.apply_move(bestMove);    // O(p + log n)

                //             if (best_move.gain <= 0.000000001) {
                //                 conseq_no_gain_moves_counter++;

                //                 if (conseq_no_gain_moves_counter > 15) {

                //                     conseq_no_gain_moves_counter = 0;
                //                     parameters.initial_penalty = 0.0;
                //                     parameters.violations_threshold = 3;
                // #ifdef KL_DEBUG
                //                     std::cout << "more than 15 moves with gain <= 0, set " <<
                //                     parameters.initial_penalty
                //                               << " violations threshold " << parameters.violations_threshold <<
                //                               std::endl;
                // #endif
                //                 }

                //             } else {
                //                 conseq_no_gain_moves_counter = 0;
                //             }

#ifdef KL_DEBUG
                BspSchedule<Graph_t> tmp_schedule(current_schedule.set_schedule);
                if (not tmp_schedule.satisfiesMemoryConstraints()) {
                    std::cout << "Mem const violated" << std::endl;
                }
#endif

                UpdateRewardPenalty();
                lockedNodes_.insert(bestMove.node);

#ifdef KL_DEBUG
                double tmp_costs = current_schedule.current_cost;
                if (tmp_costs != compute_current_costs()) {
                    std::cout << "current costs: " << current_schedule.current_cost << " best move gain: " << best_move.gain
                              << " best move costs: " << best_move.change_in_cost << " tmp cost: " << tmp_costs << std::endl;

                    std::cout << "! costs not equal " << std::endl;
                }
#endif

                if (iterFeasible != currentSchedule_.current_feasible) {
                    if (iterFeasible) {
#ifdef KL_DEBUG
                        std::cout << "===> current schedule changed from feasible to infeasible" << std::endl;
#endif

                        nodeCausingFirstViolation = bestMove.node;

                        if (iterCosts < bestScheduleCosts_) {
#ifdef KL_DEBUG
                            std::cout << "save best schedule with costs " << iter_costs << std::endl;
#endif
                            bestScheduleCosts_ = iterCosts;
                            SaveBestSchedule(currentSchedule_.vector_schedule);    // O(n)
                            ReverseMoveBestSchedule(bestMove);
#ifdef KL_DEBUG
                            std::cout << "KLBase save best schedule with (source node comm) cost "
                                      << best_schedule->computeTotalCosts() << " and number of supersteps "
                                      << best_schedule->numberOfSupersteps() << std::endl;
#endif
                        }

                    } else {
#ifdef KL_DEBUG
                        std::cout << "===> current schedule changed from infeasible to feasible" << std::endl;
#endif
                    }
                } else if (bestMove.change_in_cost > 0 && currentSchedule_.current_feasible) {
                    if (iterCosts < bestScheduleCosts_) {
#ifdef KL_DEBUG
                        std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                        bestScheduleCosts_ = iterCosts;
                        SaveBestSchedule(currentSchedule_.vector_schedule);    // O(n)
                        ReverseMoveBestSchedule(bestMove);
#ifdef KL_DEBUG
                        std::cout << "KLBase save best schedule with (source node comm) cost "
                                  << best_schedule->computeTotalCosts() << " and number of supersteps "
                                  << best_schedule->numberOfSupersteps() << std::endl;
#endif
                    }
                }

#ifdef KL_DEBUG
                std::cout << "Node selection: [";
                for (auto it = node_selection.begin(); it != node_selection.end(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << "]" << std::endl;

                std::cout << "Locked nodes: [";
                for (auto it = locked_nodes.begin(); it != locked_nodes.end(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << "]" << std::endl;

                std::cout << "Super locked nodes: [";
                for (auto it = super_locked_nodes.begin(); it != super_locked_nodes.end(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << "]" << std::endl;

#endif

                ComputeNodesToUpdate(bestMove);

                SelectUnlockNeighbors(bestMove.node);

                if (CheckViolationLocked()) {
                    if (iterFeasible != currentSchedule_.current_feasible && iterFeasible) {
                        nodeCausingFirstViolation = bestMove.node;
                    }
                    superLockedNodes_.insert(nodeCausingFirstViolation);
#ifdef KL_DEBUG
                    std::cout << "abort iteration on locked violation, super locking node " << node_causing_first_violation
                              << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                std::cout << "Nodes to update: [";
                for (auto it = nodes_to_update.begin(); it != nodes_to_update.end(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << "]" << std::endl;
#endif

                UpdateNodeGains(nodesToUpdate_);

                if (not(currentSchedule_.current_violations.size() > 4) && not iterFeasible && not maxGainHeap_.empty()) {
                    const auto &iter = maxGainHeap_.ordered_begin();
                    if (iter->gain < parameters_.gainThreshold) {
                        nodeSelection_.clear();
                        lockedNodes_.clear();
                        superLockedNodes_.clear();
                        SelectNodesViolations();

                        UpdateRewardPenalty();

                        InitializeGainHeap(nodeSelection_);

#ifdef KL_DEBUG
                        std::cout << "max gain below gain threshold" << std::endl;
#endif
                    }
                }

                if (currentSchedule_.current_cost
                    > (parameters_.maxDivBestSolBasePercent + outerCounter * parameters_.maxDivBestSolRatePercent)
                          * bestScheduleCosts_) {
#ifdef KL_DEBUG
                    std::cout << "current cost " << current_schedule.current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs << " rollback to best schedule"
                              << std::endl;
#endif

                    currentSchedule_.set_current_schedule(*bestSchedule_);

                    SetInitialRewardPenalty();
                    InitializeGainHeapUnlockedNodes(nodeSelection_);

#ifdef KL_DEBUG
                    std::cout << "new current cost " << current_schedule.current_cost << std::endl;
#endif

                    failedBranches++;
                }

            }    // while

#ifdef KL_DEBUG
            std::cout << std::setprecision(12) << "end inner loop current cost: " << current_schedule.current_cost << " with "
                      << current_schedule.current_violations.size() << " violation, best sol cost: " << best_schedule_costs
                      << " with " << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                      << parameters.max_outer_iterations << std::endl;
#endif

            if (currentSchedule_.current_feasible) {
                if (currentSchedule_.current_cost <= bestScheduleCosts_) {
                    SaveBestSchedule(currentSchedule_.vector_schedule);
                    bestScheduleCosts_ = currentSchedule_.current_cost;
#ifdef KL_DEBUG
                    std::cout << "KLBase save best schedule with (source node comm) cost " << best_schedule->computeTotalCosts()
                              << " and number of supersteps " << best_schedule->numberOfSupersteps() << std::endl;
#endif
                } else {
                    currentSchedule_.set_current_schedule(*bestSchedule_);
                }
            } else {
                currentSchedule_.set_current_schedule(*bestSchedule_);
            }

            if (computeWithTimeLimit_) {
                auto finishTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();
                if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds) {
                    break;
                }
            }

            if (outerCounter > 0 && outerCounter % 30 == 0) {
                superLockedNodes_.clear();
#ifdef KL_DEBUG
                std::cout << "---- reset super locked nodes" << std::endl;
#endif
            }

#ifdef KL_PRINT_SCHEDULE
            if (best_iter_costs > current_schedule.current_cost) {
                print_best_schedule(outer_counter + 1);
            }
#endif

            ResetLockedNodes();

            nodeSelection_.clear();

            // if (reset_superstep) {
            //     select_nodes_check_reset_superstep();
            // } else {
            SelectNodesCheckRemoveSuperstep();
            // }

            UpdateRewardPenalty();

            InitializeGainHeap(nodeSelection_);

#ifdef KL_DEBUG
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

            if (bestIterCosts <= currentSchedule_.current_cost) {
                noImprovementIterCounter++;

                if (noImprovementIterCounter > parameters_.resetEpochCounterThreshold) {
                    stepSelectionEpochCounter_ = 0;
                    parameters_.resetEpochCounterThreshold += currentSchedule_.num_steps();
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << no_improvement_iter_counter
                              << " iterations, reset epoc counter. Increase reset threshold to "
                              << parameters.reset_epoch_counter_threshold << std::endl;
#endif
                }

                //             if (no_improvement_iter_counter > 10 && no_improvement_iter_counter % 15 == 0) {

                //                 step_selection_epoch_counter = 0;

                //                 if (alternate_reset_remove_superstep) {
                //                     reset_superstep = !reset_superstep;
                //                 }

                // #ifdef KL_DEBUG
                //                 std::cout << "no improvement for " << no_improvement_iter_counter << " reset
                //                 superstep "
                //                           << reset_superstep << std::endl;
                // #endif
                //             }

                if (noImprovementIterCounter > 50 && noImprovementIterCounter % 3 == 0) {
                    parameters_.initialPenalty = 0.0;
                    parameters_.violationsThreshold = 5;

                } else if (noImprovementIterCounter > 30 && noImprovementIterCounter % 5 == 0) {
                    parameters_.initialPenalty = 0.0;
                    parameters_.violationsThreshold = 4;

                } else if (noImprovementIterCounter > 9 && noImprovementIterCounter % 10 == 0) {
                    parameters_.initialPenalty = 0.0;
                    parameters_.violationsThreshold = 3;
#ifdef KL_DEBUG
                    std::cout << "---- reset initial penalty " << parameters.initial_penalty << " violations threshold "
                              << parameters.violations_threshold << std::endl;
#endif
                }

                if (noImprovementIterCounter == 35) {
                    parameters_.maxDivBestSolBasePercent *= 1.02;
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << no_improvement_iter_counter
                              << " iterations, increase max_div_best_sol_base_percent to "
                              << parameters.max_div_best_sol_base_percent << std::endl;
#endif
                }

                if (noImprovementIterCounter >= parameters_.maxNoImprovementIterations) {
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << parameters.max_no_improvement_iterations
                              << " iterations, end local search" << std::endl;
#endif
                    break;
                }
            } else {
                noImprovementIterCounter = 0;
            }

#ifdef KL_DEBUG
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

        }    // for

        CleanupDatastructures();

#ifdef KL_DEBUG_1
        std::cout << "kl done, current cost " << best_schedule_costs << " with " << best_schedule->numberOfSupersteps()
                  << " supersteps vs " << initial_costs << " initial costs" << std::endl;
        assert(best_schedule->satisfiesPrecedenceConstraints());
#endif

        if (initialCosts > currentSchedule_.current_cost) {
            return true;
        } else {
            return false;
        }
    }

    // virtual void checkMergeSupersteps();
    // virtual void checkInsertSuperstep();

    // virtual void insertSuperstep(unsigned step);

    void PrintHeap() {
        std::cout << "heap current size: " << maxGainHeap_.size() << std::endl;
        std::cout << "heap top node " << maxGainHeap_.top().node << " gain " << maxGainHeap_.top().gain << std::endl;

        unsigned count = 0;
        for (auto it = maxGainHeap_.ordered_begin(); it != maxGainHeap_.ordered_end(); ++it) {
            std::cout << "node " << it->node << " gain " << it->gain << " to proc " << it->to_proc << " to step " << it->to_step
                      << std::endl;

            if (count++ > 15 || it->gain <= 0.0) {
                break;
            }
        }
    }

    bool computeWithTimeLimit_ = false;

#ifdef KL_PRINT_SCHEDULE
    std::string file_name_write_schedule = "kl_schedule_iter_";
    void print_best_schedule(unsigned iteration);
#endif

  public:
    KlBase(KlCurrentSchedule<GraphT, MemoryConstraintT> &currentSchedule)
        : ImprovementScheduler<GraphT>(), currentSchedule_(currentSchedule) {
        std::random_device rd;
        gen_ = std::mt19937(rd());
    }

    virtual ~KlBase() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule<GraphT> &schedule) override {
        ResetRunDatastructures();

        bestSchedule_ = &schedule;
        currentSchedule_.instance = &bestSchedule_->getInstance();

        numNodes_ = currentSchedule_.instance->numberOfVertices();
        numProcs_ = currentSchedule_.instance->numberOfProcessors();

        SetParameters();
        InitializeDatastructures();

        bool improvementFound = RunLocalSearchUnlockDelay();

        if (improvementFound) {
            return RETURN_STATUS::OSP_SUCCESS;
        } else {
            return RETURN_STATUS::BEST_FOUND;
        }
    }

    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule<GraphT> &schedule) override {
        computeWithTimeLimit_ = true;
        return improveSchedule(schedule);
    }

    virtual void SetComputeWithTimeLimit(bool computeWithTimeLimit) { computeWithTimeLimit_ = computeWithTimeLimit; }

    virtual std::string GetScheduleName() const = 0;

    virtual void SetQuickPass(bool quickPass) { parameters_.quickPass = quickPass; }

    virtual void SetAlternateResetRemoveSuperstep(bool alternateResetRemoveSuperstep) {
        autoAlternate_ = false;
        alternateResetRemoveSuperstep_ = alternateResetRemoveSuperstep;
    }
};

}    // namespace osp
