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

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

template <typename GraphT>
class HillClimbingScheduler : public ImprovementScheduler<GraphT> {
    static_assert(IsDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");
    static_assert(IsComputationalDagV<GraphT>, "GraphT must satisfy the computational_dag concept");

    using VertexIdx = VertexIdxT<GraphT>;
    using CostType = VWorkwT<GraphT>;

    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "HillClimbing requires work and comm. weights to have the same type.");

  public:
    enum Direction { EARLIER = 0, AT, LATER };

    static const int NumDirections = 3;

    // aux structure for efficiently storing the changes incurred by a potential HC step
    struct StepAuxData {
        CostType newCost;
        std::map<std::pair<unsigned, unsigned>, int> sentChange, recChange;
        bool canShrink = false;
    };

  private:
    BspSchedule<GraphT> *schedule;
    CostType cost = 0;

    // Main parameters for runnign algorithm
    bool shrink = true;
    bool steepestAscent = false;

    // aux data structures
    std::vector<std::vector<std::list<VertexIdx>>> supsteplists;
    std::vector<std::vector<std::vector<bool>>> canMove;
    std::vector<std::list<std::pair<VertexIdx, unsigned>>> moveOptions;
    std::vector<std::vector<std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>>> movePointer;
    std::vector<std::vector<std::map<unsigned, unsigned>>> succSteps;
    std::vector<std::vector<CostType>> workCost, sent, received, commCost;
    std::vector<std::set<std::pair<CostType, unsigned>>> workCostList, commCostList;
    std::vector<std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>> workCostPointer, commCostPointer;
    std::vector<typename std::list<VertexIdx>::iterator> supStepListPointer;
    std::pair<int, typename std::list<std::pair<VertexIdx, unsigned>>::iterator> nextMove;
    bool HCwithLatency = true;

    // for improved candidate selection
    std::deque<std::tuple<VertexIdx, unsigned, int>> promisingMoves;
    bool findPromisingMoves = true;

    // Initialize data structures (based on current schedule)
    void Init();
    void updatePromisingMoves();

    // Functions to compute and update the std::list of possible moves
    void updateNodeMovesEarlier(VertexIdx node);
    void updateNodeMovesAt(VertexIdx node);
    void updateNodeMovesLater(VertexIdx node);
    void updateNodeMoves(VertexIdx node);
    void updateMoveOptions(VertexIdx node, int where);

    void addMoveOption(VertexIdx node, unsigned p, Direction dir);

    void eraseMoveOption(VertexIdx node, unsigned p, Direction dir);
    void eraseMoveOptionsEarlier(VertexIdx node);
    void eraseMoveOptionsAt(VertexIdx node);
    void eraseMoveOptionsLater(VertexIdx node);
    void eraseMoveOptions(VertexIdx node);

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupstepLists();

    // For memory constraints
    bool use_memory_constraint = false;
    std::vector<std::vector<VMemwT<GraphT>>> memory_used;
    bool violatesMemConstraint(VertexIdx node, unsigned processor, int where);

    // Compute the cost change incurred by a potential move
    int moveCostChange(VertexIdx node, unsigned p, int where, StepAuxData &changing);

    // Execute a chosen move, updating the schedule and the data structures
    void executeMove(VertexIdx node, unsigned newProc, int where, const StepAuxData &changing);

    // Single hill climbing step
    bool Improve();

  public:
    HillClimbingScheduler() : ImprovementScheduler<GraphT>() {}

    virtual ~HillClimbingScheduler() = default;

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &input_schedule) override;

    // call with time/step limits
    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &input_schedule) override;
    virtual ReturnStatus ImproveScheduleWithStepLimit(BspSchedule<GraphT> &input_schedule, const unsigned stepLimit = 10);

    // setting parameters
    void SetSteepestAscend(bool steepestAscent_) { steepestAscent = steepestAscent_; }

    void SetShrink(bool shrink_) { shrink = shrink_; }

    virtual std::string GetScheduleName() const override { return "HillClimbing"; }
};

template <typename GraphT>
ReturnStatus HillClimbingScheduler<GraphT>::ImproveSchedule(BspSchedule<GraphT> &input_schedule) {
    ImprovementScheduler<GraphT>::SetTimeLimitSeconds(600U);
    return ImproveScheduleWithTimeLimit(input_schedule);
}

// Main method for hill climbing (with time limit)
template <typename GraphT>
ReturnStatus HillClimbingScheduler<GraphT>::ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &input_schedule) {
    schedule = &input_schedule;

    CreateSupstepLists();
    Init();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    int counter = 0;
    while (Improve()) {
        if ((++counter) == 10) {
            counter = 0;
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (elapsed >= ImprovementScheduler<GraphT>::timeLimitSeconds_) {
                std::cout << "Hill Climbing was shut down due to time limit." << std::endl;
                break;
            }
        }
    }

    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT>
// Hill climbing with step limit (designed as an ingredient for multilevel algorithms, no safety checks)
ReturnStatus HillClimbingScheduler<GraphT>::ImproveScheduleWithStepLimit(BspSchedule<GraphT> &input_schedule,
                                                                         const unsigned stepLimit) {
    schedule = &input_schedule;

    CreateSupstepLists();
    Init();
    for (unsigned step = 0; step < stepLimit; ++step) {
        if (!Improve()) {
            break;
        }
    }

    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::Init() {
    if (shrink) {
        schedule->ShrinkByMergingSupersteps();
        CreateSupstepLists();
    }

    const VertexIdx N = schedule->GetInstance().GetComputationalDag().NumVertices();
    const unsigned P = schedule->GetInstance().GetArchitecture().NumberOfProcessors();
    const unsigned M = schedule->NumberOfSupersteps();
    const GraphT &G = schedule->GetInstance().GetComputationalDag();

    // Movement options
    canMove.clear();
    canMove.resize(NumDirections, std::vector<std::vector<bool>>(N, std::vector<bool>(P, false)));
    moveOptions.clear();
    moveOptions.resize(NumDirections);
    movePointer.clear();
    movePointer.resize(NumDirections,
                       std::vector<std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>>(
                           N, std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>(P)));

    // Value use lists
    succSteps.clear();
    succSteps.resize(N, std::vector<std::map<unsigned, unsigned>>(P));
    for (VertexIdx node = 0; node < N; ++node) {
        for (const VertexIdx &succ : G.Children(node)) {
            if (succSteps[node][schedule->AssignedProcessor(succ)].find(schedule->AssignedSuperstep(succ))
                == succSteps[node][schedule->AssignedProcessor(succ)].end()) {
                succSteps[node][schedule->AssignedProcessor(succ)].insert({schedule->AssignedSuperstep(succ), 1U});
            } else {
                succSteps[node][schedule->AssignedProcessor(succ)].at(schedule->AssignedSuperstep(succ)) += 1;
            }
        }
    }

    // Cost data
    workCost.clear();
    workCost.resize(M, std::vector<CostType>(P, 0));
    sent.clear();
    sent.resize(M - 1, std::vector<CostType>(P, 0));
    received.clear();
    received.resize(M - 1, std::vector<CostType>(P, 0));
    commCost.clear();
    commCost.resize(M - 1, std::vector<CostType>(P));

    workCostList.clear();
    workCostList.resize(M);
    commCostList.clear();
    commCostList.resize(M - 1);
    workCostPointer.clear();
    workCostPointer.resize(M, std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>(P));
    commCostPointer.clear();
    commCostPointer.resize(M - 1, std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>(P));

    // Supstep std::list pointers
    supStepListPointer.clear();
    supStepListPointer.resize(N);
    for (unsigned step = 0; step < M; ++step) {
        for (unsigned proc = 0; proc < P; ++proc) {
            for (auto it = supsteplists[step][proc].begin(); it != supsteplists[step][proc].end(); ++it) {
                supStepListPointer[*it] = it;
            }
        }
    }

    // Compute movement options
    for (VertexIdx node = 0; node < N; ++node) {
        updateNodeMoves(node);
    }

    nextMove.first = 0;
    nextMove.second = moveOptions[0].begin();

    // Compute cost data
    std::vector<CostType> work_cost(M, 0);
    for (unsigned step = 0; step < M; ++step) {
        for (unsigned proc = 0; proc < P; ++proc) {
            for (const VertexIdx node : supsteplists[step][proc]) {
                workCost[step][proc] += schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node);
            }

            std::pair<CostType, unsigned> entry(workCost[step][proc], proc);
            workCostPointer[step][proc] = workCostList[step].insert(entry).first;
        }
        work_cost[step] = (--workCostList[step].end())->first;
    }

    cost = work_cost[0];
    std::vector<std::vector<bool>> present(N, std::vector<bool>(P, false));
    for (unsigned step = 0; step < M - schedule->GetStaleness(); ++step) {
        for (unsigned proc = 0; proc < P; ++proc) {
            for (const VertexIdx node : supsteplists[step + schedule->GetStaleness()][proc]) {
                for (const VertexIdx &pred : G.Parents(node)) {
                    if (schedule->AssignedProcessor(node) != schedule->AssignedProcessor(pred)
                        && !present[pred][schedule->AssignedProcessor(node)]) {
                        present[pred][schedule->AssignedProcessor(node)] = true;
                        sent[step][schedule->AssignedProcessor(pred)]
                            += schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                               * schedule->GetInstance().GetArchitecture().SendCosts(schedule->AssignedProcessor(pred),
                                                                                     schedule->AssignedProcessor(node));
                        received[step][schedule->AssignedProcessor(node)]
                            += schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                               * schedule->GetInstance().GetArchitecture().SendCosts(schedule->AssignedProcessor(pred),
                                                                                     schedule->AssignedProcessor(node));
                    }
                }
            }
        }
    }

    for (unsigned step = 0; step < M - 1; ++step) {
        for (unsigned proc = 0; proc < P; ++proc) {
            commCost[step][proc] = std::max(sent[step][proc], received[step][proc]);
            std::pair<CostType, unsigned> entry(commCost[step][proc], proc);
            commCostPointer[step][proc] = commCostList[step].insert(entry).first;
        }
        CostType comm_cost = schedule->GetInstance().GetArchitecture().CommunicationCosts() * commCostList[step].rbegin()->first;
        CostType sync_cost = (comm_cost > 0) ? schedule->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        if (schedule->GetStaleness() == 1) {
            cost += comm_cost + work_cost[step + 1] + sync_cost;
        } else {
            cost += std::max(comm_cost, work_cost[step + 1]) + sync_cost;
        }
    }

    updatePromisingMoves();

    // memory_constraints
    if (use_memory_constraint) {
        memory_used.clear();
        memory_used.resize(P, std::vector<VMemwT<GraphT>>(M, 0));
        for (VertexIdx node = 0; node < N; ++node) {
            memory_used[schedule->AssignedProcessor(node)][schedule->AssignedSuperstep(node)]
                += schedule->GetInstance().GetComputationalDag().VertexMemWeight(node);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::updatePromisingMoves() {
    if (!findPromisingMoves) {
        return;
    }

    const unsigned P = schedule->GetInstance().GetArchitecture().NumberOfProcessors();
    const GraphT &G = schedule->GetInstance().GetComputationalDag();

    promisingMoves.clear();
    for (VertexIdx node = 0; node < schedule->GetInstance().GetComputationalDag().NumVertices(); ++node) {
        std::vector<unsigned> nrPredOnProc(P, 0);
        for (const VertexIdx &pred : G.Parents(node)) {
            ++nrPredOnProc[schedule->AssignedProcessor(pred)];
        }

        unsigned otherProcUsed = 0;
        for (unsigned proc = 0; proc < P; ++proc) {
            if (schedule->AssignedProcessor(node) != proc && nrPredOnProc[proc] > 0) {
                ++otherProcUsed;
            }
        }

        if (otherProcUsed == 1) {
            for (unsigned proc = 0; proc < P; ++proc) {
                if (schedule->AssignedProcessor(node) != proc && nrPredOnProc[proc] > 0
                    && schedule->GetInstance().IsCompatible(node, proc)) {
                    promisingMoves.push_back(std::make_tuple(node, proc, EARLIER));
                    promisingMoves.push_back(std::make_tuple(node, proc, AT));
                    promisingMoves.push_back(std::make_tuple(node, proc, LATER));
                }
            }
        }

        std::vector<unsigned> nrSuccOnProc(P, 0);
        for (const VertexIdx &succ : G.Children(node)) {
            ++nrSuccOnProc[schedule->AssignedProcessor(succ)];
        }

        otherProcUsed = 0;
        for (unsigned proc = 0; proc < P; ++proc) {
            if (schedule->AssignedProcessor(node) != proc && nrSuccOnProc[proc] > 0) {
                ++otherProcUsed;
            }
        }

        if (otherProcUsed == 1) {
            for (unsigned proc = 0; proc < P; ++proc) {
                if (schedule->AssignedProcessor(node) != proc && nrSuccOnProc[proc] > 0
                    && schedule->GetInstance().IsCompatible(node, proc)) {
                    promisingMoves.push_back(std::make_tuple(node, proc, EARLIER));
                    promisingMoves.push_back(std::make_tuple(node, proc, AT));
                    promisingMoves.push_back(std::make_tuple(node, proc, LATER));
                }
            }
        }
    }

    for (unsigned step = 0; step < schedule->NumberOfSupersteps(); ++step) {
        std::list<unsigned> minProcs, maxProcs;
        CostType minWork = std::numeric_limits<CostType>::max(), maxWork = std::numeric_limits<CostType>::min();
        for (unsigned proc = 0; proc < P; ++proc) {
            if (workCost[step][proc] > maxWork) {
                maxWork = workCost[step][proc];
            }
            if (workCost[step][proc] < minWork) {
                minWork = workCost[step][proc];
            }
        }
        for (unsigned proc = 0; proc < P; ++proc) {
            if (workCost[step][proc] == minWork) {
                minProcs.push_back(proc);
            }
            if (workCost[step][proc] == maxWork) {
                maxProcs.push_back(proc);
            }
        }
        for (unsigned to : minProcs) {
            for (unsigned from : maxProcs) {
                for (VertexIdx node : supsteplists[step][from]) {
                    if (schedule->GetInstance().IsCompatible(node, to)) {
                        promisingMoves.push_back(std::make_tuple(node, to, AT));
                    }
                }
            }
        }
    }
}

// Functions to compute and update the std::list of possible moves
template <typename GraphT>
void HillClimbingScheduler<GraphT>::updateNodeMovesEarlier(const VertexIdx node) {
    if (schedule->AssignedSuperstep(node) == 0) {
        return;
    }

    std::set<unsigned> predProc;
    for (const VertexIdx &pred : schedule->GetInstance().GetComputationalDag().Parents(node)) {
        if (schedule->AssignedSuperstep(pred) == schedule->AssignedSuperstep(node)) {
            return;
        }
        if (static_cast<int>(schedule->AssignedSuperstep(pred))
            >= static_cast<int>(schedule->AssignedSuperstep(node)) - static_cast<int>(schedule->GetStaleness())) {
            predProc.insert(schedule->AssignedProcessor(pred));
        }
    }
    if (schedule->GetStaleness() == 2) {
        for (const VertexIdx &succ : schedule->GetInstance().GetComputationalDag().Children(node)) {
            if (schedule->AssignedSuperstep(succ) == schedule->AssignedSuperstep(node)) {
                predProc.insert(schedule->AssignedProcessor(succ));
            }
        }
    }

    if (predProc.size() > 1) {
        return;
    }

    if (predProc.size() == 1) {
        addMoveOption(node, *predProc.begin(), EARLIER);
    } else {
        for (unsigned proc = 0; proc < schedule->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
            addMoveOption(node, proc, EARLIER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::updateNodeMovesAt(const VertexIdx node) {
    for (const VertexIdx &pred : schedule->GetInstance().GetComputationalDag().Parents(node)) {
        if (static_cast<int>(schedule->AssignedSuperstep(pred))
            >= static_cast<int>(schedule->AssignedSuperstep(node)) - static_cast<int>(schedule->GetStaleness()) + 1) {
            return;
        }
    }

    for (const VertexIdx &succ : schedule->GetInstance().GetComputationalDag().Children(node)) {
        if (schedule->AssignedSuperstep(succ) <= schedule->AssignedSuperstep(node) + schedule->GetStaleness() - 1) {
            return;
        }
    }

    for (unsigned proc = 0; proc < schedule->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (proc != schedule->AssignedProcessor(node)) {
            addMoveOption(node, proc, AT);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::updateNodeMovesLater(const VertexIdx node) {
    if (schedule->AssignedSuperstep(node) == schedule->NumberOfSupersteps() - 1) {
        return;
    }

    std::set<unsigned> succProc;
    for (const VertexIdx &succ : schedule->GetInstance().GetComputationalDag().Children(node)) {
        if (schedule->AssignedSuperstep(succ) == schedule->AssignedSuperstep(node)) {
            return;
        }
        if (schedule->AssignedSuperstep(succ) <= schedule->AssignedSuperstep(node) + schedule->GetStaleness()) {
            succProc.insert(schedule->AssignedProcessor(succ));
        }
    }
    if (schedule->GetStaleness() == 2) {
        for (const VertexIdx &pred : schedule->GetInstance().GetComputationalDag().Parents(node)) {
            if (schedule->AssignedSuperstep(pred) == schedule->AssignedSuperstep(node)) {
                succProc.insert(schedule->AssignedProcessor(pred));
            }
        }
    }

    if (succProc.size() > 1) {
        return;
    }

    if (succProc.size() == 1) {
        addMoveOption(node, *succProc.begin(), LATER);
    } else {
        for (unsigned proc = 0; proc < schedule->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
            addMoveOption(node, proc, LATER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::updateNodeMoves(const VertexIdx node) {
    eraseMoveOptions(node);
    updateNodeMovesEarlier(node);
    updateNodeMovesAt(node);
    updateNodeMovesLater(node);
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::updateMoveOptions(VertexIdx node, int where) {
    const GraphT &G = schedule->GetInstance().GetComputationalDag();

    updateNodeMoves(node);
    if (where == 0) {
        for (const VertexIdx &pred : G.Parents(node)) {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
        }
        for (const VertexIdx &succ : G.Children(node)) {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
        }
    }
    if (where == -1) {
        for (const VertexIdx &pred : G.Parents(node)) {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
            eraseMoveOptionsAt(pred);
            updateNodeMovesAt(pred);
            if (schedule->GetStaleness() == 2) {
                eraseMoveOptionsEarlier(pred);
                updateNodeMovesEarlier(pred);
            }
        }
        for (const VertexIdx &succ : G.Children(node)) {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
            if (schedule->GetStaleness() == 2) {
                eraseMoveOptionsAt(succ);
                updateNodeMovesAt(succ);
            }
        }
    }
    if (where == 1) {
        for (const VertexIdx &pred : G.Parents(node)) {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
            if (schedule->GetStaleness() == 2) {
                eraseMoveOptionsAt(pred);
                updateNodeMovesAt(pred);
            }
        }
        for (const VertexIdx &succ : G.Children(node)) {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
            eraseMoveOptionsAt(succ);
            updateNodeMovesAt(succ);
            if (schedule->GetStaleness() == 2) {
                eraseMoveOptionsLater(succ);
                updateNodeMovesLater(succ);
            }
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::addMoveOption(const VertexIdx node, const unsigned p, const Direction dir) {
    if (!canMove[dir][node][p] && schedule->GetInstance().IsCompatible(node, p)) {
        canMove[dir][node][p] = true;
        moveOptions[dir].emplace_back(node, p);
        movePointer[dir][node][p] = --moveOptions[dir].end();
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::eraseMoveOption(VertexIdx node, unsigned p, Direction dir) {
    canMove[dir][node][p] = false;
    if (nextMove.first == dir && nextMove.second->first == node && nextMove.second->second == p) {
        ++nextMove.second;
    }
    moveOptions[dir].erase(movePointer[dir][node][p]);
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::eraseMoveOptionsEarlier(VertexIdx node) {
    for (unsigned proc = 0; proc < schedule->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove[EARLIER][node][proc]) {
            eraseMoveOption(node, proc, EARLIER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::eraseMoveOptionsAt(VertexIdx node) {
    for (unsigned proc = 0; proc < schedule->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove[AT][node][proc]) {
            eraseMoveOption(node, proc, AT);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::eraseMoveOptionsLater(VertexIdx node) {
    for (unsigned proc = 0; proc < schedule->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove[LATER][node][proc]) {
            eraseMoveOption(node, proc, LATER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::eraseMoveOptions(VertexIdx node) {
    eraseMoveOptionsEarlier(node);
    eraseMoveOptionsAt(node);
    eraseMoveOptionsLater(node);
}

// Compute the cost change incurred by a potential move
template <typename GraphT>
int HillClimbingScheduler<GraphT>::moveCostChange(const VertexIdx node, unsigned p, const int where, StepAuxData &changing) {
    const unsigned step = schedule->AssignedSuperstep(node);
    const unsigned new_step = static_cast<unsigned>(static_cast<int>(step) + where);
    unsigned oldProc = schedule->AssignedProcessor(node);
    int change = 0;

    const GraphT &G = schedule->GetInstance().GetComputationalDag();

    std::set<unsigned> affectedSteps;

    // Work cost change
    std::map<unsigned, CostType> newWorkCost;
    const auto itBest = --workCostList[step].end();
    CostType maxAfterRemoval = itBest->first;
    if (itBest->second == oldProc) {
        auto itNext = itBest;
        --itNext;
        maxAfterRemoval
            = std::max(itBest->first - schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node), itNext->first);
        if (itBest->first != maxAfterRemoval) {
            if (step == 0 || schedule->GetStaleness() == 1) {    // incorporate immediately into cost change
                change -= static_cast<int>(itBest->first) - static_cast<int>(maxAfterRemoval);
            } else {
                newWorkCost[step] = maxAfterRemoval;
                affectedSteps.insert(step - 1);
            }
        }
    }

    const CostType maxBeforeAddition = (where == 0) ? maxAfterRemoval : workCostList[new_step].rbegin()->first;
    if (workCost[new_step][p] + schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node) > maxBeforeAddition) {
        if (new_step == 0 || schedule->GetStaleness() == 1) {    // incorporate immediately into cost change
            change += static_cast<int>(workCost[new_step][p] + schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node))
                      - static_cast<int>(maxBeforeAddition);
        } else {
            newWorkCost[new_step] = workCost[new_step][p] + schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node);
            affectedSteps.insert(new_step - 1);
        }
    }

    // Comm cost change
    std::list<std::tuple<unsigned, unsigned, int>> sentInc, recInc;
    //  -outputs
    if (p != oldProc) {
        for (unsigned j = 0; j < schedule->GetInstance().GetArchitecture().NumberOfProcessors(); ++j) {
            if (succSteps[node][j].empty()) {
                continue;
            }

            unsigned affectedStep = succSteps[node][j].begin()->first - schedule->GetStaleness();
            if (j == p) {
                sentInc.emplace_back(affectedStep,
                                     oldProc,
                                     -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                       * schedule->GetInstance().GetArchitecture().SendCosts(oldProc, j)));
                recInc.emplace_back(affectedStep,
                                    p,
                                    -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule->GetInstance().GetArchitecture().SendCosts(oldProc, j)));
            } else if (j == oldProc) {
                recInc.emplace_back(affectedStep,
                                    oldProc,
                                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                     * schedule->GetInstance().GetArchitecture().SendCosts(p, j)));
                sentInc.emplace_back(affectedStep,
                                     p,
                                     static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule->GetInstance().GetArchitecture().SendCosts(p, j)));
            } else {
                sentInc.emplace_back(affectedStep,
                                     oldProc,
                                     -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                       * schedule->GetInstance().GetArchitecture().SendCosts(oldProc, j)));
                recInc.emplace_back(affectedStep,
                                    j,
                                    -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule->GetInstance().GetArchitecture().SendCosts(oldProc, j)));
                sentInc.emplace_back(affectedStep,
                                     p,
                                     static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule->GetInstance().GetArchitecture().SendCosts(p, j)));
                recInc.emplace_back(affectedStep,
                                    j,
                                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                     * schedule->GetInstance().GetArchitecture().SendCosts(p, j)));
            }
        }
    }

    //  -inputs
    if (p == oldProc) {
        for (const VertexIdx &pred : G.Parents(node)) {
            if (schedule->AssignedProcessor(pred) == p) {
                continue;
            }

            const auto firstUse = *succSteps[pred][p].begin();
            const bool skip = firstUse.first < step || (firstUse.first == step && where >= 0 && firstUse.second > 1);
            if (!skip) {
                sentInc.emplace_back(step - schedule->GetStaleness(),
                                     schedule->AssignedProcessor(pred),
                                     -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                       * schedule->GetInstance().GetArchitecture().SendCosts(
                                                           schedule->AssignedProcessor(pred), p)));
                recInc.emplace_back(step - schedule->GetStaleness(),
                                    p,
                                    -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule->GetInstance().GetArchitecture().SendCosts(
                                                          schedule->AssignedProcessor(pred), p)));
                sentInc.emplace_back(
                    new_step - schedule->GetStaleness(),
                    schedule->AssignedProcessor(pred),
                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                     * schedule->GetInstance().GetArchitecture().SendCosts(schedule->AssignedProcessor(pred), p)));
                recInc.emplace_back(
                    new_step - schedule->GetStaleness(),
                    p,
                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                     * schedule->GetInstance().GetArchitecture().SendCosts(schedule->AssignedProcessor(pred), p)));
            }
        }
    } else {
        for (const VertexIdx &pred : G.Parents(node)) {
            // Comm. cost of sending pred to oldProc
            auto firstUse = succSteps[pred][oldProc].begin();
            bool skip = (schedule->AssignedProcessor(pred) == oldProc) || firstUse->first < step
                        || (firstUse->first == step && firstUse->second > 1);
            if (!skip) {
                sentInc.emplace_back(step - schedule->GetStaleness(),
                                     schedule->AssignedProcessor(pred),
                                     -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                       * schedule->GetInstance().GetArchitecture().SendCosts(
                                                           schedule->AssignedProcessor(pred), oldProc)));
                recInc.emplace_back(step - schedule->GetStaleness(),
                                    oldProc,
                                    -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule->GetInstance().GetArchitecture().SendCosts(
                                                          schedule->AssignedProcessor(pred), oldProc)));
                ++firstUse;
                if (firstUse != succSteps[pred][oldProc].end()) {
                    const unsigned nextStep = firstUse->first;
                    sentInc.emplace_back(nextStep - schedule->GetStaleness(),
                                         schedule->AssignedProcessor(pred),
                                         static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                          * schedule->GetInstance().GetArchitecture().SendCosts(
                                                              schedule->AssignedProcessor(pred), oldProc)));
                    recInc.emplace_back(nextStep - schedule->GetStaleness(),
                                        oldProc,
                                        static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                         * schedule->GetInstance().GetArchitecture().SendCosts(
                                                             schedule->AssignedProcessor(pred), oldProc)));
                }
            }

            // Comm. cost of sending pred to p
            firstUse = succSteps[pred][p].begin();
            skip = (schedule->AssignedProcessor(pred) == p)
                   || ((firstUse != succSteps[pred][p].end()) && (firstUse->first <= new_step));
            if (!skip) {
                sentInc.emplace_back(
                    new_step - schedule->GetStaleness(),
                    schedule->AssignedProcessor(pred),
                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                     * schedule->GetInstance().GetArchitecture().SendCosts(schedule->AssignedProcessor(pred), p)));
                recInc.emplace_back(
                    new_step - schedule->GetStaleness(),
                    p,
                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                     * schedule->GetInstance().GetArchitecture().SendCosts(schedule->AssignedProcessor(pred), p)));
                if (firstUse != succSteps[pred][p].end()) {
                    sentInc.emplace_back(firstUse->first - schedule->GetStaleness(),
                                         schedule->AssignedProcessor(pred),
                                         -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                           * schedule->GetInstance().GetArchitecture().SendCosts(
                                                               schedule->AssignedProcessor(pred), p)));
                    recInc.emplace_back(firstUse->first - schedule->GetStaleness(),
                                        p,
                                        -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                          * schedule->GetInstance().GetArchitecture().SendCosts(
                                                              schedule->AssignedProcessor(pred), p)));
                }
            }
        }
    }

    //  -process changes
    changing.sentChange.clear();
    changing.recChange.clear();
    for (auto entry : sentInc) {
        const unsigned e_step = std::get<0>(entry);
        const unsigned e_proc = std::get<1>(entry);
        const int e_increase = std::get<2>(entry);
        affectedSteps.insert(e_step);
        auto itr = changing.sentChange.find(std::make_pair(e_step, e_proc));
        if (itr == changing.sentChange.end()) {
            changing.sentChange.insert({std::make_pair(e_step, e_proc), e_increase});
        } else {
            itr->second += e_increase;
        }
    }
    for (auto entry : recInc) {
        const unsigned e_step = std::get<0>(entry);
        const unsigned e_proc = std::get<1>(entry);
        const int e_increase = std::get<2>(entry);
        affectedSteps.insert(e_step);
        auto itr = changing.recChange.find(std::make_pair(e_step, e_proc));
        if (itr == changing.recChange.end()) {
            changing.recChange.insert({std::make_pair(e_step, e_proc), e_increase});
        } else {
            itr->second += e_increase;
        }
    }

    auto itrSent = changing.sentChange.begin(), itrRec = changing.recChange.begin();
    bool last_affected_empty = false;
    for (const unsigned sstep : affectedSteps) {
        CostType oldMax = schedule->GetInstance().GetArchitecture().CommunicationCosts() * commCostList[sstep].rbegin()->first;
        CostType oldSync = (HCwithLatency && oldMax > 0) ? schedule->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        CostType newMax = 0;
        for (unsigned j = 0; j < schedule->GetInstance().GetArchitecture().NumberOfProcessors(); ++j) {
            int diff = (itrSent != changing.sentChange.end() && itrSent->first.first == sstep && itrSent->first.second == j)
                           ? (itrSent++)->second
                           : 0;
            if (static_cast<int>(sent[sstep][j]) + diff > static_cast<int>(newMax)) {
                newMax = static_cast<CostType>(static_cast<int>(sent[sstep][j]) + diff);
            }
            diff = (itrRec != changing.recChange.end() && itrRec->first.first == sstep && itrRec->first.second == j)
                       ? (itrRec++)->second
                       : 0;
            if (static_cast<int>(received[sstep][j]) + diff > static_cast<int>(newMax)) {
                newMax = static_cast<CostType>(static_cast<int>(received[sstep][j]) + diff);
            }
        }
        newMax *= schedule->GetInstance().GetArchitecture().CommunicationCosts();
        CostType newSync = (HCwithLatency && newMax > 0) ? schedule->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        if (newMax == 0) {
            if (schedule->GetStaleness() == 1) {
                changing.canShrink = true;
            } else {
                if ((sstep > 0 && affectedSteps.find(sstep - 1) == affectedSteps.end()
                     && commCostList[sstep - 1].rbegin()->first == 0)
                    || (sstep < commCostList.size() - 1 && affectedSteps.find(sstep + 1) == affectedSteps.end()
                        && commCostList[sstep + 1].rbegin()->first == 0)
                    || (sstep > 0 && affectedSteps.find(sstep - 1) != affectedSteps.end() && last_affected_empty)) {
                    changing.canShrink = true;
                }
            }
            last_affected_empty = true;
        } else {
            last_affected_empty = false;
        }

        if (schedule->GetStaleness() == 2) {
            auto itrWork = newWorkCost.find(sstep + 1);
            oldMax = std::max(oldMax, workCostList[sstep + 1].rbegin()->first);
            newMax = std::max(newMax, itrWork != newWorkCost.end() ? itrWork->second : workCostList[sstep + 1].rbegin()->first);
        }
        change += static_cast<int>(newMax + newSync) - static_cast<int>(oldMax + oldSync);
    }

    changing.newCost = static_cast<CostType>(static_cast<int>(cost) + change);
    return change;
}

// Execute a chosen move, updating the schedule and the data structures
template <typename GraphT>
void HillClimbingScheduler<GraphT>::executeMove(const VertexIdx node,
                                                const unsigned newProc,
                                                const int where,
                                                const StepAuxData &changing) {
    unsigned oldStep = schedule->AssignedSuperstep(node);
    unsigned newStep = static_cast<unsigned>(static_cast<int>(oldStep) + where);
    const unsigned oldProc = schedule->AssignedProcessor(node);
    cost = changing.newCost;

    // Work cost change
    workCostList[oldStep].erase(workCostPointer[oldStep][oldProc]);
    workCost[oldStep][oldProc] -= schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node);
    workCostPointer[oldStep][oldProc] = workCostList[oldStep].insert(std::make_pair(workCost[oldStep][oldProc], oldProc)).first;

    workCostList[newStep].erase(workCostPointer[newStep][newProc]);
    workCost[newStep][newProc] += schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node);
    workCostPointer[newStep][newProc] = workCostList[newStep].insert(std::make_pair(workCost[newStep][newProc], newProc)).first;

    // Comm cost change
    for (const auto &update : changing.sentChange) {
        sent[update.first.first][update.first.second]
            = static_cast<CostType>(static_cast<int>(sent[update.first.first][update.first.second]) + update.second);
    }
    for (const auto &update : changing.recChange) {
        received[update.first.first][update.first.second]
            = static_cast<CostType>(static_cast<int>(received[update.first.first][update.first.second]) + update.second);
    }

    std::set<std::pair<unsigned, unsigned>> toUpdate;
    for (const auto &update : changing.sentChange) {
        if (std::max(sent[update.first.first][update.first.second], received[update.first.first][update.first.second])
            != commCost[update.first.first][update.first.second]) {
            toUpdate.insert(std::make_pair(update.first.first, update.first.second));
        }
    }

    for (const auto &update : changing.recChange) {
        if (std::max(sent[update.first.first][update.first.second], received[update.first.first][update.first.second])
            != commCost[update.first.first][update.first.second]) {
            toUpdate.insert(std::make_pair(update.first.first, update.first.second));
        }
    }

    for (const auto &update : toUpdate) {
        commCostList[update.first].erase(commCostPointer[update.first][update.second]);
        commCost[update.first][update.second] = std::max(sent[update.first][update.second], received[update.first][update.second]);
        commCostPointer[update.first][update.second]
            = commCostList[update.first].insert(std::make_pair(commCost[update.first][update.second], update.second)).first;
    }

    // update successor lists
    for (const VertexIdx &pred : schedule->GetInstance().GetComputationalDag().Parents(node)) {
        auto itr = succSteps[pred][oldProc].find(oldStep);
        if ((--(itr->second)) == 0) {
            succSteps[pred][oldProc].erase(itr);
        }

        itr = succSteps[pred][newProc].find(newStep);
        if (itr == succSteps[pred][newProc].end()) {
            succSteps[pred][newProc].insert({newStep, 1U});
        } else {
            itr->second += 1;
        }
    }

    // memory constraints, if any
    if (use_memory_constraint) {
        memory_used[schedule->AssignedProcessor(node)][schedule->AssignedSuperstep(node)]
            -= schedule->GetInstance().GetComputationalDag().VertexMemWeight(node);
        memory_used[newProc][newStep] += schedule->GetInstance().GetComputationalDag().VertexMemWeight(node);
    }

    // update data
    schedule->SetAssignedProcessor(node, newProc);
    schedule->SetAssignedSuperstep(node, newStep);
    supsteplists[oldStep][oldProc].erase(supStepListPointer[node]);
    supsteplists[newStep][newProc].push_back(node);
    supStepListPointer[node] = (--supsteplists[newStep][newProc].end());

    updateMoveOptions(node, where);
}

// Single hill climbing step
template <typename GraphT>
bool HillClimbingScheduler<GraphT>::Improve() {
    CostType bestCost = cost;
    StepAuxData bestMoveData;
    std::pair<VertexIdx, unsigned> bestMove;
    int bestDir = 0;
    int startingDir = nextMove.first;

    // pre-selected "promising" moves
    while (!promisingMoves.empty() && !steepestAscent) {
        std::tuple<VertexIdx, unsigned, int> next = promisingMoves.front();
        promisingMoves.pop_front();

        const VertexIdx node = std::get<0>(next);
        const unsigned proc = std::get<1>(next);
        const int where = std::get<2>(next);

        if (!canMove[static_cast<Direction>(where)][node][proc]) {
            continue;
        }

        if (use_memory_constraint && violatesMemConstraint(node, proc, where - 1)) {
            continue;
        }

        StepAuxData moveData;
        int costDiff = moveCostChange(node, proc, where - 1, moveData);

        if (costDiff < 0) {
            executeMove(node, proc, where - 1, moveData);
            if (shrink && moveData.canShrink) {
                Init();
            }

            return true;
        }
    }

    // standard moves
    int dir = startingDir;
    while (true) {
        bool reachedBeginning = false;
        while (nextMove.second == moveOptions[static_cast<unsigned>(nextMove.first)].end()) {
            dir = (nextMove.first + 1) % NumDirections;
            if (dir == startingDir) {
                reachedBeginning = true;
                break;
            }
            nextMove.first = dir;
            nextMove.second = moveOptions[static_cast<unsigned>(nextMove.first)].begin();
        }
        if (reachedBeginning) {
            break;
        }

        std::pair<VertexIdx, unsigned> next = *nextMove.second;
        ++nextMove.second;

        const VertexIdx node = next.first;
        const unsigned proc = next.second;

        if (use_memory_constraint && violatesMemConstraint(node, proc, dir - 1)) {
            continue;
        }

        StepAuxData moveData;
        int costDiff = moveCostChange(node, proc, dir - 1, moveData);

        if (!steepestAscent && costDiff < 0) {
            executeMove(node, proc, dir - 1, moveData);
            if (shrink && moveData.canShrink) {
                Init();
            }

            return true;
        } else if (static_cast<CostType>(static_cast<int>(cost) + costDiff) < bestCost) {
            bestCost = static_cast<CostType>(static_cast<int>(cost) + costDiff);
            bestMove = next;
            bestMoveData = moveData;
            bestDir = dir - 1;
        }
    }

    if (bestCost == cost) {
        return false;
    }

    executeMove(bestMove.first, bestMove.second, bestDir, bestMoveData);
    if (shrink && bestMoveData.canShrink) {
        Init();
    }

    return true;
}

// Check if move violates mem constraints
template <typename GraphT>
bool HillClimbingScheduler<GraphT>::violatesMemConstraint(VertexIdx node, unsigned processor, int where) {
    if (memory_used[processor][static_cast<unsigned>(static_cast<int>(schedule->AssignedSuperstep(node)) + where)]
            + schedule->GetInstance().GetComputationalDag().VertexMemWeight(node)
        > schedule->GetInstance().MemoryBound(processor)) {    // TODO ANDRAS double check change
        return true;
    }

    return false;
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::CreateSupstepLists() {
    const unsigned P = schedule->GetInstance().GetArchitecture().NumberOfProcessors();
    const GraphT &G = schedule->GetInstance().GetComputationalDag();

    schedule->UpdateNumberOfSupersteps();
    const unsigned M = schedule->NumberOfSupersteps();

    supsteplists.clear();
    supsteplists.resize(M, std::vector<std::list<VertexIdx>>(P));

    for (VertexIdx node : TopSortView(G)) {
        supsteplists[schedule->AssignedSuperstep(node)][schedule->AssignedProcessor(node)].push_back(node);
    }
}

}    // namespace osp
