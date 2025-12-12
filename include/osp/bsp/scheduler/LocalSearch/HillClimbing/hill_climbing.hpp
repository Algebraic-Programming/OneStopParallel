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
    static_assert(IsDirectedGraphV<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(IsComputationalDagV<Graph_t>, "Graph_t must satisfy the computational_dag concept");

    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = VWorkwT<Graph_t>;

    static_assert(std::is_same_v<VWorkwT<Graph_t>, v_commw_t<Graph_t>>,
                  "HillClimbing requires work and comm. weights to have the same type.");

  public:
    enum Direction { EARLIER = 0, AT, LATER };

    static const int numDirections_ = 3;

    // aux structure for efficiently storing the changes incurred by a potential HC step
    struct StepAuxData {
        cost_type newCost_;
        std::map<std::pair<unsigned, unsigned>, int> sentChange_, recChange_;
        bool canShrink_ = false;
    };

  private:
    BspSchedule<GraphT> *schedule_;
    cost_type cost_ = 0;

    // Main parameters for runnign algorithm
    bool shrink_ = true;
    bool steepestAscent_ = false;

    // aux data structures
    std::vector<std::vector<std::list<vertex_idx>>> supsteplists_;
    std::vector<std::vector<std::vector<bool>>> canMove_;
    std::vector<std::list<std::pair<vertex_idx, unsigned>>> moveOptions_;
    std::vector<std::vector<std::vector<typename std::list<std::pair<vertex_idx, unsigned>>::iterator>>> movePointer_;
    std::vector<std::vector<std::map<unsigned, unsigned>>> succSteps_;
    std::vector<std::vector<cost_type>> workCost_, sent_, received_, commCost_;
    std::vector<std::set<std::pair<cost_type, unsigned>>> workCostList_, commCostList_;
    std::vector<std::vector<typename std::set<std::pair<cost_type, unsigned>>::iterator>> workCostPointer_, commCostPointer_;
    std::vector<typename std::list<vertex_idx>::iterator> supStepListPointer_;
    std::pair<int, typename std::list<std::pair<vertex_idx, unsigned>>::iterator> nextMove_;
    bool hCwithLatency_ = true;

    // for improved candidate selection
    std::deque<std::tuple<vertex_idx, unsigned, int>> promisingMoves_;
    bool findPromisingMoves_ = true;

    // Initialize data structures (based on current schedule)
    void Init();
    void UpdatePromisingMoves();

    // Functions to compute and update the std::list of possible moves
    void UpdateNodeMovesEarlier(vertex_idx node);
    void UpdateNodeMovesAt(vertex_idx node);
    void UpdateNodeMovesLater(vertex_idx node);
    void UpdateNodeMoves(vertex_idx node);
    void UpdateMoveOptions(vertex_idx node, int where);

    void AddMoveOption(vertex_idx node, unsigned p, Direction dir);

    void EraseMoveOption(vertex_idx node, unsigned p, Direction dir);
    void EraseMoveOptionsEarlier(vertex_idx node);
    void EraseMoveOptionsAt(vertex_idx node);
    void EraseMoveOptionsLater(vertex_idx node);
    void EraseMoveOptions(vertex_idx node);

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupstepLists();

    // For memory constraints
    bool useMemoryConstraint_ = false;
    std::vector<std::vector<v_memw_t<Graph_t>>> memoryUsed_;
    bool ViolatesMemConstraint(vertex_idx node, unsigned processor, int where);

    // Compute the cost change incurred by a potential move
    int MoveCostChange(vertex_idx node, unsigned p, int where, StepAuxData &changing);

    // Execute a chosen move, updating the schedule and the data structures
    void ExecuteMove(vertex_idx node, unsigned newProc, int where, const StepAuxData &changing);

    // Single hill climbing step
    bool Improve();

  public:
    HillClimbingScheduler() : ImprovementScheduler<GraphT>() {}

    virtual ~HillClimbingScheduler() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule<GraphT> &inputSchedule) override;

    // call with time/step limits
    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule<GraphT> &inputSchedule) override;
    virtual RETURN_STATUS ImproveScheduleWithStepLimit(BspSchedule<GraphT> &inputSchedule, const unsigned stepLimit = 10);

    // setting parameters
    void SetSteepestAscend(bool steepestAscent) { steepestAscent_ = steepestAscent; }

    void SetShrink(bool shrink) { shrink_ = shrink; }

    virtual std::string getScheduleName() const override { return "HillClimbing"; }
};

template <typename GraphT>
RETURN_STATUS HillClimbingScheduler<GraphT>::ImproveSchedule(BspSchedule<GraphT> &inputSchedule) {
    ImprovementScheduler<GraphT>::setTimeLimitSeconds(600U);
    return improveScheduleWithTimeLimit(input_schedule);
}

// Main method for hill climbing (with time limit)
template <typename GraphT>
RETURN_STATUS HillClimbingScheduler<GraphT>::ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &inputSchedule) {
    schedule_ = &inputSchedule;

    CreateSupstepLists();
    Init();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    int counter = 0;
    while (Improve()) {
        if ((++counter) == 10) {
            counter = 0;
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (elapsed >= ImprovementScheduler<GraphT>::timeLimitSeconds) {
                std::cout << "Hill Climbing was shut down due to time limit." << std::endl;
                break;
            }
        }
    }

    return RETURN_STATUS::OSP_SUCCESS;
}

template <typename GraphT>
// Hill climbing with step limit (designed as an ingredient for multilevel algorithms, no safety checks)
RETURN_STATUS HillClimbingScheduler<GraphT>::ImproveScheduleWithStepLimit(BspSchedule<GraphT> &inputSchedule,
                                                                          const unsigned stepLimit) {
    schedule_ = &inputSchedule;

    CreateSupstepLists();
    Init();
    for (unsigned step = 0; step < stepLimit; ++step) {
        if (!Improve()) {
            break;
        }
    }

    return RETURN_STATUS::OSP_SUCCESS;
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::Init() {
    if (shrink_) {
        schedule_->shrinkByMergingSupersteps();
        CreateSupstepLists();
    }

    const vertex_idx n = schedule_->GetInstance().GetComputationalDag().NumVertices();
    const unsigned p = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const unsigned m = schedule_->NumberOfSupersteps();
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    // Movement options
    canMove_.clear();
    canMove_.resize(numDirections_, std::vector<std::vector<bool>>(N, std::vector<bool>(p, false)));
    moveOptions.clear();
    moveOptions.resize(NumDirections);
    movePointer.clear();
    movePointer.resize(NumDirections,
                       std::vector<std::vector<typename std::list<std::pair<vertex_idx, unsigned>>::iterator>>(
                           N, std::vector<typename std::list<std::pair<vertex_idx, unsigned>>::iterator>(P)));

    // Value use lists
    succSteps_.clear();
    succSteps_.resize(N, std::vector<std::map<unsigned, unsigned>>(p));
    for (vertex_idx node = 0; node < N; ++node) {
        for (const vertex_idx &succ : G.Children(node)) {
            if (succSteps[node][schedule->assignedProcessor(succ)].find(schedule->assignedSuperstep(succ))
                == succSteps[node][schedule->assignedProcessor(succ)].end()) {
                succSteps[node][schedule->assignedProcessor(succ)].insert({schedule->assignedSuperstep(succ), 1U});
            } else {
                succSteps[node][schedule->assignedProcessor(succ)].at(schedule->assignedSuperstep(succ)) += 1;
            }
        }
    }

    // Cost data
    workCost.clear();
    workCost.resize(M, std::vector<cost_type>(P, 0));
    sent.clear();
    sent.resize(M - 1, std::vector<cost_type>(P, 0));
    received.clear();
    received.resize(M - 1, std::vector<cost_type>(P, 0));
    commCost.clear();
    commCost.resize(M - 1, std::vector<cost_type>(P));

    workCostList.clear();
    workCostList.resize(M);
    commCostList.clear();
    commCostList.resize(M - 1);
    workCostPointer.clear();
    workCostPointer.resize(M, std::vector<typename std::set<std::pair<cost_type, unsigned>>::iterator>(P));
    commCostPointer.clear();
    commCostPointer.resize(M - 1, std::vector<typename std::set<std::pair<cost_type, unsigned>>::iterator>(P));

    // Supstep std::list pointers
    supStepListPointer.clear();
    supStepListPointer.resize(N);
    for (unsigned step = 0; step < m; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (auto it = supsteplists[step][proc].begin(); it != supsteplists[step][proc].end(); ++it) {
                supStepListPointer[*it] = it;
            }
        }
    }

    // Compute movement options
    for (vertex_idx node = 0; node < N; ++node) {
        updateNodeMoves(node);
    }

    nextMove.first = 0;
    nextMove.second = moveOptions[0].begin();

    // Compute cost data
    std::vector<cost_type> workCost(m, 0);
    for (unsigned step = 0; step < m; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (const vertex_idx node : supsteplists[step][proc]) {
                workCost[step][proc] += schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node);
            }

            std::pair<cost_type, unsigned> entry(workCost[step][proc], proc);
            workCostPointer[step][proc] = workCostList[step].insert(entry).first;
        }
        work_cost[step] = (--workCostList[step].end())->first;
    }

    cost = work_cost[0];
    std::vector<std::vector<bool>> present(n, std::vector<bool>(p, false));
    for (unsigned step = 0; step < m - schedule_->getStaleness(); ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (const vertex_idx node : supsteplists[step + schedule->getStaleness()][proc]) {
                for (const vertex_idx &pred : G.Parents(node)) {
                    if (schedule->assignedProcessor(node) != schedule->assignedProcessor(pred)
                        && !present[pred][schedule->assignedProcessor(node)]) {
                        present[pred][schedule->assignedProcessor(node)] = true;
                        sent[step][schedule->assignedProcessor(pred)]
                            += schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                               * schedule->GetInstance().GetArchitecture().sendCosts(schedule->assignedProcessor(pred),
                                                                                     schedule->assignedProcessor(node));
                        received[step][schedule->assignedProcessor(node)]
                            += schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                               * schedule->GetInstance().GetArchitecture().sendCosts(schedule->assignedProcessor(pred),
                                                                                     schedule->assignedProcessor(node));
                    }
                }
            }
        }
    }

    for (unsigned step = 0; step < m - 1; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            commCost[step][proc] = std::max(sent[step][proc], received[step][proc]);
            std::pair<cost_type, unsigned> entry(commCost[step][proc], proc);
            commCostPointer[step][proc] = commCostList[step].insert(entry).first;
        }
        cost_type commCost = schedule->GetInstance().GetArchitecture().CommunicationCosts() * commCostList[step].rbegin()->first;
        cost_type syncCost = (commCost > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        if (schedule_->getStaleness() == 1) {
            cost += comm_cost + work_cost[step + 1] + sync_cost;
        } else {
            cost += std::max(comm_cost, work_cost[step + 1]) + sync_cost;
        }
    }

    UpdatePromisingMoves();

    // memory_constraints
    if (useMemoryConstraint_) {
        memory_used.clear();
        memory_used.resize(P, std::vector<v_memw_t<Graph_t>>(M, 0));
        for (vertex_idx node = 0; node < N; ++node) {
            memory_used[schedule->assignedProcessor(node)][schedule->assignedSuperstep(node)]
                += schedule->GetInstance().GetComputationalDag().VertexMemWeight(node);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdatePromisingMoves() {
    if (!findPromisingMoves_) {
        return;
    }

    const unsigned p = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    promisingMoves.clear();
    for (vertex_idx node = 0; node < schedule_->GetInstance().GetComputationalDag().NumVertices(); ++node) {
        std::vector<unsigned> nrPredOnProc(p, 0);
        for (const vertex_idx &pred : G.Parents(node)) {
            ++nrPredOnProc[schedule->assignedProcessor(pred)];
        }

        unsigned otherProcUsed = 0;
        for (unsigned proc = 0; proc < p; ++proc) {
            if (schedule_->assignedProcessor(node) != proc && nrPredOnProc[proc] > 0) {
                ++otherProcUsed;
            }
        }

        if (otherProcUsed == 1) {
            for (unsigned proc = 0; proc < p; ++proc) {
                if (schedule_->assignedProcessor(node) != proc && nrPredOnProc[proc] > 0
                    && schedule_->GetInstance().isCompatible(node, proc)) {
                    promisingMoves.push_back(std::make_tuple(node, proc, EARLIER));
                    promisingMoves.push_back(std::make_tuple(node, proc, AT));
                    promisingMoves.push_back(std::make_tuple(node, proc, LATER));
                }
            }
        }

        std::vector<unsigned> nrSuccOnProc(p, 0);
        for (const vertex_idx &succ : G.Children(node)) {
            ++nrSuccOnProc[schedule->assignedProcessor(succ)];
        }

        otherProcUsed = 0;
        for (unsigned proc = 0; proc < p; ++proc) {
            if (schedule_->assignedProcessor(node) != proc && nrSuccOnProc[proc] > 0) {
                ++otherProcUsed;
            }
        }

        if (otherProcUsed == 1) {
            for (unsigned proc = 0; proc < p; ++proc) {
                if (schedule_->assignedProcessor(node) != proc && nrSuccOnProc[proc] > 0
                    && schedule_->GetInstance().isCompatible(node, proc)) {
                    promisingMoves.push_back(std::make_tuple(node, proc, EARLIER));
                    promisingMoves.push_back(std::make_tuple(node, proc, AT));
                    promisingMoves.push_back(std::make_tuple(node, proc, LATER));
                }
            }
        }
    }

    for (unsigned step = 0; step < schedule_->NumberOfSupersteps(); ++step) {
        std::list<unsigned> minProcs, maxProcs;
        cost_type minWork = std::numeric_limits<cost_type>::max(), maxWork = std::numeric_limits<cost_type>::min();
        for (unsigned proc = 0; proc < p; ++proc) {
            if (workCost[step][proc] > maxWork) {
                maxWork = workCost[step][proc];
            }
            if (workCost[step][proc] < minWork) {
                minWork = workCost[step][proc];
            }
        }
        for (unsigned proc = 0; proc < p; ++proc) {
            if (workCost[step][proc] == minWork) {
                minProcs.push_back(proc);
            }
            if (workCost[step][proc] == maxWork) {
                maxProcs.push_back(proc);
            }
        }
        for (unsigned to : minProcs) {
            for (unsigned from : maxProcs) {
                for (vertex_idx node : supsteplists[step][from]) {
                    if (schedule->GetInstance().isCompatible(node, to)) {
                        promisingMoves.push_back(std::make_tuple(node, to, AT));
                    }
                }
            }
        }
    }
}

// Functions to compute and update the std::list of possible moves
template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateNodeMovesEarlier(const vertex_idx node) {
    if (schedule_->assignedSuperstep(node) == 0) {
        return;
    }

    std::set<unsigned> predProc;
    for (const vertex_idx &pred : schedule->GetInstance().GetComputationalDag().Parents(node)) {
        if (schedule->assignedSuperstep(pred) == schedule->assignedSuperstep(node)) {
            return;
        }
        if (static_cast<int>(schedule->assignedSuperstep(pred))
            >= static_cast<int>(schedule->assignedSuperstep(node)) - static_cast<int>(schedule->getStaleness())) {
            predProc.insert(schedule->assignedProcessor(pred));
        }
    }
    if (schedule_->getStaleness() == 2) {
        for (const vertex_idx &succ : schedule->GetInstance().GetComputationalDag().Children(node)) {
            if (schedule->assignedSuperstep(succ) == schedule->assignedSuperstep(node)) {
                predProc.insert(schedule->assignedProcessor(succ));
            }
        }
    }

    if (predProc.size() > 1) {
        return;
    }

    if (predProc.size() == 1) {
        addMoveOption(node, *predProc.begin(), EARLIER);
    } else {
        for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
            addMoveOption(node, proc, EARLIER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateNodeMovesAt(const vertex_idx node) {
    for (const vertex_idx &pred : schedule->GetInstance().GetComputationalDag().Parents(node)) {
        if (static_cast<int>(schedule->assignedSuperstep(pred))
            >= static_cast<int>(schedule->assignedSuperstep(node)) - static_cast<int>(schedule->getStaleness()) + 1) {
            return;
        }
    }

    for (const vertex_idx &succ : schedule->GetInstance().GetComputationalDag().Children(node)) {
        if (schedule->assignedSuperstep(succ) <= schedule->assignedSuperstep(node) + schedule->getStaleness() - 1) {
            return;
        }
    }

    for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (proc != schedule_->assignedProcessor(node)) {
            addMoveOption(node, proc, AT);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateNodeMovesLater(const vertex_idx node) {
    if (schedule_->assignedSuperstep(node) == schedule_->NumberOfSupersteps() - 1) {
        return;
    }

    std::set<unsigned> succProc;
    for (const vertex_idx &succ : schedule->GetInstance().GetComputationalDag().Children(node)) {
        if (schedule->assignedSuperstep(succ) == schedule->assignedSuperstep(node)) {
            return;
        }
        if (schedule->assignedSuperstep(succ) <= schedule->assignedSuperstep(node) + schedule->getStaleness()) {
            succProc.insert(schedule->assignedProcessor(succ));
        }
    }
    if (schedule_->getStaleness() == 2) {
        for (const vertex_idx &pred : schedule->GetInstance().GetComputationalDag().Parents(node)) {
            if (schedule->assignedSuperstep(pred) == schedule->assignedSuperstep(node)) {
                succProc.insert(schedule->assignedProcessor(pred));
            }
        }
    }

    if (succProc.size() > 1) {
        return;
    }

    if (succProc.size() == 1) {
        addMoveOption(node, *succProc.begin(), LATER);
    } else {
        for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
            addMoveOption(node, proc, LATER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateNodeMoves(const vertex_idx node) {
    eraseMoveOptions(node);
    updateNodeMovesEarlier(node);
    updateNodeMovesAt(node);
    updateNodeMovesLater(node);
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateMoveOptions(vertex_idx node, int where) {
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    updateNodeMoves(node);
    if (where == 0) {
        for (const vertex_idx &pred : G.Parents(node)) {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
        }
        for (const vertex_idx &succ : G.Children(node)) {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
        }
    }
    if (where == -1) {
        for (const vertex_idx &pred : G.Parents(node)) {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
            eraseMoveOptionsAt(pred);
            updateNodeMovesAt(pred);
            if (schedule->getStaleness() == 2) {
                eraseMoveOptionsEarlier(pred);
                updateNodeMovesEarlier(pred);
            }
        }
        for (const vertex_idx &succ : G.Children(node)) {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
            if (schedule->getStaleness() == 2) {
                eraseMoveOptionsAt(succ);
                updateNodeMovesAt(succ);
            }
        }
    }
    if (where == 1) {
        for (const vertex_idx &pred : G.Parents(node)) {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
            if (schedule->getStaleness() == 2) {
                eraseMoveOptionsAt(pred);
                updateNodeMovesAt(pred);
            }
        }
        for (const vertex_idx &succ : G.Children(node)) {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
            eraseMoveOptionsAt(succ);
            updateNodeMovesAt(succ);
            if (schedule->getStaleness() == 2) {
                eraseMoveOptionsLater(succ);
                updateNodeMovesLater(succ);
            }
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::AddMoveOption(const vertex_idx node, const unsigned p, const Direction dir) {
    if (!canMove_[dir][node][p] && schedule_->GetInstance().isCompatible(node, p)) {
        canMove_[dir][node][p] = true;
        moveOptions[dir].emplace_back(node, p);
        movePointer[dir][node][p] = --moveOptions[dir].end();
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOption(vertex_idx node, unsigned p, Direction dir) {
    canMove_[dir][node][p] = false;
    if (nextMove.first == dir && nextMove.second->first == node && nextMove.second->second == p) {
        ++nextMove.second;
    }
    moveOptions[dir].erase(movePointer[dir][node][p]);
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOptionsEarlier(vertex_idx node) {
    for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove_[EARLIER][node][proc]) {
            eraseMoveOption(node, proc, EARLIER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOptionsAt(vertex_idx node) {
    for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove_[AT][node][proc]) {
            eraseMoveOption(node, proc, AT);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOptionsLater(vertex_idx node) {
    for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove_[LATER][node][proc]) {
            eraseMoveOption(node, proc, LATER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOptions(vertex_idx node) {
    eraseMoveOptionsEarlier(node);
    eraseMoveOptionsAt(node);
    eraseMoveOptionsLater(node);
}

// Compute the cost change incurred by a potential move
template <typename GraphT>
int HillClimbingScheduler<GraphT>::MoveCostChange(const vertex_idx node, unsigned p, const int where, StepAuxData &changing) {
    const unsigned step = schedule_->assignedSuperstep(node);
    const unsigned newStep = static_cast<unsigned>(static_cast<int>(step) + where);
    unsigned oldProc = schedule_->assignedProcessor(node);
    int change = 0;

    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    std::set<unsigned> affectedSteps;

    // Work cost change
    std::map<unsigned, cost_type> newWorkCost;
    const auto itBest = --workCostList[step].end();
    cost_type maxAfterRemoval = itBest->first;
    if (itBest->second == oldProc) {
        auto itNext = itBest;
        --itNext;
        maxAfterRemoval
            = std::max(itBest->first - schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node), itNext->first);
        if (itBest->first != maxAfterRemoval) {
            if (step == 0 || schedule_->getStaleness() == 1) {    // incorporate immediately into cost change
                change -= static_cast<int>(itBest->first) - static_cast<int>(maxAfterRemoval);
            } else {
                newWorkCost[step] = maxAfterRemoval;
                affectedSteps.insert(step - 1);
            }
        }
    }

    const cost_type maxBeforeAddition = (where == 0) ? maxAfterRemoval : workCostList[new_step].rbegin()->first;
    if (workCost[new_step][p] + schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node) > maxBeforeAddition) {
        if (newStep == 0 || schedule_->getStaleness() == 1) {    // incorporate immediately into cost change
            change += static_cast<int>(workCost[new_step][p] + schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node))
                      - static_cast<int>(maxBeforeAddition);
        } else {
            newWorkCost[new_step] = workCost[new_step][p] + schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node);
            affectedSteps.insert(newStep - 1);
        }
    }

    // Comm cost change
    std::list<std::tuple<unsigned, unsigned, int>> sentInc, recInc;
    //  -outputs
    if (p != oldProc) {
        for (unsigned j = 0; j < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++j) {
            if (succSteps_[node][j].empty()) {
                continue;
            }

            unsigned affectedStep = succSteps_[node][j].begin()->first - schedule_->getStaleness();
            if (j == p) {
                sentInc.emplace_back(affectedStep,
                                     oldProc,
                                     -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                       * schedule_->GetInstance().GetArchitecture().sendCosts(oldProc, j)));
                recInc.emplace_back(affectedStep,
                                    p,
                                    -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule_->GetInstance().GetArchitecture().sendCosts(oldProc, j)));
            } else if (j == oldProc) {
                recInc.emplace_back(affectedStep,
                                    oldProc,
                                    static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                     * schedule_->GetInstance().GetArchitecture().sendCosts(p, j)));
                sentInc.emplace_back(affectedStep,
                                     p,
                                     static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule_->GetInstance().GetArchitecture().sendCosts(p, j)));
            } else {
                sentInc.emplace_back(affectedStep,
                                     oldProc,
                                     -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                       * schedule_->GetInstance().GetArchitecture().sendCosts(oldProc, j)));
                recInc.emplace_back(affectedStep,
                                    j,
                                    -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule_->GetInstance().GetArchitecture().sendCosts(oldProc, j)));
                sentInc.emplace_back(affectedStep,
                                     p,
                                     static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule_->GetInstance().GetArchitecture().sendCosts(p, j)));
                recInc.emplace_back(affectedStep,
                                    j,
                                    static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                     * schedule_->GetInstance().GetArchitecture().sendCosts(p, j)));
            }
        }
    }

    //  -inputs
    if (p == oldProc) {
        for (const vertex_idx &pred : G.Parents(node)) {
            if (schedule->assignedProcessor(pred) == p) {
                continue;
            }

            const auto firstUse = *succSteps[pred][p].begin();
            const bool skip = firstUse.first < step || (firstUse.first == step && where >= 0 && firstUse.second > 1);
            if (!skip) {
                sentInc.emplace_back(step - schedule->getStaleness(),
                                     schedule->assignedProcessor(pred),
                                     -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                       * schedule->GetInstance().GetArchitecture().sendCosts(
                                                           schedule->assignedProcessor(pred), p)));
                recInc.emplace_back(step - schedule->getStaleness(),
                                    p,
                                    -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule->GetInstance().GetArchitecture().sendCosts(
                                                          schedule->assignedProcessor(pred), p)));
                sentInc.emplace_back(
                    new_step - schedule->getStaleness(),
                    schedule->assignedProcessor(pred),
                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                     * schedule->GetInstance().GetArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                recInc.emplace_back(
                    new_step - schedule->getStaleness(),
                    p,
                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                     * schedule->GetInstance().GetArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
            }
        }
    } else {
        for (const vertex_idx &pred : G.Parents(node)) {
            // Comm. cost of sending pred to oldProc
            auto firstUse = succSteps[pred][oldProc].begin();
            bool skip = (schedule->assignedProcessor(pred) == oldProc) || firstUse->first < step
                        || (firstUse->first == step && firstUse->second > 1);
            if (!skip) {
                sentInc.emplace_back(step - schedule->getStaleness(),
                                     schedule->assignedProcessor(pred),
                                     -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                       * schedule->GetInstance().GetArchitecture().sendCosts(
                                                           schedule->assignedProcessor(pred), oldProc)));
                recInc.emplace_back(step - schedule->getStaleness(),
                                    oldProc,
                                    -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule->GetInstance().GetArchitecture().sendCosts(
                                                          schedule->assignedProcessor(pred), oldProc)));
                ++firstUse;
                if (firstUse != succSteps[pred][oldProc].end()) {
                    const unsigned nextStep = firstUse->first;
                    sentInc.emplace_back(nextStep - schedule->getStaleness(),
                                         schedule->assignedProcessor(pred),
                                         static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                          * schedule->GetInstance().GetArchitecture().sendCosts(
                                                              schedule->assignedProcessor(pred), oldProc)));
                    recInc.emplace_back(nextStep - schedule->getStaleness(),
                                        oldProc,
                                        static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                         * schedule->GetInstance().GetArchitecture().sendCosts(
                                                             schedule->assignedProcessor(pred), oldProc)));
                }
            }

            // Comm. cost of sending pred to p
            firstUse = succSteps[pred][p].begin();
            skip = (schedule->assignedProcessor(pred) == p)
                   || ((firstUse != succSteps[pred][p].end()) && (firstUse->first <= new_step));
            if (!skip) {
                sentInc.emplace_back(
                    new_step - schedule->getStaleness(),
                    schedule->assignedProcessor(pred),
                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                     * schedule->GetInstance().GetArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                recInc.emplace_back(
                    new_step - schedule->getStaleness(),
                    p,
                    static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                     * schedule->GetInstance().GetArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                if (firstUse != succSteps[pred][p].end()) {
                    sentInc.emplace_back(firstUse->first - schedule->getStaleness(),
                                         schedule->assignedProcessor(pred),
                                         -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                           * schedule->GetInstance().GetArchitecture().sendCosts(
                                                               schedule->assignedProcessor(pred), p)));
                    recInc.emplace_back(firstUse->first - schedule->getStaleness(),
                                        p,
                                        -static_cast<int>(schedule->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                          * schedule->GetInstance().GetArchitecture().sendCosts(
                                                              schedule->assignedProcessor(pred), p)));
                }
            }
        }
    }

    //  -process changes
    changing.sentChange_.clear();
    changing.recChange_.clear();
    for (auto entry : sentInc) {
        const unsigned eStep = std::get<0>(entry);
        const unsigned eProc = std::get<1>(entry);
        const int eIncrease = std::get<2>(entry);
        affectedSteps.insert(eStep);
        auto itr = changing.sentChange_.find(std::make_pair(eStep, eProc));
        if (itr == changing.sentChange_.end()) {
            changing.sentChange_.insert({std::make_pair(eStep, eProc), eIncrease});
        } else {
            itr->second += eIncrease;
        }
    }
    for (auto entry : recInc) {
        const unsigned eStep = std::get<0>(entry);
        const unsigned eProc = std::get<1>(entry);
        const int eIncrease = std::get<2>(entry);
        affectedSteps.insert(eStep);
        auto itr = changing.recChange_.find(std::make_pair(eStep, eProc));
        if (itr == changing.recChange_.end()) {
            changing.recChange_.insert({std::make_pair(eStep, eProc), eIncrease});
        } else {
            itr->second += eIncrease;
        }
    }

    auto itrSent = changing.sentChange_.begin(), itrRec = changing.recChange_.begin();
    bool lastAffectedEmpty = false;
    for (const unsigned sstep : affectedSteps) {
        cost_type oldMax = schedule->GetInstance().GetArchitecture().CommunicationCosts() * commCostList[sstep].rbegin()->first;
        cost_type oldSync = (hCwithLatency_ && oldMax > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        cost_type newMax = 0;
        for (unsigned j = 0; j < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++j) {
            int diff = (itrSent != changing.sentChange_.end() && itrSent->first.first == sstep && itrSent->first.second == j)
                           ? (itrSent++)->second
                           : 0;
            if (static_cast<int>(sent[sstep][j]) + diff > static_cast<int>(newMax)) {
                newMax = static_cast<cost_type>(static_cast<int>(sent[sstep][j]) + diff);
            }
            diff = (itrRec != changing.recChange_.end() && itrRec->first.first == sstep && itrRec->first.second == j)
                       ? (itrRec++)->second
                       : 0;
            if (static_cast<int>(received[sstep][j]) + diff > static_cast<int>(newMax)) {
                newMax = static_cast<cost_type>(static_cast<int>(received[sstep][j]) + diff);
            }
        }
        newMax *= schedule_->GetInstance().GetArchitecture().CommunicationCosts();
        cost_type newSync = (hCwithLatency_ && newMax > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        if (newMax == 0) {
            if (schedule_->getStaleness() == 1) {
                changing.canShrink_ = true;
            } else {
                if ((sstep > 0 && affectedSteps.find(sstep - 1) == affectedSteps.end()
                     && commCostList[sstep - 1].rbegin()->first == 0)
                    || (sstep < commCostList.size() - 1 && affectedSteps.find(sstep + 1) == affectedSteps.end()
                        && commCostList[sstep + 1].rbegin()->first == 0)
                    || (sstep > 0 && affectedSteps.find(sstep - 1) != affectedSteps.end() && last_affected_empty)) {
                    changing.canShrink_ = true;
                }
            }
            lastAffectedEmpty = true;
        } else {
            lastAffectedEmpty = false;
        }

        if (schedule_->getStaleness() == 2) {
            auto itrWork = newWorkCost.find(sstep + 1);
            oldMax = std::max(oldMax, workCostList[sstep + 1].rbegin()->first);
            newMax = std::max(newMax, itrWork != newWorkCost.end() ? itrWork->second : workCostList[sstep + 1].rbegin()->first);
        }
        change += static_cast<int>(newMax + newSync) - static_cast<int>(oldMax + oldSync);
    }

    changing.newCost = static_cast<cost_type>(static_cast<int>(cost) + change);
    return change;
}

// Execute a chosen move, updating the schedule and the data structures
template <typename GraphT>
void HillClimbingScheduler<GraphT>::ExecuteMove(const vertex_idx node,
                                                const unsigned newProc,
                                                const int where,
                                                const StepAuxData &changing) {
    unsigned oldStep = schedule_->assignedSuperstep(node);
    unsigned newStep = static_cast<unsigned>(static_cast<int>(oldStep) + where);
    const unsigned oldProc = schedule_->assignedProcessor(node);
    cost = changing.newCost;

    // Work cost change
    workCostList[oldStep].erase(workCostPointer[oldStep][oldProc]);
    workCost[oldStep][oldProc] -= schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node);
    workCostPointer[oldStep][oldProc] = workCostList[oldStep].insert(std::make_pair(workCost[oldStep][oldProc], oldProc)).first;

    workCostList[newStep].erase(workCostPointer[newStep][newProc]);
    workCost[newStep][newProc] += schedule->GetInstance().GetComputationalDag().VertexWorkWeight(node);
    workCostPointer[newStep][newProc] = workCostList[newStep].insert(std::make_pair(workCost[newStep][newProc], newProc)).first;

    // Comm cost change
    for (const auto &update : changing.sentChange_) {
        sent[update.first.first][update.first.second]
            = static_cast<cost_type>(static_cast<int>(sent[update.first.first][update.first.second]) + update.second);
    }
    for (const auto &update : changing.recChange_) {
        received[update.first.first][update.first.second]
            = static_cast<cost_type>(static_cast<int>(received[update.first.first][update.first.second]) + update.second);
    }

    std::set<std::pair<unsigned, unsigned>> toUpdate;
    for (const auto &update : changing.sentChange_) {
        if (std::max(sent[update.first.first][update.first.second], received[update.first.first][update.first.second])
            != commCost[update.first.first][update.first.second]) {
            toUpdate.insert(std::make_pair(update.first.first, update.first.second));
        }
    }

    for (const auto &update : changing.recChange_) {
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
    for (const vertex_idx &pred : schedule->GetInstance().GetComputationalDag().Parents(node)) {
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
    if (useMemoryConstraint_) {
        memory_used[schedule->assignedProcessor(node)][schedule->assignedSuperstep(node)]
            -= schedule->GetInstance().GetComputationalDag().VertexMemWeight(node);
        memory_used[newProc][newStep] += schedule->GetInstance().GetComputationalDag().VertexMemWeight(node);
    }

    // update data
    schedule_->setAssignedProcessor(node, newProc);
    schedule_->setAssignedSuperstep(node, newStep);
    supsteplists[oldStep][oldProc].erase(supStepListPointer[node]);
    supsteplists[newStep][newProc].push_back(node);
    supStepListPointer[node] = (--supsteplists[newStep][newProc].end());

    updateMoveOptions(node, where);
}

// Single hill climbing step
template <typename GraphT>
bool HillClimbingScheduler<GraphT>::Improve() {
    cost_type bestCost = cost;
    StepAuxData bestMoveData;
    std::pair<vertex_idx, unsigned> bestMove;
    int bestDir = 0;
    int startingDir = nextMove.first;

    // pre-selected "promising" moves
    while (!promisingMoves.empty() && !steepestAscent) {
        std::tuple<vertex_idx, unsigned, int> next = promisingMoves.front();
        promisingMoves.pop_front();

        const vertex_idx node = std::get<0>(next);
        const unsigned proc = std::get<1>(next);
        const int where = std::get<2>(next);

        if (!canMove_[static_cast<Direction>(where)][node][proc]) {
            continue;
        }

        if (use_memory_constraint && violatesMemConstraint(node, proc, where - 1)) {
            continue;
        }

        StepAuxData moveData;
        int costDiff = moveCostChange(node, proc, where - 1, moveData);

        if (costDiff < 0) {
            executeMove(node, proc, where - 1, moveData);
            if (shrink_ && moveData.canShrink_) {
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

        std::pair<vertex_idx, unsigned> next = *nextMove.second;
        ++nextMove.second;

        const vertex_idx node = next.first;
        const unsigned proc = next.second;

        if (use_memory_constraint && violatesMemConstraint(node, proc, dir - 1)) {
            continue;
        }

        StepAuxData moveData;
        int costDiff = moveCostChange(node, proc, dir - 1, moveData);

        if (!steepestAscent_ && costDiff < 0) {
            executeMove(node, proc, dir - 1, moveData);
            if (shrink_ && moveData.canShrink_) {
                Init();
            }

            return true;
        } else if (static_cast<cost_type>(static_cast<int>(cost) + costDiff) < bestCost) {
            bestCost = static_cast<cost_type>(static_cast<int>(cost) + costDiff);
            bestMove = next;
            bestMoveData = moveData;
            bestDir = dir - 1;
        }
    }

    if (bestCost == cost) {
        return false;
    }

    executeMove(bestMove.first, bestMove.second, bestDir, bestMoveData);
    if (shrink_ && bestMoveData.canShrink_) {
        Init();
    }

    return true;
}

// Check if move violates mem constraints
template <typename GraphT>
bool HillClimbingScheduler<GraphT>::ViolatesMemConstraint(vertex_idx node, unsigned processor, int where) {
    if (memory_used[processor][static_cast<unsigned>(static_cast<int>(schedule->assignedSuperstep(node)) + where)]
            + schedule->GetInstance().GetComputationalDag().VertexMemWeight(node)
        > schedule->GetInstance().memoryBound(processor)) {    // TODO ANDRAS double check change
        return true;
    }

    return false;
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::CreateSupstepLists() {
    const unsigned p = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    schedule_->updateNumberOfSupersteps();
    const unsigned m = schedule_->NumberOfSupersteps();

    supsteplists.clear();
    supsteplists.resize(M, std::vector<std::list<vertex_idx>>(P));

    for (vertex_idx node : top_sort_view(G)) {
        supsteplists[schedule->assignedSuperstep(node)][schedule->assignedProcessor(node)].push_back(node);
    }
}

}    // namespace osp
