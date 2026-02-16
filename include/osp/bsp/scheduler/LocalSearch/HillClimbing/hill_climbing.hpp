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
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");
    static_assert(isComputationalDagV<GraphT>, "GraphT must satisfy the computational_dag concept");

    using VertexIdx = VertexIdxT<GraphT>;
    using CostType = VWorkwT<GraphT>;

    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "HillClimbing requires work and comm. weights to have the same type.");

  public:
    enum Direction { EARLIER = 0, AT, LATER };

    static const int numDirections_ = 3;

    // aux structure for efficiently storing the changes incurred by a potential HC step
    struct StepAuxData {
        CostType newCost_;
        std::map<std::pair<unsigned, unsigned>, int> sentChange_, recChange_;
        bool canShrink_ = false;
    };

  private:
    BspSchedule<GraphT> *schedule_;
    CostType cost_ = 0;

    // Main parameters for runnign algorithm
    bool shrink_ = true;
    bool steepestAscent_ = false;

    // aux data structures
    std::vector<std::vector<std::list<VertexIdx>>> supsteplists_;
    std::vector<std::vector<std::vector<bool>>> canMove_;
    std::vector<std::list<std::pair<VertexIdx, unsigned>>> moveOptions_;
    std::vector<std::vector<std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>>> movePointer_;
    std::vector<std::vector<std::map<unsigned, unsigned>>> succSteps_;
    std::vector<std::vector<CostType>> workCost_, sent_, received_, commCost_;
    std::vector<std::set<std::pair<CostType, unsigned>>> workCostList_, commCostList_;
    std::vector<std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>> workCostPointer_, commCostPointer_;
    std::vector<typename std::list<VertexIdx>::iterator> supStepListPointer_;
    std::pair<int, typename std::list<std::pair<VertexIdx, unsigned>>::iterator> nextMove_;
    bool hcWithLatency_ = true;

    // for improved candidate selection
    std::deque<std::tuple<VertexIdx, unsigned, int>> promisingMoves_;
    bool findPromisingMoves_ = true;

    // Initialize data structures (based on current schedule)
    void Init();
    void UpdatePromisingMoves();

    // Functions to compute and update the std::list of possible moves
    void UpdateNodeMovesEarlier(VertexIdx node);
    void UpdateNodeMovesAt(VertexIdx node);
    void UpdateNodeMovesLater(VertexIdx node);
    void UpdateNodeMoves(VertexIdx node);
    void UpdateMoveOptions(VertexIdx node, int where);

    void AddMoveOption(VertexIdx node, unsigned proc, Direction dir);

    void EraseMoveOption(VertexIdx node, unsigned proc, Direction dir);
    void EraseMoveOptionsEarlier(VertexIdx node);
    void EraseMoveOptionsAt(VertexIdx node);
    void EraseMoveOptionsLater(VertexIdx node);
    void EraseMoveOptions(VertexIdx node);

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupstepLists();

    // For memory constraints
    bool useMemoryConstraint_ = false;
    std::vector<std::vector<VMemwT<GraphT>>> memoryUsed_;
    bool ViolatesMemConstraint(VertexIdx node, unsigned processor, int where);

    // Compute the cost change incurred by a potential move
    int MoveCostChange(VertexIdx node, unsigned proc, int where, StepAuxData &changing);

    // Execute a chosen move, updating the schedule and the data structures
    void ExecuteMove(VertexIdx node, unsigned newProc, int where, const StepAuxData &changing);

    // Single hill climbing step
    bool Improve();

  public:
    HillClimbingScheduler() : ImprovementScheduler<GraphT>() {}

    virtual ~HillClimbingScheduler() = default;

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &inputSchedule) override;

    // call with time/step limits
    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &inputSchedule) override;
    virtual ReturnStatus ImproveScheduleWithStepLimit(BspSchedule<GraphT> &inputSchedule, const unsigned stepLimit = 10);

    // setting parameters
    void SetSteepestAscend(bool steepestAscent) { steepestAscent_ = steepestAscent; }

    void SetShrink(bool shrink) { shrink_ = shrink; }

    virtual std::string GetScheduleName() const override { return "HillClimbing"; }
};

template <typename GraphT>
ReturnStatus HillClimbingScheduler<GraphT>::ImproveSchedule(BspSchedule<GraphT> &inputSchedule) {
    ImprovementScheduler<GraphT>::SetTimeLimitSeconds(600U);
    return ImproveScheduleWithTimeLimit(inputSchedule);
}

// Main method for hill climbing (with time limit)
template <typename GraphT>
ReturnStatus HillClimbingScheduler<GraphT>::ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &inputSchedule) {
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
ReturnStatus HillClimbingScheduler<GraphT>::ImproveScheduleWithStepLimit(BspSchedule<GraphT> &inputSchedule,
                                                                         const unsigned stepLimit) {
    schedule_ = &inputSchedule;

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
    if (shrink_) {
        schedule_->ShrinkByMergingSupersteps();
        CreateSupstepLists();
    }

    const VertexIdx n = schedule_->GetInstance().GetComputationalDag().NumVertices();
    const unsigned p = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const unsigned m = schedule_->NumberOfSupersteps();
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    // Movement options
    canMove_.clear();
    canMove_.resize(numDirections_, std::vector<std::vector<bool>>(n, std::vector<bool>(p, false)));
    moveOptions_.clear();
    moveOptions_.resize(numDirections_);
    movePointer_.clear();
    movePointer_.resize(numDirections_,
                        std::vector<std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>>(
                            n, std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>(p)));

    // Value use lists
    succSteps_.clear();
    succSteps_.resize(n, std::vector<std::map<unsigned, unsigned>>(p));
    for (VertexIdx node = 0; node < n; ++node) {
        for (const VertexIdx &succ : g.Children(node)) {
            if (succSteps_[node][schedule_->AssignedProcessor(succ)].find(schedule_->AssignedSuperstep(succ))
                == succSteps_[node][schedule_->AssignedProcessor(succ)].end()) {
                succSteps_[node][schedule_->AssignedProcessor(succ)].insert({schedule_->AssignedSuperstep(succ), 1U});
            } else {
                succSteps_[node][schedule_->AssignedProcessor(succ)].at(schedule_->AssignedSuperstep(succ)) += 1;
            }
        }
    }

    // Cost data
    workCost_.clear();
    workCost_.resize(m, std::vector<CostType>(p, 0));
    sent_.clear();
    sent_.resize(m - 1, std::vector<CostType>(p, 0));
    received_.clear();
    received_.resize(m - 1, std::vector<CostType>(p, 0));
    commCost_.clear();
    commCost_.resize(m - 1, std::vector<CostType>(p));

    workCostList_.clear();
    workCostList_.resize(m);
    commCostList_.clear();
    commCostList_.resize(m - 1);
    workCostPointer_.clear();
    workCostPointer_.resize(m, std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>(p));
    commCostPointer_.clear();
    commCostPointer_.resize(m - 1, std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>(p));

    // Supstep std::list pointers
    supStepListPointer_.clear();
    supStepListPointer_.resize(n);
    for (unsigned step = 0; step < m; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (auto it = supsteplists_[step][proc].begin(); it != supsteplists_[step][proc].end(); ++it) {
                supStepListPointer_[*it] = it;
            }
        }
    }

    // Compute movement options
    for (VertexIdx node = 0; node < n; ++node) {
        UpdateNodeMoves(node);
    }

    nextMove_.first = 0;
    nextMove_.second = moveOptions_[0].begin();

    // Compute cost data
    std::vector<CostType> workCost(m, 0);
    for (unsigned step = 0; step < m; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (const VertexIdx node : supsteplists_[step][proc]) {
                workCost_[step][proc] += schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node);
            }

            std::pair<CostType, unsigned> entry(workCost_[step][proc], proc);
            workCostPointer_[step][proc] = workCostList_[step].insert(entry).first;
        }
        workCost[step] = (--workCostList_[step].end())->first;
    }

    cost_ = workCost[0];
    std::vector<std::vector<bool>> present(n, std::vector<bool>(p, false));
    for (unsigned step = 0; step < m - schedule_->GetStaleness(); ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (const VertexIdx node : supsteplists_[step + schedule_->GetStaleness()][proc]) {
                for (const VertexIdx &pred : g.Parents(node)) {
                    if (schedule_->AssignedProcessor(node) != schedule_->AssignedProcessor(pred)
                        && !present[pred][schedule_->AssignedProcessor(node)]) {
                        present[pred][schedule_->AssignedProcessor(node)] = true;
                        sent_[step][schedule_->AssignedProcessor(pred)]
                            += schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                               * schedule_->GetInstance().GetArchitecture().SendCosts(schedule_->AssignedProcessor(pred),
                                                                                      schedule_->AssignedProcessor(node));
                        received_[step][schedule_->AssignedProcessor(node)]
                            += schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                               * schedule_->GetInstance().GetArchitecture().SendCosts(schedule_->AssignedProcessor(pred),
                                                                                      schedule_->AssignedProcessor(node));
                    }
                }
            }
        }
    }

    for (unsigned step = 0; step < m - 1; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            commCost_[step][proc] = std::max(sent_[step][proc], received_[step][proc]);
            std::pair<CostType, unsigned> entry(commCost_[step][proc], proc);
            commCostPointer_[step][proc] = commCostList_[step].insert(entry).first;
        }
        CostType commCost = schedule_->GetInstance().GetArchitecture().CommunicationCosts() * commCostList_[step].rbegin()->first;
        CostType syncCost = (commCost > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        if (schedule_->GetStaleness() == 1) {
            cost_ += commCost + workCost[step + 1] + syncCost;
        } else {
            cost_ += std::max(commCost, workCost[step + 1]) + syncCost;
        }
    }

    UpdatePromisingMoves();

    // memory_constraints
    if (useMemoryConstraint_) {
        memoryUsed_.clear();
        memoryUsed_.resize(p, std::vector<VMemwT<GraphT>>(m, 0));
        for (VertexIdx node = 0; node < n; ++node) {
            memoryUsed_[schedule_->AssignedProcessor(node)][schedule_->AssignedSuperstep(node)]
                += schedule_->GetInstance().GetComputationalDag().VertexMemWeight(node);
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

    promisingMoves_.clear();
    for (VertexIdx node = 0; node < schedule_->GetInstance().GetComputationalDag().NumVertices(); ++node) {
        std::vector<unsigned> nrPredOnProc(p, 0);
        for (const VertexIdx &pred : g.Parents(node)) {
            ++nrPredOnProc[schedule_->AssignedProcessor(pred)];
        }

        unsigned otherProcUsed = 0;
        for (unsigned proc = 0; proc < p; ++proc) {
            if (schedule_->AssignedProcessor(node) != proc && nrPredOnProc[proc] > 0) {
                ++otherProcUsed;
            }
        }

        if (otherProcUsed == 1) {
            for (unsigned proc = 0; proc < p; ++proc) {
                if (schedule_->AssignedProcessor(node) != proc && nrPredOnProc[proc] > 0
                    && schedule_->GetInstance().IsCompatible(node, proc)) {
                    promisingMoves_.push_back(std::make_tuple(node, proc, EARLIER));
                    promisingMoves_.push_back(std::make_tuple(node, proc, AT));
                    promisingMoves_.push_back(std::make_tuple(node, proc, LATER));
                }
            }
        }

        std::vector<unsigned> nrSuccOnProc(p, 0);
        for (const VertexIdx &succ : g.Children(node)) {
            ++nrSuccOnProc[schedule_->AssignedProcessor(succ)];
        }

        otherProcUsed = 0;
        for (unsigned proc = 0; proc < p; ++proc) {
            if (schedule_->AssignedProcessor(node) != proc && nrSuccOnProc[proc] > 0) {
                ++otherProcUsed;
            }
        }

        if (otherProcUsed == 1) {
            for (unsigned proc = 0; proc < p; ++proc) {
                if (schedule_->AssignedProcessor(node) != proc && nrSuccOnProc[proc] > 0
                    && schedule_->GetInstance().IsCompatible(node, proc)) {
                    promisingMoves_.push_back(std::make_tuple(node, proc, EARLIER));
                    promisingMoves_.push_back(std::make_tuple(node, proc, AT));
                    promisingMoves_.push_back(std::make_tuple(node, proc, LATER));
                }
            }
        }
    }

    for (unsigned step = 0; step < schedule_->NumberOfSupersteps(); ++step) {
        std::list<unsigned> minProcs, maxProcs;
        CostType minWork = std::numeric_limits<CostType>::max(), maxWork = std::numeric_limits<CostType>::min();
        for (unsigned proc = 0; proc < p; ++proc) {
            if (workCost_[step][proc] > maxWork) {
                maxWork = workCost_[step][proc];
            }
            if (workCost_[step][proc] < minWork) {
                minWork = workCost_[step][proc];
            }
        }
        for (unsigned proc = 0; proc < p; ++proc) {
            if (workCost_[step][proc] == minWork) {
                minProcs.push_back(proc);
            }
            if (workCost_[step][proc] == maxWork) {
                maxProcs.push_back(proc);
            }
        }
        for (unsigned to : minProcs) {
            for (unsigned from : maxProcs) {
                for (VertexIdx node : supsteplists_[step][from]) {
                    if (schedule_->GetInstance().IsCompatible(node, to)) {
                        promisingMoves_.push_back(std::make_tuple(node, to, AT));
                    }
                }
            }
        }
    }
}

// Functions to compute and update the std::list of possible moves
template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateNodeMovesEarlier(const VertexIdx node) {
    if (schedule_->AssignedSuperstep(node) == 0) {
        return;
    }

    std::set<unsigned> predProc;
    for (const VertexIdx &pred : schedule_->GetInstance().GetComputationalDag().Parents(node)) {
        if (schedule_->AssignedSuperstep(pred) == schedule_->AssignedSuperstep(node)) {
            return;
        }
        if (static_cast<int>(schedule_->AssignedSuperstep(pred))
            >= static_cast<int>(schedule_->AssignedSuperstep(node)) - static_cast<int>(schedule_->GetStaleness())) {
            predProc.insert(schedule_->AssignedProcessor(pred));
        }
    }
    if (schedule_->GetStaleness() == 2) {
        for (const VertexIdx &succ : schedule_->GetInstance().GetComputationalDag().Children(node)) {
            if (schedule_->AssignedSuperstep(succ) == schedule_->AssignedSuperstep(node)) {
                predProc.insert(schedule_->AssignedProcessor(succ));
            }
        }
    }

    if (predProc.size() > 1) {
        return;
    }

    if (predProc.size() == 1) {
        AddMoveOption(node, *predProc.begin(), EARLIER);
    } else {
        for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
            AddMoveOption(node, proc, EARLIER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateNodeMovesAt(const VertexIdx node) {
    for (const VertexIdx &pred : schedule_->GetInstance().GetComputationalDag().Parents(node)) {
        if (static_cast<int>(schedule_->AssignedSuperstep(pred))
            >= static_cast<int>(schedule_->AssignedSuperstep(node)) - static_cast<int>(schedule_->GetStaleness()) + 1) {
            return;
        }
    }

    for (const VertexIdx &succ : schedule_->GetInstance().GetComputationalDag().Children(node)) {
        if (schedule_->AssignedSuperstep(succ) <= schedule_->AssignedSuperstep(node) + schedule_->GetStaleness() - 1) {
            return;
        }
    }

    for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (proc != schedule_->AssignedProcessor(node)) {
            AddMoveOption(node, proc, AT);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateNodeMovesLater(const VertexIdx node) {
    if (schedule_->AssignedSuperstep(node) == schedule_->NumberOfSupersteps() - 1) {
        return;
    }

    std::set<unsigned> succProc;
    for (const VertexIdx &succ : schedule_->GetInstance().GetComputationalDag().Children(node)) {
        if (schedule_->AssignedSuperstep(succ) == schedule_->AssignedSuperstep(node)) {
            return;
        }
        if (schedule_->AssignedSuperstep(succ) <= schedule_->AssignedSuperstep(node) + schedule_->GetStaleness()) {
            succProc.insert(schedule_->AssignedProcessor(succ));
        }
    }
    if (schedule_->GetStaleness() == 2) {
        for (const VertexIdx &pred : schedule_->GetInstance().GetComputationalDag().Parents(node)) {
            if (schedule_->AssignedSuperstep(pred) == schedule_->AssignedSuperstep(node)) {
                succProc.insert(schedule_->AssignedProcessor(pred));
            }
        }
    }

    if (succProc.size() > 1) {
        return;
    }

    if (succProc.size() == 1) {
        AddMoveOption(node, *succProc.begin(), LATER);
    } else {
        for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
            AddMoveOption(node, proc, LATER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateNodeMoves(const VertexIdx node) {
    EraseMoveOptions(node);
    UpdateNodeMovesEarlier(node);
    UpdateNodeMovesAt(node);
    UpdateNodeMovesLater(node);
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::UpdateMoveOptions(VertexIdx node, int where) {
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    UpdateNodeMoves(node);
    if (where == 0) {
        for (const VertexIdx &pred : g.Parents(node)) {
            EraseMoveOptionsLater(pred);
            UpdateNodeMovesLater(pred);
        }
        for (const VertexIdx &succ : g.Children(node)) {
            EraseMoveOptionsEarlier(succ);
            UpdateNodeMovesEarlier(succ);
        }
    }
    if (where == -1) {
        for (const VertexIdx &pred : g.Parents(node)) {
            EraseMoveOptionsLater(pred);
            UpdateNodeMovesLater(pred);
            EraseMoveOptionsAt(pred);
            UpdateNodeMovesAt(pred);
            if (schedule_->GetStaleness() == 2) {
                EraseMoveOptionsEarlier(pred);
                UpdateNodeMovesEarlier(pred);
            }
        }
        for (const VertexIdx &succ : g.Children(node)) {
            EraseMoveOptionsEarlier(succ);
            UpdateNodeMovesEarlier(succ);
            if (schedule_->GetStaleness() == 2) {
                EraseMoveOptionsAt(succ);
                UpdateNodeMovesAt(succ);
            }
        }
    }
    if (where == 1) {
        for (const VertexIdx &pred : g.Parents(node)) {
            EraseMoveOptionsLater(pred);
            UpdateNodeMovesLater(pred);
            if (schedule_->GetStaleness() == 2) {
                EraseMoveOptionsAt(pred);
                UpdateNodeMovesAt(pred);
            }
        }
        for (const VertexIdx &succ : g.Children(node)) {
            EraseMoveOptionsEarlier(succ);
            UpdateNodeMovesEarlier(succ);
            EraseMoveOptionsAt(succ);
            UpdateNodeMovesAt(succ);
            if (schedule_->GetStaleness() == 2) {
                EraseMoveOptionsLater(succ);
                UpdateNodeMovesLater(succ);
            }
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::AddMoveOption(const VertexIdx node, const unsigned proc, const Direction dir) {
    if (!canMove_[dir][node][proc] && schedule_->GetInstance().IsCompatible(node, proc)) {
        canMove_[dir][node][proc] = true;
        moveOptions_[dir].emplace_back(node, proc);
        movePointer_[dir][node][proc] = --moveOptions_[dir].end();
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOption(VertexIdx node, unsigned proc, Direction dir) {
    canMove_[dir][node][proc] = false;
    if (nextMove_.first == dir && nextMove_.second != moveOptions_[dir].end() && nextMove_.second->first == node && nextMove_.second->second == proc) {
        ++nextMove_.second;
    }
    moveOptions_[dir].erase(movePointer_[dir][node][proc]);
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOptionsEarlier(VertexIdx node) {
    for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove_[EARLIER][node][proc]) {
            EraseMoveOption(node, proc, EARLIER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOptionsAt(VertexIdx node) {
    for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove_[AT][node][proc]) {
            EraseMoveOption(node, proc, AT);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOptionsLater(VertexIdx node) {
    for (unsigned proc = 0; proc < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++proc) {
        if (canMove_[LATER][node][proc]) {
            EraseMoveOption(node, proc, LATER);
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOptions(VertexIdx node) {
    EraseMoveOptionsEarlier(node);
    EraseMoveOptionsAt(node);
    EraseMoveOptionsLater(node);
}

// Compute the cost change incurred by a potential move
template <typename GraphT>
int HillClimbingScheduler<GraphT>::MoveCostChange(const VertexIdx node, unsigned proc, const int where, StepAuxData &changing) {
    const unsigned step = schedule_->AssignedSuperstep(node);
    const unsigned newStep = static_cast<unsigned>(static_cast<int>(step) + where);
    unsigned oldProc = schedule_->AssignedProcessor(node);
    int change = 0;

    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    std::set<unsigned> affectedSteps;

    // Work cost change
    std::map<unsigned, CostType> newWorkCost;
    const auto itBest = --workCostList_[step].end();
    CostType maxAfterRemoval = itBest->first;
    if (itBest->second == oldProc) {
        auto itNext = itBest;
        --itNext;
        maxAfterRemoval
            = std::max(itBest->first - schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node), itNext->first);
        if (itBest->first != maxAfterRemoval) {
            if (step == 0 || schedule_->GetStaleness() == 1) {    // incorporate immediately into cost change
                change -= static_cast<int>(itBest->first) - static_cast<int>(maxAfterRemoval);
            } else {
                newWorkCost[step] = maxAfterRemoval;
                affectedSteps.insert(step - 1);
            }
        }
    }

    const CostType maxBeforeAddition = (where == 0) ? maxAfterRemoval : workCostList_[newStep].rbegin()->first;
    if (workCost_[newStep][proc] + schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node) > maxBeforeAddition) {
        if (newStep == 0 || schedule_->GetStaleness() == 1) {    // incorporate immediately into cost change
            change += static_cast<int>(workCost_[newStep][proc]
                                       + schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node))
                      - static_cast<int>(maxBeforeAddition);
        } else {
            newWorkCost[newStep] = workCost_[newStep][proc] + schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node);
            affectedSteps.insert(newStep - 1);
        }
    }

    // Comm cost change
    std::list<std::tuple<unsigned, unsigned, int>> sentInc, recInc;
    //  -outputs
    if (proc != oldProc) {
        for (unsigned j = 0; j < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++j) {
            if (succSteps_[node][j].empty()) {
                continue;
            }

            unsigned affectedStep = succSteps_[node][j].begin()->first - schedule_->GetStaleness();
            if (j == proc) {
                sentInc.emplace_back(affectedStep,
                                     oldProc,
                                     -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                       * schedule_->GetInstance().GetArchitecture().SendCosts(oldProc, j)));
                recInc.emplace_back(affectedStep,
                                    proc,
                                    -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule_->GetInstance().GetArchitecture().SendCosts(oldProc, j)));
            } else if (j == oldProc) {
                recInc.emplace_back(affectedStep,
                                    oldProc,
                                    static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                     * schedule_->GetInstance().GetArchitecture().SendCosts(proc, j)));
                sentInc.emplace_back(affectedStep,
                                     proc,
                                     static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule_->GetInstance().GetArchitecture().SendCosts(proc, j)));
            } else {
                sentInc.emplace_back(affectedStep,
                                     oldProc,
                                     -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                       * schedule_->GetInstance().GetArchitecture().SendCosts(oldProc, j)));
                recInc.emplace_back(affectedStep,
                                    j,
                                    -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule_->GetInstance().GetArchitecture().SendCosts(oldProc, j)));
                sentInc.emplace_back(affectedStep,
                                     proc,
                                     static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                      * schedule_->GetInstance().GetArchitecture().SendCosts(proc, j)));
                recInc.emplace_back(affectedStep,
                                    j,
                                    static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                                     * schedule_->GetInstance().GetArchitecture().SendCosts(proc, j)));
            }
        }
    }

    //  -inputs
    if (proc == oldProc) {
        for (const VertexIdx &pred : g.Parents(node)) {
            if (schedule_->AssignedProcessor(pred) == proc) {
                continue;
            }

            const auto firstUse = *succSteps_[pred][proc].begin();
            const bool skip = firstUse.first < step || (firstUse.first == step && where >= 0 && firstUse.second > 1);
            if (!skip) {
                sentInc.emplace_back(step - schedule_->GetStaleness(),
                                     schedule_->AssignedProcessor(pred),
                                     -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                       * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                           schedule_->AssignedProcessor(pred), proc)));
                recInc.emplace_back(step - schedule_->GetStaleness(),
                                    proc,
                                    -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                          schedule_->AssignedProcessor(pred), proc)));
                sentInc.emplace_back(newStep - schedule_->GetStaleness(),
                                     schedule_->AssignedProcessor(pred),
                                     static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                          schedule_->AssignedProcessor(pred), proc)));
                recInc.emplace_back(newStep - schedule_->GetStaleness(),
                                    proc,
                                    static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                     * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                         schedule_->AssignedProcessor(pred), proc)));
            }
        }
    } else {
        for (const VertexIdx &pred : g.Parents(node)) {
            // Comm. cost of sending pred to oldProc
            auto firstUse = succSteps_[pred][oldProc].begin();
            bool skip = (schedule_->AssignedProcessor(pred) == oldProc) || firstUse->first < step
                        || (firstUse->first == step && firstUse->second > 1);
            if (!skip) {
                sentInc.emplace_back(step - schedule_->GetStaleness(),
                                     schedule_->AssignedProcessor(pred),
                                     -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                       * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                           schedule_->AssignedProcessor(pred), oldProc)));
                recInc.emplace_back(step - schedule_->GetStaleness(),
                                    oldProc,
                                    -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                          schedule_->AssignedProcessor(pred), oldProc)));
                ++firstUse;
                if (firstUse != succSteps_[pred][oldProc].end()) {
                    const unsigned nextStep = firstUse->first;
                    sentInc.emplace_back(nextStep - schedule_->GetStaleness(),
                                         schedule_->AssignedProcessor(pred),
                                         static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                          * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                              schedule_->AssignedProcessor(pred), oldProc)));
                    recInc.emplace_back(nextStep - schedule_->GetStaleness(),
                                        oldProc,
                                        static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                         * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                             schedule_->AssignedProcessor(pred), oldProc)));
                }
            }

            // Comm. cost of sending pred to proc
            firstUse = succSteps_[pred][proc].begin();
            skip = (schedule_->AssignedProcessor(pred) == proc)
                   || ((firstUse != succSteps_[pred][proc].end()) && (firstUse->first <= newStep));
            if (!skip) {
                sentInc.emplace_back(newStep - schedule_->GetStaleness(),
                                     schedule_->AssignedProcessor(pred),
                                     static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                          schedule_->AssignedProcessor(pred), proc)));
                recInc.emplace_back(newStep - schedule_->GetStaleness(),
                                    proc,
                                    static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                     * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                         schedule_->AssignedProcessor(pred), proc)));
                if (firstUse != succSteps_[pred][proc].end()) {
                    sentInc.emplace_back(firstUse->first - schedule_->GetStaleness(),
                                         schedule_->AssignedProcessor(pred),
                                         -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                           * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                               schedule_->AssignedProcessor(pred), proc)));
                    recInc.emplace_back(firstUse->first - schedule_->GetStaleness(),
                                        proc,
                                        -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                          * schedule_->GetInstance().GetArchitecture().SendCosts(
                                                              schedule_->AssignedProcessor(pred), proc)));
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
        CostType oldMax = schedule_->GetInstance().GetArchitecture().CommunicationCosts() * commCostList_[sstep].rbegin()->first;
        CostType oldSync = (hcWithLatency_ && oldMax > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        CostType newMax = 0;
        for (unsigned j = 0; j < schedule_->GetInstance().GetArchitecture().NumberOfProcessors(); ++j) {
            int diff = (itrSent != changing.sentChange_.end() && itrSent->first.first == sstep && itrSent->first.second == j)
                           ? (itrSent++)->second
                           : 0;
            if (static_cast<int>(sent_[sstep][j]) + diff > static_cast<int>(newMax)) {
                newMax = static_cast<CostType>(static_cast<int>(sent_[sstep][j]) + diff);
            }
            diff = (itrRec != changing.recChange_.end() && itrRec->first.first == sstep && itrRec->first.second == j)
                       ? (itrRec++)->second
                       : 0;
            if (static_cast<int>(received_[sstep][j]) + diff > static_cast<int>(newMax)) {
                newMax = static_cast<CostType>(static_cast<int>(received_[sstep][j]) + diff);
            }
        }
        newMax *= schedule_->GetInstance().GetArchitecture().CommunicationCosts();
        CostType newSync = (hcWithLatency_ && newMax > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        if (newMax == 0) {
            if (schedule_->GetStaleness() == 1) {
                changing.canShrink_ = true;
            } else {
                if ((sstep > 0 && affectedSteps.find(sstep - 1) == affectedSteps.end()
                     && commCostList_[sstep - 1].rbegin()->first == 0)
                    || (sstep < commCostList_.size() - 1 && affectedSteps.find(sstep + 1) == affectedSteps.end()
                        && commCostList_[sstep + 1].rbegin()->first == 0)
                    || (sstep > 0 && affectedSteps.find(sstep - 1) != affectedSteps.end() && lastAffectedEmpty)) {
                    changing.canShrink_ = true;
                }
            }
            lastAffectedEmpty = true;
        } else {
            lastAffectedEmpty = false;
        }

        if (schedule_->GetStaleness() == 2) {
            auto itrWork = newWorkCost.find(sstep + 1);
            oldMax = std::max(oldMax, workCostList_[sstep + 1].rbegin()->first);
            newMax = std::max(newMax, itrWork != newWorkCost.end() ? itrWork->second : workCostList_[sstep + 1].rbegin()->first);
        }
        change += static_cast<int>(newMax + newSync) - static_cast<int>(oldMax + oldSync);
    }

    changing.newCost_ = static_cast<CostType>(static_cast<int>(cost_) + change);
    return change;
}

// Execute a chosen move, updating the schedule and the data structures
template <typename GraphT>
void HillClimbingScheduler<GraphT>::ExecuteMove(const VertexIdx node,
                                                const unsigned newProc,
                                                const int where,
                                                const StepAuxData &changing) {
    unsigned oldStep = schedule_->AssignedSuperstep(node);
    unsigned newStep = static_cast<unsigned>(static_cast<int>(oldStep) + where);
    const unsigned oldProc = schedule_->AssignedProcessor(node);
    cost_ = changing.newCost_;

    // Work cost change
    workCostList_[oldStep].erase(workCostPointer_[oldStep][oldProc]);
    workCost_[oldStep][oldProc] -= schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node);
    workCostPointer_[oldStep][oldProc] = workCostList_[oldStep].insert(std::make_pair(workCost_[oldStep][oldProc], oldProc)).first;

    workCostList_[newStep].erase(workCostPointer_[newStep][newProc]);
    workCost_[newStep][newProc] += schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node);
    workCostPointer_[newStep][newProc] = workCostList_[newStep].insert(std::make_pair(workCost_[newStep][newProc], newProc)).first;

    // Comm cost change
    for (const auto &update : changing.sentChange_) {
        sent_[update.first.first][update.first.second]
            = static_cast<CostType>(static_cast<int>(sent_[update.first.first][update.first.second]) + update.second);
    }
    for (const auto &update : changing.recChange_) {
        received_[update.first.first][update.first.second]
            = static_cast<CostType>(static_cast<int>(received_[update.first.first][update.first.second]) + update.second);
    }

    std::set<std::pair<unsigned, unsigned>> toUpdate;
    for (const auto &update : changing.sentChange_) {
        if (std::max(sent_[update.first.first][update.first.second], received_[update.first.first][update.first.second])
            != commCost_[update.first.first][update.first.second]) {
            toUpdate.insert(std::make_pair(update.first.first, update.first.second));
        }
    }

    for (const auto &update : changing.recChange_) {
        if (std::max(sent_[update.first.first][update.first.second], received_[update.first.first][update.first.second])
            != commCost_[update.first.first][update.first.second]) {
            toUpdate.insert(std::make_pair(update.first.first, update.first.second));
        }
    }

    for (const auto &update : toUpdate) {
        commCostList_[update.first].erase(commCostPointer_[update.first][update.second]);
        commCost_[update.first][update.second]
            = std::max(sent_[update.first][update.second], received_[update.first][update.second]);
        commCostPointer_[update.first][update.second]
            = commCostList_[update.first].insert(std::make_pair(commCost_[update.first][update.second], update.second)).first;
    }

    // update successor lists
    for (const VertexIdx &pred : schedule_->GetInstance().GetComputationalDag().Parents(node)) {
        auto itr = succSteps_[pred][oldProc].find(oldStep);
        if ((--(itr->second)) == 0) {
            succSteps_[pred][oldProc].erase(itr);
        }

        itr = succSteps_[pred][newProc].find(newStep);
        if (itr == succSteps_[pred][newProc].end()) {
            succSteps_[pred][newProc].insert({newStep, 1U});
        } else {
            itr->second += 1;
        }
    }

    // memory constraints, if any
    if (useMemoryConstraint_) {
        memoryUsed_[schedule_->AssignedProcessor(node)][schedule_->AssignedSuperstep(node)]
            -= schedule_->GetInstance().GetComputationalDag().VertexMemWeight(node);
        memoryUsed_[newProc][newStep] += schedule_->GetInstance().GetComputationalDag().VertexMemWeight(node);
    }

    // update data
    schedule_->SetAssignedProcessor(node, newProc);
    schedule_->SetAssignedSuperstep(node, newStep);
    supsteplists_[oldStep][oldProc].erase(supStepListPointer_[node]);
    supsteplists_[newStep][newProc].push_back(node);
    supStepListPointer_[node] = (--supsteplists_[newStep][newProc].end());

    UpdateMoveOptions(node, where);
}

// Single hill climbing step
template <typename GraphT>
bool HillClimbingScheduler<GraphT>::Improve() {
    CostType bestCost = cost_;
    StepAuxData bestMoveData;
    std::pair<VertexIdx, unsigned> bestMove;
    int bestDir = 0;
    int startingDir = nextMove_.first;

    // pre-selected "promising" moves
    while (!promisingMoves_.empty() && !steepestAscent_) {
        std::tuple<VertexIdx, unsigned, int> next = promisingMoves_.front();
        promisingMoves_.pop_front();

        const VertexIdx node = std::get<0>(next);
        const unsigned proc = std::get<1>(next);
        const int where = std::get<2>(next);

        if (!canMove_[static_cast<Direction>(where)][node][proc]) {
            continue;
        }

        if (useMemoryConstraint_ && ViolatesMemConstraint(node, proc, where - 1)) {
            continue;
        }

        StepAuxData moveData;
        int costDiff = MoveCostChange(node, proc, where - 1, moveData);

        if (costDiff < 0) {
            ExecuteMove(node, proc, where - 1, moveData);
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
        while (nextMove_.second == moveOptions_[static_cast<unsigned>(nextMove_.first)].end()) {
            dir = (nextMove_.first + 1) % numDirections_;
            if (dir == startingDir) {
                reachedBeginning = true;
                break;
            }
            nextMove_.first = dir;
            nextMove_.second = moveOptions_[static_cast<unsigned>(nextMove_.first)].begin();
        }
        if (reachedBeginning) {
            break;
        }

        std::pair<VertexIdx, unsigned> next = *nextMove_.second;
        ++nextMove_.second;

        const VertexIdx node = next.first;
        const unsigned proc = next.second;

        if (useMemoryConstraint_ && ViolatesMemConstraint(node, proc, dir - 1)) {
            continue;
        }

        StepAuxData moveData;
        int costDiff = MoveCostChange(node, proc, dir - 1, moveData);

        if (!steepestAscent_ && costDiff < 0) {
            ExecuteMove(node, proc, dir - 1, moveData);
            if (shrink_ && moveData.canShrink_) {
                Init();
            }

            return true;
        } else if (static_cast<CostType>(static_cast<int>(cost_) + costDiff) < bestCost) {
            bestCost = static_cast<CostType>(static_cast<int>(cost_) + costDiff);
            bestMove = next;
            bestMoveData = moveData;
            bestDir = dir - 1;
        }
    }

    if (bestCost == cost_) {
        return false;
    }

    ExecuteMove(bestMove.first, bestMove.second, bestDir, bestMoveData);
    if (shrink_ && bestMoveData.canShrink_) {
        Init();
    }

    return true;
}

// Check if move violates mem constraints
template <typename GraphT>
bool HillClimbingScheduler<GraphT>::ViolatesMemConstraint(VertexIdx node, unsigned processor, int where) {
    if (memoryUsed_[processor][static_cast<unsigned>(static_cast<int>(schedule_->AssignedSuperstep(node)) + where)]
            + schedule_->GetInstance().GetComputationalDag().VertexMemWeight(node)
        > schedule_->GetInstance().MemoryBound(processor)) {    // TODO ANDRAS double check change
        return true;
    }

    return false;
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::CreateSupstepLists() {
    const unsigned p = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    schedule_->UpdateNumberOfSupersteps();
    const unsigned m = schedule_->NumberOfSupersteps();

    supsteplists_.clear();
    supsteplists_.resize(m, std::vector<std::list<VertexIdx>>(p));

    for (VertexIdx node : TopSortView(g)) {
        supsteplists_[schedule_->AssignedSuperstep(node)][schedule_->AssignedProcessor(node)].push_back(node);
    }
}

}    // namespace osp
