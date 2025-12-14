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
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(IsComputationalDagV<GraphT>, "Graph_t must satisfy the computational_dag concept");

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
    std::vector<std::vector<std::list<VertexIdx>>> supStepLists_;
    std::vector<std::vector<std::vector<bool>>> canMove_;
    std::vector<std::list<std::pair<VertexIdx, unsigned>>> moveOptions_;
    std::vector<std::vector<std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>>> movePointer_;
    std::vector<std::vector<std::map<unsigned, unsigned>>> succSteps_;
    std::vector<std::vector<CostType>> workCost_, sent_, received_, commCost_;
    std::vector<std::set<std::pair<CostType, unsigned>>> workCostList_, commCostList_;
    std::vector<std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>> workCostPointer_, commCostPointer_;
    std::vector<typename std::list<VertexIdx>::iterator> supStepListPointer_;
    std::pair<int, typename std::list<std::pair<VertexIdx, unsigned>>::iterator> nextMove_;
    bool hCwithLatency_ = true;

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

    void AddMoveOption(VertexIdx node, unsigned p, Direction dir);

    void EraseMoveOption(VertexIdx node, unsigned p, Direction dir);
    void EraseMoveOptionsEarlier(VertexIdx node);
    void EraseMoveOptionsAt(VertexIdx node);
    void EraseMoveOptionsLater(VertexIdx node);
    void EraseMoveOptions(VertexIdx node);

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupStepLists();

    // For memory constraints
    bool useMemoryConstraint_ = false;
    std::vector<std::vector<VMemwT<GraphT>>> memoryUsed_;
    bool ViolatesMemConstraint(VertexIdx node, unsigned processor, int where);

    // Compute the cost change incurred by a potential move
    int MoveCostChange(VertexIdx node, unsigned p, int where, StepAuxData &changing);

    // Execute a chosen move, updating the schedule and the data structures
    void ExecuteMove(VertexIdx node, unsigned newProc, int where, const StepAuxData &changing);

    // Single hill climbing step
    bool Improve();

  public:
    HillClimbingScheduler() : ImprovementScheduler<GraphT>() {}

    virtual ~HillClimbingScheduler() = default;

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &inputSchedule) override;

    // call with time/step limits
    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &inputSchedule);
    virtual ReturnStatus ImproveScheduleWithStepLimit(BspSchedule<GraphT> &inputSchedule, const unsigned stepLimit = 10);

    // setting parameters
    void SetSteepestAscend(bool steepestAscent) { steepestAscent_ = steepestAscent; }

    void SetShrink(bool shrink) { shrink_ = shrink; }

    virtual std::string getScheduleName() const override { return "HillClimbing"; }
};

template <typename GraphT>
ReturnStatus HillClimbingScheduler<GraphT>::ImproveSchedule(BspSchedule<GraphT> &inputSchedule) {
    ImprovementScheduler<GraphT>::setTimeLimitSeconds(600U);
    return ImproveScheduleWithTimeLimit(inputSchedule);
}

// Main method for hill climbing (with time limit)
template <typename GraphT>
ReturnStatus HillClimbingScheduler<GraphT>::ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &inputSchedule) {
    schedule_ = &inputSchedule;

    CreateSupStepLists();
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

    CreateSupStepLists();
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
        schedule_->shrinkByMergingSupersteps();
        CreateSupStepLists();
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
            for (auto it = supStepLists_[step][proc].begin(); it != supStepLists_[step][proc].end(); ++it) {
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
            for (const VertexIdx node : supStepLists_[step][proc]) {
                workCost_[step][proc] += schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node);
            }

            std::pair<CostType, unsigned> entry(workCost_[step][proc], proc);
            workCostPointer_[step][proc] = workCostList_[step].insert(entry).first;
        }
        workCost[step] = (--workCostList_[step].end())->first;
    }

    cost_ = workCost[0];
    std::vector<std::vector<bool>> present(n, std::vector<bool>(p, false));
    for (unsigned step = 0; step < m - schedule_->getStaleness(); ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (const VertexIdx node : supStepLists_[step + schedule_->getStaleness()][proc]) {
                for (const VertexIdx &pred : g.Parents(node)) {
                    if (schedule_->AssignedProcessor(node) != schedule_->AssignedProcessor(pred)
                        && !present[pred][schedule_->AssignedProcessor(node)]) {
                        present[pred][schedule_->AssignedProcessor(node)] = true;
                        sent_[step][schedule_->AssignedProcessor(pred)]
                            += schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                               * schedule_->GetInstance().GetArchitecture().sendCosts(schedule_->AssignedProcessor(pred),
                                                                                      schedule_->AssignedProcessor(node));
                        received_[step][schedule_->AssignedProcessor(node)]
                            += schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                               * schedule_->GetInstance().GetArchitecture().sendCosts(schedule_->AssignedProcessor(pred),
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

        if (schedule_->getStaleness() == 1) {
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
                    && schedule_->GetInstance().isCompatible(node, proc)) {
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
                    && schedule_->GetInstance().isCompatible(node, proc)) {
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
                for (VertexIdx node : supStepLists_[step][from]) {
                    if (schedule_->GetInstance().isCompatible(node, to)) {
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
            >= static_cast<int>(schedule_->AssignedSuperstep(node)) - static_cast<int>(schedule_->getStaleness())) {
            predProc.insert(schedule_->AssignedProcessor(pred));
        }
    }
    if (schedule_->getStaleness() == 2) {
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
            >= static_cast<int>(schedule_->AssignedSuperstep(node)) - static_cast<int>(schedule_->getStaleness()) + 1) {
            return;
        }
    }

    for (const VertexIdx &succ : schedule_->GetInstance().GetComputationalDag().Children(node)) {
        if (schedule_->AssignedSuperstep(succ) <= schedule_->AssignedSuperstep(node) + schedule_->getStaleness() - 1) {
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
        if (schedule_->AssignedSuperstep(succ) <= schedule_->AssignedSuperstep(node) + schedule_->getStaleness()) {
            succProc.insert(schedule_->AssignedProcessor(succ));
        }
    }
    if (schedule_->getStaleness() == 2) {
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
            if (schedule_->getStaleness() == 2) {
                EraseMoveOptionsEarlier(pred);
                UpdateNodeMovesEarlier(pred);
            }
        }
        for (const VertexIdx &succ : g.Children(node)) {
            EraseMoveOptionsEarlier(succ);
            UpdateNodeMovesEarlier(succ);
            if (schedule_->getStaleness() == 2) {
                EraseMoveOptionsAt(succ);
                UpdateNodeMovesAt(succ);
            }
        }
    }
    if (where == 1) {
        for (const VertexIdx &pred : g.Parents(node)) {
            EraseMoveOptionsLater(pred);
            UpdateNodeMovesLater(pred);
            if (schedule_->getStaleness() == 2) {
                EraseMoveOptionsAt(pred);
                UpdateNodeMovesAt(pred);
            }
        }
        for (const VertexIdx &succ : g.Children(node)) {
            EraseMoveOptionsEarlier(succ);
            UpdateNodeMovesEarlier(succ);
            EraseMoveOptionsAt(succ);
            UpdateNodeMovesAt(succ);
            if (schedule_->getStaleness() == 2) {
                EraseMoveOptionsLater(succ);
                UpdateNodeMovesLater(succ);
            }
        }
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::AddMoveOption(const VertexIdx node, const unsigned p, const Direction dir) {
    if (!canMove_[dir][node][p] && schedule_->GetInstance().isCompatible(node, p)) {
        canMove_[dir][node][p] = true;
        moveOptions_[dir].emplace_back(node, p);
        movePointer_[dir][node][p] = --moveOptions_[dir].end();
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::EraseMoveOption(VertexIdx node, unsigned p, Direction dir) {
    canMove_[dir][node][p] = false;
    if (nextMove_.first == dir && nextMove_.second->first == node && nextMove_.second->second == p) {
        ++nextMove_.second;
    }
    moveOptions_[dir].erase(movePointer_[dir][node][p]);
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
int HillClimbingScheduler<GraphT>::MoveCostChange(const VertexIdx node, unsigned p, const int where, StepAuxData &changing) {
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
            if (step == 0 || schedule_->getStaleness() == 1) {    // incorporate immediately into cost change
                change -= static_cast<int>(itBest->first) - static_cast<int>(maxAfterRemoval);
            } else {
                newWorkCost[step] = maxAfterRemoval;
                affectedSteps.insert(step - 1);
            }
        }
    }

    const CostType maxBeforeAddition = (newWorkCost.find(newStep) != newWorkCost.end())
                                           ? newWorkCost[newStep]
                                           : ((where == 0) ? maxAfterRemoval : workCostList_[newStep].rbegin()->first);

    if (workCost_[newStep][p] + schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node) > maxBeforeAddition) {
        if (newStep == 0 || schedule_->getStaleness() == 1) {    // incorporate immediately into cost change
            change
                += static_cast<int>(workCost_[newStep][p] + schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node))
                   - static_cast<int>(maxBeforeAddition);
        } else {
            newWorkCost[newStep] = workCost_[newStep][p] + schedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node);
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
        for (const VertexIdx &pred : g.Parents(node)) {
            if (schedule_->AssignedProcessor(pred) == p) {
                continue;
            }

            const auto firstUse = *succSteps_[pred][p].begin();
            const bool skip = firstUse.first < step || (firstUse.first == step && where >= 0 && firstUse.second > 1);
            if (!skip) {
                sentInc.emplace_back(step - schedule_->getStaleness(),
                                     schedule_->AssignedProcessor(pred),
                                     -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                       * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                           schedule_->AssignedProcessor(pred), p)));
                recInc.emplace_back(step - schedule_->getStaleness(),
                                    p,
                                    -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                          schedule_->AssignedProcessor(pred), p)));
                sentInc.emplace_back(newStep - schedule_->getStaleness(),
                                     schedule_->AssignedProcessor(pred),
                                     static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                          schedule_->AssignedProcessor(pred), p)));
                recInc.emplace_back(newStep - schedule_->getStaleness(),
                                    p,
                                    static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                     * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                         schedule_->AssignedProcessor(pred), p)));
            }
        }
    } else {
        for (const VertexIdx &pred : g.Parents(node)) {
            // Comm. cost of sending pred to oldProc
            auto firstUse = succSteps_[pred][oldProc].begin();
            bool skip = (schedule_->AssignedProcessor(pred) == oldProc) || firstUse->first < step
                        || (firstUse->first == step && firstUse->second > 1);
            if (!skip) {
                sentInc.emplace_back(step - schedule_->getStaleness(),
                                     schedule_->AssignedProcessor(pred),
                                     -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                       * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                           schedule_->AssignedProcessor(pred), oldProc)));
                recInc.emplace_back(step - schedule_->getStaleness(),
                                    oldProc,
                                    -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                          schedule_->AssignedProcessor(pred), oldProc)));
                ++firstUse;
                if (firstUse != succSteps_[pred][oldProc].end()) {
                    const unsigned nextStep = firstUse->first;
                    sentInc.emplace_back(nextStep - schedule_->getStaleness(),
                                         schedule_->AssignedProcessor(pred),
                                         static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                          * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                              schedule_->AssignedProcessor(pred), oldProc)));
                    recInc.emplace_back(nextStep - schedule_->getStaleness(),
                                        oldProc,
                                        static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                         * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                             schedule_->AssignedProcessor(pred), oldProc)));
                }
            }

            // Comm. cost of sending pred to p
            firstUse = succSteps_[pred][p].begin();
            skip = (schedule_->AssignedProcessor(pred) == p)
                   || ((firstUse != succSteps_[pred][p].end()) && (firstUse->first <= newStep));
            if (!skip) {
                sentInc.emplace_back(newStep - schedule_->getStaleness(),
                                     schedule_->AssignedProcessor(pred),
                                     static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                      * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                          schedule_->AssignedProcessor(pred), p)));
                recInc.emplace_back(newStep - schedule_->getStaleness(),
                                    p,
                                    static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                     * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                         schedule_->AssignedProcessor(pred), p)));
                if (firstUse != succSteps_[pred][p].end()) {
                    sentInc.emplace_back(firstUse->first - schedule_->getStaleness(),
                                         schedule_->AssignedProcessor(pred),
                                         -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                           * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                               schedule_->AssignedProcessor(pred), p)));
                    recInc.emplace_back(firstUse->first - schedule_->getStaleness(),
                                        p,
                                        -static_cast<int>(schedule_->GetInstance().GetComputationalDag().VertexCommWeight(pred)
                                                          * schedule_->GetInstance().GetArchitecture().sendCosts(
                                                              schedule_->AssignedProcessor(pred), p)));
                }
            }
        }
    }

    // Comm cost change calculation logic...
    return change;
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::CreateSupStepLists() {
    supStepLists_.clear();
    supStepLists_.resize(schedule_->NumberOfSupersteps(),
                         std::vector<std::list<VertexIdx>>(schedule_->GetInstance().GetArchitecture().NumberOfProcessors()));

    for (const VertexIdx &node : schedule_->GetInstance().GetComputationalDag().Vertices()) {
        supStepLists_[schedule_->AssignedSuperstep(node)][schedule_->AssignedProcessor(node)].push_back(node);
    }
}

template <typename GraphT>
void HillClimbingScheduler<GraphT>::ExecuteMove(VertexIdx node, unsigned newProc, int where, const StepAuxData &changing) {
    // Implementation of ExecuteMove
}

template <typename GraphT>
bool HillClimbingScheduler<GraphT>::Improve() {
    // Implementation of Improve
    return false;
}

template <typename GraphT>
bool HillClimbingScheduler<GraphT>::ViolatesMemConstraint(VertexIdx node, unsigned processor, int where) {
    if (useMemoryConstraint_) {
        // Implementation of memory constraint check
    }
    return false;
}

}    // namespace osp
