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

#include "bsp/model/BspSchedule.hpp"
#include "bsp/scheduler/ImprovementScheduler.hpp"
#include "graph_algorithms/directed_graph_top_sort.hpp"

namespace osp{

template<typename Graph_t>
class HillClimbingScheduler : public ImprovementScheduler<Graph_t> {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(is_computational_dag_v<Graph_t>, "Graph_t must satisfy the computational_dag concept");

    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = v_workw_t<Graph_t>;

    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>, "HillClimbing requires work and comm. weights to have the same type.");

  public:
    enum Direction { EARLIER = 0, AT, LATER };
    static const int NumDirections = 3;

    // aux structure for efficiently storing the changes incurred by a potential HC step
    struct stepAuxData {
        cost_type newCost;
        std::map<std::pair<unsigned, unsigned>, int> sentChange, recChange;
        bool canShrink = false;
    };

  private:

    BspSchedule<Graph_t> *schedule;
    cost_type cost=0;

    // Main parameters for runnign algorithm
    bool shrink = true;
    bool steepestAscent = false;

    // aux data structures
    std::vector<std::vector<std::list<vertex_idx>>> supsteplists;
    std::vector<std::vector<std::vector<bool>>> canMove;
    std::vector<std::list<std::pair<vertex_idx, unsigned> > > moveOptions;
    std::vector<std::vector<std::vector<typename std::list<std::pair<vertex_idx, unsigned> >::iterator>>> movePointer;
    std::vector<std::vector<std::map<unsigned, unsigned>>> succSteps;
    std::vector<std::vector<cost_type> > workCost, sent, received, commCost;
    std::vector<std::set<std::pair<cost_type, unsigned> > > workCostList, commCostList;
    std::vector<std::vector<typename std::set<std::pair<cost_type, unsigned> >::iterator> > workCostPointer, commCostPointer;
    std::vector<typename std::list<vertex_idx>::iterator> supStepListPointer;
    std::pair<int, typename std::list<std::pair<vertex_idx, unsigned> >::iterator> nextMove;
    bool HCwithLatency = true;

    // for improved candidate selection
    std::deque<std::tuple<vertex_idx, unsigned, int> > promisingMoves;
    bool findPromisingMoves = true;

    // Initialize data structures (based on current schedule)
    void Init();
    void updatePromisingMoves();

    // Functions to compute and update the std::list of possible moves
    void updateNodeMovesEarlier(vertex_idx node);
    void updateNodeMovesAt(vertex_idx node);
    void updateNodeMovesLater(vertex_idx node);
    void updateNodeMoves(vertex_idx node);
    void updateMoveOptions(vertex_idx node, int where);

    void addMoveOption(vertex_idx node, unsigned p, Direction dir);

    void eraseMoveOption(vertex_idx node, unsigned p, Direction dir);
    void eraseMoveOptionsEarlier(vertex_idx node);
    void eraseMoveOptionsAt(vertex_idx node);
    void eraseMoveOptionsLater(vertex_idx node);
    void eraseMoveOptions(vertex_idx node);

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupstepLists();

    // Combine subsequent supersteps whenever there is no communication inbetween
    void RemoveNeedlessSupSteps();

    // For memory constraints
    bool use_memory_constraint = false;
    std::vector<std::vector<v_memw_t<Graph_t>>> memory_used;
    bool violatesMemConstraint(vertex_idx node, unsigned processor, int where);

    // Compute the cost change incurred by a potential move
    int moveCostChange(vertex_idx node, unsigned p, int where, stepAuxData &changing);

    // Execute a chosen move, updating the schedule and the data structures
    void executeMove(vertex_idx node, unsigned newProc, int where, const stepAuxData &changing);

    // Single hill climbing step
    bool Improve();

  public:
    HillClimbingScheduler() : ImprovementScheduler<Graph_t>() {}

    virtual ~HillClimbingScheduler() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule<Graph_t> &input_schedule) override;

    //call with time/step limits
    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule<Graph_t> &input_schedule) override;
    virtual RETURN_STATUS improveScheduleWithStepLimit(BspSchedule<Graph_t> &input_schedule, const unsigned stepLimit = 10);

    //setting parameters
    void setSteepestAscend(bool steepestAscent_) {steepestAscent = steepestAscent_;}
    void setShrink(bool shrink_) {shrink = shrink_;}

    virtual std::string getScheduleName() const override { return "HillClimbing"; }
};

template<typename Graph_t>
RETURN_STATUS HillClimbingScheduler<Graph_t>::improveSchedule(BspSchedule<Graph_t> &input_schedule) {

    ImprovementScheduler<Graph_t>::setTimeLimitSeconds(600U);
    return improveScheduleWithTimeLimit(input_schedule);
}

// Main method for hill climbing (with time limit)
template<typename Graph_t>
RETURN_STATUS HillClimbingScheduler<Graph_t>::improveScheduleWithTimeLimit(BspSchedule<Graph_t> &input_schedule) {

    schedule = &input_schedule;

    std::cout<<schedule->computeCosts()<<" "<<schedule->computeWorkCosts()<<std::endl;

    CreateSupstepLists();
    Init();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    int counter = 0;
    while (Improve())
        if ((++counter) == 10) {
            counter = 0;
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (elapsed >= ImprovementScheduler<Graph_t>::timeLimitSeconds) {
                std::cout << "Hill Climbing was shut down due to time limit." << std::endl;
                break;
            }
        }

    return RETURN_STATUS::OSP_SUCCESS;
}

template<typename Graph_t>
// Hill climbing with step limit (designed as an ingredient for multilevel algorithms, no safety checks)
RETURN_STATUS HillClimbingScheduler<Graph_t>::improveScheduleWithStepLimit(BspSchedule<Graph_t> &input_schedule, const unsigned stepLimit) {

    schedule = &input_schedule;
    
    CreateSupstepLists();
    Init();
    for (unsigned step = 0; step < stepLimit; ++step)
        if (!Improve())
            break;

    return RETURN_STATUS::OSP_SUCCESS;
}

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::Init() {
    if(shrink)
    {
        RemoveNeedlessSupSteps();
        CreateSupstepLists();
    }

    const vertex_idx N = schedule->getInstance().getComputationalDag().num_vertices();
    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const unsigned M = schedule->numberOfSupersteps();
    const Graph_t &G = schedule->getInstance().getComputationalDag();

    // Movement options
    canMove.clear();
    canMove.resize(NumDirections, std::vector<std::vector<bool>>(N, std::vector<bool>(P, false)));
    moveOptions.clear();
    moveOptions.resize(NumDirections);
    movePointer.clear();
    movePointer.resize(NumDirections, std::vector<std::vector<typename std::list<std::pair<vertex_idx, unsigned> >::iterator>>(
                                          N, std::vector<typename std::list<std::pair<vertex_idx, unsigned> >::iterator>(P)));

    // Value use lists
    succSteps.clear();
    succSteps.resize(N, std::vector<std::map<unsigned, unsigned>>(P));
    for (vertex_idx node = 0; node < N; ++node)
        for (const vertex_idx &succ : G.children(node)) {
            if (succSteps[node][schedule->assignedProcessor(succ)].find(schedule->assignedSuperstep(succ)) ==
                succSteps[node][schedule->assignedProcessor(succ)].end())
                succSteps[node][schedule->assignedProcessor(succ)].insert({schedule->assignedSuperstep(succ), 1U});
            else
                succSteps[node][schedule->assignedProcessor(succ)].at(schedule->assignedSuperstep(succ)) += 1;
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
    workCostPointer.resize(M, std::vector<typename std::set<std::pair<cost_type, unsigned> >::iterator>(P));
    commCostPointer.clear();
    commCostPointer.resize(M - 1, std::vector<typename std::set<std::pair<cost_type, unsigned> >::iterator>(P));

    // Supstep std::list pointers
    supStepListPointer.clear();
    supStepListPointer.resize(N);
    for (unsigned step = 0; step < M; ++step)
        for (unsigned proc = 0; proc < P; ++proc)
            for (auto it = supsteplists[step][proc].begin(); it != supsteplists[step][proc].end(); ++it)
                supStepListPointer[*it] = it;

    // Compute movement options
    for (vertex_idx node = 0; node < N; ++node)
        updateNodeMoves(node);

    nextMove.first = 0;
    nextMove.second = moveOptions[0].begin();

    // Compute cost data
    cost = 0;
    for (unsigned step = 0; step < M; ++step) {
        for (unsigned proc = 0; proc < P; ++proc) {
            for (const vertex_idx node : supsteplists[step][proc])
                workCost[step][proc] += schedule->getInstance().getComputationalDag().vertex_work_weight(node);

            std::pair<cost_type, unsigned> entry(workCost[step][proc], proc);
            workCostPointer[step][proc] = workCostList[step].insert(entry).first;
        }
        cost += (--workCostList[step].end())->first;
    }

    std::vector<std::vector<bool>> present(N, std::vector<bool>(P, false));
    for (unsigned step = 0; step < M - 1; ++step) {
        for (unsigned proc = 0; proc < P; ++proc)
            for (const vertex_idx node : supsteplists[step + 1][proc])
                for (const vertex_idx &pred : G.parents(node))
                    if (schedule->assignedProcessor(node) != schedule->assignedProcessor(pred) && !present[pred][schedule->assignedProcessor(node)]) {
                        present[pred][schedule->assignedProcessor(node)] = true;
                        sent[step][schedule->assignedProcessor(pred)] +=
                            schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), schedule->assignedProcessor(node));
                        received[step][schedule->assignedProcessor(node)] +=
                            schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), schedule->assignedProcessor(node));
                    }

        for (unsigned proc = 0; proc < P; ++proc) {
            commCost[step][proc] = std::max(sent[step][proc], received[step][proc]);
            std::pair<cost_type, unsigned> entry(commCost[step][proc], proc);
            commCostPointer[step][proc] = commCostList[step].insert(entry).first;
        }
        cost += schedule->getInstance().getArchitecture().communicationCosts() * commCostList[step].rbegin()->first
                + schedule->getInstance().getArchitecture().synchronisationCosts();
    }

    updatePromisingMoves();

    // memory_constraints
    if(use_memory_constraint)
    {
        memory_used.clear();
        memory_used.resize(P, std::vector<v_memw_t<Graph_t>>(M, 0));
        for (vertex_idx node = 0; node < N; ++node)
            memory_used[schedule->assignedProcessor(node)][schedule->assignedSuperstep(node)] += schedule->getInstance().getComputationalDag().vertex_mem_weight(node);
    }

};

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::updatePromisingMoves()
{
    if(!findPromisingMoves)
        return;

    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const Graph_t &G = schedule->getInstance().getComputationalDag();

    promisingMoves.clear();
    for(vertex_idx node=0; node < schedule->getInstance().getComputationalDag().num_vertices(); ++node)
    {
        std::vector<unsigned> nrPredOnProc(P, 0);
        for(const vertex_idx &pred : G.parents(node))
            ++nrPredOnProc[schedule->assignedProcessor(pred)];

        unsigned otherProcUsed = 0;
        for(unsigned proc=0; proc<P; ++proc)
            if(schedule->assignedProcessor(node)!=proc && nrPredOnProc[proc]>0)
                ++otherProcUsed;
                
        if(otherProcUsed==1)
            for(unsigned proc=0; proc<P; ++proc)
                if(schedule->assignedProcessor(node)!=proc && nrPredOnProc[proc]>0 && schedule->getInstance().isCompatible(node,proc))
                {
                    promisingMoves.push_back(std::make_tuple(node, proc, EARLIER));
                    promisingMoves.push_back(std::make_tuple(node, proc, AT));
                    promisingMoves.push_back(std::make_tuple(node, proc, LATER));
                }

        std::vector<unsigned> nrSuccOnProc(P, 0);
        for(const vertex_idx &succ : G.children(node))
            ++nrSuccOnProc[schedule->assignedProcessor(succ)];

        otherProcUsed = 0;
        for(unsigned proc=0; proc<P; ++proc)
            if(schedule->assignedProcessor(node)!=proc && nrSuccOnProc[proc]>0)
                ++otherProcUsed;

        if(otherProcUsed==1)
            for(unsigned proc=0; proc<P; ++proc)
                if(schedule->assignedProcessor(node)!=proc && nrSuccOnProc[proc]>0 && schedule->getInstance().isCompatible(node,proc))
                {
                    promisingMoves.push_back(std::make_tuple(node, proc, EARLIER));
                    promisingMoves.push_back(std::make_tuple(node, proc, AT));
                    promisingMoves.push_back(std::make_tuple(node, proc, LATER));
                }
        }

    for(unsigned step=0; step < schedule->numberOfSupersteps(); ++step)
    {
        std::list<unsigned> minProcs, maxProcs;
        cost_type minWork=std::numeric_limits<cost_type>::max(), maxWork=std::numeric_limits<cost_type>::min();
        for(unsigned proc=0; proc<P; ++proc)
        {
            if(workCost[step][proc]> maxWork)
                maxWork=workCost[step][proc];
            if(workCost[step][proc]< minWork)
                minWork=workCost[step][proc];
        }
        for(unsigned proc=0; proc<P; ++proc)
        {
            if(workCost[step][proc]==minWork)
                minProcs.push_back(proc);
            if(workCost[step][proc]==maxWork)
                maxProcs.push_back(proc);
        }
        for(unsigned to: minProcs)
            for(unsigned from: maxProcs)
                for(vertex_idx node : supsteplists[step][from])
                    if(schedule->getInstance().isCompatible(node, to))
                        promisingMoves.push_back(std::make_tuple(node,to, AT));
    }
}

// Functions to compute and update the std::list of possible moves
template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::updateNodeMovesEarlier(const vertex_idx node) {
    if (schedule->assignedSuperstep(node) == 0)
        return;

    std::set<unsigned> predProc; 
    for (const vertex_idx &pred : schedule->getInstance().getComputationalDag().parents(node)) {
        if (schedule->assignedSuperstep(pred) == schedule->assignedSuperstep(node))
            return;
        if (schedule->assignedSuperstep(pred) == schedule->assignedSuperstep(node) - 1)
            predProc.insert(schedule->assignedProcessor(pred));
    }

    if (predProc.size() > 1)
        return;

    if (predProc.size() == 1)
        addMoveOption(node, *predProc.begin(), EARLIER);
    else
        for (unsigned proc = 0; proc < schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
            addMoveOption(node, proc, EARLIER);
};

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::updateNodeMovesAt(const vertex_idx node) {
    for (const vertex_idx &pred : schedule->getInstance().getComputationalDag().parents(node))
        if (schedule->assignedSuperstep(pred) == schedule->assignedSuperstep(node))
            return;

    for (const vertex_idx &succ : schedule->getInstance().getComputationalDag().children(node))
        if (schedule->assignedSuperstep(succ) == schedule->assignedSuperstep(node))
            return;

    for (unsigned proc = 0; proc < schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
        if (proc != schedule->assignedProcessor(node))
            addMoveOption(node, proc, AT);
};

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::updateNodeMovesLater(const vertex_idx node) {
    if (schedule->assignedSuperstep(node) == schedule->numberOfSupersteps() - 1)
        return;

    std::set<unsigned> succProc;
    for (const vertex_idx &succ : schedule->getInstance().getComputationalDag().children(node)) {
        if (schedule->assignedSuperstep(succ) == schedule->assignedSuperstep(node))
            return;
        if (schedule->assignedSuperstep(succ) == schedule->assignedSuperstep(node) + 1)
            succProc.insert(schedule->assignedProcessor(succ));
    }

    if (succProc.size() > 1)
        return;

    if (succProc.size() == 1)
        addMoveOption(node, *succProc.begin(), LATER);
    else
        for (unsigned proc = 0; proc < schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
            addMoveOption(node, proc, LATER);
};

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::updateNodeMoves(const vertex_idx node) {
    eraseMoveOptions(node);
    updateNodeMovesEarlier(node);
    updateNodeMovesAt(node);
    updateNodeMovesLater(node);
};

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::updateMoveOptions(vertex_idx node, int where)
{
    const Graph_t &G = schedule->getInstance().getComputationalDag();
    
    updateNodeMoves(node);
    if(where==0)
    {
        for(const vertex_idx &pred : G.parents(node))
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
        }
        for(const vertex_idx &succ : G.children(node))
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
        }
    }
    if(where==-1)
    {
        for(const vertex_idx &pred : G.parents(node))
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
            eraseMoveOptionsAt(pred);
            updateNodeMovesAt(pred);
        }
        for(const vertex_idx &succ : G.children(node))
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
        }
    }
    if(where==1)
    {
        for(const vertex_idx &pred : G.parents(node))
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
        }
        for(const vertex_idx &succ : G.children(node))
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
            eraseMoveOptionsAt(succ);
            updateNodeMovesAt(succ);
        }
    }
}

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::addMoveOption(const vertex_idx node, const unsigned p, const Direction dir) {
    if (!canMove[dir][node][p] && schedule->getInstance().isCompatible(node, p)) {
        canMove[dir][node][p] = true;
        moveOptions[dir].emplace_back(node, p);
        movePointer[dir][node][p] = --moveOptions[dir].end();
    }
};

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::eraseMoveOption(vertex_idx node, unsigned p, Direction dir)
{
    canMove[dir][node][p] = false;
    if(nextMove.first == dir && nextMove.second->first == node && nextMove.second->second == p)
        ++nextMove.second;
    moveOptions[dir].erase(movePointer[dir][node][p]);
}

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::eraseMoveOptionsEarlier(vertex_idx node)
{
    for(unsigned proc=0; proc<schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
        if(canMove[EARLIER][node][proc])
            eraseMoveOption(node, proc, EARLIER);
}

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::eraseMoveOptionsAt(vertex_idx node)
{
    for(unsigned proc=0; proc<schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
        if(canMove[AT][node][proc])
            eraseMoveOption(node, proc, AT);
}

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::eraseMoveOptionsLater(vertex_idx node)
{
    for(unsigned proc=0; proc<schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
        if(canMove[LATER][node][proc])
            eraseMoveOption(node, proc, LATER);
}

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::eraseMoveOptions(vertex_idx node)
{
    eraseMoveOptionsEarlier(node);
    eraseMoveOptionsAt(node);
    eraseMoveOptionsLater(node);
}

// Compute the cost change incurred by a potential move
template<typename Graph_t>
int HillClimbingScheduler<Graph_t>::moveCostChange(const vertex_idx node, unsigned p, const int where, stepAuxData &changing) {
    const unsigned step = schedule->assignedSuperstep(node);
    const unsigned new_step = static_cast<unsigned>(static_cast<int>(step) + where);
    unsigned oldProc = schedule->assignedProcessor(node);
    int change = 0;

    const Graph_t &G = schedule->getInstance().getComputationalDag();

    // Work cost change
    const auto itBest = --workCostList[step].end();
    cost_type maxAfterRemoval = itBest->first;
    if (itBest->second == oldProc) {
        auto itNext = itBest;
        --itNext;
        maxAfterRemoval = std::max(itBest->first - schedule->getInstance().getComputationalDag().vertex_work_weight(node), itNext->first);
        change -= static_cast<int>(itBest->first) - static_cast<int>(maxAfterRemoval);
    }

    const cost_type maxBeforeAddition = (where == 0) ? maxAfterRemoval : workCostList[new_step].rbegin()->first;
    if (workCost[new_step][p] + schedule->getInstance().getComputationalDag().vertex_work_weight(node) > maxBeforeAddition)
        change += static_cast<int>(workCost[new_step][p] + schedule->getInstance().getComputationalDag().vertex_work_weight(node)) - static_cast<int>(maxBeforeAddition);

    // Comm cost change
    std::list<std::tuple<unsigned, unsigned, int> > sentInc, recInc;
    //  -outputs
    if (p != oldProc) {
        for (unsigned j = 0; j < schedule->getInstance().getArchitecture().numberOfProcessors(); ++j) {
            if (succSteps[node][j].empty())
                continue;

            unsigned affectedStep = succSteps[node][j].begin()->first - 1U;
            if (j == p) {
                sentInc.emplace_back(affectedStep, oldProc, 
                                     -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(node) * schedule->getInstance().getArchitecture().sendCosts(oldProc, j)));
                recInc.emplace_back(affectedStep, p, -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(node) * schedule->getInstance().getArchitecture().sendCosts(oldProc, j)));
            } else if (j == oldProc) {
                recInc.emplace_back(affectedStep, oldProc, static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(node) * schedule->getInstance().getArchitecture().sendCosts(p, j)));
                sentInc.emplace_back(affectedStep, p, static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(node) * schedule->getInstance().getArchitecture().sendCosts(p, j)));
            } else {
                sentInc.emplace_back(affectedStep, oldProc,
                                     -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(node) * schedule->getInstance().getArchitecture().sendCosts(oldProc, j)));
                recInc.emplace_back(affectedStep, j, -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(node) * schedule->getInstance().getArchitecture().sendCosts(oldProc, j)));
                sentInc.emplace_back(affectedStep, p, static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(node) * schedule->getInstance().getArchitecture().sendCosts(p, j)));
                recInc.emplace_back(affectedStep, j, static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(node) * schedule->getInstance().getArchitecture().sendCosts(p, j)));
            }
        }
    }

    //  -inputs
    if (p == oldProc)
        for (const vertex_idx &pred : G.parents(node)) {
            if (schedule->assignedProcessor(pred) == p)
                continue;

            const auto firstUse = *succSteps[pred][p].begin();
            const bool skip = firstUse.first < step || (firstUse.first == step && where >= 0 && firstUse.second > 1);
            if (!skip) {
                sentInc.emplace_back(step - 1, schedule->assignedProcessor(pred),
                                     -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                recInc.emplace_back(step - 1, p,
                                    -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                sentInc.emplace_back(new_step - 1, schedule->assignedProcessor(pred),
                                     static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                recInc.emplace_back(new_step - 1, p,
                                    static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
            }
        }
    else
        for (const vertex_idx &pred : G.parents(node)) {
            // Comm. cost of sending pred to oldProc
            auto firstUse = succSteps[pred][oldProc].begin();
            bool skip = (schedule->assignedProcessor(pred) == oldProc) || firstUse->first < step ||
                        (firstUse->first == step && firstUse->second > 1);
            if (!skip) {
                sentInc.emplace_back(step - 1, schedule->assignedProcessor(pred),
                                     -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), oldProc)));
                recInc.emplace_back(step - 1, oldProc,
                                    -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), oldProc)));
                ++firstUse;
                if (firstUse != succSteps[pred][oldProc].end()) {
                    const unsigned nextStep = firstUse->first;
                    sentInc.emplace_back(nextStep - 1, schedule->assignedProcessor(pred),
                                         static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) *
                                             schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), oldProc)));
                    recInc.emplace_back(nextStep - 1, oldProc,
                                        static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) *
                                            schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), oldProc)));
                }
            }

            // Comm. cost of sending pred to p
            firstUse = succSteps[pred][p].begin();
            skip = (schedule->assignedProcessor(pred) == p) ||
                   ((firstUse != succSteps[pred][p].end()) && (firstUse->first <= new_step));
            if (!skip) {
                sentInc.emplace_back(new_step - 1, schedule->assignedProcessor(pred),
                                     static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                recInc.emplace_back(new_step - 1, p,
                                    static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                if (firstUse != succSteps[pred][p].end()) {
                    sentInc.emplace_back(firstUse->first - 1, schedule->assignedProcessor(pred),
                                         -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                    recInc.emplace_back(firstUse->first - 1, p,
                                        -static_cast<int>(schedule->getInstance().getComputationalDag().vertex_comm_weight(pred) * schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p)));
                }
            }
        }

    //  -process changes
    changing.sentChange.clear();
    changing.recChange.clear();
    std::set<unsigned> affectedSteps;
    for (auto entry : sentInc) {
        const unsigned e_step = std::get<0>(entry);
        const unsigned e_proc = std::get<1>(entry);
        const int e_increase = std::get<2>(entry);
        affectedSteps.insert(e_step);
        auto itr = changing.sentChange.find(std::make_pair(e_step, e_proc));
        if (itr == changing.sentChange.end())
            changing.sentChange.insert({std::make_pair(e_step, e_proc), e_increase});
        else
            itr->second += e_increase;
    }
    for (auto entry : recInc) {
        const unsigned e_step = std::get<0>(entry);
        const unsigned e_proc = std::get<1>(entry);
        const int e_increase = std::get<2>(entry);
        affectedSteps.insert(e_step);
        auto itr = changing.recChange.find(std::make_pair(e_step, e_proc));
        if (itr == changing.recChange.end())
            changing.recChange.insert({std::make_pair(e_step, e_proc), e_increase});
        else
            itr->second += e_increase;
    }

    auto itrSent = changing.sentChange.begin(), itrRec = changing.recChange.begin();
    for (const unsigned sstep : affectedSteps) {
        int newMax = 0;
        for (unsigned j = 0; j < schedule->getInstance().getArchitecture().numberOfProcessors(); ++j) {
            int diff = (itrSent != changing.sentChange.end() && itrSent->first.first == sstep && itrSent->first.second == j)
                           ? (itrSent++)->second
                           : 0;
            if (static_cast<int>(sent[sstep][j]) + diff > newMax)
                newMax = static_cast<int>(sent[sstep][j]) + diff;
            diff = (itrRec != changing.recChange.end() && itrRec->first.first == sstep && itrRec->first.second == j)
                       ? (itrRec++)->second
                       : 0;
            if (static_cast<int>(received[sstep][j]) + diff > newMax)
                newMax = static_cast<int>(received[sstep][j]) + diff;
        }
        change += static_cast<int>(schedule->getInstance().getArchitecture().communicationCosts()) * (newMax - static_cast<int>(commCostList[sstep].rbegin()->first));

        if (HCwithLatency) {
            if (newMax > 0 && commCostList[sstep].rbegin()->first == 0) {
                change += static_cast<int>(schedule->getInstance().getArchitecture().synchronisationCosts());
            }
            if (newMax == 0 && commCostList[sstep].rbegin()->first > 0) {
                change -= static_cast<int>(schedule->getInstance().getArchitecture().synchronisationCosts());
                changing.canShrink = true;
            }
        }
    }

    changing.newCost = static_cast<cost_type>(static_cast<int>(cost) + change);
    return change;
};

// Execute a chosen move, updating the schedule and the data structures
template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::executeMove(const vertex_idx node, const unsigned newProc, const int where, const stepAuxData &changing) {
    unsigned oldStep = schedule->assignedSuperstep(node);
    unsigned newStep = static_cast<unsigned>(static_cast<int>(oldStep) + where);
    const unsigned oldProc = schedule->assignedProcessor(node);
    cost = changing.newCost;

    // Work cost change
    workCostList[oldStep].erase(workCostPointer[oldStep][oldProc]);
    workCost[oldStep][oldProc] -= schedule->getInstance().getComputationalDag().vertex_work_weight(node);
    workCostPointer[oldStep][oldProc] =
        workCostList[oldStep].insert(std::make_pair(workCost[oldStep][oldProc], oldProc)).first;

    workCostList[newStep].erase(workCostPointer[newStep][newProc]);
    workCost[newStep][newProc] += schedule->getInstance().getComputationalDag().vertex_work_weight(node);
    workCostPointer[newStep][newProc] =
        workCostList[newStep].insert(std::make_pair(workCost[newStep][newProc], newProc)).first;

    // Comm cost change
    for (const auto& update : changing.sentChange)
        sent[update.first.first][update.first.second] = static_cast<cost_type>(static_cast<int>(sent[update.first.first][update.first.second]) + update.second);
    for (const auto& update : changing.recChange)
        received[update.first.first][update.first.second] = static_cast<cost_type>(static_cast<int>(received[update.first.first][update.first.second]) + update.second);

    std::set<std::pair<unsigned, unsigned> > toUpdate;
    for (const auto& update : changing.sentChange)
        if (std::max(sent[update.first.first][update.first.second], received[update.first.first][update.first.second]) !=
            commCost[update.first.first][update.first.second])
            toUpdate.insert(std::make_pair(update.first.first, update.first.second));

    for (const auto& update : changing.recChange)
        if (std::max(sent[update.first.first][update.first.second], received[update.first.first][update.first.second]) !=
            commCost[update.first.first][update.first.second])
            toUpdate.insert(std::make_pair(update.first.first, update.first.second));

    for (const auto& update : toUpdate) {
        commCostList[update.first].erase(commCostPointer[update.first][update.second]);
        commCost[update.first][update.second] = std::max(sent[update.first][update.second], received[update.first][update.second]);
        commCostPointer[update.first][update.second] =
            commCostList[update.first].insert(std::make_pair(commCost[update.first][update.second], update.second)).first;
    }

    // update successor lists
    for (const vertex_idx &pred : schedule->getInstance().getComputationalDag().parents(node)) {
        auto itr = succSteps[pred][oldProc].find(oldStep);
        if ((--(itr->second)) == 0)
            succSteps[pred][oldProc].erase(itr);

        itr = succSteps[pred][newProc].find(newStep);
        if (itr == succSteps[pred][newProc].end())
            succSteps[pred][newProc].insert({newStep, 1U});
        else
            itr->second += 1;
    }

    // memory constraints, if any
    if(use_memory_constraint)
    {
        memory_used[schedule->assignedProcessor(node)][schedule->assignedSuperstep(node)] -= schedule->getInstance().getComputationalDag().vertex_mem_weight(node);
        memory_used[newProc][newStep] += schedule->getInstance().getComputationalDag().vertex_mem_weight(node);
    }

    // update data
    schedule->setAssignedProcessor(node, newProc);
    schedule->setAssignedSuperstep(node, newStep);
    supsteplists[oldStep][oldProc].erase(supStepListPointer[node]);
    supsteplists[newStep][newProc].push_back(node);
    supStepListPointer[node] = (--supsteplists[newStep][newProc].end());

    updateMoveOptions(node, where);
};

// Single hill climbing step
template<typename Graph_t>
bool HillClimbingScheduler<Graph_t>::Improve() {
    cost_type bestCost = cost;
    stepAuxData bestMoveData;
    std::pair<vertex_idx, unsigned> bestMove;
    int bestDir = 0;
    int startingDir = nextMove.first;

    // pre-selected "promising" moves
    while(!promisingMoves.empty() && !steepestAscent)
    {
        std::tuple<vertex_idx, unsigned, int> next = promisingMoves.front();
        promisingMoves.pop_front();

        const vertex_idx node = std::get<0>(next);
        const unsigned proc = std::get<1>(next);
        const int where = std::get<2>(next);

        if(!canMove[static_cast<Direction>(where)][node][proc])
            continue;
        
        if(use_memory_constraint && violatesMemConstraint(node, proc, where-1))
            continue;

        stepAuxData moveData;
        int costDiff = moveCostChange(node, proc, where-1, moveData);

        if(costDiff<0)
        {
            executeMove(node, proc, where-1, moveData);
            if(shrink && moveData.canShrink)
                Init();
            
            return true;
        }

    }

    // standard moves
    int dir = startingDir;
    while(true)
    {
        bool reachedBeginning = false;
        while(nextMove.second == moveOptions[static_cast<unsigned>(nextMove.first)].end())
        {
            dir = (nextMove.first+1)%NumDirections;
            if(dir == startingDir)
            {
                reachedBeginning = true;
                break;
            }
            nextMove.first = dir;
            nextMove.second = moveOptions[static_cast<unsigned>(nextMove.first)].begin();
        }
        if(reachedBeginning)
            break;

        std::pair<vertex_idx, unsigned> next = *nextMove.second;
        ++nextMove.second;

        const vertex_idx node = next.first;
        const unsigned proc = next.second;

        if(use_memory_constraint && violatesMemConstraint(node, proc, dir-1))
            continue;

        stepAuxData moveData;
        int costDiff = moveCostChange(node, proc, dir-1, moveData);

        if(!steepestAscent && costDiff<0)
        {
            executeMove(node, proc, dir-1, moveData);
            if(shrink && moveData.canShrink)
                Init();

            return true;
        }
        else if(static_cast<cost_type>(static_cast<int>(cost)+costDiff) < bestCost)
        {
            bestCost = static_cast<cost_type>(static_cast<int>(cost)+costDiff);
            bestMove = next;
            bestMoveData = moveData;
            bestDir = dir-1;
        }


    }

    if (bestCost == cost)
        return false;

    executeMove(bestMove.first, bestMove.second, bestDir, bestMoveData);
    if(shrink && bestMoveData.canShrink)
        Init();

    return true;
};

// Check if move violates mem constraints
template<typename Graph_t>
bool HillClimbingScheduler<Graph_t>::violatesMemConstraint(vertex_idx node, unsigned processor, int where)
{
    if(memory_used[processor][static_cast<unsigned>(static_cast<int>(schedule->assignedSuperstep(node))+where)]
        + schedule->getInstance().getComputationalDag().vertex_mem_weight(node) > schedule->getInstance().memoryBound(processor)) // TODO ANDRAS double check change
        return true;
    
    return false;
}

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::RemoveNeedlessSupSteps() {

    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const unsigned M = schedule->numberOfSupersteps();
    const Graph_t &G = schedule->getInstance().getComputationalDag();
    
    unsigned current_step = 0;

    auto nextBreak = schedule->numberOfSupersteps();
    for (unsigned step = 0; step < M; ++step) {
        if (nextBreak == step) {
            ++current_step;
            nextBreak = M;
        }
        for (unsigned proc = 0; proc < P; ++proc)
            for (const vertex_idx node : supsteplists[step][proc]) {
                schedule->setAssignedSuperstep(node, current_step);
                for (const vertex_idx &succ : G.children(node))
                    if (schedule->assignedProcessor(node) != schedule->assignedProcessor(succ) && schedule->assignedSuperstep(succ) < nextBreak)
                        nextBreak = schedule->assignedSuperstep(succ);
            }
    }

    schedule->updateNumberOfSupersteps();
};

template<typename Graph_t>
void HillClimbingScheduler<Graph_t>::CreateSupstepLists() {
    
    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const Graph_t &G = schedule->getInstance().getComputationalDag();

    schedule->updateNumberOfSupersteps();
    const unsigned M = schedule->numberOfSupersteps();

    supsteplists.clear();
    supsteplists.resize(M, std::vector<std::list<vertex_idx>>(P));

    for (vertex_idx node : top_sort_view(G))
        supsteplists[schedule->assignedSuperstep(node)][schedule->assignedProcessor(node)].push_back(node);

};

} // namespace osp