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

#include <callbackbase.h>
#include <coptcpp_pch.h>

#include "osp/bsp/model/BspScheduleCS.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

/**
 * @class CoptPartialScheduler
 * @brief A class that represents a scheduler using the COPT solver for optimizing a specific segment of
 * a BSP schedule, from a starting superstep to and ending superstep.
 */

template <typename GraphT>
class CoptPartialScheduler {
    static_assert(isComputationalDagV<GraphT>, "CoptPartialScheduler can only be used with computational DAGs.");

    using KeyTriple = std::tuple<VertexIdxT<GraphT>, unsigned int, unsigned int>;

    unsigned int timeLimitSeconds_ = 600;

  protected:
    unsigned startSuperstep_ = 1, endSuperstep_ = 3;

    std::vector<VertexIdxT<GraphT>> nodeGlobalId_;
    std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<GraphT>> nodeLocalId_;

    std::vector<VertexIdxT<GraphT>> sourceGlobalId_;
    std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<GraphT>> sourceLocalId_;

    std::vector<std::pair<unsigned, unsigned>> nodeNeededAfterOnProc_, sourceNeededAfterOnProc_;
    std::vector<std::tuple<VertexIdxT<GraphT>, unsigned, unsigned, unsigned>> fixedCommSteps_;
    std::set<std::pair<unsigned, unsigned>> sourcePresentBefore_;

    unsigned maxNumberSupersteps_;

    VarArray superstepUsedVar_;
    VarArray keepFixedCommStep_;

    std::vector<std::vector<VarArray>> nodeToProcessorSuperstepVar_;
    std::vector<std::vector<std::vector<VarArray>>> commProcessorToProcessorSuperstepNodeVar_;
    std::vector<std::vector<VarArray>> commToProcessorSuperstepSourceVar_;

    bool hasFixedCommInPrecedingStep_;

    void SetupVariablesConstraintsObjective(const BspScheduleCS<GraphT> &schedule, Model &model);

    void SetInitialSolution(const BspScheduleCS<GraphT> &schedule, Model &model);

    void UpdateSchedule(BspScheduleCS<GraphT> &schedule) const;

    void SetupVertexMaps(const BspScheduleCS<GraphT> &schedule);

  public:
    virtual ReturnStatus ImproveSchedule(BspScheduleCS<GraphT> &schedule);

    virtual std::string GetScheduleName() const { return "ILPPartial"; }

    virtual void SetTimeLimitSeconds(unsigned int limit) { timeLimitSeconds_ = limit; }

    inline unsigned int GetTimeLimitSeconds() const { return timeLimitSeconds_; }

    virtual void SetStartAndEndSuperstep(unsigned start, unsigned end) {
        startSuperstep_ = start;
        endSuperstep_ = end;
    }

    virtual ~CoptPartialScheduler() = default;
};

template <typename GraphT>
ReturnStatus CoptPartialScheduler<GraphT>::ImproveSchedule(BspScheduleCS<GraphT> &schedule) {
    Envr env;
    Model model = env.CreateModel("bsp_schedule_partial");

    SetupVertexMaps(schedule);

    SetupVariablesConstraintsObjective(schedule, model);

    SetInitialSolution(schedule, model);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds_);
    model.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model.Solve();

    if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
        UpdateSchedule(schedule);
    }

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        return ReturnStatus::OSP_SUCCESS;
    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return ReturnStatus::ERROR;
    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            return ReturnStatus::BEST_FOUND;
        } else {
            return ReturnStatus::TIMEOUT;
        }
    }
}

template <typename GraphT>
void CoptPartialScheduler<GraphT>::SetInitialSolution(const BspScheduleCS<GraphT> &schedule, Model &model) {
    const GraphT &dag = schedule.GetInstance().GetComputationalDag();
    const unsigned &numProcessors = schedule.GetInstance().NumberOfProcessors();
    const auto &cs = schedule.GetCommunicationSchedule();

    for (const VertexIdxT<GraphT> &node : dag.Vertices()) {
        if (nodeLocalId_.find(node) == nodeLocalId_.end()) {
            continue;
        }
        for (unsigned proc = 0; proc < numProcessors; proc++) {
            for (unsigned step = 0; step < maxNumberSupersteps_; ++step) {
                if (schedule.AssignedProcessor(node) == proc && schedule.AssignedSuperstep(node) == startSuperstep_ + step) {
                    model.SetMipStart(nodeToProcessorSuperstepVar_[nodeLocalId_[node]][proc][static_cast<int>(step)], 1);
                } else {
                    model.SetMipStart(nodeToProcessorSuperstepVar_[nodeLocalId_[node]][proc][static_cast<int>(step)], 0);
                }
            }
        }
    }

    for (unsigned index = 0; index < fixedCommSteps_.size(); ++index) {
        model.SetMipStart(keepFixedCommStep_[static_cast<int>(index)], 1);
    }

    for (const auto &node : dag.Vertices()) {
        if (nodeLocalId_.find(node) == nodeLocalId_.end()) {
            continue;
        }

        for (unsigned p1 = 0; p1 < numProcessors; p1++) {
            for (unsigned p2 = 0; p2 < numProcessors; p2++) {
                if (p1 == p2) {
                    continue;
                }

                for (unsigned step = 0; step < maxNumberSupersteps_ && step <= endSuperstep_ - startSuperstep_; step++) {
                    const auto &key = std::make_tuple(node, p1, p2);
                    if (cs.find(key) != cs.end() && cs.at(key) == startSuperstep_ + step) {
                        model.SetMipStart(
                            commProcessorToProcessorSuperstepNodeVar_[p1][p2][step][static_cast<int>(nodeLocalId_[node])], 1);
                    } else {
                        model.SetMipStart(
                            commProcessorToProcessorSuperstepNodeVar_[p1][p2][step][static_cast<int>(nodeLocalId_[node])], 0);
                    }
                }
            }
        }
    }

    for (const auto &source : dag.Vertices()) {
        if (sourceLocalId_.find(source) == sourceLocalId_.end()) {
            continue;
        }

        for (unsigned proc = 0; proc < numProcessors; proc++) {
            if (proc == schedule.AssignedProcessor(source)) {
                continue;
            }

            for (unsigned step = 0; step < maxNumberSupersteps_ + 1 && step <= endSuperstep_ - startSuperstep_ + 1; step++) {
                const auto &key = std::make_tuple(source, schedule.AssignedProcessor(source), proc);
                if (cs.find(key) != cs.end() && cs.at(key) == startSuperstep_ + step - 1) {
                    model.SetMipStart(commToProcessorSuperstepSourceVar_[proc][step][static_cast<int>(sourceLocalId_[source])], 1);
                } else if (step > 0) {
                    model.SetMipStart(commToProcessorSuperstepSourceVar_[proc][step][static_cast<int>(sourceLocalId_[source])], 0);
                }
            }
        }
    }

    model.LoadMipStart();
    model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
}

template <typename GraphT>
void CoptPartialScheduler<GraphT>::UpdateSchedule(BspScheduleCS<GraphT> &schedule) const {
    unsigned numberOfSupersteps = 0;

    while (numberOfSupersteps < maxNumberSupersteps_
           && superstepUsedVar_[static_cast<int>(numberOfSupersteps)].Get(COPT_DBLINFO_VALUE) >= .99) {
        numberOfSupersteps++;
    }

    const int offset = static_cast<int>(numberOfSupersteps) - static_cast<int>(endSuperstep_ - startSuperstep_ + 1);

    for (VertexIdxT<GraphT> node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        if (schedule.AssignedSuperstep(node) > endSuperstep_) {
            schedule.SetAssignedSuperstep(node, static_cast<unsigned>(static_cast<int>(schedule.AssignedSuperstep(node)) + offset));
        }
    }

    for (VertexIdxT<GraphT> node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        if (nodeLocalId_.find(node) == nodeLocalId_.end()) {
            continue;
        }

        for (unsigned processor = 0; processor < schedule.GetInstance().NumberOfProcessors(); processor++) {
            for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                if (nodeToProcessorSuperstepVar_[nodeLocalId_.at(node)][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE)
                    >= .99) {
                    schedule.SetAssignedSuperstep(node, startSuperstep_ + step);
                    schedule.SetAssignedProcessor(node, processor);
                }
            }
        }
    }

    std::map<KeyTriple, unsigned int> &commSchedule = schedule.GetCommunicationSchedule();

    std::vector<KeyTriple> toErase;
    for (const auto &[key, val] : schedule.GetCommunicationSchedule()) {
        if (val > endSuperstep_) {
            commSchedule[key] = static_cast<unsigned>(static_cast<int>(val) + offset);
        } else if (static_cast<int>(val) >= static_cast<int>(startSuperstep_) - 1) {
            toErase.push_back(key);
        }
    }
    for (const KeyTriple &key : toErase) {
        commSchedule.erase(key);
    }

    for (unsigned index = 0; index < fixedCommSteps_.size(); ++index) {
        const auto &entry = fixedCommSteps_[index];
        if (keepFixedCommStep_[static_cast<int>(index)].Get(COPT_DBLINFO_VALUE) >= .99
            && std::get<3>(entry) < startSuperstep_ + numberOfSupersteps) {
            commSchedule[std::make_tuple(std::get<0>(entry), std::get<1>(entry), std::get<2>(entry))] = std::get<3>(entry);
        } else {
            commSchedule[std::make_tuple(std::get<0>(entry), std::get<1>(entry), std::get<2>(entry))] = startSuperstep_ - 1;
        }
    }

    for (VertexIdxT<GraphT> node = 0; node < nodeGlobalId_.size(); node++) {
        for (unsigned int pFrom = 0; pFrom < schedule.GetInstance().NumberOfProcessors(); pFrom++) {
            for (unsigned int pTo = 0; pTo < schedule.GetInstance().NumberOfProcessors(); pTo++) {
                if (pFrom != pTo) {
                    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                        if (commProcessorToProcessorSuperstepNodeVar_[pFrom][pTo][step][static_cast<int>(node)].Get(
                                COPT_DBLINFO_VALUE)
                            >= .99) {
                            commSchedule[std::make_tuple(nodeGlobalId_[node], pFrom, pTo)] = startSuperstep_ + step;
                            break;
                        }
                    }
                }
            }
        }
    }

    for (VertexIdxT<GraphT> source = 0; source < sourceGlobalId_.size(); source++) {
        for (unsigned int pTo = 0; pTo < schedule.GetInstance().NumberOfProcessors(); pTo++) {
            if (sourcePresentBefore_.find(std::make_pair(source, pTo)) == sourcePresentBefore_.end()) {
                for (unsigned int step = 0; step < maxNumberSupersteps_ + 1; step++) {
                    if (commToProcessorSuperstepSourceVar_[pTo][step][static_cast<int>(source)].Get(COPT_DBLINFO_VALUE) >= .99) {
                        commSchedule[std::make_tuple(sourceGlobalId_[source], schedule.AssignedProcessor(sourceGlobalId_[source]), pTo)]
                            = startSuperstep_ - 1 + step;
                        break;
                    }
                }
            }
        }
    }

    schedule.CleanCommSchedule();
    schedule.ShrinkByMergingSupersteps();
}

template <typename GraphT>
void CoptPartialScheduler<GraphT>::SetupVariablesConstraintsObjective(const BspScheduleCS<GraphT> &schedule, Model &model) {
    const VertexIdxT<GraphT> numVertices = static_cast<VertexIdxT<GraphT>>(nodeGlobalId_.size());
    const VertexIdxT<GraphT> numSources = static_cast<VertexIdxT<GraphT>>(sourceGlobalId_.size());
    const unsigned numProcessors = schedule.GetInstance().NumberOfProcessors();

    /*
    Variables
    */
    // variables indicating if superstep is used at all
    superstepUsedVar_ = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_BINARY, "superstep_used");
    VarArray superstepHasComm = model.AddVars(static_cast<int>(maxNumberSupersteps_ + 1), COPT_BINARY, "superstepHasComm");
    VarArray hasCommAtEnd = model.AddVars(1, COPT_BINARY, "hasCommAtEnd");

    // variables for assigments of nodes to processor and superstep
    nodeToProcessorSuperstepVar_ = std::vector<std::vector<VarArray>>(numVertices, std::vector<VarArray>(numProcessors));

    for (unsigned int node = 0; node < numVertices; node++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            nodeToProcessorSuperstepVar_[node][processor]
                = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_BINARY, "node_to_processor_superstep");
        }
    }

    // communicate node from p1 to p2 at superstep

    commProcessorToProcessorSuperstepNodeVar_ = std::vector<std::vector<std::vector<VarArray>>>(
        numProcessors, std::vector<std::vector<VarArray>>(numProcessors, std::vector<VarArray>(maxNumberSupersteps_)));

    for (unsigned int p1 = 0; p1 < numProcessors; p1++) {
        for (unsigned int p2 = 0; p2 < numProcessors; p2++) {
            for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                commProcessorToProcessorSuperstepNodeVar_[p1][p2][step]
                    = model.AddVars(static_cast<int>(numVertices), COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    // communicate nodes in supersteps smaller than startSuperstep_
    commToProcessorSuperstepSourceVar_
        = std::vector<std::vector<VarArray>>(numProcessors, std::vector<VarArray>(maxNumberSupersteps_ + 1));
    std::vector<std::vector<VarArray>> presentOnProcessorSuperstepSourceVar
        = std::vector<std::vector<VarArray>>(numProcessors, std::vector<VarArray>(maxNumberSupersteps_));

    for (unsigned int proc = 0; proc < numProcessors; proc++) {
        for (unsigned int step = 0; step < maxNumberSupersteps_ + 1; step++) {
            commToProcessorSuperstepSourceVar_[proc][step]
                = model.AddVars(static_cast<int>(numSources), COPT_BINARY, "comm_to_processor_superstep_source");

            if (step < maxNumberSupersteps_) {
                presentOnProcessorSuperstepSourceVar[proc][step]
                    = model.AddVars(static_cast<int>(numSources), COPT_BINARY, "present_on_processor_superstep_source");
            }
        }
    }

    VarArray maxCommSuperstepVar = model.AddVars(static_cast<int>(maxNumberSupersteps_ + 1), COPT_INTEGER, "max_comm_superstep");

    VarArray maxWorkSuperstepVar = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_INTEGER, "max_work_superstep");

    keepFixedCommStep_ = model.AddVars(static_cast<int>(fixedCommSteps_.size()), COPT_BINARY, "keepFixedCommStep_");

    /*
    Constraints
      */

    //  use consecutive supersteps starting from 0
    model.AddConstr(superstepUsedVar_[0] == 1);

    for (unsigned int step = 0; step < maxNumberSupersteps_ - 1; step++) {
        model.AddConstr(superstepUsedVar_[static_cast<int>(step)] >= superstepUsedVar_[static_cast<int>(step + 1)]);
    }

    // check whether superstep is used at all (work or comm), and whether superstep has any communication at all
    unsigned largeConstantWork = static_cast<unsigned>(numVertices) * numProcessors;
    unsigned largeConstantComm = static_cast<unsigned>(numVertices + numSources) * numProcessors * numProcessors
                                 + static_cast<unsigned>(fixedCommSteps_.size());
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        Expr exprWork, exprComm;
        for (VertexIdxT<GraphT> node = 0; node < numVertices; node++) {
            for (unsigned int processor = 0; processor < numProcessors; processor++) {
                exprWork += nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];

                for (unsigned int pOther = 0; pOther < numProcessors; pOther++) {
                    if (processor != pOther) {
                        exprComm += commProcessorToProcessorSuperstepNodeVar_[processor][pOther][step][static_cast<int>(node)];
                    }
                }
            }
        }
        for (VertexIdxT<GraphT> source = 0; source < numSources; source++) {
            for (unsigned int processor = 0; processor < numProcessors; processor++) {
                if (sourcePresentBefore_.find(std::make_pair(source, processor)) == sourcePresentBefore_.end()) {
                    exprComm += commToProcessorSuperstepSourceVar_[processor][step + 1][static_cast<int>(source)];
                }
            }
        }

        for (unsigned index = 0; index < fixedCommSteps_.size(); ++index) {
            if (std::get<3>(fixedCommSteps_[index]) == startSuperstep_ + step) {
                exprComm += keepFixedCommStep_[static_cast<int>(index)];
            }
        }

        model.AddConstr(exprComm <= largeConstantComm * superstepHasComm[static_cast<int>(step + 1)]);
        model.AddConstr(exprWork <= largeConstantWork * superstepUsedVar_[static_cast<int>(step)]);
        model.AddConstr(superstepHasComm[static_cast<int>(step + 1)] <= superstepUsedVar_[static_cast<int>(step)]);
    }

    // check communication usage in edge case: comm phase before the segment
    if (hasFixedCommInPrecedingStep_) {
        model.AddConstr(superstepHasComm[0] == 1);
    } else {
        Expr exprComm0;
        for (VertexIdxT<GraphT> source = 0; source < numSources; source++) {
            for (unsigned int processor = 0; processor < numProcessors; processor++) {
                if (sourcePresentBefore_.find(std::make_pair(source, processor)) == sourcePresentBefore_.end()) {
                    exprComm0 += commToProcessorSuperstepSourceVar_[processor][0][static_cast<int>(source)];
                }
            }
        }
        for (unsigned index = 0; index < fixedCommSteps_.size(); ++index) {
            exprComm0 += 1 - keepFixedCommStep_[static_cast<int>(index)];
        }
        model.AddConstr(exprComm0
                        <= (static_cast<unsigned>(numSources) * numProcessors + static_cast<unsigned>(fixedCommSteps_.size()))
                               * superstepHasComm[0]);
    }

    // check if there is any communication at the end of the subschedule
    for (unsigned int step = 0; step < maxNumberSupersteps_ - 1; step++) {
        model.AddConstr(superstepUsedVar_[static_cast<int>(step)] - superstepUsedVar_[static_cast<int>(step + 1)]
                            + superstepHasComm[static_cast<int>(step + 1)] - 1
                        <= hasCommAtEnd[0]);
    }
    model.AddConstr(superstepUsedVar_[static_cast<int>(maxNumberSupersteps_ - 1)]
                        + superstepHasComm[static_cast<int>(maxNumberSupersteps_)] - 1
                    <= hasCommAtEnd[0]);

    // nodes are assigend
    for (VertexIdxT<GraphT> node = 0; node < numVertices; node++) {
        Expr expr;
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                expr += nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];
            }
        }

        model.AddConstr(expr == 1);
    }

    // precedence constraint: if task is computed then all of its predecessors must have been present
    for (VertexIdxT<GraphT> node = 0; node < numVertices; node++) {
        for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
            for (unsigned int processor = 0; processor < numProcessors; processor++) {
                Expr expr;
                unsigned numTerms = 0;
                for (const auto &pred : schedule.GetInstance().GetComputationalDag().Parents(nodeGlobalId_[node])) {
                    if (nodeLocalId_.find(pred) != nodeLocalId_.end()) {
                        ++numTerms;
                        expr += commProcessorToProcessorSuperstepNodeVar_[processor][processor][step]
                                                                         [static_cast<int>(nodeLocalId_[pred])];
                    } else if (sourceLocalId_.find(pred) != sourceLocalId_.end()
                               && sourcePresentBefore_.find(std::make_pair(sourceLocalId_[pred], processor))
                                      == sourcePresentBefore_.end()) {
                        ++numTerms;
                        expr += presentOnProcessorSuperstepSourceVar[processor][step][static_cast<int>(sourceLocalId_[pred])];
                    }
                }

                if (numTerms > 0) {
                    model.AddConstr(expr >= numTerms * nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)]);
                }
            }
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            for (VertexIdxT<GraphT> node = 0; node < numVertices; node++) {
                Expr expr1, expr2;
                if (step > 0) {
                    for (unsigned int pFrom = 0; pFrom < numProcessors; pFrom++) {
                        expr1 += commProcessorToProcessorSuperstepNodeVar_[pFrom][processor][step - 1][static_cast<int>(node)];
                    }
                }

                expr1 += nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];

                for (unsigned int pTo = 0; pTo < numProcessors; pTo++) {
                    expr2 += commProcessorToProcessorSuperstepNodeVar_[processor][pTo][step][static_cast<int>(node)];
                }

                model.AddConstr(numProcessors * (expr1) >= expr2);
            }
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            for (VertexIdxT<GraphT> sourceNode = 0; sourceNode < numSources; sourceNode++) {
                if (sourcePresentBefore_.find(std::make_pair(sourceNode, processor)) != sourcePresentBefore_.end()) {
                    continue;
                }

                Expr expr1 = commToProcessorSuperstepSourceVar_[processor][step][static_cast<int>(sourceNode)];
                if (step > 0) {
                    expr1 += presentOnProcessorSuperstepSourceVar[processor][step - 1][static_cast<int>(sourceNode)];
                }

                Expr expr2 = presentOnProcessorSuperstepSourceVar[processor][step][static_cast<int>(sourceNode)];

                model.AddConstr(expr1 >= expr2);
            }
        }
    }

    // boundary conditions at the end
    for (const std::pair<VertexIdxT<GraphT>, unsigned> nodeAndProc : nodeNeededAfterOnProc_) {
        Expr expr;
        for (unsigned int pFrom = 0; pFrom < numProcessors; pFrom++) {
            expr += commProcessorToProcessorSuperstepNodeVar_[pFrom][nodeAndProc.second][maxNumberSupersteps_ - 1]
                                                             [static_cast<int>(nodeAndProc.first)];
        }

        model.AddConstr(expr >= 1);
    }

    for (const std::pair<VertexIdxT<GraphT>, unsigned> sourceAndProc : sourceNeededAfterOnProc_) {
        Expr expr = presentOnProcessorSuperstepSourceVar[sourceAndProc.second][maxNumberSupersteps_ - 1]
                                                        [static_cast<int>(sourceAndProc.first)];
        expr
            += commToProcessorSuperstepSourceVar_[sourceAndProc.second][maxNumberSupersteps_][static_cast<int>(sourceAndProc.first)];
        model.AddConstr(expr >= 1);
    }

    // cost calculation - work
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            Expr expr;
            for (unsigned int node = 0; node < numVertices; node++) {
                expr += schedule.GetInstance().GetComputationalDag().VertexWorkWeight(nodeGlobalId_[node])
                        * nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];
            }

            model.AddConstr(maxWorkSuperstepVar[static_cast<int>(step)] >= expr);
        }
    }

    // cost calculation - comm
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            Expr expr1, expr2;
            for (VertexIdxT<GraphT> node = 0; node < numVertices; node++) {
                for (unsigned int pOther = 0; pOther < numProcessors; pOther++) {
                    if (processor != pOther) {
                        expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(nodeGlobalId_[node])
                                 * schedule.GetInstance().SendCosts(processor, pOther)
                                 * commProcessorToProcessorSuperstepNodeVar_[processor][pOther][step][static_cast<int>(node)];
                        expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(nodeGlobalId_[node])
                                 * schedule.GetInstance().SendCosts(pOther, processor)
                                 * commProcessorToProcessorSuperstepNodeVar_[pOther][processor][step][static_cast<int>(node)];
                    }
                }
            }

            for (VertexIdxT<GraphT> source = 0; source < numSources; source++) {
                const unsigned originProc = schedule.AssignedProcessor(sourceGlobalId_[source]);
                if (originProc == processor) {
                    for (unsigned int pOther = 0; pOther < numProcessors; pOther++) {
                        expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(sourceGlobalId_[source])
                                 * schedule.GetInstance().SendCosts(processor, pOther)
                                 * commToProcessorSuperstepSourceVar_[pOther][step + 1][static_cast<int>(source)];
                    }
                }
                expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(sourceGlobalId_[source])
                         * schedule.GetInstance().SendCosts(originProc, processor)
                         * commToProcessorSuperstepSourceVar_[processor][step + 1][static_cast<int>(source)];
            }

            for (unsigned index = 0; index < fixedCommSteps_.size(); ++index) {
                const auto &entry = fixedCommSteps_[index];
                if (std::get<3>(entry) != startSuperstep_ + step) {
                    continue;
                }
                if (std::get<1>(entry) == processor) {
                    expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(entry))
                             * schedule.GetInstance().SendCosts(processor, std::get<2>(entry))
                             * keepFixedCommStep_[static_cast<int>(index)];
                }
                if (std::get<2>(entry) == processor) {
                    expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(entry))
                             * schedule.GetInstance().SendCosts(std::get<1>(entry), processor)
                             * keepFixedCommStep_[static_cast<int>(index)];
                }
            }

            model.AddConstr(maxCommSuperstepVar[static_cast<int>(step + 1)] >= expr1);
            model.AddConstr(maxCommSuperstepVar[static_cast<int>(step + 1)] >= expr2);
        }
    }

    // cost calculation - first comm phase handled separately
    for (unsigned int processor = 0; processor < numProcessors; processor++) {
        Expr expr1, expr2;
        for (VertexIdxT<GraphT> source = 0; source < numSources; source++) {
            const unsigned originProc = schedule.AssignedProcessor(sourceGlobalId_[source]);
            if (originProc == processor) {
                for (unsigned int pOther = 0; pOther < numProcessors; pOther++) {
                    expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(sourceGlobalId_[source])
                             * schedule.GetInstance().SendCosts(processor, pOther)
                             * commToProcessorSuperstepSourceVar_[pOther][0][static_cast<int>(source)];
                }
            }
            expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(sourceGlobalId_[source])
                     * schedule.GetInstance().SendCosts(originProc, processor)
                     * commToProcessorSuperstepSourceVar_[processor][0][static_cast<int>(source)];
        }

        for (unsigned index = 0; index < fixedCommSteps_.size(); ++index) {
            const auto &entry = fixedCommSteps_[index];
            if (std::get<1>(entry) == processor) {
                expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(entry))
                         * schedule.GetInstance().SendCosts(processor, std::get<2>(entry))
                         * (1 - keepFixedCommStep_[static_cast<int>(index)]);
            }
            if (std::get<2>(entry) == processor) {
                expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(entry))
                         * schedule.GetInstance().SendCosts(std::get<1>(entry), processor)
                         * (1 - keepFixedCommStep_[static_cast<int>(index)]);
            }
        }

        model.AddConstr(maxCommSuperstepVar[0] >= expr1);
        model.AddConstr(maxCommSuperstepVar[0] >= expr2);
    }

    /*
    Objective function
    */
    Expr expr;

    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        expr += maxWorkSuperstepVar[static_cast<int>(step)]
                + schedule.GetInstance().CommunicationCosts() * maxCommSuperstepVar[static_cast<int>(step + 1)]
                + schedule.GetInstance().SynchronisationCosts() * superstepUsedVar_[static_cast<int>(step)];
    }

    expr += schedule.GetInstance().CommunicationCosts() * maxCommSuperstepVar[0];
    expr += schedule.GetInstance().SynchronisationCosts() * superstepHasComm[0];
    expr += schedule.GetInstance().SynchronisationCosts() * hasCommAtEnd[0];

    model.SetObjective(expr - schedule.GetInstance().SynchronisationCosts(), COPT_MINIMIZE);
}

template <typename GraphT>
void CoptPartialScheduler<GraphT>::SetupVertexMaps(const BspScheduleCS<GraphT> &schedule) {
    nodeLocalId_.clear();
    nodeGlobalId_.clear();
    sourceLocalId_.clear();
    sourceGlobalId_.clear();

    nodeNeededAfterOnProc_.clear();
    sourceNeededAfterOnProc_.clear();
    fixedCommSteps_.clear();
    sourcePresentBefore_.clear();

    std::vector<std::vector<unsigned>> firstAt = schedule.GetFirstPresence();

    maxNumberSupersteps_ = endSuperstep_ - startSuperstep_ + 3;

    for (unsigned node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        if (schedule.AssignedSuperstep(node) >= startSuperstep_ && schedule.AssignedSuperstep(node) <= endSuperstep_) {
            nodeLocalId_[node] = static_cast<VertexIdxT<GraphT>>(nodeGlobalId_.size());
            nodeGlobalId_.push_back(node);

            for (const auto &pred : schedule.GetInstance().GetComputationalDag().Parents(node)) {
                if (schedule.AssignedSuperstep(pred) < startSuperstep_) {
                    if (sourceLocalId_.find(pred) == sourceLocalId_.end()) {
                        sourceLocalId_[pred] = static_cast<VertexIdxT<GraphT>>(sourceGlobalId_.size());
                        sourceGlobalId_.push_back(pred);
                    }

                } else if (schedule.AssignedSuperstep(pred) > endSuperstep_) {
                    throw std::invalid_argument("Initial Schedule might be invalid?!");
                }
            }
        }
    }

    // find where the sources are already present before the segment
    for (const auto &sourceAndId : sourceLocalId_) {
        VertexIdxT<GraphT> source = sourceAndId.first;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            if (firstAt[source][proc] < startSuperstep_) {
                sourcePresentBefore_.emplace(std::make_pair(sourceAndId.second, proc));
            }
        }
    }

    // collect values that are needed by the end of the segment
    for (const auto &sourceAndId : sourceLocalId_) {
        VertexIdxT<GraphT> source = sourceAndId.first;

        std::set<unsigned> procsNeedingThis;
        for (const auto &succ : schedule.GetInstance().GetComputationalDag().Children(source)) {
            if (schedule.AssignedProcessor(succ) != schedule.AssignedProcessor(source)
                && schedule.AssignedSuperstep(succ) > endSuperstep_) {
                procsNeedingThis.insert(schedule.AssignedProcessor(succ));
            }
        }

        for (unsigned proc1 = 0; proc1 < schedule.GetInstance().NumberOfProcessors(); ++proc1) {
            for (unsigned proc2 = 0; proc2 < schedule.GetInstance().NumberOfProcessors(); ++proc2) {
                if (proc1 == proc2) {
                    continue;
                }
                auto itr = schedule.GetCommunicationSchedule().find(std::make_tuple(source, proc1, proc2));
                if (itr != schedule.GetCommunicationSchedule().end() && itr->second > endSuperstep_) {
                    procsNeedingThis.insert(schedule.AssignedProcessor(proc1));
                }
            }
        }

        for (unsigned proc : procsNeedingThis) {
            if (firstAt[source][proc] >= startSuperstep_ && firstAt[source][proc] <= endSuperstep_ + 1) {
                sourceNeededAfterOnProc_.emplace_back(sourceAndId.second, proc);
            }
        }
    }
    for (const auto &nodeAndId : nodeLocalId_) {
        VertexIdxT<GraphT> node = nodeAndId.first;

        std::set<unsigned> procsNeedingThis;
        for (const auto &succ : schedule.GetInstance().GetComputationalDag().Children(node)) {
            if (schedule.AssignedSuperstep(succ) > endSuperstep_) {
                procsNeedingThis.insert(schedule.AssignedProcessor(succ));
            }
        }

        for (unsigned proc1 = 0; proc1 < schedule.GetInstance().NumberOfProcessors(); ++proc1) {
            for (unsigned proc2 = 0; proc2 < schedule.GetInstance().NumberOfProcessors(); ++proc2) {
                auto itr = schedule.GetCommunicationSchedule().find(std::make_tuple(node, proc1, proc2));
                if (itr != schedule.GetCommunicationSchedule().end() && proc1 != proc2 && itr->second > endSuperstep_) {
                    procsNeedingThis.insert(schedule.AssignedProcessor(proc1));
                }
            }
        }

        for (unsigned proc : procsNeedingThis) {
            if (firstAt[node][proc] <= endSuperstep_ + 1) {
                nodeNeededAfterOnProc_.emplace_back(nodeAndId.second, proc);
            }
        }
    }

    // comm steps that just happen to be in this interval, but not connected to the nodes within
    hasFixedCommInPrecedingStep_ = false;
    for (const auto &[key, val] : schedule.GetCommunicationSchedule()) {
        VertexIdxT<GraphT> source = std::get<0>(key);
        if (sourceLocalId_.find(source) == sourceLocalId_.end() && schedule.AssignedSuperstep(source) < startSuperstep_
            && val >= startSuperstep_ - 1 && val <= endSuperstep_) {
            fixedCommSteps_.emplace_back(std::get<0>(key), std::get<1>(key), std::get<2>(key), val);
            if (val == startSuperstep_ - 1) {
                hasFixedCommInPrecedingStep_ = true;
            }
        }
    }
}

}    // namespace osp
