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
    static_assert(IsComputationalDagV<Graph_t>, "CoptPartialScheduler can only be used with computational DAGs.");

    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;

    unsigned int timeLimitSeconds_ = 600;

  protected:
    unsigned startSuperstep_ = 1, endSuperstep_ = 3;

    std::vector<vertex_idx_t<Graph_t>> nodeGlobalId_;
    std::unordered_map<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>> node_local_ID;

    std::vector<vertex_idx_t<Graph_t>> sourceGlobalId_;
    std::unordered_map<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>> source_local_ID;

    std::vector<std::pair<unsigned, unsigned>> nodeNeededAfterOnProc_, sourceNeededAfterOnProc_;
    std::vector<std::tuple<vertex_idx_t<Graph_t>, unsigned, unsigned, unsigned>> fixedCommSteps_;
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
    virtual RETURN_STATUS ImproveSchedule(BspScheduleCS<GraphT> &schedule);

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
RETURN_STATUS CoptPartialScheduler<GraphT>::ImproveSchedule(BspScheduleCS<GraphT> &schedule) {
    Envr env;
    Model model = env.CreateModel("bsp_schedule_partial");

    SetupVertexMaps(schedule);

    setupVariablesConstraintsObjective(schedule, model);

    setInitialSolution(schedule, model);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
    model.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model.Solve();

    if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
        UpdateSchedule(schedule);
    }

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        return RETURN_STATUS::OSP_SUCCESS;
    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;
    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            return RETURN_STATUS::BEST_FOUND;
        } else {
            return RETURN_STATUS::TIMEOUT;
        }
    }
}

template <typename GraphT>
void CoptPartialScheduler<GraphT>::SetInitialSolution(const BspScheduleCS<GraphT> &schedule, Model &model) {
    const GraphT &dag = schedule.GetInstance().GetComputationalDag();
    const unsigned &numProcessors = schedule.GetInstance().NumberOfProcessors();
    const auto &cs = schedule.getCommunicationSchedule();

    for (const vertex_idx_t<Graph_t> &node : DAG.vertices()) {
        if (node_local_ID.find(node) == node_local_ID.end()) {
            continue;
        }
        for (unsigned proc = 0; proc < num_processors; proc++) {
            for (unsigned step = 0; step < max_number_supersteps; ++step) {
                if (schedule.assignedProcessor(node) == proc && schedule.assignedSuperstep(node) == start_superstep + step) {
                    model.SetMipStart(node_to_processor_superstep_var[node_local_ID[node]][proc][static_cast<int>(step)], 1);
                } else {
                    model.SetMipStart(node_to_processor_superstep_var[node_local_ID[node]][proc][static_cast<int>(step)], 0);
                }
            }
        }
    }

    for (unsigned index = 0; index < fixed_comm_steps.size(); ++index) {
        model.SetMipStart(keep_fixed_comm_step[static_cast<int>(index)], 1);
    }

    for (const auto &node : dag.vertices()) {
        if (node_local_ID.find(node) == node_local_ID.end()) {
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
                            comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node_local_ID[node])], 1);
                    } else {
                        model.SetMipStart(
                            comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node_local_ID[node])], 0);
                    }
                }
            }
        }
    }

    for (const auto &source : dag.vertices()) {
        if (source_local_ID.find(source) == source_local_ID.end()) {
            continue;
        }

        for (unsigned proc = 0; proc < numProcessors; proc++) {
            if (proc == schedule.assignedProcessor(source)) {
                continue;
            }

            for (unsigned step = 0; step < maxNumberSupersteps_ + 1 && step <= endSuperstep_ - startSuperstep_ + 1; step++) {
                const auto &key = std::make_tuple(source, schedule.assignedProcessor(source), proc);
                if (cs.find(key) != cs.end() && cs.at(key) == startSuperstep_ + step - 1) {
                    model.SetMipStart(
                        comm_to_processor_superstep_source_var[proc][step][static_cast<int>(source_local_ID[source])], 1);
                } else if (step > 0) {
                    model.SetMipStart(
                        comm_to_processor_superstep_source_var[proc][step][static_cast<int>(source_local_ID[source])], 0);
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

    while (number_of_supersteps < max_number_supersteps
           && superstep_used_var[static_cast<int>(number_of_supersteps)].Get(COPT_DBLINFO_VALUE) >= .99) {
        numberOfSupersteps++;
    }

    const int offset = static_cast<int>(numberOfSupersteps) - static_cast<int>(endSuperstep_ - startSuperstep_ + 1);

    for (vertex_idx_t<Graph_t> node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        if (schedule.assignedSuperstep(node) > endSuperstep_) {
            schedule.setAssignedSuperstep(node, static_cast<unsigned>(static_cast<int>(schedule.assignedSuperstep(node)) + offset));
        }
    }

    for (vertex_idx_t<Graph_t> node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        if (node_local_ID.find(node) == node_local_ID.end()) {
            continue;
        }

        for (unsigned processor = 0; processor < schedule.GetInstance().NumberOfProcessors(); processor++) {
            for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                if (node_to_processor_superstep_var[node_local_ID.at(node)][processor][static_cast<int>(step)].Get(
                        COPT_DBLINFO_VALUE)
                    >= .99) {
                    schedule.setAssignedSuperstep(node, startSuperstep_ + step);
                    schedule.setAssignedProcessor(node, processor);
                }
            }
        }
    }

    std::map<KeyTriple, unsigned int> &commSchedule = schedule.getCommunicationSchedule();

    std::vector<KeyTriple> toErase;
    for (const auto &[key, val] : schedule.getCommunicationSchedule()) {
        if (val > endSuperstep_) {
            commSchedule[key] = static_cast<unsigned>(static_cast<int>(val) + offset);
        } else if (static_cast<int>(val) >= static_cast<int>(startSuperstep_) - 1) {
            toErase.push_back(key);
        }
    }
    for (const KeyTriple &key : toErase) {
        commSchedule.erase(key);
    }

    for (unsigned index = 0; index < fixed_comm_steps.size(); ++index) {
        const auto &entry = fixed_comm_steps[index];
        if (keep_fixed_comm_step[static_cast<int>(index)].Get(COPT_DBLINFO_VALUE) >= .99
            && std::get<3>(entry) < start_superstep + number_of_supersteps) {
            commSchedule[std::make_tuple(std::get<0>(entry), std::get<1>(entry), std::get<2>(entry))] = std::get<3>(entry);
        } else {
            commSchedule[std::make_tuple(std::get<0>(entry), std::get<1>(entry), std::get<2>(entry))] = startSuperstep_ - 1;
        }
    }

    for (vertex_idx_t<Graph_t> node = 0; node < node_global_ID.size(); node++) {
        for (unsigned int pFrom = 0; pFrom < schedule.GetInstance().NumberOfProcessors(); pFrom++) {
            for (unsigned int pTo = 0; pTo < schedule.GetInstance().NumberOfProcessors(); pTo++) {
                if (pFrom != pTo) {
                    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                        if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][static_cast<int>(node)].Get(
                                COPT_DBLINFO_VALUE)
                            >= .99) {
                            commSchedule[std::make_tuple(node_global_ID[node], p_from, p_to)] = start_superstep + step;
                            break;
                        }
                    }
                }
            }
        }
    }

    for (vertex_idx_t<Graph_t> source = 0; source < source_global_ID.size(); source++) {
        for (unsigned int pTo = 0; pTo < schedule.GetInstance().NumberOfProcessors(); pTo++) {
            if (source_present_before.find(std::make_pair(source, p_to)) == source_present_before.end()) {
                for (unsigned int step = 0; step < maxNumberSupersteps_ + 1; step++) {
                    if (comm_to_processor_superstep_source_var[p_to][step][static_cast<int>(source)].Get(COPT_DBLINFO_VALUE)
                        >= .99) {
                        commSchedule[std::make_tuple(
                            source_global_ID[source], schedule.assignedProcessor(source_global_ID[source]), p_to)]
                            = start_superstep - 1 + step;
                        break;
                    }
                }
            }
        }
    }

    schedule.cleanCommSchedule();
    schedule.shrinkByMergingSupersteps();
};

template <typename GraphT>
void CoptPartialScheduler<GraphT>::SetupVariablesConstraintsObjective(const BspScheduleCS<GraphT> &schedule, Model &model) {
    const vertex_idx_t<Graph_t> numVertices = static_cast<vertex_idx_t<Graph_t>>(node_global_ID.size());
    const vertex_idx_t<Graph_t> numSources = static_cast<vertex_idx_t<Graph_t>>(source_global_ID.size());
    const unsigned numProcessors = schedule.GetInstance().NumberOfProcessors();

    /*
    Variables
    */
    // variables indicating if superstep is used at all
    superstep_used_var = model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "superstep_used");
    VarArray superstepHasComm = model.AddVars(static_cast<int>(max_number_supersteps + 1), COPT_BINARY, "superstep_has_comm");
    VarArray hasCommAtEnd = model.AddVars(1, COPT_BINARY, "has_comm_at_end");

    // variables for assigments of nodes to processor and superstep
    node_to_processor_superstep_var = std::vector<std::vector<VarArray>>(numVertices, std::vector<VarArray>(numProcessors));

    for (unsigned int node = 0; node < numVertices; node++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            node_to_processor_superstep_var[node][processor]
                = model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "node_to_processor_superstep");
        }
    }

    // communicate node from p1 to p2 at superstep

    comm_processor_to_processor_superstep_node_var = std::vector<std::vector<std::vector<VarArray>>>(
        num_processors, std::vector<std::vector<VarArray>>(num_processors, std::vector<VarArray>(max_number_supersteps)));

    for (unsigned int p1 = 0; p1 < numProcessors; p1++) {
        for (unsigned int p2 = 0; p2 < numProcessors; p2++) {
            for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                comm_processor_to_processor_superstep_node_var[p1][p2][step]
                    = model.AddVars(static_cast<int>(numVertices), COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    // communicate nodes in supersteps smaller than start_superstep
    comm_to_processor_superstep_source_var
        = std::vector<std::vector<VarArray>>(num_processors, std::vector<VarArray>(max_number_supersteps + 1));
    std::vector<std::vector<VarArray>> presentOnProcessorSuperstepSourceVar
        = std::vector<std::vector<VarArray>>(num_processors, std::vector<VarArray>(max_number_supersteps));

    for (unsigned int proc = 0; proc < numProcessors; proc++) {
        for (unsigned int step = 0; step < maxNumberSupersteps_ + 1; step++) {
            comm_to_processor_superstep_source_var[proc][step]
                = model.AddVars(static_cast<int>(num_sources), COPT_BINARY, "comm_to_processor_superstep_source");

            if (step < maxNumberSupersteps_) {
                present_on_processor_superstep_source_var[proc][step]
                    = model.AddVars(static_cast<int>(num_sources), COPT_BINARY, "present_on_processor_superstep_source");
            }
        }
    }

    VarArray maxCommSuperstepVar = model.AddVars(static_cast<int>(max_number_supersteps + 1), COPT_INTEGER, "max_comm_superstep");

    VarArray maxWorkSuperstepVar = model.AddVars(static_cast<int>(max_number_supersteps), COPT_INTEGER, "max_work_superstep");

    keep_fixed_comm_step = model.AddVars(static_cast<int>(fixed_comm_steps.size()), COPT_BINARY, "keep_fixed_comm_step");

    /*
    Constraints
      */

    //  use consecutive supersteps starting from 0
    model.AddConstr(superstep_used_var[0] == 1);

    for (unsigned int step = 0; step < maxNumberSupersteps_ - 1; step++) {
        model.AddConstr(superstep_used_var[static_cast<int>(step)] >= superstep_used_var[static_cast<int>(step + 1)]);
    }

    // check whether superstep is used at all (work or comm), and whether superstep has any communication at all
    unsigned largeConstantWork = static_cast<unsigned>(numVertices) * numProcessors;
    unsigned largeConstantComm = static_cast<unsigned>(numVertices + num_sources) * num_processors * num_processors
                                 + static_cast<unsigned>(fixed_comm_steps.size());
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        Expr exprWork, expr_comm;
        for (vertex_idx_t<Graph_t> node = 0; node < numVertices; node++) {
            for (unsigned int processor = 0; processor < numProcessors; processor++) {
                expr_work += node_to_processor_superstep_var[node][processor][static_cast<int>(step)];

                for (unsigned int pOther = 0; pOther < numProcessors; pOther++) {
                    if (processor != pOther) {
                        expr_comm
                            += comm_processor_to_processor_superstep_node_var[processor][p_other][step][static_cast<int>(node)];
                    }
                }
            }
        }
        for (vertex_idx_t<Graph_t> source = 0; source < num_sources; source++) {
            for (unsigned int processor = 0; processor < numProcessors; processor++) {
                if (source_present_before.find(std::make_pair(source, processor)) == source_present_before.end()) {
                    expr_comm += comm_to_processor_superstep_source_var[processor][step + 1][static_cast<int>(source)];
                }
            }
        }

        for (unsigned index = 0; index < fixed_comm_steps.size(); ++index) {
            if (std::get<3>(fixed_comm_steps[index]) == start_superstep + step) {
                expr_comm += keep_fixed_comm_step[static_cast<int>(index)];
            }
        }

        model.AddConstr(expr_comm <= large_constant_comm * superstep_has_comm[static_cast<int>(step + 1)]);
        model.AddConstr(expr_work <= large_constant_work * superstep_used_var[static_cast<int>(step)]);
        model.AddConstr(superstep_has_comm[static_cast<int>(step + 1)] <= superstep_used_var[static_cast<int>(step)]);
    }

    // check communication usage in edge case: comm phase before the segment
    if (hasFixedCommInPrecedingStep_) {
        model.AddConstr(superstep_has_comm[0] == 1);
    } else {
        Expr exprComm0;
        for (vertex_idx_t<Graph_t> source = 0; source < num_sources; source++) {
            for (unsigned int processor = 0; processor < numProcessors; processor++) {
                if (source_present_before.find(std::make_pair(source, processor)) == source_present_before.end()) {
                    expr_comm_0 += comm_to_processor_superstep_source_var[processor][0][static_cast<int>(source)];
                }
            }
        }
        for (unsigned index = 0; index < fixed_comm_steps.size(); ++index) {
            expr_comm_0 += 1 - keep_fixed_comm_step[static_cast<int>(index)];
        }
        model.AddConstr(expr_comm_0
                        <= (static_cast<unsigned>(num_sources) * num_processors + static_cast<unsigned>(fixed_comm_steps.size()))
                               * superstep_has_comm[0]);
    }

    // check if there is any communication at the end of the subschedule
    for (unsigned int step = 0; step < maxNumberSupersteps_ - 1; step++) {
        model.AddConstr(superstep_used_var[static_cast<int>(step)] - superstep_used_var[static_cast<int>(step + 1)]
                            + superstep_has_comm[static_cast<int>(step + 1)] - 1
                        <= has_comm_at_end[0]);
    }
    model.AddConstr(superstep_used_var[static_cast<int>(max_number_supersteps - 1)]
                        + superstep_has_comm[static_cast<int>(max_number_supersteps)] - 1
                    <= has_comm_at_end[0]);

    // nodes are assigend
    for (vertex_idx_t<Graph_t> node = 0; node < numVertices; node++) {
        Expr expr;
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                expr += node_to_processor_superstep_var[node][processor][static_cast<int>(step)];
            }
        }

        model.AddConstr(expr == 1);
    }

    // precedence constraint: if task is computed then all of its predecessors must have been present
    for (vertex_idx_t<Graph_t> node = 0; node < numVertices; node++) {
        for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
            for (unsigned int processor = 0; processor < numProcessors; processor++) {
                Expr expr;
                unsigned numTerms = 0;
                for (const auto &pred : schedule.GetInstance().GetComputationalDag().Parents(node_global_ID[node])) {
                    if (node_local_ID.find(pred) != node_local_ID.end()) {
                        ++num_terms;
                        expr += comm_processor_to_processor_superstep_node_var[processor][processor][step]
                                                                              [static_cast<int>(node_local_ID[pred])];
                    } else if (source_local_ID.find(pred) != source_local_ID.end()
                               && source_present_before.find(std::make_pair(source_local_ID[pred], processor))
                                      == source_present_before.end()) {
                        ++num_terms;
                        expr += present_on_processor_superstep_source_var[processor][step][static_cast<int>(source_local_ID[pred])];
                    }
                }

                if (numTerms > 0) {
                    model.AddConstr(expr >= num_terms * node_to_processor_superstep_var[node][processor][static_cast<int>(step)]);
                }
            }
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            for (vertex_idx_t<Graph_t> node = 0; node < numVertices; node++) {
                Expr expr1, expr2;
                if (step > 0) {
                    for (unsigned int pFrom = 0; pFrom < numProcessors; pFrom++) {
                        expr1
                            += comm_processor_to_processor_superstep_node_var[p_from][processor][step - 1][static_cast<int>(node)];
                    }
                }

                expr1 += node_to_processor_superstep_var[node][processor][static_cast<int>(step)];

                for (unsigned int pTo = 0; pTo < numProcessors; pTo++) {
                    expr2 += comm_processor_to_processor_superstep_node_var[processor][p_to][step][static_cast<int>(node)];
                }

                model.AddConstr(num_processors * (expr1) >= expr2);
            }
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            for (vertex_idx_t<Graph_t> sourceNode = 0; source_node < num_sources; source_node++) {
                if (source_present_before.find(std::make_pair(source_node, processor)) != source_present_before.end()) {
                    continue;
                }

                Expr expr1 = comm_to_processor_superstep_source_var[processor][step][static_cast<int>(source_node)];
                if (step > 0) {
                    expr1 += present_on_processor_superstep_source_var[processor][step - 1][static_cast<int>(source_node)];
                }

                Expr expr2 = present_on_processor_superstep_source_var[processor][step][static_cast<int>(source_node)];

                model.AddConstr(expr1 >= expr2);
            }
        }
    }

    // boundary conditions at the end
    for (const std::pair<vertex_idx_t<Graph_t>, unsigned> node_and_proc : node_needed_after_on_proc) {
        Expr expr;
        for (unsigned int p_from = 0; p_from < num_processors; p_from++) {
            expr += comm_processor_to_processor_superstep_node_var[p_from][node_and_proc.second][max_number_supersteps - 1]
                                                                  [static_cast<int>(node_and_proc.first)];
        }

        model.AddConstr(expr >= 1);
    }

    for (const std::pair<vertex_idx_t<Graph_t>, unsigned> source_and_proc : source_needed_after_on_proc) {
        Expr expr = present_on_processor_superstep_source_var[source_and_proc.second][max_number_supersteps - 1]
                                                             [static_cast<int>(source_and_proc.first)];
        expr += comm_to_processor_superstep_source_var[source_and_proc.second][max_number_supersteps]
                                                      [static_cast<int>(source_and_proc.first)];
        model.AddConstr(expr >= 1);
    }

    // cost calculation - work
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            Expr expr;
            for (unsigned int node = 0; node < numVertices; node++) {
                expr += schedule.GetInstance().GetComputationalDag().VertexWorkWeight(node_global_ID[node])
                        * node_to_processor_superstep_var[node][processor][static_cast<int>(step)];
            }

            model.AddConstr(max_work_superstep_var[static_cast<int>(step)] >= expr);
        }
    }

    // cost calculation - comm
    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            Expr expr1, expr2;
            for (vertex_idx_t<Graph_t> node = 0; node < numVertices; node++) {
                for (unsigned int pOther = 0; pOther < numProcessors; pOther++) {
                    if (processor != pOther) {
                        expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(node_global_ID[node])
                                 * schedule.GetInstance().sendCosts(processor, p_other)
                                 * comm_processor_to_processor_superstep_node_var[processor][p_other][step][static_cast<int>(node)];
                        expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(node_global_ID[node])
                                 * schedule.GetInstance().sendCosts(p_other, processor)
                                 * comm_processor_to_processor_superstep_node_var[p_other][processor][step][static_cast<int>(node)];
                    }
                }
            }

            for (vertex_idx_t<Graph_t> source = 0; source < num_sources; source++) {
                const unsigned originProc = schedule.assignedProcessor(source_global_ID[source]);
                if (originProc == processor) {
                    for (unsigned int pOther = 0; pOther < numProcessors; pOther++) {
                        expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(source_global_ID[source])
                                 * schedule.GetInstance().sendCosts(processor, p_other)
                                 * comm_to_processor_superstep_source_var[p_other][step + 1][static_cast<int>(source)];
                    }
                }
                expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(source_global_ID[source])
                         * schedule.GetInstance().sendCosts(origin_proc, processor)
                         * comm_to_processor_superstep_source_var[processor][step + 1][static_cast<int>(source)];
            }

            for (unsigned index = 0; index < fixed_comm_steps.size(); ++index) {
                const auto &entry = fixed_comm_steps[index];
                if (std::get<3>(entry) != startSuperstep_ + step) {
                    continue;
                }
                if (std::get<1>(entry) == processor) {
                    expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(entry))
                             * schedule.GetInstance().sendCosts(processor, std::get<2>(entry))
                             * keep_fixed_comm_step[static_cast<int>(index)];
                }
                if (std::get<2>(entry) == processor) {
                    expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(entry))
                             * schedule.GetInstance().sendCosts(std::get<1>(entry), processor)
                             * keep_fixed_comm_step[static_cast<int>(index)];
                }
            }

            model.AddConstr(max_comm_superstep_var[static_cast<int>(step + 1)] >= expr1);
            model.AddConstr(max_comm_superstep_var[static_cast<int>(step + 1)] >= expr2);
        }
    }

    // cost calculation - first comm phase handled separately
    for (unsigned int processor = 0; processor < numProcessors; processor++) {
        Expr expr1, expr2;
        for (vertex_idx_t<Graph_t> source = 0; source < num_sources; source++) {
            const unsigned originProc = schedule.assignedProcessor(source_global_ID[source]);
            if (originProc == processor) {
                for (unsigned int pOther = 0; pOther < numProcessors; pOther++) {
                    expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(source_global_ID[source])
                             * schedule.GetInstance().sendCosts(processor, p_other)
                             * comm_to_processor_superstep_source_var[p_other][0][static_cast<int>(source)];
                }
            }
            expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(source_global_ID[source])
                     * schedule.GetInstance().sendCosts(origin_proc, processor)
                     * comm_to_processor_superstep_source_var[processor][0][static_cast<int>(source)];
        }

        for (unsigned index = 0; index < fixed_comm_steps.size(); ++index) {
            const auto &entry = fixed_comm_steps[index];
            if (std::get<1>(entry) == processor) {
                expr1 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(entry))
                         * schedule.GetInstance().sendCosts(processor, std::get<2>(entry))
                         * (1 - keep_fixed_comm_step[static_cast<int>(index)]);
            }
            if (std::get<2>(entry) == processor) {
                expr2 += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(entry))
                         * schedule.GetInstance().sendCosts(std::get<1>(entry), processor)
                         * (1 - keep_fixed_comm_step[static_cast<int>(index)]);
            }
        }

        model.AddConstr(max_comm_superstep_var[0] >= expr1);
        model.AddConstr(max_comm_superstep_var[0] >= expr2);
    }

    /*
    Objective function
    */
    Expr expr;

    for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
        expr += max_work_superstep_var[static_cast<int>(step)]
                + schedule.GetInstance().CommunicationCosts() * max_comm_superstep_var[static_cast<int>(step + 1)]
                + schedule.GetInstance().SynchronisationCosts() * superstep_used_var[static_cast<int>(step)];
    }

    expr += schedule.GetInstance().CommunicationCosts() * max_comm_superstep_var[0];
    expr += schedule.GetInstance().SynchronisationCosts() * superstep_has_comm[0];
    expr += schedule.GetInstance().SynchronisationCosts() * has_comm_at_end[0];

    model.SetObjective(expr - schedule.GetInstance().SynchronisationCosts(), COPT_MINIMIZE);
};

template <typename GraphT>
void CoptPartialScheduler<GraphT>::SetupVertexMaps(const BspScheduleCS<GraphT> &schedule) {
    node_local_ID.clear();
    node_global_ID.clear();
    source_local_ID.clear();
    source_global_ID.clear();

    node_needed_after_on_proc.clear();
    source_needed_after_on_proc.clear();
    fixed_comm_steps.clear();
    source_present_before.clear();

    std::vector<std::vector<unsigned>> firstAt = schedule.getFirstPresence();

    maxNumberSupersteps_ = endSuperstep_ - startSuperstep_ + 3;

    for (unsigned node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        if (schedule.assignedSuperstep(node) >= startSuperstep_ && schedule.assignedSuperstep(node) <= endSuperstep_) {
            node_local_ID[node] = static_cast<vertex_idx_t<Graph_t>>(node_global_ID.size());
            node_global_ID.push_back(node);

            for (const auto &pred : schedule.GetInstance().GetComputationalDag().Parents(node)) {
                if (schedule.assignedSuperstep(pred) < startSuperstep_) {
                    if (source_local_ID.find(pred) == source_local_ID.end()) {
                        source_local_ID[pred] = static_cast<vertex_idx_t<Graph_t>>(source_global_ID.size());
                        source_global_ID.push_back(pred);
                    }

                } else if (schedule.assignedSuperstep(pred) > endSuperstep_) {
                    throw std::invalid_argument("Initial Schedule might be invalid?!");
                }
            }
        }
    }

    // find where the sources are already present before the segment
    for (const auto &source_and_ID : source_local_ID) {
        vertex_idx_t<Graph_t> source = source_and_ID.first;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            if (first_at[source][proc] < start_superstep) {
                source_present_before.emplace(std::make_pair(source_and_ID.second, proc));
            }
        }
    }

    // collect values that are needed by the end of the segment
    for (const auto &source_and_ID : source_local_ID) {
        vertex_idx_t<Graph_t> source = source_and_ID.first;

        std::set<unsigned> procs_needing_this;
        for (const auto &succ : schedule.GetInstance().GetComputationalDag().Children(source)) {
            if (schedule.assignedProcessor(succ) != schedule.assignedProcessor(source)
                && schedule.assignedSuperstep(succ) > end_superstep) {
                procs_needing_this.insert(schedule.assignedProcessor(succ));
            }
        }

        for (unsigned proc1 = 0; proc1 < schedule.GetInstance().NumberOfProcessors(); ++proc1) {
            for (unsigned proc2 = 0; proc2 < schedule.GetInstance().NumberOfProcessors(); ++proc2) {
                if (proc1 == proc2) {
                    continue;
                }
                auto itr = schedule.getCommunicationSchedule().find(std::make_tuple(source, proc1, proc2));
                if (itr != schedule.getCommunicationSchedule().end() && itr->second > end_superstep) {
                    procs_needing_this.insert(schedule.assignedProcessor(proc1));
                }
            }
        }

        for (unsigned proc : procs_needing_this) {
            if (first_at[source][proc] >= start_superstep && first_at[source][proc] <= end_superstep + 1) {
                source_needed_after_on_proc.emplace_back(source_and_ID.second, proc);
            }
        }
    }
    for (const auto &node_and_ID : node_local_ID) {
        vertex_idx_t<Graph_t> node = node_and_ID.first;

        std::set<unsigned> procs_needing_this;
        for (const auto &succ : schedule.GetInstance().GetComputationalDag().Children(node)) {
            if (schedule.assignedSuperstep(succ) > end_superstep) {
                procs_needing_this.insert(schedule.assignedProcessor(succ));
            }
        }

        for (unsigned proc1 = 0; proc1 < schedule.GetInstance().NumberOfProcessors(); ++proc1) {
            for (unsigned proc2 = 0; proc2 < schedule.GetInstance().NumberOfProcessors(); ++proc2) {
                auto itr = schedule.getCommunicationSchedule().find(std::make_tuple(node, proc1, proc2));
                if (itr != schedule.getCommunicationSchedule().end() && proc1 != proc2 && itr->second > end_superstep) {
                    procs_needing_this.insert(schedule.assignedProcessor(proc1));
                }
            }
        }

        for (unsigned proc : procs_needing_this) {
            if (first_at[node][proc] <= end_superstep + 1) {
                node_needed_after_on_proc.emplace_back(node_and_ID.second, proc);
            }
        }
    }

    // comm steps that just happen to be in this interval, but not connected to the nodes within
    hasFixedCommInPrecedingStep_ = false;
    for (const auto &[key, val] : schedule.getCommunicationSchedule()) {
        vertex_idx_t<Graph_t> source = std::get<0>(key);
        if (source_local_ID.find(source) == source_local_ID.end() && schedule.assignedSuperstep(source) < start_superstep
            && val >= start_superstep - 1 && val <= end_superstep) {
            fixed_comm_steps.emplace_back(std::get<0>(key), std::get<1>(key), std::get<2>(key), val);
            if (val == startSuperstep_ - 1) {
                hasFixedCommInPrecedingStep_ = true;
            }
        }
    }
};

}    // namespace osp
