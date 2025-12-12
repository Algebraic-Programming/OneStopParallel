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
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

/**
 * @class CoptCommScheduleOptimizer
 * @brief A class that represents a scheduler using the COPT solver for optimizing the communication schedule part of
 * a BSP schedule, with the assignment of vertices to processors and supersteps fixed.
 */

template <typename GraphT>
class CoptCommScheduleOptimizer {
    static_assert(IsComputationalDagV<Graph_t>, "CoptFullScheduler can only be used with computational DAGs.");

    bool ignoreLatency_ = false;

    unsigned int timeLimitSeconds_ = 600;

  protected:
    VarArray superstepHasComm_;
    VarArray maxCommSuperstepVar_;
    std::vector<std::vector<std::vector<VarArray>>> commProcessorToProcessorSuperstepNodeVar_;

    void SetupVariablesConstraintsObjective(const BspScheduleCS<GraphT> &schedule, Model &model);

    void SetInitialSolution(BspScheduleCS<GraphT> &schedule, Model &model);

    bool CanShrinkResultingSchedule(unsigned numberOfSupersteps) const;

    void UpdateCommSchedule(BspScheduleCS<GraphT> &schedule) const;

  public:
    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;
    virtual ~CoptCommScheduleOptimizer() = default;

    virtual RETURN_STATUS ImproveSchedule(BspScheduleCS<GraphT> &schedule);

    virtual std::string GetScheduleName() const { return "ILPCommunication"; }

    virtual void SetTimeLimitSeconds(unsigned int limit) { timeLimitSeconds_ = limit; }

    inline unsigned int GetTimeLimitSeconds() const { return timeLimitSeconds_; }

    virtual void SetIgnoreLatency(bool ignoreLatency) { ignoreLatency_ = ignoreLatency; }
};

template <typename GraphT>
RETURN_STATUS CoptCommScheduleOptimizer<GraphT>::ImproveSchedule(BspScheduleCS<GraphT> &schedule) {
    Envr env;
    Model model = env.CreateModel("bsp_schedule_cs");

    setupVariablesConstraintsObjective(schedule, model);

    setInitialSolution(schedule, model);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
    model.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model.Solve();

    if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
        UpdateCommSchedule(schedule);
        if (CanShrinkResultingSchedule(schedule.NumberOfSupersteps())) {
            schedule.shrinkByMergingSupersteps();
        }
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
bool CoptCommScheduleOptimizer<GraphT>::CanShrinkResultingSchedule(unsigned numberOfSupersteps) const {
    for (unsigned step = 0; step < numberOfSupersteps - 1; step++) {
        if (superstep_has_comm[static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) <= 0.01) {
            return true;
        }
    }
    return false;
}

template <typename GraphT>
void CoptCommScheduleOptimizer<GraphT>::UpdateCommSchedule(BspScheduleCS<GraphT> &schedule) const {
    std::map<KeyTriple, unsigned int> &cs = schedule.getCommunicationSchedule();
    cs.clear();

    for (const auto &node : schedule.GetInstance().vertices()) {
        for (unsigned int pFrom = 0; pFrom < schedule.GetInstance().NumberOfProcessors(); pFrom++) {
            for (unsigned int pTo = 0; pTo < schedule.GetInstance().NumberOfProcessors(); pTo++) {
                if (pFrom != pTo) {
                    for (unsigned int step = 0; step < schedule.NumberOfSupersteps(); step++) {
                        if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][static_cast<int>(node)].Get(
                                COPT_DBLINFO_VALUE)
                            >= .99) {
                            cs[std::make_tuple(node, pFrom, pTo)] = step;
                        }
                    }
                }
            }
        }
    }
}

template <typename GraphT>
void CoptCommScheduleOptimizer<GraphT>::SetInitialSolution(BspScheduleCS<GraphT> &schedule, Model &model) {
    const GraphT &dag = schedule.GetInstance().getComputationalDag();
    const BspArchitecture<GraphT> &arch = schedule.GetInstance().GetArchitecture();
    const unsigned &numProcessors = schedule.GetInstance().NumberOfProcessors();
    const unsigned &numSupersteps = schedule.NumberOfSupersteps();
    const auto &cs = schedule.getCommunicationSchedule();

    std::vector<std::vector<unsigned>> firstAt(DAG.NumVertices(),
                                               std::vector<unsigned>(num_processors, std::numeric_limits<unsigned>::max()));
    for (const auto &node : dag.vertices()) {
        firstAt[node][schedule.assignedProcessor(node)] = schedule.assignedSuperstep(node);
    }

    for (const auto &node : dag.vertices()) {
        for (unsigned p1 = 0; p1 < numProcessors; p1++) {
            for (unsigned p2 = 0; p2 < numProcessors; p2++) {
                if (p1 == p2) {
                    continue;
                }

                for (unsigned step = 0; step < numSupersteps; step++) {
                    const auto &key = std::make_tuple(node, p1, p2);
                    if (cs.find(key) != cs.end() && cs.at(key) == step) {
                        model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node)], 1);
                        firstAt[node][p2] = std::min(first_at[node][p2], step + 1);
                    } else {
                        model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node)], 0);
                    }
                }
            }
        }
    }

    for (const auto &node : dag.vertices()) {
        for (unsigned proc = 0; proc < numProcessors; proc++) {
            for (unsigned step = 0; step < numSupersteps; step++) {
                if (step >= first_at[node][proc]) {
                    model.SetMipStart(comm_processor_to_processor_superstep_node_var[proc][proc][step][static_cast<int>(node)], 1);
                } else {
                    model.SetMipStart(comm_processor_to_processor_superstep_node_var[proc][proc][step][static_cast<int>(node)], 0);
                }
            }
        }
    }

    if (!ignoreLatency_) {
        std::vector<unsigned> commPhaseUsed(numSupersteps, 0);
        for (auto const &[key, val] : cs) {
            commPhaseUsed[val] = 1;
        }
        for (unsigned step = 0; step < numSupersteps; step++) {
            model.SetMipStart(superstep_has_comm[static_cast<int>(step)], comm_phase_used[step]);
        }
    }

    std::vector<std::vector<v_commw_t<Graph_t>>> send(num_supersteps, std::vector<v_commw_t<Graph_t>>(num_processors, 0));
    std::vector<std::vector<v_commw_t<Graph_t>>> rec(num_supersteps, std::vector<v_commw_t<Graph_t>>(num_processors, 0));

    for (const auto &[key, val] : cs) {
        send[val][std::get<1>(key)] += dag.VertexCommWeight(std::get<0>(key)) * arch.sendCosts(std::get<1>(key), std::get<2>(key));
        rec[val][std::get<2>(key)] += dag.VertexCommWeight(std::get<0>(key)) * arch.sendCosts(std::get<1>(key), std::get<2>(key));
    }

    for (unsigned step = 0; step < numSupersteps; step++) {
        v_commw_t<Graph_t> maxComm = 0;
        for (unsigned proc = 0; proc < numProcessors; proc++) {
            maxComm = std::max(max_comm, send[step][proc]);
            maxComm = std::max(max_comm, rec[step][proc]);
        }

        model.SetMipStart(max_comm_superstep_var[static_cast<int>(step)], max_comm);
    }

    model.LoadMipStart();
    model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
}

template <typename GraphT>
void CoptCommScheduleOptimizer<GraphT>::SetupVariablesConstraintsObjective(const BspScheduleCS<GraphT> &schedule, Model &model) {
    const unsigned &maxNumberSupersteps = schedule.NumberOfSupersteps();
    const unsigned &numProcessors = schedule.GetInstance().NumberOfProcessors();
    const unsigned numVertices = static_cast<unsigned>(schedule.GetInstance().numberOfVertices());

    // variables indicating if superstep is used at all
    if (!ignoreLatency_) {
        superstep_has_comm = model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "superstep_has_comm");
    }

    max_comm_superstep_var = model.AddVars(static_cast<int>(max_number_supersteps), COPT_INTEGER, "max_comm_superstep");

    // communicate node from p1 to p2 at superstep

    comm_processor_to_processor_superstep_node_var = std::vector<std::vector<std::vector<VarArray>>>(
        num_processors, std::vector<std::vector<VarArray>>(num_processors, std::vector<VarArray>(max_number_supersteps)));

    for (unsigned p1 = 0; p1 < numProcessors; p1++) {
        for (unsigned p2 = 0; p2 < numProcessors; p2++) {
            for (unsigned step = 0; step < maxNumberSupersteps; step++) {
                comm_processor_to_processor_superstep_node_var[p1][p2][step]
                    = model.AddVars(static_cast<int>(numVertices), COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    if (!ignoreLatency_) {
        unsigned m = numProcessors * numProcessors * numVertices;
        for (unsigned int step = 0; step < schedule.NumberOfSupersteps(); step++) {
            Expr expr;

            for (unsigned p1 = 0; p1 < numProcessors; p1++) {
                for (unsigned p2 = 0; p2 < numProcessors; p2++) {
                    if (p1 != p2) {
                        for (unsigned node = 0; node < numVertices; node++) {
                            expr += comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node)];
                        }
                    }
                }
            }

            model.AddConstr(expr <= M * superstep_has_comm[static_cast<int>(step)]);
        }
    }
    // precedence constraint: if task is computed then all of its predecessors must have been present
    // and vertex is present where it was computed
    for (unsigned node = 0; node < numVertices; node++) {
        const unsigned &processor = schedule.assignedProcessor(node);
        const unsigned &superstep = schedule.assignedSuperstep(node);
        Expr expr;
        unsigned numComEdges = 0;
        for (const auto &pred : schedule.GetInstance().getComputationalDag().parents(node)) {
            if (schedule.assignedProcessor(node) != schedule.assignedProcessor(pred)) {
                numComEdges += 1;
                expr += comm_processor_to_processor_superstep_node_var[processor][processor][superstep][static_cast<int>(pred)];

                model.AddConstr(
                    comm_processor_to_processor_superstep_node_var[schedule.assignedProcessor(pred)][schedule.assignedProcessor(
                        pred)][schedule.assignedSuperstep(pred)][static_cast<int>(pred)]
                    == 1);
            }
        }

        if (numComEdges > 0) {
            model.AddConstr(expr >= numComEdges);
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < maxNumberSupersteps; step++) {
        for (unsigned int processor = 0; processor < numProcessors; processor++) {
            for (unsigned int node = 0; node < numVertices; node++) {
                if (processor == schedule.assignedProcessor(node) && step >= schedule.assignedSuperstep(node)) {
                    continue;
                }

                Expr expr1, expr2;
                if (step > 0) {
                    for (unsigned int pFrom = 0; pFrom < numProcessors; pFrom++) {
                        expr1
                            += comm_processor_to_processor_superstep_node_var[p_from][processor][step - 1][static_cast<int>(node)];
                    }
                }

                for (unsigned int pTo = 0; pTo < numProcessors; pTo++) {
                    expr2 += comm_processor_to_processor_superstep_node_var[processor][p_to][step][static_cast<int>(node)];
                }

                model.AddConstr(num_processors * expr1 >= expr2);
            }
        }
    }

    for (unsigned step = 0; step < maxNumberSupersteps; step++) {
        for (unsigned processor = 0; processor < numProcessors; processor++) {
            Expr expr1, expr2;
            for (unsigned node = 0; node < numVertices; node++) {
                for (unsigned pTo = 0; pTo < numProcessors; pTo++) {
                    if (processor != pTo) {
                        expr1 += schedule.GetInstance().getComputationalDag().VertexCommWeight(node)
                                 * schedule.GetInstance().sendCosts(processor, p_to)
                                 * comm_processor_to_processor_superstep_node_var[processor][p_to][step][static_cast<int>(node)];
                    }
                }

                for (unsigned int pFrom = 0; pFrom < numProcessors; pFrom++) {
                    if (processor != pFrom) {
                        expr2 += schedule.GetInstance().getComputationalDag().VertexCommWeight(node)
                                 * schedule.GetInstance().sendCosts(p_from, processor)
                                 * comm_processor_to_processor_superstep_node_var[p_from][processor][step][static_cast<int>(node)];
                    }
                }
            }

            model.AddConstr(max_comm_superstep_var[static_cast<int>(step)] >= expr1);
            model.AddConstr(max_comm_superstep_var[static_cast<int>(step)] >= expr2);
        }
    }

    /*
    Objective function
      */
    Expr expr;

    if (!ignoreLatency_) {
        for (unsigned int step = 0; step < maxNumberSupersteps; step++) {
            expr += schedule.GetInstance().CommunicationCosts() * max_comm_superstep_var[static_cast<int>(step)]
                    + schedule.GetInstance().SynchronisationCosts() * superstep_has_comm[static_cast<int>(step)];
        }
    } else {
        for (unsigned int step = 0; step < maxNumberSupersteps; step++) {
            expr += schedule.GetInstance().CommunicationCosts() * max_comm_superstep_var[static_cast<int>(step)];
        }
    }
    model.SetObjective(expr - schedule.GetInstance().SynchronisationCosts(), COPT_MINIMIZE);
}

}    // namespace osp
