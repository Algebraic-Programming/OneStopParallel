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

template<typename Graph_t>
class CoptCommScheduleOptimizer {

    static_assert(is_computational_dag_v<Graph_t>, "CoptFullScheduler can only be used with computational DAGs.");

    bool num_supersteps_can_change = true;

    unsigned int timeLimitSeconds = 600;

  protected:

    VarArray superstep_used_var;
    VarArray max_comm_superstep_var;
    std::vector<std::vector<std::vector<VarArray>>> comm_processor_to_processor_superstep_node_var;

    void setupVariablesConstraintsObjective(const BspScheduleCS<Graph_t>& schedule, Model& model);

    void setInitialSolution(BspScheduleCS<Graph_t>& schedule, Model &model);

    bool canShrinkResultingSchedule(unsigned number_of_supersteps) const;

    void updateCommSchedule(BspScheduleCS<Graph_t>& schedule) const;

  public:

    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;
    virtual ~CoptCommScheduleOptimizer() = default;

    virtual RETURN_STATUS improveSchedule(BspScheduleCS<Graph_t> &schedule);

    virtual std::string getScheduleName() const { return "ILPCommunication"; }

    virtual void setTimeLimitSeconds(unsigned int limit) { timeLimitSeconds = limit; }
    inline unsigned int getTimeLimitSeconds() const { return timeLimitSeconds; }
    virtual void setNumSuperstepsCanChange(bool can_change_) { num_supersteps_can_change = can_change_; }
};


template<typename Graph_t>
RETURN_STATUS CoptCommScheduleOptimizer<Graph_t>::improveSchedule(BspScheduleCS<Graph_t>& schedule) {

    Envr env;
    Model model = env.CreateModel("bsp_schedule_cs");

    setupVariablesConstraintsObjective(schedule, model);

    setInitialSolution(schedule, model);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
    model.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model.Solve();

    if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL))
    {
        updateCommSchedule(schedule);
        if (canShrinkResultingSchedule(schedule.numberOfSupersteps()))
           schedule.shrinkByMergingSupersteps();
    }

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        return RETURN_STATUS::OSP_SUCCESS;
    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;
    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL))
            return RETURN_STATUS::BEST_FOUND;
        else
            return RETURN_STATUS::TIMEOUT;
    }
}

template<typename Graph_t>
bool CoptCommScheduleOptimizer<Graph_t>::canShrinkResultingSchedule(unsigned number_of_supersteps) const {

    for (unsigned step = 0; step < number_of_supersteps - 1; step++) {

        if (superstep_used_var[static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) <= 0.01)
            return true;
    }
    return false;
}

template<typename Graph_t>
void CoptCommScheduleOptimizer<Graph_t>::updateCommSchedule(BspScheduleCS<Graph_t>& schedule) const {

    std::map<KeyTriple, unsigned int>& cs = schedule.getCommunicationSchedule();
    cs.clear();

    for (const auto &node : schedule.getInstance().vertices()) {

        for (unsigned int p_from = 0; p_from < schedule.getInstance().numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < schedule.getInstance().numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (unsigned int step = 0; step < schedule.numberOfSupersteps(); step++) {
                        if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][static_cast<int>(node)].Get(
                                COPT_DBLINFO_VALUE) >= .99) {
                            cs[std::make_tuple(node, p_from, p_to)] = step;
                        }
                    }
                }
            }
        }
    }
}

template<typename Graph_t>
void CoptCommScheduleOptimizer<Graph_t>::setInitialSolution(BspScheduleCS<Graph_t>& schedule, Model &model){

    const Graph_t& DAG = schedule.getInstance().getComputationalDag();
    const BspArchitecture<Graph_t>& arch = schedule.getInstance().getArchitecture();
    const unsigned& num_processors = schedule.getInstance().numberOfProcessors();
    const unsigned& num_supersteps = schedule.numberOfSupersteps();
    const auto &cs = schedule.getCommunicationSchedule();

    std::vector<std::vector<unsigned> > first_at(DAG.num_vertices(), std::vector<unsigned>(num_processors, std::numeric_limits<unsigned>::max()));
    for (const auto &node : DAG.vertices())
            first_at[node][schedule.assignedProcessor(node)] = schedule.assignedSuperstep(node);

    for (const auto &node : DAG.vertices()) {

        for (unsigned p1 = 0; p1 < num_processors; p1++) {

            for (unsigned p2 = 0; p2 < num_processors; p2++) {

                if(p1 == p2)
                    continue;

                for (unsigned step = 0; step < num_supersteps; step++) {

                    const auto &key = std::make_tuple(node, p1, p2);
                    if (cs.find(key) != cs.end() && cs.at(key) == step) {
                        model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node)], 1);
                        first_at[node][p2] = std::min(first_at[node][p2], step+1);
                    } else {
                        model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node)], 0);
                    }
                }
            }
        }
    }

    for (const auto &node : DAG.vertices())
        for (unsigned proc = 0; proc < num_processors; proc++)
                for (unsigned step = 0; step < num_supersteps; step++)
                {
                    if(step >= first_at[node][proc])
                        model.SetMipStart(comm_processor_to_processor_superstep_node_var[proc][proc][step]
                                                                                        [static_cast<int>(node)], 1);
                    else
                        model.SetMipStart(comm_processor_to_processor_superstep_node_var[proc][proc][step]
                                                                                        [static_cast<int>(node)], 0);
                }

    if(num_supersteps_can_change)
    {
        std::vector<unsigned> comm_phase_used(num_supersteps, 0);
        for (auto const &[key, val] : cs)
            comm_phase_used[val] = 1;
        for (unsigned step = 0; step < num_supersteps; step++)
            model.SetMipStart(superstep_used_var[static_cast<int>(step)], comm_phase_used[step]);
    }

    std::vector<std::vector<v_commw_t<Graph_t>>> send(num_supersteps, std::vector<v_commw_t<Graph_t>>(num_processors, 0));
    std::vector<std::vector<v_commw_t<Graph_t>>> rec(num_supersteps, std::vector<v_commw_t<Graph_t>>(num_processors, 0));

    for (const auto &[key, val] : cs) {
        send[val][std::get<1>(key)] += DAG.vertex_comm_weight(std::get<0>(key)) * arch.sendCosts(std::get<1>(key), std::get<2>(key));
        rec[val][std::get<2>(key)] += DAG.vertex_comm_weight(std::get<0>(key)) * arch.sendCosts(std::get<1>(key), std::get<2>(key));
    }

    for (unsigned step = 0; step < num_supersteps; step++) {

        v_commw_t<Graph_t> max_comm = 0;
        for (unsigned proc = 0; proc < num_processors; proc++) {
            max_comm = std::max(max_comm, send[step][proc]);
            max_comm = std::max(max_comm, rec[step][proc]);
        }

        model.SetMipStart(max_comm_superstep_var[static_cast<int>(step)], max_comm);
    }

    model.LoadMipStart();
    model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
}

template<typename Graph_t>
void CoptCommScheduleOptimizer<Graph_t>::setupVariablesConstraintsObjective(const BspScheduleCS<Graph_t>& schedule, Model& model) {

    const unsigned &max_number_supersteps = schedule.numberOfSupersteps();
    const unsigned &num_processors = schedule.getInstance().numberOfProcessors();
    const unsigned num_vertices = static_cast<unsigned>(schedule.getInstance().numberOfVertices());

    // variables indicating if superstep is used at all
    if (num_supersteps_can_change) {
        superstep_used_var = model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "superstep_used");
    }

    max_comm_superstep_var = model.AddVars(static_cast<int>(max_number_supersteps), COPT_INTEGER, "max_comm_superstep");

    // communicate node from p1 to p2 at superstep

    comm_processor_to_processor_superstep_node_var = std::vector<std::vector<std::vector<VarArray>>>(num_processors,
                                        std::vector<std::vector<VarArray>>(num_processors,  std::vector<VarArray>(max_number_supersteps)));

    for (unsigned p1 = 0; p1 < num_processors; p1++) {

        for (unsigned p2 = 0; p2 < num_processors; p2++) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {

                comm_processor_to_processor_superstep_node_var[p1][p2][step] = model.AddVars(static_cast<int>(num_vertices),
                                                        COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    if (num_supersteps_can_change) {
        unsigned M = num_processors * num_processors * num_vertices;
        for (unsigned int step = 0; step < schedule.numberOfSupersteps(); step++) {

            Expr expr;

            for (unsigned p1 = 0; p1 < num_processors; p1++) {

                for (unsigned p2 = 0; p2 < num_processors; p2++) {

                    if (p1 != p2) {
                        for (unsigned node = 0; node < num_vertices; node++) {

                            expr += comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node)];
                        }
                    }
                }
            }

            model.AddConstr(expr <= M * superstep_used_var[static_cast<int>(step)]);
        }
    }
    // precedence constraint: if task is computed then all of its predecessors must have been present
    // and vertex is present where it was computed
    for (unsigned node = 0; node < num_vertices; node++) {

        const unsigned &processor = schedule.assignedProcessor(node);
        const unsigned &superstep = schedule.assignedSuperstep(node);
        Expr expr;
        unsigned num_com_edges = 0;
        for (const auto &pred : schedule.getInstance().getComputationalDag().parents(node)) {

            if (schedule.assignedProcessor(node) != schedule.assignedProcessor(pred)) {
                num_com_edges += 1;
                expr += comm_processor_to_processor_superstep_node_var[processor][processor][superstep][static_cast<int>(pred)];

                model.AddConstr(
                    comm_processor_to_processor_superstep_node_var[schedule.assignedProcessor(pred)][schedule.assignedProcessor(pred)]
                                                                  [schedule.assignedSuperstep(pred)][static_cast<int>(pred)] == 1);
            }
        }

        if (num_com_edges > 0)
            model.AddConstr(expr >= num_com_edges);
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < num_processors; processor++) {
            for (unsigned int node = 0; node < num_vertices; node++) {

                if (processor == schedule.assignedProcessor(node) && step >= schedule.assignedSuperstep(node))
                    continue;

                Expr expr1, expr2;
                if (step > 0) {

                    for (unsigned int p_from = 0; p_from < num_processors; p_from++) {
                        expr1 += comm_processor_to_processor_superstep_node_var[p_from][processor][step - 1][static_cast<int>(node)];
                    }
                }

                for (unsigned int p_to = 0; p_to < num_processors; p_to++) {
                    expr2 += comm_processor_to_processor_superstep_node_var[processor][p_to][step][static_cast<int>(node)];
                }

                model.AddConstr(num_processors * expr1 >= expr2);
            }
        }
    }

    for (unsigned step = 0; step < max_number_supersteps; step++) {
        for (unsigned processor = 0; processor < num_processors; processor++) {

            Expr expr1, expr2;
            for (unsigned node = 0; node < num_vertices; node++) {

                for (unsigned p_to = 0; p_to < num_processors; p_to++) {
                    if (processor != p_to) {
                        expr1 += schedule.getInstance().getComputationalDag().vertex_comm_weight(node) *
                                schedule.getInstance().sendCosts(processor, p_to) *
                                comm_processor_to_processor_superstep_node_var[processor][p_to][step][static_cast<int>(node)];
                    }
                }

                for (unsigned int p_from = 0; p_from < num_processors; p_from++) {
                    if (processor != p_from) {
                        expr2 += schedule.getInstance().getComputationalDag().vertex_comm_weight(node) *
                                schedule.getInstance().sendCosts(p_from, processor) *
                                comm_processor_to_processor_superstep_node_var[p_from][processor][step][static_cast<int>(node)];
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

    if (num_supersteps_can_change) {

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            expr += schedule.getInstance().communicationCosts() * max_comm_superstep_var[static_cast<int>(step)] +
                    schedule.getInstance().synchronisationCosts() * superstep_used_var[static_cast<int>(step)];
        }
    } else {

        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            expr += schedule.getInstance().communicationCosts() * max_comm_superstep_var[static_cast<int>(step)];
        }
    }
    model.SetObjective(expr - schedule.getInstance().synchronisationCosts(), COPT_MINIMIZE);
}

} // namespace osp