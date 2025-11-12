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

template<typename Graph_t>
class CoptPartialScheduler {

    static_assert(is_computational_dag_v<Graph_t>, "CoptPartialScheduler can only be used with computational DAGs.");

    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;

    unsigned int timeLimitSeconds = 600;

  protected:

    unsigned start_superstep = 1, end_superstep = 3;

    std::vector<vertex_idx_t<Graph_t>> node_global_ID;
    std::unordered_map<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>> node_local_ID;

    std::vector<vertex_idx_t<Graph_t>> source_global_ID;
    std::unordered_map<vertex_idx_t<Graph_t>, vertex_idx_t<Graph_t>> source_local_ID;

    std::vector<std::pair<unsigned, unsigned> > node_needed_after_on_proc, source_needed_after_on_proc;
    std::vector<std::tuple<vertex_idx_t<Graph_t>, unsigned, unsigned, unsigned> > fixed_comm_steps;
    std::set<std::pair<unsigned, unsigned> > source_present_before;

    unsigned max_number_supersteps;

    VarArray superstep_used_var;
    VarArray keep_fixed_comm_step;

    std::vector<std::vector<VarArray>> node_to_processor_superstep_var;
    std::vector<std::vector<std::vector<VarArray>>> comm_processor_to_processor_superstep_node_var;
    std::vector<std::vector<VarArray>> comm_to_processor_superstep_source_var;

    void setupVariablesConstraintsObjective(const BspScheduleCS<Graph_t>& schedule, Model& model);

    void setInitialSolution(const BspScheduleCS<Graph_t>& schedule, Model &model);

    void updateSchedule(BspScheduleCS<Graph_t>& schedule) const;

    void setupVertexMaps(const BspScheduleCS<Graph_t>& schedule);

  public:

    virtual RETURN_STATUS improveSchedule(BspScheduleCS<Graph_t> &schedule);

    virtual std::string getScheduleName() const { return "ILPPartial"; }

    virtual void setTimeLimitSeconds(unsigned int limit) { timeLimitSeconds = limit; }
    inline unsigned int getTimeLimitSeconds() const { return timeLimitSeconds; }
    virtual void setStartAndEndSuperstep(unsigned start_, unsigned end_) { start_superstep = start_; end_superstep = end_; }
};

template<typename Graph_t>
RETURN_STATUS CoptPartialScheduler<Graph_t>::improveSchedule(BspScheduleCS<Graph_t>& schedule) {

    Envr env;
    Model model = env.CreateModel("bsp_schedule_partial");

    setupVertexMaps(schedule);

    setupVariablesConstraintsObjective(schedule, model);

    setInitialSolution(schedule, model);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds);
    model.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model.Solve();

    if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL))
        updateSchedule(schedule);

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
void CoptPartialScheduler<Graph_t>::setInitialSolution(const BspScheduleCS<Graph_t>& schedule, Model &model){

    const Graph_t& DAG = schedule.getInstance().getComputationalDag();
    const unsigned& num_processors = schedule.getInstance().numberOfProcessors();
    const auto &cs = schedule.getCommunicationSchedule();

    for (const vertex_idx_t<Graph_t> &node : DAG.vertices())
    {
        if(node_local_ID.find(node) == node_local_ID.end())
            continue;
        for (unsigned proc = 0; proc < num_processors; proc++)
            for(unsigned step = 0; step < max_number_supersteps; ++step)
            {
                if (schedule.assignedProcessor(node) == proc && schedule.assignedSuperstep(node) == start_superstep + step)
                    model.SetMipStart(node_to_processor_superstep_var[node_local_ID[node]][proc][static_cast<int>(step)], 1);
                else
                    model.SetMipStart(node_to_processor_superstep_var[node_local_ID[node]][proc][static_cast<int>(step)], 0);
            }
    }

    for (unsigned index = 0; index < fixed_comm_steps.size(); ++index)
        model.SetMipStart(keep_fixed_comm_step[static_cast<int>(index)], 1);

    for (const auto &node : DAG.vertices()) {

        if(node_local_ID.find(node) == node_local_ID.end())
            continue;

        for (unsigned p1 = 0; p1 < num_processors; p1++) {

            for (unsigned p2 = 0; p2 < num_processors; p2++) {

                if(p1 == p2)
                    continue;

                for (unsigned step = 0; step < max_number_supersteps && step <= end_superstep - start_superstep; step++) {

                    const auto &key = std::make_tuple(node, p1, p2);
                    if (cs.find(key) != cs.end() && cs.at(key) == start_superstep + step) 
                        model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node_local_ID[node])], 1);
                    else 
                        model.SetMipStart(comm_processor_to_processor_superstep_node_var[p1][p2][step][static_cast<int>(node_local_ID[node])], 0);
                }
            }
        }
    }

    for (const auto &source : DAG.vertices()) {

        if(source_local_ID.find(source) == source_local_ID.end())
            continue;

        for (unsigned proc = 0; proc < num_processors; proc++)
        {
            if(proc == schedule.assignedProcessor(source))
                continue;

            for (unsigned step = 0; step < max_number_supersteps + 1 && step <= end_superstep - start_superstep + 1; step++) {

                const auto &key = std::make_tuple(source, schedule.assignedProcessor(source), proc);
                if (cs.find(key) != cs.end() && cs.at(key) == start_superstep + step - 1) 
                    model.SetMipStart(comm_to_processor_superstep_source_var[proc][step][static_cast<int>(source_local_ID[source])], 1);
                else if(step > 0)
                    model.SetMipStart(comm_to_processor_superstep_source_var[proc][step][static_cast<int>(source_local_ID[source])], 0);
            }
        }
    }

    model.LoadMipStart();
    model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
}

template<typename Graph_t>
void CoptPartialScheduler<Graph_t>::updateSchedule(BspScheduleCS<Graph_t>& schedule) const {

    unsigned number_of_supersteps = 0;

    while (number_of_supersteps < max_number_supersteps &&
           superstep_used_var[static_cast<int>(number_of_supersteps)].Get(COPT_DBLINFO_VALUE) >= .99) {
        number_of_supersteps++;
    }

    const int offset = static_cast<int>(number_of_supersteps) - static_cast<int>(end_superstep - start_superstep + 1);

    for (vertex_idx_t<Graph_t> node = 0; node < schedule.getInstance().numberOfVertices(); node++)
        if(schedule.assignedSuperstep(node) > end_superstep)
            schedule.setAssignedSuperstep(node, static_cast<unsigned>(static_cast<int>(schedule.assignedSuperstep(node)) + offset));

    for (vertex_idx_t<Graph_t> node = 0; node < schedule.getInstance().numberOfVertices(); node++) {

        if(node_local_ID.find(node) == node_local_ID.end())
            continue;

        for (unsigned processor = 0; processor < schedule.getInstance().numberOfProcessors(); processor++) {

            for (unsigned step = 0; step < max_number_supersteps; step++) {

                if (node_to_processor_superstep_var[node_local_ID.at(node)][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99)
                {
                    schedule.setAssignedSuperstep(node, start_superstep + step);
                    schedule.setAssignedProcessor(node, processor);
                }
            }
        }
    }

    std::map<KeyTriple, unsigned int>& commSchedule = schedule.getCommunicationSchedule();

    std::vector<KeyTriple> toErase;
    for (const auto &[key, val] : schedule.getCommunicationSchedule())
    {
        if (val > end_superstep)
            commSchedule[key] = static_cast<unsigned>(static_cast<int>(val) + offset);
        else if (static_cast<int>(val) >= static_cast<int>(start_superstep) - 1)
            toErase.push_back(key);
    }
    for(const KeyTriple& key : toErase)
        commSchedule.erase(key);

    for (unsigned index = 0; index < fixed_comm_steps.size(); ++index)
    {
        const auto& entry = fixed_comm_steps[index];
        if (keep_fixed_comm_step[static_cast<int>(index)].Get(COPT_DBLINFO_VALUE) >= .99 &&
            std::get<3>(entry) < start_superstep + number_of_supersteps)
            commSchedule[std::make_tuple(std::get<0>(entry), std::get<1>(entry), std::get<2>(entry))] = std::get<3>(entry);
        else
            commSchedule[std::make_tuple(std::get<0>(entry), std::get<1>(entry), std::get<2>(entry))] = start_superstep-1;
    }

    for (vertex_idx_t<Graph_t> node = 0; node < node_global_ID.size(); node++) {

        for (unsigned int p_from = 0; p_from < schedule.getInstance().numberOfProcessors(); p_from++) {
            for (unsigned int p_to = 0; p_to < schedule.getInstance().numberOfProcessors(); p_to++) {
                if (p_from != p_to) {
                    for (unsigned int step = 0; step < max_number_supersteps; step++) {
                        if (comm_processor_to_processor_superstep_node_var[p_from][p_to][step][static_cast<int>(node)].Get(COPT_DBLINFO_VALUE) >= .99) {
                            commSchedule[std::make_tuple(node_global_ID[node], p_from, p_to)] = start_superstep + step;
                            break;
                        }
                    }
                }
            }
        }
    }

    for (vertex_idx_t<Graph_t> source = 0; source < source_global_ID.size(); source++) {

        for (unsigned int p_to = 0; p_to < schedule.getInstance().numberOfProcessors(); p_to++) {
            if (source_present_before.find(std::make_pair(source, p_to)) == source_present_before.end()) {
                for (unsigned int step = 0; step < max_number_supersteps + 1; step++) {
                    if (comm_to_processor_superstep_source_var[p_to][step][static_cast<int>(source)].Get(COPT_DBLINFO_VALUE) >= .99) {
                        commSchedule[std::make_tuple(source_global_ID[source], schedule.assignedProcessor(source_global_ID[source]), p_to)] =
                            start_superstep - 1 + step;
                        break;
                    }
                }
            }
        }
    }

    schedule.cleanCommSchedule();
    schedule.shrinkSchedule();

};


template<typename Graph_t>
void CoptPartialScheduler<Graph_t>::setupVariablesConstraintsObjective(const BspScheduleCS<Graph_t>& schedule, Model& model) {

    const vertex_idx_t<Graph_t> num_vertices = static_cast<vertex_idx_t<Graph_t>>(node_global_ID.size());
    const vertex_idx_t<Graph_t> num_sources = static_cast<vertex_idx_t<Graph_t>>(source_global_ID.size());
    const unsigned num_processors = schedule.getInstance().numberOfProcessors();

    /*
    Variables
    */
    // variables indicating if superstep is used at all
    superstep_used_var = model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "superstep_used");

    // variables for assigments of nodes to processor and superstep
    node_to_processor_superstep_var = std::vector<std::vector<VarArray>>(num_vertices, std::vector<VarArray>(num_processors));

    for (unsigned int node = 0; node < num_vertices; node++) {

        for (unsigned int processor = 0; processor < num_processors; processor++) {

            node_to_processor_superstep_var[node][processor] =
                model.AddVars(static_cast<int>(max_number_supersteps), COPT_BINARY, "node_to_processor_superstep");
        }
    }

    // communicate node from p1 to p2 at superstep

    comm_processor_to_processor_superstep_node_var = std::vector<std::vector<std::vector<VarArray>>>(num_processors,
        std::vector<std::vector<VarArray>>(num_processors, std::vector<VarArray>(max_number_supersteps)));

    for (unsigned int p1 = 0; p1 < num_processors; p1++) {
        for (unsigned int p2 = 0; p2 < num_processors; p2++) {
            for (unsigned int step = 0; step < max_number_supersteps; step++) {

                comm_processor_to_processor_superstep_node_var[p1][p2][step] =
                    model.AddVars(static_cast<int>(num_vertices), COPT_BINARY, "comm_processor_to_processor_superstep_node");
            }
        }
    }

    // communicate nodes in supersteps smaller than start_superstep
    comm_to_processor_superstep_source_var = std::vector<std::vector<VarArray>>(num_processors, std::vector<VarArray>(max_number_supersteps + 1));
    std::vector<std::vector<VarArray>> present_on_processor_superstep_source_var = std::vector<std::vector<VarArray>>(num_processors, std::vector<VarArray>(max_number_supersteps));

    for (unsigned int proc = 0; proc < num_processors; proc++) {
        for (unsigned int step = 0; step < max_number_supersteps + 1; step++) {

            comm_to_processor_superstep_source_var[proc][step] =
                model.AddVars(static_cast<int>(num_sources), COPT_BINARY, "comm_to_processor_superstep_source");
            
            if(step < max_number_supersteps)
                present_on_processor_superstep_source_var[proc][step] =
                    model.AddVars(static_cast<int>(num_sources), COPT_BINARY, "present_on_processor_superstep_source");
        }
    }

    VarArray max_comm_superstep_var = model.AddVars(static_cast<int>(max_number_supersteps + 1), COPT_INTEGER, "max_comm_superstep");

    VarArray max_work_superstep_var = model.AddVars(static_cast<int>(max_number_supersteps), COPT_INTEGER, "max_work_superstep");

    keep_fixed_comm_step = model.AddVars(static_cast<int>(fixed_comm_steps.size()), COPT_BINARY, "keep_fixed_comm_step");

    /*
    Constraints
      */

    //  use consecutive supersteps starting from 0
    model.AddConstr(superstep_used_var[0] == 1);

    for (unsigned int step = 0; step < max_number_supersteps - 1; step++) {
        model.AddConstr(superstep_used_var[static_cast<int>(step)] >= superstep_used_var[static_cast<int>(step + 1)]);
    }

    // superstep is used at all
    unsigned large_constant = static_cast<unsigned>(num_vertices+num_sources) * num_processors * num_processors * 2;
    for (unsigned int step = 0; step < max_number_supersteps; step++) {

        Expr expr;
        for (vertex_idx_t<Graph_t> node = 0; node < num_vertices; node++) {

            for (unsigned int processor = 0; processor < num_processors; processor++) {
                expr += node_to_processor_superstep_var[node][processor][static_cast<int>(step)];
                
                for (unsigned int p_other = 0; p_other < num_processors; p_other++)
                    if(processor != p_other)
                        expr += comm_processor_to_processor_superstep_node_var[processor][p_other][step][static_cast<int>(node)];
            }
        }
        for (vertex_idx_t<Graph_t> source = 0; source < num_sources; source++)
            for (unsigned int processor = 0; processor < num_processors; processor++)
                if(source_present_before.find(std::make_pair(source, processor)) == source_present_before.end())
                    expr += comm_to_processor_superstep_source_var[processor][step+1][static_cast<int>(source)]; 

        model.AddConstr(expr <= large_constant * superstep_used_var[static_cast<int>(step)]);
    }

    // nodes are assigend
    for (vertex_idx_t<Graph_t> node = 0; node < num_vertices; node++) {

        Expr expr;
        for (unsigned int processor = 0; processor < num_processors; processor++) {

            for (unsigned int step = 0; step < max_number_supersteps; step++) {
                expr += node_to_processor_superstep_var[node][processor][static_cast<int>(step)];
            }
        }

        model.AddConstr(expr == 1);
    }

    // precedence constraint: if task is computed then all of its predecessors must have been present
    for (vertex_idx_t<Graph_t> node = 0; node < num_vertices; node++) {
        for (unsigned int step = 0; step < max_number_supersteps; step++) {
            for (unsigned int processor = 0; processor < num_processors; processor++) {

                Expr expr;
                unsigned num_terms = 0;
                for (const auto &pred : schedule.getInstance().getComputationalDag().parents(node_global_ID[node]))
                {
                    if(node_local_ID.find(pred) != node_local_ID.end())
                    {
                        ++num_terms;
                        expr += comm_processor_to_processor_superstep_node_var[processor][processor][step][static_cast<int>(node_local_ID[pred])];
                    }
                    else if(source_local_ID.find(pred) != source_local_ID.end() &&
                            source_present_before.find(std::make_pair(source_local_ID[pred], processor)) == source_present_before.end())
                    {
                        ++num_terms;
                        expr += present_on_processor_superstep_source_var[processor][step][static_cast<int>(source_local_ID[pred])];
                    }
                }

                if(num_terms > 0)
                    model.AddConstr(expr >= num_terms * node_to_processor_superstep_var[node][processor][static_cast<int>(step)]);
            }
        }
    }
    
    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < num_processors; processor++) {
            for (vertex_idx_t<Graph_t> node = 0; node < num_vertices; node++) {

                Expr expr1, expr2;
                if (step > 0) {

                    for (unsigned int p_from = 0; p_from < num_processors; p_from++) {
                        expr1 += comm_processor_to_processor_superstep_node_var[p_from][processor][step - 1][static_cast<int>(node)];
                    }
                }

                expr1 += node_to_processor_superstep_var[node][processor][static_cast<int>(step)];

                for (unsigned int p_to = 0; p_to < num_processors; p_to++) {
                    expr2 += comm_processor_to_processor_superstep_node_var[processor][p_to][step][static_cast<int>(node)];
                }

                model.AddConstr(num_processors * (expr1) >= expr2);
            }
        }
    }

    // combines two constraints: node can only be communicated if it is present; and node is present if it was computed
    // or communicated
    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < num_processors; processor++) {
            for (vertex_idx_t<Graph_t> source_node = 0; source_node < num_sources; source_node++) {

                if(source_present_before.find(std::make_pair(source_node, processor)) != source_present_before.end())
                    continue;

                Expr expr1 = comm_to_processor_superstep_source_var[processor][step][static_cast<int>(source_node)];
                if (step > 0)
                    expr1 += present_on_processor_superstep_source_var[processor][step-1][static_cast<int>(source_node)];

                Expr expr2 = present_on_processor_superstep_source_var[processor][step][static_cast<int>(source_node)];

                model.AddConstr(expr1 >= expr2);
            }
        }
    }

    // boundary conditions at the end
    for(const std::pair<vertex_idx_t<Graph_t>, unsigned>&  node_and_proc : node_needed_after_on_proc)
    {
        Expr expr;
        for (unsigned int p_from = 0; p_from < num_processors; p_from++)
            expr += comm_processor_to_processor_superstep_node_var[p_from][node_and_proc.second][max_number_supersteps - 1][static_cast<int>(node_and_proc.first)];

        model.AddConstr(expr >= 1);
    }

    for(const std::pair<vertex_idx_t<Graph_t>, unsigned>&  source_and_proc : source_needed_after_on_proc)
    {
        Expr expr = present_on_processor_superstep_source_var[source_and_proc.second][max_number_supersteps - 1][static_cast<int>(source_and_proc.first)];
        expr += comm_to_processor_superstep_source_var[source_and_proc.second][max_number_supersteps][static_cast<int>(source_and_proc.first)];
        model.AddConstr(expr >= 1);
    }

    // cost calculation - work
    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < num_processors; processor++) {

            Expr expr;
            for (unsigned int node = 0; node < num_vertices; node++) {
                expr += schedule.getInstance().getComputationalDag().vertex_work_weight(node_global_ID[node]) *
                        node_to_processor_superstep_var[node][processor][static_cast<int>(step)];
            }

            model.AddConstr(max_work_superstep_var[static_cast<int>(step)] >= expr);
        }
    }

    // cost calculation - comm
    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        for (unsigned int processor = 0; processor < num_processors; processor++) {

            Expr expr1, expr2;
            for (vertex_idx_t<Graph_t> node = 0; node < num_vertices; node++) {
                for (unsigned int p_other = 0; p_other < num_processors; p_other++) {
                    if (processor != p_other) {
                        expr1 += schedule.getInstance().getComputationalDag().vertex_comm_weight(node_global_ID[node]) *
                                schedule.getInstance().sendCosts(processor, p_other) *
                                comm_processor_to_processor_superstep_node_var[processor][p_other][step][static_cast<int>(node)];
                        expr2 += schedule.getInstance().getComputationalDag().vertex_comm_weight(node_global_ID[node]) *
                                schedule.getInstance().sendCosts(p_other, processor) *
                                comm_processor_to_processor_superstep_node_var[p_other][processor][step][static_cast<int>(node)];
                    }
                }
            }

            for (vertex_idx_t<Graph_t> source = 0; source < num_sources; source++) {
                const unsigned origin_proc = schedule.assignedProcessor(source_global_ID[source]);
                if(origin_proc == processor)
                {
                    for (unsigned int p_other = 0; p_other < num_processors; p_other++)
                    {
                        expr1 += schedule.getInstance().getComputationalDag().vertex_comm_weight(source_global_ID[source]) *
                        schedule.getInstance().sendCosts(processor, p_other) *
                        comm_to_processor_superstep_source_var[p_other][step + 1][static_cast<int>(source)];
                    }
                }
                expr2 +=
                    schedule.getInstance().getComputationalDag().vertex_comm_weight(source_global_ID[source]) *
                    schedule.getInstance().sendCosts(origin_proc, processor) *
                    comm_to_processor_superstep_source_var[processor][step + 1][static_cast<int>(source)];
            }

            for (unsigned index = 0; index < fixed_comm_steps.size(); ++index)
            {
                const auto& entry = fixed_comm_steps[index];
                if(std::get<3>(entry) != start_superstep + step)
                    continue;
                if(std::get<1>(entry) == processor)
                    expr1 += schedule.getInstance().getComputationalDag().vertex_comm_weight(std::get<0>(entry)) *
                        schedule.getInstance().sendCosts(processor, std::get<2>(entry)) *
                        keep_fixed_comm_step[static_cast<int>(index)];
                if(std::get<2>(entry) == processor)
                    expr2 += schedule.getInstance().getComputationalDag().vertex_comm_weight(std::get<0>(entry)) *
                        schedule.getInstance().sendCosts(std::get<1>(entry), processor) *
                        keep_fixed_comm_step[static_cast<int>(index)];
            }

            model.AddConstr(max_comm_superstep_var[static_cast<int>(step + 1)] >= expr1);
            model.AddConstr(max_comm_superstep_var[static_cast<int>(step + 1)] >= expr2);
        }
    }

    // cost calculation - first comm phase handled separately
    for (unsigned int processor = 0; processor < num_processors; processor++) {

        Expr expr1, expr2;
        for (vertex_idx_t<Graph_t> source = 0; source < num_sources; source++) {
            const unsigned origin_proc = schedule.assignedProcessor(source_global_ID[source]);
            if(origin_proc == processor)
            {
                for (unsigned int p_other = 0; p_other < num_processors; p_other++)
                {
                    expr1 += schedule.getInstance().getComputationalDag().vertex_comm_weight(source_global_ID[source]) *
                    schedule.getInstance().sendCosts(processor, p_other) *
                    comm_to_processor_superstep_source_var[p_other][0][static_cast<int>(source)];
                }
            }
            expr2 +=
                schedule.getInstance().getComputationalDag().vertex_comm_weight(source_global_ID[source]) *
                schedule.getInstance().sendCosts(origin_proc, processor) *
                comm_to_processor_superstep_source_var[processor][0][static_cast<int>(source)];
        }

        for (unsigned index = 0; index < fixed_comm_steps.size(); ++index)
        {
            const auto& entry = fixed_comm_steps[index];
            if(std::get<1>(entry) == processor)
                expr1 += schedule.getInstance().getComputationalDag().vertex_comm_weight(std::get<0>(entry)) *
                    schedule.getInstance().sendCosts(processor, std::get<2>(entry)) *
                    (1-keep_fixed_comm_step[static_cast<int>(index)]);
            if(std::get<2>(entry) == processor)
                expr2 += schedule.getInstance().getComputationalDag().vertex_comm_weight(std::get<0>(entry)) *
                    schedule.getInstance().sendCosts(std::get<1>(entry), processor) *
                    (1-keep_fixed_comm_step[static_cast<int>(index)]);
        }

        model.AddConstr(max_comm_superstep_var[0] >= expr1);
        model.AddConstr(max_comm_superstep_var[0] >= expr2);
    }

    /*
    Objective function
    */
    Expr expr;

    for (unsigned int step = 0; step < max_number_supersteps; step++) {
        expr += max_work_superstep_var[static_cast<int>(step)] + schedule.getInstance().communicationCosts() * max_comm_superstep_var[static_cast<int>(step + 1)] +
                schedule.getInstance().synchronisationCosts() * superstep_used_var[static_cast<int>(step)];
    }

    expr += schedule.getInstance().communicationCosts() * max_comm_superstep_var[0];

    model.SetObjective(expr, COPT_MINIMIZE);
};

template<typename Graph_t>
void CoptPartialScheduler<Graph_t>::setupVertexMaps(const BspScheduleCS<Graph_t>& schedule) {

    node_local_ID.clear();
    node_global_ID.clear();
    source_local_ID.clear();
    source_global_ID.clear();

    node_needed_after_on_proc.clear();
    source_needed_after_on_proc.clear();
    fixed_comm_steps.clear();
    source_present_before.clear();

    std::vector<std::vector<unsigned> > first_at = schedule.getFirstPresence();

    max_number_supersteps = end_superstep - start_superstep + 3;

    for (unsigned node = 0; node < schedule.getInstance().numberOfVertices(); node++) {

        if (schedule.assignedSuperstep(node) >= start_superstep && schedule.assignedSuperstep(node) <= end_superstep) {

            node_local_ID[node] = static_cast<vertex_idx_t<Graph_t>>(node_global_ID.size());
            node_global_ID.push_back(node);

            for (const auto &pred : schedule.getInstance().getComputationalDag().parents(node)) {

                if (schedule.assignedSuperstep(pred) < start_superstep) {

                    if (source_local_ID.find(pred) == source_local_ID.end()) {
                        source_local_ID[pred] = static_cast<vertex_idx_t<Graph_t>>(source_global_ID.size());
                        source_global_ID.push_back(pred);
                    }

                } else if (schedule.assignedSuperstep(pred) > end_superstep) {

                    throw std::invalid_argument("Initial Schedule might be invalid?!");
                }
            }
        }
    }

    // find where the sources are already present before the segment
    for(const auto& source_and_ID : source_local_ID)
    {
        vertex_idx_t<Graph_t> source = source_and_ID.first;
        for(unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc)
            if(first_at[source][proc] < start_superstep)
                source_present_before.emplace(std::make_pair(source_and_ID.second, proc));
    }

    // collect values that are needed by the end of the segment
    for(const auto& source_and_ID : source_local_ID)
    {
        vertex_idx_t<Graph_t> source = source_and_ID.first;

        std::set<unsigned> procs_needing_this;
        for (const auto &succ : schedule.getInstance().getComputationalDag().children(source))
            if(schedule.assignedProcessor(succ) != schedule.assignedProcessor(source) &&
                schedule.assignedSuperstep(succ) > end_superstep)
                procs_needing_this.insert(schedule.assignedProcessor(succ));

        for(unsigned proc1 = 0; proc1 < schedule.getInstance().numberOfProcessors(); ++proc1)
            for(unsigned proc2 = 0; proc2 < schedule.getInstance().numberOfProcessors(); ++proc2)
            {
                if(proc1 == proc2)
                    continue;          
                auto itr = schedule.getCommunicationSchedule().find(std::make_tuple(source, proc1, proc2));
                if (itr != schedule.getCommunicationSchedule().end() && itr->second > end_superstep)
                    procs_needing_this.insert(schedule.assignedProcessor(proc1));
            }
        
        for(unsigned proc : procs_needing_this)
            if(first_at[source][proc] >= start_superstep && first_at[source][proc] <= end_superstep + 1)
                source_needed_after_on_proc.emplace_back(source_and_ID.second, proc);
    }
    for(const auto& node_and_ID : node_local_ID)
    {
        vertex_idx_t<Graph_t> node = node_and_ID.first;

        std::set<unsigned> procs_needing_this;
        for (const auto &succ : schedule.getInstance().getComputationalDag().children(node))
            if(schedule.assignedSuperstep(succ) > end_superstep)
                procs_needing_this.insert(schedule.assignedProcessor(succ));

        for(unsigned proc1 = 0; proc1 < schedule.getInstance().numberOfProcessors(); ++proc1)
            for(unsigned proc2 = 0; proc2 < schedule.getInstance().numberOfProcessors(); ++proc2)
            {                
                auto itr = schedule.getCommunicationSchedule().find(std::make_tuple(node, proc1, proc2));
                if (itr != schedule.getCommunicationSchedule().end() && proc1 != proc2 && itr->second > end_superstep)
                    procs_needing_this.insert(schedule.assignedProcessor(proc1));
            }
        
        for(unsigned proc : procs_needing_this)
            if(first_at[node][proc] <= end_superstep + 1)
                node_needed_after_on_proc.emplace_back(node_and_ID.second, proc);
    }


    // comm steps that just happen to be in this interval, but not connected to the nodes within
    for (const auto &[key, val] : schedule.getCommunicationSchedule())
    {
        vertex_idx_t<Graph_t> source = std::get<0>(key);
        if(source_local_ID.find(source) == source_local_ID.end() && 
            schedule.assignedSuperstep(source) < start_superstep &&
            val >= start_superstep - 1 && val <= end_superstep)
                fixed_comm_steps.emplace_back(std::get<0>(key), std::get<1>(key), std::get<2>(key), val);
    }

};

} // namespace osp