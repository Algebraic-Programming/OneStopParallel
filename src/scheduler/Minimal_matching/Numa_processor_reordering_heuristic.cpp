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

#include "scheduler/Minimal_matching/Numa_processor_reordering_heuristic.hpp"

std::vector<std::vector<unsigned>> numa_processor_reordering_heuristic::comp_p2p_comm(const BspSchedule & schedule) {
    std::vector<std::vector<unsigned>> proc_to_proc_comm = std::vector<std::vector<unsigned>>(schedule.getInstance().numberOfProcessors(), std::vector<unsigned>(schedule.getInstance().numberOfProcessors(), 0 ));

    for (size_t node = 0; node < schedule.getInstance().numberOfVertices(); node++) {
        for (const auto& chld_edge : schedule.getInstance().getComputationalDag().out_edges(node)) {
            if ( schedule.assignedProcessor(node) != schedule.assignedProcessor(chld_edge.m_target) ) {
                proc_to_proc_comm[schedule.assignedProcessor(node)][schedule.assignedProcessor(chld_edge.m_target)] += schedule.getInstance().getComputationalDag().nodeCommunicationWeight(node);
            }
        }
    }

    return proc_to_proc_comm;
}


RETURN_STATUS numa_processor_reordering_heuristic::improveSchedule(BspSchedule &schedule) {
    std::pair<RETURN_STATUS, BspSchedule> out = constructImprovedSchedule(schedule);
    schedule = out.second;
    return out.first;
}

std::pair<RETURN_STATUS, BspSchedule> numa_processor_reordering_heuristic::constructImprovedSchedule(const BspSchedule &schedule) {
    if (! schedule.getInstance().getArchitecture().isNumaArchitecture() ) return {SUCCESS, schedule};

    BspSchedule new_sched(schedule.getInstance());
    new_sched.setAssignedSupersteps( schedule.assignedSupersteps() );
    
    std::vector<unsigned> proc_reorder = compute_best_reordering(schedule);
    for (size_t node = 0 ; node < schedule.getInstance().numberOfVertices(); node++) {
        new_sched.setAssignedProcessor(node, proc_reorder[schedule.assignedProcessor(node)]);
    }

    new_sched.setAutoCommunicationSchedule();
    return std::make_pair(SUCCESS, new_sched);
}


std::vector<unsigned> numa_processor_reordering_heuristic::compute_best_reordering(const BspSchedule & schedule) {
    size_t num_proc = schedule.getInstance().numberOfProcessors();
    std::vector<std::vector<unsigned>> proc_to_proc_comm = comp_p2p_comm(schedule);
    std::vector<std::vector<unsigned>> numa_coeff = schedule.getInstance().getArchitecture().sendCostMatrixCopy();

    Envr env;
    Model coptModel = env.CreateModel("Processor_Reordering");

    std::vector<VarArray> proc_to_proc_assignment(num_proc);
    for (size_t proc = 0; proc < num_proc; proc++) {
        proc_to_proc_assignment[proc] = coptModel.AddVars(num_proc, COPT_BINARY, "p2p");
    }

    // Partition constraints
    // function
    for (size_t p_dom = 0; p_dom < num_proc; p_dom++) {
        Expr expr;
        for (size_t p_img = 0; p_img < num_proc; p_img++) {
            expr += proc_to_proc_assignment[p_dom][p_img];
        }
        coptModel.AddConstr(expr == 1);
    }
    // bijective
    for (size_t p_img = 0; p_img < num_proc; p_img++) {
        Expr expr;
        for (size_t p_dom = 0; p_dom < num_proc; p_dom++) {
            expr += proc_to_proc_assignment[p_dom][p_img];
        }
        coptModel.AddConstr(expr == 1);
    }

    //And variables
    std::vector<std::vector<std::vector<std::vector<VarArray>>>> proc_and_assignments(num_proc, std::vector<std::vector<std::vector<VarArray>>>(num_proc, std::vector<std::vector<VarArray>>(num_proc, std::vector<VarArray>(num_proc))));
    for (size_t p1 = 0; p1 < num_proc; p1++) {
        for (size_t p2 = 0; p2 < num_proc; p2++) {
            for (size_t p1_img = 0; p1_img < num_proc; p1_img++) {
                for (size_t p2_img = 0; p2_img < num_proc; p2_img++) {
                    proc_and_assignments[p1][p2][p1_img][p2_img] = coptModel.AddVars(1,COPT_BINARY, "and");
                    coptModel.AddConstr(proc_and_assignments[p1][p2][p1_img][p2_img][0] >= proc_to_proc_assignment[p1][p1_img] + proc_to_proc_assignment[p2][p2_img] - 1.001);
                }
            }
        }
    }

    Expr expr_objective;
    for (size_t p1 = 0; p1 < num_proc; p1++) {
        for (size_t p2 = 0; p2 < num_proc; p2++) {
            if (p1 == p2) continue;
            for (size_t p1_img = 0; p1_img < num_proc; p1_img++) {
                for (size_t p2_img = 0; p2_img < num_proc; p2_img++) {
                    expr_objective += proc_and_assignments[p1][p2][p1_img][p2_img][0] * proc_to_proc_comm[p1][p2] * numa_coeff[p1_img][p2_img];
                }
            }
        }
    }
    coptModel.SetObjective(expr_objective, COPT_MINIMIZE);

    for (size_t p_dom = 0; p_dom < num_proc; p_dom++) {
        for (size_t p_img = 0; p_img < num_proc; p_img++) {
            coptModel.SetMipStart(proc_to_proc_assignment[p_dom][p_img], p_dom == p_img? 1 : 0);
        }
    }

    coptModel.LoadMipStart();
    coptModel.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
    coptModel.SetIntParam(COPT_INTPARAM_THREADS, 16);

    coptModel.Solve();

    // generating output
    std::vector<unsigned> output(num_proc, 0);
    if (coptModel.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

        for (size_t proc = 0; proc < num_proc; proc++) {
            for (size_t proc_img = 0; proc_img < num_proc; proc_img++) {
                if (proc_to_proc_assignment[proc][proc_img].Get(COPT_DBLINFO_VALUE) >= 0.99) {
                    output[proc] = proc_img;
                }
            }
        }

        std::vector<unsigned> id_vec(num_proc);
        std::iota(id_vec.begin(), id_vec.end(), 0);
        assert( std::is_permutation(output.begin(), output.end(), id_vec.begin(), id_vec.end()) );

        return output;

    } else {
        throw std::runtime_error("ILP partitioner did not find a solution :(");
    }
}