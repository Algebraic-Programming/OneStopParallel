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

@author Christos Matzoros, Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
*/

#include "simulation/SmSptrsvCSR.hpp"
#include <thread>

void SmSptrsvCSR::setup_csr_no_permutation(const SmSchedule &schedule) {
    vector_step_processor_vertices = std::vector<std::vector<std::vector<int>>>(
        schedule.numberOfSupersteps(),
        std::vector<std::vector<int>>(schedule.getInstance().numberOfProcessors()));

    vector_step_processor_vertices_u = std::vector<std::vector<std::vector<int>>>(
        schedule.numberOfSupersteps(),
        std::vector<std::vector<int>>(schedule.getInstance().numberOfProcessors()));

    bounds_array_l = std::vector<std::vector<std::vector<unsigned int>>>(
        schedule.numberOfSupersteps(),
        std::vector<std::vector<unsigned int>>(schedule.getInstance().numberOfProcessors()));
    bounds_array_u = std::vector<std::vector<std::vector<unsigned int>>>(
        schedule.numberOfSupersteps(),
        std::vector<std::vector<unsigned int>>(schedule.getInstance().numberOfProcessors()));

    num_supersteps = schedule.numberOfSupersteps();
    unsigned int number_of_edges = instance->getMatrix().numberOfEdges();
    unsigned int number_of_vertices = instance->getMatrix().numberOfVertices();

#pragma omp parallel num_threads(2)
{
    int id = omp_get_thread_num();
    switch(id) {
        case 0:
        {
            for (int node=0; node < number_of_vertices; ++node){
                vector_step_processor_vertices[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].push_back(node);
            }  

            for (unsigned int step=0; step<schedule.numberOfSupersteps(); ++step){
                for (unsigned int proc=0; proc<instance->numberOfProcessors(); ++proc){
                    if (!vector_step_processor_vertices[step][proc].empty()){
                        int start = vector_step_processor_vertices[step][proc][0];
                        int prev = vector_step_processor_vertices[step][proc][0];

                        for (size_t i=1; i<vector_step_processor_vertices[step][proc].size(); ++i){
                            if(vector_step_processor_vertices[step][proc][i] != prev + 1){
                                bounds_array_l[step][proc].push_back(start);
                                bounds_array_l[step][proc].push_back(prev);
                                start = vector_step_processor_vertices[step][proc][i];
                            }
                            prev = vector_step_processor_vertices[step][proc][i];
                        }

                        bounds_array_l[step][proc].push_back(start);
                        bounds_array_l[step][proc].push_back(prev);
                    }
                }
            }

            break;
        }
        case 1:
        {
            for (int node=number_of_vertices-1; node>=0; node--){
                vector_step_processor_vertices_u[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].push_back(node);
            } 
            
            for (unsigned int step=0; step<schedule.numberOfSupersteps(); ++step){
                for (unsigned int proc=0; proc<instance->numberOfProcessors(); ++proc){
                    if (!vector_step_processor_vertices_u[step][proc].empty()){
                        int start_u = vector_step_processor_vertices_u[step][proc][0];
                        int prev_u = vector_step_processor_vertices_u[step][proc][0];

                        for (size_t i=1; i<vector_step_processor_vertices_u[step][proc].size(); ++i){
                            if(vector_step_processor_vertices_u[step][proc][i] != prev_u - 1){
                                bounds_array_u[step][proc].push_back(start_u);
                                bounds_array_u[step][proc].push_back(prev_u);
                                start_u = vector_step_processor_vertices_u[step][proc][i];
                            }
                            prev_u = vector_step_processor_vertices_u[step][proc][i];
                        }

                        bounds_array_u[step][proc].push_back(start_u);
                        bounds_array_u[step][proc].push_back(prev_u);
                    }
                }
            }

            break;
        }
        default:{
            std::cout << "Unexpected Behaviour" << std::endl;
        }

    }

}

}


void SmSptrsvCSR::lsolve_serial_in_place(){
    for (unsigned i = 0; i < instance->numberOfVertices(); ++i){
        for (unsigned j = (*(instance->getMatrix().getCSR())).outerIndexPtr()[i]; j < (*(instance->getMatrix().getCSR())).outerIndexPtr()[i + 1] - 1; ++j){
            x[i] -= (*(instance->getMatrix().getCSR())).valuePtr()[j] * x[(*(instance->getMatrix().getCSR())).innerIndexPtr()[j]];
        }
        x[i] /= (*(instance->getMatrix().getCSR())).valuePtr()[(*(instance->getMatrix().getCSR())).outerIndexPtr()[i + 1] - 1];
    }
    
}

void SmSptrsvCSR::lsolve_serial(){
    for (unsigned i = 0; i < instance->numberOfVertices(); ++i){
        x[i] = b[i];
        for (unsigned j = (*(instance->getMatrix().getCSR())).outerIndexPtr()[i]; j < (*(instance->getMatrix().getCSR())).outerIndexPtr()[i + 1] - 1; ++j){
            x[i] -= (*(instance->getMatrix().getCSR())).valuePtr()[j] * x[(*(instance->getMatrix().getCSR())).innerIndexPtr()[j]];
        }
        x[i] /= (*(instance->getMatrix().getCSR())).valuePtr()[(*(instance->getMatrix().getCSR())).outerIndexPtr()[i + 1] - 1];
    }
}

void SmSptrsvCSR::usolve_serial_in_place(){
    // Start from the last row and move upwards
    for (unsigned i = instance->numberOfVertices() - 1; i < instance->numberOfVertices(); --i){
        for (unsigned j = (*(instance->getMatrix().getCSC())).outerIndexPtr()[i] + 1; j < (*(instance->getMatrix().getCSC())).outerIndexPtr()[i + 1]; ++j){
            x[i] -= (*(instance->getMatrix().getCSC())).valuePtr()[j] * x[(*(instance->getMatrix().getCSC())).innerIndexPtr()[j]];
        }
        x[i] /= (*(instance->getMatrix().getCSC())).valuePtr()[(*(instance->getMatrix().getCSC())).outerIndexPtr()[i]];
    }
}

void SmSptrsvCSR::usolve_serial(){
    // Start from the last row and move upwards
    for (unsigned i = instance->numberOfVertices() - 1; i < instance->numberOfVertices(); --i){
        x[i] = b[i];
        for (unsigned j = (*(instance->getMatrix().getCSC())).outerIndexPtr()[i] + 1; j < (*(instance->getMatrix().getCSC())).outerIndexPtr()[i + 1]; ++j){
            x[i] -= (*(instance->getMatrix().getCSC())).valuePtr()[j] * x[(*(instance->getMatrix().getCSC())).innerIndexPtr()[j]];
        }
        x[i] /= (*(instance->getMatrix().getCSC())).valuePtr()[(*(instance->getMatrix().getCSC())).outerIndexPtr()[i]];
    }
}

void SmSptrsvCSR::lsolve_no_permutation_in_place(){
#pragma omp parallel num_threads(instance->numberOfProcessors())
    {
        const unsigned int proc = omp_get_thread_num();
        for (unsigned step = 0; step < num_supersteps; ++step){
            const unsigned int bounds_str_size = bounds_array_l[step][proc].size();
            
            for (unsigned int index = 0; index < bounds_str_size; index+=2){
                unsigned int lower_b = bounds_array_l[step][proc][index];
                const unsigned int upper_b = bounds_array_l[step][proc][index+1];
                
                for (unsigned int node = lower_b; node<=upper_b; ++node){
                    for (unsigned int i = (*(instance->getMatrix().getCSR())).outerIndexPtr()[node]; i < (*(instance->getMatrix().getCSR())).outerIndexPtr()[node + 1] - 1; ++i){
                        x[node] -= (*(instance->getMatrix().getCSR())).valuePtr()[i] * x[(*(instance->getMatrix().getCSR())).innerIndexPtr()[i]];
                    }
                    x[node] /= (*(instance->getMatrix().getCSR())).valuePtr()[(*(instance->getMatrix().getCSR())).outerIndexPtr()[node + 1] - 1];
                }
            }
#pragma omp barrier
        }        
    }
}

void SmSptrsvCSR::lsolve_no_permutation(){
#pragma omp parallel num_threads(instance->numberOfProcessors())
    {
        const unsigned int proc = omp_get_thread_num();
        for (unsigned step = 0; step < num_supersteps; ++step){
            const unsigned int bounds_str_size = bounds_array_l[step][proc].size();
            
            for (unsigned int index = 0; index < bounds_str_size; index+=2){
                unsigned int lower_b = bounds_array_l[step][proc][index];
                const unsigned int upper_b = bounds_array_l[step][proc][index+1];
                
                for (unsigned int node = lower_b; node<=upper_b; ++node){
                    x[node] = b[node];
                    for (unsigned int i = (*(instance->getMatrix().getCSR())).outerIndexPtr()[node]; i < (*(instance->getMatrix().getCSR())).outerIndexPtr()[node + 1] - 1; ++i){
                        x[node] -= (*(instance->getMatrix().getCSR())).valuePtr()[i] * x[(*(instance->getMatrix().getCSR())).innerIndexPtr()[i]];
                    }
                    x[node] /= (*(instance->getMatrix().getCSR())).valuePtr()[(*(instance->getMatrix().getCSR())).outerIndexPtr()[node + 1] - 1];
                }
            }
#pragma omp barrier
        }        
    }
}

void SmSptrsvCSR::usolve_no_permutation_in_place(){
#pragma omp parallel num_threads(instance->numberOfProcessors())
    {
        // Process each superstep starting from the last one (opposite of lsolve)
        const unsigned int proc = omp_get_thread_num();
        unsigned step = num_supersteps;
        do {
            step--;
            const unsigned int bounds_str_size = bounds_array_u[step][proc].size();
            for (unsigned int index = 0; index < bounds_str_size; index+=2){
                unsigned int node = bounds_array_u[step][proc][index] + 1;
                const unsigned int lower_b = bounds_array_u[step][proc][index+1];

                do {
                    node--;
                    for (unsigned int i=(*(instance->getMatrix().getCSC())).outerIndexPtr()[node] + 1; i < (*(instance->getMatrix().getCSC())).outerIndexPtr()[node + 1]; ++i){
                        x[node] -= (*(instance->getMatrix().getCSC())).valuePtr()[i] * x[(*(instance->getMatrix().getCSC())).innerIndexPtr()[i]];
                    }
                    x[node] /= (*(instance->getMatrix().getCSC())).valuePtr()[(*(instance->getMatrix().getCSC())).outerIndexPtr()[node]];
                } while (node != lower_b);
            }
#pragma omp barrier
        } while (step!=0);    
    }
}

void SmSptrsvCSR::usolve_no_permutation(){
#pragma omp parallel num_threads(instance->numberOfProcessors())
    {
        // Process each superstep starting from the last one (opposite of lsolve)
        const unsigned int proc = omp_get_thread_num();
        unsigned step = num_supersteps;
        do {
            step--;
            const unsigned int bounds_str_size = bounds_array_u[step][proc].size();
            for (unsigned int index = 0; index < bounds_str_size; index+=2){
                unsigned int node = bounds_array_u[step][proc][index] + 1;
                const unsigned int lower_b = bounds_array_u[step][proc][index+1];

                do {
                    node--;
                    x[node] = b[node];
                    for (unsigned int i=(*(instance->getMatrix().getCSC())).outerIndexPtr()[node] + 1; i < (*(instance->getMatrix().getCSC())).outerIndexPtr()[node + 1]; ++i){
                        x[node] -= (*(instance->getMatrix().getCSC())).valuePtr()[i] * x[(*(instance->getMatrix().getCSC())).innerIndexPtr()[i]];
                    }
                    x[node] /= (*(instance->getMatrix().getCSC())).valuePtr()[(*(instance->getMatrix().getCSC())).outerIndexPtr()[node]];
                } while (node != lower_b);
            }
#pragma omp barrier
        } while (step!=0);    
    }
}


void SmSptrsvCSR::reset_x() {
    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
        x[i] = 1.0;
    }
}

void SmSptrsvCSR::setup_csr_with_permutation(const SmSchedule &schedule, std::vector<size_t> &perm) {
    std::vector<size_t> perm_inv(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        perm_inv[perm[i]] = i;
    }

    num_supersteps = schedule.numberOfSupersteps();

    val.clear();
    // val.reserve(instance->getComputationalDag().numberOfEdges() + instance->getComputationalDag().numberOfVertices());
    val.reserve(instance->getMatrix().getCSR()->nonZeros());

    col_idx.clear();
    // col_idx.reserve(instance->getComputationalDag().numberOfEdges() +
    //                 instance->getComputationalDag().numberOfVertices());
    col_idx.reserve(instance->getMatrix().getCSR()->nonZeros());

    row_ptr.clear();
    row_ptr.reserve(instance->numberOfVertices() + 1);

    step_ptr.clear();
    step_ptr.reserve(instance->numberOfVertices());

    step_proc_ptr =
        std::vector<std::vector<unsigned>>(num_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

    step_proc_num = schedule.num_assigned_nodes_per_superstep_processor();

    unsigned current_step = 0;
    unsigned current_processor = 0;

    step_proc_ptr[current_step][current_processor] = 0;

    for (const auto &node : perm_inv) {

        if (schedule.assignedProcessor(node) != current_processor || schedule.assignedSuperstep(node) != current_step) {
            
            while (schedule.assignedProcessor(node) != current_processor ||
                   schedule.assignedSuperstep(node) != current_step) {

                if (current_processor < instance->numberOfProcessors() - 1) {
                    current_processor++;
                } else {
                    current_processor = 0;
                    current_step++;
                }
            }

            step_proc_ptr[current_step][current_processor] = row_ptr.size();

        }

        row_ptr.push_back(col_idx.size());

        std::set<unsigned> parents;

        // for (const auto &edge : instance->getComputationalDag().in_edges(node)) {
        //     parents.insert(perm[instance->getComputationalDag().source(edge)]);
        // }
        for (unsigned par : instance->getMatrix().parents(node)) {
            parents.insert(perm[par]);
        }

        // for (const auto &p : parents) {
        //     col_idx.push_back(p);
        //     auto pair = boost::edge(perm_inv[p], node, instance->getComputationalDag().getGraph());
        //     assert(pair.second);
        //     val.push_back(instance->getComputationalDag().edge_mtx_entry(pair.first));
        // }
        for (const auto &par : parents) {
            col_idx.push_back(par);
            unsigned found = 0;
            for (unsigned par_ind = instance->getMatrix().getCSR()->outerIndexPtr()[node]; par_ind < instance->getMatrix().getCSR()->outerIndexPtr()[node + 1] - 1; par_ind++) {
                if (instance->getMatrix().getCSR()->innerIndexPtr()[par_ind] == perm_inv[par] ) {
                    val.push_back(instance->getMatrix().getCSR()->valuePtr()[par_ind]);
                    found++;
                }
            }
            assert(found == 1);
        }

        col_idx.push_back(perm[node]);
        // val.push_back(instance->getComputationalDag().node_mtx_entry(node));
        val.push_back(instance->getMatrix().getCSR()->valuePtr()[ instance->getMatrix().getCSR()->outerIndexPtr()[node + 1] - 1 ]);
    }

    row_ptr.push_back(col_idx.size());
}

void SmSptrsvCSR::lsolve_with_permutation_in_place() {
    #pragma omp parallel num_threads(instance->numberOfProcessors())
    {
        for (unsigned step = 0; step < num_supersteps; step++) {

            const unsigned int proc = omp_get_thread_num();
            // #pragma omp for schedule(static, 1)
            //             for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            // nodes assigned to proc
            const unsigned upper_limit = step_proc_ptr[step][proc] + step_proc_num[step][proc];

            for (unsigned row_idx = step_proc_ptr[step][proc]; row_idx < upper_limit; row_idx++) {

                for (unsigned i = row_ptr[row_idx]; i < row_ptr[row_idx + 1] - 1; i++) {

                    x[row_idx] -= val[i] * x[col_idx[i]];
                }

                x[row_idx] /= val[row_ptr[row_idx + 1] - 1];
            }

#pragma omp barrier
        }
    }
}

void SmSptrsvCSR::permute_vector(std::vector<double> &vec, const std::vector<size_t> &perm) {

    std::vector<double> vec_perm(vec.size());

    for (size_t i = 0; i < perm.size(); i++) {
        vec_perm[i] = vec[perm[i]];
    }

    vec = vec_perm;
}