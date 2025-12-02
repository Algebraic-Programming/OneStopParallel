/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#ifdef EIGEN_FOUND

#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"

namespace osp {

template<typename eigen_idx_type>
class Sptrsv {
    using uVertType = typename SparseMatrixImp<eigen_idx_type>::vertex_idx;

  private:
    const BspInstance<SparseMatrixImp<eigen_idx_type>> *instance;

  public:
    std::vector<double> val;
    std::vector<double> csc_val;

    std::vector<uVertType> col_idx;
    std::vector<uVertType> row_ptr;

    std::vector<uVertType> row_idx;
    std::vector<uVertType> col_ptr;

    std::vector<std::vector<unsigned>> step_proc_ptr;
    std::vector<std::vector<unsigned>> step_proc_num;

    double *x;
    const double *b;

    unsigned num_supersteps;

    std::vector<std::vector<std::vector<eigen_idx_type>>> vector_step_processor_vertices;
    std::vector<std::vector<std::vector<eigen_idx_type>>> vector_step_processor_vertices_u;
    std::vector<int> ready;

    std::vector<std::vector<std::vector<eigen_idx_type>>> bounds_array_l;
    std::vector<std::vector<std::vector<eigen_idx_type>>> bounds_array_u;

    Sptrsv() = default;

    Sptrsv(BspInstance<SparseMatrixImp<eigen_idx_type>> &inst) : instance(&inst) {};

    void setup_csr_no_permutation(const BspSchedule<SparseMatrixImp<eigen_idx_type>> &schedule) {
        vector_step_processor_vertices = std::vector<std::vector<std::vector<eigen_idx_type>>>(
            schedule.numberOfSupersteps(),
            std::vector<std::vector<eigen_idx_type>>(schedule.getInstance().numberOfProcessors()));

        vector_step_processor_vertices_u = std::vector<std::vector<std::vector<eigen_idx_type>>>(
            schedule.numberOfSupersteps(),
            std::vector<std::vector<eigen_idx_type>>(schedule.getInstance().numberOfProcessors()));

        bounds_array_l = std::vector<std::vector<std::vector<eigen_idx_type>>>(
            schedule.numberOfSupersteps(),
            std::vector<std::vector<eigen_idx_type>>(schedule.getInstance().numberOfProcessors()));
        bounds_array_u = std::vector<std::vector<std::vector<eigen_idx_type>>>(
            schedule.numberOfSupersteps(),
            std::vector<std::vector<eigen_idx_type>>(schedule.getInstance().numberOfProcessors()));

        num_supersteps = schedule.numberOfSupersteps();
        size_t number_of_vertices = instance->getComputationalDag().num_vertices();

#pragma omp parallel num_threads(2)
        {
            int id = omp_get_thread_num();
            switch (id) {
            case 0: {
                for (size_t node = 0; node < number_of_vertices; ++node) {
                    vector_step_processor_vertices[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].push_back(static_cast<eigen_idx_type>(node));
                }

                for (unsigned int step = 0; step < schedule.numberOfSupersteps(); ++step) {
                    for (unsigned int proc = 0; proc < instance->numberOfProcessors(); ++proc) {
                        if (!vector_step_processor_vertices[step][proc].empty()) {
                            eigen_idx_type start = vector_step_processor_vertices[step][proc][0];
                            eigen_idx_type prev = vector_step_processor_vertices[step][proc][0];

                            for (size_t i = 1; i < vector_step_processor_vertices[step][proc].size(); ++i) {
                                if (vector_step_processor_vertices[step][proc][i] != prev + 1) {
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
            case 1: {
                size_t node = number_of_vertices;
                do {
                    node--;
                    vector_step_processor_vertices_u[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].push_back(static_cast<eigen_idx_type>(node));
                } while (node > 0);

                for (unsigned int step = 0; step < schedule.numberOfSupersteps(); ++step) {
                    for (unsigned int proc = 0; proc < instance->numberOfProcessors(); ++proc) {
                        if (!vector_step_processor_vertices_u[step][proc].empty()) {
                            eigen_idx_type start_u = static_cast<eigen_idx_type>(vector_step_processor_vertices_u[step][proc][0]);
                            eigen_idx_type prev_u = static_cast<eigen_idx_type>(vector_step_processor_vertices_u[step][proc][0]);

                            for (size_t i = 1; i < vector_step_processor_vertices_u[step][proc].size(); ++i) {
                                if (static_cast<eigen_idx_type>(vector_step_processor_vertices_u[step][proc][i]) != prev_u - 1) {
                                    bounds_array_u[step][proc].push_back(start_u);
                                    bounds_array_u[step][proc].push_back(prev_u);
                                    start_u = static_cast<eigen_idx_type>(vector_step_processor_vertices_u[step][proc][i]);
                                }
                                prev_u = static_cast<eigen_idx_type>(vector_step_processor_vertices_u[step][proc][i]);
                            }

                            bounds_array_u[step][proc].push_back(start_u);
                            bounds_array_u[step][proc].push_back(prev_u);
                        }
                    }
                }

                break;
            }
            default: {
                std::cout << "Unexpected Behaviour" << std::endl;
            }
            }
        }
    }

    void setup_csr_with_permutation(const BspSchedule<SparseMatrixImp<eigen_idx_type>> &schedule, std::vector<size_t> &perm) {
        std::vector<size_t> perm_inv(perm.size());
        for (size_t i = 0; i < perm.size(); i++) {
            perm_inv[perm[i]] = i;
        }

        num_supersteps = schedule.numberOfSupersteps();

        val.clear();
        val.reserve(static_cast<size_t>(instance->getComputationalDag().getCSR()->nonZeros()));

        col_idx.clear();
        col_idx.reserve(static_cast<size_t>(instance->getComputationalDag().getCSR()->nonZeros()));

        row_ptr.clear();
        row_ptr.reserve(instance->numberOfVertices() + 1);

        step_proc_ptr =
            std::vector<std::vector<unsigned>>(num_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

        step_proc_num = schedule.numAssignedNodesPerSuperstepProcessor();

        unsigned current_step = 0;
        unsigned current_processor = 0;

        step_proc_ptr[current_step][current_processor] = 0;

        for (const uVertType &node : perm_inv) {

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

                step_proc_ptr[current_step][current_processor] = static_cast<unsigned>(row_ptr.size());
            }

            row_ptr.push_back(col_idx.size());

            std::set<uVertType> parents;

            for (uVertType par : instance->getComputationalDag().parents(node)) {
                parents.insert(perm[par]);
            }

            for (const uVertType &par : parents) {
                col_idx.push_back(par);
                unsigned found = 0;

                const auto *outer = instance->getComputationalDag().getCSR()->outerIndexPtr();
                for (uVertType par_ind = static_cast<uVertType>(outer[node]); par_ind < static_cast<uVertType>(outer[node + 1] - 1); ++par_ind) {

                    if (static_cast<size_t>(instance->getComputationalDag().getCSR()->innerIndexPtr()[par_ind]) == perm_inv[par]) {
                        val.push_back(instance->getComputationalDag().getCSR()->valuePtr()[par_ind]);
                        found++;
                    }
                }
                assert(found == 1);
            }

            col_idx.push_back(perm[node]);
            val.push_back(instance->getComputationalDag().getCSR()->valuePtr()[instance->getComputationalDag().getCSR()->outerIndexPtr()[node + 1] - 1]);
        }

        row_ptr.push_back(col_idx.size());
    }

    void lsolve_serial() {
        eigen_idx_type number_of_vertices = static_cast<eigen_idx_type>(instance->numberOfVertices());
        for (eigen_idx_type i = 0; i < number_of_vertices; ++i) {
            x[i] = b[i];
            for (eigen_idx_type j = (*(instance->getComputationalDag().getCSR())).outerIndexPtr()[i]; j < (*(instance->getComputationalDag().getCSR())).outerIndexPtr()[i + 1] - 1; ++j) {
                x[i] -= (*(instance->getComputationalDag().getCSR())).valuePtr()[j] * x[(*(instance->getComputationalDag().getCSR())).innerIndexPtr()[j]];
            }
            x[i] /= (*(instance->getComputationalDag().getCSR())).valuePtr()[(*(instance->getComputationalDag().getCSR())).outerIndexPtr()[i + 1] - 1];
        }
    }

    void usolve_serial() {
        eigen_idx_type number_of_vertices = static_cast<eigen_idx_type>(instance->numberOfVertices());

        eigen_idx_type i = number_of_vertices;
        do {
            i--;
            x[i] = b[i];
            for (eigen_idx_type j = (*(instance->getComputationalDag().getCSC())).outerIndexPtr()[i] + 1; j < (*(instance->getComputationalDag().getCSC())).outerIndexPtr()[i + 1]; ++j) {
                x[i] -= (*(instance->getComputationalDag().getCSC())).valuePtr()[j] * x[(*(instance->getComputationalDag().getCSC())).innerIndexPtr()[j]];
            }
            x[i] /= (*(instance->getComputationalDag().getCSC())).valuePtr()[(*(instance->getComputationalDag().getCSC())).outerIndexPtr()[i]];
        } while (i != 0);
    }

    void lsolve_no_permutation_in_place() {
#pragma omp parallel num_threads(instance->numberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < num_supersteps; ++step) {
                const size_t bounds_str_size = bounds_array_l[step][proc].size();

                for (size_t index = 0; index < bounds_str_size; index += 2) {
                    eigen_idx_type lower_b = bounds_array_l[step][proc][index];
                    const eigen_idx_type upper_b = bounds_array_l[step][proc][index + 1];

                    for (eigen_idx_type node = lower_b; node <= upper_b; ++node) {
                        for (eigen_idx_type i = (*(instance->getComputationalDag().getCSR())).outerIndexPtr()[node]; i < (*(instance->getComputationalDag().getCSR())).outerIndexPtr()[node + 1] - 1; ++i) {
                            x[node] -= (*(instance->getComputationalDag().getCSR())).valuePtr()[i] * x[(*(instance->getComputationalDag().getCSR())).innerIndexPtr()[i]];
                        }
                        x[node] /= (*(instance->getComputationalDag().getCSR())).valuePtr()[(*(instance->getComputationalDag().getCSR())).outerIndexPtr()[node + 1] - 1];
                    }
                }
#pragma omp barrier
            }
        }
    }

    void usolve_no_permutation_in_place() {
#pragma omp parallel num_threads(instance->numberOfProcessors())
        {
            // Process each superstep starting from the last one (opposite of lsolve)
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            unsigned step = num_supersteps;
            do {
                step--;
                const size_t bounds_str_size = bounds_array_u[step][proc].size();
                for (size_t index = 0; index < bounds_str_size; index += 2) {
                    eigen_idx_type node = bounds_array_u[step][proc][index] + 1;
                    const eigen_idx_type lower_b = bounds_array_u[step][proc][index + 1];

                    do {
                        node--;
                        for (eigen_idx_type i = (*(instance->getComputationalDag().getCSC())).outerIndexPtr()[node] + 1; i < (*(instance->getComputationalDag().getCSC())).outerIndexPtr()[node + 1]; ++i) {
                            x[node] -= (*(instance->getComputationalDag().getCSC())).valuePtr()[i] * x[(*(instance->getComputationalDag().getCSC())).innerIndexPtr()[i]];
                        }
                        x[node] /= (*(instance->getComputationalDag().getCSC())).valuePtr()[(*(instance->getComputationalDag().getCSC())).outerIndexPtr()[node]];
                    } while (node != lower_b);
                }
#pragma omp barrier
            } while (step != 0);
        }
    }

    void lsolve_no_permutation() {
#pragma omp parallel num_threads(instance->numberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < num_supersteps; ++step) {
                const size_t bounds_str_size = bounds_array_l[step][proc].size();

                for (size_t index = 0; index < bounds_str_size; index += 2) {
                    eigen_idx_type lower_b = bounds_array_l[step][proc][index];
                    const eigen_idx_type upper_b = bounds_array_l[step][proc][index + 1];

                    for (eigen_idx_type node = lower_b; node <= upper_b; ++node) {
                        x[node] = b[node];
                        for (eigen_idx_type i = (*(instance->getComputationalDag().getCSR())).outerIndexPtr()[node]; i < (*(instance->getComputationalDag().getCSR())).outerIndexPtr()[node + 1] - 1; ++i) {
                            x[node] -= (*(instance->getComputationalDag().getCSR())).valuePtr()[i] * x[(*(instance->getComputationalDag().getCSR())).innerIndexPtr()[i]];
                        }
                        x[node] /= (*(instance->getComputationalDag().getCSR())).valuePtr()[(*(instance->getComputationalDag().getCSR())).outerIndexPtr()[node + 1] - 1];
                    }
                }
#pragma omp barrier
            }
        }
    }

    void usolve_no_permutation() {
#pragma omp parallel num_threads(instance->numberOfProcessors())
        {
            // Process each superstep starting from the last one (opposite of lsolve)
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            unsigned step = num_supersteps;
            do {
                step--;
                const size_t bounds_str_size = bounds_array_u[step][proc].size();
                for (size_t index = 0; index < bounds_str_size; index += 2) {
                    eigen_idx_type node = bounds_array_u[step][proc][index] + 1;
                    const eigen_idx_type lower_b = bounds_array_u[step][proc][index + 1];

                    do {
                        node--;
                        x[node] = b[node];
                        for (eigen_idx_type i = (*(instance->getComputationalDag().getCSC())).outerIndexPtr()[node] + 1; i < (*(instance->getComputationalDag().getCSC())).outerIndexPtr()[node + 1]; ++i) {
                            x[node] -= (*(instance->getComputationalDag().getCSC())).valuePtr()[i] * x[(*(instance->getComputationalDag().getCSC())).innerIndexPtr()[i]];
                        }
                        x[node] /= (*(instance->getComputationalDag().getCSC())).valuePtr()[(*(instance->getComputationalDag().getCSC())).outerIndexPtr()[node]];
                    } while (node != lower_b);
                }
#pragma omp barrier
            } while (step != 0);
        }
    }

    void lsolve_serial_in_place() {
        eigen_idx_type number_of_vertices = static_cast<eigen_idx_type>(instance->numberOfVertices());
        for (eigen_idx_type i = 0; i < number_of_vertices; ++i) {
            for (eigen_idx_type j = (*(instance->getComputationalDag().getCSR())).outerIndexPtr()[i]; j < (*(instance->getComputationalDag().getCSR())).outerIndexPtr()[i + 1] - 1; ++j) {
                x[i] -= (*(instance->getComputationalDag().getCSR())).valuePtr()[j] * x[(*(instance->getComputationalDag().getCSR())).innerIndexPtr()[j]];
            }
            x[i] /= (*(instance->getComputationalDag().getCSR())).valuePtr()[(*(instance->getComputationalDag().getCSR())).outerIndexPtr()[i + 1] - 1];
        }
    }

    void usolve_serial_in_place() {
        eigen_idx_type number_of_vertices = static_cast<eigen_idx_type>(instance->numberOfVertices());
        eigen_idx_type i = number_of_vertices;
        do {
            i--;
            for (eigen_idx_type j = (*(instance->getComputationalDag().getCSC())).outerIndexPtr()[i] + 1; j < (*(instance->getComputationalDag().getCSC())).outerIndexPtr()[i + 1]; ++j) {
                x[i] -= (*(instance->getComputationalDag().getCSC())).valuePtr()[j] * x[(*(instance->getComputationalDag().getCSC())).innerIndexPtr()[j]];
            }
            x[i] /= (*(instance->getComputationalDag().getCSC())).valuePtr()[(*(instance->getComputationalDag().getCSC())).outerIndexPtr()[i]];
        } while (i != 0);
    }

    void lsolve_with_permutation_in_place() {
#pragma omp parallel num_threads(instance->numberOfProcessors())
        {
            for (unsigned step = 0; step < num_supersteps; step++) {

                const size_t proc = static_cast<size_t>(omp_get_thread_num());
                const uVertType upper_limit = step_proc_ptr[step][proc] + step_proc_num[step][proc];
                for (uVertType _row_idx = step_proc_ptr[step][proc]; _row_idx < upper_limit; _row_idx++) {

                    for (uVertType i = row_ptr[_row_idx]; i < row_ptr[_row_idx + 1] - 1; i++) {
                        x[_row_idx] -= val[i] * x[col_idx[i]];
                    }

                    x[_row_idx] /= val[row_ptr[_row_idx + 1] - 1];
                }

#pragma omp barrier
            }
        }
    }

    void lsolve_with_permutation() {
#pragma omp parallel num_threads(instance->numberOfProcessors())
        {
            for (unsigned step = 0; step < num_supersteps; step++) {

                const size_t proc = static_cast<size_t>(omp_get_thread_num());
                const uVertType upper_limit = step_proc_ptr[step][proc] + step_proc_num[step][proc];
                for (uVertType _row_idx = step_proc_ptr[step][proc]; _row_idx < upper_limit; _row_idx++) {
                    x[_row_idx] = b[_row_idx];
                    for (uVertType i = row_ptr[_row_idx]; i < row_ptr[_row_idx + 1] - 1; i++) {
                        x[_row_idx] -= val[i] * x[col_idx[i]];
                    }

                    x[_row_idx] /= val[row_ptr[_row_idx + 1] - 1];
                }

#pragma omp barrier
            }
        }
    }

    void reset_x() {
        eigen_idx_type number_of_vertices = static_cast<eigen_idx_type>(instance->numberOfVertices());
        for (eigen_idx_type i = 0; i < number_of_vertices; i++) {
            x[i] = 1.0;
        }
    }

    void permute_x_vector(const std::vector<size_t> &perm) {
        std::vector<double> vec_perm(perm.size());
        for (size_t i = 0; i < perm.size(); i++) {
            vec_perm[i] = x[perm[i]];
        }
        for (size_t i = 0; i < perm.size(); i++) {
            x[i] = vec_perm[i];
        }
    }

    void permute_x_vector_inverse(const std::vector<size_t> &perm) {
        std::vector<double> vec_unperm(perm.size());
        for (size_t i = 0; i < perm.size(); i++) {
            vec_unperm[perm[i]] = x[i];
        }
        for (size_t i = 0; i < perm.size(); i++) {
            x[i] = vec_unperm[i];
        }
    }

    std::size_t get_number_of_vertices() {
        return instance->numberOfVertices();
    }

    virtual ~Sptrsv() = default;
};

} // namespace osp

#endif