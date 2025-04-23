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

#pragma once

#ifdef EIGEN_FOUND

#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>

#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"
#include "model/SmSchedule.hpp"
#include "model/SetSchedule.hpp"
#include "scheduler/SchedulePermutations/ScheduleNodePermuter.hpp"

using perm_t = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t>;

class SmSptrsvCSR {

  private:
    const SmInstance *instance;

  public:
    std::vector<double> val;
    std::vector<double> csc_val;

    std::vector<unsigned> col_idx;
    std::vector<unsigned> row_ptr;
    
    std::vector<unsigned> row_idx;
    std::vector<unsigned> col_ptr;

    std::vector<unsigned> num_row_entries;

    std::vector<unsigned> step_ptr;
    std::vector<std::vector<unsigned>> step_proc_ptr;
    std::vector<std::vector<unsigned>> step_proc_num;

    double * x;
    const double *b;

    unsigned num_supersteps;

    std::vector<std::vector<std::vector<int>>> vector_step_processor_vertices;
    std::vector<std::vector<std::vector<int>>> vector_step_processor_vertices_u;
    std::vector<int> ready;

    std::vector<std::vector<std::vector<unsigned int>>>  bounds_array_l;
    std::vector<std::vector<std::vector<unsigned int>>>  bounds_array_u;

    SmSptrsvCSR() = default;

    SmSptrsvCSR(SmInstance &inst) : instance(&inst) {};

    void setup_csr_no_permutation(const SmSchedule &schedule);
    void setup_csr_with_permutation(const SmSchedule &schedule, std::vector<size_t> &perm);

    void lsolve_serial();
    void usolve_serial();

    void lsolve_no_permutation_in_place();
    void usolve_no_permutation_in_place();
    void lsolve_no_permutation();
    void usolve_no_permutation();

    void lsolve_in_place();
    void lsolve_serial_in_place();
    void usolve_in_place();
    void usolve_serial_in_place();

    void lsolve_with_permutation_in_place();

    void reset_x();

    void permute_vector(std::vector<double> &vec, const std::vector<size_t> &perm);

    unsigned int get_number_of_vertices(){
      return instance->getMatrix().numberOfVertices();
    }

    virtual ~SmSptrsvCSR() = default;
};

#endif