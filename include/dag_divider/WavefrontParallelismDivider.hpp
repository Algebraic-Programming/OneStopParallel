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
#include "scheduler/Scheduler.hpp"
#include "structures/union_find.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "WavefrontDivider.hpp"
#include "file_interactions/ComputationalDagWriter.hpp"
#include "model/dag_algorithms/subgraph_algorithms.hpp"



class WavefrontParallelismDivider : public IWavefrontDivider {

  private:

    double var_mult = 0.6;
    double var_threshold = 1000.0;

    unsigned min_number_wavefronts = 100;
    unsigned max_depth = 2;
        
    unsigned depth = 0;

    struct wavefron_statistics {

        unsigned number_of_connected_components;
        double parallelism;
        double max_weight;
        double max_acc_weight;
        double total_weight;
        double total_acc_weight;
        std::vector<unsigned> connected_components_weights;
        std::vector<unsigned> connected_components_memories;
        std::vector<std::vector<unsigned>> connected_components_vertices;
    };

    const ComputationalDag *dag;

    std::vector<wavefron_statistics> forward_statistics;

    void split_sequence(const std::vector<double> &seq, std::vector<size_t> &splits, size_t offset = 0, unsigned depth = 0);

    bool compute_split(const std::vector<double> &parallelism, size_t &split);
  
    void compute_variance(const std::vector<double> &data, double &mean, double &variance);    

    void print_wavefront_statistics(const std::vector<wavefron_statistics> &statistics, bool reverse = false);

    void compute_forward_statistics(const std::vector<std::vector<unsigned>> &level_sets, const ComputationalDag &dag);

  public:

    WavefrontParallelismDivider() = default;

    std::vector<std::vector<std::vector<unsigned>>> divide(const ComputationalDag& dag_) override;
};