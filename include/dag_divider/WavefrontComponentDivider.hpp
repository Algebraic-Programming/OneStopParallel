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

#include "WavefrontDivider.hpp"
#include "file_interactions/ComputationalDagWriter.hpp"
#include "model/dag_algorithms/subgraph_algorithms.hpp"

/**
 * @class WavefrontComponentDivider
 * @brief Divides the wavefronts of a computational DAG into consecutive groups or sections.
 * The sections are created with the aim of containing a high number of connected components.
 * The class also provides functionality to detect groups of isomorphic components within the sections.
 *
 *
 */
class WavefrontComponentDivider : public IWavefrontDivider {

  public:
    enum class SplitMethod { MIN_DIFF, VARIANCE };

  private:
    double var_mult = 0.5;
    double var_threshold = 1.0;

    size_t diff_threshold = 3;

    int min_subseq_len = 4;

    SplitMethod split_method = SplitMethod::MIN_DIFF;

    struct wavefron_statistics {

        unsigned number_of_connected_components;
        std::vector<unsigned> connected_components_weights;
        std::vector<unsigned> connected_components_memories;
        std::vector<std::vector<unsigned>> connected_components_vertices;
    };

    const ComputationalDag *dag;

    std::vector<wavefron_statistics> forward_statistics;
    std::vector<wavefron_statistics> backward_statistics;

    void split_sequence_min_diff_fwd(const std::vector<double> &seq, std::vector<size_t> &splits, size_t offset = 0);

    void split_sequence_min_diff_bwd(const std::vector<double> &seq, std::vector<size_t> &splits, size_t offset = 0);

    bool compute_split_min_diff(const std::vector<double> &parallelism, size_t &split, bool reverse = true);

    void split_sequence_var(const std::vector<double> &seq, std::vector<size_t> &splits, size_t offset = 0);

    bool compute_split_var(const std::vector<double> &parallelism, size_t &split);

    void compute_variance(const std::vector<double> &data, double &mean, double &variance);

    void print_wavefront_statistics(const std::vector<wavefron_statistics> &statistics, bool reverse = false);

    void compute_forward_statistics(const std::vector<std::vector<unsigned>> &level_sets, const ComputationalDag &dag);

    void compute_backward_statistics(const std::vector<std::vector<unsigned>> &level_sets, const ComputationalDag &dag);

    std::vector<size_t> combine_split_sequences(std::vector<size_t> &fwd_splits, std::vector<size_t> &bwd_splits);

    std::vector<size_t> compute_cut_levels_fwd_bwd_var(std::vector<std::vector<unsigned>> &level_sets);

    std::vector<size_t> compute_cut_levels_fwd_bwd_min_diff(std::vector<std::vector<unsigned>> &level_sets);

  public:
    WavefrontComponentDivider() = default;

    std::vector<std::vector<std::vector<unsigned>>> divide(const ComputationalDag &dag_) override;

    inline void set_split_method(SplitMethod method) { split_method = method; }
};