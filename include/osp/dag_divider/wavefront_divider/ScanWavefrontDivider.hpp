/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/
#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include "AbstractWavefrontDivider.hpp"
#include "SequenceSplitter.hpp"
#include "SequenceGenerator.hpp"

namespace osp {

/**
 * @class ScanWavefrontDivider
 * @brief Divides a DAG by scanning all wavefronts and applying a splitting algorithm.
 */
template<typename Graph_t>
class ScanWavefrontDivider : public AbstractWavefrontDivider<Graph_t> {
public:
    constexpr static bool enable_debug_print = true;

    ScanWavefrontDivider() = default;

    std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const Graph_t &dag) override {
        this->dag_ptr_ = &dag;
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG] Starting scan-all division." << std::endl;
        }
        std::vector<std::vector<vertex_idx_t<Graph_t>>> level_sets = this->compute_wavefronts(dag);
        if (level_sets.empty()) {
            return {};
        }

        SequenceGenerator<Graph_t> generator(dag, level_sets);
        std::vector<double> sequence = generator.generate(sequence_metric_);
        
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG]   Metric: " << static_cast<int>(sequence_metric_) 
                      << ", Algorithm: " << static_cast<int>(split_algorithm_) << std::endl;
            std::cout << "[DEBUG]   Generated sequence: ";
            for(const auto& val : sequence) std::cout << val << " ";
            std::cout << std::endl;
        }

        std::unique_ptr<SequenceSplitter> splitter;
        switch(split_algorithm_) {
            case SplitAlgorithm::VARIANCE:
                splitter = std::make_unique<VarianceSplitter>(var_mult_, var_threshold_);
                break;
            case SplitAlgorithm::THRESHOLD_SCAN:
                splitter = std::make_unique<ThresholdScanSplitter>(diff_threshold_, absolute_threshold_);
                break;
            case SplitAlgorithm::LARGEST_STEP:
            default:
                splitter = std::make_unique<LargestStepSplitter>(diff_threshold_, min_subseq_len_);
                break;
        }
        
        std::vector<size_t> cut_levels = splitter->split(sequence);
        std::sort(cut_levels.begin(), cut_levels.end());
        cut_levels.erase(std::unique(cut_levels.begin(), cut_levels.end()), cut_levels.end());
        
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG]   Final cut levels: ";
            for(const auto& level : cut_levels) std::cout << level << " ";
            std::cout << std::endl;
        }
        
        return create_vertex_maps_from_cuts(cut_levels, level_sets);
    }

    void set_metric(SequenceMetric metric) { sequence_metric_ = metric; }
    void set_algorithm(SplitAlgorithm algorithm) { split_algorithm_ = algorithm; }
    void set_variance_params(double mult, double threshold) {
        var_mult_ = mult;
        var_threshold_ = threshold;
    }
    void set_largest_step_params(double threshold, size_t min_len) {
        diff_threshold_ = threshold;
        min_subseq_len_ = min_len;
    }
    void set_threshold_scan_params(double diff_threshold, double abs_threshold) {
        diff_threshold_ = diff_threshold;
        absolute_threshold_ = abs_threshold;
    }

private:
    using VertexType = vertex_idx_t<Graph_t>;

    SequenceMetric sequence_metric_ = SequenceMetric::COMPONENT_COUNT;
    SplitAlgorithm split_algorithm_ = SplitAlgorithm::LARGEST_STEP;
    
    double var_mult_ = 0.5;
    double var_threshold_ = 1.0;
    double diff_threshold_ = 3.0;
    size_t min_subseq_len_ = 4;
    double absolute_threshold_ = 10.0;

    std::vector<std::vector<std::vector<VertexType>>> create_vertex_maps_from_cuts(
        const std::vector<size_t>& cut_levels,
        const std::vector<std::vector<VertexType>>& level_sets) const {
        
        if (cut_levels.empty()) {
            // If there are no cuts, return a single section with all components.
            return { this->get_components_for_range(0, level_sets.size(), level_sets) };
        }

        std::vector<std::vector<std::vector<VertexType>>> vertex_maps;
        size_t start_level = 0;

        for (const auto& cut_level : cut_levels) {
            if (start_level < cut_level) { // Avoid creating empty sections
                vertex_maps.push_back(this->get_components_for_range(start_level, cut_level, level_sets));
            }
            start_level = cut_level;
        }
        // Add the final section from the last cut to the end of the levels
        if (start_level < level_sets.size()) {
            vertex_maps.push_back(this->get_components_for_range(start_level, level_sets.size(), level_sets));
        }

        return vertex_maps;
    }
};

} // namespace osp
