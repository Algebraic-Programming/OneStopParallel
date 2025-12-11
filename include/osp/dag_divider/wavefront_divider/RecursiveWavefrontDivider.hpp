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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

#include "AbstractWavefrontDivider.hpp"
#include "SequenceGenerator.hpp"
#include "SequenceSplitter.hpp"

namespace osp {

/**
 * @class RecursiveWavefrontDivider
 * @brief Recursively divides a DAG by applying a splitting algorithm to subgraphs.
 *
 * This divider first computes the wavefronts for the entire DAG. It then uses a
 * configured splitting algorithm to find all cut points. For each resulting
 * section, it recursively repeats the process, allowing for a hierarchical
 * division of the DAG.
 */
template <typename Graph_t>
class RecursiveWavefrontDivider : public AbstractWavefrontDivider<Graph_t> {
  public:
    constexpr static bool enable_debug_print = true;

    RecursiveWavefrontDivider() {
        // Set a sensible default splitter on construction.
        use_largest_step_splitter(3.0, 4);
    }

    std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const Graph_t &dag) override {
        this->dag_ptr_ = &dag;
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG] Starting recursive-scan division." << std::endl;
        }

        auto global_level_sets = this->compute_wavefronts();
        if (global_level_sets.empty()) {
            return {};
        }

        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> all_sections;
        divide_recursive(global_level_sets.cbegin(), global_level_sets.cend(), global_level_sets, all_sections, 0);
        return all_sections;
    }

    RecursiveWavefrontDivider &set_metric(SequenceMetric metric) {
        sequence_metric_ = metric;
        return *this;
    }

    RecursiveWavefrontDivider &use_variance_splitter(double mult, double threshold, size_t min_len = 1) {
        splitter_ = std::make_unique<VarianceSplitter>(mult, threshold, min_len);
        min_subseq_len_ = min_len;
        return *this;
    }

    RecursiveWavefrontDivider &use_largest_step_splitter(double threshold, size_t min_len) {
        splitter_ = std::make_unique<LargestStepSplitter>(threshold, min_len);
        min_subseq_len_ = min_len;
        return *this;
    }

    RecursiveWavefrontDivider &use_threshold_scan_splitter(double diff_threshold, double abs_threshold, size_t min_len = 1) {
        splitter_ = std::make_unique<ThresholdScanSplitter>(diff_threshold, abs_threshold, min_len);
        min_subseq_len_ = min_len;
        return *this;
    }

    RecursiveWavefrontDivider &set_max_depth(size_t max_depth) {
        max_depth_ = max_depth;
        return *this;
    }

  private:
    using VertexType = vertex_idx_t<Graph_t>;
    using LevelSetConstIterator = typename std::vector<std::vector<VertexType>>::const_iterator;
    using DifferenceType = typename std::iterator_traits<LevelSetConstIterator>::difference_type;

    SequenceMetric sequence_metric_ = SequenceMetric::COMPONENT_COUNT;
    std::unique_ptr<SequenceSplitter> splitter_;
    size_t min_subseq_len_ = 4;
    size_t max_depth_ = std::numeric_limits<size_t>::max();

    void divide_recursive(LevelSetConstIterator level_begin,
                          LevelSetConstIterator level_end,
                          const std::vector<std::vector<VertexType>> &global_level_sets,
                          std::vector<std::vector<std::vector<VertexType>>> &all_sections,
                          size_t current_depth) const {
        const auto current_range_size = static_cast<size_t>(std::distance(level_begin, level_end));
        size_t start_level_idx = static_cast<size_t>(std::distance(global_level_sets.cbegin(), level_begin));
        size_t end_level_idx = static_cast<size_t>(std::distance(global_level_sets.cbegin(), level_end));

        // --- Base Cases for Recursion ---
        if (current_depth >= max_depth_ || current_range_size < min_subseq_len_) {
            if constexpr (enable_debug_print) {
                std::cout << "[DEBUG depth " << current_depth << "] Base case reached. Creating section from levels "
                          << start_level_idx << " to " << end_level_idx << "." << std::endl;
            }
            // Ensure the section is not empty before adding
            if (start_level_idx < end_level_idx) {
                all_sections.push_back(this->get_components_for_range(start_level_idx, end_level_idx, global_level_sets));
            }
            return;
        }

        // --- Create a view of the levels for the current sub-problem ---
        std::vector<std::vector<VertexType>> sub_level_sets(level_begin, level_end);

        SequenceGenerator<Graph_t> generator(*(this->dag_ptr_), sub_level_sets);
        std::vector<double> sequence = generator.generate(sequence_metric_);

        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG depth " << current_depth << "] Analyzing sequence: ";
            for (const auto &val : sequence) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }

        std::vector<size_t> local_cuts = splitter_->split(sequence);

        // --- Base Case: No further cuts found ---
        if (local_cuts.empty()) {
            if constexpr (enable_debug_print) {
                std::cout << "[DEBUG depth " << current_depth << "] No cuts found. Creating section from levels "
                          << start_level_idx << " to " << end_level_idx << "." << std::endl;
            }
            all_sections.push_back(this->get_components_for_range(start_level_idx, end_level_idx, global_level_sets));
            return;
        }

        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG depth " << current_depth << "] Found " << local_cuts.size() << " cuts: ";
            for (const auto c : local_cuts) {
                std::cout << c << ", ";
            }
            std::cout << "in level range [" << start_level_idx << ", " << end_level_idx << "). Recursing." << std::endl;
        }

        // --- Recurse on the new, smaller sub-problems ---
        std::sort(local_cuts.begin(), local_cuts.end());
        local_cuts.erase(std::unique(local_cuts.begin(), local_cuts.end()), local_cuts.end());

        auto current_sub_begin = level_begin;
        for (const auto &local_cut_idx : local_cuts) {
            auto cut_iterator = level_begin + static_cast<DifferenceType>(local_cut_idx);
            if (cut_iterator > current_sub_begin) {
                divide_recursive(current_sub_begin, cut_iterator, global_level_sets, all_sections, current_depth + 1);
            }
            current_sub_begin = cut_iterator;
        }
        // Recurse on the final segment from the last cut to the end.
        if (current_sub_begin < level_end) {
            divide_recursive(current_sub_begin, level_end, global_level_sets, all_sections, current_depth + 1);
        }
    }
};

}    // namespace osp
