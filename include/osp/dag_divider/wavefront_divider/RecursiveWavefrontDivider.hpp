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

#include "AbstractWavefrontDivider.hpp"
#include <iostream>
#include <numeric>

namespace osp {

/**
 * @class RecursiveWavefrontDivider
 * @brief Divides a DAG recursively by finding the best split point in the wavefronts.
 */
template<typename Graph_t>
class RecursiveWavefrontDivider : public AbstractWavefrontDivider<Graph_t> {
public:
    constexpr static bool enable_debug_print = true;

    RecursiveWavefrontDivider(double diff_threshold = 3.0, size_t min_subseq_len = 4, size_t max_depth = std::numeric_limits<size_t>::max())
        : diff_threshold_(diff_threshold), min_subseq_len_(min_subseq_len), max_depth_(max_depth) {}

    std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const Graph_t &dag) override {
        this->dag_ptr_ = &dag;
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG] Starting recursive division." << std::endl;
        }
        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> all_sections;
        std::vector<vertex_idx_t<Graph_t>> all_vertices(this->dag_ptr_->num_vertices());
        std::iota(all_vertices.begin(), all_vertices.end(), 0);
        divide_recursive(all_vertices, all_sections, 0);
        return all_sections;
    }

private:
    using VertexType = vertex_idx_t<Graph_t>;

    double diff_threshold_;
    size_t min_subseq_len_;
    size_t max_depth_;

    void divide_recursive(const std::vector<VertexType>& subgraph_vertices, 
                          std::vector<std::vector<std::vector<VertexType>>>& all_sections,
                          size_t current_depth) const {
        
        if constexpr (enable_debug_print) {
            std::cout << "\n[DEBUG] --- Entering divide_recursive with " << subgraph_vertices.size() << " vertices (depth " << current_depth << ") ---" << std::endl;
        }
        
        if (subgraph_vertices.empty()) {
            return;
        }

        auto level_sets = this->compute_wavefronts_for_subgraph(subgraph_vertices);

        // Base case: max depth reached or subgraph is too small to divide further.
        if (current_depth >= max_depth_ || level_sets.size() < min_subseq_len_) {
            if constexpr (enable_debug_print) {
                if (current_depth >= max_depth_) {
                    std::cout << "[DEBUG] Max recursion depth reached." << std::endl;
                } else {
                    std::cout << "[DEBUG] Subgraph too small (" << level_sets.size() << " wavefronts)." << std::endl;
                }
                std::cout << "[DEBUG] Treating remaining subgraph as a single section." << std::endl;
            }
            all_sections.push_back(this->get_components_for_range(0, level_sets.size(), level_sets));
            return;
        }

        SequenceGenerator<Graph_t> generator(*(this->dag_ptr_), level_sets);
        std::vector<double> sequence = generator.generate(SequenceMetric::COMPONENT_COUNT);
        
        size_t cut_point = find_best_split_point(sequence);

        if (cut_point == 0) { // Base case: no valid cut found
            if constexpr (enable_debug_print) {
                std::cout << "[DEBUG] No significant cut found. Treating remaining subgraph as a single section." << std::endl;
            }
            all_sections.push_back(this->get_components_for_range(0, level_sets.size(), level_sets));
            return;
        }

        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG] Found best cut point after wavefront " << cut_point -1 << ". Creating section." << std::endl;
        }

        all_sections.push_back(this->get_components_for_range(0, cut_point, level_sets));
        
        std::vector<VertexType> remaining_vertices;
        for (size_t i = cut_point; i < level_sets.size(); ++i) {
            remaining_vertices.insert(remaining_vertices.end(), level_sets[i].begin(), level_sets[i].end());
        }
        
        divide_recursive(remaining_vertices, all_sections, current_depth + 1);
    }

    size_t find_best_split_point(const std::vector<double>& seq) const {
        if (seq.size() < min_subseq_len_) return 0;
        
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG] Analyzing component count sequence: ";
            for(const auto& p : seq) std::cout << p << " ";
            std::cout << std::endl;
        }
        
        double max_diff = 0.0;
        size_t best_split = 0;
        for (size_t i = 0; i < seq.size() - 1; ++i) {
            double diff = seq[i] - seq[i+1];
            if (diff > max_diff) {
                max_diff = diff;
                best_split = i + 1;
            }
        }
        
        if constexpr (enable_debug_print) {
            if (max_diff > diff_threshold_) {
                std::cout << "[DEBUG]   -> Max drop is " << max_diff << " at index " << best_split -1 << " -> " << best_split << ". (Cut after wavefront " << best_split - 1 << ")" << std::endl;
            } else {
                std::cout << "[DEBUG]   -> Max drop is " << max_diff << ", which is below threshold " << diff_threshold_ << ". No cut." << std::endl;
            }
        }
        
        return (max_diff > diff_threshold_) ? best_split : 0;
    }
};

};
