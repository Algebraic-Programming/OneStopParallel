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

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <queue>
#include <memory>
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/auxiliary/datastructures/union_find.hpp"
#include "osp/dag_divider/DagDivider.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "SequenceSplitter.hpp"
#include "WavefrontStatisticsCollector.hpp"

namespace osp {

/**
 * @class WavefrontComponentDivider
 * @brief Divides the wavefronts of a computational DAG into consecutive groups or sections.
 */
template<typename Graph_t>
class WavefrontComponentDivider : public IDagDivider<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>,
                  "WavefrontComponentDivider can only be used with computational DAGs.");

public:
    // --- Configuration Enums ---
    enum class DivisionStrategy { SCAN_ALL, RECURSIVE };
    enum class SequenceMetric { COMPONENT_COUNT, AVAILABLE_PARALLELISM };
    enum class SplitAlgorithm { LARGEST_STEP, VARIANCE, THRESHOLD_SCAN };

    constexpr static bool enable_debug_print = true;

    WavefrontComponentDivider() = default;

    std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const Graph_t &dag_) override {
        dag_ptr_ = &dag_;
        if (division_strategy_ == DivisionStrategy::RECURSIVE) {
            if constexpr (enable_debug_print) {
                std::cout << "[DEBUG] Starting recursive division." << std::endl;
            }
            std::vector<std::vector<std::vector<VertexType>>> all_sections;
            std::vector<VertexType> all_vertices(dag_ptr_->num_vertices());
            std::iota(all_vertices.begin(), all_vertices.end(), 0);
            divide_recursive(all_vertices, all_sections);
            return all_sections;
        }
        
        return divide_scan_all();
    }

    // --- Setters for configuration ---
    inline void set_division_strategy(DivisionStrategy strategy) { division_strategy_ = strategy; }
    inline void set_scan_metric(SequenceMetric metric) { sequence_metric_ = metric; }
    inline void set_scan_algorithm(SplitAlgorithm algorithm) { split_algorithm_ = algorithm; }
    inline void set_absolute_threshold(double threshold) { absolute_threshold_ = threshold; }

private:
    using VertexType = vertex_idx_t<Graph_t>;

    // --- Member Variables ---
    const Graph_t* dag_ptr_ = nullptr;

    // Configuration
    DivisionStrategy division_strategy_ = DivisionStrategy::SCAN_ALL;
    SequenceMetric sequence_metric_ = SequenceMetric::COMPONENT_COUNT;
    SplitAlgorithm split_algorithm_ = SplitAlgorithm::LARGEST_STEP;
    
    // Parameters for splitters
    double var_mult_ = 0.5;
    double var_threshold_ = 1.0;
    double diff_threshold_ = 3.0;
    size_t min_subseq_len_ = 4;
    double absolute_threshold_ = 10.0;

    /**
     * @class SequenceGenerator
     * @brief Helper to generate a numerical sequence based on a chosen metric.
     */
    class SequenceGenerator {
    public:
        SequenceGenerator(const Graph_t& dag, const std::vector<std::vector<VertexType>>& level_sets)
            : dag_(dag), level_sets_(level_sets) {}

        std::vector<double> generate(SequenceMetric metric) {
            switch (metric) {
                case SequenceMetric::AVAILABLE_PARALLELISM:
                    return generate_available_parallelism();
                case SequenceMetric::COMPONENT_COUNT:
                default:
                    return generate_component_count();
            }
        }
    private:
        std::vector<double> generate_component_count() {
            detail::StatisticsCollector<Graph_t> collector(dag_, level_sets_);
            auto fwd_stats = collector.compute_forward();
            std::vector<double> seq;
            seq.reserve(fwd_stats.size());
            for(const auto& stat : fwd_stats) {
                seq.push_back(static_cast<double>(stat.number_of_connected_components));
            }
            return seq;
        }

        std::vector<double> generate_available_parallelism() {
            std::vector<double> seq;
            seq.reserve(level_sets_.size());
            double cumulative_work = 0.0;
            for (size_t i = 0; i < level_sets_.size(); ++i) {
                for (const auto& vertex : level_sets_[i]) {
                    cumulative_work += dag_.vertex_work_weight(vertex);
                }
                seq.push_back(cumulative_work / (i + 1.0));
            }
            return seq;
        }

        const Graph_t& dag_;
        const std::vector<std::vector<VertexType>>& level_sets_;
    };

    /**
     * @brief Global scan implementation for all non-recursive methods.
     */
    std::vector<std::vector<std::vector<VertexType>>> divide_scan_all() {
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG] Starting scan-all division." << std::endl;
        }
        std::vector<std::vector<VertexType>> level_sets = compute_wavefronts(*dag_ptr_);
        if (level_sets.empty()) return {};

        // 1. Generate the sequence based on the chosen metric
        SequenceGenerator generator(*dag_ptr_, level_sets);
        std::vector<double> sequence = generator.generate(sequence_metric_);
        
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG]   Metric: " << static_cast<int>(sequence_metric_) 
                      << ", Algorithm: " << static_cast<int>(split_algorithm_) << std::endl;
            std::cout << "[DEBUG]   Generated sequence: ";
            for(const auto& val : sequence) std::cout << val << " ";
            std::cout << std::endl;
        }

        // 2. Split the sequence using the chosen algorithm
        std::unique_ptr<detail::SequenceSplitter> splitter;
        switch(split_algorithm_) {
            case SplitAlgorithm::VARIANCE:
                splitter = std::make_unique<detail::VarianceSplitter>(var_mult_, var_threshold_);
                break;
            case SplitAlgorithm::THRESHOLD_SCAN:
                splitter = std::make_unique<detail::ThresholdScanSplitter>(diff_threshold_, absolute_threshold_);
                break;
            case SplitAlgorithm::LARGEST_STEP:
            default:
                splitter = std::make_unique<detail::LargestStepSplitter>(diff_threshold_, min_subseq_len_);
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
        
        // 3. Create the final vertex maps from the cuts
        detail::StatisticsCollector<Graph_t> collector(*dag_ptr_, level_sets);
        auto final_stats = collector.compute_forward();
        
        return create_vertex_maps_from_cuts(cut_levels, level_sets, final_stats);
    }

    /**
     * @brief Recursive implementation: finds the best cut, partitions, and recurses.
     */
    void divide_recursive(const std::vector<VertexType>& subgraph_vertices, 
                          std::vector<std::vector<std::vector<VertexType>>>& all_sections) {
        
        if constexpr (enable_debug_print) {
            std::cout << "\n[DEBUG] --- Entering divide_recursive with " << subgraph_vertices.size() << " vertices ---" << std::endl;
        }

        auto level_sets = compute_wavefronts_for_subgraph(subgraph_vertices);

        if (level_sets.size() < min_subseq_len_) {
            if constexpr (enable_debug_print) {
                std::cout << "[DEBUG] Subgraph too small (" << level_sets.size() << " wavefronts), treating as a single section." << std::endl;
            }
            if (!subgraph_vertices.empty()) {
                all_sections.push_back(get_components_for_range(0, level_sets.size(), level_sets));
            }
            return;
        }

        SequenceGenerator generator(*dag_ptr_, level_sets);
        std::vector<double> sequence = generator.generate(SequenceMetric::COMPONENT_COUNT); // Recursive uses component count by default
        
        size_t cut_point = find_best_split_point(sequence);

        if (cut_point == 0) { // Base case: no valid cut found
            if constexpr (enable_debug_print) {
                std::cout << "[DEBUG] No significant cut found. Treating remaining subgraph as a single section." << std::endl;
            }
            all_sections.push_back(get_components_for_range(0, level_sets.size(), level_sets));
            return;
        }

        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG] Found best cut point after wavefront " << cut_point -1 << ". Creating section." << std::endl;
        }

        all_sections.push_back(get_components_for_range(0, cut_point, level_sets));
        
        std::vector<VertexType> remaining_vertices;
        for (size_t i = cut_point; i < level_sets.size(); ++i) {
            remaining_vertices.insert(remaining_vertices.end(), level_sets[i].begin(), level_sets[i].end());
        }
        
        divide_recursive(remaining_vertices, all_sections);
    }

    /**
     * @brief Finds the single best split point in a sequence (largest drop).
     * @return The index of the wavefront to cut AFTER. Returns 0 if no valid cut is found.
     */
    size_t find_best_split_point(const std::vector<double>& seq) {
        if (seq.size() < min_subseq_len_) return 0;
        
        if constexpr (enable_debug_print) {
            std::cout << "[DEBUG] Analyzing parallelism sequence: ";
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

    /**
     * @brief Creates the final vertex map sections from the computed cut levels.
     */
    std::vector<std::vector<std::vector<VertexType>>> create_vertex_maps_from_cuts(
        const std::vector<size_t>& cut_levels,
        const std::vector<std::vector<VertexType>>& level_sets,
        const std::vector<detail::WavefrontStatistics<Graph_t>>& final_stats) {
        
        if (cut_levels.empty()) {
            if (!final_stats.empty() && final_stats.back().number_of_connected_components > 1) {
                return { final_stats.back().connected_components_vertices };
            }
            return {};
        }

        std::vector<std::vector<std::vector<VertexType>>> vertex_maps(cut_levels.size() + 1);
        size_t level_set_idx = 0;

        for (size_t i = 0; i < cut_levels.size(); ++i) {
            vertex_maps[i] = get_components_for_range(level_set_idx, cut_levels[i], level_sets);
            level_set_idx = cut_levels[i];
        }
        vertex_maps.back() = get_components_for_range(level_set_idx, level_sets.size(), level_sets);

        return vertex_maps;
    }

    /**
     * @brief Helper to get connected components for a specific range of levels.
     */
    std::vector<std::vector<VertexType>> get_components_for_range(
        size_t start_level, size_t end_level,
        const std::vector<std::vector<VertexType>>& level_sets) {
        
        union_find_universe_t<Graph_t> uf;
        for (size_t i = start_level; i < end_level; ++i) {
            for (const auto vertex : level_sets[i]) {
                uf.add_object(vertex, dag_ptr_->vertex_work_weight(vertex), dag_ptr_->vertex_mem_weight(vertex));
            }
            for (const auto& node : level_sets[i]) {
                for (const auto& child : dag_ptr_->children(node)) {
                    if (uf.is_in_universe(child)) uf.join_by_name(node, child);
                }
                for (const auto& parent : dag_ptr_->parents(node)) {
                    if (uf.is_in_universe(parent)) uf.join_by_name(parent, node);
                }
            }
        }
        return uf.get_connected_components();
    }

    /**
     * @brief Computes wavefronts for the entire DAG.
     */
    std::vector<std::vector<VertexType>> compute_wavefronts(const Graph_t& dag) const {
        std::vector<VertexType> all_vertices(dag.num_vertices());
        std::iota(all_vertices.begin(), all_vertices.end(), 0);
        return compute_wavefronts_for_subgraph(all_vertices);
    }

    /**
     * @brief Computes wavefronts for a specific subset of vertices.
     */
    std::vector<std::vector<VertexType>> compute_wavefronts_for_subgraph(
        const std::vector<VertexType>& vertices) const {
        
        if (vertices.empty()) return {};

        std::vector<std::vector<VertexType>> level_sets;
        std::unordered_set<VertexType> vertex_set(vertices.begin(), vertices.end());
        std::unordered_map<VertexType, int> in_degree;
        std::queue<VertexType> q;

        for (const auto& v : vertices) {
            in_degree[v] = 0;
            for (const auto& p : dag_ptr_->parents(v)) {
                if (vertex_set.count(p)) {
                    in_degree[v]++;
                }
            }
            if (in_degree[v] == 0) {
                q.push(v);
            }
        }

        while (!q.empty()) {
            size_t level_size = q.size();
            std::vector<VertexType> current_level;
            for (size_t i = 0; i < level_size; ++i) {
                VertexType u = q.front();
                q.pop();
                current_level.push_back(u);
                for (const auto& v : dag_ptr_->children(u)) {
                    if (vertex_set.count(v)) {
                        in_degree[v]--;
                        if (in_degree[v] == 0) {
                            q.push(v);
                        }
                    }
                }
            }
            level_sets.push_back(current_level);
        }
        return level_sets;
    }
};

} // namespace osp