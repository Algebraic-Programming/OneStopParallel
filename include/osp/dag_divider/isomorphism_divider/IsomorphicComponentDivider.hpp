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

#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <map>

#include "osp/auxiliary/misc.hpp"
#include "osp/dag_divider/IsomorphismGroups.hpp"
#include "osp/dag_divider/DagDivider.hpp"
#include "osp/dag_divider/ConnectedComponentDivider.hpp"
#include "MerkleHashComputer.hpp"
#include "WavefrontOrbitProcessor.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"

namespace osp {

/**
 * @class IsomorphicLayoutDivider
 * @brief Identifies large, structurally isomorphic subgraphs within a DAG.
 *
 * This algorithm processes the DAG wavefront by wavefront. It uses a child-centric
 * approach: for each new wavefront, it analyzes each orbit (a group of structurally
 * identical nodes) and traces its parentage. Based on the merge dynamics, the
 * orbit decides whether to (A) continue a parent group of subgraphs or (B) break
 * the parent group and start a new one if the merge would shrink the group below
 * a size threshold. In a second step, any parent groups that were broken or did
 * not produce children are finalized.
 */
template<typename Graph_t, typename node_hash_func_t = default_node_hash_func<vertex_idx_t<Graph_t>>>
class IsomorphicComponentDivider : public IDagDivider<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>,
                  "IsomorphicComponentDivider can only be used with computational DAGs.");

private:
    using VertexType = vertex_idx_t<Graph_t>;
    using Subgraph = subgraph<Graph_t>;
    using MerkleHashComputer_t = MerkleHashComputer<Graph_t, node_hash_func_t, true>;

    using InternalConstrGraph_t = Graph_t;

public:
    explicit IsomorphicComponentDivider() {}

    std::vector<std::vector<std::vector<VertexType>>> divide(const Graph_t &dag) override {
        if (dag.num_vertices() == 0) return {};

        ConnectedComponentDivider<Graph_t, InternalConstrGraph_t> cc_divider_;
        bool has_more_than_one_connected_component = cc_divider_.compute_connected_components(dag);

        if (has_more_than_one_connected_component) {
            std::cout << "Found " << cc_divider_.get_sub_dags().size() << " connected components in original dag.\n";

            // IsomorphismGroups<Graph_t, InternalConstrGraph_t> iso_groups;
            // std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> vertex_maps = cc_divider_.compute_vertex_maps(dag);
            // iso_groups.compute_isomorphism_groups(vertex_maps, dag);

            // for (auto & sub_dag : iso_groups.get_isomorphism_groups_subgraphs()[0]) {
            //     analyze_connected_dag(sub_dag);
            // }

            for (auto & sub_dag : cc_divider_.get_sub_dags()) {
                analyze_connected_dag(sub_dag);
            }

        } else {
            std::cout << "Found 1 connected component in original dag.\n";
            return analyze_connected_dag(dag);
        }

        return {};
    }

private:

    std::unordered_map<size_t, std::vector<VertexType>> get_orbits(const std::vector<VertexType>& level, MerkleHashComputer_t& hasher) const {
        std::unordered_map<size_t, std::vector<VertexType>> orbits;
        for (const auto v : level) {
            orbits[hasher.get_vertex_hash(v)].push_back(v);
        }
        return orbits;
    }

    struct WavefrontSection {
        unsigned start_wavefront;        
        unsigned end_wavefront;
    };

    struct WavefrontAnalysis {
        std::vector<size_t> cut_levels;
        std::vector<WavefrontSection> pipeline_sections;
        std::vector<WavefrontSection> complex_sections;

        double percentage_complex() const {
            size_t complex_count = 0;
            for (const auto& sec : complex_sections) {
                complex_count += (sec.end_wavefront - sec.start_wavefront + 1);
            }

            size_t total_count = complex_count;
            for (const auto& sec : pipeline_sections) {
                total_count += (sec.end_wavefront - sec.start_wavefront + 1);
            }

            if (total_count == 0) {
                return 0.0;
            }

            return (static_cast<double>(complex_count) / static_cast<double>(total_count)) * 100.0;
        }

    };

    // Option A: simple factor-based "same magnitude"
    inline bool same_magnitude(size_t a, size_t b, double factor = 2.0) const {
        if (a == 0 || b == 0) return false;
        double ratio = static_cast<double>(std::max(a,b)) / static_cast<double>(std::min(a,b));
        return ratio <= factor;
    }

    inline bool same_magnitude_log(size_t a, size_t b, int max_log_diff = 1) const {
        if (a == 0 || b == 0) return false;
        int diff = std::abs(static_cast<int>(std::log2(static_cast<double>(a))) -
                            static_cast<int>(std::log2(static_cast<double>(b))));
        return diff <= max_log_diff;
    }

    // Helper function to check if a wavefront with N-1 orbits is structurally
    // similar to one with N orbits, allowing for one orbit to have disappeared/merged.
    inline bool can_extend_peak(const std::vector<size_t>& smaller_list,
                                const std::vector<size_t>& larger_list) const {
        // We are looking for a small structural change, ideally one orbit merging/disappearing.
        if (smaller_list.empty() || smaller_list.size() + 1 != larger_list.size()) {
            return false;
        }

        const size_t min_smaller_set = *std::min_element(smaller_list.begin(), smaller_list.end());
        if (min_smaller_set == 1) {
            return true;
        }

        // Greedily match elements from the smaller list to the larger one.
        std::multiset<size_t> large_set(larger_list.begin(), larger_list.end());

        for (size_t s_val : smaller_list) {
            auto it = large_set.begin();
            bool found_match = false;
            while (it != large_set.end()) {
                if (same_magnitude_log(s_val, *it)) {
                    large_set.erase(it);
                    found_match = true;
                    break;
                }
                ++it;
            }
            if (!found_match) {
                return false; // Cannot find a match for s_val
            }
        }
        return true; // Found a match for every element in smaller_list.
    }

    WavefrontAnalysis identify_peak_sections(const std::vector<size_t>& num_orbits, const std::vector<std::vector<size_t>>& orbit_sizes) const {
        WavefrontAnalysis analysis;
        if (num_orbits.size() < 3) {
            return analysis;
        }

        auto max_it = std::max_element(num_orbits.begin(), num_orbits.end());
        size_t max_val = *max_it;

        if (max_val <= 2) { // Not a "peak" if it's small enough to be a pipeline candidate
            return analysis;
        }

        // 1. Find initial peak plateaus (where num_orbits == max_val)
        std::vector<std::pair<unsigned, unsigned>> peaks;
        for (size_t i = 0; i < num_orbits.size(); ++i) {
            if (num_orbits[i] == max_val) {
                size_t start = i;
                while (i + 1 < num_orbits.size() && num_orbits[i + 1] == max_val) {
                    i++;
                }
                peaks.push_back({static_cast<unsigned>(start), static_cast<unsigned>(i)});
            }
        }

        if (peaks.empty()) {
            return analysis;
        }

        // 2. Extend peaks backwards and forwards with structurally similar wavefronts (N-1 orbits)
        for (auto& peak : peaks) {
            // Extend backwards
            while (peak.first > 0 && num_orbits[peak.first - 1] == num_orbits[peak.first] - 1) {
                if (can_extend_peak(orbit_sizes[peak.first - 1], orbit_sizes[peak.first])) {
                    peak.first--;
                } else {
                    break;
                }
            }
            // Extend forwards
            while (peak.second + 1 < num_orbits.size() && num_orbits[peak.second + 1] == num_orbits[peak.second] - 1) {
                if (can_extend_peak(orbit_sizes[peak.second + 1], orbit_sizes[peak.second])) {
                    peak.second++;
                } else {
                    break;
                }
            }
        }

        // 3. Merge overlapping/adjacent peaks
        if (peaks.size() > 1) {
            std::sort(peaks.begin(), peaks.end());
            std::vector<std::pair<unsigned, unsigned>> merged_peaks;
            merged_peaks.push_back(peaks[0]);
            for (size_t i = 1; i < peaks.size(); ++i) {
                if (peaks[i].first <= merged_peaks.back().second + 1) {
                    merged_peaks.back().second = std::max(merged_peaks.back().second, peaks[i].second);
                } else {
                    merged_peaks.push_back(peaks[i]);
                }
            }
            peaks = merged_peaks;
        }

        if (peaks.empty() || (peaks.size() == 1 && peaks[0].first == 0 && peaks[0].second == num_orbits.size() - 1)) {
            return analysis;
        }

        // 4. Create cuts at the boundaries of the final peak sections
        for (const auto& peak : peaks) {
            if (peak.first > 0) {
                analysis.cut_levels.push_back(peak.first);
            }
            if (peak.second < num_orbits.size() - 1) {
                analysis.cut_levels.push_back(peak.second + 1);
            }
        }

        // Sort and remove duplicates
        std::sort(analysis.cut_levels.begin(), analysis.cut_levels.end());
        analysis.cut_levels.erase(std::unique(analysis.cut_levels.begin(), analysis.cut_levels.end()), analysis.cut_levels.end());

        // Reconstruct sections from cuts
        unsigned last_cut = 0;
        for (size_t cut : analysis.cut_levels) {
            if (cut > last_cut) {
                analysis.complex_sections.push_back({last_cut, static_cast<unsigned>(cut - 1)});
            }
            last_cut = static_cast<unsigned>(cut);
        }
        if (last_cut < num_orbits.size()) {
            analysis.complex_sections.push_back({last_cut, static_cast<unsigned>(num_orbits.size() - 1)});
        }
        return analysis;
    }

    template<typename T>
    T quantile(std::vector<T> data, double q) const {
        if (data.empty()) return T{};
        std::sort(data.begin(), data.end());
        size_t idx = static_cast<size_t>(q * static_cast<double>((data.size() - 1)));
        return data[idx];
    }

   // New helper for variance
    double calculate_coefficient_of_variation(const std::vector<double>& data) const {
        if (data.size() < 2) return 0.0;
        const double sum = std::accumulate(data.begin(), data.end(), 0.0);
        const double data_size = static_cast<double>(data.size());
        const double mean = sum / data_size;
        if (mean == 0.0) return 0.0;

        const double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
        double var = sq_sum / data_size - mean * mean;
        if (var < 0.0) var = 0.0; // Handle floating point inaccuracies
        return std::sqrt(var) / mean;
    }

    WavefrontAnalysis identify_quantile_sections(const std::vector<double>& sequence, double quantile_val = 0.75, size_t min_section_size = 3) const {
        WavefrontAnalysis analysis;
        if (sequence.size() < min_section_size * 2) {
            return analysis;
        }

        if (calculate_coefficient_of_variation(sequence) < 0.1) {
            std::cout << "    - Sequence has low variance, skipping quantile analysis.\n";
            return analysis;
        }

        double q_value = quantile(sequence, quantile_val);

        std::vector<bool> is_high(sequence.size());
        for(size_t i = 0; i < sequence.size(); ++i) {
            is_high[i] = (sequence[i] >= q_value);
        }

        // Find cut points where the state (high/low) changes
        for (size_t i = 1; i < is_high.size(); ++i) {
            if (is_high[i] != is_high[i-1]) {
                analysis.cut_levels.push_back(i);
            }
        }

        if (analysis.cut_levels.empty()) {
            return analysis; // No change points, the whole sequence is either high or low.
        }

        // Validate the partitioning
        unsigned last_cut = 0;
        for (size_t cut : analysis.cut_levels) {
            if ((cut - last_cut) < min_section_size) {
                std::cout << "    - Quantile analysis produced a fractured partition (section size < " << min_section_size << "). Rejecting.\n";
                return WavefrontAnalysis{}; // Return empty analysis
            }
            last_cut = static_cast<unsigned>(cut);
        }
        if ((sequence.size() - last_cut) < min_section_size) {
            std::cout << "    - Quantile analysis produced a fractured partition (last section size < " << min_section_size << "). Rejecting.\n";
            return WavefrontAnalysis{};
        }

        // If valid, populate the analysis object
        last_cut = 0;
        for (size_t cut : analysis.cut_levels) {
            analysis.complex_sections.push_back({last_cut, static_cast<unsigned>(cut - 1)});
            last_cut = static_cast<unsigned>(cut);
        }
        if (last_cut < sequence.size()) {
            analysis.complex_sections.push_back({last_cut, static_cast<unsigned>(sequence.size() - 1)});
        }

        return analysis;
    }

    WavefrontAnalysis analyze_orbit_wavefront_structure(const Graph_t & dag, const std::vector<std::vector<VertexType>>& levels, MerkleHashComputer_t& hasher) const {
                
        v_workw_t<Graph_t> total_dag_work = 0;
        v_commw_t<Graph_t> total_dag_comm_weight = 0;
        for (const auto& level : levels) {
            for (const auto& v : level) {
                total_dag_work += dag.vertex_work_weight(v);
                total_dag_comm_weight += dag.vertex_comm_weight(v);
            }
        }                

        std::vector<size_t> num_orbits(levels.size(), 0);
        std::vector<std::vector<size_t>> orbit_sizes(levels.size());
        std::vector<double> work(levels.size(), 0.0);
        std::vector<double> comm(levels.size(), 0.0);

        // Collect stats
        for (size_t i = 0; i < levels.size(); ++i) {
            auto orbits = get_orbits(levels[i], hasher);

            v_workw_t<Graph_t> wf_work = 0;
            v_commw_t<Graph_t> wf_comm = 0;
            for (const auto& v : levels[i]) {
                wf_work += dag.vertex_work_weight(v);
                wf_comm += dag.vertex_comm_weight(v);
            }

            work[i] = (total_dag_work > 0) ? (100.0 * static_cast<double>(wf_work) / static_cast<double>(total_dag_work)) : 0.0;
            comm[i] = (total_dag_comm_weight > 0) ? (100.0 * static_cast<double>(wf_comm) / static_cast<double>(total_dag_comm_weight)) : 0.0;

            for (const auto& [hash, vertices] : orbits) {
                orbit_sizes[i].push_back(vertices.size());
            }
            std::sort(orbit_sizes[i].begin(), orbit_sizes[i].end());

            num_orbits[i] = orbits.size();
        }

        const auto analysis = identify_pipeline_sections(num_orbits, orbit_sizes);
        double percent_complex = analysis.percentage_complex();

        if (percent_complex < 75.0) {
            std::cout << "Pipeline analysis successful (" << std::fixed << std::setprecision(1) << (100.0 - percent_complex) << "% pipeline sections found).\n";
            return analysis;
        }
        
        std::cout << "Pipeline analysis ineffective (" << std::fixed << std::setprecision(1) << percent_complex << "% complex). Trying other structural analyses...\n";
        
        std::cout << "  -> Trying peak-based analysis...\n";
        auto peak_analysis = identify_peak_sections(num_orbits, orbit_sizes);
        if (!peak_analysis.cut_levels.empty()) {
            std::cout << "  Peak-based analysis successful.\n";
            //print_wavefront_analysis(quantile_analysis);
            return peak_analysis;
        }

        std::cout << "  Peak-based analysis could not find a partition.\n";
        std::cout << "  -> Trying quantile-based analysis...\n";

        std::vector<double> num_orbits_d(num_orbits.begin(), num_orbits.end());
        std::vector<double> max_orbit_size_d;
        for(const auto& vec : orbit_sizes) {
            if (vec.empty()) max_orbit_size_d.push_back(0);
            else max_orbit_size_d.push_back(static_cast<double>(*std::max_element(vec.begin(), vec.end())));
        }

        std::vector<std::pair<std::string, std::vector<double>>> sequences_to_try = {
            {"Num Orbits", num_orbits_d}, {"Max Orbit Size", max_orbit_size_d}, {"Work %", work}, {"Comm %", comm}
        };

        for (const auto& pair : sequences_to_try) {
            std::cout << "    - Analyzing sequence: " << pair.first << "\n";
            auto quantile_analysis = identify_quantile_sections(pair.second);
            if (!quantile_analysis.cut_levels.empty()) {
                std::cout << "  Quantile analysis successful on " << pair.first << ".\n";
                //print_wavefront_analysis(quantile_analysis);
                return quantile_analysis;
            }
        }
        
        std::cout << "All structural analyses failed. Using original complex sections from pipeline analysis.\n";
        return analysis; // Fallback to original analysis
    }

    // --------------------------------------------------
    // Pipeline/Complex detection
    // --------------------------------------------------
    
    // Helper to check if a wavefront can be part of a pipeline.
    inline bool is_pipeline_candidate(size_t num_orbits_at_idx) const {
        return num_orbits_at_idx <= 2;
    }

    // Helper to check if a pipeline can be extended from a previous to a current wavefront.
    inline bool can_extend_pipeline(size_t current_num_orbits, const std::vector<size_t>& current_sizes,
                                    const std::vector<size_t>& prev_sizes) const {
        if (current_num_orbits == 1) {
            return true;
        }

        const size_t current_min_size = current_sizes[0] > current_sizes[1] ? current_sizes[1] : current_sizes[0];
        if (current_min_size == 1) {
            return true;
        }

        // At this point, current_num_orbits is 2.
        // The check is asymmetric: every orbit in the current wavefront must find a
        // similarly-sized orbit in the previous one.
        const size_t current_max_size = current_sizes[0] > current_sizes[1] ? current_sizes[0] : current_sizes[1];
        bool found_match = false;
        for (size_t p : prev_sizes) {
            if (same_magnitude(current_max_size, p)) {
                found_match = true;
                break;
            
            }
        }

        if (!found_match) return false;

        const size_t prev_max_size = *std::max_element(prev_sizes.begin(), prev_sizes.end());
        for (size_t c : current_sizes) {
            if (same_magnitude(prev_max_size, c)) {
                return true;
            }
        }

        return false;
    }

    WavefrontAnalysis identify_pipeline_sections(const std::vector<size_t>& num_orbits, const std::vector<std::vector<size_t>>& orbit_sizes) const {
        WavefrontAnalysis analysis;
        const size_t n = num_orbits.size();
        if (n == 0) return analysis;

        // Helper to add a complex section, merging with the last one if adjacent.
        auto add_complex_section = [&](unsigned start, unsigned end) {
            if (start > end) return;
            if (!analysis.complex_sections.empty() && analysis.complex_sections.back().end_wavefront + 1 == start) {
                analysis.complex_sections.back().end_wavefront = end;
            } else {
                analysis.complex_sections.push_back({start, end});
            }
        };

        // Helper to add a pipeline section, merging with the last one if adjacent.
        auto add_pipeline_section = [&](unsigned start, unsigned end) {
            if (start > end) return;
            if (!analysis.pipeline_sections.empty() && analysis.pipeline_sections.back().end_wavefront + 1 == start) {
                analysis.pipeline_sections.back().end_wavefront = end;
            } else {
                analysis.pipeline_sections.push_back({start, end});
            }
        };

        size_t i = 0;
        while (i < n) {
            // Find the start of a potential pipeline section (must have 1 orbit)
            size_t section_start = i;
            while (section_start < n && num_orbits[section_start] != 1) {
                section_start++;
            }

            // Try to extend this potential pipeline start backwards.
            // A 2-orbit wavefront can be prepended if it's structurally similar.
            while (section_start < n && section_start > 0 && num_orbits[section_start - 1] == 2) {
                if (can_extend_pipeline(num_orbits[section_start - 1], orbit_sizes[section_start - 1], orbit_sizes[section_start])) {
                    section_start--;
                } else {
                    break;
                }
            }

            // Any region before this is considered complex.
            if (section_start > i) {
                add_complex_section(static_cast<unsigned>(i), static_cast<unsigned>(section_start - 1));
            }

            // If we've scanned to the end, we're done.
            if (section_start == n) {
                break;
            }

            // We have a potential pipeline starting at section_start. Try to extend it forward.
            size_t section_end = section_start;
            while (section_end + 1 < n) {
                if (!is_pipeline_candidate(num_orbits[section_end + 1]) ||
                    !can_extend_pipeline(num_orbits[section_end + 1], orbit_sizes[section_end + 1], orbit_sizes[section_end])) {
                    break;
                }
                section_end++;
            }

            // Now, classify the identified section [section_start, section_end].
            size_t one_orbit_count = 0;
            for (size_t j = section_start; j <= section_end; ++j) {
                if (num_orbits[j] == 1) {
                    one_orbit_count++;
                }
            }

            // A pipeline must have at least one single-orbit wavefront (guaranteed by the anchor search)
            // and at least half its wavefronts must have a single orbit. It also must be longer than 1 wavefront.
            // one_orbit_count * 2 >= (section_end - section_start + 1) &&
            if ((section_end - section_start + 1) > 1) {
                add_pipeline_section(static_cast<unsigned>(section_start), static_cast<unsigned>(section_end));
            } else {
                add_complex_section(static_cast<unsigned>(section_start), static_cast<unsigned>(section_end));
            }

            // Continue searching from the next wavefront.
            i = section_end + 1;
        }

        return analysis;
    }

    void print_wavefront_analysis(const WavefrontAnalysis& analysis) const {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "📊 Wavefront Structural Analysis Results\n";
        std::cout << std::string(80, '=') << std::endl;

        if (analysis.pipeline_sections.empty() && analysis.complex_sections.empty()) {
            std::cout << "No structural sections were identified.\n";
        } else {
            std::cout << std::fixed << std::setprecision(1)
                      << "Complexity: " << analysis.percentage_complex() << "% of wavefronts are complex.\n\n";
        }

        if (!analysis.pipeline_sections.empty()) {
            std::cout << "Pipeline Sections (" << analysis.pipeline_sections.size() << "):\n";
            for (const auto& sec : analysis.pipeline_sections) {
                std::cout << "  - Section [" << std::setw(3) << sec.start_wavefront << " - " << std::setw(3) << sec.end_wavefront
                          << "]" << "\n";
            }
        }

        if (!analysis.complex_sections.empty()) {
            std::cout << "Complex Sections (" << analysis.complex_sections.size() << "):\n";
            for (const auto& sec : analysis.complex_sections) {
                std::cout << "  - Section [" << std::setw(3) << sec.start_wavefront << " - " << std::setw(3) << sec.end_wavefront
                          << "] " << "\n";
            }
        }

        if (!analysis.cut_levels.empty()) {
            std::cout << "Suggested Cut Levels: ";
            for (size_t i = 0; i < analysis.cut_levels.size(); ++i) {
                std::cout << analysis.cut_levels[i] << (i < analysis.cut_levels.size() - 1 ? ", " : "");
            }
            std::cout << "\n";
        }
        std::cout << std::string(80, '=') << std::endl;
    }

    void process_complex_subgraph(const Graph_t &dag) const {

        ConnectedComponentDivider<Graph_t, InternalConstrGraph_t> cc_divider_;
        bool has_more_than_one_connected_component = cc_divider_.compute_connected_components(dag);

        if (has_more_than_one_connected_component) {
            std::cout << "Found " << cc_divider_.get_sub_dags().size() << " connected components in complex subdag.\n";
            
            IsomorphismGroups<Graph_t, InternalConstrGraph_t> iso_groups;
            std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> vertex_maps = cc_divider_.compute_vertex_maps(dag);
            iso_groups.compute_isomorphism_groups(vertex_maps, dag);

            for (auto & sub_dag : iso_groups.get_isomorphism_groups_subgraphs()[0]) {
                analyze_connected_dag(sub_dag);
            }

        } else {
            std::cout << "Found 1 connected component in complex subdag.\n";
        }
    }


    std::vector<std::vector<std::vector<VertexType>>> analyze_connected_dag(const Graph_t &dag) const {
            std::vector<std::vector<VertexType>> level_sets = compute_wavefronts(dag);
            MerkleHashComputer_t m_fw_hash(dag);
            if (level_sets.empty()) {
                std::cout << "DAG is empty or has no nodes, skipping analysis.\n";
                return {};
            }

            print_orbit_wavefront_summary(dag, level_sets, m_fw_hash);
            auto analysis = analyze_orbit_wavefront_structure(dag, level_sets, m_fw_hash);
            print_wavefront_analysis(analysis);

            for (auto & pipeline_section : analysis.pipeline_sections) {

                if (pipeline_section.start_wavefront == 0 && pipeline_section.end_wavefront == (level_sets.size() - 1)) {
                    std::cout << "Processing pipline section, spanning entire dag." << std::endl;

                    WavefrontOrbitProcessor<Graph_t,node_hash_func_t> wf_orbit_processor(8);
                    return wf_orbit_processor.process_wavefronts(dag, level_sets);

                } else {
                    std::cout << "Processing pipline section, spanning wavefront " << pipeline_section.start_wavefront << " to " << pipeline_section.end_wavefront << std::endl;

                    std::vector<VertexType> selected_nodes;
                    std::vector<std::vector<VertexType>> subgraph_level_sets(pipeline_section.end_wavefront - pipeline_section.start_wavefront + 1);
                    for (size_t i = pipeline_section.start_wavefront; i <= pipeline_section.end_wavefront; ++i) {
                        for (const auto& v : level_sets[i]) {
                            selected_nodes.push_back(v);
                            subgraph_level_sets[i - pipeline_section.start_wavefront].push_back(v);
                        }
                    }

                    InternalConstrGraph_t subgraph;
                    create_induced_subgraph(dag, subgraph, selected_nodes);

                    WavefrontOrbitProcessor<InternalConstrGraph_t,node_hash_func_t> wf_orbit_processor(8);
                    wf_orbit_processor.process_wavefronts(subgraph, subgraph_level_sets);
                }
            }

            for (auto & complex_section : analysis.complex_sections) {
                
                std::vector<VertexType> selected_nodes;
                for (size_t i = complex_section.start_wavefront; i <= complex_section.end_wavefront; ++i) {
                    for (const auto& v : level_sets[i]) {
                        selected_nodes.push_back(v);
                    }
                }

                InternalConstrGraph_t subgraph;
                create_induced_subgraph(dag, subgraph, selected_nodes);
                process_complex_subgraph(subgraph);
            }

    }

    void print_orbit_wavefront_summary(const Graph_t & dag, const std::vector<std::vector<VertexType>>& levels, MerkleHashComputer_t& hasher) const {
        v_workw_t<Graph_t> total_dag_work = 0;
        v_commw_t<Graph_t> total_dag_comm_weight = 0;
        for (const auto& level : levels) {
            for (const auto& v : level) {
                total_dag_work += dag.vertex_work_weight(v);
                total_dag_comm_weight += dag.vertex_comm_weight(v);
            }
        }

        // Print table header
        std::cout << std::string(230, '=') << "\n";
        std::cout << "📈 DAG Structural Rhythm Analysis\n";
        std::cout << std::string(230, '=') << std::endl;
        std::cout << std::left
                << std::setw(4)  << "WF"       << "| "
                << std::setw(7)  << "Orbits"   << "| "
                << std::setw(10) << "Min Orbit"<< "| "
                << std::setw(10) << "Max Orbit"<< "| "
                << std::setw(9)  << "WF Work"  << "| "
                << std::setw(10) << "WF Work %" << "| "
                << std::setw(22) << "Min-W Orbit (size)" << "| "
                << std::setw(22) << "Max-W Orbit (size)" << "| "
                << std::setw(9)  << "WF Comm"  << "| "
                << std::setw(10) << "WF Comm %" << "| "
                << std::setw(22) << "Min-C Orbit (size)" << "| "
                << std::setw(22) << "Max-C Orbit (size)" << "| "
                << std::setw(20) << "Comment"  << "| "
                << "Orbit Sizes\n";

        std::cout << "----+--------+-----------+-----------+----------+-----------+------------------------+------------------------+----------+-----------+------------------------+------------------------+---------------------+---------------------------\n";

        size_t prev_orbits = 0;
        for (size_t i = 0; i < levels.size(); ++i) {
            auto orbits = get_orbits(levels[i], hasher);

            v_workw_t<Graph_t> wf_work = 0;
            v_commw_t<Graph_t> wf_comm = 0;
            for (const auto& v : levels[i]) {
                wf_work += dag.vertex_work_weight(v);
                wf_comm += dag.vertex_comm_weight(v);
            }

            size_t max_size = 0;
            size_t min_size = 0;
            v_workw_t<Graph_t> max_work = 0;
            size_t size_at_max_work = 0;
            v_workw_t<Graph_t> min_work = 0;
            size_t size_at_min_work = 0;
            v_commw_t<Graph_t> max_comm = 0;
            size_t size_at_max_comm = 0;
            v_commw_t<Graph_t> min_comm = 0;
            size_t size_at_min_comm = 0;
            std::vector<size_t> orbit_sizes;

            if (!orbits.empty()) {
                min_size = std::numeric_limits<size_t>::max();
                min_work = std::numeric_limits<v_workw_t<Graph_t>>::max();
                min_comm = std::numeric_limits<v_commw_t<Graph_t>>::max();

                for (const auto& [hash, vertices] : orbits) {
                    size_t current_size = vertices.size();
                    v_workw_t<Graph_t> current_orbit_work = 0;
                    v_commw_t<Graph_t> current_orbit_comm = 0;
                    for (const auto& v : vertices) {
                        current_orbit_work += dag.vertex_work_weight(v);
                        current_orbit_comm += dag.vertex_comm_weight(v);
                    }

                    orbit_sizes.push_back(current_size);

                    if (current_size > max_size) max_size = current_size;
                    if (current_size < min_size) min_size = current_size;

                    if (current_orbit_work > max_work) {
                        max_work = current_orbit_work;
                        size_at_max_work = current_size;
                    }
                    if (current_orbit_work < min_work) {
                        min_work = current_orbit_work;
                        size_at_min_work = current_size;
                    }

                    if (current_orbit_comm > max_comm) {
                        max_comm = current_orbit_comm;
                        size_at_max_comm = current_size;
                    }
                    if (current_orbit_comm < min_comm) {
                        min_comm = current_orbit_comm;
                        size_at_min_comm = current_size;
                    }
                }
            }

            std::stringstream sizes_ss;
            for (size_t j = 0; j < orbit_sizes.size(); ++j) {
                sizes_ss << orbit_sizes[j] << (j < orbit_sizes.size() - 1 ? "," : "");
            }
            std::string sizes_str = sizes_ss.str();

            std::string comment;
            if (i > 0) {
                if (static_cast<double>(orbits.size()) > static_cast<double>(prev_orbits) * 1.5) {
                    comment = "Diverging";
                } else if (static_cast<double>(orbits.size()) < static_cast<double>(prev_orbits) * 0.6) {
                    comment = "Converging";
                } else if (orbits.size() == 1 && prev_orbits == 1) {
                    comment = "Uniform/Pipeline";
                } else {
                    comment = "Stable";
                }
            }

            std::stringstream min_work_ss, max_work_ss, min_comm_ss, max_comm_ss;
            min_work_ss << min_work << " (" << size_at_min_work << ")";
            max_work_ss << max_work << " (" << size_at_max_work << ")";
            min_comm_ss << min_comm << " (" << size_at_min_comm << ")";
            max_comm_ss << max_comm << " (" << size_at_max_comm << ")";

            double wf_work_percent = (total_dag_work > 0)
                ? (100.0 * static_cast<double>(wf_work) / static_cast<double>(total_dag_work))
                : 0.0;
            double wf_comm_percent = (total_dag_comm_weight > 0)
                ? (100.0 * static_cast<double>(wf_comm) / static_cast<double>(total_dag_comm_weight))
                : 0.0;

            std::cout << std::left << std::setw(4)  << i << "| "
                    << std::setw(7)  << orbits.size() << "| "
                    << std::setw(10) << min_size << "| "
                    << std::setw(10) << max_size << "| "
                    << std::setw(9)  << wf_work << "| "
                    << std::fixed << std::setprecision(1)
                    << std::setw(10) << wf_work_percent << "| " << std::defaultfloat
                    << std::setw(22) << min_work_ss.str() << "| "
                    << std::setw(22) << max_work_ss.str() << "| "
                    << std::setw(9)  << wf_comm << "| "
                    << std::fixed << std::setprecision(1)
                    << std::setw(10) << wf_comm_percent << "| " << std::defaultfloat
                    << std::setw(22) << min_comm_ss.str() << "| "
                    << std::setw(22) << max_comm_ss.str() << "| "
                    << std::setw(20) << comment << "| "
                    << sizes_str << std::endl;

            prev_orbits = orbits.size();

        }
        std::cout << std::string(230, '=') << std::endl;
    }
 

};

} // namespace osp
