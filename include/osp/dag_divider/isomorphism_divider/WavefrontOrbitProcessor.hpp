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
#include "osp/dag_divider/DagDivider.hpp"
#include "osp/dag_divider/ConnectedComponentDivider.hpp"
#include "MerkleHashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/dag_divider/DagDivider.hpp"

namespace osp {

// The subgraph struct remains largely the same. It's a container for a
// growing, structurally-coherent component of the DAG.
template<typename Graph_t> 
struct subgraph {
    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> vertices;
    size_t current_hash; // Hash of the orbit this subgraph currently belongs to
    unsigned family_id;

    v_workw_t<Graph_t> work_weight;
    v_memw_t<Graph_t> memory_weight;
    
    unsigned start_wavefront;
    unsigned end_wavefront;

    subgraph() = default;

    // Constructor for a new subgraph starting with a single vertex.
    subgraph(VertexType vertex, size_t hash_arg, const Graph_t& dag, unsigned wavefront_arg, unsigned family_id_arg)
        : vertices({vertex}), current_hash(hash_arg), family_id(family_id_arg),
          work_weight(dag.vertex_work_weight(vertex)),
          memory_weight(dag.vertex_mem_weight(vertex)),
          start_wavefront(wavefront_arg), end_wavefront(wavefront_arg) {}
};


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
class WavefrontOrbitProcessor : public IDagDivider<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>,
                  "IsomorphicComponentDivider can only be used with computational DAGs.");

private:
    using VertexType = vertex_idx_t<Graph_t>;
    using Subgraph = subgraph<Graph_t>;
    using MerkleHashComputer_t = MerkleHashComputer<Graph_t, node_hash_func_t, true>;

    using InternalConstrGraph_t = Graph_t;

    // A group is uniquely identified by its structure (hash) and its lineage (family_id).
    using GroupKey = std::pair<size_t, unsigned>;
    struct GroupKeyHash {
        std::size_t operator()(const GroupKey& k) const {
            size_t seed = 0;
            hash_combine(seed, k.first);
            hash_combine(seed, k.second);
            return seed;
        }
    };

    // --- Algorithm Parameters ---
    // If a merge event causes an isomorphic group to shrink below this size,
    // the parent group is broken, and a new group is started.
    size_t min_iso_group_size_threshold_;
   

    // --- Algorithm State ---
    const Graph_t* dag_;
    unsigned next_subgraph_id_ = 0;
    unsigned next_family_id_ = 0;

    std::unordered_map<unsigned, Subgraph> active_subgraphs_;
    std::vector<Subgraph> finalized_subgraphs_;
    std::vector<unsigned> vertex_to_subgraph_id_;

    // Maps the ID of a finalized subgraph to its family_id.
    std::unordered_map<unsigned, unsigned> finalized_sg_id_to_family_id_;

    // Groups active subgraphs by their isomorphism hash and family ID.
    std::unordered_map<GroupKey, std::vector<unsigned>, GroupKeyHash> isomorphic_groups_;

    // Enum to track the fate of a parent group during a wavefront transition.
    enum class GroupStatus { UNCHANGED, CONTINUED, BROKEN };

    // Struct for pre-analysis of parent group evolution
    struct ParentAnalysis {
        std::vector<std::pair<size_t, const std::vector<VertexType>*>> continuing_orbits;
        std::vector<std::pair<size_t, const std::vector<VertexType>*>> breaking_orbits;
    };

public:
    explicit WavefrontOrbitProcessor(size_t min_iso_group_size_threshold = 4)
        : min_iso_group_size_threshold_(min_iso_group_size_threshold) {}


    std::vector<std::vector<std::vector<VertexType>>> divide(const Graph_t &dag) override {
        dag_ = &dag;
        next_subgraph_id_ = 0;
        next_family_id_ = 0;
        active_subgraphs_.clear();
        finalized_subgraphs_.clear();
        vertex_to_subgraph_id_.assign(dag.num_vertices(), std::numeric_limits<unsigned>::max());
        finalized_sg_id_to_family_id_.clear();
        isomorphic_groups_.clear();
       
        std::vector<std::vector<VertexType>> level_sets = compute_wavefronts(dag);

        MerkleHashComputer_t m_fw_hash(dag);    

        // --- 3. Main Loop: Process Wavefronts ---
        for (unsigned wf_idx = 0; wf_idx < level_sets.size(); ++wf_idx) {
            const auto& level = level_sets[wf_idx];
            std::cout << "\n" << std::string(80, '=') << "\n";
            std::cout << "🌊 Processing Wavefront " << wf_idx << " (Size: " << level.size() << ")\n";
            std::cout << std::string(80, '=') << std::endl;

            // Group vertices in the current wavefront by their forward Merkle hash.
            std::unordered_map<size_t, std::vector<VertexType>> orbits = get_orbits(level, m_fw_hash);
            print_orbit_summary(orbits);

            if (wf_idx == 0) {
                // For the first wavefront (sources), all vertices start new subgraphs.
                for (const auto& [hash, vertices] : orbits) {
                    // All vertices in the same orbit in WF0 get the same new family ID.
                    const unsigned new_family_id = next_family_id_++;
                    for (const auto& v : vertices) {
                        create_new_subgraph(v, hash, wf_idx, new_family_id);
                    }
                }
                rebuild_isomorphic_groups();
                print_active_groups_summary(wf_idx);
                continue;
            }            

            process_wavefront_orbits(orbits, wf_idx);
            
            print_active_groups_summary(wf_idx);
        }

        // --- 4. Finalization ---
        // Any remaining active subgraphs are finalized at the end.
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "✅ Finalizing all remaining active subgraphs." << std::endl;

        for (const auto& [id, sg] : active_subgraphs_) {
            finalize_subgraph(id, sg);
        }
 
        std::cout << "Total finalized subgraphs: " << finalized_subgraphs_.size() << std::endl;

        print_final_layout_summary();

        return convert_to_output_format();
    }

    std::vector<std::vector<std::vector<VertexType>>> process_wavefronts(const Graph_t &dag, std::vector<std::vector<VertexType>> & level_sets) {
        dag_ = &dag;
        next_subgraph_id_ = 0;
        next_family_id_ = 0;
        active_subgraphs_.clear();
        finalized_subgraphs_.clear();
        vertex_to_subgraph_id_.assign(dag.num_vertices(), std::numeric_limits<unsigned>::max());
        finalized_sg_id_to_family_id_.clear();
        isomorphic_groups_.clear();
       
        MerkleHashComputer_t m_fw_hash(dag);    

        // --- 3. Main Loop: Process Wavefronts ---
        for (unsigned wf_idx = 0; wf_idx < level_sets.size(); ++wf_idx) {
            const auto& level = level_sets[wf_idx];
            std::cout << "\n" << std::string(80, '=') << "\n";
            std::cout << "🌊 Processing Wavefront " << wf_idx << " (Size: " << level.size() << ")\n";
            std::cout << std::string(80, '=') << std::endl;

            // Group vertices in the current wavefront by their forward Merkle hash.
            std::unordered_map<size_t, std::vector<VertexType>> orbits = get_orbits(level, m_fw_hash);
            print_orbit_summary(orbits);

            if (wf_idx == 0) {
                // For the first wavefront (sources), all vertices start new subgraphs.
                for (const auto& [hash, vertices] : orbits) {
                    // All vertices in the same orbit in WF0 get the same new family ID.
                    const unsigned new_family_id = next_family_id_++;
                    for (const auto& v : vertices) {
                        create_new_subgraph(v, hash, wf_idx, new_family_id);
                    }
                }
                rebuild_isomorphic_groups();
                print_active_groups_summary(wf_idx);
                continue;
            }            

            process_wavefront_orbits(orbits, wf_idx);
            
            print_active_groups_summary(wf_idx);
        }

        // --- 4. Finalization ---
        // Any remaining active subgraphs are finalized at the end.
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "✅ Finalizing all remaining active subgraphs." << std::endl;

        for (const auto& [id, sg] : active_subgraphs_) {
            finalize_subgraph(id, sg);
        }
 
        std::cout << "Total finalized subgraphs: " << finalized_subgraphs_.size() << std::endl;

        print_final_layout_summary();

        return convert_to_output_format();
    }


private:



    void process_wavefront_orbits(const std::unordered_map<size_t, std::vector<VertexType>>& orbits, unsigned wf_idx) {
        std::unordered_map<unsigned, Subgraph> next_active_subgraphs;
        std::unordered_map<GroupKey, GroupStatus, GroupKeyHash> group_fate;
        for (const auto& [key, group] : isomorphic_groups_) {
            group_fate[key] = GroupStatus::UNCHANGED;
        }

        // Make a mutable copy of active subgraphs to consume from during continuation.
        auto consumable_active_subgraphs = active_subgraphs_;

        std::cout << "\n--- ⚖️ Phase 1: Orbit-Driven Decision Making ---\n";

        // --- Step 1: Analyze parentage of all orbits ---
        std::map<GroupKey, ParentAnalysis> parent_analysis_map;
        std::vector<const typename std::unordered_map<size_t, std::vector<VertexType>>::value_type*> unhandled_orbits;
        analyze_parent_fates(orbits, parent_analysis_map, unhandled_orbits);

        // --- Step 2: Process single-parent-group continuations ---
        for (auto& [parent_key, analysis] : parent_analysis_map) {
            const unsigned parent_family_id = parent_key.second;

            if (analysis.continuing_orbits.empty()) {
                // Case: All child orbits are breaking away. The parent group is broken.
                group_fate[parent_key] = GroupStatus::BROKEN;
                std::cout << "  - Group (H:" << parent_key.first << "/F:" << parent_family_id << ") is being BROKEN by " << analysis.breaking_orbits.size() << " orbit(s).\n";
            } else {
                // Case: At least one child orbit is continuing the group.
                group_fate[parent_key] = GroupStatus::CONTINUED;

                // Check for fan-out from a single parent subgraph to multiple continuing orbits
                std::unordered_map<unsigned, int> parent_sg_orbit_count;
                for (const auto& [hash, vertices_ptr] : analysis.continuing_orbits) {
                    const auto& rep_v = vertices_ptr->front();
                    for (const auto& p : dag_->parents(rep_v)) {
                        unsigned p_sg_id = vertex_to_subgraph_id_[p];
                        if (p_sg_id != std::numeric_limits<unsigned>::max() && consumable_active_subgraphs.count(p_sg_id) &&
                            consumable_active_subgraphs.at(p_sg_id).current_hash == parent_key.first && consumable_active_subgraphs.at(p_sg_id).family_id == parent_family_id) {
                            parent_sg_orbit_count[p_sg_id]++;
                        }
                    }
                }

                bool fan_out_detected = std::any_of(parent_sg_orbit_count.begin(), parent_sg_orbit_count.end(),
                                                    [](const auto& pair){ return pair.second > 1; });

                if (fan_out_detected) {
                    resolve_fan_out_continuation(parent_key, analysis, wf_idx, next_active_subgraphs, consumable_active_subgraphs);
                } else {
                    handle_simple_continuation(parent_key, analysis, wf_idx, next_active_subgraphs, consumable_active_subgraphs);
                }
            }

            // Handle all orbits that were designated as "breaking" for this parent group.
            for (const auto& [hash, vertices_ptr] : analysis.breaking_orbits) {
                std::cout << "    -> 💥 Breaking off with orbit " << hash << " (starting new group, inheriting Family " << parent_family_id << ").\n";
                start_new_group(hash, *vertices_ptr, wf_idx, parent_family_id, next_active_subgraphs);
            }
        }

        // --- Step 3: Handle Orbits from Complex Merges or with No Active Parents ---
        handle_complex_and_new_orbits(unhandled_orbits, wf_idx, next_active_subgraphs, consumable_active_subgraphs, group_fate);

        // --- Finalize groups based on their determined fate ---
        std::cout << "\n--- 📋 Phase 2: Finalizing Old Groups ---\n";
        for (const auto& [key, status] : group_fate) {
            if (status == GroupStatus::UNCHANGED) {
                std::cout << "  - Group (H:" << key.first << "/F:" << key.second << ") did not evolve. Finalizing.\n";
            } else if (status == GroupStatus::BROKEN) {
                std::cout << "  - Group (H:" << key.first << "/F:" << key.second << ") was broken. Finalizing.\n";
            } else { // CONTINUED
                 std::cout << "  - Group (H:" << key.first << "/F:" << key.second << ") successfully continued.\n";
            }
        }
        // Any subgraphs remaining in `consumable_active_subgraphs` were not continued and must be finalized.
        for (const auto& [id, sg] : consumable_active_subgraphs) {
            finalize_subgraph(id, sg);
        }

        // Update state for the next wavefront
        active_subgraphs_ = std::move(next_active_subgraphs);
        rebuild_isomorphic_groups();
    }

    bool is_viable_continuation(const std::vector<VertexType>& orbit_vertices, const GroupKey& parent_key) const {
        if (orbit_vertices.empty()) return false;

        const auto& parent_group_members = isomorphic_groups_.at(parent_key);
        size_t old_size = parent_group_members.size();
        
        std::map<GroupKey, size_t> current_orbit_parents;
        get_parent_group_counts(orbit_vertices.front(), current_orbit_parents);
        
        if (current_orbit_parents.find(parent_key) == current_orbit_parents.end()) {
            return false; // Should not happen if called correctly
        }
        
        size_t merge_width = current_orbit_parents.at(parent_key);
        size_t new_size = (merge_width > 0) ? (old_size / merge_width) : 0;

        return new_size >= min_iso_group_size_threshold_;
    }

    void analyze_parent_fates(
        const std::unordered_map<size_t, std::vector<VertexType>>& orbits,
        std::map<GroupKey, ParentAnalysis>& parent_analysis_map,
        std::vector<const typename std::unordered_map<size_t, std::vector<VertexType>>::value_type*>& unhandled_orbits)
    {
        for (const auto& orbit_pair : orbits) {
            const auto& orbit_hash = orbit_pair.first;
            const auto& orbit_vertices = orbit_pair.second;

            std::map<GroupKey, size_t> active_parent_counts;
            std::set<unsigned> finalized_parent_families;
            collect_parent_info(orbit_vertices.front(), active_parent_counts, finalized_parent_families);

            // Simple case: children of a single, un-finalized parent group.
            if (active_parent_counts.size() == 1) { // && finalized_parent_families.empty()
                const auto& parent_key = active_parent_counts.begin()->first;
                
                // Decide if this orbit continues the parent group or breaks from it.
                if (is_viable_continuation(orbit_vertices, parent_key)) {
                    parent_analysis_map[parent_key].continuing_orbits.push_back({orbit_hash, &orbit_vertices});
                } else {
                    parent_analysis_map[parent_key].breaking_orbits.push_back({orbit_hash, &orbit_vertices});
                }
            } else {
                // Complex case: multiple parent groups, finalized parents, or no parents.
                // These are handled separately.
                unhandled_orbits.push_back(&orbit_pair);
            }
        }
    }

    void extend_and_merge_in_place(
        size_t child_orbit_hash, const std::vector<VertexType>& orbit_vertices, unsigned wf_idx,
        const std::set<unsigned>& original_parent_ids,
        std::unordered_map<unsigned, Subgraph>& evolving_parents)
    {
        // Group children by the set of their parents from the evolving group.
        std::map<std::set<unsigned>, std::vector<VertexType>> parent_set_to_children;
        for (const auto& v : orbit_vertices) {
            std::set<unsigned> parents_from_group;
            for (const auto& p : dag_->parents(v)) {
                unsigned p_sg_id = vertex_to_subgraph_id_[p];
                if (original_parent_ids.count(p_sg_id)) { // Check against original members
                    parents_from_group.insert(p_sg_id);
                }
            }
            if (!parents_from_group.empty()) {
                parent_set_to_children[parents_from_group].push_back(v);
            }
        }

        // For each merge pattern, extend/merge the parents in the evolving_parents map.
        for (const auto& [parent_set, children] : parent_set_to_children) {
            if (parent_set.empty() || children.empty()) continue;

            // Choose a survivor to merge into.
            unsigned survivor_id = *parent_set.begin();
            
            // Merge other parents into the survivor.
            for (unsigned parent_id : parent_set) {
                if (parent_id == survivor_id) continue;
                if (!evolving_parents.count(parent_id)) continue; // Already merged

                auto& survivor_sg = evolving_parents.at(survivor_id);
                const auto& other_sg = evolving_parents.at(parent_id);

                survivor_sg.vertices.insert(survivor_sg.vertices.end(), other_sg.vertices.begin(), other_sg.vertices.end());
                survivor_sg.work_weight += other_sg.work_weight;
                survivor_sg.memory_weight += other_sg.memory_weight;
                survivor_sg.start_wavefront = std::min(survivor_sg.start_wavefront, other_sg.start_wavefront);
                
                evolving_parents.erase(parent_id);
            }

            // Extend the survivor with the children.
            auto& survivor_sg = evolving_parents.at(survivor_id);
            for (const auto& child : children) {
                survivor_sg.vertices.push_back(child);
                survivor_sg.work_weight += dag_->vertex_work_weight(child);
                survivor_sg.memory_weight += dag_->vertex_mem_weight(child);
            }
            survivor_sg.end_wavefront = wf_idx;
            survivor_sg.current_hash = child_orbit_hash; // The group's structure has evolved.
        }
    }

    void resolve_fan_out_continuation(
        const GroupKey& parent_key,
        ParentAnalysis& analysis,
        unsigned wf_idx,
        std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
        std::unordered_map<unsigned, Subgraph>& consumable_active_subgraphs)
    {
        const unsigned parent_family_id = parent_key.second;
        std::cout << "  - Group (H:" << parent_key.first << "/F:" << parent_family_id 
                  << ") has a fan-out. Resolving via iterative extension.\n";

        // 1. Create a local, mutable copy of the parent group to be extended.
        std::unordered_map<unsigned, Subgraph> evolving_parents;
        std::set<unsigned> original_parent_ids;
        for (unsigned id : isomorphic_groups_.at(parent_key)) {
            evolving_parents.emplace(id, consumable_active_subgraphs.at(id));
            original_parent_ids.insert(id);
        }

        // 2. Iteratively extend the group with the most viable orbits.
        std::list<std::pair<size_t, const std::vector<VertexType>*>> remaining_orbits(
            analysis.continuing_orbits.begin(), analysis.continuing_orbits.end());

        bool extension_was_made = true;
        while (extension_was_made) {
            extension_was_made = false;
            auto best_orbit_it = remaining_orbits.end();
            size_t min_width = std::numeric_limits<size_t>::max();

            std::cout << "    --- Iteration Start (Evolving group size: " << evolving_parents.size() << ") ---\n";
            for (auto it = remaining_orbits.begin(); it != remaining_orbits.end(); ++it) {
                const auto& vertices = *it->second;
                if (vertices.empty()) continue;

                // Calculate merge width against the evolving parent group.
                size_t current_width = 0;
                for (const auto& p : dag_->parents(vertices.front())) {
                    if (original_parent_ids.count(vertex_to_subgraph_id_[p])) {
                        current_width++;
                    }
                }

                // Check viability based on the current state of the evolving group.
                size_t parent_group_size = evolving_parents.size();
                size_t new_size = (current_width > 0) ? (parent_group_size / current_width) : 0;
                bool is_viable = (current_width > 0) && (new_size >= min_iso_group_size_threshold_);
                
                std::cout << "      - Checking orbit " << it->first << " (width " << current_width << "). Predicted new size: " << new_size << (is_viable ? " (Viable)" : " (Not Viable)") << "\n";

                if (is_viable && current_width < min_width) {
                    min_width = current_width;
                    best_orbit_it = it;
                }
            }

            if (best_orbit_it != remaining_orbits.end()) {
                const auto& [hash, vertices_ptr] = *best_orbit_it;
                std::cout << "    -> Extending group with orbit " << hash << " (width: " << min_width << ").\n";
                
                extend_and_merge_in_place(hash, *vertices_ptr, wf_idx, original_parent_ids, evolving_parents);

                extension_was_made = true;
                remaining_orbits.erase(best_orbit_it);
            }
        }

        // 3. Any remaining orbits could not be viably extended and must break off.
        for (const auto& orbit_info : remaining_orbits) {
            std::cout << "    -> 💥 Unmerged orbit " << orbit_info.first << " is breaking off.\n";
            analysis.breaking_orbits.push_back(orbit_info);
        }

        // 4. Commit the changes.
        // Move the final, evolved subgraphs to the next generation.
        for (auto& [id, sg] : evolving_parents) {
            next_active_subgraphs.emplace(id, std::move(sg));
        }
        // Remove the original parent subgraphs from the consumable pool.
        for (unsigned id : original_parent_ids) {
            consumable_active_subgraphs.erase(id);
        }
    }

    void handle_simple_continuation(
        const GroupKey& parent_key,
        const ParentAnalysis& analysis,
        unsigned wf_idx,
        std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
        std::unordered_map<unsigned, Subgraph>& consumable_active_subgraphs)
    {
        std::cout << "  - Group (H:" << parent_key.first << "/F:" << parent_key.second << ") is being CONTINUED by " << analysis.continuing_orbits.size() << " orbit(s).\n";

        for (const auto& [hash, vertices_ptr] : analysis.continuing_orbits) {
            std::cout << "    -> ✅ Continuing with orbit " << hash << ".\n";
            (void)merge_into_group(parent_key, hash, *vertices_ptr, wf_idx, next_active_subgraphs, consumable_active_subgraphs);
        }
    }

    void handle_complex_and_new_orbits(
        const std::vector<const typename std::unordered_map<size_t, std::vector<VertexType>>::value_type*>& unhandled_orbits,
        unsigned wf_idx,
        std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
        std::unordered_map<unsigned, Subgraph>& consumable_active_subgraphs,
        std::unordered_map<GroupKey, GroupStatus, GroupKeyHash>& group_fate)
    {
        for (const auto* orbit_pair_ptr : unhandled_orbits) {
            const auto& orbit_hash = orbit_pair_ptr->first;
            const auto& orbit_vertices = orbit_pair_ptr->second;

            std::map<GroupKey, size_t> active_parent_counts;
            std::set<unsigned> finalized_parent_families;
            collect_parent_info(orbit_vertices.front(), active_parent_counts, finalized_parent_families);

            if (active_parent_counts.empty()) {
                // Case: No active parents. This orbit must start a new group.
                // It might inherit a family ID from a finalized parent, but for simplicity
                // and to ensure distinctness after a merge, we start a new family.
                std::cout << "  - Orbit " << orbit_hash << " has no active parents. Starting new group (Family " << next_family_id_ << ").\n";
                start_new_group(orbit_hash, orbit_vertices, wf_idx, next_family_id_++, next_active_subgraphs);
            } else {
                // Case: Complex merge (multiple active parents, or mix of active/finalized).
                // This also starts a new group, but it consumes the active parents.
                std::cout << "  - Orbit " << orbit_hash << " is a complex merge of " << active_parent_counts.size() << " active groups. Starting new group (Family " << next_family_id_ << ").\n";
                continue_complex_group(orbit_hash, orbit_vertices, wf_idx, next_family_id_++, next_active_subgraphs, consumable_active_subgraphs);
                
                // Mark all consumed parent groups as BROKEN, as their lineage is now merged and terminated.
                for (const auto& [parent_key, count] : active_parent_counts) {
                    if (group_fate.count(parent_key)) { group_fate[parent_key] = GroupStatus::BROKEN; }
                }
            }
        }
    }

    void get_parent_group_counts(const VertexType& representative, std::map<GroupKey, size_t>& counts, const std::unordered_map<unsigned, Subgraph>& parent_pool) const {
        for (const auto& p : dag_->parents(representative)) {
            unsigned p_sg_id = vertex_to_subgraph_id_[p];
            if (p_sg_id != std::numeric_limits<unsigned>::max() && parent_pool.count(p_sg_id)) {
                const auto& parent_sg = parent_pool.at(p_sg_id);
                counts[{parent_sg.current_hash, parent_sg.family_id}]++;
            }
        }
    }

    void get_parent_group_counts(const VertexType& representative, std::map<GroupKey, size_t>& counts) const {
        get_parent_group_counts(representative, counts, active_subgraphs_);
    }

    void collect_parent_info(const VertexType& representative,
                             std::map<GroupKey, size_t>& active_counts,
                             std::set<unsigned>& finalized_family_ids) const {
        for (const auto& p : dag_->parents(representative)) {
            unsigned p_sg_id = vertex_to_subgraph_id_[p];
            if (p_sg_id != std::numeric_limits<unsigned>::max()) {
                if (active_subgraphs_.count(p_sg_id)) {
                    const auto& parent_sg = active_subgraphs_.at(p_sg_id);
                    active_counts[{parent_sg.current_hash, parent_sg.family_id}]++;
                } else if (finalized_sg_id_to_family_id_.count(p_sg_id)) {
                    finalized_family_ids.insert(finalized_sg_id_to_family_id_.at(p_sg_id));
                }
            }
        }
    }

    bool merge_into_group(const GroupKey& parent_group_key, size_t child_orbit_hash, const std::vector<VertexType>& orbit_vertices,
                        unsigned wf_idx, std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
                         std::unordered_map<unsigned, Subgraph>& parent_map) {
        const unsigned family_id = parent_group_key.second;
        // Group children by the set of their parents from the target group
        std::map<std::set<unsigned>, std::vector<VertexType>> parent_set_to_children;
        for (const auto& v : orbit_vertices) {
            std::set<unsigned> parents_from_group;
            for (const auto& p : dag_->parents(v)) {
                unsigned p_sg_id = vertex_to_subgraph_id_[p];
                if (p_sg_id != std::numeric_limits<unsigned>::max() && parent_map.count(p_sg_id) && 
                    parent_map.at(p_sg_id).current_hash == parent_group_key.first && parent_map.at(p_sg_id).family_id == family_id) {
                    parents_from_group.insert(p_sg_id);
                }
            }
            if (!parents_from_group.empty()) {
                parent_set_to_children[parents_from_group].push_back(v);
            }
        }

        if (parent_set_to_children.empty()) {
            return false; // No merge was possible.
        }

        // --- Validation Phase ---
        // Collect all required parents and check for internal conflicts (non-disjoint parent sets).
        // This prevents a single parent from being merged into multiple new subgraphs within this call.
        std::set<unsigned> all_required_parents;
        for (const auto& [parent_set, children] : parent_set_to_children) {
            for (unsigned p_id : parent_set) {
                if (!all_required_parents.insert(p_id).second) {
                    // This parent is required by multiple child groups in this single merge operation.
                    // This indicates an irregular merge structure that this function cannot resolve.
                    // The calling logic (fan-out) should have caught this.
                    return false;
                }
            }
        }

        // --- Creation Phase ---
        std::vector<Subgraph> created_subgraphs;
        // Create one new subgraph for each group of children that share the same set of parents
        for (const auto& [parent_set, children] : parent_set_to_children) {
            const auto& rep_child = children.front();
            Subgraph new_sg(rep_child, child_orbit_hash, *dag_, wf_idx, family_id);
            unsigned min_start_wf = new_sg.start_wavefront;

            for (unsigned parent_id : parent_set) {
                const auto& parent_sg = parent_map.at(parent_id);
                new_sg.vertices.insert(new_sg.vertices.end(), parent_sg.vertices.begin(), parent_sg.vertices.end());
                new_sg.work_weight += parent_sg.work_weight;
                new_sg.memory_weight += parent_sg.memory_weight;
                min_start_wf = std::min(min_start_wf, parent_sg.start_wavefront);
            }
            new_sg.start_wavefront = min_start_wf;

            // Add the other children in this group
            for (size_t i = 1; i < children.size(); ++i) {
                const auto& child = children[i];
                new_sg.vertices.push_back(child);
                new_sg.work_weight += dag_->vertex_work_weight(child);
                new_sg.memory_weight += dag_->vertex_mem_weight(child);
            }
            created_subgraphs.push_back(std::move(new_sg));
        }

        // --- Commit Phase ---
        // 1. Move created subgraphs to a temporary holding to get their new IDs.
        std::vector<std::pair<unsigned, Subgraph>> new_entries;
        for (auto& sg : created_subgraphs) {
            new_entries.emplace_back(next_subgraph_id_++, std::move(sg));
        }

        // 2. Erase all consumed parents from the source map.
        for (unsigned id : all_required_parents) {
            parent_map.erase(id);
        }

        // 3. Add the new subgraphs to the destination map and update vertex mappings.
        for (auto& entry : new_entries) {
            unsigned new_id = entry.first;
            next_active_subgraphs.emplace(new_id, std::move(entry.second));
            for (const auto& v : next_active_subgraphs.at(new_id).vertices) {
                vertex_to_subgraph_id_[v] = new_id;
            }
        }
        return true;
    }

    void continue_complex_group(size_t child_orbit_hash, const std::vector<VertexType>& orbit_vertices,
                                unsigned wf_idx, unsigned new_family_id, std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
                                std::unordered_map<unsigned, Subgraph>& consumable_parents) {
        // Group children by the set of ALL their active parents
        std::map<std::set<unsigned>, std::vector<VertexType>> parent_set_to_children;
        for (const auto& v : orbit_vertices) {
            std::set<unsigned> current_parents;
            for (const auto& p : dag_->parents(v)) {
                unsigned p_sg_id = vertex_to_subgraph_id_[p];
                if (p_sg_id != std::numeric_limits<unsigned>::max() && consumable_parents.count(p_sg_id)) {
                    current_parents.insert(p_sg_id);
                }
            }
            if (!current_parents.empty()) {
                parent_set_to_children[current_parents].push_back(v);
            }
        }

        if (parent_set_to_children.empty()) return;

        // --- Creation Phase ---
        std::vector<std::pair<unsigned, Subgraph>> new_entries;
        std::set<unsigned> all_consumed_parents;

        // Create one new subgraph for each group of children that share the same set of parents
        for (const auto& [parent_set, children] : parent_set_to_children) {
            if (children.empty()) continue;
            
            const auto& rep_child = children.front();
            Subgraph new_sg(rep_child, child_orbit_hash, *dag_, wf_idx, new_family_id);
            unsigned min_start_wf = new_sg.start_wavefront;

            for (unsigned parent_id : parent_set) {
                const auto& parent_sg = consumable_parents.at(parent_id);
                new_sg.vertices.insert(new_sg.vertices.end(), parent_sg.vertices.begin(), parent_sg.vertices.end());
                new_sg.work_weight += parent_sg.work_weight;
                new_sg.memory_weight += parent_sg.memory_weight;
                min_start_wf = std::min(min_start_wf, parent_sg.start_wavefront);
                all_consumed_parents.insert(parent_id);
            }
            new_sg.start_wavefront = min_start_wf;

            // Add the other children in this group that share this parent set
            for (size_t i = 1; i < children.size(); ++i) {
                const auto& child = children[i];
                new_sg.vertices.push_back(child);
                new_sg.work_weight += dag_->vertex_work_weight(child);
                new_sg.memory_weight += dag_->vertex_mem_weight(child);
            }
            new_entries.emplace_back(next_subgraph_id_++, std::move(new_sg));
        }

        // --- Commit Phase ---
        // First, remove all consumed parents.
        for (unsigned parent_id : all_consumed_parents) {
            consumable_parents.erase(parent_id);
        }

        // Second, add new subgraphs to the next active set and update vertex mappings.
        for (auto& entry : new_entries) {
            unsigned new_id = entry.first;
            next_active_subgraphs.emplace(new_id, std::move(entry.second));
            for (const auto& v : next_active_subgraphs.at(new_id).vertices) {
                vertex_to_subgraph_id_[v] = new_id;
            }
        }
    }

    void start_new_group(size_t orbit_hash, const std::vector<VertexType>& orbit_vertices, unsigned wf_idx,
                         unsigned family_id, std::unordered_map<unsigned, Subgraph>& next_active_subgraphs) {
        
        // When starting a new group, each vertex in the orbit begins a new subgraph.
        // These subgraphs are considered isomorphic from the start.
        for (const auto& v : orbit_vertices) {
            // Only create if not already claimed by a different, continuing group
            if (vertex_to_subgraph_id_[v] == std::numeric_limits<unsigned>::max()) {
                create_new_subgraph_in_map(v, orbit_hash, wf_idx, family_id, next_active_subgraphs);
            }
        }
    }

    // --- Helper Methods ---

    std::unordered_map<size_t, std::vector<VertexType>> get_orbits(const std::vector<VertexType>& level, MerkleHashComputer_t& hasher) const {
        std::unordered_map<size_t, std::vector<VertexType>> orbits;
        for (const auto v : level) {
            orbits[hasher.get_vertex_hash(v)].push_back(v);
        }
        return orbits;
    }

    void create_new_subgraph(VertexType v, size_t hash, unsigned wf_idx, unsigned family_id) {
        unsigned id = next_subgraph_id_++;
        active_subgraphs_.emplace(id, Subgraph(v, hash, *dag_, wf_idx, family_id));
        vertex_to_subgraph_id_[v] = id;
    }
    
    unsigned create_new_subgraph_in_map(Subgraph sg, std::unordered_map<unsigned, Subgraph>& map) {
        unsigned id = next_subgraph_id_++;
        map.emplace(id, std::move(sg));
        return id;
    }

    unsigned create_new_subgraph_in_map(VertexType v, size_t hash, unsigned wf_idx, unsigned family_id, std::unordered_map<unsigned, Subgraph>& map) {
        unsigned id = next_subgraph_id_++;
        map.emplace(id, Subgraph(v, hash, *dag_, wf_idx, family_id));
        vertex_to_subgraph_id_[v] = id;
        return id;
    }

    void finalize_subgraph(unsigned id, const Subgraph& sg) {
        finalized_sg_id_to_family_id_[id] = sg.family_id;
        finalized_subgraphs_.push_back(sg);
    }

    void rebuild_isomorphic_groups() {
        isomorphic_groups_.clear();
        for (const auto& [id, sg] : active_subgraphs_) {
            isomorphic_groups_[{sg.current_hash, sg.family_id}].push_back(id);
        }
    }
    
    std::vector<std::vector<std::vector<VertexType>>> convert_to_output_format() {
        if (finalized_subgraphs_.empty()) {
            return {};
        }

        std::unordered_map<GroupKey, std::vector<const Subgraph*>, GroupKeyHash> layout;
        for (const auto& sg : finalized_subgraphs_) {
            layout[{sg.current_hash, sg.family_id}].push_back(&sg);
        }

        struct GroupSpan {
            unsigned min_wf = std::numeric_limits<unsigned>::max();
            unsigned max_wf = 0;
        };

        std::unordered_map<GroupKey, GroupSpan, GroupKeyHash> group_spans;
        for (const auto& [key, group] : layout) {
            for (const auto* sg : group) {
                auto& span = group_spans[key];
                span.min_wf = std::min(span.min_wf, sg->start_wavefront);
                span.max_wf = std::max(span.max_wf, sg->end_wavefront);
            }
        }

        std::vector<std::vector<std::vector<VertexType>>> sections;
        
        std::unordered_set<GroupKey, GroupKeyHash> assigned_groups;
        
        unsigned current_start_wf = 0;
        bool is_first_section = true;

        while (assigned_groups.size() < layout.size()) {
            // --- Find anchor group for the new section ---
            const GroupKey* anchor_key = nullptr;
            unsigned max_end_wf = 0;

            if (is_first_section) {
                // Special rule for the first section: anchor must start at WF 0.
                for (const auto& [key, span] : group_spans) {
                    if (assigned_groups.count(key)) continue;
                    if (span.min_wf == 0) {
                        if (anchor_key == nullptr || span.max_wf > max_end_wf) {
                            max_end_wf = span.max_wf;
                            anchor_key = &key;
                        }
                    }
                }
                is_first_section = false;
            } else {
                // General rule: anchor must cross the current start wavefront.
                for (const auto& [key, span] : group_spans) {
                    if (assigned_groups.count(key)) continue;
                    if (span.min_wf <= current_start_wf && span.max_wf >= current_start_wf) {
                        if (anchor_key == nullptr || span.max_wf > max_end_wf) {
                            max_end_wf = span.max_wf;
                            anchor_key = &key;
                        }
                    }
                }
            }

            // If no anchor found, there might be a gap. Find the next unassigned group.
            if (anchor_key == nullptr) {
                unsigned next_min_start_wf = std::numeric_limits<unsigned>::max();
                for (const auto& [key, span] : group_spans) {
                    if (assigned_groups.count(key) == 0) {
                        next_min_start_wf = std::min(next_min_start_wf, span.min_wf);
                    }
                }

                if (next_min_start_wf == std::numeric_limits<unsigned>::max()) {
                    break;
                }
                current_start_wf = next_min_start_wf;
                is_first_section = false; // No longer the "first" section in the strict sense
                continue; // Restart the loop with the new start wavefront.
            }

            unsigned section_end_wf = max_end_wf;
            std::cout << "Defining new section from WF " << current_start_wf << " to " << section_end_wf << std::endl;

            // --- Collect subgraphs for this section ---
            std::vector<std::vector<VertexType>> current_section_components;
            
            // The anchor group is always included.
            for (const auto* sg_ptr : layout.at(*anchor_key)) {
                current_section_components.push_back(sg_ptr->vertices);
            }
            assigned_groups.insert(*anchor_key);

            // Add any other groups that fit entirely in the new section's range.
            for (const auto& [key, span] : group_spans) {
                if (assigned_groups.count(key)) continue;

                if (span.min_wf >= current_start_wf && span.max_wf <= section_end_wf) {
                    for (const auto* sg_ptr : layout.at(key)) {
                        current_section_components.push_back(sg_ptr->vertices);
                    }
                    assigned_groups.insert(key);
                }
            }
            
            if (!current_section_components.empty()) {
                sections.push_back(std::move(current_section_components));
            }

            // --- Prepare for next section ---
            current_start_wf = section_end_wf + 1;
        }

        return sections;
    }

    void print_orbit_summary(const std::unordered_map<size_t, std::vector<VertexType>>& orbits) {
        std::cout << "\n--- 📊 Found " << orbits.size() << " orbits in this wavefront ---\n";
        for (const auto& [hash, vertices] : orbits) {
            std::cout << "  - Orbit " << hash << " (Size: " << vertices.size() << ")\n";
        }
    }
    
    void print_active_groups_summary(unsigned wf_idx) {
        std::cout << "\n--- 📦 Active Groups Summary (End of WF " << wf_idx << ") ---\n";
        std::cout << "Total active groups: " << isomorphic_groups_.size() << "\n";
        for (const auto& [key, sg_ids] : isomorphic_groups_) {
            std::cout << "  - Group (H:" << key.first << "/F:" << key.second << ") (Size: " << sg_ids.size() << "):\n";
            if (!sg_ids.empty() && active_subgraphs_.count(sg_ids.front())) {
                 const auto& sg = active_subgraphs_.at(sg_ids.front());
                 std::cout << "    - Rep. Subgraph " << sg_ids.front() << " (nodes: " << sg.vertices.size() 
                           << ", span: " << sg.start_wavefront << "-" << sg.end_wavefront << ")\n";
            }
        }
        std::cout << "-------------------------------------------\n";
    }

    void print_final_layout_summary() {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "📜 Final Layout Summary\n";
        std::cout << std::string(80, '=') << std::endl;

        if (finalized_subgraphs_.empty()) {
            std::cout << "No subgraphs were finalized.\n";
            std::cout << std::string(80, '=') << std::endl;
            return;
        }

        std::unordered_map<GroupKey, std::vector<const Subgraph*>, GroupKeyHash> layout;
        for (const auto& sg : finalized_subgraphs_) {
            layout[{sg.current_hash, sg.family_id}].push_back(&sg);
        }

        std::cout << std::left
                  << std::setw(22) << "Isomorphism Hash"
                  << "| " << std::setw(8) << "Family"
                  << "| " << std::setw(10) << "# Subgraphs"
                  << "| " << std::setw(15) << "Total Work"
                  << "| " << std::setw(15) << "Total Memory"
                  << "| " << "Wavefront Span" << std::endl;
        std::cout << std::string(92, '-') << std::endl;

        for (const auto& [key, group] : layout) {
            v_workw_t<Graph_t> total_work = 0;
            v_memw_t<Graph_t> total_mem = 0;
            unsigned min_wf = std::numeric_limits<unsigned>::max();
            unsigned max_wf = 0;

            for (const auto* sg : group) {
                total_work += sg->work_weight;
                total_mem += sg->memory_weight;
                if (sg->start_wavefront < min_wf) min_wf = sg->start_wavefront;
                if (sg->end_wavefront > max_wf) max_wf = sg->end_wavefront;
            }
            
            std::string wf_span = std::to_string(min_wf) + " - " + std::to_string(max_wf);

            std::cout << std::left
                      << std::setw(22) << key.first
                      << "| " << std::setw(8) << key.second
                      << "| " << std::setw(10) << group.size()
                      << "| " << std::setw(15) << total_work
                      << "| " << std::setw(15) << total_mem
                      << "| " << wf_span << std::endl;
        }
        std::cout << std::string(92, '=') << std::endl;
    }
};

} // namespace osp
