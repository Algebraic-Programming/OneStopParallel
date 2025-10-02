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
#include "osp/auxiliary/datastructures/union_find.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/dag_divider/DagDivider.hpp"
#include "osp/coarser/coarser_util.hpp"

namespace osp {

// The subgraph struct remains largely the same. It's a container for a
// growing, structurally-coherent component of the DAG.
template<typename Graph_t> 
struct subgraph {
    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> vertices;
    size_t current_hash = 0; // Hash of the orbit this subgraph currently belongs to
    unsigned family_id = 0;

    v_workw_t<Graph_t> work_weight = 0;
    v_memw_t<Graph_t> memory_weight = 0;
    
    unsigned start_wavefront = 0;
    unsigned end_wavefront = 0;

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
template<typename Graph_t, typename node_hash_func_t = uniform_node_hash_func<vertex_idx_t<Graph_t>>>
class WavefrontOrbitProcessor {
    static_assert(is_computational_dag_v<Graph_t>,
                  "IsomorphicComponentDivider can only be used with computational DAGs.");

public:
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

private:

    static constexpr bool verbose = false;

    using VertexType = vertex_idx_t<Graph_t>;
    using Subgraph = subgraph<Graph_t>;
    using MerkleHashComputer_t = MerkleHashComputer<Graph_t, bwd_merkle_node_hash_func<Graph_t>, true>; //MerkleHashComputer<Graph_t, node_hash_func_t, true>;
    using InternalConstrGraph_t = Graph_t;

    // Represents a node in the family lineage tree.
    struct FamilyTreeNode {
        unsigned id;
        unsigned parent_id;
        std::vector<unsigned> children_ids;
        unsigned creation_wavefront;
    };

    // --- Algorithm Parameters ---
    // If a merge event causes an isomorphic group to shrink below this size,
    // the parent group is broken, and a new group is started.
    size_t min_iso_group_size_threshold_;
   
    // --- Algorithm State ---
    const Graph_t* dag_;
    unsigned next_subgraph_id_ = 0;
    unsigned next_family_id_ = 0;
    std::unordered_map<unsigned, FamilyTreeNode> family_tree_;

    std::unordered_map<unsigned, Subgraph> active_subgraphs_;
    std::vector<Subgraph> finalized_subgraphs_;
    std::vector<unsigned> vertex_to_subgraph_id_;

    // Maps the ID of a finalized subgraph to its family_id.
    std::unordered_map<unsigned, unsigned> finalized_sg_id_to_family_id_;

    // Groups active subgraphs by their isomorphism hash and family ID.
    std::unordered_map<GroupKey, std::vector<unsigned>, GroupKeyHash> isomorphic_groups_;

    // Enum to track the fate of a parent group during a wavefront transition.
    enum class GroupStatus { UNCHANGED, CONTINUED, BROKEN };

    // For each vertex in an orbit, list its parent subgraph IDs.
    using VertexParentConnections = std::unordered_map<VertexType, std::set<unsigned>>;

    // For an orbit, map each parent group to the specific subgraphs within that group that it connects to.
    using OrbitParentGroupConnections = std::map<GroupKey, std::set<unsigned>>;

    // Holds all connectivity information and the final decision for a single orbit.
    struct OrbitAnalysis {
        const std::vector<VertexType>* vertices;
        size_t hash;

        // --- Connectivity Information ---
        // Maps parent group keys to the specific parent subgraph IDs from that group.
        OrbitParentGroupConnections parent_groups;
        // Maps each vertex in this orbit to its direct parent subgraph IDs.
        VertexParentConnections vertex_parents;
        // Stores the family IDs of any finalized parent subgraphs.
        std::set<unsigned> finalized_parent_families;

        // --- Decision ---
        enum class Fate { UNCLASSIFIED, CONTINUE, BREAK, COMPLEX_MERGE, NEW };
        Fate fate = Fate::UNCLASSIFIED;

        // If fate is CONTINUE or BREAK, this is the key of the single parent group involved.
        GroupKey target_parent_group;
    };

    // Tracks how each parent group will evolve based on its child orbits.
    struct ParentGroupEvolution {
        // Orbits that will continue this group.
        std::vector<const OrbitAnalysis*> continuing_orbits;
        // Orbits that break away from this group to start their own.
        std::vector<const OrbitAnalysis*> breaking_orbits;
    };

public:
    explicit WavefrontOrbitProcessor(size_t min_iso_group_size_threshold = 4)
        : min_iso_group_size_threshold_(min_iso_group_size_threshold) {}


    void discover_isomorphic_groups(const Graph_t &dag) {       
        std::vector<std::vector<VertexType>> level_sets = compute_wavefronts(dag);
        process_wavefronts(dag, level_sets);
    }

    inline const std::vector<unsigned> get_vertex_color_map() const { return vertex_to_subgraph_id_; }
    inline const std::vector<Subgraph>& get_finalized_subgraphs() const { return finalized_subgraphs_; }
    inline std::vector<std::vector<unsigned>> get_isomorphic_groups() { return build_isomorphic_groups_from_finalized(); }

private:

    void process_wavefronts(const Graph_t &dag, const std::vector<std::vector<VertexType>> & level_sets) {
        dag_ = &dag;
        next_subgraph_id_ = 0;
        next_family_id_ = 0;
        active_subgraphs_.clear();
        family_tree_.clear();
        finalized_subgraphs_.clear();
        vertex_to_subgraph_id_.assign(dag.num_vertices(), std::numeric_limits<unsigned>::max());
        finalized_sg_id_to_family_id_.clear();
        isomorphic_groups_.clear();
       
        MerkleHashComputer_t m_fw_hash(dag, dag);    

        if constexpr (verbose) {
            print_orbit_wavefront_summary(dag, level_sets, m_fw_hash);
        }

        // --- 3. Main Loop: Process Wavefronts ---
        for (unsigned wf_idx = 0; wf_idx < level_sets.size(); ++wf_idx) {
            const auto& level = level_sets[wf_idx];
            if constexpr (verbose) {
                std::cout << "\n" << std::string(80, '=') << "\n";
                std::cout << "🌊 Processing Wavefront " << wf_idx << " (Size: " << level.size() << ")\n";
                std::cout << std::string(80, '=') << std::endl;
            }

            // Group vertices in the current wavefront by their forward Merkle hash.
            std::unordered_map<size_t, std::vector<VertexType>> orbits = get_orbits(level, m_fw_hash);
            if constexpr (verbose) {
                print_orbit_summary(orbits);
            }

            if (wf_idx == 0) {
                // For the first wavefront (sources), all vertices start new subgraphs.
                for (const auto& [hash, vertices] : orbits) {
                    // All vertices in the same orbit in WF0 get the same new family ID.
                    add_family_node(next_family_id_, std::numeric_limits<unsigned>::max(), wf_idx);
                    const unsigned new_family_id = next_family_id_++;
                    for (const auto& v : vertices) {
                        create_new_subgraph(v, hash, wf_idx, new_family_id);
                    }
                }
                rebuild_isomorphic_groups();
                if constexpr (verbose) {
                    print_active_groups_summary(wf_idx);
                }
                continue;
            }            

            process_wavefront_orbits(orbits, wf_idx, m_fw_hash);
            
            if constexpr (verbose) {
                print_family_tree();
                print_active_groups_summary(wf_idx);
            }
        }

        // --- 4. Finalization ---
        // Any remaining active subgraphs are finalized at the end.
        if constexpr (verbose) {
            std::cout << "\n" << std::string(80, '=') << "\n";
            std::cout << "✅ Finalizing all remaining active subgraphs." << std::endl;
        }

        for (const auto& [id, sg] : active_subgraphs_) {
            finalize_subgraph(id, sg);
        }
 
        if constexpr (verbose) {
            std::cout << "Total finalized subgraphs: " << finalized_subgraphs_.size() << std::endl;

            print_final_layout_summary();
        }
    }


private:

    /**
     * @brief Checks if two families share a common ancestor in the lineage tree.
     *
     * This is useful for determining if two groups of subgraphs originated from the
     * same initial structure, even after multiple splits.
     *
     * @param family1_id The ID of the first family.
     * @param family2_id The ID of the second family.
     * @return True if they have a common ancestor, false otherwise.
     */
    bool share_common_ancestor(unsigned family1_id, unsigned family2_id) const {
        if (family1_id == family2_id) return true;
        return find_root_ancestor(family1_id) == find_root_ancestor(family2_id);
    }

    // Helper to find the root ancestor of a family.
    unsigned find_root_ancestor(unsigned family_id) const {
        if (family_tree_.find(family_id) == family_tree_.end()) {
            return family_id; // A family not in the tree is its own root.
        }
        unsigned current_id = family_id;
        while (family_tree_.at(current_id).parent_id != std::numeric_limits<unsigned>::max()) {
            current_id = family_tree_.at(current_id).parent_id;
            if (family_tree_.find(current_id) == family_tree_.end()) {
                 // This indicates a malformed tree, break to avoid infinite loop.
                 return current_id;
            }
        }
        return current_id;
    }

    // Checks if all families in a set share a single common ancestor (root).
    // Returns {true, common_root_id} or {false, max_unsigned}.
    std::pair<bool, unsigned> get_common_root(const std::set<unsigned>& family_ids) const {
        if (family_ids.empty()) {
            return {true, std::numeric_limits<unsigned>::max()};
        }

        unsigned first_root = find_root_ancestor(*family_ids.begin());
        for (auto it = std::next(family_ids.begin()); it != family_ids.end(); ++it) {
            if (find_root_ancestor(*it) != first_root) {
                return {false, std::numeric_limits<unsigned>::max()};
            }
        }
        return {true, first_root};
    }

private:

    void process_wavefront_orbits(const std::unordered_map<size_t, std::vector<VertexType>>& orbits, unsigned wf_idx, MerkleHashComputer_t& hasher) {        
        std::unordered_map<unsigned, Subgraph> next_active_subgraphs;
        std::set<unsigned> consumed_sg_ids;

        if constexpr (verbose) {
            std::cout << "\n--- ⚖️ Phase 1: Analyze & Decide ---\n";
        }

        // This map will own the analysis objects for this wavefront.
        std::unordered_map<size_t, OrbitAnalysis> analysis_storage;        
        std::map<GroupKey, ParentGroupEvolution> parent_evolutions;
        std::vector<const OrbitAnalysis*> new_orbits_list;
        std::vector<const OrbitAnalysis*> complex_orbits;
        
        analyze_and_classify_orbits(orbits, parent_evolutions, new_orbits_list, complex_orbits, analysis_storage);

        if constexpr (verbose) {
            std::cout << "\n--- 🚀 Phase 2: Execute Actions ---\n";
        }

        // --- Step 1: Execute simple continuations, breaks, and new orbits ---
        for (auto& [parent_key, evolution] : parent_evolutions) {            
            if (!evolution.continuing_orbits.empty()) {
                if (evolution.continuing_orbits.size() == 1) {
                    handle_simple_continuation(parent_key, *evolution.continuing_orbits[0], wf_idx, next_active_subgraphs, consumed_sg_ids);
                } else {
                    handle_multi_orbit_continuation(parent_key, evolution, wf_idx, next_active_subgraphs, consumed_sg_ids);
                }
            }

            if (!evolution.breaking_orbits.empty()) {
                // A group's lineage is only inherited by a single breaking orbit if there are no other continuing orbits.
                // Otherwise, any breaking orbit is considered a split and starts a new family.
                bool inherit_family = evolution.continuing_orbits.empty() && evolution.breaking_orbits.size() == 1;

                for (const auto* analysis : evolution.breaking_orbits) {
                    unsigned family_id_for_break;
                    if (inherit_family) {
                        family_id_for_break = parent_key.second;
                        if constexpr (verbose) {
                            std::cout << "    -> 💥 Breaking off with orbit " << analysis->hash << " (starting new group, inheriting Family " << family_id_for_break << ").\n";
                        }
                    } else {
                        family_id_for_break = next_family_id_++;
                        add_family_node(family_id_for_break, parent_key.second, wf_idx);
                        if constexpr (verbose) {
                            std::cout << "    -> 🔱 Breaking off with orbit " << analysis->hash << " (SPLITTING into new Family " << family_id_for_break << ").\n";
                        }
                    }
                    start_new_group(analysis->hash, *analysis->vertices, wf_idx, family_id_for_break, next_active_subgraphs);
                }
            }
        }

        for (const auto* analysis : new_orbits_list) {
            unsigned new_family_id = next_family_id_++;
            add_family_node(new_family_id, std::numeric_limits<unsigned>::max(), wf_idx);
            if constexpr (verbose) {
                std::cout << "  - Orbit " << analysis->hash << " has no active parents. Starting new group (Family " << new_family_id << ").\n";
            }
            start_new_group(analysis->hash, *analysis->vertices, wf_idx, new_family_id, next_active_subgraphs);
        }
        
        // --- Step 2: Handle complex merges ---
        if (!complex_orbits.empty()) {
            handle_complex_merges(complex_orbits, wf_idx, next_active_subgraphs, consumed_sg_ids, hasher);
        }

        // --- Phase 3: Finalize Old Groups ---
        if constexpr (verbose) {
            std::cout << "\n--- 📋 Phase 3: Finalizing Old Groups ---\n";
        }
        // Any active subgraph that was not consumed (i.e., its lineage was not continued or broken) must be finalized.
        for (const auto& [id, sg] : active_subgraphs_) {
            if (consumed_sg_ids.find(id) == consumed_sg_ids.end()) {
                if constexpr (verbose) {
                    std::cout << "  - Subgraph " << id << " (H:" << sg.current_hash << "/F:" << sg.family_id << ") had no children. Finalizing.\n";
                }
                finalize_subgraph(id, sg);
            }
        }

        // Update state for the next wavefront
        active_subgraphs_ = std::move(next_active_subgraphs);
        rebuild_isomorphic_groups();
    }

    void handle_complex_merges(
        const std::vector<const OrbitAnalysis*>& complex_orbits,
        unsigned wf_idx,
        std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
        std::set<unsigned>& consumed_sg_ids,
        MerkleHashComputer_t& hasher)
    {
        if constexpr (verbose) {
            std::cout << "  - Handling " << complex_orbits.size() << " complex orbit(s) by re-analyzing connectivity.\n";
        }

        // 1. Re-analyze connectivity for each complex orbit to find its *currently active* parents.
        std::unordered_map<size_t, OrbitParentGroupConnections> live_parent_connections;
        std::unordered_map<size_t, std::set<unsigned>> live_finalized_parent_families;

        for (const auto* analysis : complex_orbits) {
            OrbitParentGroupConnections current_live_parents;
            std::set<unsigned> current_finalized_families;

            for (const auto& v : *analysis->vertices) {
                for (const auto& p : dag_->parents(v)) {
                    unsigned p_sg_id = vertex_to_subgraph_id_[p];
                    assert(p_sg_id != std::numeric_limits<unsigned>::max());

                    // Check if the parent is in a newly created subgraph for this wavefront
                    if (next_active_subgraphs.count(p_sg_id)) {
                        const auto& parent_sg = next_active_subgraphs.at(p_sg_id);
                        GroupKey parent_key = {parent_sg.current_hash, parent_sg.family_id};
                        current_live_parents[parent_key].insert(p_sg_id);
                    }
                    // Check if the parent is in an active, unconsumed subgraph from the previous wavefront
                    else if (active_subgraphs_.count(p_sg_id) && consumed_sg_ids.find(p_sg_id) == consumed_sg_ids.end()) {
                        const auto& parent_sg = active_subgraphs_.at(p_sg_id);
                        GroupKey parent_key = {parent_sg.current_hash, parent_sg.family_id};
                        current_live_parents[parent_key].insert(p_sg_id);
                    }
                    // Check if the parent is in an already finalized subgraph (this info is static and correct)
                    else if (finalized_sg_id_to_family_id_.count(p_sg_id)) {
                        current_finalized_families.insert(finalized_sg_id_to_family_id_.at(p_sg_id));
                    }
                }
            }
            live_parent_connections[analysis->hash] = std::move(current_live_parents);
            live_finalized_parent_families[analysis->hash] = std::move(current_finalized_families);
        }

        // 2. Group complex orbits by shared *live* parent groups AND finalized families using Union-Find
        Union_Find_Universe<size_t, size_t, int, int> uf;
        std::unordered_map<GroupKey, std::vector<size_t>, GroupKeyHash> parent_to_orbits;
        std::unordered_map<unsigned, std::vector<size_t>> finalized_family_to_orbits;

        for (const auto* analysis : complex_orbits) {
            uf.add_object(analysis->hash);
            const auto& live_parents = live_parent_connections.at(analysis->hash);
            for (const auto& [pkey, sgs] : live_parents) {
                parent_to_orbits[pkey].push_back(analysis->hash);
            }
            const auto& finalized_fams = live_finalized_parent_families.at(analysis->hash);
            for (unsigned fam_id : finalized_fams) {
                finalized_family_to_orbits[fam_id].push_back(analysis->hash);
            }
        }

        // Join by shared live parents
        for (const auto& [pkey, hashes] : parent_to_orbits) {
            if (hashes.size() > 1) {
                for (size_t i = 1; i < hashes.size(); ++i) {
                    uf.join_by_name(hashes[0], hashes[i]);
                }
            }
        }

        // Join by shared finalized families
        for (const auto& [fam_id, hashes] : finalized_family_to_orbits) {
            if (hashes.size() > 1) {
                for (size_t i = 1; i < hashes.size(); ++i) {
                    uf.join_by_name(hashes[0], hashes[i]);
                }
            }
        }

        auto components = uf.get_connected_components();
        if constexpr (verbose) {
            std::cout << "    -> Grouped into " << components.size() << " complex merge region(s) based on live and finalized parents.\n";
        }

        // 3. Process each component of complex orbits
        std::unordered_map<size_t, const OrbitAnalysis*> hash_to_analysis;
        for (const auto* analysis : complex_orbits) {
            hash_to_analysis[analysis->hash] = analysis;
        }

        for (const auto& component : components) {
            if constexpr (verbose) {
                std::cout << "      - Processing complex region with " << component.size() << " orbit(s).\n";
            }

            // A. Collect all unique parent families and subgraphs for this component
            std::set<unsigned> all_parent_families;
            std::set<unsigned> all_parent_sg_ids;
            for (size_t orbit_hash : component) {
                const auto& live_parents = live_parent_connections.at(orbit_hash);
                for (const auto& [pkey, sgs] : live_parents) {
                    all_parent_families.insert(pkey.second);
                    all_parent_sg_ids.insert(sgs.begin(), sgs.end());
                }
                const auto& finalized_fams = live_finalized_parent_families.at(orbit_hash);
                all_parent_families.insert(finalized_fams.begin(), finalized_fams.end());
            }

            // B. Decide fate based on ancestry
            auto [has_common_root, root_id] = get_common_root(all_parent_families);

            if (has_common_root) {
                // CASE A: Re-joining related branches (or trivial case with <=1 parent family).
                // Here, we finalize the parents and start a new family that is a child of the common ancestor.
                if constexpr (verbose) {
                    std::cout << "        - Complex merge joins branches of a common ancestor. Finalizing parents and creating new family.\n";
                }

                if (!all_parent_sg_ids.empty()) {
                    if constexpr (verbose) {
                        std::cout << "        - Consuming and finalizing " << all_parent_sg_ids.size() << " parent subgraphs involved in this merge.\n";
                    }
                    for (unsigned sg_id : all_parent_sg_ids) {
                        if (next_active_subgraphs.count(sg_id)) {
                            finalize_subgraph(sg_id, next_active_subgraphs.at(sg_id));
                            next_active_subgraphs.erase(sg_id);
                        } else if (active_subgraphs_.count(sg_id)) {
                            finalize_subgraph(sg_id, active_subgraphs_.at(sg_id));
                            consumed_sg_ids.insert(sg_id);
                        }
                    }
                }

                unsigned parent_for_tree = all_parent_families.empty() ? std::numeric_limits<unsigned>::max() : root_id;

                for (size_t orbit_hash : component) {
                    const auto* analysis = hash_to_analysis.at(orbit_hash);
                    
                    // Create a new family for EACH orbit.
                    unsigned new_family_id = next_family_id_++;
                    add_family_node(new_family_id, parent_for_tree, wf_idx);
                    if constexpr (verbose) {
                        std::cout << "        - Orbit " << orbit_hash << " starts new group in new Family " << new_family_id 
                                  << " (parent: " << (parent_for_tree == std::numeric_limits<unsigned>::max() ? "none" : std::to_string(parent_for_tree)) << ").\n";
                    }
                    start_new_group(analysis->hash, *analysis->vertices, wf_idx, new_family_id, next_active_subgraphs);
                }
            } else {
                // CASE B: Merging unrelated branches. We must check if a true merge is viable.
                if constexpr (verbose) {
                    std::cout << "        - Complex merge joins distinct family trees. Checking for merge viability...\n";
                }

                // 1. Collect all vertices and build a map of all potential parent subgraphs for this component.
                std::vector<VertexType> component_vertices;
                for (size_t orbit_hash : component) {
                    const auto* analysis = hash_to_analysis.at(orbit_hash);
                    component_vertices.insert(component_vertices.end(), analysis->vertices->begin(), analysis->vertices->end());
                }

                std::unordered_map<unsigned, Subgraph> all_potential_parents;
                for (unsigned sg_id : all_parent_sg_ids) {
                    if (next_active_subgraphs.count(sg_id)) {
                        all_potential_parents.emplace(sg_id, next_active_subgraphs.at(sg_id));
                    } else if (active_subgraphs_.count(sg_id)) {
                        all_potential_parents.emplace(sg_id, active_subgraphs_.at(sg_id));
                    }
                }

                // 2. Group child vertices by their connection to the parent set to determine the resulting structure.
                std::map<std::set<unsigned>, std::vector<VertexType>> parent_set_to_children;
                for (const auto& v : component_vertices) {
                    std::set<unsigned> parents_from_set;
                    for (const auto& p_node : dag_->parents(v)) {
                        unsigned p_sg_id = vertex_to_subgraph_id_[p_node];
                        if (all_parent_sg_ids.count(p_sg_id)) {
                            parents_from_set.insert(p_sg_id);
                        }
                    }
                    if (!parents_from_set.empty()) {
                        parent_set_to_children[parents_from_set].push_back(v);
                    }
                }

                // 3. Check viability: if the resulting group is not viable, perform a full break.
                const size_t new_size = parent_set_to_children.size();
                const size_t current_size = all_parent_sg_ids.size();
                const size_t orbit_size = component_vertices.size();

                bool parent_sets_overlap = false;
                if (parent_set_to_children.size() > 1) {
                    std::set<unsigned> seen_parents;
                    size_t total_parents_in_sets = 0;
                    for (const auto& [pset, children] : parent_set_to_children) {
                        total_parents_in_sets += pset.size();
                        seen_parents.insert(pset.begin(), pset.end());
                    }
                    if (seen_parents.size() < total_parents_in_sets) {
                        parent_sets_overlap = true;
                    }
                }

                bool is_viable = new_size >= min_iso_group_size_threshold_ || new_size >= current_size;
                if (orbit_size >= min_iso_group_size_threshold_ && current_size < min_iso_group_size_threshold_) {
                    is_viable = false;
                }

                if constexpr (verbose) {
                    std::cout << "        - Viability check: merging " << current_size << " parents with "
                              << component.size() << " orbits would result in " << new_size << " new subgraphs.\n";
                }

                if ((!is_viable || parent_sets_overlap) && !parent_set_to_children.empty()) {
                    if constexpr (verbose) {
                        std::cout << "        - Merge not viable or parent sets overlap. Opting for full break.\n";
                    }
                    // Perform the "full break" action: finalize parents and start new groups.
                    if (!all_parent_sg_ids.empty()) {
                        for (unsigned sg_id : all_parent_sg_ids) {
                            if (next_active_subgraphs.count(sg_id)) {
                                finalize_subgraph(sg_id, next_active_subgraphs.at(sg_id));
                                next_active_subgraphs.erase(sg_id);
                            } else if (active_subgraphs_.count(sg_id)) {
                                finalize_subgraph(sg_id, active_subgraphs_.at(sg_id));
                                consumed_sg_ids.insert(sg_id);
                            }
                        }
                    }
                    for (size_t orbit_hash : component) {
                        const auto* analysis = hash_to_analysis.at(orbit_hash);
                        // Create a new root family for EACH orbit.
                        unsigned new_family_id = next_family_id_++;
                        add_family_node(new_family_id, std::numeric_limits<unsigned>::max(), wf_idx);
                        if constexpr (verbose) {
                            std::cout << "        - Orbit " << orbit_hash << " starts new group in new root Family " << new_family_id << ".\n";
                        }
                        start_new_group(analysis->hash, *analysis->vertices, wf_idx, new_family_id, next_active_subgraphs);
                    }
                } else {
                    if constexpr (verbose) {
                        std::cout << "        - Merge is viable. Performing full merge of unrelated families.\n";
                    }
                    // Perform a true merge.
                    perform_full_merge(component, all_parent_sg_ids, parent_set_to_children, all_potential_parents, hasher,
                                       live_parent_connections, wf_idx, next_active_subgraphs, consumed_sg_ids);
                }
            }
        }
    }

    void perform_full_merge(
        const std::vector<size_t>& component,
        const std::set<unsigned>& all_parent_sg_ids,
        const std::map<std::set<unsigned>, std::vector<VertexType>>& parent_set_to_children,
        const std::unordered_map<unsigned, Subgraph>& all_potential_parents,
        MerkleHashComputer_t& hasher,
        const std::unordered_map<size_t, OrbitParentGroupConnections>& live_parent_connections,
        unsigned wf_idx,
        std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
        std::set<unsigned>& consumed_sg_ids)
    {
        // a. Consume all parent subgraphs
        for (unsigned sg_id : all_parent_sg_ids) {
            if (next_active_subgraphs.count(sg_id)) {
                next_active_subgraphs.erase(sg_id);
            } else {
                consumed_sg_ids.insert(sg_id);
            }
        }

        // b. Create a new root family for the merged result
        unsigned new_family_id = next_family_id_++;
        add_family_node(new_family_id, std::numeric_limits<unsigned>::max(), wf_idx);
        if constexpr (verbose) {
            std::cout << "        - Creating new root Family " << new_family_id << " for the merged group.\n";
        }

        // c. Compute a new hash for the merged group from all involved parent and orbit hashes
        size_t merged_group_hash = 0;
        std::set<GroupKey> parent_keys_in_component;
        for (size_t orbit_hash : component) {
            hash_combine(merged_group_hash, orbit_hash);
            if (live_parent_connections.count(orbit_hash)) {
                const auto& live_parents = live_parent_connections.at(orbit_hash);
                for (const auto& [pkey, sgs] : live_parents) {
                    parent_keys_in_component.insert(pkey);
                }
            }
        }
        for (const auto& pkey : parent_keys_in_component) {
            hash_combine(merged_group_hash, pkey.first);
        }

        // d. Create the new merged subgraphs
        for (const auto& [parent_set, children] : parent_set_to_children) {
            if (children.empty()) continue;
            
            size_t new_hash = 0;
            // Combine parent hashes for multiplicity. parent_set is a std::set, so iteration is sorted.
            for (unsigned p_id : parent_set) {
                hash_combine(new_hash, all_potential_parents.at(p_id).current_hash);
            }
            // Combine orbit hash for each child.
            for (const auto& v : children) {
                hash_combine(new_hash, hasher.get_vertex_hash(v));
            }
            Subgraph new_sg = create_merged_subgraph(parent_set, children, all_potential_parents, new_hash, new_family_id, wf_idx);
            unsigned new_id = next_subgraph_id_++;
            next_active_subgraphs.emplace(new_id, std::move(new_sg));
            for (const auto& v : next_active_subgraphs.at(new_id).vertices) {
                vertex_to_subgraph_id_[v] = new_id;
            }
        }
    }

    std::map<std::set<unsigned>, std::vector<VertexType>>
    group_children_by_parent_set(const OrbitAnalysis& analysis, const GroupKey& parent_key) const {
        std::map<std::set<unsigned>, std::vector<VertexType>> parent_set_to_children;

        const auto& parent_group_members_vec = isomorphic_groups_.at(parent_key);
        const std::set<unsigned> parent_group_members(parent_group_members_vec.begin(), parent_group_members_vec.end());

        for (const auto& v : *analysis.vertices) {
            std::set<unsigned> parents_from_group;
            if (analysis.vertex_parents.count(v)) {
                const auto& vertex_parent_sgs = analysis.vertex_parents.at(v);
                std::set_intersection(vertex_parent_sgs.begin(), vertex_parent_sgs.end(),
                                      parent_group_members.begin(), parent_group_members.end(),
                                      std::inserter(parents_from_group, parents_from_group.begin()));
            }
            
            if (!parents_from_group.empty()) {
                parent_set_to_children[parents_from_group].push_back(v);
            }
        }
        return parent_set_to_children;
    }

    bool is_viable_continuation(const OrbitAnalysis& analysis, const GroupKey& parent_key) const {
        if (analysis.vertices->empty()) return false;

        auto parent_set_to_children = group_children_by_parent_set(analysis, parent_key);
        // The number of new subgraphs is the number of unique parent sets found.
        size_t new_size = parent_set_to_children.size();
        const size_t current_size = isomorphic_groups_.at(parent_key).size();
        const size_t orbit_size = analysis.vertices->size();

        bool is_viable = new_size >= min_iso_group_size_threshold_ || new_size >= current_size;

        // New condition: if a large orbit can be formed, and it comes from a small
        // parent group, it's better to break off and start a new large group to
        // exploit the parallelism, rather than continuing the small group.
        if (orbit_size >= min_iso_group_size_threshold_ && current_size < min_iso_group_size_threshold_) {
            return false;
        }

        return is_viable;
    }

    void analyze_and_classify_orbits(
        const std::unordered_map<size_t, std::vector<VertexType>>& orbits,
        std::map<GroupKey, ParentGroupEvolution>& parent_evolutions,
        std::vector<const OrbitAnalysis*>& new_orbits,
        std::vector<const OrbitAnalysis*>& complex_orbits,
        std::unordered_map<size_t, OrbitAnalysis>& analysis_storage)
    {
        // 1. Gather raw connectivity data for each orbit.
        for (const auto& [hash, vertices] : orbits) {
            OrbitAnalysis analysis;
            analysis.hash = hash;
            analysis.vertices = &vertices;

            for (const auto& v : vertices) {
                for (const auto& p : dag_->parents(v)) {
                    unsigned p_sg_id = vertex_to_subgraph_id_[p];
                    if (p_sg_id == std::numeric_limits<unsigned>::max()) continue;

                    analysis.vertex_parents[v].insert(p_sg_id);

                    if (active_subgraphs_.count(p_sg_id)) {
                        const auto& parent_sg = active_subgraphs_.at(p_sg_id);
                        GroupKey parent_key = {parent_sg.current_hash, parent_sg.family_id};
                        analysis.parent_groups[parent_key].insert(p_sg_id);
                    } else if (finalized_sg_id_to_family_id_.count(p_sg_id)) {
                        analysis.finalized_parent_families.insert(finalized_sg_id_to_family_id_.at(p_sg_id));
                    }
                }
            }
            analysis_storage[hash] = std::move(analysis);
        }

        // 2. Classify orbits and populate evolution tracking structures.
        for (auto& [hash, analysis] : analysis_storage) {
            if (analysis.parent_groups.empty()) {
                analysis.fate = OrbitAnalysis::Fate::NEW;
                new_orbits.push_back(&analysis);
            } else if (analysis.parent_groups.size() == 1) {
                const auto& parent_key = analysis.parent_groups.begin()->first;
                const bool is_simple = analysis.finalized_parent_families.empty() ||
                                 (analysis.finalized_parent_families.count(parent_key.second) &&
                                  analysis.finalized_parent_families.size() == 1);

                if (is_simple) {
                    analysis.target_parent_group = parent_key;
                    if (is_viable_continuation(analysis, parent_key)) {
                        analysis.fate = OrbitAnalysis::Fate::CONTINUE;
                        parent_evolutions[parent_key].continuing_orbits.push_back(&analysis);
                    } else {
                        analysis.fate = OrbitAnalysis::Fate::BREAK;
                        parent_evolutions[parent_key].breaking_orbits.push_back(&analysis);
                    }
                } else {
                    analysis.fate = OrbitAnalysis::Fate::COMPLEX_MERGE;
                }
            } else { // More than one active parent group
                analysis.fate = OrbitAnalysis::Fate::COMPLEX_MERGE;
            }

            if (analysis.fate == OrbitAnalysis::Fate::COMPLEX_MERGE) {
                complex_orbits.push_back(&analysis);
            }
        }

        if constexpr (verbose)
            print_analysis_summary(analysis_storage);
    }

    void print_analysis_summary(
        const std::unordered_map<size_t, OrbitAnalysis>& analysis_storage) const
    {
            std::cout << "\n--- 📊 Orbit Analysis Summary ---\n";
            for (const auto& [hash, analysis] : analysis_storage) {
                std::cout << "  - Orbit " << hash << " (size: " << analysis.vertices->size() << "): ";
                switch (analysis.fate) {
                    case OrbitAnalysis::Fate::CONTINUE:
                        std::cout << "CONTINUE Group (H:" << analysis.target_parent_group.first 
                                  << "/F:" << analysis.target_parent_group.second << ")\n";
                        break;
                    case OrbitAnalysis::Fate::BREAK:
                        std::cout << "BREAK from Group (H:" << analysis.target_parent_group.first 
                                  << "/F:" << analysis.target_parent_group.second << ")\n";
                        break;
                    case OrbitAnalysis::Fate::NEW:
                        std::cout << "NEW group\n";
                        if (!analysis.finalized_parent_families.empty()) {
                            std::cout << "    - Connects to finalized families: ";
                            for (unsigned fam_id : analysis.finalized_parent_families) {
                                std::cout << fam_id << " ";
                            }
                            std::cout << "\n";
                        }
                        break;
                    case OrbitAnalysis::Fate::COMPLEX_MERGE:
                        std::cout << "COMPLEX_MERGE involving groups:\n";
                        for (const auto& [pkey, sgs] : analysis.parent_groups) {
                            std::cout << "    - Group (H:" << pkey.first << "/F:" << pkey.second << ")\n";
                        }
                        if (!analysis.finalized_parent_families.empty()) {
                            std::cout << "    - Also connects to finalized families: ";
                            for (unsigned fam_id : analysis.finalized_parent_families) {
                                std::cout << fam_id << " ";
                            }
                            std::cout << "\n";
                        }
                        break;
                    case OrbitAnalysis::Fate::UNCLASSIFIED:
                        std::cout << "UNCLASSIFIED\n";
                        break;
                }
            }
            std::cout << "--------------------------------\n";
    }

    void sequentially_merge_orbits(
        const std::vector<unsigned>& initial_parent_ids,
        std::vector<const OrbitAnalysis*>& orbits,
        unsigned family_id,
        unsigned wf_idx,
        std::unordered_map<unsigned, Subgraph>& next_active_subgraphs)
    {
        if constexpr (verbose) {
            std::cout << "      -> Sequentially merging " << orbits.size() << " orbits into " << initial_parent_ids.size() << " parents (Family " << family_id << ").\n";
        }

        // 1. Sort orbits by simplicity (parent degree). Since all vertices in an orbit are
        // structurally identical, we only need to check the first vertex.
        std::sort(orbits.begin(), orbits.end(),
            [this](const OrbitAnalysis* a, const OrbitAnalysis* b) {
                size_t degree_a = 0;
                if (!a->vertices->empty()) {
                    const auto& v_a = a->vertices->front();
                    if (a->vertex_parents.count(v_a)) degree_a = a->vertex_parents.at(v_a).size();
                }
                size_t degree_b = 0;
                if (!b->vertices->empty()) {
                    const auto& v_b = b->vertices->front();
                    if (b->vertex_parents.count(v_b)) degree_b = b->vertex_parents.at(v_b).size();
                }
                return degree_a < degree_b;
            });

        if constexpr (verbose) {
            std::cout << "         - Sorted orbit order (by simplicity): ";
            for (const auto* o : orbits) {
                std::cout << o->hash << " ";
            }
            std::cout << "\n";
        }

        // 2. Initial state: the subgraphs to evolve are the initial parents.
        std::unordered_map<unsigned, Subgraph> evolving_sgs;
        for (unsigned id : initial_parent_ids) {
            evolving_sgs.emplace(id, active_subgraphs_.at(id));
        }

        // 3. Sequentially merge each orbit
        for (const auto* orbit : orbits) {
            if constexpr (verbose) {
                std::cout << "         - Processing orbit " << orbit->hash << " (size " << orbit->vertices->size() << ") against " << evolving_sgs.size() << " evolving subgraphs.\n";
            }
            if (evolving_sgs.empty()) { // All parents were consumed by a break
                if constexpr (verbose) {
                    std::cout << "           - No evolving parents left. Starting new group for this orbit.\n";
                }
                start_new_group(orbit->hash, *orbit->vertices, wf_idx, family_id, next_active_subgraphs);
                continue;
            }

            // A. Re-analyze connectivity of this orbit against the current `evolving_sgs`
            std::unordered_map<VertexType, unsigned> temp_v_to_sg_id;
            for (const auto& [id, sg] : evolving_sgs) {
                for (const auto& v : sg.vertices) {
                    temp_v_to_sg_id[v] = id;
                }
            }

            std::map<std::set<unsigned>, std::vector<VertexType>> parent_set_to_children;
            for (const auto& v : *orbit->vertices) {
                std::set<unsigned> parents_from_evolving;
                for (const auto& p_node : dag_->parents(v)) {
                    if (temp_v_to_sg_id.count(p_node)) {
                        parents_from_evolving.insert(temp_v_to_sg_id.at(p_node));
                    }
                }
                if (!parents_from_evolving.empty()) {
                    parent_set_to_children[parents_from_evolving].push_back(v);
                }
            }

            if constexpr (verbose) {
                std::cout << "           - Re-analyzed connectivity: " << parent_set_to_children.size() << " unique parent sets found.\n";
                if (parent_set_to_children.size() < 10) { // Avoid spamming for large merges
                    for (const auto& [pset, children] : parent_set_to_children) {
                        std::cout << "             - Set of " << pset.size() << " parents connects to " << children.size() << " children.\n";
                    }
                }
            }

            // B. Check viability
            size_t new_size = parent_set_to_children.size();
            if (new_size < min_iso_group_size_threshold_ && new_size > 0 && new_size < evolving_sgs.size()) {
                unsigned new_family_for_break = next_family_id_++;
                add_family_node(new_family_for_break, family_id, wf_idx);
                if constexpr (verbose) {
                    std::cout << "           - Orbit " << orbit->hash << " would break the evolving group (new size " << new_size 
                              << "). Splitting into new Family " << new_family_for_break << ".\n";
                }
                start_new_group(orbit->hash, *orbit->vertices, wf_idx, new_family_for_break, next_active_subgraphs);
                continue; // This orbit breaks off, but the evolving group continues to the next orbit.
            }

            // C. Perform merge
            std::unordered_map<unsigned, Subgraph> next_evolving_sgs;
            std::set<unsigned> consumed_ids;
            size_t temp_sg_id_counter = 0; // Use temporary IDs for this merge step

            for (const auto& [parent_set, children] : parent_set_to_children) {
                if (children.empty()) continue;

                size_t new_hash = 0;
                const size_t parent_hash = evolving_sgs.at(*parent_set.begin()).current_hash;
                for (size_t i = 0; i < parent_set.size(); ++i) {
                    hash_combine(new_hash, parent_hash);
                }
                for (size_t i = 0; i < children.size(); ++i) {
                    hash_combine(new_hash, orbit->hash);
                }
                
                Subgraph new_sg = create_merged_subgraph(parent_set, children, evolving_sgs, new_hash, family_id, wf_idx);
                next_evolving_sgs.emplace(temp_sg_id_counter++, std::move(new_sg));
                consumed_ids.insert(parent_set.begin(), parent_set.end());
            }

            // size_t num_created_sgs = next_evolving_sgs.size();
            //size_t num_unconsumed = evolving_sgs.size() - consumed_ids.size();

            // D. Carry over unconsumed subgraphs
            for (const auto& [id, sg] : evolving_sgs) {
                if (consumed_ids.find(id) == consumed_ids.end()) {
                    next_evolving_sgs.emplace(temp_sg_id_counter++, sg);
                }
            }
            
            evolving_sgs = std::move(next_evolving_sgs);
            // if constexpr (verbose) {
            //     std::cout << "           - Merging... Consumed " << consumed_ids.size() << " parents, carried over " << num_unconsumed
            //               << ", created " << num_created_sgs << " new subgraphs.\n";
            //     std::cout << "           - State after merge: " << evolving_sgs.size() << " evolving subgraphs.\n";
            // }
        }

        // 4. Finalize: move the fully evolved subgraphs to the main `next_active_subgraphs` map
        if constexpr (verbose) {
            std::cout << "      -> Sequential merge complete. Finalizing " << evolving_sgs.size() << " subgraphs for Family " << family_id << ".\n";
        }
        for (auto& [_, sg] : evolving_sgs) {
            unsigned new_id = next_subgraph_id_++;
            for (const auto& v : sg.vertices) {
                vertex_to_subgraph_id_[v] = new_id;
            }
            next_active_subgraphs.emplace(new_id, std::move(sg));
        }
    }

    void handle_multi_orbit_continuation(
        const GroupKey& parent_key,
        ParentGroupEvolution& evolution,
        unsigned wf_idx,
        std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
        std::set<unsigned>& consumed_sg_ids)
    {
        if constexpr (verbose) {
            std::cout << "  - Handling multi-orbit continuation for Group (H:" << parent_key.first << "/F:" << parent_key.second << ") with "
                      << evolution.continuing_orbits.size() << " orbits.\n";
        }

        // 1. Check for splitting by grouping parents by the set of orbits that connect to them.
        const auto& parent_sg_ids_vec = isomorphic_groups_.at(parent_key);
        std::set<unsigned> parent_sg_ids(parent_sg_ids_vec.begin(), parent_sg_ids_vec.end());

        std::map<unsigned, std::set<size_t>> parent_to_orbit_hashes;
        for (const auto* orbit : evolution.continuing_orbits) {
            if (orbit->parent_groups.count(parent_key)) {
                for (unsigned sg_id : orbit->parent_groups.at(parent_key)) {
                    if (parent_sg_ids.count(sg_id)) { // Ensure it's from the correct group
                        parent_to_orbit_hashes[sg_id].insert(orbit->hash);
                    }
                }
            }
        }

        std::map<std::set<size_t>, std::vector<unsigned>> orbit_set_to_parents;
        for (const auto& [p_id, orbit_hashes] : parent_to_orbit_hashes) {
            if (!orbit_hashes.empty()) {
                orbit_set_to_parents[orbit_hashes].push_back(p_id);
            }
        }

        // Mark all original parents as consumed. They will be handled here.
        consumed_sg_ids.insert(parent_sg_ids.begin(), parent_sg_ids.end());

        if (orbit_set_to_parents.size() > 1) {
            // --- Handle Splitting ---
            if constexpr (verbose) {
                std::cout << "    -> 🔱 Group (H:" << parent_key.first << "/F:" << parent_key.second << ") is SPLIT by "
                          << evolution.continuing_orbits.size() << " orbits into " << orbit_set_to_parents.size() << " new families.\n";
            }

            for (const auto& [orbit_hashes, parent_subset_ids] : orbit_set_to_parents) {
                unsigned new_family_id = next_family_id_++;
                add_family_node(new_family_id, parent_key.second, wf_idx);
                if constexpr (verbose) {
                    std::cout << "      - Split part: " << parent_subset_ids.size() << " parents evolving with "
                              << orbit_hashes.size() << " orbits into new Family " << new_family_id << ".\n";

                    std::cout << "        - Parent SG IDs: ";
                    for(auto id : parent_subset_ids) {
                        std::cout << id << " ";
                    }
                    std::cout << "\n";
                    std::cout << "        - Orbit Hashes: ";
                    for(auto h : orbit_hashes) {
                        std::cout << h << " ";
                    }
                    std::cout << "\n";
                }

                std::vector<const OrbitAnalysis*> relevant_orbits;
                for (const auto* o : evolution.continuing_orbits) {
                    if (orbit_hashes.count(o->hash)) {
                        relevant_orbits.push_back(o);
                    }
                }
                
                if (!relevant_orbits.empty()) {
                    sequentially_merge_orbits(parent_subset_ids, relevant_orbits, new_family_id, wf_idx, next_active_subgraphs);
                }
            }
        } else {
            // --- Handle Non-Splitting Multi-Orbit Merge ---
            if constexpr (verbose) {
                std::cout << "  - Group (H:" << parent_key.first << "/F:" << parent_key.second << ") of size "   
                          << parent_sg_ids.size() << " is being CONTINUED--multiple--by " << evolution.continuing_orbits.size() << " orbits." << std::endl; 
                std::cout << "    - Orbit Hashes involved: ";
                for (const auto* orbit : evolution.continuing_orbits) {
                    std::cout << orbit->hash << " ";
                }
                std::cout << "\n";
            }
            
            sequentially_merge_orbits(parent_sg_ids_vec, evolution.continuing_orbits, parent_key.second, wf_idx, next_active_subgraphs);
        }
    }

    void handle_simple_continuation(
        const GroupKey& parent_key,
        const OrbitAnalysis& analysis,
        unsigned wf_idx,
        std::unordered_map<unsigned, Subgraph>& next_active_subgraphs,
        std::set<unsigned>& consumed_sg_ids)
    {
        if constexpr (verbose) {
            std::cout << "  - Group (H:" << parent_key.first << "/F:" << parent_key.second << ") of size "   
                      << isomorphic_groups_.at(parent_key).size() << " is being CONTINUED--simple--by one orbit of size " << analysis.vertices->size() << std::endl; 
        }

        auto just_consumed = merge_and_continue_group(parent_key, analysis, wf_idx, next_active_subgraphs);
        consumed_sg_ids.insert(just_consumed.begin(), just_consumed.end());
    }

    // Helper to create a new subgraph by merging a set of parent subgraphs and adding a set of child vertices.
    Subgraph create_merged_subgraph(
        const std::set<unsigned>& parent_ids,
        const std::vector<VertexType>& children,
        const std::unordered_map<unsigned, Subgraph>& parent_sg_map,
        size_t new_hash,
        unsigned family_id,
        unsigned wf_idx)
    {
        Subgraph new_sg(children.front(), new_hash, *dag_, wf_idx, family_id);
        unsigned min_start_wf = new_sg.start_wavefront;

        for (unsigned parent_id : parent_ids) {
            const auto& parent_sg = parent_sg_map.at(parent_id);
            new_sg.vertices.insert(new_sg.vertices.end(), parent_sg.vertices.begin(), parent_sg.vertices.end());
            new_sg.work_weight += parent_sg.work_weight;
            new_sg.memory_weight += parent_sg.memory_weight;
            min_start_wf = std::min(min_start_wf, parent_sg.start_wavefront);
        }
        new_sg.start_wavefront = min_start_wf;

        for (size_t i = 1; i < children.size(); ++i) {
            const auto& child = children[i];
            new_sg.vertices.push_back(child);
            new_sg.work_weight += dag_->vertex_work_weight(child);
            new_sg.memory_weight += dag_->vertex_mem_weight(child);
        }
        return new_sg;
    }

    // This function reads from `active_subgraphs_`, creates new merged subgraphs,
    // places them in `next_active_subgraphs`, and returns the set of parent IDs
    // that were consumed in the process.
    std::set<unsigned> merge_and_continue_group(const GroupKey& parent_group_key, const OrbitAnalysis& analysis,
                                                unsigned wf_idx, std::unordered_map<unsigned, Subgraph>& next_active_subgraphs) {
        const unsigned family_id = parent_group_key.second;
        
        auto parent_set_to_children = group_children_by_parent_set(analysis, parent_group_key);
        
        if (parent_set_to_children.empty()) {
            return {}; // Should not happen if called for a continuing orbit.
        }

        // Create new subgraphs for the next generation.
        std::set<unsigned> consumed_parents;
        for (const auto& [parent_set, children] : parent_set_to_children) {
            if (children.empty()) continue;

            consumed_parents.insert(parent_set.begin(), parent_set.end());

            size_t new_hash = 0;
            // Combine parent hash for each parent to distinguish merge topology.
            for (size_t i = 0; i < parent_set.size(); ++i) {
                hash_combine(new_hash, parent_group_key.first);
            }
            // Combine orbit hash for each child.
            for (size_t i = 0; i < children.size(); ++i) {
                hash_combine(new_hash, analysis.hash);
            }
            Subgraph new_sg = create_merged_subgraph(parent_set, children, active_subgraphs_, new_hash, family_id, wf_idx);
            unsigned new_id = next_subgraph_id_++;
            next_active_subgraphs.emplace(new_id, std::move(new_sg));
            for (const auto& v : next_active_subgraphs.at(new_id).vertices) {
                vertex_to_subgraph_id_[v] = new_id;
            }
        }
        return consumed_parents;
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
    
    std::vector<std::vector<unsigned>> build_isomorphic_groups_from_finalized() {
        isomorphic_groups_.clear();
        unsigned idx = 0;
        for (const auto& sg : finalized_subgraphs_) {
            isomorphic_groups_[{sg.current_hash, sg.family_id}].push_back(idx++);
        }

        std::vector<std::vector<unsigned>> result;
        result.reserve(isomorphic_groups_.size());
        for (const auto& [key, sg_ids] : isomorphic_groups_) {
            result.push_back(sg_ids);
        }
        return result;
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

    void add_family_node(unsigned family_id, unsigned parent_id, unsigned wf_idx) {
        if (family_tree_.count(family_id)) return; // Should not happen, but safe.

        family_tree_[family_id] = {family_id, parent_id, {}, wf_idx};
        if (parent_id != std::numeric_limits<unsigned>::max() && family_tree_.count(parent_id)) {
            family_tree_.at(parent_id).children_ids.push_back(family_id);
        }
    }

    void print_family_tree_node(unsigned family_id, unsigned depth, std::unordered_set<unsigned>& visited) const {
            if (visited.count(family_id)) return;
            visited.insert(family_id);

            const auto& node = family_tree_.at(family_id);
            std::cout << std::string(depth * 2, ' ') << "-> Family " << node.id 
                      << " (created at WF " << node.creation_wavefront << ")\n";

            for (unsigned child_id : node.children_ids) {
                print_family_tree_node(child_id, depth + 1, visited);
        }
    }

    void print_family_tree() const {
            std::cout << "\n--- 🌳 Family Lineage Tree ---\n";
            if (family_tree_.empty()) {
                std::cout << "  (No families tracked yet.)\n";
            } else {
                std::unordered_set<unsigned> visited;
                for (const auto& [id, node] : family_tree_) {
                    if (node.parent_id == std::numeric_limits<unsigned>::max()) {
                        print_family_tree_node(node.id, 0, visited);
                    }
                }
            }
            std::cout << "---------------------------\n";
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

template<typename Graph_t, typename Constr_Graph_t>
struct subgrah_scheduler_input {

    using GroupKey = typename WavefrontOrbitProcessor<Graph_t>::GroupKey;
    using GroupKeyHash = typename WavefrontOrbitProcessor<Graph_t>::GroupKeyHash;

    BspInstance<Constr_Graph_t> instance;
    std::vector<unsigned> multiplicities;
    std::vector<std::vector<v_workw_t<Graph_t>>> required_proc_types;

    static constexpr bool verbose = false;

    void prepare_subgraph_scheduling_input(const BspInstance<Graph_t>& original_instance, const std::vector<subgraph<Graph_t>> & finalized_subgraphs, const std::vector<std::vector<unsigned>> & isomorphic_groups) {
        instance.setArchitecture(original_instance.getArchitecture());
        const unsigned num_proc_types = original_instance.getArchitecture().getNumberOfProcessorTypes();
        const unsigned min_proc_type_count = original_instance.getArchitecture().getMinProcessorTypeCount();

        multiplicities.resize(isomorphic_groups.size());
        required_proc_types.resize(isomorphic_groups.size());
        std::vector<vertex_idx_t<Constr_Graph_t>> contraction_map(original_instance.numberOfVertices());

        size_t i = 0;
        for (const auto &sg_ids : isomorphic_groups) {
            multiplicities[i] = std::gcd(min_proc_type_count, static_cast<unsigned>(sg_ids.size()));
            required_proc_types[i].resize(num_proc_types, 0);

            for (const auto sg_id : sg_ids) {
                const auto &subgraph = finalized_subgraphs.at(sg_id);
                for (const auto &vertex : subgraph.vertices) {
                    contraction_map[vertex] = static_cast<vertex_idx_t<Constr_Graph_t>>(i);
                    const auto vertex_work = original_instance.getComputationalDag().vertex_work_weight(vertex);
                    const auto vertex_type = original_instance.getComputationalDag().vertex_type(vertex);
                    for (unsigned j = 0; j < num_proc_types; ++j) {
                        if (original_instance.isCompatibleType(vertex_type, j)) {
                            required_proc_types[i][j] += vertex_work;
                        }
                    }
                }
            }
            ++i;
        }
        coarser_util::construct_coarse_dag(original_instance.getComputationalDag(), instance.getComputationalDag(),
                                        contraction_map);

        if constexpr (verbose) {
        std::cout << "\n--- 🏗️ Preparing Subgraph Scheduling Input ---\n";
        std::cout << "Found " << isomorphic_groups.size() << " isomorphic groups to schedule as coarse nodes.\n";
        std::cout << std::string(80, '-') << std::endl;
        for (size_t j = 0; j < isomorphic_groups.size(); ++j) {
            std::cout << "  - Coarse Node " << j << " (from " << isomorphic_groups[j].size()
                    << " isomorphic subgraphs):\n";
            std::cout << "    - Multiplicity for scheduling: " << multiplicities[j] << "\n";
            std::cout << "    - Total Work (in coarse graph): " << instance.getComputationalDag().vertex_work_weight(j)
                    << "\n";
            std::cout << "    - Required Proc Work (for whole group): [";
            for (unsigned k = 0; k < num_proc_types; ++k) {
                std::cout << "T" << k << ":" << required_proc_types[j][k] << (k == num_proc_types - 1 ? "" : ", ");
            }
            std::cout << "]\n";
        }
        std::cout << std::string(80, '-') << std::endl;
        }
    }
};

} // namespace osp
