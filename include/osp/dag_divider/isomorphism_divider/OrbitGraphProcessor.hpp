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

#include <vector>

#include "osp/coarser/coarser_util.hpp"
#include "osp/dag_divider/isomorphism_divider/MerkleHashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_algorithms/transitive_reduction.hpp"
#include <numeric>

namespace osp {

/**
 * @class OrbitGraphProcessor
 * @brief A simple processor that groups nodes of a DAG based on their Merkle hash.
 *
 * This class uses a MerkleHashComputer to assign a structural hash to each node.
 * It then partitions the DAG by grouping all nodes with the same hash into an "orbit".
 * A coarse graph is constructed where each node represents one such orbit.
 */
template<typename Graph_t, typename Constr_Graph_t>
class OrbitGraphProcessor {
public:
    static_assert(is_computational_dag_v<Graph_t>, "Graph must be a computational DAG");
    static_assert(is_computational_dag_v<Constr_Graph_t>, "Constr_Graph_t must be a computational DAG");
    static_assert(is_constructable_cdag_v<Constr_Graph_t>,
                  "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>>,
                  "Graph_t and Constr_Graph_t must have the same vertex_idx types");

    using VertexType = vertex_idx_t<Graph_t>;
    
    // Represents a group of isomorphic subgraphs, corresponding to a single node in a coarse graph.
    struct Group {
        // Each vector of vertices represents one of the isomorphic subgraphs in this group.
        std::vector<std::vector<VertexType>> subgraphs;

        inline size_t size() const { return subgraphs.size(); }
        // v_workw_t<Graph_t> work_weight_per_subgraph = 0;
    };

private:
    using MerkleHashComputer_t = MerkleHashComputer<Graph_t, bwd_merkle_node_hash_func<Graph_t>, true>; //MerkleHashComputer<Graph_t, node_hash_func_t, true>;
    // using MerkleHashComputer_t = MerkleHashComputer<Graph_t, node_hash_func_t, true>;

    // Results from the first (orbit) coarsening step
    Constr_Graph_t coarse_graph_;
    std::vector<VertexType> contraction_map_;

    // Results from the second (custom) coarsening step
    Constr_Graph_t final_coarse_graph_;
    std::vector<VertexType> final_contraction_map_;
    std::vector<Group> final_groups_;

    size_t symmetry_threshold_ = 2;
    static constexpr bool verbose = false;

public:
    explicit OrbitGraphProcessor(size_t symmetry_threshold = 2)
        : symmetry_threshold_(symmetry_threshold) {}

    /**
     * @brief Sets the minimum number of isomorphic subgraphs a merged group must have.
     * @param threshold The symmetry threshold.
     */
    void set_symmetry_threshold(size_t threshold) {
        symmetry_threshold_ = threshold;
    }

    /**
     * @brief Discovers isomorphic groups (orbits) and constructs a coarse graph.
     * @param dag The input computational DAG.
     */
    void discover_isomorphic_groups(const Graph_t &dag) {
        coarse_graph_ = Constr_Graph_t();
        contraction_map_.clear();
        final_coarse_graph_ = Constr_Graph_t();
        final_contraction_map_.clear();
        final_groups_.clear();

        if (dag.num_vertices() == 0) {
            return;
        }
  
        MerkleHashComputer_t hasher(dag, dag); // The second 'dag' is for the bwd_merkle_node_hash_func
        const auto orbits = hasher.get_orbits(); 

        contraction_map_.assign(dag.num_vertices(), 0);
        VertexType coarse_node_idx = 0;

        for (const auto& [hash, vertices] : orbits) {
            for (const auto v : vertices) {
                contraction_map_[v] = coarse_node_idx;
            }
            coarse_node_idx++;
        }

        coarser_util::construct_coarse_dag(dag, coarse_graph_, contraction_map_);

        // Constr_Graph_t transitive_reduction; 
        // transitive_reduction_sparse(coarse_graph_, transitive_reduction);
        // coarse_graph_ = std::move(transitive_reduction);

        perform_coarsening(dag, coarse_graph_);
    }

private:
    /**
     * @brief Greedily merges nodes in the orbit graph based on structural and symmetry constraints.
     */
    void perform_coarsening(const Graph_t& original_dag, const Constr_Graph_t& initial_coarse_graph) {
        final_coarse_graph_ = Constr_Graph_t();
        final_contraction_map_.clear();
        
        if (initial_coarse_graph.num_vertices() == 0) {
            return;
        }

        Constr_Graph_t current_coarse_graph = initial_coarse_graph;
        std::vector<Group> current_groups(initial_coarse_graph.num_vertices());
        std::vector<VertexType> current_contraction_map = contraction_map_;

        // Initialize groups: each group corresponds to an orbit.
        for (VertexType i = 0; i < original_dag.num_vertices(); ++i) {
            const VertexType coarse_node = contraction_map_[i];
            current_groups[coarse_node].subgraphs.push_back({i});
        }


        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& edge : edges(current_coarse_graph)) {
                VertexType u = source(edge, current_coarse_graph);
                VertexType v = target(edge, current_coarse_graph);

                if (current_coarse_graph.in_degree(v) != 1) {
                    if constexpr (verbose) { std::cout << "  - Skipping edge " << u << " -> " << v << " target in-degree > 1" << std::endl; }
                    continue;
                }

                std::vector<std::vector<VertexType>> new_subgraphs;

                // --- Check Constraints ---
                // Symmetry Threshold
                const bool merge_viable = is_merge_viable(original_dag, current_groups[u], current_groups[v], new_subgraphs);
                const bool both_below_symmetry_threshold = (current_groups[u].size() < symmetry_threshold_) && (current_groups[v].size() < symmetry_threshold_);                
                if (!merge_viable && !both_below_symmetry_threshold) {
                    if constexpr (verbose) { std::cout << "  - Merge of " << u << " and " << v << " not viable (symmetry threshold)\n"; }
                    continue;
                }
                
                // Acyclicity & Critical Path
                Constr_Graph_t temp_coarse_graph;
                std::vector<VertexType> temp_contraction_map(current_coarse_graph.num_vertices());
                VertexType new_idx = 0;
                for (VertexType i = 0; i < temp_contraction_map.size(); ++i) {
                    if (i != v) {
                        temp_contraction_map[i] = new_idx++;
                    }
                }
                // Assign 'v' the same new index as 'u'.
                temp_contraction_map[v] = temp_contraction_map[u];
                coarser_util::construct_coarse_dag(current_coarse_graph, temp_coarse_graph, temp_contraction_map);

                if (!is_acyclic(temp_coarse_graph)) {
                    if constexpr (verbose) { std::cout << "  - Merge of " << u << " and " << v << " creates a cycle. Skipping.\n"; }
                    continue;
                }

                if (critical_path_weight(temp_coarse_graph) > critical_path_weight(current_coarse_graph)) {
                    if constexpr (verbose) { std::cout << "  - Merge of " << u << " and " << v << " increases critical path. Skipping.\n"; }
                    continue;
                }

                // --- If all checks pass, execute the merge ---
                if constexpr (verbose) { std::cout << "  - Merging " << v << " into " << u << ". New coarse graph has " << temp_coarse_graph.num_vertices() << " nodes.\n"; }
                // The new coarse graph is the one we just tested
                current_coarse_graph = std::move(temp_coarse_graph);

                // Update groups
                std::vector<Group> next_groups(current_coarse_graph.num_vertices());
                std::vector<VertexType> group_remap(current_groups.size());
                new_idx = 0;
                for (VertexType i = 0; i < group_remap.size(); ++i) {
                    if (i != v) {
                        group_remap[i] = new_idx++;
                    }
                }
                group_remap[v] = group_remap[u];

                // Move existing groups that are not part of the merge
                for (VertexType i = 0; i < current_groups.size(); ++i) {
                    if (i != u && i != v) {
                        next_groups[group_remap[i]] = std::move(current_groups[i]);
                    }
                }
                // Install the newly computed merged group
                next_groups[group_remap[u]].subgraphs = std::move(new_subgraphs);
                current_groups = std::move(next_groups);

                // Update the main contraction map
                for (VertexType i = 0; i < current_contraction_map.size(); ++i) {
                    current_contraction_map[i] = group_remap[current_contraction_map[i]];
                }

                changed = true;
                break; // Restart scan on the new, smaller graph
            }
        }

        // --- Finalize ---
        final_coarse_graph_ = std::move(current_coarse_graph);
        final_contraction_map_ = std::move(current_contraction_map);
        final_groups_ = std::move(current_groups);

        if constexpr (verbose) {
            print_final_groups_summary();
        }
    }

    void print_final_groups_summary() const {
        std::cout << "\n--- 📦 Final Groups Summary ---\n";
        std::cout << "Total final groups: " << final_groups_.size() << "\n";
        for (size_t i = 0; i < final_groups_.size(); ++i) {
            const auto& group = final_groups_[i];
            std::cout << "  - Group " << i << " (Size: " << group.subgraphs.size() << ")\n";
            if (!group.subgraphs.empty() && !group.subgraphs[0].empty()) {
                 std::cout << "    - Rep. Subgraph size: " << group.subgraphs[0].size() << " nodes\n";
            }
        }
        std::cout << "--------------------------------\n";
    }

    /**
     * @brief Checks if merging two groups is viable based on the resulting number of isomorphic subgraphs.
     * This is analogous to WavefrontOrbitProcessor::is_viable_continuation.
     * If viable, it populates the `out_new_subgraphs` with the structure of the merged group.
     */
    bool is_merge_viable(const Graph_t& original_dag, const Group& group_u, const Group& group_v,
                         std::vector<std::vector<VertexType>>& out_new_subgraphs) const {

        std::vector<VertexType> all_nodes;
        all_nodes.reserve(group_u.subgraphs.size() + group_v.subgraphs.size());
        for (const auto& sg : group_u.subgraphs) {
            all_nodes.insert(all_nodes.end(), sg.begin(), sg.end());
        }
        for (const auto& sg : group_v.subgraphs) {
            all_nodes.insert(all_nodes.end(), sg.begin(), sg.end());
        }

        assert([&]() {
            std::vector<VertexType> temp_nodes_for_check = all_nodes;
            std::sort(temp_nodes_for_check.begin(), temp_nodes_for_check.end());
            return std::unique(temp_nodes_for_check.begin(), temp_nodes_for_check.end()) == temp_nodes_for_check.end();
        }() && "Assumption failed: Vertices in groups being merged are not disjoint.");

        std::sort(all_nodes.begin(), all_nodes.end());

        Constr_Graph_t induced_subgraph;
        create_induced_subgraph(original_dag, induced_subgraph, all_nodes);

        std::vector<VertexType> components; // local -> component_id
        size_t num_components = compute_weakly_connected_components(induced_subgraph, components);

        out_new_subgraphs.assign(num_components, std::vector<VertexType>());
        for (VertexType i = 0; i < induced_subgraph.num_vertices(); ++i) {
            out_new_subgraphs[components[i]].push_back(all_nodes[i]);
        }

        if (num_components < symmetry_threshold_) {
            return false;
        }

        if (num_components > 1) {
            const size_t first_sg_size = out_new_subgraphs[0].size();
            Constr_Graph_t rep_sg;
            create_induced_subgraph(original_dag, rep_sg, out_new_subgraphs[0]);

            for (size_t i = 1; i < num_components; ++i) {
                if (out_new_subgraphs[i].size() != first_sg_size) {
                    return false;
                }
                
                Constr_Graph_t current_sg;
                create_induced_subgraph(original_dag, current_sg, out_new_subgraphs[i]);
                if (!are_isomorphic_by_merkle_hash(rep_sg, current_sg)) {
                    return false;
                }
            }
        }

        return true;
    }

public:
    const Graph_t& get_coarse_graph() const { return coarse_graph_; }
    const std::vector<VertexType>& get_contraction_map() const { return contraction_map_; }
    const Graph_t& get_final_coarse_graph() const { return final_coarse_graph_; }
    const std::vector<VertexType>& get_final_contraction_map() const { return final_contraction_map_; }
    const std::vector<Group>& get_final_groups() const { return final_groups_; }
};

} // namespace osp