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
#include "osp/auxiliary/hash_util.hpp"
#include "osp/graph_algorithms/transitive_reduction.hpp"
#include "osp/coarser/Sarkar/Sarkar.hpp"
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
    static constexpr bool verbose = true;

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
        perform_coarsening(dag, coarse_graph_);
    }

private:

    std::vector<Group> m_current_groups;
    std::unordered_map<std::pair<VertexType, VertexType>, std::vector<std::vector<VertexType>>, pair_hash> new_subgraphs;
    const Graph_t * original_dag_ptr = nullptr;

    SarkarParams::Parameters<v_workw_t<Graph_t>> get_sarkar_params() const {
        SarkarParams::Parameters<v_workw_t<Graph_t>> params;
        params.geomDecay = 0.0;
        params.leniency = 0.0;
        params.mode = SarkarParams::Mode::LINES; // Default mode, can be changed
        params.commCost = 0; // Default communication cost
        params.maxWeight = std::numeric_limits<v_workw_t<Graph_t>>::max(); // Default max weight
        params.smallWeightThreshold = std::numeric_limits<v_workw_t<Graph_t>>::lowest(); // Default small weight threshold
        params.useTopPoset = true; // Default        
        return params;
    }


    /**
     * @brief Greedily merges nodes in the orbit graph based on structural and symmetry constraints.
     */
    void perform_coarsening(const Graph_t& original_dag, const Constr_Graph_t& initial_coarse_graph) {
        final_coarse_graph_ = Constr_Graph_t();
        final_contraction_map_.clear();
        
        if (initial_coarse_graph.num_vertices() == 0) {
            return;
        }

        original_dag_ptr = &original_dag;
        Constr_Graph_t current_coarse_graph = initial_coarse_graph;
        for (const auto & v : current_coarse_graph.vertices())
            current_coarse_graph.set_vertex_type(v, 0);


        m_current_groups.resize(current_coarse_graph.num_vertices());
        std::vector<VertexType> current_contraction_map = contraction_map_;

        // Initialize groups: each group corresponds to an orbit.
        for (VertexType i = 0; i < original_dag.num_vertices(); ++i) {
            const VertexType coarse_node = contraction_map_[i];
            m_current_groups[coarse_node].subgraphs.push_back({i});
        }

        Sarkar<Constr_Graph_t, Constr_Graph_t> sarkar(get_sarkar_params());
        sarkar.addContractionConstraint([&](const Constr_Graph_t& graph, VertexType u, VertexType v) {
                return is_merge_viable_constr(graph, u, v);
            });
            

        bool changed = true;
        while (changed) {
            
            new_subgraphs.clear();
            Constr_Graph_t new_coarse_graph;
            std::vector<vertex_idx_t<Constr_Graph_t>> contraction_map; 
            sarkar.coarsenDag(current_coarse_graph, new_coarse_graph, contraction_map);

            if (current_coarse_graph.num_vertices() == new_coarse_graph.num_vertices()) {
                changed = false;
                break;
            } 

            current_coarse_graph = std::move(new_coarse_graph);

            // Update groups
            std::vector<Group> next_groups(current_coarse_graph.num_vertices());
            std::vector<bool> already_matched(contraction_map.size(), false);
            for (std::size_t i = 0; i < contraction_map.size(); ++i) {
                if (already_matched[i]) continue;

                const VertexType & v = contraction_map[i];
                VertexType u;
                bool i_is_merged_with_u = false;
                for (std::size_t j = i + 1; j < contraction_map.size(); ++j) {                    
                    if (contraction_map[j] == v) {
                        u = j;
                        already_matched[j] = true;
                        i_is_merged_with_u = true;
                        break;
                    }
                }

                if (i_is_merged_with_u) {
                    const auto pair = std::make_pair(i, u);
                    if (new_subgraphs.find(pair) == new_subgraphs.end()) {
                        next_groups[v] = {std::move(new_subgraphs[std::make_pair(u, i)])};
                    } else {
                        next_groups[v] = {std::move(new_subgraphs[pair])};
                    }                    
                } else {
                    next_groups[v] = std::move(m_current_groups[i]);
                }
            }
            m_current_groups = std::move(next_groups);  
        
            for (VertexType i = 0; i < current_contraction_map.size(); ++i) {
                    current_contraction_map[i] = contraction_map[current_contraction_map[i]];
            }
        }

        // --- Finalize ---
        final_coarse_graph_ = std::move(current_coarse_graph);
        final_contraction_map_ = std::move(current_contraction_map);
        final_groups_ = std::move(m_current_groups);

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
    bool is_merge_viable_constr(const Constr_Graph_t& , const VertexType& u, const VertexType& v) {
        auto & group_u = m_current_groups[u];
        auto & group_v = m_current_groups[v];

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
        create_induced_subgraph(*original_dag_ptr, induced_subgraph, all_nodes);

        std::vector<VertexType> components; // local -> component_id
        size_t num_components = compute_weakly_connected_components(induced_subgraph, components);      

        std::vector<std::vector<VertexType>> new_subgraphs_group{num_components, std::vector<VertexType>()};
        for (VertexType i = 0; i < induced_subgraph.num_vertices(); ++i) {
            new_subgraphs_group[components[i]].push_back(all_nodes[i]);
        }
        
        if (num_components < symmetry_threshold_) {
            new_subgraphs[std::make_pair(u, v)] = std::move(new_subgraphs_group);
            const bool both_below_symmetry_threshold = (group_u.size() < symmetry_threshold_) && (group_v.size() < symmetry_threshold_); 
            return both_below_symmetry_threshold;
        }
       
        if (num_components > 1) {
            const size_t first_sg_size = new_subgraphs_group[0].size();
            Constr_Graph_t rep_sg;
            create_induced_subgraph(*original_dag_ptr, rep_sg, new_subgraphs_group[0]);

            for (size_t i = 1; i < num_components; ++i) {
                if (new_subgraphs_group[i].size() != first_sg_size) {
                    return false;
                }
                
                Constr_Graph_t current_sg;
                create_induced_subgraph(*original_dag_ptr, current_sg, new_subgraphs_group[i]);
                if (!are_isomorphic_by_merkle_hash(rep_sg, current_sg)) {
                    return false;
                }
            }
        }
        new_subgraphs[std::make_pair(u, v)] = std::move(new_subgraphs_group);
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