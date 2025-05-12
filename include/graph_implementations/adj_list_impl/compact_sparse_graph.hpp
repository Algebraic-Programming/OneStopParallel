/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/
#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <type_traits>
#include <vector>

#include "auxiliary/math_helper.hpp"
#include "concepts/computational_dag_concept.hpp"
#include "graph_algorithms/directed_graph_edge_view.hpp"
#include "graph_implementations/vertex_iterator.hpp"


namespace osp {

template<bool keep_vertex_order, bool use_work_weights = false, bool use_comm_weights = false, bool use_mem_weights = false, bool use_vert_types = false, typename vert_t = std::size_t, typename edge_t = std::size_t, typename work_weight_type = unsigned, typename comm_weight_type = unsigned, typename mem_weight_type = unsigned, typename vertex_type_template_type = unsigned>
class Compact_Sparse_Graph {
    static_assert(std::is_integral<vert_t>::value && std::is_integral<edge_t>::value, "Vertex and edge type must be of integral nature.");
    static_assert(std::is_arithmetic_v<work_weight_type> && "Work weight must be of arithmetic type.");
    static_assert(std::is_arithmetic_v<comm_weight_type> && "Communication weight must be of arithmetic type.");
    static_assert(std::is_arithmetic_v<mem_weight_type> && "Memory weight must be of arithmetic type.");
    static_assert(std::is_integral_v<vertex_type_template_type> && "Vertex type type must be of integral type.");

    public:
        using vertex_idx = vert_t;

        using vertex_work_weight_type = std::conditional_t<use_work_weights, work_weight_type, edge_t>;
        using vertex_comm_weight_type = comm_weight_type;
        using vertex_mem_weight_type = mem_weight_type;
        using vertex_type_type = vertex_type_template_type;

    private:
        using ThisT = Compact_Sparse_Graph<keep_vertex_order, use_work_weights, use_comm_weights, use_mem_weights, use_vert_types, vert_t, edge_t, work_weight_type, comm_weight_type, mem_weight_type, vertex_type_template_type>;

        class Compact_Parent_Edges {
            private:
                // Compressed Sparse Row (CSR)
                std::vector<vertex_idx> csr_edge_parents;
                std::vector<edge_t> csr_target_ptr;

            public:
                Compact_Parent_Edges() = default;
                Compact_Parent_Edges(const Compact_Parent_Edges &other) = default;
                Compact_Parent_Edges(Compact_Parent_Edges &&other) = default;
                Compact_Parent_Edges &operator=(const Compact_Parent_Edges &other) = default;
                Compact_Parent_Edges &operator=(Compact_Parent_Edges &&other) = default;
                virtual ~Compact_Parent_Edges() = default;

                Compact_Parent_Edges(const std::vector<vertex_idx> &csr_edge_parents_, const std::vector<edge_t> &csr_target_ptr_) : csr_edge_parents(csr_edge_parents_), csr_target_ptr(csr_target_ptr_) {};
                Compact_Parent_Edges(std::vector<vertex_idx> &&csr_edge_parents_, std::vector<edge_t> &&csr_target_ptr_) : csr_edge_parents(std::move(csr_edge_parents_)), csr_target_ptr(std::move(csr_target_ptr_)) {};

                inline edge_t number_of_parents(const vertex_idx v) const {
                    return csr_target_ptr[v + 1] - csr_target_ptr[v];
                }

                class Parent_range {
                    private:
                        const std::vector<vertex_idx> &_csr_edge_parents;
                        const std::vector<edge_t> &_csr_target_ptr;
                        const vertex_idx _vert;

                    public:
                        Parent_range (const std::vector<vertex_idx> &csr_edge_parents, const std::vector<edge_t> &csr_target_ptr, const vertex_idx vert) : _csr_edge_parents(csr_edge_parents), _csr_target_ptr(csr_target_ptr), _vert(vert) {};

                        inline auto cbegin() const { auto it = _csr_edge_parents.cbegin(); std::advance(it, _csr_target_ptr[_vert]); return it; }
                        inline auto cend() const { auto it = _csr_edge_parents.cbegin(); std::advance(it, _csr_target_ptr[_vert + 1]); return it; }
                        
                        inline auto begin() const { return cbegin(); }
                        inline auto end() const { return cend(); }

                        inline auto crbegin() const { auto it = _csr_edge_parents.crbegin(); std::advance(it, _csr_target_ptr[_csr_target_ptr.size() - 1] - _csr_target_ptr[_vert + 1]); return it; };
                        inline auto crend() const { auto it = _csr_edge_parents.crbegin(); std::advance(it, _csr_target_ptr[_csr_target_ptr.size() - 1] - _csr_target_ptr[_vert]); return it; };

                        inline auto rbegin() const { return crbegin(); };
                        inline auto rend() const { return crend(); };
                };

                inline Parent_range parents(const vertex_idx vert) const { return Parent_range(csr_edge_parents, csr_target_ptr, vert); }
        };

        class Compact_Children_Edges {
            private:
                // Compressed Sparse Column (CSC)
                std::vector<vertex_idx> csc_edge_children;
                std::vector<edge_t> csc_source_ptr;

            public:
                Compact_Children_Edges() = default;
                Compact_Children_Edges(const Compact_Children_Edges &other) = default;
                Compact_Children_Edges(Compact_Children_Edges &&other) = default;
                Compact_Children_Edges &operator=(const Compact_Children_Edges &other) = default;
                Compact_Children_Edges &operator=(Compact_Children_Edges &&other) = default;
                virtual ~Compact_Children_Edges() = default;

                Compact_Children_Edges(const std::vector<vertex_idx> &csc_edge_children_, const std::vector<edge_t> &csc_source_ptr_) : csc_edge_children(csc_edge_children_), csc_source_ptr(csc_source_ptr_) {};
                Compact_Children_Edges(std::vector<vertex_idx> &&csc_edge_children_, std::vector<edge_t> &&csc_source_ptr_) : csc_edge_children(std::move(csc_edge_children_)), csc_source_ptr(std::move(csc_source_ptr_)) {};

                inline edge_t number_of_children(const vertex_idx v) const {
                    return csc_source_ptr[v + 1] - csc_source_ptr[v];
                }

                class Children_range {
                    private:
                        const std::vector<vertex_idx> &_csc_edge_children;
                        const std::vector<edge_t> &_csc_source_ptr;
                        const vertex_idx _vert;

                    public:
                        Children_range (const std::vector<vertex_idx> &csc_edge_children, const std::vector<edge_t> &csc_source_ptr, const vertex_idx vert) : _csc_edge_children(csc_edge_children), _csc_source_ptr(csc_source_ptr), _vert(vert) {};

                        inline auto cbegin() const { auto it = _csc_edge_children.cbegin(); std::advance(it, _csc_source_ptr[_vert]); return it; };
                        inline auto cend() const { auto it = _csc_edge_children.cbegin(); std::advance(it, _csc_source_ptr[_vert + 1]); return it; };

                        inline auto begin() const { return cbegin(); };
                        inline auto end() const { return cend(); };

                        inline auto crbegin() const { auto it = _csc_edge_children.crbegin(); std::advance(it, _csc_source_ptr[_csc_source_ptr.size() - 1] - _csc_source_ptr[_vert + 1]); return it; };
                        inline auto crend() const { auto it = _csc_edge_children.crbegin(); std::advance(it, _csc_source_ptr[_csc_source_ptr.size() - 1] - _csc_source_ptr[_vert]); return it; };

                        inline auto rbegin() const { return crbegin(); };
                        inline auto rend() const { return crend(); };
                };

                inline Children_range children(const vertex_idx vert) const { return Children_range(csc_edge_children, csc_source_ptr, vert); }
        };



        const vertex_idx number_of_vertices = static_cast<vert_t>(0);
        const edge_t number_of_edges = static_cast<edge_t>(0);

        Compact_Parent_Edges csr_in_edges;
        Compact_Children_Edges csc_out_edges;

        vertex_type_type number_of_vertex_types = static_cast<vertex_type_type>(1);

        std::vector<vertex_work_weight_type> vert_work_weights;
        std::vector<vertex_comm_weight_type> vert_comm_weights;
        std::vector<vertex_mem_weight_type> vert_mem_weights;
        std::vector<vertex_type_type> vert_types;


        std::vector<vertex_idx> vertex_permutation_from_internal_to_original;
        std::vector<vertex_idx> vertex_permutation_from_original_to_internal;

        template<typename RetT = void>
        std::enable_if_t<not use_vert_types, RetT> _update_num_vertex_types() {
            number_of_vertex_types = static_cast<vertex_type_type>(1);
        }

        template<typename RetT = void>
        std::enable_if_t<use_vert_types, RetT> _update_num_vertex_types() {
            number_of_vertex_types = static_cast<vertex_type_type>(1);
            for (const auto vt : vert_types) {
                number_of_vertex_types = std::max(number_of_vertex_types, vt);
            }
        }
    

    public:
        Compact_Sparse_Graph() = default;
        Compact_Sparse_Graph(const Compact_Sparse_Graph &other) = default;
        Compact_Sparse_Graph(Compact_Sparse_Graph &&other) = default;
        Compact_Sparse_Graph &operator=(const Compact_Sparse_Graph &other) = delete;
        Compact_Sparse_Graph &operator=(Compact_Sparse_Graph &&other) = delete;
        virtual ~Compact_Sparse_Graph() = default;

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, const edge_list_type & edges) : number_of_vertices(num_vertices_), number_of_edges(static_cast<edge_t>(edges.size())) {
            static_assert( std::is_same<edge_list_type, std::vector<std::pair<vertex_idx, vertex_idx>> >::value
                        || is_edge_list_type<edge_list_type, vertex_idx, edge_t>::value);
            
            assert((0 <= num_vertices_) && "Number of vertices must be non-negative.");
            assert((edges.size() < static_cast<size_t>(std::numeric_limits<edge_t>::max())) && "Number of edge must be strictly smaller than the maximally representable number.");
            
            if constexpr ( std::is_same_v<edge_list_type, std::vector<std::pair<vertex_idx, vertex_idx>>> ) {
                assert(std::all_of(edges.cbegin(), edges.cend(), [num_vertices_](const auto &edge) { return (0 <= edge.first) && (edge.first < num_vertices_) && (0 <= edge.second) && (edge.second < num_vertices_); } ) && "Source and target of edges must be non-negative and less than the number of vertices.");
            }

            if constexpr ( is_edge_list_type_v<edge_list_type, vertex_idx, edge_t> ) {
                assert(std::all_of(edges.begin(), edges.end(), [num_vertices_](const auto &edge) { return (0 <= edge.source) && (edge.source < num_vertices_) && (0 <= edge.target) && (edge.target < num_vertices_); } ) && "Source and target of edges must be non-negative and less than the number of vertices.");
            }

            if constexpr (keep_vertex_order) {
                if constexpr ( std::is_same_v<edge_list_type, std::vector<std::pair<vertex_idx, vertex_idx>>> ) {
                    assert(std::all_of(edges.cbegin(), edges.cend(), [](const auto &edge) { return edge.first < edge.second; } ) && "Vertex order must be a topological order.");
                }
                if constexpr ( is_edge_list_type_v<edge_list_type, vertex_idx, edge_t> ) {
                    assert(std::all_of(edges.begin(), edges.end(), [](const auto &edge) { return edge.source < edge.target; } ) && "Vertex order must be a topological order.");
                }
            }

            if constexpr (use_work_weights) {
                vert_work_weights = std::vector<vertex_work_weight_type>(num_vertices(), 1);
            }
            if constexpr (use_comm_weights) {
                vert_comm_weights = std::vector<vertex_comm_weight_type>(num_vertices(), 0);
            }
            if constexpr (use_mem_weights) {
                vert_mem_weights = std::vector<vertex_mem_weight_type>(num_vertices(), 0);
            }
            if constexpr (use_vert_types) {
                number_of_vertex_types = 1;
                vert_types = std::vector<vertex_type_type>(num_vertices(), 0);
            }
            if constexpr (!keep_vertex_order) {
                vertex_permutation_from_internal_to_original.reserve(num_vertices());
                vertex_permutation_from_original_to_internal.reserve(num_vertices());
            }

            // Construction
            std::vector<std::vector<vertex_idx>> children_tmp(num_vertices());
            std::vector<edge_t> num_parents_tmp(num_vertices(), 0);

            if constexpr ( std::is_same_v<edge_list_type, std::vector<std::pair<vertex_idx, vertex_idx>>> ) {
                for (const auto &edge : edges) {
                    children_tmp[edge.first].push_back(edge.second);
                    num_parents_tmp[edge.second]++;
                }
            }
            if constexpr ( is_edge_list_type_v<edge_list_type, vertex_idx, edge_t> ) {
                for (const auto &edge : edges) {
                    children_tmp[edge.source].push_back(edge.target);
                    num_parents_tmp[edge.target]++;
                }
            }

            std::vector<vertex_idx> csc_edge_children;
            csc_edge_children.reserve(num_edges());
            std::vector<edge_t> csc_source_ptr(num_vertices() + 1);
            std::vector<vertex_idx> csr_edge_parents(num_edges());
            std::vector<edge_t> csr_target_ptr;
            csr_target_ptr.reserve(num_vertices() + 1);

            if constexpr (keep_vertex_order) {
                for (vertex_idx vert = 0; vert < num_vertices(); ++vert) {
                    csc_source_ptr[vert] = static_cast<edge_t>( csc_edge_children.size() );
                    
                    std::sort(children_tmp[vert].begin(), children_tmp[vert].end());
                    for (const auto &chld : children_tmp[vert]) {
                        csc_edge_children.emplace_back(chld);
                    }
                }
                csc_source_ptr[num_vertices()] = static_cast<edge_t>( csc_edge_children.size() );

                csr_target_ptr = std::vector<edge_t>(num_vertices() + 1, 0);
                std::exclusive_scan(num_parents_tmp.cbegin(), num_parents_tmp.cend(), csr_target_ptr.begin(), 0);
                csr_target_ptr[num_vertices()] = num_edges();

                std::vector<edge_t> offset = csr_target_ptr;
                for (vertex_idx vert = 0; vert < num_vertices(); ++vert) {
                    for (const auto &chld : children_tmp[vert]) {
                        csr_edge_parents[offset[chld]++] = vert;
                    }
                }
                
            } else {
                std::vector<std::vector<vertex_idx>> parents_tmp(num_vertices());

                if constexpr ( std::is_same_v<edge_list_type, std::vector<std::pair<vertex_idx, vertex_idx>>> ) {
                    for (const auto &edge : edges) {
                        parents_tmp[edge.second].push_back(edge.first);
                    }
                }
                if constexpr ( is_edge_list_type_v<edge_list_type, vertex_idx, edge_t> ) {
                    for (const auto &edge : edges) {
                        parents_tmp[edge.target].push_back(edge.source);
                    }
                }

                // Generating modified Gorder topological order cf. "Speedup Graph Processing by Graph Ordering" by Hao Wei, Jeffrey Xu Yu, Can Lu, and Xuemin Lin
                const double decay = 8.0;

                std::vector<edge_t> prec_remaining = num_parents_tmp;
                std::vector<double> priorities(num_vertices(), 0.0);

                auto v_cmp = [&priorities, &children_tmp] (const vertex_idx &lhs, const vertex_idx &rhs) {
                    return  (priorities[lhs] < priorities[rhs]) ||
                            ((priorities[lhs] == priorities[rhs]) && (children_tmp[lhs].size() < children_tmp[rhs].size())) ||
                            ((priorities[lhs] == priorities[rhs]) && (children_tmp[lhs].size() == children_tmp[rhs].size()) && (lhs > rhs));
                };

                std::priority_queue<vertex_idx, std::vector<vertex_idx>, decltype(v_cmp)> ready_q(v_cmp);
                for (vertex_idx vert = 0; vert < num_vertices(); ++vert) {
                    if (prec_remaining[vert] == 0) {
                        ready_q.push(vert);
                    } 
                }

                while (!ready_q.empty()) {
                    vertex_idx vert = ready_q.top();
                    ready_q.pop();

                    double pos = static_cast<double>(vertex_permutation_from_internal_to_original.size());
                    pos /= decay;

                    vertex_permutation_from_internal_to_original.push_back(vert);

                    // update priorities
                    for (vertex_idx chld : children_tmp[vert]) {
                        priorities[chld] = log_sum_exp(priorities[chld], pos);
                    }
                    for (vertex_idx par : parents_tmp[vert]) {
                        for (vertex_idx sibling : children_tmp[par]) {
                            priorities[sibling] = log_sum_exp(priorities[sibling], pos);
                        }
                    }
                    for (vertex_idx chld : children_tmp[vert]) {
                        for (vertex_idx couple : parents_tmp[chld]) {
                            priorities[couple] = log_sum_exp(priorities[couple], pos);
                        }
                    }

                    // update constraints and push to queue
                    for (vertex_idx chld : children_tmp[vert]) {
                        --prec_remaining[chld];
                        if (prec_remaining[chld] == 0) {
                            ready_q.push(chld);
                        }
                    }
                }

                assert(vertex_permutation_from_internal_to_original.size() == static_cast<size_t>(num_vertices()));


                // constructing the csr and csc
                vertex_permutation_from_original_to_internal = std::vector<vertex_idx>(num_vertices(), 0);
                for (vertex_idx new_pos = 0; new_pos < num_vertices(); ++new_pos) {
                    vertex_permutation_from_original_to_internal[vertex_permutation_from_internal_to_original[new_pos]] = new_pos;
                }

                for (vertex_idx vert_new_pos = 0; vert_new_pos < num_vertices(); ++vert_new_pos) {
                    csc_source_ptr[vert_new_pos] = static_cast<edge_t>( csc_edge_children.size() );

                    vertex_idx vert_old_name = vertex_permutation_from_internal_to_original[vert_new_pos];

                    std::vector<vertex_idx> children_new_name;
                    children_new_name.reserve( children_tmp[vert_old_name].size() );

                    for (vertex_idx chld_old_name : children_tmp[vert_old_name]) {
                        children_new_name.push_back( vertex_permutation_from_original_to_internal[chld_old_name] );
                    }
                    
                    
                    std::sort(children_new_name.begin(), children_new_name.end());
                    for (const auto &chld : children_new_name) {
                        csc_edge_children.emplace_back(chld);
                    }
                }
                csc_source_ptr[num_vertices()] = static_cast<edge_t>( csc_edge_children.size() );

                edge_t acc = 0;
                for (vertex_idx vert_old_name : vertex_permutation_from_internal_to_original) {
                    csr_target_ptr.push_back(acc);
                    acc += num_parents_tmp[vert_old_name];
                }
                csr_target_ptr.push_back(acc);

                std::vector<edge_t> offset = csr_target_ptr;
                for (vertex_idx vert = 0; vert < num_vertices(); ++vert) {
                    for (edge_t indx = csc_source_ptr[vert]; indx < csc_source_ptr[vert + 1]; ++indx) {
                        const vertex_idx chld = csc_edge_children[indx];
                        csr_edge_parents[offset[chld]++] = vert;
                    }
                }
            }

            csc_out_edges = Compact_Children_Edges(std::move(csc_edge_children), std::move(csc_source_ptr));
            csr_in_edges = Compact_Parent_Edges(std::move(csr_edge_parents), std::move(csr_target_ptr));
        };

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, const edge_list_type & edges, const std::vector<vertex_work_weight_type> &ww) : Compact_Sparse_Graph(num_vertices_, edges) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
            assert((ww.size() == static_cast<std::size_t>(num_vertices())) && "Work weights vector must have the same length as the number of vertices.");
            
            if constexpr (keep_vertex_order) {
                vert_work_weights = ww;
            } else {
                for (auto vert : vertices()) {
                    vert_work_weights[vert] = ww[vertex_permutation_from_internal_to_original[vert]];
                }
            }
        }

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, edge_list_type && edges, const std::vector<vertex_work_weight_type> &ww) : Compact_Sparse_Graph(num_vertices_, edges) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
            assert((ww.size() == static_cast<std::size_t>(num_vertices())) && "Work weights vector must have the same length as the number of vertices.");

            if constexpr (keep_vertex_order) {
                vert_work_weights = std::move(ww);
            } else {
                for (auto vert : vertices()) {
                    vert_work_weights[vert] = ww[vertex_permutation_from_internal_to_original[vert]];
                }
            }
        }

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, const edge_list_type & edges, const std::vector<vertex_work_weight_type> &ww, const std::vector<vertex_comm_weight_type> &cw) : Compact_Sparse_Graph(num_vertices_, edges) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
            static_assert(use_comm_weights, "To set communication weight, graph type must allow communication weights.");
            assert((ww.size() == static_cast<std::size_t>(num_vertices())) && "Work weights vector must have the same length as the number of vertices.");
            assert((cw.size() == static_cast<std::size_t>(num_vertices())) && "Communication weights vector must have the same length as the number of vertices.");

            if constexpr (keep_vertex_order) {
                vert_work_weights = ww;
            } else {
                for (auto vert : vertices()) {
                    vert_work_weights[vert] = ww[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_comm_weights = cw;
            } else {
                for (auto vert : vertices()) {
                    vert_comm_weights[vert] = cw[vertex_permutation_from_internal_to_original[vert]];
                }
            }
        }

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, const edge_list_type & edges, std::vector<vertex_work_weight_type> &&ww, std::vector<vertex_comm_weight_type> &&cw) : Compact_Sparse_Graph(num_vertices_, edges) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
            static_assert(use_comm_weights, "To set communication weight, graph type must allow communication weights.");
            assert((ww.size() == static_cast<std::size_t>(num_vertices())) && "Work weights vector must have the same length as the number of vertices.");
            assert((cw.size() == static_cast<std::size_t>(num_vertices())) && "Communication weights vector must have the same length as the number of vertices.");

            if constexpr (keep_vertex_order) {
                vert_work_weights = std::move(ww);
            } else {
                for (auto vert : vertices()) {
                    vert_work_weights[vert] = ww[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_comm_weights = std::move(cw);
            } else {
                for (auto vert : vertices()) {
                    vert_comm_weights[vert] = cw[vertex_permutation_from_internal_to_original[vert]];
                }
            }
        }

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, const edge_list_type & edges, const std::vector<vertex_work_weight_type> &ww, const std::vector<vertex_comm_weight_type> &cw, const std::vector<vertex_mem_weight_type> &mw) : Compact_Sparse_Graph(num_vertices_, edges) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
            static_assert(use_comm_weights, "To set communication weight, graph type must allow communication weights.");
            static_assert(use_mem_weights, "To set memory weight, graph type must allow memory weights.");
            assert((ww.size() == static_cast<std::size_t>(num_vertices())) && "Work weights vector must have the same length as the number of vertices.");
            assert((cw.size() == static_cast<std::size_t>(num_vertices())) && "Communication weights vector must have the same length as the number of vertices.");
            assert((mw.size() == static_cast<std::size_t>(num_vertices())) && "Memory weights vector must have the same length as the number of vertices.");

            if constexpr (keep_vertex_order) {
                vert_work_weights = ww;
            } else {
                for (auto vert : vertices()) {
                    vert_work_weights[vert] = ww[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_comm_weights = cw;
            } else {
                for (auto vert : vertices()) {
                    vert_comm_weights[vert] = cw[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_mem_weights = mw;
            } else {
                for (auto vert : vertices()) {
                    vert_mem_weights[vert] = mw[vertex_permutation_from_internal_to_original[vert]];
                }
            }
        }

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, const edge_list_type & edges, std::vector<vertex_work_weight_type> &&ww, std::vector<vertex_comm_weight_type> &&cw, std::vector<vertex_mem_weight_type> &&mw) : Compact_Sparse_Graph(num_vertices_, edges) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
            static_assert(use_comm_weights, "To set communication weight, graph type must allow communication weights.");
            static_assert(use_mem_weights, "To set memory weight, graph type must allow memory weights.");
            assert((ww.size() == static_cast<std::size_t>(num_vertices())) && "Work weights vector must have the same length as the number of vertices.");
            assert((cw.size() == static_cast<std::size_t>(num_vertices())) && "Communication weights vector must have the same length as the number of vertices.");
            assert((mw.size() == static_cast<std::size_t>(num_vertices())) && "Memory weights vector must have the same length as the number of vertices.");

            if constexpr (keep_vertex_order) {
                vert_work_weights = std::move(ww);
            } else {
                for (auto vert : vertices()) {
                    vert_work_weights[vert] = ww[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_comm_weights = std::move(cw);
            } else {
                for (auto vert : vertices()) {
                    vert_comm_weights[vert] = cw[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_mem_weights = std::move(mw);
            } else {
                for (auto vert : vertices()) {
                    vert_mem_weights[vert] = mw[vertex_permutation_from_internal_to_original[vert]];
                }
            }
        }

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, const edge_list_type & edges, const std::vector<vertex_work_weight_type> &ww, const std::vector<vertex_comm_weight_type> &cw, const std::vector<vertex_mem_weight_type> &mw, const std::vector<vertex_type_type> &vt) : Compact_Sparse_Graph(num_vertices_, edges) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
            static_assert(use_comm_weights, "To set communication weight, graph type must allow communication weights.");
            static_assert(use_mem_weights, "To set memory weight, graph type must allow memory weights.");
            static_assert(use_vert_types, "To set vertex types, graph type must allow vertex types.");
            assert((ww.size() == static_cast<std::size_t>(num_vertices())) && "Work weights vector must have the same length as the number of vertices.");
            assert((cw.size() == static_cast<std::size_t>(num_vertices())) && "Communication weights vector must have the same length as the number of vertices.");
            assert((mw.size() == static_cast<std::size_t>(num_vertices())) && "Memory weights vector must have the same length as the number of vertices.");
            assert((vt.size() == static_cast<std::size_t>(num_vertices())) && "Vertex type vector must have the same length as the number of vertices.");

            if constexpr (keep_vertex_order) {
                vert_work_weights = ww;
            } else {
                for (auto vert : vertices()) {
                    vert_work_weights[vert] = ww[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_comm_weights = cw;
            } else {
                for (auto vert : vertices()) {
                    vert_comm_weights[vert] = cw[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_mem_weights = mw;
            } else {
                for (auto vert : vertices()) {
                    vert_mem_weights[vert] = mw[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_types = vt;
            } else {
                for (auto vert : vertices()) {
                    vert_types[vert] = vt[vertex_permutation_from_internal_to_original[vert]];
                }
            }
        }

        template <typename edge_list_type>
        Compact_Sparse_Graph(vertex_idx num_vertices_, const edge_list_type & edges, std::vector<vertex_work_weight_type> &&ww, std::vector<vertex_comm_weight_type> &&cw, std::vector<vertex_mem_weight_type> &&mw, std::vector<vertex_type_type> &&vt) : Compact_Sparse_Graph(num_vertices_, edges) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
            static_assert(use_comm_weights, "To set communication weight, graph type must allow communication weights.");
            static_assert(use_mem_weights, "To set memory weight, graph type must allow memory weights.");
            static_assert(use_vert_types, "To set vertex types, graph type must allow vertex types.");
            assert((ww.size() == static_cast<std::size_t>(num_vertices())) && "Work weights vector must have the same length as the number of vertices.");
            assert((cw.size() == static_cast<std::size_t>(num_vertices())) && "Communication weights vector must have the same length as the number of vertices.");
            assert((mw.size() == static_cast<std::size_t>(num_vertices())) && "Memory weights vector must have the same length as the number of vertices.");
            assert((vt.size() == static_cast<std::size_t>(num_vertices())) && "Vertex type vector must have the same length as the number of vertices.");

            if constexpr (keep_vertex_order) {
                vert_work_weights = std::move(ww);
            } else {
                for (auto vert : vertices()) {
                    vert_work_weights[vert] = ww[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_comm_weights = std::move(cw);
            } else {
                for (auto vert : vertices()) {
                    vert_comm_weights[vert] = cw[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_mem_weights = std::move(mw);
            } else {
                for (auto vert : vertices()) {
                    vert_mem_weights[vert] = mw[vertex_permutation_from_internal_to_original[vert]];
                }
            }

            if constexpr (keep_vertex_order) {
                vert_types = std::move(vt);
            } else {
                for (auto vert : vertices()) {
                    vert_types[vert] = vt[vertex_permutation_from_internal_to_original[vert]];
                }
            }
        }

        template <typename Graph_type>
        Compact_Sparse_Graph(const Graph_type  & graph) : Compact_Sparse_Graph(graph.num_vertices(), edge_view(graph)) {
            static_assert(is_directed_graph_v<Graph_type>);

            if constexpr (is_computational_dag_v<Graph_type> && use_work_weights) {
                for (const auto &vert : graph.vertices()) {
                    set_vertex_work_weight(vert, graph.vertex_work_weight(vert));
                }
            }

            if constexpr (is_computational_dag_v<Graph_type> && use_comm_weights) {
                for (const auto &vert : graph.vertices()) {
                    set_vertex_comm_weight(vert, graph.vertex_comm_weight(vert));
                }
            }

            if constexpr (is_computational_dag_v<Graph_type> && use_mem_weights) {
                for (const auto &vert : graph.vertices()) {
                    set_vertex_mem_weight(vert, graph.vertex_mem_weight(vert));
                }
            }

            if constexpr (is_computational_dag_typed_vertices_v<Graph_type> && use_vert_types) {
                for (const auto &vert : graph.vertices()) {
                    set_vertex_type(vert, graph.vertex_type(vert));
                }
            }
        }

        inline auto vertices() const { return vertex_range<vertex_idx>(number_of_vertices); };

        inline vert_t num_vertices() const { return number_of_vertices; };
        inline edge_t num_edges() const { return number_of_edges; }

        inline auto parents(const vertex_idx v) const { return csr_in_edges.parents(v); };
        inline auto children(const vertex_idx v) const { return csc_out_edges.children(v); };

        inline edge_t in_degree(const vertex_idx v) const {
            return csr_in_edges.number_of_parents(v);
        };
        inline edge_t out_degree(const vertex_idx v) const {
            return csc_out_edges.number_of_children(v);
        };

        template<typename RetT = vertex_work_weight_type>
        inline std::enable_if_t<use_work_weights, RetT> vertex_work_weight(const vertex_idx v) const {
            return vert_work_weights[v];
        };
        template<typename RetT = vertex_work_weight_type>
        inline std::enable_if_t<not use_work_weights, RetT> vertex_work_weight(const vertex_idx v) const {
            return static_cast<RetT>(1) + in_degree(v);
        };

        template<typename RetT = vertex_comm_weight_type>
        inline std::enable_if_t<use_comm_weights, RetT> vertex_comm_weight(const vertex_idx v) const {
            return vert_comm_weights[v];
        };
        template<typename RetT = vertex_comm_weight_type>
        inline std::enable_if_t<not use_comm_weights, RetT> vertex_comm_weight(const vertex_idx) const {
            return static_cast<RetT>(0);
        };

        template<typename RetT = vertex_mem_weight_type>
        inline std::enable_if_t<use_mem_weights, RetT> vertex_mem_weight(const vertex_idx v) const {
            return vert_mem_weights[v];
        };
        template<typename RetT = vertex_mem_weight_type>
        inline std::enable_if_t<not use_mem_weights, RetT> vertex_mem_weight(const vertex_idx) const {
            return static_cast<RetT>(0);
        };

        template<typename RetT = vertex_type_type>
        inline std::enable_if_t<use_vert_types, RetT> vertex_type(const vertex_idx v) const {
            return vert_types[v];
        };
        template<typename RetT = vertex_type_type>
        inline std::enable_if_t<not use_vert_types, RetT> vertex_type(const vertex_idx) const {
            return static_cast<RetT>(0);
        };

        inline vertex_type_type num_vertex_types() const { return number_of_vertex_types; };

        template<typename RetT = void>
        inline std::enable_if_t<use_work_weights, RetT> set_vertex_work_weight(const vertex_idx v, const vertex_work_weight_type work_weight) {
            if constexpr (keep_vertex_order) {
                vert_work_weights[v] = work_weight;
            } else {
                vert_work_weights[vertex_permutation_from_original_to_internal[v]] = work_weight;
            }
        };
        template<typename RetT = void>
        inline std::enable_if_t<not use_work_weights, RetT> set_vertex_work_weight(const vertex_idx v, const vertex_work_weight_type work_weight) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
        };

        template<typename RetT = void>
        inline std::enable_if_t<use_comm_weights, RetT> set_vertex_comm_weight(const vertex_idx v, const vertex_comm_weight_type comm_weight) {
            if constexpr (keep_vertex_order) {
                vert_comm_weights[v] = comm_weight;
            } else {
                vert_comm_weights[vertex_permutation_from_original_to_internal[v]] = comm_weight;
            }
        };
        template<typename RetT = void>
        inline std::enable_if_t<not use_comm_weights, RetT> set_vertex_comm_weight(const vertex_idx v, const vertex_comm_weight_type comm_weight) {
            static_assert(use_comm_weights, "To set comm weight, graph type must allow comm weights.");
        };
        
        template<typename RetT = void>
        inline std::enable_if_t<use_mem_weights, RetT> set_vertex_mem_weight(const vertex_idx v, const vertex_mem_weight_type mem_weight) {
            if constexpr (keep_vertex_order) {
                vert_mem_weights[v] = mem_weight;
            } else {
                vert_mem_weights[vertex_permutation_from_original_to_internal[v]] = mem_weight;
            }
        };
        template<typename RetT = void>
        inline std::enable_if_t<not use_mem_weights, RetT> set_vertex_mem_weight(const vertex_idx v, const vertex_mem_weight_type mem_weight) {
            static_assert(use_mem_weights, "To set mem weight, graph type must allow mem weights.");
        };
        
        template<typename RetT = void>
        inline std::enable_if_t<use_vert_types, RetT> set_vertex_type(const vertex_idx v, const vertex_type_type vertex_type_) {
            if constexpr (keep_vertex_order) {
                vert_types[v] = vertex_type_;
            } else {
                vert_types[vertex_permutation_from_original_to_internal[v]] = vertex_type_;
            }
            number_of_vertex_types = std::max(number_of_vertex_types, vertex_type_);
        };
        template<typename RetT = void>
        inline std::enable_if_t<not use_vert_types, RetT> set_vertex_type(const vertex_idx v, const vertex_type_type vertex_type_) {
            static_assert(use_vert_types, "To set vert type, graph type must allow vertex types.");
        };

        template<typename RetT = const std::vector<vertex_idx> &>
        inline std::enable_if_t<keep_vertex_order, RetT> get_pullback_permutation() const {
            static_assert(!keep_vertex_order, "No permutation was applied. This is a deleted function.");
            return {};
        }

        template<typename RetT = const std::vector<vertex_idx> &>
        inline std::enable_if_t<not keep_vertex_order, RetT> get_pullback_permutation() const {
            return vertex_permutation_from_internal_to_original;
        }

        template<typename RetT = const std::vector<vertex_idx> &>
        inline std::enable_if_t<keep_vertex_order, RetT> get_pushforward_permutation() const {
            static_assert(!keep_vertex_order, "No permutation was applied. This is a deleted function.");
            return {};
        }

        template<typename RetT = const std::vector<vertex_idx> &>
        inline std::enable_if_t<not keep_vertex_order, RetT> get_pushforward_permutation() const {
            return vertex_permutation_from_original_to_internal;
        }
};

static_assert(has_vertex_weights_v<Compact_Sparse_Graph<true, true>>, 
    "Compact_Sparse_Graph must satisfy the has_vertex_weights concept");

static_assert(has_vertex_weights_v<Compact_Sparse_Graph<false, true>>, 
    "Compact_Sparse_Graph must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph<false, false, false, false, false>>, 
    "Compact_Sparse_Graph must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph<false, true, true, true, true>>, 
    "Compact_Sparse_Graph must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph<true, false, false, false, false>>, 
    "Compact_Sparse_Graph must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph<true, true, true, true, true>>, 
    "Compact_Sparse_Graph must satisfy the directed_graph concept");

static_assert(is_computational_dag_v<Compact_Sparse_Graph<false, true, true, true, false>>, 
    "Compact_Sparse_Graph must satisfy the is_computation_dag concept");

static_assert(is_computational_dag_v<Compact_Sparse_Graph<true, true, true, true, false>>, 
    "Compact_Sparse_Graph must satisfy the is_computation_dag concept");

static_assert(is_computational_dag_typed_vertices_v<Compact_Sparse_Graph<false, true, true, true, true>>,
    "Compact_Sparse_Graph must satisfy the is_computation_dag with types concept");

static_assert(is_computational_dag_typed_vertices_v<Compact_Sparse_Graph<true, true, true, true, true>>,
    "Compact_Sparse_Graph must satisfy the is_computation_dag with types concept");


template<bool keep_vertex_order, bool use_work_weights, bool use_comm_weights, bool use_mem_weights, bool use_vert_types, typename vert_t, typename edge_t, typename work_weight_type, typename comm_weight_type, typename mem_weight_type, typename vertex_type_template_type>
std::vector<vert_t> GetTopOrder(const Compact_Sparse_Graph<keep_vertex_order, use_work_weights, use_comm_weights, use_mem_weights, use_vert_types, vert_t, edge_t, work_weight_type, comm_weight_type, mem_weight_type, vertex_type_template_type> &graph) {
    std::vector<vert_t> topOrd(graph.num_vertices());
    std::iota(topOrd.begin(), topOrd.end(), static_cast<vert_t>(0));
    return topOrd;
}

} // namespace osp