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
#include "graph_implementations/vertex_iterator.hpp"

namespace osp {

template<bool keep_vertex_order, bool use_comm_weights = false, bool use_mem_weights = false, bool use_vert_types = false, bool use_work_weights = true, typename vert_t = std::size_t, typename edge_t = std::size_t>
class Compact_Sparse_Graph {
    static_assert(std::is_integral<vert_t>::value && std::is_integral<edge_t>::value, "Vertex and edge type must be of integral nature.");

    public:
        using vertex_idx = vert_t;

        using vertex_work_weight_type = edge_t;
        using vertex_comm_weight_type = unsigned;
        using vertex_mem_weight_type = unsigned;
        using vertex_type_type = unsigned;

        using edge_comm_weight_type = unsigned;

    private:
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
                Compact_Parent_Edges(std::vector<vertex_idx> &&csr_edge_parents_, std::vector<edge_t> &&csr_target_ptr_) : csr_edge_parents(csr_edge_parents_), csr_target_ptr(csr_target_ptr_) {};

                inline edge_t number_of_parents(const vertex_idx v) const {
                    return csr_target_ptr[v + 1] - csr_target_ptr[v];
                }

                class iterator {
                    public:
                        using iterator_category = std::random_access_iterator_tag;
                        using difference_type = vertex_idx;
                        using value_type = vertex_idx;
                        using pointer = vertex_idx *;
                        using reference = vertex_idx &;

                    private:
                        const std::vector<vertex_idx> &_csr_edge_parents;
                        edge_t _index;

                    public:
                        iterator(const std::vector<vertex_idx> &_csr_edge_parents_, edge_t index) : _csr_edge_parents(_csr_edge_parents_), _index(index) {};

                        inline edge_t get_index() { return _index; }

                        inline reference operator*() const { return _csr_edge_parents[_index]; }
                        inline pointer operator->() const { auto it = _csr_edge_parents.begin(); std::advance(it, _index); return it; }
                        inline reference operator[](difference_type diff) const { return _csr_edge_parents[_index + diff]; }

                        inline iterator &operator++() { ++_index; return *this; }
                        inline iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }

                        inline iterator &operator--() { --_index; return *this; }
                        inline iterator operator--(int) { iterator tmp = *this; --(*this); return tmp; }

                        inline iterator operator+(difference_type diff) const { return iterator(_csr_edge_parents, _index + diff); }
                        inline iterator operator-(difference_type diff) const { return iterator(_csr_edge_parents, _index - diff); }

                        inline bool operator==(const iterator &other) const { return _index == other._index; }
                        inline bool operator!=(const iterator &other) const { return !(*this == other); }
                        
                        inline bool operator<=(const iterator &other) const { return _index <= other._index; }
                        inline bool operator<(const iterator &other) const { return (*this <= other) && (*this != other); }
                        inline bool operator>=(const iterator &other) const { return (!(*this <= other)) || (*this == other); }
                        inline bool operator>(const iterator &other) const { return !(*this <= other); }
                };

                class const_iterator {
                    public:
                        using iterator_category = std::random_access_iterator_tag;
                        using difference_type = vertex_idx;
                        using value_type = vertex_idx;
                        using pointer = const vertex_idx *;
                        using reference = const vertex_idx &;

                    private:
                        const std::vector<vertex_idx> &_csr_edge_parents;
                        edge_t _index;

                    public:
                        const_iterator(const std::vector<vertex_idx> &_csr_edge_parents_, edge_t index) : _csr_edge_parents(_csr_edge_parents_), _index(index) {};

                        inline edge_t get_index() { return _index; }

                        inline const reference operator*() const { return _csr_edge_parents[_index]; }
                        inline pointer operator->() const { auto it = _csr_edge_parents.begin(); std::advance(it, _index); return it; }
                        inline reference operator[](difference_type diff) const { return _csr_edge_parents[_index + diff]; }

                        inline const_iterator &operator++() { ++_index; return *this; }
                        inline const_iterator operator++(int) { const_iterator tmp = *this; ++(*this); return tmp; }

                        inline const_iterator &operator--() { --_index; return *this; }
                        inline const_iterator operator--(int) { const_iterator tmp = *this; --(*this); return tmp; }

                        inline const_iterator operator+(difference_type diff) const { return const_iterator(_csr_edge_parents, _index + diff); }
                        inline const_iterator operator-(difference_type diff) const { return const_iterator(_csr_edge_parents, _index - diff); }

                        inline bool operator==(const const_iterator &other) const { return _index == other._index; }
                        inline bool operator!=(const const_iterator &other) const { return !(*this == other); }
                        
                        inline bool operator<=(const const_iterator &other) const { return _index <= other._index; }
                        inline bool operator<(const const_iterator &other) const { return (*this <= other) && (*this != other); }
                        inline bool operator>=(const const_iterator &other) const { return (!(*this <= other)) || (*this == other); }
                        inline bool operator>(const const_iterator &other) const { return !(*this <= other); }
                };

                class reverse_iterator {
                    public:
                        using iterator_category = std::random_access_iterator_tag;
                        using difference_type = vertex_idx;
                        using value_type = vertex_idx;
                        using pointer = vertex_idx *;
                        using reference = vertex_idx &;

                    private:
                        const std::vector<vertex_idx> &_csr_edge_parents;
                        edge_t _index;

                    public:
                        reverse_iterator(const std::vector<vertex_idx> &_csr_edge_parents_, edge_t index) : _csr_edge_parents(_csr_edge_parents_), _index(index) {};

                        inline edge_t get_index() { return _index - 1; }

                        inline reference operator*() const { return csr_edge_parents[_index - 1]; }
                        inline pointer operator->() const { auto it = csr_edge_parents.begin(); std::advance(it, _index - 1); return it; }
                        inline reference operator[](difference_type diff) const { return csr_edge_parents[_index - 1 - diff]; }

                        inline reverse_iterator &operator++() { --_index; return *this; }
                        inline reverse_iterator operator++(int) { reverse_iterator tmp = *this; ++(*this); return tmp; }

                        inline reverse_iterator &operator--() { ++_index; return *this; }
                        inline reverse_iterator operator--(int) { reverse_iterator tmp = *this; --(*this); return tmp; }

                        inline reverse_iterator operator+(difference_type diff) const { return reverse_iterator(_csr_edge_parents, _index - diff); }
                        inline reverse_iterator operator-(difference_type diff) const { return reverse_iterator(_csr_edge_parents, _index + diff); }

                        inline bool operator==(const reverse_iterator &other) const { return _index == other._index; }
                        inline bool operator!=(const reverse_iterator &other) const { return !(*this == other); }
                        
                        inline bool operator<=(const reverse_iterator &other) const { return _index >= other._index; }
                        inline bool operator<(const reverse_iterator &other) const { return (*this <= other) && (*this != other); }
                        inline bool operator>=(const reverse_iterator &other) const { return (!(*this <= other)) || (*this == other); }
                        inline bool operator>(const reverse_iterator &other) const { return !(*this <= other); }
                };

                class const_reverse_iterator {
                    public:
                        using iterator_category = std::random_access_iterator_tag;
                        using difference_type = vertex_idx;
                        using value_type = vertex_idx;
                        using pointer = const vertex_idx *;
                        using reference = const vertex_idx &;

                    private:
                        const std::vector<vertex_idx> &_csr_edge_parents;
                        edge_t _index;

                    public:
                        const_reverse_iterator(const std::vector<vertex_idx> &_csr_edge_parents_, edge_t index) : _csr_edge_parents(_csr_edge_parents_), _index(index) {};

                        inline edge_t get_index() { return _index - 1; }

                        inline reference operator*() const { return csr_edge_parents[_index - 1]; }
                        inline pointer operator->() const { auto it = csr_edge_parents.begin(); std::advance(it, _index - 1); return it; }
                        inline reference operator[](difference_type diff) const { return csr_edge_parents[_index - 1 - diff]; }

                        inline const_reverse_iterator &operator++() { --_index; return *this; }
                        inline const_reverse_iterator operator++(int) { const_reverse_iterator tmp = *this; ++(*this); return tmp; }

                        inline const_reverse_iterator &operator--() { ++_index; return *this; }
                        inline const_reverse_iterator operator--(int) { const_reverse_iterator tmp = *this; --(*this); return tmp; }

                        inline const_reverse_iterator operator+(difference_type diff) const { return const_reverse_iterator(_csr_edge_parents, _index - diff); }
                        inline const_reverse_iterator operator-(difference_type diff) const { return const_reverse_iterator(_csr_edge_parents, _index + diff); }

                        inline bool operator==(const const_reverse_iterator &other) const { return _index == other._index; }
                        inline bool operator!=(const const_reverse_iterator &other) const { return !(*this == other); }
                        
                        inline bool operator<=(const const_reverse_iterator &other) const { return _index >= other._index; }
                        inline bool operator<(const const_reverse_iterator &other) const { return (*this <= other) && (*this != other); }
                        inline bool operator>=(const const_reverse_iterator &other) const { return (!(*this <= other)) || (*this == other); }
                        inline bool operator>(const const_reverse_iterator &other) const { return !(*this <= other); }
                };

                class Parent_range {
                    private:
                        const std::vector<vertex_idx> &_csr_edge_parents;
                        const std::vector<edge_t> &_csr_target_ptr;
                        const vertex_idx _vert;

                    public:
                        Parent_range (const std::vector<vertex_idx> &csr_edge_parents, const std::vector<edge_t> &csr_target_ptr, const vertex_idx vert) : _csr_edge_parents(csr_edge_parents), _csr_target_ptr(csr_target_ptr), _vert(vert) {};

                        inline auto begin() const { return iterator(_csr_edge_parents, _csr_target_ptr[_vert]); }
                        inline auto end() const { return iterator(_csr_edge_parents, _csr_target_ptr[_vert + 1]); }

                        inline auto cbegin() const { return const_iterator(_csr_edge_parents, _csr_target_ptr[_vert]); }
                        inline auto cend() const { return const_iterator(_csr_edge_parents, _csr_target_ptr[_vert + 1]); }

                        inline auto rbegin() const { return reverse_iterator(_csr_edge_parents, _csr_target_ptr[_vert + 1]); }
                        inline auto rend() const { return reverse_iterator(_csr_edge_parents, _csr_target_ptr[_vert]); }

                        inline auto crbegin() const { return const_reverse_iterator(_csr_edge_parents, _csr_target_ptr[_vert + 1]); }
                        inline auto crend() const { return const_reverse_iterator(_csr_edge_parents, _csr_target_ptr[_vert]); }
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
                Compact_Children_Edges(std::vector<vertex_idx> &&csc_edge_children_, std::vector<edge_t> &&csc_source_ptr_) : csc_edge_children(csc_edge_children_), csc_source_ptr(csc_source_ptr_) {};

                inline edge_t number_of_children(const vertex_idx v) const {
                    return csc_source_ptr[v + 1] - csc_source_ptr[v];
                }

                class iterator {
                    public:
                        using iterator_category = std::random_access_iterator_tag;
                        using difference_type = vertex_idx;
                        using value_type = vertex_idx;
                        using pointer = vertex_idx *;
                        using reference = vertex_idx &;

                    private:
                        const std::vector<vertex_idx> &_csc_edge_children;
                        edge_t _index;

                    public:
                        iterator(const std::vector<vertex_idx> &csc_edge_children, edge_t index) : _csc_edge_children(csc_edge_children), _index(index) {};

                        inline edge_t get_index() { return _index; }

                        inline reference operator*() const { return _csc_edge_children[_index]; }
                        inline pointer operator->() const { auto it = _csc_edge_children.begin(); std::advance(it, _index); return it; }
                        inline reference operator[](difference_type diff) const { return _csc_edge_children[_index + diff]; }

                        inline iterator &operator++() { ++_index; return *this; }
                        inline iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }

                        inline iterator &operator--() { --_index; return *this; }
                        inline iterator operator--(int) { iterator tmp = *this; --(*this); return tmp; }

                        inline iterator operator+(difference_type diff) const { return iterator(_csc_edge_children, _index + diff); }
                        inline iterator operator-(difference_type diff) const { return iterator(_csc_edge_children, _index - diff); }

                        inline bool operator==(const iterator &other) const { return _index == other._index; }
                        inline bool operator!=(const iterator &other) const { return !(*this == other); }
                        
                        inline bool operator<=(const iterator &other) const { return _index <= other._index; }
                        inline bool operator<(const iterator &other) const { return (*this <= other) && (*this != other); }
                        inline bool operator>=(const iterator &other) const { return (!(*this <= other)) || (*this == other); }
                        inline bool operator>(const iterator &other) const { return !(*this <= other); }
                };

                class const_iterator {
                    public:
                        using iterator_category = std::random_access_iterator_tag;
                        using difference_type = vertex_idx;
                        using value_type = vertex_idx;
                        using pointer = const vertex_idx *;
                        using reference = const vertex_idx &;

                    private:
                        const std::vector<vertex_idx> &_csc_edge_children;
                        edge_t _index;

                    public:
                        const_iterator(const std::vector<vertex_idx> &csc_edge_children, edge_t index) : _csc_edge_children(csc_edge_children), _index(index) {};

                        inline edge_t get_index() { return _index; }

                        inline const reference operator*() const { return _csc_edge_children[_index]; }
                        inline pointer operator->() const { auto it = _csc_edge_children.begin(); std::advance(it, _index); return it; }
                        inline reference operator[](difference_type diff) const { return _csc_edge_children[_index + diff]; }

                        inline const_iterator &operator++() { ++_index; return *this; }
                        inline const_iterator operator++(int) { const_iterator tmp = *this; ++(*this); return tmp; }

                        inline const_iterator &operator--() { --_index; return *this; }
                        inline const_iterator operator--(int) { const_iterator tmp = *this; --(*this); return tmp; }

                        inline const_iterator operator+(difference_type diff) const { return const_iterator(_csc_edge_children, _index + diff); }
                        inline const_iterator operator-(difference_type diff) const { return const_iterator(_csc_edge_children, _index - diff); }

                        inline bool operator==(const const_iterator &other) const { return _index == other._index; }
                        inline bool operator!=(const const_iterator &other) const { return !(*this == other); }
                        
                        inline bool operator<=(const const_iterator &other) const { return _index <= other._index; }
                        inline bool operator<(const const_iterator &other) const { return (*this <= other) && (*this != other); }
                        inline bool operator>=(const const_iterator &other) const { return (!(*this <= other)) || (*this == other); }
                        inline bool operator>(const const_iterator &other) const { return !(*this <= other); }
                };

                class reverse_iterator {
                    public:
                        using iterator_category = std::random_access_iterator_tag;
                        using difference_type = vertex_idx;
                        using value_type = vertex_idx;
                        using pointer = vertex_idx *;
                        using reference = vertex_idx &;

                    private:
                        const std::vector<vertex_idx> &_csc_edge_children;
                        edge_t _index;

                    public:
                        reverse_iterator(const std::vector<vertex_idx> &csc_edge_children, edge_t index) : _csc_edge_children(csc_edge_children), _index(index) {};

                        inline edge_t get_index() { return _index - 1; }

                        inline reference operator*() const { return _csc_edge_children[_index - 1]; }
                        inline pointer operator->() const { auto it = _csc_edge_children.begin(); std::advance(it, _index - 1); return it; }
                        inline reference operator[](difference_type diff) const { return _csc_edge_children[_index - 1 - diff]; }

                        inline reverse_iterator &operator++() { --_index; return *this; }
                        inline reverse_iterator operator++(int) { reverse_iterator tmp = *this; ++(*this); return tmp; }

                        inline reverse_iterator &operator--() { ++_index; return *this; }
                        inline reverse_iterator operator--(int) { reverse_iterator tmp = *this; --(*this); return tmp; }

                        inline reverse_iterator operator+(difference_type diff) const { return reverse_iterator(_csc_edge_children, _index - diff); }
                        inline reverse_iterator operator-(difference_type diff) const { return reverse_iterator(_csc_edge_children, _index + diff); }

                        inline bool operator==(const reverse_iterator &other) const { return _index == other._index; }
                        inline bool operator!=(const reverse_iterator &other) const { return !(*this == other); }
                        
                        inline bool operator<=(const reverse_iterator &other) const { return _index >= other._index; }
                        inline bool operator<(const reverse_iterator &other) const { return (*this <= other) && (*this != other); }
                        inline bool operator>=(const reverse_iterator &other) const { return (!(*this <= other)) || (*this == other); }
                        inline bool operator>(const reverse_iterator &other) const { return !(*this <= other); }
                };

                class const_reverse_iterator {
                    public:
                        using iterator_category = std::random_access_iterator_tag;
                        using difference_type = vertex_idx;
                        using value_type = vertex_idx;
                        using pointer = const vertex_idx *;
                        using reference = const vertex_idx &;

                    private:
                        const std::vector<vertex_idx> &_csc_edge_children;
                        edge_t _index;

                    public:
                        const_reverse_iterator(const std::vector<vertex_idx> &csc_edge_children, edge_t index) : _csc_edge_children(csc_edge_children), _index(index) {};

                        inline edge_t get_index() { return _index - 1; }

                        inline reference operator*() const { return _csc_edge_children[_index - 1]; }
                        inline pointer operator->() const { auto it = _csc_edge_children.begin(); std::advance(it, _index - 1); return it; }
                        inline reference operator[](difference_type diff) const { return _csc_edge_children[_index - 1 - diff]; }

                        inline const_reverse_iterator &operator++() { --_index; return *this; }
                        inline const_reverse_iterator operator++(int) { const_reverse_iterator tmp = *this; ++(*this); return tmp; }

                        inline const_reverse_iterator &operator--() { ++_index; return *this; }
                        inline const_reverse_iterator operator--(int) { const_reverse_iterator tmp = *this; --(*this); return tmp; }

                        inline const_reverse_iterator operator+(difference_type diff) const { return const_reverse_iterator(_csc_edge_children, _index - diff); }
                        inline const_reverse_iterator operator-(difference_type diff) const { return const_reverse_iterator(_csc_edge_children, _index + diff); }

                        inline bool operator==(const const_reverse_iterator &other) const { return _index == other._index; }
                        inline bool operator!=(const const_reverse_iterator &other) const { return !(*this == other); }
                        
                        inline bool operator<=(const const_reverse_iterator &other) const { return _index >= other._index; }
                        inline bool operator<(const const_reverse_iterator &other) const { return (*this <= other) && (*this != other); }
                        inline bool operator>=(const const_reverse_iterator &other) const { return (!(*this <= other)) || (*this == other); }
                        inline bool operator>(const const_reverse_iterator &other) const { return !(*this <= other); }
                };

                class Children_range {
                    private:
                        const std::vector<vertex_idx> &_csc_edge_children;
                        const std::vector<edge_t> &_csc_source_ptr;
                        const vertex_idx _vert;

                    public:
                        Children_range (const std::vector<vertex_idx> &csc_edge_children, const std::vector<edge_t> &csc_source_ptr, const vertex_idx vert) : _csc_edge_children(csc_edge_children), _csc_source_ptr(csc_source_ptr), _vert(vert) {};

                        inline auto begin() const { return iterator(_csc_edge_children, _csc_source_ptr[_vert]); }
                        inline auto end() const { return iterator(_csc_edge_children, _csc_source_ptr[_vert + 1]); }

                        inline auto cbegin() const { return const_iterator(_csc_edge_children, _csc_source_ptr[_vert]); }
                        inline auto cend() const { return const_iterator(_csc_edge_children, _csc_source_ptr[_vert + 1]); }

                        inline auto rbegin() const { return reverse_iterator(_csc_edge_children, _csc_source_ptr[_vert + 1]); }
                        inline auto rend() const { return reverse_iterator(_csc_edge_children, _csc_source_ptr[_vert]); }

                        inline auto crbegin() const { return const_reverse_iterator(_csc_edge_children, _csc_source_ptr[_vert + 1]); }
                        inline auto crend() const { return const_reverse_iterator(_csc_edge_children, _csc_source_ptr[_vert]); }
                };

                inline Children_range children(const vertex_idx vert) const { return Children_range(csc_edge_children, csc_source_ptr, vert); }
        };

    public:
        Compact_Sparse_Graph() = default;
        Compact_Sparse_Graph(const Compact_Sparse_Graph &other) = default;
        Compact_Sparse_Graph(Compact_Sparse_Graph &&other) = default;
        Compact_Sparse_Graph &operator=(const Compact_Sparse_Graph &other) = default;
        Compact_Sparse_Graph &operator=(Compact_Sparse_Graph &&other) = default;
        virtual ~Compact_Sparse_Graph() = default;

        // TODO more constructors
        Compact_Sparse_Graph(vertex_idx num_vertices_, const std::vector<std::pair<vertex_idx, vertex_idx>> &edges) {
            assert((0 <= num_vertices_) && "Number of vertices must be non-negative.");
            assert(std::all_of(edges.cbegin(), edges.cend(), [num_vertices_](const std::pair<vertex_idx, vertex_idx> &edge) { return (0 <= edge.first) && (edge.first < num_vertices_) && (0 <= edge.second) && (edge.second < num_vertices_); } ) && "Source and target of edges must be non-negative and less than the number of vertices.");
            assert((edges.size() < static_cast<size_t>(std::numeric_limits<edge_t>::max())) && "Number of edge must be strictly smaller than the maximally representable number.");
            if constexpr (keep_vertex_order) {
                assert(std::all_of(edges.cbegin(), edges.cend(), [](const std::pair<vertex_idx, vertex_idx> &edge) { return edge.first < edge.second; } ) && "Vertex order must be a topological order.");
            }

            number_of_vertices = num_vertices_;
            number_of_edges = static_cast<edge_t>(edges.size());

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
            }

            // Construction
            std::vector<std::vector<vertex_idx>> children_tmp(num_vertices());
            std::vector<edge_t> num_parents_tmp(num_vertices(), 0);
            for (const auto &edge : edges) {
                children_tmp[edge.first].push_back(edge.second);
                num_parents_tmp[edge.second]++;
            }

            std::vector<vertex_idx> csc_edge_children;
            csc_edge_children.reserve(num_edges());
            std::vector<edge_t> csc_source_ptr;
            csc_source_ptr.reserve(num_vertices() + 1);
            std::vector<vertex_idx> csr_edge_parents;
            csr_edge_parents.reserve(num_edges());
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
                for (const auto &edge : edges) {
                    parents_tmp[edge.second].push_back(edge.first);
                }

                // Generating modified Gorder topological order cf. "Speedup Graph Processing by Graph Ordering" by Hao Wei, Jeffrey Xu Yu, Can Lu, and Xuemin Lin
                const double decay = 5.0;

                std::vector<edge_t> prec_remaining = num_parents_tmp;
                std::vector<double> priorities(num_vertices(), 0.0);

                auto v_cmp = [&priorities, &children_tmp] (const vertex_idx &lhs, const vertex_idx &rhs) {
                    return  (priorities[lhs] < priorities[rhs]) ||
                            ((priorities[lhs] == priorities[rhs]) && (children_tmp[lhs].size() > children_tmp[rhs].size())) ||
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

                    // update constraints and push to queue
                    for (vertex_idx chld : children_tmp[vert]) {
                        --prec_remaining[chld];
                        if (prec_remaining[chld] == 0) {
                            ready_q.push(chld);
                        }
                    }
                }

                assert(vertex_permutation_from_internal_to_original.size() == static_cast<size_t>(num_vertices()));

                std::vector<vertex_idx> vert_position(num_vertices(), 0);
                for (vertex_idx new_pos = 0; new_pos < num_vertices(); ++new_pos) {
                    vert_position[vertex_permutation_from_internal_to_original[new_pos]] = new_pos;
                }

                for (vertex_idx vert_new_pos = 0; vert_new_pos < num_vertices(); ++vert_new_pos) {
                    csc_source_ptr[vert_new_pos] = static_cast<edge_t>( csc_edge_children.size() );

                    vertex_idx vert_old_name = vertex_permutation_from_internal_to_original[vert_new_pos];

                    std::vector<vertex_idx> children_new_name;
                    children_new_name.reserve( children_tmp[vert_old_name].size() );

                    for (vertex_idx chld_old_name : children_tmp[vert_old_name]) {
                        children_new_name.push_back( vert_position[chld_old_name] );
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

            csc_out_edges = Compact_Children_Edges(csc_edge_children, csc_source_ptr);
            csr_in_edges = Compact_Parent_Edges(csr_edge_parents, csr_target_ptr);
        };

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
        inline std::enable_if_t<not use_comm_weights, RetT> vertex_comm_weight(const vertex_idx v) const {
            return static_cast<RetT>(0);
        };

        template<typename RetT = vertex_mem_weight_type>
        inline std::enable_if_t<use_mem_weights, RetT> vertex_mem_weight(const vertex_idx v) const {
            return vert_mem_weights[v];
        };
        template<typename RetT = vertex_mem_weight_type>
        inline std::enable_if_t<not use_mem_weights, RetT> vertex_mem_weight(const vertex_idx v) const {
            return static_cast<RetT>(0);
        };

        template<typename RetT = vertex_type_type>
        inline std::enable_if_t<use_vert_types, RetT> vertex_type(const vertex_idx v) const {
            return vert_types[v];
        };
        template<typename RetT = vertex_type_type>
        inline std::enable_if_t<not use_vert_types, RetT> vertex_type(const vertex_idx v) const {
            return static_cast<RetT>(0);
        };

        inline vertex_type_type num_vertex_types() const { return number_of_vertex_types; };

        template<typename RetT = void>
        inline std::enable_if_t<use_work_weights, RetT> set_vertex_work_weight(const vertex_idx v, const vertex_work_weight_type work_weight) {
            vert_work_weights[v] = work_weight;
        };
        template<typename RetT = void>
        inline std::enable_if_t<not use_work_weights, RetT> set_vertex_work_weight(const vertex_idx v, const vertex_work_weight_type work_weight) {
            static_assert(use_work_weights, "To set work weight, graph type must allow work weights.");
        };

        template<typename RetT = void>
        inline std::enable_if_t<use_comm_weights, RetT> set_vertex_comm_weight(const vertex_idx v, const vertex_comm_weight_type comm_weight) {
            vert_comm_weights[v] = comm_weight;
        };
        template<typename RetT = void>
        inline std::enable_if_t<not use_comm_weights, RetT> set_vertex_comm_weight(const vertex_idx v, const vertex_comm_weight_type comm_weight) {
            static_assert(use_comm_weights, "To set comm weight, graph type must allow comm weights.");
        };
        
        template<typename RetT = void>
        inline std::enable_if_t<use_mem_weights, RetT> set_vertex_mem_weight(const vertex_idx v, const vertex_mem_weight_type mem_weight) {
            vert_mem_weights[v] = mem_weight;
        };
        template<typename RetT = void>
        inline std::enable_if_t<not use_mem_weights, RetT> set_vertex_mem_weight(const vertex_idx v, const vertex_mem_weight_type mem_weight) {
            static_assert(use_mem_weights, "To set mem weight, graph type must allow mem weights.");
        };
        
        template<typename RetT = void>
        inline std::enable_if_t<use_vert_types, RetT> set_vertex_type(const vertex_idx v, const vertex_type_type vertex_type_) {
            vert_types[v] = vertex_type_;
            number_of_vertex_types = std::max(number_of_vertex_types, vertex_type_);
        };
        template<typename RetT = void>
        inline std::enable_if_t<not use_vert_types, RetT> set_vertex_type(const vertex_idx v, const vertex_type_type vertex_type_) {
            static_assert(use_vert_types, "To set vert type, graph type must allow vertex types.");
        };


    private:
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

        template<typename RetT = void>
        std::enable_if_t<use_vert_types, RetT> _update_num_vertex_types() {
            number_of_vertex_types = static_cast<vertex_type_type>(1);
            for (const auto vt : vert_types) {
                number_of_vertex_types = std::max(number_of_vertex_types, vt);
            }
        }
            
        template<typename RetT = void>
        std::enable_if_t<not use_vert_types, RetT> _update_num_vertex_types() {
            number_of_vertex_types = static_cast<vertex_type_type>(1);
        }
};

static_assert(has_vertex_weights_v<Compact_Sparse_Graph<true>>, 
    "Compact_Sparse_Graph must satisfy the has_vertex_weights concept");

static_assert(has_vertex_weights_v<Compact_Sparse_Graph<false>>, 
    "Compact_Sparse_Graph must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph<false, false, false, false, false>>, 
    "Compact_Sparse_Graph must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph<false, true, true, true, true>>, 
    "Compact_Sparse_Graph must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph<true, false, false, false, false>>, 
    "Compact_Sparse_Graph must satisfy the directed_graph concept");

static_assert(is_directed_graph_v<Compact_Sparse_Graph<true, true, true, true, true>>, 
    "Compact_Sparse_Graph must satisfy the directed_graph concept");

static_assert(is_computational_dag_v<Compact_Sparse_Graph<false, true, true, false, true>>, 
    "Compact_Sparse_Graph must satisfy the is_computation_dag concept");

static_assert(is_computational_dag_v<Compact_Sparse_Graph<true, true, true, false, true>>, 
    "Compact_Sparse_Graph must satisfy the is_computation_dag concept");

static_assert(is_computational_dag_typed_vertices_v<Compact_Sparse_Graph<false, true, true, true, true>>,
    "Compact_Sparse_Graph must satisfy the is_computation_dag with types concept");

static_assert(is_computational_dag_typed_vertices_v<Compact_Sparse_Graph<true, true, true, true, true>>,
    "Compact_Sparse_Graph must satisfy the is_computation_dag with types concept");

} // namespace osp