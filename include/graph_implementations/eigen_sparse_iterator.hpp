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

@author Christos Matzoros, Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
*/



#pragma once

#ifdef EIGEN_FOUND

#include <Eigen/SparseCore>
#include "concepts/graph_traits.hpp"

namespace osp {

template<typename Graph, typename eigen_idx_type>
class EigenCSRRange {
    const Graph& graph_;
    eigen_idx_type index_;

public:
    using CSRMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
    using Inner = typename CSRMatrix::InnerIterator;

    class iterator {
        Inner it_;
        eigen_idx_type skip_;
        bool at_end_;

        void skip_diagonal() {
            while (!at_end_ && it_.row() == static_cast<Eigen::Index>(skip_) && it_.col() == static_cast<Eigen::Index>(skip_)) {
                ++(*this);
            }
        }

    public:
        using value_type = std::size_t;
        using reference = value_type;
        using pointer = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        iterator(const CSRMatrix& mat, eigen_idx_type idx, bool end = false)
            : skip_(idx), at_end_(end) {
            if (!end) {
                it_ = Inner(mat, static_cast<Eigen::Index>(idx));
                at_end_ = !it_;
                skip_diagonal();
            }
        }

        reference operator*() const { return static_cast<std::size_t>(it_.col()); }
        iterator& operator++() {
            ++it_;
            at_end_ = !it_;
            skip_diagonal();
            return *this;
        }

        bool operator!=(const iterator&) const { return !at_end_; }
    };

    EigenCSRRange(const Graph& graph, eigen_idx_type idx)
        : graph_(graph), index_(idx) {}

    iterator begin() const {
        return iterator(*static_cast<const CSRMatrix*>(graph_.getCSR()), index_);
    }

    iterator end() const {
        return iterator(*static_cast<const CSRMatrix*>(graph_.getCSR()), index_, true);
    }
};


template<typename Graph, typename eigen_idx_type>
class EigenCSCRange {
    const Graph& graph_;
    eigen_idx_type index_;

public:
    using CSCMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
    using Inner = typename CSCMatrix::InnerIterator;

    class iterator {
        Inner it_;
        eigen_idx_type skip_;
        bool at_end_;

        void skip_diagonal() {
            while (!at_end_ && it_.row() == static_cast<Eigen::Index>(skip_) && it_.col() == static_cast<Eigen::Index>(skip_)) {
                ++(*this);
            }
        }
        
    public:
        using value_type = std::size_t;
        using reference = value_type;
        using pointer = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        iterator(const CSCMatrix& mat, eigen_idx_type idx, bool end = false)
            : skip_(idx), at_end_(end) {
            if (!end) {
                it_ = Inner(mat, static_cast<Eigen::Index>(idx));
                at_end_ = !it_;
                skip_diagonal();
            }
        }

        reference operator*() const { return static_cast<std::size_t>(it_.row()); }
        iterator& operator++() {
            ++it_;
            at_end_ = !it_;
            skip_diagonal();
            return *this;
        }

        bool operator!=(const iterator&) const { return !at_end_; }
    };

    EigenCSCRange(const Graph& graph, eigen_idx_type idx)
        : graph_(graph), index_(idx) {}

    iterator begin() const {
        return iterator(*static_cast<const CSCMatrix*>(graph_.getCSC()), index_);
    }

    iterator end() const {
        return iterator(*static_cast<const CSCMatrix*>(graph_.getCSC()), index_, true);
    }
};
} // namespace osp

#endif
