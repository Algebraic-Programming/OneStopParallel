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

#    include <Eigen/SparseCore>

#    include "osp/concepts/graph_traits.hpp"

namespace osp {

template <typename Graph, typename EigenIdxType>
class EigenCSRRange {
    const Graph &graph_;
    EigenIdxType index_;

  public:
    using CSRMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, EigenIdxType>;
    using Inner = typename CSRMatrix::InnerIterator;

    class Iterator {
        Inner it_;
        EigenIdxType skip_;
        bool atEnd_;

        void SkipDiagonal() {
            while (((!atEnd_) && (it_.row() == skip_)) & (it_.col() == skip_)) {
                ++(*this);
            }
        }

      public:
        using value_type = std::size_t;
        using reference = value_type;
        using pointer = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        Iterator() = default;

        Iterator(const Iterator &other) : it_(other.it_), skip_(other.skip_), atEnd_(other.atEnd_) {}

        Iterator &operator=(const Iterator &other) {
            it_ = other.it_;
            skip_ = other.skip_;
            atEnd_ = other.atEnd_;
            return *this;
        }

        Iterator(const CSRMatrix &mat, EigenIdxType idx, bool end = false) : skip_(idx), atEnd_(end) {
            if (!end) {
                it_ = Inner(mat, idx);
                atEnd_ = !it_;
                SkipDiagonal();
            }
        }

        reference operator*() const { return static_cast<std::size_t>(it_.col()); }

        Iterator &operator++() {
            ++it_;
            atEnd_ = !it_;
            SkipDiagonal();
            return *this;
        }

        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const Iterator &) const { return atEnd_; }

        bool operator!=(const Iterator &) const { return !atEnd_; }
    };

    EigenCSRRange(const Graph &graph, EigenIdxType idx) : graph_(graph), index_(idx) {}

    Iterator begin() const { return Iterator(*graph_.GetCSR(), index_); }

    Iterator end() const { return Iterator(*graph_.GetCSR(), index_, true); }
};

template <typename Graph, typename EigenIdxType>
class EigenCSCRange {
    const Graph &graph_;
    EigenIdxType index_;

  public:
    using CSCMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, EigenIdxType>;
    using Inner = typename CSCMatrix::InnerIterator;

    class Iterator {
        Inner it_;
        EigenIdxType skip_;
        bool atEnd_;

        void SkipDiagonal() {
            while ((!atEnd_) & (it_.row() == skip_) & (it_.col() == skip_)) {
                ++(*this);
            }
        }

      public:
        using value_type = std::size_t;
        using reference = value_type;
        using pointer = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        Iterator() = default;

        Iterator(const Iterator &other) : it_(other.it_), skip_(other.skip_), atEnd_(other.atEnd_) {}

        Iterator &operator=(const Iterator &other) {
            it_ = other.it_;
            skip_ = other.skip_;
            atEnd_ = other.atEnd_;
            return *this;
        }

        Iterator(const CSCMatrix &mat, EigenIdxType idx, bool end = false) : skip_(idx), atEnd_(end) {
            if (!end) {
                it_ = Inner(mat, idx);
                atEnd_ = !it_;
                SkipDiagonal();
            }
        }

        reference operator*() const { return static_cast<std::size_t>(it_.row()); }

        Iterator &operator++() {
            ++it_;
            atEnd_ = !it_;
            SkipDiagonal();
            return *this;
        }

        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const Iterator &) const { return atEnd_; }

        bool operator!=(const Iterator &) const { return !atEnd_; }
    };

    EigenCSCRange(const Graph &graph, EigenIdxType idx) : graph_(graph), index_(idx) {}

    Iterator begin() const { return Iterator(*graph_.GetCSC(), index_); }

    Iterator end() const { return Iterator(*graph_.GetCSC(), index_, true); }
};

}    // namespace osp

#endif
