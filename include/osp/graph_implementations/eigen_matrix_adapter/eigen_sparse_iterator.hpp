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

        void SkipDiagonal() {
            while (it_ && (it_.row() == it_.col())) {
                ++it_;
            }
        }

      public:
        using value_type = std::size_t;
        using reference = value_type;
        using pointer = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        Iterator() = default;

        Iterator(const Iterator &other) : it_(other.it_) {}

        Iterator &operator=(const Iterator &other) {
            it_ = other.it_;
            return *this;
        }

        Iterator(const CSRMatrix &mat, EigenIdxType idx) : it_(mat, idx) { SkipDiagonal(); }

        reference operator*() const { return static_cast<std::size_t>(it_.col()); }

        Iterator &operator++() {
            ++it_;
            SkipDiagonal();
            return *this;
        }

        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const Iterator &other) const {return it_ == other.it_;}
        bool operator!=(const Iterator &other) const { return not (*this == other);}
    };

    EigenCSRRange(const Graph &graph, EigenIdxType idx) : graph_(graph), index_(idx) {}

    Iterator begin() const { return Iterator(*graph_.GetCSR(), index_); }

    Iterator end() const { return Iterator(); }
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

        void SkipDiagonal() {
            while (it_ && (it_.row() == it_.col())) {
                ++it_;
            }
        }

      public:
        using value_type = std::size_t;
        using reference = value_type;
        using pointer = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        Iterator() = default;

        Iterator(const Iterator &other) : it_(other.it_) {}

        Iterator &operator=(const Iterator &other) {
            it_ = other.it_;
            return *this;
        }

        Iterator(const CSCMatrix &mat, EigenIdxType idx) : it_(mat, idx) { SkipDiagonal(); }

        reference operator*() const { return static_cast<std::size_t>(it_.row()); }

        Iterator &operator++() {
            ++it_;
            return *this;
        }

        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const Iterator &other) const {return it_ == other.it_;}
        bool operator!=(const Iterator &other) const { return not (*this == other);}
    };

    EigenCSCRange(const Graph &graph, EigenIdxType idx) : graph_(graph), index_(idx) {}

    Iterator begin() const { return Iterator(*graph_.GetCSC(), index_); }

    Iterator end() const { return Iterator(); }
};

}    // namespace osp

#endif
