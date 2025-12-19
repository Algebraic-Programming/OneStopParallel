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

namespace osp {

/**
 * @brief A view that casts elements of a vector to a different type upon access.
 *
 * This class provides a lightweight view over a std::vector<from_t>, exposing its elements
 * as type to_t. It is useful for adapting interfaces that expect a range of a specific type
 * without copying the underlying data.
 *
 * @tparam from_t The original type of elements in the vector.
 * @tparam to_t The target type to cast elements to.
 */
template <typename FromT, typename ToT>
class VectorCastView {
    using Iter = typename std::vector<FromT>::const_iterator;
    const std::vector<FromT> &vec_;

    /**
     * @brief Iterator for vector_cast_view.
     *
     * This iterator wraps the underlying vector iterator and performs a static_cast
     * on dereference. It satisfies the RandomAccessIterator concept.
     */
    struct CastIterator {
        using iterator_category = std::random_access_iterator_tag;
        using value_type = ToT;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        Iter currentEdge_;

        CastIterator() = default;

        explicit CastIterator(Iter currentEdge) : currentEdge_(currentEdge) {}

        value_type operator*() const { return static_cast<ToT>(*currentEdge_); }

        CastIterator &operator++() {
            ++currentEdge_;
            return *this;
        }

        CastIterator operator++(int) {
            CastIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        CastIterator &operator--() {
            --currentEdge_;
            return *this;
        }

        CastIterator operator--(int) {
            CastIterator tmp = *this;
            --(*this);
            return tmp;
        }

        CastIterator &operator+=(difference_type n) {
            currentEdge_ += n;
            return *this;
        }

        CastIterator &operator-=(difference_type n) {
            currentEdge_ -= n;
            return *this;
        }

        CastIterator operator+(difference_type n) const { return CastIterator(currentEdge_ + n); }

        CastIterator operator-(difference_type n) const { return CastIterator(currentEdge_ - n); }

        difference_type operator-(const CastIterator &other) const { return currentEdge_ - other.currentEdge_; }

        bool operator==(const CastIterator &other) const { return currentEdge_ == other.currentEdge_; }

        bool operator!=(const CastIterator &other) const { return currentEdge_ != other.currentEdge_; }

        bool operator<(const CastIterator &other) const { return currentEdge_ < other.currentEdge_; }

        bool operator>(const CastIterator &other) const { return currentEdge_ > other.currentEdge_; }

        bool operator<=(const CastIterator &other) const { return currentEdge_ <= other.currentEdge_; }

        bool operator>=(const CastIterator &other) const { return currentEdge_ >= other.currentEdge_; }
    };

  public:
    /**
     * @brief Constructs a vector_cast_view from a vector.
     *
     * @param vec_ The vector to view. The view holds a reference to this vector,
     *             so the vector must outlive the view.
     */
    explicit VectorCastView(const std::vector<FromT> &vec) : vec_(vec) {}

    /**
     * @brief Returns an iterator to the beginning of the view.
     * @return An iterator to the first element.
     */
    [[nodiscard]] auto begin() const { return CastIterator(vec_.begin()); }

    /**
     * @brief Returns an iterator to the end of the view.
     * @return An iterator to the element following the last element.
     */
    [[nodiscard]] auto end() const { return CastIterator(vec_.end()); }

    /**
     * @brief Returns the number of elements in the view.
     * @return The number of elements.
     */
    [[nodiscard]] auto size() const { return vec_.size(); }

    /**
     * @brief Checks if the view is empty.
     * @return True if the view is empty, false otherwise.
     */
    [[nodiscard]] bool empty() const { return vec_.empty(); }

    /**
     * @brief Accesses the element at the specified index.
     * @param i The index of the element to access.
     * @return The element at index i, cast to to_t.
     */
    [[nodiscard]] auto operator[](std::size_t i) const { return static_cast<ToT>(vec_[i]); }
};

}    // namespace osp
