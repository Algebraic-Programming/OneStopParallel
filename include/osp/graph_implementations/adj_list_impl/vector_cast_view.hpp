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
template <typename from_t, typename to_t>
class vector_cast_view {
    using iter = typename std::vector<from_t>::const_iterator;
    const std::vector<from_t> &vec;

    /**
     * @brief Iterator for vector_cast_view.
     *
     * This iterator wraps the underlying vector iterator and performs a static_cast
     * on dereference. It satisfies the RandomAccessIterator concept.
     */
    struct cast_iterator {
        using iterator_category = std::random_access_iterator_tag;
        using value_type = to_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        iter current_edge;

        cast_iterator() = default;

        explicit cast_iterator(iter current_edge_) : current_edge(current_edge_) {}

        value_type operator*() const { return static_cast<to_t>(*current_edge); }

        cast_iterator &operator++() {
            ++current_edge;
            return *this;
        }

        cast_iterator operator++(int) {
            cast_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        cast_iterator &operator--() {
            --current_edge;
            return *this;
        }

        cast_iterator operator--(int) {
            cast_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        cast_iterator &operator+=(difference_type n) {
            current_edge += n;
            return *this;
        }

        cast_iterator &operator-=(difference_type n) {
            current_edge -= n;
            return *this;
        }

        cast_iterator operator+(difference_type n) const { return cast_iterator(current_edge + n); }

        cast_iterator operator-(difference_type n) const { return cast_iterator(current_edge - n); }

        difference_type operator-(const cast_iterator &other) const { return current_edge - other.current_edge; }

        bool operator==(const cast_iterator &other) const { return current_edge == other.current_edge; }

        bool operator!=(const cast_iterator &other) const { return current_edge != other.current_edge; }

        bool operator<(const cast_iterator &other) const { return current_edge < other.current_edge; }

        bool operator>(const cast_iterator &other) const { return current_edge > other.current_edge; }

        bool operator<=(const cast_iterator &other) const { return current_edge <= other.current_edge; }

        bool operator>=(const cast_iterator &other) const { return current_edge >= other.current_edge; }
    };

  public:
    /**
     * @brief Constructs a vector_cast_view from a vector.
     *
     * @param vec_ The vector to view. The view holds a reference to this vector,
     *             so the vector must outlive the view.
     */
    explicit vector_cast_view(const std::vector<from_t> &vec_) : vec(vec_) {}

    /**
     * @brief Returns an iterator to the beginning of the view.
     * @return An iterator to the first element.
     */
    [[nodiscard]] auto begin() const { return cast_iterator(vec.begin()); }

    /**
     * @brief Returns an iterator to the end of the view.
     * @return An iterator to the element following the last element.
     */
    [[nodiscard]] auto end() const { return cast_iterator(vec.end()); }

    /**
     * @brief Returns the number of elements in the view.
     * @return The number of elements.
     */
    [[nodiscard]] auto size() const { return vec.size(); }

    /**
     * @brief Checks if the view is empty.
     * @return True if the view is empty, false otherwise.
     */
    [[nodiscard]] bool empty() const { return vec.empty(); }

    /**
     * @brief Accesses the element at the specified index.
     * @param i The index of the element to access.
     * @return The element at index i, cast to to_t.
     */
    [[nodiscard]] auto operator[](std::size_t i) const { return static_cast<to_t>(vec[i]); }
};

}    // namespace osp
