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

#include <iterator>
#include <type_traits>

namespace osp {

/**
 * @brief A lightweight range class for iterating over a sequence of integral values.
 *
 * This class provides a view over a range of integers [start, finish), allowing iteration
 * without allocating memory for a container. It is useful for iterating over vertex indices
 * in a graph or any other sequence of numbers.
 *
 * @tparam T The integral type of the values (e.g., int, unsigned, size_t).
 */
template <typename T>
class integral_range {
    static_assert(std::is_integral<T>::value, "integral_range requires an integral type");

    T start;
    T finish;

  public:
    /**
     * @brief Iterator for the integral_range.
     *
     * This iterator satisfies the RandomAccessIterator concept.
     */
    class integral_iterator {    // public for std::reverse_iterator
      public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = void;    // Not a real pointer
        using reference = T;     // Not a real reference

        /**
         * @brief Proxy object to support operator-> for integral types.
         */
        struct arrow_proxy {
            T value;

            constexpr const T *operator->() const noexcept { return &value; }
        };

      private:
        value_type current;

      public:
        /**
         * @brief Default constructor. Initializes iterator to 0.
         */
        constexpr integral_iterator() noexcept : current(0) {}

        /**
         * @brief Constructs an iterator pointing to the given value.
         * @param start The starting value.
         */
        explicit constexpr integral_iterator(value_type start) noexcept : current(start) {}

        constexpr integral_iterator(const integral_iterator &) noexcept = default;
        constexpr integral_iterator &operator=(const integral_iterator &) noexcept = default;
        ~integral_iterator() = default;

        /**
         * @brief Dereference operator.
         * @return The current integral value.
         */
        [[nodiscard]] constexpr value_type operator*() const noexcept { return current; }

        /**
         * @brief Arrow operator.
         * @return A proxy object that allows access to the address of the value.
         */
        [[nodiscard]] constexpr arrow_proxy operator->() const noexcept { return arrow_proxy{current}; }

        constexpr integral_iterator &operator++() noexcept {
            ++current;
            return *this;
        }

        constexpr integral_iterator operator++(int) noexcept {
            integral_iterator temp = *this;
            ++(*this);
            return temp;
        }

        constexpr integral_iterator &operator--() noexcept {
            --current;
            return *this;
        }

        constexpr integral_iterator operator--(int) noexcept {
            integral_iterator temp = *this;
            --(*this);
            return temp;
        }

        [[nodiscard]] constexpr bool operator==(const integral_iterator &other) const noexcept {
            return current == other.current;
        }

        [[nodiscard]] constexpr bool operator!=(const integral_iterator &other) const noexcept { return !(*this == other); }

        constexpr integral_iterator &operator+=(difference_type n) noexcept {
            current = static_cast<value_type>(current + n);
            return *this;
        }

        [[nodiscard]] constexpr integral_iterator operator+(difference_type n) const noexcept {
            integral_iterator temp = *this;
            return temp += n;
        }

        [[nodiscard]] friend constexpr integral_iterator operator+(difference_type n, const integral_iterator &it) noexcept {
            return it + n;
        }

        constexpr integral_iterator &operator-=(difference_type n) noexcept {
            current = static_cast<value_type>(current - n);
            return *this;
        }

        [[nodiscard]] constexpr integral_iterator operator-(difference_type n) const noexcept {
            integral_iterator temp = *this;
            return temp -= n;
        }

        [[nodiscard]] constexpr difference_type operator-(const integral_iterator &other) const noexcept {
            return static_cast<difference_type>(current) - static_cast<difference_type>(other.current);
        }

        [[nodiscard]] constexpr value_type operator[](difference_type n) const noexcept { return *(*this + n); }

        [[nodiscard]] constexpr bool operator<(const integral_iterator &other) const noexcept { return current < other.current; }

        [[nodiscard]] constexpr bool operator>(const integral_iterator &other) const noexcept { return current > other.current; }

        [[nodiscard]] constexpr bool operator<=(const integral_iterator &other) const noexcept {
            return current <= other.current;
        }

        [[nodiscard]] constexpr bool operator>=(const integral_iterator &other) const noexcept {
            return current >= other.current;
        }
    };

    using reverse_integral_iterator = std::reverse_iterator<integral_iterator>;

  public:
    /**
     * @brief Constructs a range [0, end).
     * @param end_ The exclusive upper bound.
     */
    constexpr integral_range(T end_) noexcept : start(static_cast<T>(0)), finish(end_) {}

    /**
     * @brief Constructs a range [start, end).
     * @param start_ The inclusive lower bound.
     * @param end_ The exclusive upper bound.
     */
    constexpr integral_range(T start_, T end_) noexcept : start(start_), finish(end_) {}

    [[nodiscard]] constexpr integral_iterator begin() const noexcept { return integral_iterator(start); }

    [[nodiscard]] constexpr integral_iterator cbegin() const noexcept { return integral_iterator(start); }

    [[nodiscard]] constexpr integral_iterator end() const noexcept { return integral_iterator(finish); }

    [[nodiscard]] constexpr integral_iterator cend() const noexcept { return integral_iterator(finish); }

    [[nodiscard]] constexpr reverse_integral_iterator rbegin() const noexcept { return reverse_integral_iterator(end()); }

    [[nodiscard]] constexpr reverse_integral_iterator crbegin() const noexcept { return reverse_integral_iterator(cend()); }

    [[nodiscard]] constexpr reverse_integral_iterator rend() const noexcept { return reverse_integral_iterator(begin()); }

    [[nodiscard]] constexpr reverse_integral_iterator crend() const noexcept { return reverse_integral_iterator(cbegin()); }

    /**
     * @brief Returns the number of elements in the range.
     * @return The size of the range.
     */
    [[nodiscard]] constexpr auto size() const noexcept { return finish - start; }

    /**
     * @brief Checks if the range is empty.
     * @return True if the range is empty, false otherwise.
     */
    [[nodiscard]] constexpr bool empty() const noexcept { return start == finish; }
};

}    // namespace osp
