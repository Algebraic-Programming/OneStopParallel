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
class IntegralRange {
    static_assert(std::is_integral<T>::value, "integral_range requires an integral type");

    T start_;
    T finish_;

  public:
    /**
     * @brief Iterator for the integral_range.
     *
     * This iterator satisfies the RandomAccessIterator concept.
     */
    class IntegralIterator {    // public for std::reverse_iterator
      public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = void;    // Not a real pointer
        using reference = T;     // Not a real reference

        /**
         * @brief Proxy object to support operator-> for integral types.
         */
        struct ArrowProxy {
            T value_;

            constexpr const T *operator->() const noexcept { return &value_; }
        };

      private:
        value_type current_;

      public:
        /**
         * @brief Default constructor. Initializes iterator to 0.
         */
        constexpr IntegralIterator() noexcept : current_(0) {}

        /**
         * @brief Constructs an iterator pointing to the given value.
         * @param start The starting value.
         */
        explicit constexpr IntegralIterator(value_type start) noexcept : current_(start) {}

        constexpr IntegralIterator(const IntegralIterator &) noexcept = default;
        constexpr IntegralIterator &operator=(const IntegralIterator &) noexcept = default;
        ~IntegralIterator() = default;

        /**
         * @brief Dereference operator.
         * @return The current integral value.
         */
        [[nodiscard]] constexpr value_type operator*() const noexcept { return current_; }

        /**
         * @brief Arrow operator.
         * @return A proxy object that allows access to the address of the value.
         */
        [[nodiscard]] constexpr ArrowProxy operator->() const noexcept { return ArrowProxy{current_}; }

        constexpr IntegralIterator &operator++() noexcept {
            ++current_;
            return *this;
        }

        constexpr IntegralIterator operator++(int) noexcept {
            IntegralIterator temp = *this;
            ++(*this);
            return temp;
        }

        constexpr IntegralIterator &operator--() noexcept {
            --current_;
            return *this;
        }

        constexpr IntegralIterator operator--(int) noexcept {
            IntegralIterator temp = *this;
            --(*this);
            return temp;
        }

        [[nodiscard]] constexpr bool operator==(const IntegralIterator &other) const noexcept {
            return current_ == other.current_;
        }

        [[nodiscard]] constexpr bool operator!=(const IntegralIterator &other) const noexcept { return !(*this == other); }

        constexpr IntegralIterator &operator+=(difference_type n) noexcept {
            current_ = static_cast<value_type>(current_ + n);
            return *this;
        }

        [[nodiscard]] constexpr IntegralIterator operator+(difference_type n) const noexcept {
            IntegralIterator temp = *this;
            return temp += n;
        }

        [[nodiscard]] friend constexpr IntegralIterator operator+(difference_type n, const IntegralIterator &it) noexcept {
            return it + n;
        }

        constexpr IntegralIterator &operator-=(difference_type n) noexcept {
            current_ = static_cast<value_type>(current_ - n);
            return *this;
        }

        [[nodiscard]] constexpr IntegralIterator operator-(difference_type n) const noexcept {
            IntegralIterator temp = *this;
            return temp -= n;
        }

        [[nodiscard]] constexpr difference_type operator-(const IntegralIterator &other) const noexcept {
            return static_cast<difference_type>(current_) - static_cast<difference_type>(other.current_);
        }

        [[nodiscard]] constexpr value_type operator[](difference_type n) const noexcept { return *(*this + n); }

        [[nodiscard]] constexpr bool operator<(const IntegralIterator &other) const noexcept { return current_ < other.current_; }

        [[nodiscard]] constexpr bool operator>(const IntegralIterator &other) const noexcept { return current_ > other.current_; }

        [[nodiscard]] constexpr bool operator<=(const IntegralIterator &other) const noexcept {
            return current_ <= other.current_;
        }

        [[nodiscard]] constexpr bool operator>=(const IntegralIterator &other) const noexcept {
            return current_ >= other.current_;
        }
    };

    using ReverseIntegralIterator = std::reverse_iterator<IntegralIterator>;

  public:
    /**
     * @brief Constructs a range [0, end).
     * @param end_ The exclusive upper bound.
     */
    constexpr IntegralRange(T end) noexcept : start_(static_cast<T>(0)), finish_(end) {}

    /**
     * @brief Constructs a range [start, end).
     * @param start_ The inclusive lower bound.
     * @param end_ The exclusive upper bound.
     */
    constexpr IntegralRange(T start, T end) noexcept : start_(start), finish_(end) {}

    [[nodiscard]] constexpr IntegralIterator begin() const noexcept { return IntegralIterator(start_); }

    [[nodiscard]] constexpr IntegralIterator cbegin() const noexcept { return IntegralIterator(start_); }

    [[nodiscard]] constexpr IntegralIterator end() const noexcept { return IntegralIterator(finish_); }

    [[nodiscard]] constexpr IntegralIterator cend() const noexcept { return IntegralIterator(finish_); }

    [[nodiscard]] constexpr ReverseIntegralIterator rbegin() const noexcept { return ReverseIntegralIterator(end()); }

    [[nodiscard]] constexpr ReverseIntegralIterator crbegin() const noexcept { return ReverseIntegralIterator(cend()); }

    [[nodiscard]] constexpr ReverseIntegralIterator rend() const noexcept { return ReverseIntegralIterator(begin()); }

    [[nodiscard]] constexpr ReverseIntegralIterator crend() const noexcept { return ReverseIntegralIterator(cbegin()); }

    /**
     * @brief Returns the number of elements in the range.
     * @return The size of the range.
     */
    [[nodiscard]] constexpr auto size() const noexcept { return finish_ - start_; }

    /**
     * @brief Checks if the range is empty.
     * @return True if the range is empty, false otherwise.
     */
    [[nodiscard]] constexpr bool empty() const noexcept { return start_ == finish_; }
};

}    // namespace osp
