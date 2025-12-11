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
 * @file iterator_concepts.hpp
 * @brief C++17 compatible concept checks (type traits) for iterators and ranges.
 *
 * This file provides type traits that emulate C++20 concepts for iterators and ranges.
 * These are used to ensure type safety and correct usage of templates within the library
 * while maintaining compatibility with C++17.
 */

/**
 * @brief Checks if a type is a forward iterator.
 *
 * This type trait checks if `T` satisfies the requirements of a forward iterator.
 * It verifies the existence of standard iterator typedefs and checks if the iterator category
 * is derived from `std::forward_iterator_tag`.
 *
 * @note Equivalent to C++20 `std::forward_iterator`.
 *
 * @tparam T The type to check.
 */
template <typename T, typename = void>
struct IsForwardIterator : std::false_type {};

template <typename T>
struct IsForwardIterator<T,
                         std::void_t<typename std::iterator_traits<T>::difference_type,
                                     typename std::iterator_traits<T>::value_type,
                                     typename std::iterator_traits<T>::pointer,
                                     typename std::iterator_traits<T>::reference,
                                     typename std::iterator_traits<T>::iterator_category>>
    : std::conjunction<std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<T>::iterator_category>> {};

template <typename T>
inline constexpr bool isForwardIteratorV = IsForwardIterator<T>::value;

/**
 * @brief Checks if a type is a range of forward iterators with a specific value type.
 *
 * This type trait checks if `T` is a range (provides `begin()` and `end()`) whose iterator
 * satisfies `is_forward_iterator` and whose value type matches `ValueType`.
 *
 * @note Equivalent to C++20 `std::ranges::forward_range` combined with a value type check.
 *
 * @tparam T The range type to check.
 * @tparam ValueType The expected value type of the range.
 */
template <typename T, typename ValueType, typename = void>
struct IsForwardRangeOf : std::false_type {};

template <typename T, typename ValueType>
struct IsForwardRangeOf<T, ValueType, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::conjunction<IsForwardIterator<decltype(std::begin(std::declval<T>()))>,
                       std::is_same<ValueType, typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>> {
};

template <typename T, typename ValueType>
inline constexpr bool isForwardRangeOfV = IsForwardRangeOf<T, ValueType>::value;

/**
 * @brief Checks if a type is a container (sized forward range).
 *
 * This type trait checks if `T` satisfies `is_forward_range_of` and additionally provides
 * a `size()` member function.
 *
 * @note Equivalent to C++20 `std::ranges::sized_range` combined with `std::ranges::forward_range`.
 *
 * @tparam T The container type to check.
 * @tparam ValueType The expected value type of the container.
 */
template <typename T, typename ValueType, typename = void>
struct IsContainerOf : std::false_type {};

template <typename T, typename ValueType>
struct IsContainerOf<T, ValueType, std::void_t<decltype(std::size(std::declval<T>()))>>
    : std::conjunction<IsForwardRangeOf<T, ValueType>> {};

template <typename T, typename ValueType>
inline constexpr bool isContainerOfV = IsContainerOf<T, ValueType>::value;

/**
 * @brief Checks if a type is an input iterator.
 *
 * This type trait checks if `T` satisfies the requirements of an input iterator.
 * It verifies the existence of standard iterator typedefs and checks if the iterator category
 * is derived from `std::input_iterator_tag`.
 *
 * @note Equivalent to C++20 `std::input_iterator`.
 *
 * @tparam T The type to check.
 */
template <typename T, typename = void>
struct IsInputIterator : std::false_type {};

template <typename T>
struct IsInputIterator<T,
                       std::void_t<typename std::iterator_traits<T>::difference_type,
                                   typename std::iterator_traits<T>::value_type,
                                   typename std::iterator_traits<T>::pointer,
                                   typename std::iterator_traits<T>::reference,
                                   typename std::iterator_traits<T>::iterator_category>>
    : std::conjunction<std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<T>::iterator_category>> {};

template <typename T>
inline constexpr bool isInputIteratorV = IsInputIterator<T>::value;

/**
 * @brief Checks if a type is a range of input iterators with a specific value type.
 *
 * This type trait checks if `T` is a range (provides `begin()` and `end()`) whose iterator
 * satisfies `is_input_iterator` and whose value type matches `ValueType`.
 *
 * @note Equivalent to C++20 `std::ranges::input_range` combined with a value type check.
 *
 * @tparam T The range type to check.
 * @tparam ValueType The expected value type of the range.
 */
template <typename T, typename ValueType, typename = void>
struct IsInputRangeOf : std::false_type {};

template <typename T, typename ValueType>
struct IsInputRangeOf<T, ValueType, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::conjunction<IsInputIterator<decltype(std::begin(std::declval<T>()))>,
                       std::is_same<ValueType, typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>> {
};

template <typename T, typename ValueType>
inline constexpr bool isInputRangeOfV = IsInputRangeOf<T, ValueType>::value;

}    // namespace osp
