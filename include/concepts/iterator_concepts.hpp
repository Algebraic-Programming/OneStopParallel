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

// Concept for const_iterator with forward iterator category
template<typename T, typename = void>
struct is_forward_iterator : std::false_type {};

template<typename T>
struct is_forward_iterator<
    T, std::void_t<typename std::iterator_traits<T>::difference_type,
                   typename std::iterator_traits<T>::value_type,
                   typename std::iterator_traits<T>::pointer,
                   typename std::iterator_traits<T>::reference,
                   typename std::iterator_traits<T>::iterator_category>>
    : std::conjunction<
          std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<T>::iterator_category>> {};

template<typename T>
inline constexpr bool is_forward_iterator_v = is_forward_iterator<T>::value;

// Concept for ranges with const forward iterators supporting begin or cbegin
template<typename T, typename ValueType, typename = void>
struct is_forward_range_of : std::false_type {};

template<typename T, typename ValueType>
struct is_forward_range_of<
    T, ValueType,
    std::void_t<decltype(std::begin(std::declval<T>())), 
                decltype(std::end(std::declval<T>()))>>
    : std::conjunction<
          is_forward_iterator<decltype(std::begin(std::declval<T>()))>,
          std::is_same<ValueType, typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>> {};

template<typename T, typename ValueType>
inline constexpr bool is_forward_range_of_v = is_forward_range_of<T, ValueType>::value;

// Concept for containers
template<typename T, typename ValueType, typename = void>
struct is_container_of : std::false_type {};

template<typename T, typename ValueType>
struct is_container_of<
    T, ValueType,
    std::void_t<decltype(std::size(std::declval<T>()))>>
    : std::conjunction<
          is_forward_range_of<T, ValueType>> {};

template<typename T, typename ValueType>
inline constexpr bool is_container_of_v = is_container_of<T, ValueType>::value;



// Concept for const_iterator with forward iterator category
template<typename T, typename = void>
struct is_input_iterator : std::false_type {};

template<typename T>
struct is_input_iterator<
    T, std::void_t<typename std::iterator_traits<T>::difference_type,
                   typename std::iterator_traits<T>::value_type,
                   typename std::iterator_traits<T>::pointer,
                   typename std::iterator_traits<T>::reference,
                   typename std::iterator_traits<T>::iterator_category>>
    : std::conjunction<
          std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<T>::iterator_category>> {};

template<typename T>
inline constexpr bool is_input_iterator_v = is_input_iterator<T>::value;

// Concept for ranges with const input iterators supporting begin or cbegin
template<typename T, typename ValueType, typename = void>
struct is_input_range_of : std::false_type {};

template<typename T, typename ValueType>
struct is_input_range_of<
    T, ValueType,
    std::void_t<decltype(std::begin(std::declval<T>())), 
                decltype(std::end(std::declval<T>()))>>
    : std::conjunction<
          is_input_iterator<decltype(std::begin(std::declval<T>()))>,
          std::is_same<ValueType, typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>> {};

template<typename T, typename ValueType>
inline constexpr bool is_input_range_of_v = is_input_range_of<T, ValueType>::value;


} // namespace osp