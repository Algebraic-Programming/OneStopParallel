
#pragma once

#include <iterator>
#include <type_traits>

namespace osp {

// Helper type traits
template<typename T, typename = void>
struct is_input_range : std::false_type {};

template<typename T>
struct is_input_range<
    T, std::void_t<decltype(std::begin(std::declval<T>())), 
                    decltype(std::end(std::declval<T>())),
                   typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::iterator_category>>
    : std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::iterator_category> {};

template<typename T>
inline constexpr bool is_input_range_v = is_input_range<T>::value;

// Concept for input range over a specific type, e.g., vertex_idx
template<typename T, typename ValueType, typename = void>
struct is_input_range_of : std::false_type {};

template<typename T, typename ValueType>
struct is_input_range_of<
    T, ValueType,
    std::void_t<decltype(std::begin(std::declval<T>())), 
                decltype(std::end(std::declval<T>())),
                typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::iterator_category,
                typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>>
    : std::conjunction<
          std::is_same<ValueType, typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>,
          std::is_base_of<std::input_iterator_tag,
                          typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::iterator_category>> {
};

template<typename T, typename ValueType>
inline constexpr bool is_input_range_of_v = is_input_range_of<T, ValueType>::value;

// Helper type traits for random access iterator
template<typename T, typename = void>
struct is_random_access_range : std::false_type {};

template<typename T>
struct is_random_access_range<
    T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>())),
                   typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::iterator_category>>
    : std::is_base_of<std::random_access_iterator_tag,
                      typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::iterator_category> {};

template<typename T>
inline constexpr bool is_random_access_range_v = is_random_access_range<T>::value;

template<typename T, typename ValueType, typename = void>
struct is_random_access_range_of : std::false_type {};

template<typename T, typename ValueType>
struct is_random_access_range_of<
    T, ValueType,
    std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>())),
                typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::iterator_category,
                typename std::enable_if_t<std::is_same_v<ValueType, typename std::iterator_traits<decltype(std::begin(
                                                                        std::declval<T>()))>::value_type>,
                                          void>>>
    : std::is_base_of<std::random_access_iterator_tag,
                      typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::iterator_category> {};

template<typename T, typename ValueType>
inline constexpr bool is_random_access_range_of_v = is_random_access_range_of<T, ValueType>::value;

} // namespace osp