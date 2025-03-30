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
#include <memory>
#include <type_traits>
#include <utility>

namespace osp {

/// @brief hides the unique_ptr storage and returns the reference to the stored
/// object.
/// @tparam T the stored type
template<typename T>
class RemoveUniquePtr {
  public:
    T &operator()(std::unique_ptr<T> &p) { return *(p.get()); }
    const T &operator()(std::unique_ptr<T> const &p) const { return *(p.get()); }
};

template<typename T>
class RemovePtr {
  public:
    T &operator()(T *p) { return *p; }
    const T &operator()(T const *p) const { return *p; }
};

template<typename TransformerT, typename DefValueT>
struct value_type_extractor {

    template<typename TransT>
    static std::remove_reference_t<decltype(std::declval<typename TransT::value_type>())> f(int) {
        return typename TransT::value_type{};
    }

    template<typename TransT>
    static DefValueT f(...) {
        return DefValueT{};
    }
    using type = decltype(f<TransformerT>(0));
    static constexpr bool value = not std::is_same_v<type, DefValueT>;
};

template<typename TransformerT, typename DefValueT>
using value_type_extractor_t = typename value_type_extractor<TransformerT, DefValueT>::type;

/// @brief adapter class for iterator, which adapts the value via the `*`
/// operator.
/// @tparam IterT iterator type
/// @tparam TypeTransformerT transformer type, which must be callable
template<typename IterT, typename TypeTransformerT>
class IterAdaptor {
  public:
    static_assert(std::is_copy_constructible_v<TypeTransformerT>);

    using iterator_category = typename std::iterator_traits<IterT>::iterator_category;
    using reference = decltype(std::declval<TypeTransformerT>()(*std::declval<IterT>()));
    using pointer = decltype(&std::declval<reference>());
    using const_reference = decltype(std::declval<TypeTransformerT>()(*std::declval<const IterT>()));
    using value_type = value_type_extractor_t<TypeTransformerT, std::remove_cv_t<std::remove_reference_t<reference>>>;

    using difference_type = typename std::iterator_traits<IterT>::difference_type;
    using ThisT = IterAdaptor<IterT, TypeTransformerT>;

    static constexpr bool _is_random = std::is_convertible_v<iterator_category, std::random_access_iterator_tag>;

    IterAdaptor() = delete;
    explicit IterAdaptor(IterT const &it) noexcept : iter(it) {}
    explicit IterAdaptor(IterT &&it) noexcept : iter(std::move(it)) {}
    IterAdaptor(ThisT const &) = default;
    IterAdaptor(ThisT &&) = default;

    ThisT &operator=(ThisT const &) = default;
    ThisT &operator=(ThisT &&) = default;

    ThisT &operator++() {
        iter++;
        return *this;
    }
    ThisT operator++(int) {
        ThisT ret(iter);
        ++iter;
        return ret;
    }
    bool operator==(ThisT const &other) const { return this->iter == other.iter; }
    bool operator!=(ThisT const &other) const { return !(*this == other); }

    reference operator*() { return transformer(*iter); }
    const reference operator*() const { return transformer(*iter); }

    ThisT &operator+=(const std::enable_if_t<_is_random, std::size_t> count) {
        iter += count;
        return *this;
    }

    inline difference_type operator-(std::enable_if_t<_is_random, ThisT const &> other) const {
        return this->iter - other.iter;
    }

  private:
    TypeTransformerT transformer{};
    IterT iter;
};

/// @brief adapts a container via its iterators, converting the contained value
/// to something else (e.g., to remove memory management facilities like
/// std::unique_ptr).
///
/// An example is converting a `std::vector<unique_ptr<AscendEdge>>` to iterate
/// simply over `AscendEdge`.
///
/// @tparam ConverterT type for container's value_type conversion
/// @tparam ContainerT container type
template<typename ConverterT, typename ContainerT>
class ContainerAdaptor {
  public:
    using iterator = IterAdaptor<typename ContainerT::iterator, ConverterT>;
    using const_iterator = IterAdaptor<typename ContainerT::const_iterator, ConverterT>;
    using value_type = typename iterator::value_type;
    using size_type = typename ContainerT::size_type;
    using difference_type = typename ContainerT::difference_type;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using ThisT = ContainerAdaptor<ConverterT, ContainerT>;

    ContainerAdaptor(ContainerT &_container) : container(_container) {}

    ContainerAdaptor(ThisT const &original) : container(original.container) {}

    inline std::size_t size() const { return container.size(); }

    inline iterator begin() noexcept { return iterator(container.begin()); }
    inline iterator end() noexcept { return iterator(container.end()); }

    inline const_iterator begin() const noexcept { return const_iterator(container.cbegin()); }
    inline const_iterator end() const noexcept { return const_iterator(container.cend()); }

    inline const_iterator cbegin() const noexcept { return const_pointer(container.cbegin()); }
    inline const_iterator cend() const noexcept { return const_pointer(container.cend()); }

  private:
    ContainerT &container;
};

} // namespace osp