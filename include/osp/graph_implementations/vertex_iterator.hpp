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

namespace osp {

template<typename T>
class vertex_range {
    static_assert(std::is_integral<T>::value);

    T start;
    T finish;
public:
    class vertex_iterator { // public for std::reverse_iterator
      public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        // pointer and reference are not real pointers/references to the value
        using pointer = const T *;
        using reference = const T;

      private:
        value_type current;

      public:
        vertex_iterator() : current(0) {}
        explicit vertex_iterator(value_type start) : current(start) {}
        vertex_iterator(const vertex_iterator &) = default;
        vertex_iterator &operator=(const vertex_iterator &) = default;
        ~vertex_iterator() = default;

        inline value_type operator*() const { return current; }

        inline vertex_iterator &operator++() {
            ++current;
            return *this;
        }

        inline vertex_iterator operator++(int) {
            vertex_iterator temp = *this;
            ++(*this);
            return temp;
        }

        inline vertex_iterator &operator--() {
            --current;
            return *this;
        }

        inline vertex_iterator operator--(int) {
            vertex_iterator temp = *this;
            --(*this);
            return temp;
        }

        inline bool operator==(const vertex_iterator &other) const { return current == other.current; }
        inline bool operator!=(const vertex_iterator &other) const { return !(*this == other); }

        inline vertex_iterator &operator+=(difference_type n) { current += n; return *this; }
        inline vertex_iterator operator+(difference_type n) const { vertex_iterator temp = *this; return temp += n; }
        friend inline vertex_iterator operator+(difference_type n, const vertex_iterator& it) { return it + n; }

        inline vertex_iterator &operator-=(difference_type n) { current -= n; return *this; }
        inline vertex_iterator operator-(difference_type n) const { vertex_iterator temp = *this; return temp -= n; }
        inline difference_type operator-(const vertex_iterator& other) const { return current - other.current; }

        inline value_type operator[](difference_type n) const { return *(*this + n); }

        inline bool operator<(const vertex_iterator &other) const { return current < other.current; }
        inline bool operator>(const vertex_iterator &other) const { return current > other.current; }
        inline bool operator<=(const vertex_iterator &other) const { return current <= other.current; }
        inline bool operator>=(const vertex_iterator &other) const { return current >= other.current; }
    };

    using reverse_vertex_iterator = std::reverse_iterator<vertex_iterator>;

  public:
    vertex_range(T end_) : start(static_cast<T>(0)), finish(end_) {}
    vertex_range(T start_, T end_) : start(start_), finish(end_) {}

    inline vertex_iterator begin() const { return vertex_iterator(start); }
    inline vertex_iterator cbegin() const { return vertex_iterator(start); }

    inline vertex_iterator end() const { return vertex_iterator(finish); }
    inline vertex_iterator cend() const { return vertex_iterator(finish); }

    inline reverse_vertex_iterator rbegin() const { return reverse_vertex_iterator(end()); }
    inline reverse_vertex_iterator crbegin() const { return reverse_vertex_iterator(cend()); }
    
    inline reverse_vertex_iterator rend() const { return reverse_vertex_iterator(begin()); }
    inline reverse_vertex_iterator crend() const { return reverse_vertex_iterator(cbegin()); }

    inline auto size() const { return finish - start; }
};

} // namespace osp