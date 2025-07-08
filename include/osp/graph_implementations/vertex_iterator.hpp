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

    class vertex_iterator {
      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using reference = T &;

      private:
        value_type current;

      public:
        vertex_iterator() : current(0) {}
        explicit vertex_iterator(value_type start) : current(start) {}
        vertex_iterator(const vertex_iterator &other) : current(other.current) {}
        vertex_iterator &operator=(const vertex_iterator &other) {
            if (this != &other) {
                current = other.current;
            }
            return *this;
        }

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

        inline bool operator<=(const vertex_iterator &other) const { return current <= other.current; }
        inline bool operator<(const vertex_iterator &other) const { return (*this <= other) && (*this != other); }
        inline bool operator>=(const vertex_iterator &other) const { return (!(*this <= other)) || (*this == other); }
        inline bool operator>(const vertex_iterator &other) const { return !(*this <= other); }
    };


    class reverse_vertex_iterator {
      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using reference = T &;

      private:
        value_type current;

      public:
        reverse_vertex_iterator() : current(0) {}
        explicit reverse_vertex_iterator(value_type start) : current(start) {}
        reverse_vertex_iterator(const reverse_vertex_iterator &other) : current(other.current) {}
        reverse_vertex_iterator &operator=(const reverse_vertex_iterator &other) {
            if (this != &other) {
                current = other.current;
            }
            return *this;
        }

        inline value_type operator*() const { return current; }

        inline reverse_vertex_iterator &operator++() {
            --current;
            return *this;
        }

        inline reverse_vertex_iterator operator++(int) {
            reverse_vertex_iterator temp = *this;
            ++(*this);
            return temp;
        }

        inline reverse_vertex_iterator &operator--() {
            ++current;
            return *this;
        }

        inline reverse_vertex_iterator operator--(int) {
            reverse_vertex_iterator temp = *this;
            --(*this);
            return temp;
        }

        inline bool operator==(const reverse_vertex_iterator &other) const { return current == other.current; }
        inline bool operator!=(const reverse_vertex_iterator &other) const { return !(*this == other); }

        inline bool operator<=(const reverse_vertex_iterator &other) const { return current <= other.current; }
        inline bool operator<(const reverse_vertex_iterator &other) const { return (*this <= other) && (*this != other); }
        inline bool operator>=(const reverse_vertex_iterator &other) const { return (!(*this <= other)) || (*this == other); }
        inline bool operator>(const reverse_vertex_iterator &other) const { return !(*this <= other); }
    };

  public:
    vertex_range(T end_) : start(static_cast<T>(0)), finish(end_) {}
    vertex_range(T start_, T end_) : start(start_), finish(end_) {}

    inline auto begin() const { return vertex_iterator(start); }
    inline auto cbegin() const { return vertex_iterator(start); }

    inline auto end() const { return vertex_iterator(finish); }
    inline auto cend() const { return vertex_iterator(finish); }

    inline auto rbegin() const { return reverse_vertex_iterator(finish - 1); }
    inline auto crbegin() const { return reverse_vertex_iterator(finish - 1); }
    
    inline auto rend() const { return reverse_vertex_iterator(start - 1); }
    inline auto crend() const { return reverse_vertex_iterator(start - 1); }

    inline auto size() const { return finish - start; }
};

} // namespace osp