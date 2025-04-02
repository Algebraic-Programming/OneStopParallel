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
    T num_vertices;

    class vertex_iterator {
        T current;
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        explicit vertex_iterator(T start) : current(start) {}

        T operator*() const { return current; }

        vertex_iterator& operator++() {
            ++current;
            return *this;
        }

        vertex_iterator operator++(int) {
            vertex_iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const vertex_iterator& other) const {
            return current == other.current;
        }

        bool operator!=(const vertex_iterator& other) const {
            return !(*this == other);
        }
    };

  public:
    vertex_range(T num_vertices_) : num_vertices(num_vertices_) {}

    auto begin() const {
        return vertex_iterator(0);
    }

    auto end() const {
        return vertex_iterator(num_vertices);
    }
};

} // namespace osp