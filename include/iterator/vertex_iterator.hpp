#pragma once

#include <iterator>

namespace osp {

template<typename T = unsigned>
class vertex_range {
    T num_vertices;

    class vertex_iterator {
        T current;
      public:
        using iterator_category = std::input_iterator_tag;
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