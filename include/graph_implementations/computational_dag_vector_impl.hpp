#pragma once

#include "concepts/computational_dag_concept.hpp"
#include "iterator/vertex_iterator.hpp"
#include <vector>

namespace osp {

struct cdag_vertex_impl {

    cdag_vertex_impl(vertex_idx vertex_idx, int work_weight, int comm_weight, int mem_weight, unsigned vertex_type)
        : id(vertex_idx), work_weight(work_weight), comm_weight(comm_weight), mem_weight(mem_weight),
          vertex_type(vertex_type) {}

    vertex_idx id;

    int work_weight;
    int comm_weight;
    int mem_weight;

    unsigned vertex_type;
};

template<typename v_impl = cdag_vertex_impl>
class computational_dag_vector_impl {

  public:
    static_assert(std::is_base_of<cdag_vertex_impl, v_impl>::value, "v_impl must be derived from cdag_vertex_impl");

    computational_dag_vector_impl() = default;

    ~computational_dag_vector_impl() = default;

    inline auto vertices() const { return vertex_range<vertex_idx>(vertices_.size()); }

    inline size_t num_vertices() const { return vertices_.size(); }

    inline size_t num_edges() const { return num_edges_; }

    inline const std::vector<vertex_idx> &parents(const vertex_idx v) const { return in_neigbors[v]; }

    inline const std::vector<vertex_idx> &children(const vertex_idx v) const { return out_neigbors[v]; }

    inline size_t in_degree(const vertex_idx v) const { return in_neigbors[v].size(); }

    inline size_t out_degree(const vertex_idx v) const { return out_neigbors[v].size(); }

    inline int vertex_work_weight(const vertex_idx v) const { return vertices_[v].work_weight; }

    inline int vertex_comm_weight(const vertex_idx v) const { return vertices_[v].comm_weight; }

    inline int vertex_mem_weight(const vertex_idx v) const { return vertices_[v].mem_weight; }

    inline unsigned vertex_type(const vertex_idx v) const { return vertices_[v].vertex_type; }

    inline unsigned num_vertex_types() const { return num_vertex_types_; }

    inline const v_impl &get_vertex_impl(const vertex_idx v) const { return vertices_[v]; }

    vertex_idx add_vertex(int work_weight, int comm_weight, int mem_weight, unsigned vertex_type) {

        vertices_.emplace_back(vertices_.size(), work_weight, comm_weight, mem_weight, vertex_type);
        out_neigbors.push_back({});
        in_neigbors.push_back({});

        num_vertex_types_ = std::max(num_vertex_types_, vertex_type);

        return vertices_.back().id;
    }

    bool add_edge(vertex_idx source, vertex_idx target) {

        if (source >= vertices_.size() || target >= vertices_.size())
            return false;

        for (const vertex_idx v_idx : out_neigbors[source]) {
            if (v_idx == target) {
                return false;
            }
        }

        out_neigbors[source].push_back(target);
        in_neigbors[target].push_back(source);
        num_edges_++;

        return true;
    }

  private:
    std::vector<v_impl> vertices_;

    std::vector<std::vector<vertex_idx>> out_neigbors;
    std::vector<std::vector<vertex_idx>> in_neigbors;

    size_t num_edges_ = 0;
    unsigned num_vertex_types_ = 0;
};

static_assert(has_vertex_weights_v<computational_dag_vector_impl<cdag_vertex_impl>>,
              "computational_dag_vector_impl must satisfy the has_vertex_weights concept");

static_assert(is_directed_graph_v<computational_dag_vector_impl<cdag_vertex_impl>>,
              "computational_dag_vector_impl must satisfy the directed_graph concept");

static_assert(is_computation_dag_typed_vertices_v<computational_dag_vector_impl<cdag_vertex_impl>>,
              "computational_dag_vector_impl must satisfy the is_computation_dag concept");

} // namespace osp