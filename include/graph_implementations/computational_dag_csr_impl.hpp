// #pragma once

// #include "concepts/computational_dag_concept.hpp"
// #include "computational_dag_vector_impl.hpp"
// #include "container_iterator_adaptor.hpp"
// #include <vector>

// template<typename v_impl = cdag_vertex_impl>
// class computational_dag_csr_impl {
//   public:
//     static_assert(std::is_base_of<cdag_vertex_impl, v_impl>::value, "v_impl must be derived from cdag_vertex_impl");

//   private:
//     struct parents_iterator {

//         using iterator_category = std::input_iterator_tag;
//         using value_type = vertex_idx;

//         std::vector<vertex_idx>::iterator current_parent;
//         vertex_idx current_vertex;

//         computational_dag_csr_impl &graph_;

//       public:

//         computational_dag_csr_impl(computational_dag_csr_impl &graph, vertex_idx v_idx)
//             : current_parent(graph_.out_neighbors.begin() + out_neigbors_pos[v_idx]), current_vertex(v_idx),
//             graph_(graph) {}

//         const value_type &operator*() const { return *current_parent; }
//         value_type *operator->() { return current_parent.operator->(); }

//         // Prefix increment
//         parents_iterator &operator++() {
//             current_parent++;

//             return *this;
//         }

//         // Postfix increment
//         parents_iterator operator++(int) {
//             parents_iterator tmp = *this;
//             ++(*this);
//             return tmp;
//         }

//         friend bool operator==(const parents_iterator &one, const parents_iterator &other) {
//             return one.current_vertex == other.current_vertex && one.current_parent.operator*() ==
//             other.current_parent.operator*();
//         };
//         friend bool operator!=(const parents_iterator &one, const parents_iterator &other) {
//             return one.current_vertex != other.current_vertex || one.current_parent.operator*() !=
//             other.current_parent.operator*();
//         };
//     };

//     using vertex_adapter_idx_t = ContainerAdaptor<cdag_vertex_idx_view, std::vector<v_impl>>;

//     std::vector<v_impl> vertices_;

//     std::vector<vertex_idx> out_neigbors;
//     std::vector<size_t> out_neigbors_pos;

//     std::vector<vertex_idx> in_neigbors;
//     std::vector<size_t> in_neigbors_pos;

//     unsigned int num_vertex_types_ = 0;

//   public:
//     computational_dag_csr_impl() = default;
//     ~computational_dag_csr_impl() = default;

//     inline auto vertices() { return vertex_adapter_idx_t(vertices_); }

//     // inline std::vector<vertex_idx> vertices() const {
//     //   std::vector<vertex_idx> vec;
//     //   for (const cdag_vertex_impl v : vertices_) {
//     //     vec.push_back(v.vertex_idx_);
//     //   }
//     //   return vec;
//     // }

//     inline unsigned num_vertices() const { return vertices_.size(); }

//     inline unsigned num_edges() const { return out_neighbors.size(); }

//     inline const std::vector<vertex_idx> &parents(const vertex_idx v) const { return in_neigbors[v]; }

//     inline const std::vector<vertex_idx> &children(const vertex_idx v) const { return out_neigbors[v]; }

//     inline unsigned in_degree(const vertex_idx v) const { return in_neigbors[v].size(); }

//     inline unsigned out_degree(const vertex_idx v) const { return out_neigbors[v].size(); }

//     inline int vertex_work_weight(const vertex_idx v) const { return vertices_[v].work_weight; }

//     inline int vertex_comm_weight(const vertex_idx v) const { return vertices_[v].comm_weight; }

//     inline int vertex_mem_weight(const vertex_idx v) const { return vertices_[v].mem_weight; }

//     inline unsigned vertex_type(const vertex_idx v) const { return vertices_[v].vertex_type; }

//     inline unsigned num_vertex_types() const { return num_vertex_types_; }

//     inline const v_impl &get_vertex_impl(const vertex_idx v) const { return vertices_[v]; }

//     vertex_idx add_vertex(int work_weight, int comm_weight, int mem_weight, unsigned vertex_type) {

//         vertices_.emplace_back(static_cast<vertex_idx>(vertices_.size()), work_weight, comm_weight, mem_weight,
//                                vertex_type);
//         out_neigbors.push_back({});
//         in_neigbors.push_back({});

//         num_vertex_types_ = std::max(num_vertex_types_, vertex_type);

//         return vertices_.back().id;
//     }

//     bool add_edge(vertex_idx source, vertex_idx target) {

//         if (source >= vertices_.size() || target >= vertices_.size())
//             return false;

//         for (const vertex_idx v_idx : out_neigbors[source]) {
//             if (v_idx == target) {
//                 return false;
//             }
//         }

//         out_neigbors[source].push_back(target);
//         in_neigbors[target].push_back(source);
//         num_edges_++;

//         return true;
//     }
// };