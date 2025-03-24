#pragma once

#include "iterator_concepts.hpp"

namespace osp {

using vertex_idx = size_t;

// directed_graph concept without explicit edges
template<typename T, typename = void>
struct is_directed_graph : std::false_type {};

template<typename T>
struct is_directed_graph<T, std::void_t<
    decltype(std::declval<T>().vertices()),
    decltype(std::declval<T>().num_vertices()),
    decltype(std::declval<T>().num_edges()),
    decltype(std::declval<T>().parents(std::declval<vertex_idx>())),
    decltype(std::declval<T>().children(std::declval<vertex_idx>())),
    decltype(std::declval<T>().in_degree(std::declval<vertex_idx>())),
    decltype(std::declval<T>().out_degree(std::declval<vertex_idx>()))>> 
    : std::conjunction<
    is_input_range_of<decltype(std::declval<T>().vertices()), vertex_idx>,
    std::is_unsigned<decltype(std::declval<T>().num_vertices())>,
    std::is_unsigned<decltype(std::declval<T>().num_edges())>,
    is_input_range_of<decltype(std::declval<T>().parents(std::declval<vertex_idx>())), vertex_idx>,
    is_input_range_of<decltype(std::declval<T>().children(std::declval<vertex_idx>())), vertex_idx>,
    std::is_unsigned<decltype(std::declval<T>().in_degree(std::declval<vertex_idx>()))>,
    std::is_unsigned<decltype(std::declval<T>().out_degree(std::declval<vertex_idx>()))>> {};

template<typename T>
inline constexpr bool is_directed_graph_v = is_directed_graph<T>::value;


// template<typename T>
// struct is_constructable_from_directed_graph : std::false_type {};

// template<typename T>
// struct is_constructable_from_directed_graph<std::vector<T>> : std::conjunction<
//     is_directed_graph<T>> {};

// template<typename T>
// inline constexpr bool is_constructable_from_directed_graph_v = is_constructable_from_directed_graph<T>::value;



using edge_idx = size_t;

struct directed_edge_descriptor {

    edge_idx idx;

    vertex_idx source;
    vertex_idx target;

    directed_edge_descriptor() = default;
    directed_edge_descriptor(edge_idx idx, vertex_idx source, vertex_idx target) : idx(idx), source(source), target(target) {}
    ~directed_edge_descriptor() = default;
};

// directed_graph_edge_idx concept
template<typename T, typename = void>
struct is_directed_graph_edge_desc : std::false_type {};

template<typename T>
struct is_directed_graph_edge_desc<T, std::void_t<
    //decltype(std::declval<T>().edges()),
    decltype(std::declval<T>().out_edges(std::declval<vertex_idx>())),
    decltype(std::declval<T>().in_edges(std::declval<vertex_idx>()))>> : std::conjunction<
    is_directed_graph<T>,
    //is_input_range_of<decltype(std::declval<T>().edges()), directed_edge_descriptor>,
    is_input_range_of<decltype(std::declval<T>().out_edges(std::declval<vertex_idx>())), directed_edge_descriptor>,
    is_input_range_of<decltype(std::declval<T>().in_edges(std::declval<vertex_idx>())), directed_edge_descriptor>
    > {};

template<typename T>
inline constexpr bool is_directed_graph_edge_desc_v = is_directed_graph_edge_desc<T>::value;


} // namespace osp





// using vertex_idx = unsigned int;

// template<typename T>
// concept directed_graph = requires(T graph, vertex_idx v_idx) {
    
//     { graph.vertices() } -> std::ranges::input_range;
//     { graph.num_vertices() } -> std::unsigned_integral;

//     { graph.parents(v_idx) } -> std::ranges::input_range;
//     { graph.children(v_idx) } -> std::ranges::input_range;

//     { graph.in_degree(v_idx) } -> std::unsigned_integral;
//     { graph.out_degree(v_idx) } -> std::unsigned_integral;
// };


// using edge_idx = unsigned int;

// template<typename T>
// concept edge_type = requires(T edge_t) {
//     { edge_t.source } -> std::same_as<vertex_idx>;
//     { edge_t.target } -> std::same_as<vertex_idx>;
//     { edge_t.edge_idx_ } -> std::same_as<edge_idx>;
// };

// template<typename T>
// concept directed_graph_edge_idx = requires(T graph, vertex_idx v_idx, edge_idx e_idx, edge_type edge) {
//     requires directed_graph<T>;

//     { graph.edges() } -> std::ranges::input_range;
//     { graph.num_edges() } -> std::unsigned_integral;

//     { graph.out_edges(v_idx) } -> std::ranges::input_range;
//     { graph.in_edges(v_idx) } -> std::ranges::input_range;

//     { graph.get_edge_type(e_idx) } -> std::same_as<edge_type>;
//     { graph.source(e_idx) } -> std::same_as<vertex_idx>;
//     { graph.target(e_idx) } -> std::same_as<vertex_idx>;
// };



// template<typename T>
// concept cdag_vertex = requires(T v) {
//     { v.work_weight } -> std::numeric;
//     { v.comm_weight } -> std::numeric;
//     { v.mem_weight } -> std::numeric;
// };

// template<typename T>
// concept cdag_typed_vertex = requires(T v) {
    
//     requires cdag_vertex<T>;
//     { v.vertex_type } -> std::unsigned_integral;
// };

// template<typename T>
// concept cdag_edge = requires(T e) {
//     { e.comm_weight } -> std::numeric;
// };

// template<typename T>
// concept computation_dag = requires(T graph, vertex_idx v_idx, edge_idx e_idx) {
//     requires directed_graph<T>;

//     { graph.get_vertex(v_idx) } -> std::same_as<cdag_vertex>;

//     { graph.vertex_work_weight(v_idx) } -> std::numeric;
//     { graph.vertex_comm_weight(v_idx) } -> std::numeric;
//     { graph.vertex_mem_weight(v_idx) } -> std::numeric;
// };

// template<typename T>
// concept computation_dag_edge_idx = requires(T graph, vertex_idx v_idx, edge_idx e_idx) {
//     requires directed_graph_edge_idx<T> && computation_dag<T>;

//     { graph.get_edge(e_idx) } -> std::same_as<cdag_edge>;
//     { graph.edge_comm_weight(e_idx) } -> std::numeric;
// };