#define BOOST_TEST_MODULE ComputationalDag
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>
#include <filesystem>
#include "file_interactions/FileReader.hpp"

#include "model/ComputationalDag.hpp"

std::vector<std::string> test_graphs() {

    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", 
            "data/spaa/test/test1.txt"}; 
}

BOOST_AUTO_TEST_CASE(longest_edge_triangle_parallel) {

    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {


        auto [status_graph, graph] = FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
        // auto [status_graph, graph] = FileReader::readComputationalDagMartixMarketFormat((cwd / filename_graph).string());

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        auto deleted_edges = graph.long_edges_in_triangles();
        auto finish_time = std::chrono::high_resolution_clock::now();

        std::cout << "\n" << filename_graph << std::endl;

        std::cout << "Time for long_edges_in_triangles: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count() << "ms"
                  << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        auto deleted_edges_parallel = graph.long_edges_in_triangles_parallel();
        finish_time = std::chrono::high_resolution_clock::now();

        std::cout << "Time for long_edges_in_triangles_parallel: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count() << "ms"
                  << std::endl;

        BOOST_CHECK_EQUAL(deleted_edges.size(), deleted_edges_parallel.size());

        for (const auto &edge : deleted_edges) {
            BOOST_CHECK(deleted_edges_parallel.find(edge) != deleted_edges_parallel.cend());
        }

        for (const auto &edge : deleted_edges_parallel) {
            BOOST_CHECK(deleted_edges.find(edge) != deleted_edges.cend());
        }
    }
};

BOOST_AUTO_TEST_CASE(DAG1) {

    ComputationalDag dag;

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 0);
    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 0);

    VertexType v1 = dag.addVertex(2, 9);
    VertexType v2 = dag.addVertex(3, 8);
    VertexType v3 = dag.addVertex(4, 7);
    VertexType v4 = dag.addVertex(5, 6);

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 4);

    VertexType v5 = dag.addVertex(6, 5);
    VertexType v6 = dag.addVertex(7, 4);
    VertexType v7 = dag.addVertex(8, 3);
    VertexType v8 = dag.addVertex(9, 2);

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 8);
    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 0);

    // EdgeType e12 =
    dag.addEdge(v1, v2, 2);
    // EdgeType e13 =
    dag.addEdge(v1, v3, 3);
    // EdgeType e14 =
    dag.addEdge(v1, v4, 4);
    // EdgeType e25 =
    dag.addEdge(v2, v5, 5);

    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 4);

    // EdgeType e35 =
    dag.addEdge(v3, v5, 6);
    // EdgeType e36 =
    dag.addEdge(v3, v6, 7);
    // EdgeType e27 =
    dag.addEdge(v2, v7, 8);
    // EdgeType e58 =
    dag.addEdge(v5, v8, 9);
    // EdgeType e48 =
    dag.addEdge(v4, v8, 9);

    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 9);

    BOOST_CHECK_EQUAL(dag.sourceVertices().size(), 1);
    BOOST_CHECK_EQUAL(dag.sourceVertices()[0], v1);
    BOOST_CHECK_EQUAL(dag.sinkVertices().size(), 3);

    BOOST_CHECK_EQUAL(dag.has_path(v1,v2), true);
    BOOST_CHECK_EQUAL(dag.has_path(v1,v3), true);
    BOOST_CHECK_EQUAL(dag.has_path(v1,v4), true);
    BOOST_CHECK_EQUAL(dag.has_path(v1,v5), true);
    BOOST_CHECK_EQUAL(dag.has_path(v1,v6), true);
    BOOST_CHECK_EQUAL(dag.has_path(v1,v7), true);
    BOOST_CHECK_EQUAL(dag.has_path(v1,v8), true);

    BOOST_CHECK_EQUAL(dag.has_path(v2,v1), false);
    BOOST_CHECK_EQUAL(dag.has_path(v3,v1), false);
    BOOST_CHECK_EQUAL(dag.has_path(v4,v1), false);
    BOOST_CHECK_EQUAL(dag.has_path(v5,v1), false);
    BOOST_CHECK_EQUAL(dag.has_path(v6,v1), false);
    BOOST_CHECK_EQUAL(dag.has_path(v7,v1), false);
    BOOST_CHECK_EQUAL(dag.has_path(v8,v1), false);

    BOOST_CHECK_EQUAL(dag.has_path(v2,v5), true);
    BOOST_CHECK_EQUAL(dag.has_path(v2,v8), true);
    BOOST_CHECK_EQUAL(dag.has_path(v2,v7), true);

    BOOST_CHECK_EQUAL(dag.has_path(v3,v5), true);
    BOOST_CHECK_EQUAL(dag.has_path(v3,v6), true);
    BOOST_CHECK_EQUAL(dag.has_path(v3,v8), true);

    BOOST_CHECK_EQUAL(dag.has_path(v4,v8), true);

    BOOST_CHECK_EQUAL(dag.has_path(v5,v8), true);


    BOOST_CHECK_EQUAL(dag.has_path(v2,v3), false);
    BOOST_CHECK_EQUAL(dag.has_path(v2,v4), false);
    
    BOOST_CHECK_EQUAL(dag.has_path(v3,v2), false);
    BOOST_CHECK_EQUAL(dag.has_path(v3,v4), false);
    BOOST_CHECK_EQUAL(dag.has_path(v3,v7), false);

    BOOST_CHECK_EQUAL(dag.has_path(v4,v2), false);
    BOOST_CHECK_EQUAL(dag.has_path(v4,v3), false);
    BOOST_CHECK_EQUAL(dag.has_path(v4,v5), false);
    BOOST_CHECK_EQUAL(dag.has_path(v4,v6), false);
    BOOST_CHECK_EQUAL(dag.has_path(v4,v7), false);

    BOOST_CHECK_EQUAL(dag.has_path(v5,v2), false);
    BOOST_CHECK_EQUAL(dag.has_path(v5,v3), false);
    BOOST_CHECK_EQUAL(dag.has_path(v5,v4), false);
    BOOST_CHECK_EQUAL(dag.has_path(v5,v6), false);
    BOOST_CHECK_EQUAL(dag.has_path(v5,v7), false);

    BOOST_CHECK_EQUAL(dag.has_path(v6,v2), false);
    BOOST_CHECK_EQUAL(dag.has_path(v6,v3), false);
    BOOST_CHECK_EQUAL(dag.has_path(v6,v4), false);
    BOOST_CHECK_EQUAL(dag.has_path(v6,v5), false);
    BOOST_CHECK_EQUAL(dag.has_path(v6,v7), false);
    BOOST_CHECK_EQUAL(dag.has_path(v6,v8), false);

    BOOST_CHECK_EQUAL(dag.has_path(v7,v2), false);
    BOOST_CHECK_EQUAL(dag.has_path(v7,v3), false);
    BOOST_CHECK_EQUAL(dag.has_path(v7,v4), false);
    BOOST_CHECK_EQUAL(dag.has_path(v7,v5), false);
    BOOST_CHECK_EQUAL(dag.has_path(v7,v6), false);
    BOOST_CHECK_EQUAL(dag.has_path(v7,v8), false);

    BOOST_CHECK_EQUAL(dag.has_path(v8,v2), false);
    BOOST_CHECK_EQUAL(dag.has_path(v8,v3), false);
    BOOST_CHECK_EQUAL(dag.has_path(v8,v4), false);
    BOOST_CHECK_EQUAL(dag.has_path(v8,v5), false);
    BOOST_CHECK_EQUAL(dag.has_path(v8,v6), false);
    BOOST_CHECK_EQUAL(dag.has_path(v8,v7), false);




}

BOOST_AUTO_TEST_CASE(ComputationalDagConstructor) {
    const std::vector<std::vector<int>> out(

        {{7}, {}, {0}, {2}, {}, {2, 0}, {1, 2, 0}, {}, {4}, {6, 1, 5}}

    );
    const std::vector<int> workW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});
    const std::vector<int> commW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    const ComputationalDag graph(out, workW, commW);
    const ComputationalDag graph_empty;

    BOOST_CHECK_EQUAL(graph.numberOfVertices(), std::distance(graph.vertices().begin(), graph.vertices().end()));
    BOOST_CHECK_EQUAL(graph.numberOfEdges(), std::distance(graph.edges().begin(), graph.edges().end()));
    for (const auto &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.numberOfParents(v), std::distance(graph.parents(v).begin(), graph.parents(v).end()));
        BOOST_CHECK_EQUAL(graph.numberOfChildren(v), std::distance(graph.children(v).begin(), graph.children(v).end()));
    }

    for (const VertexType i : graph.vertices()) {
        const Vertex v = graph.getGraph()[i];
        BOOST_CHECK_EQUAL(v.workWeight, workW[i]);
        BOOST_CHECK_EQUAL(v.workWeight, graph.nodeWorkWeight(i));
        BOOST_CHECK_EQUAL(v.communicationWeight, commW[i]);
        BOOST_CHECK_EQUAL(v.communicationWeight, graph.nodeCommunicationWeight(i));
    }

    std::cout << "Checking workW sums:" << std::endl;
    BOOST_CHECK_EQUAL(graph.sumOfVerticesWorkWeights({0, 1}), 2);
    {
        int sum_of_work_weights = graph.nodeWorkWeight(0) + graph.nodeWorkWeight(1);
        BOOST_CHECK_EQUAL(2, sum_of_work_weights);
    }
    BOOST_CHECK_EQUAL(graph.sumOfVerticesWorkWeights({5, 3}), 4);
    BOOST_CHECK_EQUAL(graph.sumOfVerticesWorkWeights({}), 0);
    BOOST_CHECK_EQUAL(graph.sumOfVerticesWorkWeights({0, 1, 2, 3, 4, 5}), 9);

    BOOST_CHECK_EQUAL(graph_empty.sumOfVerticesWorkWeights({}), 0);

    std::cout << "Checking every in edge is contained in out edge:" << std::endl;

    int num_edges = 0;
    for (const auto &vertex : graph.vertices()) {
        num_edges += graph.numberOfChildren(vertex);
        for (const auto &parent : graph.parents(vertex)) {
            BOOST_CHECK(std::any_of(graph.children(parent).cbegin(), graph.children(parent).cend(),
                                    [vertex](VertexType k) { return k == vertex; }));
        }
    }

    std::cout << "Checking every out edge is contained in in edge:" << std::endl;

    for (const auto &vertex : graph.vertices()) {
        for (const auto &child : graph.children(vertex)) {

            BOOST_CHECK(std::any_of(graph.parents(child).cbegin(), graph.parents(child).cend(),
                                    [vertex](VertexType k) { return k == vertex; }));
        }
    }

    std::cout << "Checking topological order:" << std::endl;
    std::vector<VertexType> top_order = graph.GetTopOrder(ComputationalDag::AS_IT_COMES);
    BOOST_CHECK(top_order.size() == graph.numberOfVertices());
    BOOST_CHECK(graph_empty.GetTopOrder().size() == graph_empty.numberOfVertices());

    std::vector<size_t> index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    top_order = graph.GetTopOrder(ComputationalDag::MAX_CHILDREN);
    BOOST_CHECK(top_order.size() == graph.numberOfVertices());
    BOOST_CHECK(graph_empty.GetTopOrder().size() == graph_empty.numberOfVertices());

    index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    top_order = graph.GetTopOrder(ComputationalDag::RANDOM);
    BOOST_CHECK(top_order.size() == graph.numberOfVertices());
    BOOST_CHECK(graph_empty.GetTopOrder().size() == graph_empty.numberOfVertices());

    index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    top_order = graph.GetTopOrder(ComputationalDag::MINIMAL_NUMBER);
    BOOST_CHECK(top_order.size() == graph.numberOfVertices());
    BOOST_CHECK(graph_empty.GetTopOrder().size() == graph_empty.numberOfVertices());

    index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    std::set<VertexType> all_nodes;
    for (const auto &vertex : graph.vertices()) {
        all_nodes.emplace(vertex);
    }
    std::set<VertexType> nodes_a({8, 0});
    std::set<VertexType> nodes_b({6, 2, 5, 3});
    std::set<VertexType> nodes_c({6, 9, 1});

    std::vector<bool> bool_a(graph.numberOfVertices(), false);
    std::vector<bool> bool_b(graph.numberOfVertices(), false);
    std::vector<bool> bool_c(graph.numberOfVertices(), false);

    for (auto &i : nodes_a) {
        bool_a[i] = true;
    }
    for (auto &i : nodes_b) {
        bool_b[i] = true;
    }
    for (auto &i : nodes_c) {
        bool_c[i] = true;
    }

    BOOST_CHECK(graph.GetFilteredTopOrder(bool_a) == std::vector<VertexType>({0, 8}) ||
                graph.GetFilteredTopOrder(bool_a) == std::vector<VertexType>({8, 0}));
    BOOST_CHECK(graph.GetFilteredTopOrder(bool_b)[3] == 2);
    BOOST_CHECK(graph.GetFilteredTopOrder(bool_c) == std::vector<VertexType>({9, 6, 1}));

    std::cout << "Checking longest path:" << std::endl;
    BOOST_CHECK_EQUAL(graph.longestPath(all_nodes), 4);
    BOOST_CHECK_EQUAL(graph.longestPath(nodes_a), 0);
    BOOST_CHECK_EQUAL(graph.longestPath(nodes_b), 1);
    BOOST_CHECK_EQUAL(graph.longestPath(nodes_c), 2);

    BOOST_CHECK_EQUAL(graph_empty.longestPath({}), 0);

    std::vector<VertexType> longest_path = graph.longestChain();

    std::vector<VertexType> long_chain1({9, 6, 2, 0, 7});
    std::vector<VertexType> long_chain2({9, 5, 2, 0, 7});

    BOOST_CHECK_EQUAL(graph.longestPath(all_nodes) + 1, graph.longestChain().size());
    BOOST_CHECK(longest_path == long_chain1 || longest_path == long_chain2);

    BOOST_CHECK(graph_empty.longestChain() == std::vector<VertexType>({}));

    std::cout << "Checking ancestors and descendants:" << std::endl;
    /*
        BOOST_CHECK(graph.ancestors(9) == std::vector<VertexType>({9}));
        BOOST_CHECK(graph.ancestors(2) == std::vector<VertexType>({2, 6, 9, 5, 3}));
        BOOST_CHECK(graph.ancestors(4) == std::vector<VertexType>({4, 8}));
        BOOST_CHECK(graph.ancestors(5) == std::vector<VertexType>({5, 9}));

        BOOST_CHECK(graph.successors(9) == std::vector<VertexType>({9, 5, 6, 1, 2, 0, 7}));
        BOOST_CHECK(graph.successors(3) == std::vector<VertexType>({3, 2, 0, 7}));
        BOOST_CHECK(graph.successors(0) == std::vector<VertexType>({0, 7}));
        BOOST_CHECK(graph.successors(8) == std::vector<VertexType>({8, 4}));
        BOOST_CHECK(graph.successors(4) == std::vector<VertexType>({4}));
    */
    std::vector<unsigned> top_dist({4, 3, 3, 1, 2, 2, 2, 5, 1, 1});
    std::vector<unsigned> bottom_dist({2, 1, 3, 4, 1, 4, 4, 1, 2, 5});

    BOOST_CHECK(graph.get_top_node_distance() == top_dist);
    BOOST_CHECK(graph.get_bottom_node_distance() == bottom_dist);

    const std::vector<std::vector<int>> graph_second_Out = {
        {1, 2}, {3, 4}, {4, 5}, {6}, {}, {6}, {},
    };
    const std::vector<int> graph_second_workW = {1, 1, 1, 1, 1, 1, 3};
    const std::vector<int> graph_second_commW = graph_second_workW;

    ComputationalDag graph_second(graph_second_Out, graph_second_workW, graph_second_commW);

    std::vector<unsigned> top_dist_second({1, 2, 2, 3, 3, 3, 4});
    std::vector<unsigned> bottom_dist_second({4, 3, 3, 2, 1, 2, 1});

    BOOST_CHECK(graph_second.get_top_node_distance() == top_dist_second);
    BOOST_CHECK(graph_second.get_bottom_node_distance() == bottom_dist_second);

    std::cout << "Checking strict poset integer map:" << std::endl;

    std::vector<double> poisson_params({0, 0.08, 0.1, 0.2, 0.5, 1, 4});

    for (unsigned loops = 0; loops < 10; loops++) {
        for (unsigned noise = 0; noise < 6; noise++) {
            for (auto &pois_para : poisson_params) {

                std::vector<int> poset_int_map = graph.get_strict_poset_integer_map(noise, pois_para);

                for (const auto &vertex : graph.vertices()) {
                    for (const auto &child : graph.children(vertex)) {
                        BOOST_CHECK_LE(poset_int_map[vertex] + 1, poset_int_map[child]);
                    }
                }
            }
        }
    }

    BOOST_CHECK(graph.critical_path_weight() == 7);

    const std::pair<std::vector<VertexType>, ComputationalDag> rev_graph_pair = graph.reverse_graph();
    const std::vector<VertexType> &vertex_mapping_rev_graph = rev_graph_pair.first;
    const ComputationalDag &rev_graph = rev_graph_pair.second;

    BOOST_CHECK_EQUAL(graph.numberOfVertices(), rev_graph.numberOfVertices());
    BOOST_CHECK_EQUAL(graph.numberOfEdges(), rev_graph.numberOfEdges());

    for (VertexType vert = 0; vert < graph.numberOfVertices(); vert++) {
        BOOST_CHECK_EQUAL(graph.nodeWorkWeight(vert), rev_graph.nodeWorkWeight(vertex_mapping_rev_graph[vert]));
        BOOST_CHECK_EQUAL(graph.nodeCommunicationWeight(vert), rev_graph.nodeCommunicationWeight(vertex_mapping_rev_graph[vert]));
        BOOST_CHECK_EQUAL(graph.nodeMemoryWeight(vert), rev_graph.nodeMemoryWeight(vertex_mapping_rev_graph[vert]));
        BOOST_CHECK_EQUAL(graph.nodeType(vert), rev_graph.nodeType(vertex_mapping_rev_graph[vert]));
    }

    for (VertexType vert_1 = 0; vert_1 < graph.numberOfVertices(); vert_1++) {
        for (VertexType vert_2 = 0; vert_2 < graph.numberOfVertices(); vert_2++) {
            bool edge_in_graph = boost::edge(vert_1, vert_2, graph.getGraph()).second;
            bool rev_edge_in_rev_graph = boost::edge(vertex_mapping_rev_graph[vert_2], vertex_mapping_rev_graph[vert_1], rev_graph.getGraph()).second;
            BOOST_CHECK_EQUAL(edge_in_graph, rev_edge_in_rev_graph);
        }
    }
}