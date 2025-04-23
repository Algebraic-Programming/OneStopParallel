#define BOOST_TEST_MODULE cuthill_mckee
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "file_interactions/FileReader.hpp"
#include "model/dag_algorithms/cuthill_mckee.hpp"
#include "model/dag_algorithms/top_sort.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_k-means.txt",
            "data/spaa/test/test1.txt"};
}

BOOST_AUTO_TEST_CASE(cuthill_mckee_1) {

    ComputationalDag dag;

    const VertexType v1 = dag.addVertex(2, 9);
    const VertexType v2 = dag.addVertex(3, 8);
    const VertexType v3 = dag.addVertex(4, 7);
    const VertexType v4 = dag.addVertex(5, 6);
    const VertexType v5 = dag.addVertex(6, 5);
    const VertexType v6 = dag.addVertex(7, 4);
    const VertexType v7 = dag.addVertex(8, 3);
    const VertexType v8 = dag.addVertex(9, 2);

    dag.addEdge(v1, v2, 2);
    dag.addEdge(v1, v3, 3);
    dag.addEdge(v1, v4, 4);
    dag.addEdge(v2, v5, 5);
    dag.addEdge(v3, v5, 6);
    dag.addEdge(v3, v6, 7);
    dag.addEdge(v2, v7, 8);
    dag.addEdge(v5, v8, 9);
    dag.addEdge(v4, v8, 9);

    std::vector<VertexType> cm_wavefront = dag_algorithms::cuthill_mckee_wavefront(dag);
    std::vector<unsigned> expected_cm_wavefront = {0, 3, 1, 2, 6, 4, 5, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_wavefront.begin(), cm_wavefront.end(), expected_cm_wavefront.begin(),
                                  expected_cm_wavefront.end());

    cm_wavefront = dag_algorithms::cuthill_mckee_wavefront(dag, true);
    expected_cm_wavefront = {0, 2, 3, 1, 5, 6, 4, 7};

    BOOST_CHECK_EQUAL_COLLECTIONS(cm_wavefront.begin(), cm_wavefront.end(), expected_cm_wavefront.begin(),
                                  expected_cm_wavefront.end());

    std::vector<VertexType> cm_undirected;
    std::vector<unsigned> expected_cm_undirected;

    cm_undirected = dag_algorithms::cuthill_mckee_undirected(dag, true);
    expected_cm_undirected = {7, 3, 4, 0, 1, 2, 6, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    cm_undirected = dag_algorithms::cuthill_mckee_undirected(dag, false);
    expected_cm_undirected = {0, 3, 1, 2, 7, 6, 4, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    cm_undirected = dag_algorithms::cuthill_mckee_undirected(dag, true, true);
    expected_cm_undirected = {3, 4, 5, 1, 2, 7, 6, 0};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    std::vector<VertexType> top_sort = dag_algorithms::top_sort_priority(dag, cm_undirected);
    std::vector<unsigned> expected_top_sort = {0, 3, 1, 2, 4, 7, 6, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(top_sort.begin(), top_sort.end(), expected_top_sort.begin(), expected_top_sort.end());

    cm_undirected = dag_algorithms::cuthill_mckee_undirected(dag, false, true);
    expected_cm_undirected = {0, 2, 3, 1, 6, 7, 5, 4};

    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    dag.addEdge(8, 9);
    dag.addEdge(9, 10);

    cm_undirected = dag_algorithms::cuthill_mckee_undirected(dag, true);
    expected_cm_undirected = {7, 3, 4, 0, 1, 2, 6, 5, 10, 9, 8};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());

    cm_undirected = dag_algorithms::cuthill_mckee_undirected(dag, false);
    expected_cm_undirected = {0, 3, 1, 2, 7, 6, 4, 5, 8, 9, 10};
    BOOST_CHECK_EQUAL_COLLECTIONS(cm_undirected.begin(), cm_undirected.end(), expected_cm_undirected.begin(),
                                  expected_cm_undirected.end());
};


bool is_permutation(const std::vector<VertexType>& vec) {
    std::vector<VertexType> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());
    for (unsigned i = 0; i < sorted_vec.size(); ++i) {
        if (sorted_vec[i] != i) {
            return false;
        }
    }
    return true;
}


bool is_top_sort(const std::vector<VertexType>& vec, const ComputationalDag& dag) {
    std::unordered_map<unsigned, unsigned> position;
    for (unsigned i = 0; i < vec.size(); ++i) {
        position[vec[i]] = i;
    }

    for(const auto& vertex : dag.vertices()) {

        for(const auto& child : dag.children(vertex)) {
            if (position[vertex] > position[child]) {
                return false;
            }
        }

    }

    return true;
}

BOOST_AUTO_TEST_CASE(cuthill_mckee_2) {

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

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filename_graph << std::endl;
        }

        std::vector<VertexType> wavefront = dag_algorithms::cuthill_mckee_wavefront(graph);
        BOOST_CHECK(is_permutation(wavefront));

        wavefront = dag_algorithms::cuthill_mckee_wavefront(graph, true);
        BOOST_CHECK(is_permutation(wavefront));

        const auto cm_undirected = dag_algorithms::cuthill_mckee_undirected(graph, true, true);
        BOOST_CHECK(is_permutation(cm_undirected));

        std::vector<VertexType> top_sort = dag_algorithms::top_sort_priority(graph, cm_undirected);

        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));

    }
};


BOOST_AUTO_TEST_CASE(top_sort) {

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

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filename_graph << std::endl;
        }
        
        std::vector<VertexType> top_sort = dag_algorithms::top_sort_bfs(graph);
        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));

        top_sort = dag_algorithms::top_sort_dfs(graph);
        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));


        top_sort = dag_algorithms::top_sort_heavy_edges(graph, true);
        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));


        top_sort = dag_algorithms::top_sort_heavy_edges(graph, false);
        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));


        top_sort = dag_algorithms::top_sort_locality(graph);
        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));

        top_sort = dag_algorithms::top_sort_max_children(graph);
        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));

        top_sort = dag_algorithms::top_sort_random(graph);
        BOOST_CHECK(is_permutation(top_sort));
        BOOST_CHECK(is_top_sort(top_sort, graph));

    }


};