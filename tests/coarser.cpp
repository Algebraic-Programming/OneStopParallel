#define BOOST_TEST_MODULE COARSER_TEST
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>

#include "coarser/BspScheduleCoarser.hpp"
#include "coarser/hdagg/hdagg_coarser.hpp"
#include "coarser/hdagg/hdagg_coarser_variant.hpp"
#include "coarser/heavy_edges/HeavyEdgeCoarser.hpp"
#include "coarser/top_order/top_order.hpp"
#include "file_interactions/FileReader.hpp"
#include "model/dag_algorithms/cuthill_mckee.hpp"
#include "model/dag_algorithms/top_sort.hpp"
#include "scheduler/GreedySchedulers/GreedyBspLocking.hpp"
#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "scheduler/LocalSearchSchedulers/KernighanLin/kl_total_cut.hpp"
#include "dag_divider/WavefrontComponentScheduler.hpp"
#include "dag_divider/WavefrontComponentDivider.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.txt", "data/spaa/test/test1.txt"};
}



bool check_vertex_map(std::vector<std::vector<VertexType>> &map, unsigned size) {

    std::unordered_set<VertexType> vertices;

    for (auto &v : map) {
        for (auto &v2 : v) {
            if (vertices.find(v2) != vertices.end()) {
                return false;
            }
            vertices.insert(v2);
        }
    }

    return vertices.size() == size;
}

bool check_vertex_map_constraints(std::vector<std::vector<VertexType>> &map, ComputationalDag &dag,
                                  unsigned size_threshold, int memory_threshold, int work_threshold,
                                  int communication_threshold) {

    std::unordered_set<VertexType> vertices;

    for (auto &super_node : map) {

        int memory = 0;
        int work = 0;
        int communication = 0;

        if (super_node.size() > size_threshold) {
            return false;
        }

        if (super_node.size() == 0) {
            return false;
        }

        for (auto &v : super_node) {

            memory += dag.nodeMemoryWeight(v);
            work += dag.nodeWorkWeight(v);
            communication += dag.nodeCommunicationWeight(v);

            if (dag.nodeType(v) != dag.nodeType(super_node[0])) {
                return false;
            }
        }

        if (memory > memory_threshold || work > work_threshold || communication > communication_threshold) {
            return false;
        }
    }
    return true;
}

BOOST_AUTO_TEST_CASE(HeavyEdgeCoarser_test) {

    ComputationalDag dag;

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 0);
    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 0);

    VertexType v1 = dag.addVertex(2, 1);
    VertexType v2 = dag.addVertex(3, 1);
    VertexType v3 = dag.addVertex(4, 1);
    VertexType v4 = dag.addVertex(5, 1);

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 4);

    VertexType v5 = dag.addVertex(6, 1);
    VertexType v6 = dag.addVertex(7, 1);
    VertexType v7 = dag.addVertex(8, 1);
    VertexType v8 = dag.addVertex(9, 1);

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 8);
    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 0);

    // EdgeType e12 =
    dag.addEdge(v1, v2, 1);
    // EdgeType e13 =
    dag.addEdge(v1, v3, 1);
    // EdgeType e14 =
    dag.addEdge(v1, v4, 1000);
    // EdgeType e25 =
    dag.addEdge(v2, v5, 1);

    // EdgeType e35 =
    dag.addEdge(v3, v5, 1);
    // EdgeType e36 =
    dag.addEdge(v3, v6, 1);
    // EdgeType e27 =
    dag.addEdge(v2, v7, 1);
    // EdgeType e58 =
    dag.addEdge(v5, v8, 1);
    // EdgeType e48 =
    dag.addEdge(v4, v8, 1);

    HeavyEdgeCoarser coarser(2.0, 0.7, 1.0);

    ComputationalDag coarse_dag;
    std::vector<std::vector<VertexType>> vertex_map;

    coarser.coarseDag(dag, coarse_dag, vertex_map);

    GreedyBspLocking scheduler;
    BspArchitecture arch;
    BspInstance coarse_instance(coarse_dag, arch);
    BspInstance instance(dag, arch);

    auto [status, schedule] = scheduler.computeSchedule(coarse_instance);

    auto [status_pull_back, schedule_pulled] = pull_back_schedule(instance, schedule, vertex_map);

    schedule_pulled.setAutoCommunicationSchedule();

    BOOST_CHECK(schedule_pulled.satisfiesPrecedenceConstraints());
}

BOOST_AUTO_TEST_CASE(BspScheduleCoarser_test) {

    ComputationalDag dag;

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 0);
    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 0);

    VertexType v1 = dag.addVertex(2, 1);
    VertexType v2 = dag.addVertex(3, 1);
    VertexType v3 = dag.addVertex(4, 1);
    VertexType v4 = dag.addVertex(5, 1);

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 4);

    VertexType v5 = dag.addVertex(6, 1);
    VertexType v6 = dag.addVertex(7, 1);
    VertexType v7 = dag.addVertex(8, 1);
    VertexType v8 = dag.addVertex(9, 1);

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 8);
    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 0);

    // EdgeType e12 =
    dag.addEdge(v1, v2, 1);
    // EdgeType e13 =
    dag.addEdge(v1, v3, 1);
    // EdgeType e14 =
    dag.addEdge(v1, v4, 1000);
    // EdgeType e25 =
    dag.addEdge(v2, v5, 1);

    // EdgeType e35 =
    dag.addEdge(v3, v5, 1);
    // EdgeType e36 =
    dag.addEdge(v3, v6, 1);
    // EdgeType e27 =
    dag.addEdge(v2, v7, 1);
    // EdgeType e58 =
    dag.addEdge(v5, v8, 1);
    // EdgeType e48 =
    dag.addEdge(v4, v8, 1);

    GreedyBspScheduler scheduler;
    BspArchitecture arch;

    BspInstance instance(dag, arch);

    auto [status, schedule] = scheduler.computeSchedule(instance);

    SetSchedule set_schedule(schedule);

    unsigned num_nodes = 0;
    for (auto step : set_schedule.step_processor_vertices) {

        for (auto proc : step) {

            if (proc.size() > 0)
                num_nodes++;
        }
    }

    BspScheduleCoarser coarser(schedule);

    ComputationalDag coarse_dag;
    std::vector<std::vector<VertexType>> vertex_map;
    coarser.coarseDag(dag, coarse_dag, vertex_map);

    BOOST_CHECK(num_nodes == coarse_dag.numberOfVertices());

    BspInstance coarse_instance(coarse_dag, arch);

    auto [c_status, c_schedule] = scheduler.computeSchedule(coarse_instance);

    auto [status_pull_back, schedule_pulled] = pull_back_schedule(instance, c_schedule, vertex_map);

    schedule_pulled.setAutoCommunicationSchedule();

    BOOST_CHECK(schedule_pulled.satisfiesPrecedenceConstraints());

    CoarseAndSchedule sc(coarser, scheduler);

    sc.computeSchedule(instance);
}

BOOST_AUTO_TEST_CASE(coarser_top_order_test) {
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

        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        auto [status_graph, graph] = FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());

        if (!status_graph) {
            std::cout << "Reading files failed." << std::endl;
        }

        std::vector<std::vector<VertexType>> top_sort_vec;

        top_sort_vec.push_back(dag_algorithms::top_sort_bfs(graph));
        top_sort_vec.push_back(dag_algorithms::top_sort_dfs(graph));
        top_sort_vec.push_back(dag_algorithms::top_sort_locality(graph));
        top_sort_vec.push_back(dag_algorithms::top_sort_max_children(graph));
        top_sort_vec.push_back(dag_algorithms::top_sort_random(graph));
        top_sort_vec.push_back(dag_algorithms::top_sort_heavy_edges(graph, true));
        top_sort_vec.push_back(
            dag_algorithms::top_sort_priority_node_type(graph, dag_algorithms::cuthill_mckee_wavefront(graph)));
        top_sort_vec.push_back(dag_algorithms::top_sort_priority_node_type(
            graph, dag_algorithms::cuthill_mckee_undirected(graph, true, true)));

        for (const auto &top_sort : top_sort_vec) {

            ComputationalDag coarse_dag;
            std::vector<std::vector<VertexType>> vertex_map;

            top_order coarser(top_sort);
            coarser.set_work_threshold(4);
            coarser.set_degree_threshold(2);
            coarser.coarseDag(graph, coarse_dag, vertex_map);

            BOOST_CHECK(check_vertex_map(vertex_map, graph.numberOfVertices()));

            GreedyBspLocking scheduler;
            BspArchitecture arch;
            BspInstance coarse_instance(coarse_dag, arch);
            BspInstance instance(graph, arch);

            auto [status, schedule] = scheduler.computeSchedule(coarse_instance);

            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

            auto [status_pull_back, schedule_pulled] = pull_back_schedule(instance, schedule, vertex_map);

            BOOST_CHECK(schedule_pulled.satisfiesPrecedenceConstraints());
        }
    }
};


BOOST_AUTO_TEST_CASE(coarser_hdagg_test) {
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

        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        auto [status_graph, graph] = FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());

        if (!status_graph) {
            std::cout << "Reading files failed." << std::endl;
        }

        ComputationalDag coarse_dag;
        std::vector<std::vector<VertexType>> vertex_map;

        hdagg_coarser coarser;
        // coarser.set_work_threshold(4);
        // coarser.set_degree_threshold(2);
        coarser.coarseDag(graph, coarse_dag, vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, graph.numberOfVertices()));

        GreedyBspLocking scheduler;
        BspArchitecture arch;
        BspInstance coarse_instance(coarse_dag, arch);
        BspInstance instance(graph, arch);

        auto [status, schedule] = scheduler.computeSchedule(coarse_instance);

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        auto [status_pull_back, schedule_pulled] = pull_back_schedule(instance, schedule, vertex_map);
        BOOST_CHECK(schedule_pulled.satisfiesPrecedenceConstraints());
    }
};