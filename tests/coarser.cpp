#define BOOST_TEST_MODULE COARSER_TEST
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>

#include "coarser/hdagg/hdagg_coarser.hpp"
#include "graph_implementations/computational_dag_edge_idx_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"
#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"

std::vector<std::string> test_graphs() { return {"data/spaa/tiny/instance_bicgstab.txt", "data/spaa/tiny/instance_k-means.txt", 
"data/spaa/tiny/instance_pregel.txt", "data/spaa/tiny/instance_spmv_N6_nzP0d4.txt"}; }

using namespace osp;

using VertexType = vertex_idx_t<computational_dag_edge_idx_vector_impl_def_t>;

bool check_vertex_map(std::vector<std::vector<VertexType>> &map, std::size_t size) {

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

template<typename ComputationalDag>
bool check_vertex_map_constraints(std::vector<std::vector<VertexType>> &map, ComputationalDag &dag,
                                  v_type_t<ComputationalDag> size_threshold,
                                  v_memw_t<ComputationalDag> memory_threshold,
                                  v_workw_t<ComputationalDag> work_threshold,
                                  v_commw_t<ComputationalDag> communication_threshold) {

    std::unordered_set<VertexType> vertices;

    for (auto &super_node : map) {

        v_memw_t<ComputationalDag> memory = 0;
        v_workw_t<ComputationalDag> work = 0;
        v_commw_t<ComputationalDag> communication = 0;

        if (super_node.size() > size_threshold) {
            return false;
        }

        if (super_node.size() == 0) {
            return false;
        }

        for (auto &v : super_node) {

            memory += dag.vertex_mem_weight(v);
            work += dag.vertex_work_weight(v);
            communication += dag.vertex_comm_weight(v);

            if (dag.vertex_type(v) != dag.vertex_type(super_node[0])) {
                return false;
            }
        }

        if (memory > memory_threshold || work > work_threshold || communication > communication_threshold) {
            return false;
        }
    }
    return true;
}

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

        using graph_t = computational_dag_edge_idx_vector_impl_def_t;

        BspInstance<graph_t> instance;

        bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        bool status_architecture =
            file_reader::readBspArchitecture((cwd / "data/machine_params/p3.txt").string(), instance.getArchitecture());

        if (!status_graph || !status_architecture) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<graph_t> coarse_instance;
        coarse_instance.setArchitecture(instance.getArchitecture());
        std::vector<std::vector<VertexType>> vertex_map;
        std::vector<VertexType> reverse_vertex_map;

        hdagg_coarser<graph_t, graph_t> coarser;

        coarser.coarseDag(instance.getComputationalDag(), coarse_instance.getComputationalDag(), vertex_map,
                          reverse_vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, instance.getComputationalDag().num_vertices()));

        GreedyBspScheduler<graph_t> scheduler;

        auto [status_sched, schedule] = scheduler.computeSchedule(coarse_instance);

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BspSchedule<graph_t> schedule_out(instance);

        BOOST_CHECK_EQUAL(pull_back_schedule(schedule, vertex_map, schedule_out), true);
        BOOST_CHECK(schedule_out.satisfiesPrecedenceConstraints());
    }
};