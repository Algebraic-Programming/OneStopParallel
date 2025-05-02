#define BOOST_TEST_MODULE COARSER_TEST
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>

#include "coarser/hdagg/hdagg_coarser.hpp"
#include "coarser/CoarseAndSchedule.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"

std::vector<std::string> tiny_spaa_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.hdag",
            "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.hdag",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.hdag",
            "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag",
            "data/spaa/tiny/instance_exp_N4_K2_nzP0d5.hdag",
            "data/spaa/tiny/instance_exp_N5_K3_nzP0d4.hdag",
            "data/spaa/tiny/instance_exp_N6_K4_nzP0d25.hdag",
            "data/spaa/tiny/instance_k-means.hdag",
            "data/spaa/tiny/instance_k-NN_3_gyro_m.hdag",
            "data/spaa/tiny/instance_kNN_N4_K3_nzP0d5.hdag",
            "data/spaa/tiny/instance_kNN_N5_K3_nzP0d3.hdag",
            "data/spaa/tiny/instance_kNN_N6_K4_nzP0d2.hdag",
            "data/spaa/tiny/instance_pregel.hdag",
            "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag",
            "data/spaa/tiny/instance_spmv_N7_nzP0d35.hdag",
            "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag"};
}

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
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

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
            file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.getArchitecture());

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

        CoarseAndSchedule<graph_t, graph_t> coarse_and_schedule(coarser, scheduler);
        auto [status, schedule2] = coarse_and_schedule.computeSchedule(instance);
        BOOST_CHECK(status == RETURN_STATUS::SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());

    }
};


BOOST_AUTO_TEST_CASE(coarser_hdagg_test_diff_graph_impl) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

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

        using graph_t1 = computational_dag_edge_idx_vector_impl_def_t;
        using graph_t2 = computational_dag_vector_impl_def_t;


        BspInstance<graph_t1> instance;

        bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        bool status_architecture =
            file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.getArchitecture());

        if (!status_graph || !status_architecture) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<graph_t2> coarse_instance;
        BspArchitecture<graph_t2> architecture_t2(instance.getArchitecture());
        coarse_instance.setArchitecture(architecture_t2);
        std::vector<std::vector<VertexType>> vertex_map;
        std::vector<VertexType> reverse_vertex_map;

        
        hdagg_coarser<graph_t1, graph_t2> coarser;

        coarser.coarseDag(instance.getComputationalDag(), coarse_instance.getComputationalDag(), vertex_map,
                          reverse_vertex_map);

        BOOST_CHECK(check_vertex_map(vertex_map, instance.getComputationalDag().num_vertices()));

        GreedyBspScheduler<graph_t2> scheduler;

        auto [status_sched, schedule] = scheduler.computeSchedule(coarse_instance);

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BspSchedule<graph_t1> schedule_out(instance);

        BOOST_CHECK_EQUAL(pull_back_schedule(schedule, vertex_map, schedule_out), true);
        BOOST_CHECK(schedule_out.satisfiesPrecedenceConstraints());


        CoarseAndSchedule<graph_t1, graph_t2> coarse_and_schedule(coarser, scheduler);
        auto [status, schedule2] = coarse_and_schedule.computeSchedule(instance);
        BOOST_CHECK(status == RETURN_STATUS::SUCCESS || status == RETURN_STATUS::BEST_FOUND);
        BOOST_CHECK(schedule2.satisfiesPrecedenceConstraints());

    }
};