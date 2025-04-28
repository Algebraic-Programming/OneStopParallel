#define BOOST_TEST_MODULE kl
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_base.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"

#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_k-means.txt", "data/spaa/tiny/instance_bicgstab.txt",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.txt"};
}


BOOST_AUTO_TEST_CASE(kl_base_1) {

    using graph = computational_dag_edge_idx_vector_impl_def_t;
    using VertexType = graph::vertex_idx;

    graph dag;

    const VertexType v1 = dag.add_vertex(2, 9, 2);
    const VertexType v2 = dag.add_vertex(3, 8, 4);
    const VertexType v3 = dag.add_vertex(4, 7, 3);
    const VertexType v4 = dag.add_vertex(5, 6, 2);
    const VertexType v5 = dag.add_vertex(6, 5, 6);
    const VertexType v6 = dag.add_vertex(7, 4, 2);
    const VertexType v7 = dag.add_vertex(8, 3, 4);
    const VertexType v8 = dag.add_vertex(9, 2, 1);

    dag.add_edge(v1, v2, 2);
    dag.add_edge(v1, v3, 2);
    dag.add_edge(v1, v4, 2);
    dag.add_edge(v2, v5, 12);
    dag.add_edge(v3, v5, 6);
    dag.add_edge(v3, v6, 7);
    dag.add_edge(v5, v8, 9);
    dag.add_edge(v4, v8, 9);

    BspArchitecture<graph> arch;

    BspInstance<graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.setAssignedProcessors({0, 0, 0, 0, 0, 0, 0, 0});
    schedule.setAssignedSupersteps({0, 0, 0, 0, 0, 0, 0, 0});

    schedule.updateNumberOfSupersteps();

    using kl_move = kl_move<graph>;

    kl_total_comm_test<graph> kl;

    kl.test_setup_schedule(schedule);

    kl_current_schedule_total<graph> &kl_current_schedule = kl.get_current_schedule();

    BOOST_CHECK_EQUAL(kl_current_schedule.step_max_work[0], 44.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.step_second_max_work[0], 0.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.num_steps(), 1);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_cost, 44.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_feasible, true);

    kl_move move_1(v1, 0, 6.0 - 2.0, 0, 0, 1, 0);

    kl_current_schedule.apply_move(move_1);

    BOOST_CHECK_EQUAL(kl_current_schedule.step_max_work[0], 42.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.step_second_max_work[0], 2.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.num_steps(), 1);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_cost, 48.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_feasible, false);
    BOOST_CHECK_EQUAL(kl_current_schedule.cost_f->compute_current_costs(), 48.0);

    kl_move move_2(v2, 0, 7.0, 0, 0, 1, 0);

    kl_current_schedule.apply_move(move_2);

    BOOST_CHECK_EQUAL(kl_current_schedule.step_max_work[0], 39.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.step_second_max_work[0], 5.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.num_steps(), 1);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_cost, 55.0);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_feasible, false);
    BOOST_CHECK_EQUAL(kl_current_schedule.cost_f->compute_current_costs(), 55.0);

    kl.initialize_gain_heap_test({0, 1, 2, 3, 4, 5, 6, 7});

    auto &node_gains = kl.get_node_gains();
    auto &node_change_in_costs = kl.get_node_change_in_costs();

    BOOST_CHECK_EQUAL(node_gains[v1][0][1], 4.0);
    BOOST_CHECK_EQUAL(node_change_in_costs[v1][0][1], 0.0);

    BOOST_CHECK_EQUAL(node_gains[v1][1][1], std::numeric_limits<double>::lowest());
    BOOST_CHECK_EQUAL(node_change_in_costs[v1][1][1], 0.0);

    BOOST_CHECK_EQUAL(node_gains[v2][0][1], 19.0);
    BOOST_CHECK_EQUAL(node_change_in_costs[v2][0][1], -7.0);

    kl_move move_3(v7, 0, 7.0, 0, 0, 1, 0);
    kl_current_schedule.apply_move(move_3);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_feasible, false);

    kl_move move_4(v2, 0, 7.0, 1, 0, 0, 0);
    kl_current_schedule.apply_move(move_4);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_feasible, false);

    kl_move move_5(v1, 0, 7.0, 1, 0, 0, 0);
    kl_current_schedule.apply_move(move_5);
    BOOST_CHECK_EQUAL(kl_current_schedule.current_feasible, true);
};

BOOST_AUTO_TEST_CASE(kl_total_comm_test_1) {

    std::vector<std::string> filenames_graph = test_graphs();

    using graph = computational_dag_edge_idx_vector_impl_def_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> test_scheduler;

    for (auto &filename_graph : filenames_graph) {

        BspInstance<graph> instance;

        bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        instance.getArchitecture().setSynchronisationCosts(5);
        instance.getArchitecture().setCommunicationCosts(5);
        instance.getArchitecture().setNumberOfProcessors(4);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        std::pair<RETURN_STATUS, BspSchedule<graph>> result = test_scheduler.computeSchedule(instance);

        BOOST_CHECK_EQUAL(SUCCESS, result.first);
        BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
        BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());

        kl_total_comm_test<graph> kl;

        auto status = kl.improve_schedule_test_1(result.second);

        BOOST_CHECK(status == SUCCESS || status == BEST_FOUND);
        BOOST_CHECK_EQUAL(result.second.satisfiesPrecedenceConstraints(), true);
    }
}

BOOST_AUTO_TEST_CASE(kl_total_comm_test_2) {

    std::vector<std::string> filenames_graph = test_graphs();

    using graph = computational_dag_edge_idx_vector_impl_def_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> test_scheduler;

    for (auto &filename_graph : filenames_graph) {

        BspInstance<graph> instance;

        bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());

        instance.getArchitecture().setSynchronisationCosts(5);
        instance.getArchitecture().setCommunicationCosts(5);
        instance.getArchitecture().setNumberOfProcessors(4);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        std::pair<RETURN_STATUS, BspSchedule<graph>> result = test_scheduler.computeSchedule(instance);

        BOOST_CHECK_EQUAL(SUCCESS, result.first);
        BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
        BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());

        kl_total_comm_test<graph> kl;

        auto status = kl.improve_schedule_test_2(result.second);

        BOOST_CHECK(status == SUCCESS || status == BEST_FOUND);
        BOOST_CHECK_EQUAL(result.second.satisfiesPrecedenceConstraints(), true);
    }
}

BOOST_AUTO_TEST_CASE(kl_total_cut_test_1) {

    std::vector<std::string> filenames_graph = test_graphs();

    using graph = computational_dag_edge_idx_vector_impl_def_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> test_scheduler;

    for (auto &filename_graph : filenames_graph) {

        BspInstance<graph> instance;

        bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());
        instance.getArchitecture().setSynchronisationCosts(5);
        instance.getArchitecture().setCommunicationCosts(5);
        instance.getArchitecture().setNumberOfProcessors(4);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        std::pair<RETURN_STATUS, BspSchedule<graph>> result = test_scheduler.computeSchedule(instance);

        BOOST_CHECK_EQUAL(SUCCESS, result.first);
        BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
        BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());

        kl_total_cut_test<graph> kl;

        auto status = kl.improve_schedule_test_1(result.second);

        BOOST_CHECK(status == SUCCESS || status == BEST_FOUND);
        BOOST_CHECK_EQUAL(result.second.satisfiesPrecedenceConstraints(), true);
    }
}

BOOST_AUTO_TEST_CASE(kl_total_cut_test_2) {

    std::vector<std::string> filenames_graph = test_graphs();

    using graph = computational_dag_edge_idx_vector_impl_def_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> test_scheduler;

    for (auto &filename_graph : filenames_graph) {

        BspInstance<graph> instance;

        bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                            instance.getComputationalDag());
        instance.getArchitecture().setSynchronisationCosts(5);
        instance.getArchitecture().setCommunicationCosts(5);
        instance.getArchitecture().setNumberOfProcessors(4);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        std::pair<RETURN_STATUS, BspSchedule<graph>> result = test_scheduler.computeSchedule(instance);

        BOOST_CHECK_EQUAL(SUCCESS, result.first);
        BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
        BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());

        kl_total_cut_test<graph> kl;

        auto status = kl.improve_schedule_test_2(result.second);

        BOOST_CHECK(status == SUCCESS || status == BEST_FOUND);
        BOOST_CHECK_EQUAL(result.second.satisfiesPrecedenceConstraints(), true);
    }
}
