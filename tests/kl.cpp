#define BOOST_TEST_MODULE kl
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_base.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "io/graph_file_reader.hpp"

#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p4_g3_l5.txt"};
}

std::vector<std::string> test_graphs_2() { return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt"}; }

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_k-means.txt", "data/spaa/tiny/instance_bicgstab.txt",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.txt", "data/spaa/small/instance_CG_N7_K2_nzP0d6.txt"};
}

// void add_mem_weights(ComputationalDag &dag) {
//     int weight = 1;
//     for (const auto &v : dag.vertices()) {
//         dag.setNodeMemoryWeight(v, weight++ % 3 + 1);
//         dag.setNodeCommunicationWeight(v, weight++ % 3 + 1);
//     }
// }

// BOOST_AUTO_TEST_CASE(kl_memory_test_local_inc) {
//     // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
//     std::vector<std::string> filenames_graph = test_graphs_2();
//     std::vector<std::string> filenames_architectures = test_architectures();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         for (auto &filename_machine : filenames_architectures) {
//             std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
//             name_graph = name_graph.substr(0, name_graph.find_last_of("."));
//             std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
//             name_machine = name_machine.substr(0, name_machine.rfind("."));

//             std::cout << std::endl << "Graph: " << name_graph << std::endl;
//             std::cout << "Architecture: " << name_machine << std::endl;

//             auto [status_graph, graph] =
//                 FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
//             auto [status_architecture, architecture] =
//                 FileReader::readBspArchitecture((cwd / filename_machine).string());

//             add_mem_weights(graph);

//             if (!status_graph || !status_architecture) {

//                 std::cout << "Reading files failed." << std::endl;
//                 BOOST_CHECK(false);
//             }

//             architecture.setSynchronisationCosts(80);
//             architecture.setMemoryConstraintType(LOCAL_INC_EDGES);

//             BspInstance instance(graph, architecture);

//             unsigned max_in_degree = 0;
//             for(const auto &node : graph.vertices()) {

//                 max_in_degree = std::max(max_in_degree, graph.numberOfParents(node));

//             }

//             max_in_degree *= 3;

//             const std::vector<unsigned> bounds_to_test = {max_in_degree + 1, max_in_degree + 4};

//             for (const auto &bound : bounds_to_test) {

//                 instance.getArchitecture().setMemoryBound(bound);

//                 GreedyBspLocking greedy_scheduler;
//                 greedy_scheduler.setUseMemoryConstraint(true);

//                 auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//                 BOOST_CHECK_EQUAL(status, SUCCESS);
//                 BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//                 kl_total_comm_test kl;
//                 kl.setUseMemoryConstraint(true);

//                 kl.improve_schedule_test_2(schedule);

//                 BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//                 auto [status_2, schedule_2] = greedy_scheduler.computeSchedule(instance);

//                 BOOST_CHECK_EQUAL(status_2, SUCCESS);
//                 BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

//                 kl_total_cut_test kl_cut;
//                 kl_cut.setUseMemoryConstraint(true);

//                 kl_cut.improve_schedule_test_2(schedule_2);

//                 BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

//             }
//         }
//     }
// };

// BOOST_AUTO_TEST_CASE(kl_memory_test_local_inc_2) {
//     // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
//     std::vector<std::string> filenames_graph = test_graphs_2();
//     std::vector<std::string> filenames_architectures = test_architectures();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         for (auto &filename_machine : filenames_architectures) {
//             std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
//             name_graph = name_graph.substr(0, name_graph.find_last_of("."));
//             std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
//             name_machine = name_machine.substr(0, name_machine.rfind("."));

//             std::cout << std::endl << "Graph: " << name_graph << std::endl;
//             std::cout << "Architecture: " << name_machine << std::endl;

//             auto [status_graph, graph] =
//                 FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
//             auto [status_architecture, architecture] =
//                 FileReader::readBspArchitecture((cwd / filename_machine).string());

//             add_mem_weights(graph);

//             if (!status_graph || !status_architecture) {

//                 std::cout << "Reading files failed." << std::endl;
//                 BOOST_CHECK(false);
//             }

//             architecture.setSynchronisationCosts(80);
//             architecture.setMemoryConstraintType(LOCAL_INC_EDGES_2);

//             BspInstance instance(graph, architecture);

//             unsigned max_in_degree = 0;
//             for(const auto &node : graph.vertices()) {

//                 max_in_degree = std::max(max_in_degree, graph.numberOfParents(node));

//             }

//             max_in_degree *= 3;

//             const std::vector<unsigned> bounds_to_test = {max_in_degree + 1, max_in_degree + 4};

//             for (const auto &bound : bounds_to_test) {

//                 instance.getArchitecture().setMemoryBound(bound);

//                 GreedyBspLocking greedy_scheduler;
//                 greedy_scheduler.setUseMemoryConstraint(true);

//                 auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//                 BOOST_CHECK_EQUAL(status, SUCCESS);
//                 BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//                 kl_total_comm_test kl;
//                 kl.setUseMemoryConstraint(true);

//                 kl.improve_schedule_test_2(schedule);

//                 BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//                 auto [status_2, schedule_2] = greedy_scheduler.computeSchedule(instance);

//                 BOOST_CHECK_EQUAL(status_2, SUCCESS);
//                 BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

//                 kl_total_cut_test kl_cut;
//                 kl_cut.setUseMemoryConstraint(true);

//                 kl_cut.improve_schedule_test_2(schedule_2);

//                 BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

//             }
//         }
//     }
// };

// BOOST_AUTO_TEST_CASE(kl_memory_test_local_in_out) {
//     // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
//     std::vector<std::string> filenames_graph = test_graphs_2();
//     std::vector<std::string> filenames_architectures = test_architectures();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         for (auto &filename_machine : filenames_architectures) {
//             std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
//             name_graph = name_graph.substr(0, name_graph.find_last_of("."));
//             std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
//             name_machine = name_machine.substr(0, name_machine.rfind("."));

//             std::cout << std::endl << "Graph: " << name_graph << std::endl;
//             std::cout << "Architecture: " << name_machine << std::endl;

//             auto [status_graph, graph] =
//                 FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
//             auto [status_architecture, architecture] =
//                 FileReader::readBspArchitecture((cwd / filename_machine).string());

//             add_mem_weights(graph);

//             if (!status_graph || !status_architecture) {

//                 std::cout << "Reading files failed." << std::endl;
//                 BOOST_CHECK(false);
//             }

//             architecture.setSynchronisationCosts(80);
//             architecture.setMemoryConstraintType(LOCAL_IN_OUT);

//             BspInstance instance(graph, architecture);

//             unsigned max_in_degree = 0;
//             for(const auto &node : graph.vertices()) {

//                 max_in_degree = std::max(max_in_degree, graph.numberOfParents(node));

//             }

//             max_in_degree *= 7;

//             const std::vector<unsigned> bounds_to_test = {max_in_degree, max_in_degree + 4};

//             for (const auto &bound : bounds_to_test) {

//                 instance.getArchitecture().setMemoryBound(bound);

//                 GreedyBspLocking greedy_scheduler;
//                 greedy_scheduler.setUseMemoryConstraint(true);

//                 auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//                 BOOST_CHECK_EQUAL(status, SUCCESS);
//                 BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//                 kl_total_comm_test kl;
//                 kl.setUseMemoryConstraint(true);

//                 kl.improve_schedule_test_1(schedule);

//                 BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//                 auto [status_2, schedule_2] = greedy_scheduler.computeSchedule(instance);

//                 BOOST_CHECK_EQUAL(status_2, SUCCESS);
//                 BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

//                 kl_total_cut_test kl_cut;
//                 kl_cut.setUseMemoryConstraint(true);

//                 kl_cut.improve_schedule_test_1(schedule_2);

//                 BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

//             }
//         }
//     }
// };

// BOOST_AUTO_TEST_CASE(kl_memory_test_local) {
//     // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
//     std::vector<std::string> filenames_graph = test_graphs_2();
//     std::vector<std::string> filenames_architectures = test_architectures();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         for (auto &filename_machine : filenames_architectures) {
//             std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
//             name_graph = name_graph.substr(0, name_graph.find_last_of("."));
//             std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
//             name_machine = name_machine.substr(0, name_machine.rfind("."));

//             std::cout << std::endl << "Graph: " << name_graph << std::endl;
//             std::cout << "Architecture: " << name_machine << std::endl;

//             auto [status_graph, graph] =
//                 FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
//             auto [status_architecture, architecture] =
//                 FileReader::readBspArchitecture((cwd / filename_machine).string());

//             add_mem_weights(graph);

//             if (!status_graph || !status_architecture) {

//                 std::cout << "Reading files failed." << std::endl;
//                 BOOST_CHECK(false);
//             }

//             architecture.setMemoryConstraintType(LOCAL);
//             architecture.setSynchronisationCosts(80);

//             BspInstance instance(graph, architecture);

//             const std::vector<unsigned> bounds_to_test = {10, 20};

//             for (const auto &bound : bounds_to_test) {

//                 instance.getArchitecture().setMemoryBound(bound);

//                 GreedyBspLocking greedy_scheduler;
//                 greedy_scheduler.setUseMemoryConstraint(true);

//                 auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//                 BOOST_CHECK_EQUAL(status, SUCCESS);
//                 BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//                 kl_total_comm_test kl;
//                 kl.setUseMemoryConstraint(true);

//                 kl.improve_schedule_test_2(schedule);

//                 BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//                 auto [status_2, schedule_2] = greedy_scheduler.computeSchedule(instance);

//                 BOOST_CHECK_EQUAL(status_2, SUCCESS);
//                 BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

//                 kl_total_cut_test kl_cut;
//                 kl_cut.setUseMemoryConstraint(true);

//                 kl_cut.improve_schedule_test_2(schedule_2);

//                 BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
//                 BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

//             }
//         }
//     }
// };

// BOOST_AUTO_TEST_CASE(kl_base_medium) {

//     std::vector<std::string> filenames_graph = test_graphs_medium();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {

//         auto [status_graph, graph] = FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         } else {
//             std::cout << "File read:" << filename_graph << std::endl;
//         }

//         BspArchitecture arch;
//         arch.setNumberOfProcessors(3);
//         arch.setSynchronisationCosts(25);
//         arch.setCommunicationCosts(10);

//         BspInstance instance(graph, arch);

//         GreedyVarianceFillupScheduler greedy_scheduler;

//         auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//         BOOST_CHECK_EQUAL(status, SUCCESS);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         kl_total_comm_test kl;

//         kl.improve_schedule_test_2(schedule);

//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);
//     }
// }

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

// BOOST_AUTO_TEST_CASE(kl_base_2) {

//     std::vector<std::string> filenames_graph = test_graphs();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {

//         auto [status_graph, graph] = FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         } else {
//             std::cout << "File read:" << filename_graph << std::endl;
//         }

//         BspArchitecture arch(4,3,5);

//         BspInstance instance(graph, arch);

//         GreedyBspLocking greedy_scheduler;

//         kl_total_comm_test kl;

//         FunnelBfs coarser(&greedy_scheduler, &kl);

//         auto [status, schedule] = coarser.computeSchedule(instance);

//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);
//     }
// }

// BOOST_AUTO_TEST_CASE(kl_base_3) {

//     std::vector<std::string> filenames_graph = test_graphs_dot();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {

//         auto [status_graph, graph] = FileReader::readComputationalDagDotFormat((cwd / filename_graph).string());

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         } else {
//             std::cout << "File read:" << filename_graph << std::endl;
//         }

//         BspArchitecture arch;
//         arch.setCommunicationCosts(200);
//         arch.setSynchronisationCosts(1000000);
//         arch.setProcessorsWithTypes({0,0,1,1,1,1});
//         arch.setMemoryConstraintType(LOCAL);
//         arch.setMemoryBound({6000000, 6000000, 6000000, 6000000, 6000000, 600000});

//         BspInstance instance(graph, arch);
//         instance.setDiagonalCompatibilityMatrix(graph.getNumberOfNodeTypes());

//         GreedyBspLocking greedy_scheduler;
//         greedy_scheduler.setUseMemoryConstraint(true);

//         auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//         BOOST_CHECK_EQUAL(status, SUCCESS);
//         BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//         BOOST_CHECK(schedule.satisfiesNodeTypeConstraints());
//         BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//         kl_total_comm_test kl;

//         kl.set_quick_pass(true);
//         kl.setUseMemoryConstraint(true);
//         kl.set_alternate_reset_remove_superstep(false);

//         //kl.get_current_schedule().use_node_communication_costs = true;

//         kl.improve_schedule_test_2(schedule);

//         BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
//         BOOST_CHECK(schedule.satisfiesNodeTypeConstraints());
//         BOOST_CHECK(schedule.satisfiesMemoryConstraints());

//     }
// }

// BOOST_AUTO_TEST_CASE(kl_base_4) {

//     std::vector<std::string> filenames_graph = test_graphs();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {

//         auto [status_graph, graph] = FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         } else {
//             std::cout << "File read:" << filename_graph << std::endl;
//         }

//         BspArchitecture arch;
//         arch.setSynchronisationCosts(25);
//         arch.setCommunicationCosts(5);

//         BspInstance instance(graph, arch);

//         GreedyVarianceFillupScheduler greedy_scheduler;

//         auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//         BOOST_CHECK_EQUAL(status, SUCCESS);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         kl_total_comm_test kl;
//         kl.improve_schedule_test_2(schedule);

//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//     }
// }

// BOOST_AUTO_TEST_CASE(kl_base_5) {

//     std::vector<std::string> filenames_graph = test_graphs();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {

//         auto [status_graph, graph] = FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         } else {
//             std::cout << "File read:" << filename_graph << std::endl;
//         }

//         BspArchitecture arch;

//         BspInstance instance(graph, arch);

//         GreedyVarianceFillupScheduler greedy_scheduler;

//         auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//         BOOST_CHECK_EQUAL(status, SUCCESS);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         kl_total_cut_test kl;

//         kl.improve_schedule_test_1(schedule);

//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);
//     }
// }

// BOOST_AUTO_TEST_CASE(kl_base_7) {

//     std::vector<std::string> filenames_graph = test_graphs();

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "one-stop-parallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {

//         auto [status_graph, graph] = FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         } else {
//             std::cout << "File read:" << filename_graph << std::endl;
//         }

//         BspArchitecture arch;
//         arch.setSynchronisationCosts(25);

//         BspInstance instance(graph, arch);

//         GreedyVarianceFillupScheduler greedy_scheduler;

//         auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

//         BOOST_CHECK_EQUAL(status, SUCCESS);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         kl_total_cut_test kl;
//         kl.improve_schedule_test_2(schedule);

//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);
//     }
// }