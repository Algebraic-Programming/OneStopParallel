#define BOOST_TEST_MODULE SCHEDULE_NODE_PERMUTER_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "scheduler/ContractRefineScheduler/BalDMixR.hpp"
#include "scheduler/SchedulePermutations/ScheduleNodePermuter.hpp"
#include "scheduler/Scheduler.hpp"
#include "file_interactions/FileReader.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", "data/spaa/small/instance_kNN_N20_K5_nzP0d2.txt",
            "data/spaa/small/instance_exp_N10_K8_nzP0d2.txt"};
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p16_g1_l5_numa2.txt"};
}

void print_bsp_schedule(BspSchedule bsp_schedule) {
    std::vector<std::vector<std::vector<unsigned>>> schedule(
        bsp_schedule.numberOfSupersteps(),
        std::vector<std::vector<unsigned>>(bsp_schedule.getInstance().numberOfProcessors(), std::vector<unsigned>()));

    for (unsigned node = 0; node < bsp_schedule.getInstance().numberOfVertices(); node++) {
        schedule[bsp_schedule.assignedSuperstep(node)][bsp_schedule.assignedProcessor(node)].push_back(node);
    }

    std::cout << std::endl << "Schedule:" << std::endl;
    for (size_t i = 0; i < schedule.size(); i++) {
        std::cout << std::endl << "Superstep " << i << std::endl;
        for (size_t j = 0; j < schedule[i].size(); j++) {
            std::cout << "Processor " << j << ": ";
            for (auto &node : schedule[i][j]) {
                std::cout << node << ", ";
            }
            std::cout << std::endl;
        }
    }
}

void run_test(Scheduler *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph =
                filename_graph.substr(filename_machine.find_last_of("/\\") + 1, filename_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            auto [status_graph, graph] =
                FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspInstance instance(graph, architecture);

            std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);

            print_bsp_schedule(result.second);

            BOOST_CHECK_EQUAL(SUCCESS, result.first);

            BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
            BOOST_CHECK(result.second.hasValidCommSchedule());

            std::vector<size_t> perm = schedule_node_permuter(result.second, 8, SNAKE_PROCESSORS, true);
            std::vector<size_t> id_perm(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter(result.second, 16, SNAKE_PROCESSORS, false);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter(result.second, 4, LOOP_PROCESSORS, false);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter(result.second, 8, LOOP_PROCESSORS, true);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter_basic(result.second, LOOP_PROCESSORS);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter_basic(result.second, SNAKE_PROCESSORS);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter_small_relation(result.second, LOOP_PROCESSORS);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter_small_relation(result.second, SNAKE_PROCESSORS);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter_dfs(result.second, LOOP_PROCESSORS);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter_dfs(result.second, SNAKE_PROCESSORS);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                }
            }

            perm = schedule_node_permuter(result.second, 8, PROCESSOR_FIRST, true);
            id_perm = std::vector<size_t>(perm.size(), 0);
            std::iota(id_perm.begin(), id_perm.end(), 0);
            BOOST_CHECK(std::is_permutation(perm.begin(), perm.end(), id_perm.begin(), id_perm.end()));
            for (size_t old_node_name = 0; old_node_name < perm.size(); old_node_name++) {
                for (const auto &old_in_edge : result.second.getInstance().getComputationalDag().in_edges(old_node_name)) {
                    if (result.second.assignedProcessor(old_in_edge.m_source) <= result.second.assignedProcessor(old_node_name)) {
                        BOOST_CHECK_LE(perm[old_in_edge.m_source], perm[old_node_name]);
                    } else {
                        BOOST_CHECK_GE(perm[old_in_edge.m_source], perm[old_node_name]);
                    }
                }
            }
        }
    }
};

BOOST_AUTO_TEST_CASE(Schedule_Node_Permuter_test) {
    BalDMixR test;
    run_test(&test);
}