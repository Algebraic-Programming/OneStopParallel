#define BOOST_TEST_MODULE BSP_SCHEDULERS_CSR
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "scheduler/Coarsers/CacheLineGluer.hpp"
#include "scheduler/Coarsers/HDaggCoarser.hpp"
#include "scheduler/Coarsers/SquashA.hpp"
#include "scheduler/HDagg/HDagg_simple.hpp"
#include "scheduler/LocalSearchSchedulers/HillClimbingScheduler.hpp"
#include "scheduler/Scheduler.hpp"
#include "scheduler/Serial/Serial.hpp"
#include "scheduler/SubArchitectureSchedulers/SubArchitectures.hpp"
#include "file_interactions/FileReader.hpp"

#include "scheduler/SchedulePermutations/ScheduleNodePermuter.hpp"
#include "model/BspSchedule.hpp"
#include "simulation/BspSptrsvCSR.hpp"

void print_x_vector(const std::vector<double> &vec) {
    std::cout << "[";
    for (unsigned i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
};

BOOST_AUTO_TEST_CASE(BspScheduler_csr) {

    const std::vector<std::vector<int>> out(

        {{7}, {}, {0}, {2}, {}, {2, 0}, {1, 2, 0}, {}, {4}, {6, 1, 5}}

    );
    const std::vector<int> workW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});
    const std::vector<int> commW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    ComputationalDag graph(out, workW, commW);
    BspArchitecture architecture(2, 1, 1);

    BspInstance instance(graph, architecture);

    BspSchedule schedule(instance);

    schedule.setAssignedProcessors({1, 0, 1, 0, 1, 0, 0, 1, 1, 0});
    schedule.setAssignedSupersteps({2, 2, 2, 1, 1, 1, 1, 3, 0, 0});

    BOOST_CHECK(schedule.numberOfSupersteps() == 4);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
    schedule.setAutoCommunicationSchedule();
    BOOST_CHECK(schedule.hasValidCommSchedule());

    std::vector<size_t> perm = {8, 6, 7, 4, 5,
                                2, 3, 9, 1, 0}; // = schedule_node_permuter(schedule, 8, LOOP_PROCESSORS);
    std::vector<size_t> perm_snake = {8, 6, 7, 3, 2,
                                      4, 5, 9, 1, 0}; // = schedule_node_permuter_basic(schedule, SNAKE_PROCESSORS);
    std::vector<size_t> test_perm = schedule_node_permuter_basic(schedule, LOOP_PROCESSORS);

    //  std::vector<size_t> perm_inv(perm.size());
    //  for (size_t i = 0; i < perm.size(); i++) {
    //      perm_inv[perm[i]] = i;
    //   }

    BspSptrsvCSR schedule_csr(instance, true);

    schedule_csr.setup_csr(schedule, perm);

    BOOST_CHECK_EQUAL(schedule_csr.num_supersteps, 4);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[0][0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[0][1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[1][0], 2);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[1][1], 5);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[2][0], 6);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[2][1], 7);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[3][0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[3][1], 9);

    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[0][0], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[0][1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[1][0], 3);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[1][1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[2][0], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[2][1], 2);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[3][0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[3][1], 1);

    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[2], 2);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[3], 4);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[4], 6);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[5], 7);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[6], 9);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[7], 12);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[8], 16);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[9], 20);

    BOOST_CHECK_EQUAL(schedule_csr.col_idx[0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[2], 0);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[3], 2);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[4], 0);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[5], 3);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[6], 4);

    BOOST_CHECK_EQUAL(schedule_csr.col_idx[7], 1);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[8], 5);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[9], 0);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[10], 3);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[11], 6);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[12], 2);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[13], 3);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[14], 4);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[15], 7);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[16], 2);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[17], 3);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[18], 7);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[19], 8);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[20], 8);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[21], 9);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx.size(), schedule_csr.val.size());
    BOOST_CHECK_EQUAL(schedule_csr.col_idx.size(),
                      instance.numberOfVertices() + instance.getComputationalDag().numberOfEdges());

    std::vector<double> result_vector = schedule_csr.get_result();
    std::vector<double> x_initial(instance.numberOfVertices(), 0);
    for (unsigned i = 0; i < perm.size(); i++) {
        x_initial[i] = result_vector[perm[i]];
    }

    schedule_csr.simulate_sptrsv_serial();
    result_vector = schedule_csr.get_result();
    std::vector<double> x_serial(instance.numberOfVertices(), 0);
    for (unsigned i = 0; i < perm.size(); i++) {
        x_serial[i] = result_vector[perm[i]];
    }

    schedule_csr.reset_x();
    result_vector = schedule_csr.get_result();
    std::vector<double> x_reset(instance.numberOfVertices(), 0);
    for (unsigned i = 0; i < perm.size(); i++) {
        x_reset[i] = result_vector[perm[i]];
    }

    schedule_csr.simulate_sptrsv();
    result_vector = schedule_csr.get_result();
    std::vector<double> x_sptrsv(instance.numberOfVertices(), 0);
    for (unsigned i = 0; i < perm.size(); i++) {
        x_sptrsv[i] = result_vector[perm[i]];
    }

    schedule_csr.setup_csr_snake(schedule, perm_snake);
    schedule_csr.simulate_sptrsv();
    result_vector = schedule_csr.get_result();
    std::vector<double> x_snake(instance.numberOfVertices(), 0);
    for (unsigned i = 0; i < perm_snake.size(); i++) {
        x_snake[i] = result_vector[perm_snake[i]];
    }

    std::cout << "\nSnake tests\n";

    BOOST_CHECK_EQUAL(schedule_csr.num_supersteps, 4);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[0][0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[0][1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[1][0], 3);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[1][1], 2);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[2][0], 6);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[2][1], 7);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[3][0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_ptr[3][1], 9);

    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[0][0], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[0][1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[1][0], 3);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[1][1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[2][0], 1);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[2][1], 2);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[3][0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.step_proc_num[3][1], 1);

    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[2], 2);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[3], 4);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[4], 5);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[5], 7);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[6], 9);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[7], 12);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[8], 16);
    BOOST_CHECK_EQUAL(schedule_csr.row_ptr[9], 20);

    BOOST_CHECK_EQUAL(schedule_csr.col_idx[0], 0);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[1], 1);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[2], 1);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[3], 2);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[4], 3);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[5], 0);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[6], 4);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[7], 0);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[8], 5);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[9], 0);

    BOOST_CHECK_EQUAL(schedule_csr.col_idx[10], 5);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[11], 6);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[12], 3);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[13], 4);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[14], 5);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[15], 7);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[16], 4);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[17], 5);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[18], 7);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[19], 8);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[20], 8);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx[21], 9);
    BOOST_CHECK_EQUAL(schedule_csr.col_idx.size(), schedule_csr.val.size());
    BOOST_CHECK_EQUAL(schedule_csr.col_idx.size(),
                      instance.numberOfVertices() + instance.getComputationalDag().numberOfEdges());

    for (const auto &val : x_initial) {
        BOOST_CHECK_CLOSE(val, 1.0, 0.00001);
    }
    for (const auto &val : x_reset) {
        BOOST_CHECK_CLOSE(val, 1.0, 0.00001);
    }

    for (unsigned i = 0; i < x_serial.size(); i++) {
        BOOST_CHECK_CLOSE(x_serial[i], x_sptrsv[i], 0.00001);
    }

    for (unsigned i = 0; i < x_serial.size(); i++) {
        BOOST_CHECK_CLOSE(x_serial[i], x_snake[i], 0.00001);
    }

    print_x_vector(x_serial);
    print_x_vector(x_sptrsv);
    print_x_vector(x_snake);

    schedule_csr.setup_csr_no_permutation(schedule);
    BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), 4);

    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices.size(), 4);

    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[0][0].size(), 1);
    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[0][1].size(), 1);

    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[1][0].size(), 3);
    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[1][1].size(), 1);

    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[2][0].size(), 1);
    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[2][1].size(), 2);

    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[3].size(), 2);

    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[3][0].size(), 0);
    BOOST_CHECK_EQUAL(schedule_csr.vector_step_processor_vertices[3][1].size(), 1);

    schedule_csr.simulate_sptrsv_no_permutation();

    result_vector = schedule_csr.get_result();
    std::vector<double> x_noperm(instance.numberOfVertices(), 0);
    for (unsigned i = 0; i < perm.size(); i++) {
        x_noperm[i] = result_vector[i];
    }

    for (unsigned i = 0; i < x_serial.size(); i++) {
        BOOST_CHECK_CLOSE(x_serial[i], x_noperm[i], 0.00001);
    }

    print_x_vector(x_noperm);

    auto xx = schedule_csr.compute_sptrsv();

    print_x_vector(xx);


    for (unsigned i = 0; i < x_serial.size(); i++) {
        BOOST_CHECK_CLOSE(x_serial[i], xx[i], 0.00001);
    }

    schedule_csr.setup_csr_no_barrier(schedule, perm);

    schedule_csr.simulate_sptrsv_no_barrier();

    result_vector = schedule_csr.get_result();
    std::vector<double> x_nobarrier(instance.numberOfVertices(), 0);
    for (unsigned i = 0; i < perm.size(); i++) {
        x_nobarrier[i] = result_vector[i];
    }


    print_x_vector(x_nobarrier);

    for (unsigned i = 0; i < x_serial.size(); i++) {
        BOOST_CHECK_CLOSE(x_serial[i], x_nobarrier[i], 0.00001);
    }



}
