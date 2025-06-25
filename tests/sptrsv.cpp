/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

// #define EIGEN_FOUND 1


#ifdef EIGEN_FOUND

#define BOOST_TEST_MODULE SPTRSV

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>


#include "graph_algorithms/directed_graph_util.hpp"
#include "graph_algorithms/directed_graph_path_util.hpp"
#include "graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"
#include "supplemental/sptrsv.hpp"
#include "bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "supplemental/ScheduleNodePermuter.hpp"
#include "supplemental/sptrsv.hpp"

using namespace osp;

#define EPSILON 1e-10 
bool compare_vectors(Eigen::VectorXd &v1, Eigen::VectorXd &v2) {
    std::cout << std::fixed;
    std::cout << std::setprecision(15);

    assert(v1.size() == v2.size());
    bool same = true;

    for (long long int i=0; i < v1.size(); ++i){
        //std::cout << "Ind: " << i << ": | " << v1[i] << " - " << v2[i] << " | = " << abs(v1[i]-v2[i]) << "\n";  
        if( abs(v1[i] - v2[i]) / (abs(v1[i]) + abs(v2[i]) + EPSILON) > EPSILON ){
            std::cout << "We have differences in the matrix in position: " << i << std::endl;
            std::cout << v1[i] << " , " << v2[i] << std::endl;
            same = false;
            break;
        }
    }
    return same;
};

BOOST_AUTO_TEST_CASE(test_eigen_sptrsv) {
    using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
    using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }
    const std::string filename  = (cwd / "data/mtx_tests/ErdosRenyi_2k_14k_A.mtx").string();
    
    SparseMatrixImp<int32_t> graph;

    SM_csr L_csr;
    bool matrix_load_success = Eigen::loadMarket(L_csr, filename);
    BOOST_CHECK(matrix_load_success);

    if (!matrix_load_success) {
        std::cerr << "Failed to read matrix from " << filename << std::endl;
        return;
    }

    std::cout << "Loaded matrix of size " << L_csr.rows() << " x " << L_csr.cols()
              << " with " << L_csr.nonZeros() << " non-zeros.\n";

    graph.setCSR(&L_csr);
    SM_csc L_csc{};
    L_csc = L_csr;
    graph.setCSC(&L_csc);

    BspArchitecture<SparseMatrixImp<int32_t>> architecture(16, 1, 500);
    BspInstance<SparseMatrixImp<int32_t>> instance(graph, architecture);
    GrowLocalAutoCores<SparseMatrixImp<int32_t>> scheduler;
    BspSchedule<SparseMatrixImp<int32_t>> schedule(instance);
    auto result = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(result, SUCCESS);
    BOOST_CHECK_EQUAL(&schedule.getInstance(), &instance);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    BspScheduleCS<SparseMatrixImp<int32_t>> schedule_cs(instance);
    auto result_cs = scheduler.computeScheduleCS(schedule_cs);

    /*
    for (const auto &node : instance.vertices()) {
        std::cout << "Vertex " << node << " children:" <<  std::endl;
        for (const auto &target : instance.getComputationalDag().children(node)) {
            std::cout << "target:" << target << std::endl;
        }
        std::cout << std::endl;
    }
    */
   
    BOOST_CHECK_EQUAL(result_cs, SUCCESS);
    BOOST_CHECK(schedule_cs.hasValidCommSchedule());

    //std::cout << "Scheduling Costs:" << schedule_cs.computeCosts() << std::endl;
    //std::cout << "lazy com Costs:" <<schedule_cs.compute_lazy_communication_costs() << std::endl;

    // Eigen L solve
    Eigen::VectorXd L_b_ref, L_x_ref; // Declare vectors
    auto n = L_csc.cols(); // Get the number of columns (assuming square matrix)
    L_x_ref.resize(n); // Resize solution vector
    L_b_ref.resize(n); // Resize RHS vector
    auto L_view = L_csc.triangularView<Eigen::Lower>();
    L_b_ref.setOnes();  // Initialize RHS vector with all ones
    L_x_ref.setZero();
    L_x_ref = L_view.solve(L_b_ref);

    // OSP no permutation setup
    Sptrsv<int32_t> sim{instance};
    sim.setup_csr_no_permutation(schedule_cs);


    //osp no permutation L_solve
    auto L_x_osp = L_x_ref;
    auto L_b_osp = L_b_ref;
    L_b_osp.setOnes();
    //L_x_osp.setZero();
    sim.x = &L_x_osp[0];
    sim.b = &L_b_osp[0];
    sim.lsolve_no_permutation();
    BOOST_CHECK(compare_vectors(L_x_ref,L_x_osp));

    // Comparisson with osp serial L solve
    // Eigen
    L_b_ref.setOnes();
    L_x_ref.setZero();
    L_x_ref = L_view.solve(L_b_ref);
    // OSP
    L_b_osp.setOnes();
    //L_x_osp.setZero();
    sim.lsolve_serial();
    BOOST_CHECK(compare_vectors(L_x_ref,L_x_osp));


    // INPLACE case eigen L solve vs osp L solve
    // Eigen
    L_b_ref.setConstant(0.1);
    L_x_ref.setConstant(0.1);
    L_x_ref = L_view.solve(L_b_ref);
    // OSP
    L_x_osp.setConstant(0.1);
    L_b_osp.setZero(); // this will not be used as x will take the values that already has instead of the b values
    sim.lsolve_no_permutation_in_place();
    BOOST_CHECK(compare_vectors(L_x_ref,L_x_osp));

    // Comparisson with osp serial in place L solve
    // Eigen
    L_b_ref.setConstant(0.1);
    L_x_ref.setConstant(0.1);
    L_x_ref = L_view.solve(L_b_ref);
    // OSP
    L_x_osp.setConstant(0.1);
    L_b_osp.setZero(); // this will not be used as x will take the values that already has instead of the b values
    sim.lsolve_serial_in_place();
    BOOST_CHECK(compare_vectors(L_x_ref,L_x_osp));

    // Upper Solve
    SM_csr U_csr = L_csc.transpose();
    SM_csc U_csc = U_csr;  // Convert to column-major
    Eigen::VectorXd U_b_ref(n), U_x_ref(n);
    Eigen::VectorXd U_b_osp(n), U_x_osp(n);
    // Eigen reference U solve
    U_b_ref.setOnes();
    U_x_ref.setZero();
    auto U_view = U_csc.triangularView<Eigen::Upper>();
    U_x_ref = U_view.solve(U_b_ref);
    // OSP U solve
    U_b_osp.setOnes();
    U_x_osp.setZero();
    sim.x = &U_x_osp[0];
    sim.b = &U_b_osp[0];
    sim.usolve_no_permutation();
    BOOST_CHECK(compare_vectors(U_x_ref, U_x_osp));

    // Comparisson with osp serial U solve
    // Eigen
    U_b_ref.setOnes();
    U_x_ref.setZero();
    U_x_ref = U_view.solve(U_b_ref);
    // OSP
    U_b_osp.setOnes();
    U_x_osp.setZero();
    sim.usolve_serial();
    BOOST_CHECK(compare_vectors(U_x_ref,U_x_osp));
    
    // INPLACE case eigen U solve vs osp U solve
    // Eigen
    U_b_ref.setConstant(0.1);
    U_x_ref.setConstant(0.1);
    U_x_ref = U_view.solve(U_b_ref);
    // OSP
    U_x_osp.setConstant(0.1);
    U_b_osp.setZero(); // this will not be used as x will take the values that already has instead of the b values
    sim.usolve_no_permutation_in_place();
    BOOST_CHECK(compare_vectors(U_x_ref,U_x_osp));

    // Comparisson with osp serial in place U solve
    // Eigen
    U_b_ref.setConstant(0.1);
    U_x_ref.setConstant(0.1);
    U_x_ref = U_view.solve(U_b_ref);
    // OSP
    U_x_osp.setConstant(0.1);
    U_b_osp.setZero(); // this will not be used as x will take the values that already has instead of the b values
    sim.usolve_serial_in_place();
    BOOST_CHECK(compare_vectors(U_x_ref,U_x_osp));


    // Lsolve in-place With PERMUTATION
    std::vector<size_t> perm = schedule_node_permuter_basic(schedule_cs, LOOP_PROCESSORS);
    sim.setup_csr_with_permutation (schedule_cs, perm);

    // Comparisson with osp serial in place L solve
    // Eigen
    L_b_ref.setConstant(0.1);
    L_x_ref.setConstant(0.1);
    L_x_ref = L_view.solve(L_b_ref);
    // OSP
    L_x_osp.setConstant(0.1);
    L_b_osp.setZero(); // this will not be used as x will take the values that already has instead of the b values
    sim.x = &L_x_osp[0];
    sim.b = &L_b_osp[0];
    //sim.permute_x_vector(perm);
    sim.lsolve_with_permutation_in_place();

    sim.permute_x_vector(perm);
    BOOST_CHECK(compare_vectors(L_x_ref,L_x_osp));

};



#endif
