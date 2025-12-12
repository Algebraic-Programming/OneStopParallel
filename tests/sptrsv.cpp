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

#    define BOOST_TEST_MODULE SPTRSV

#    include "osp/auxiliary/sptrsv_simulator/sptrsv.hpp"

#    include <Eigen/Sparse>
#    include <boost/test/unit_test.hpp>
#    include <filesystem>
#    include <iostream>
#    include <unsupported/Eigen/SparseExtra>
#    include <vector>

#    include "osp/auxiliary/sptrsv_simulator/ScheduleNodePermuter.hpp"
#    include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#    include "osp/graph_algorithms/directed_graph_path_util.hpp"
#    include "osp/graph_algorithms/directed_graph_util.hpp"
#    include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"

using namespace osp;

bool CompareVectors(Eigen::VectorXd &v1, Eigen::VectorXd &v2) {
    std::cout << std::fixed;
    std::cout << std::setprecision(15);

    assert(v1.size() == v2.size());
    bool same = true;
    const double epsilon = 1e-10;
    for (long long int i = 0; i < v1.size(); ++i) {
        // std::cout << "Ind: " << i << ": | " << v1[i] << " - " << v2[i] << " | = " << abs(v1[i]-v2[i]) << "\n";
        if (std::abs(v1[i] - v2[i]) / (std::abs(v1[i]) + std::abs(v2[i]) + epsilon) > epsilon) {
            std::cout << "We have differences in the matrix in position: " << i << std::endl;
            std::cout << v1[i] << " , " << v2[i] << std::endl;
            same = false;
            break;
        }
    }
    return same;
}

BOOST_AUTO_TEST_CASE(TestEigenSptrsv) {
    using SmCsr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
    using SmCsc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }
    const std::string filename = (cwd / "data/mtx_tests/ErdosRenyi_2k_14k_A.mtx").string();

    SparseMatrixImp<int32_t> graph;

    SmCsr lCsr;
    bool matrixLoadSuccess = Eigen::loadMarket(lCsr, filename);
    BOOST_CHECK(matrixLoadSuccess);

    if (!matrixLoadSuccess) {
        std::cerr << "Failed to read matrix from " << filename << std::endl;
        return;
    }

    std::cout << "Loaded matrix of size " << lCsr.rows() << " x " << lCsr.cols() << " with " << lCsr.nonZeros() << " non-zeros.\n";

    graph.setCSR(&lCsr);
    SmCsc lCsc{};
    lCsc = lCsr;
    graph.setCSC(&lCsc);

    BspArchitecture<SparseMatrixImp<int32_t>> architecture(16, 1, 500);
    BspInstance<SparseMatrixImp<int32_t>> instance(graph, architecture);
    GrowLocalAutoCores<SparseMatrixImp<int32_t>> scheduler;
    BspSchedule<SparseMatrixImp<int32_t>> schedule(instance);
    auto result = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(result, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    BspScheduleCS<SparseMatrixImp<int32_t>> scheduleCs(instance);
    auto resultCs = scheduler.computeScheduleCS(scheduleCs);

    /*
    for (const auto &node : instance.vertices()) {
        std::cout << "Vertex " << node << " children:" <<  std::endl;
        for (const auto &target : instance.getComputationalDag().children(node)) {
            std::cout << "target:" << target << std::endl;
        }
        std::cout << std::endl;
    }
    */

    BOOST_CHECK_EQUAL(resultCs, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK(scheduleCs.hasValidCommSchedule());

    // std::cout << "Scheduling Costs:" << schedule_cs.computeCosts() << std::endl;
    // std::cout << "lazy com Costs:" <<schedule_cs.compute_lazy_communication_costs() << std::endl;

    // Eigen L solve
    Eigen::VectorXd lBRef, lXRef;    // Declare vectors
    auto n = lCsc.cols();            // Get the number of columns (assuming square matrix)
    lXRef.resize(n);                 // Resize solution vector
    lBRef.resize(n);                 // Resize RHS vector
    auto lView = lCsc.triangularView<Eigen::Lower>();
    lBRef.setOnes();    // Initialize RHS vector with all ones
    lXRef.setZero();
    lXRef = lView.solve(lBRef);

    // OSP no permutation setup
    Sptrsv<int32_t> sim{instance};
    sim.setup_csr_no_permutation(scheduleCs);

    // osp no permutation L_solve
    auto lXOsp = lXRef;
    auto lBOsp = lBRef;
    lBOsp.setOnes();
    // L_x_osp.setZero();
    sim.x = &lXOsp[0];
    sim.b = &lBOsp[0];
    sim.lsolve_no_permutation();
    BOOST_CHECK(CompareVectors(lXRef, lXOsp));

    // Comparisson with osp serial L solve
    // Eigen
    lBRef.setOnes();
    lXRef.setZero();
    lXRef = lView.solve(lBRef);
    // OSP
    lBOsp.setOnes();
    // L_x_osp.setZero();
    sim.lsolve_serial();
    BOOST_CHECK(CompareVectors(lXRef, lXOsp));

    // INPLACE case eigen L solve vs osp L solve
    // Eigen
    lBRef.setConstant(0.1);
    lXRef.setConstant(0.1);
    lXRef = lView.solve(lBRef);
    // OSP
    lXOsp.setConstant(0.1);
    lBOsp.setZero();    // this will not be used as x will take the values that already has instead of the b values
    sim.lsolve_no_permutation_in_place();
    BOOST_CHECK(CompareVectors(lXRef, lXOsp));

    // Comparisson with osp serial in place L solve
    // Eigen
    lBRef.setConstant(0.1);
    lXRef.setConstant(0.1);
    lXRef = lView.solve(lBRef);
    // OSP
    lXOsp.setConstant(0.1);
    lBOsp.setZero();    // this will not be used as x will take the values that already has instead of the b values
    sim.lsolve_serial_in_place();
    BOOST_CHECK(CompareVectors(lXRef, lXOsp));

    // Upper Solve
    SmCsr uCsr = lCsc.transpose();
    SmCsc uCsc = uCsr;    // Convert to column-major
    Eigen::VectorXd uBRef(n), uXRef(n);
    Eigen::VectorXd uBOsp(n), uXOsp(n);
    // Eigen reference U solve
    uBRef.setOnes();
    uXRef.setZero();
    auto uView = uCsc.triangularView<Eigen::Upper>();
    uXRef = uView.solve(uBRef);
    // OSP U solve
    uBOsp.setOnes();
    uXOsp.setZero();
    sim.x = &uXOsp[0];
    sim.b = &uBOsp[0];
    sim.usolve_no_permutation();
    BOOST_CHECK(CompareVectors(uXRef, uXOsp));

    // Comparisson with osp serial U solve
    // Eigen
    uBRef.setOnes();
    uXRef.setZero();
    uXRef = uView.solve(uBRef);
    // OSP
    uBOsp.setOnes();
    uXOsp.setZero();
    sim.usolve_serial();
    BOOST_CHECK(CompareVectors(uXRef, uXOsp));

    // INPLACE case eigen U solve vs osp U solve
    // Eigen
    uBRef.setConstant(0.1);
    uXRef.setConstant(0.1);
    uXRef = uView.solve(uBRef);
    // OSP
    uXOsp.setConstant(0.1);
    uBOsp.setZero();    // this will not be used as x will take the values that already has instead of the b values
    sim.usolve_no_permutation_in_place();
    BOOST_CHECK(CompareVectors(uXRef, uXOsp));

    // Comparisson with osp serial in place U solve
    // Eigen
    uBRef.setConstant(0.1);
    uXRef.setConstant(0.1);
    uXRef = uView.solve(uBRef);
    // OSP
    uXOsp.setConstant(0.1);
    uBOsp.setZero();    // this will not be used as x will take the values that already has instead of the b values
    sim.usolve_serial_in_place();
    BOOST_CHECK(CompareVectors(uXRef, uXOsp));

    // Lsolve in-place With PERMUTATION
    std::vector<size_t> perm = schedule_node_permuter_basic(scheduleCs, LOOP_PROCESSORS);
    sim.setup_csr_with_permutation(scheduleCs, perm);

    // Comparisson with osp serial in place L solve
    // Eigen
    lBRef.setConstant(0.1);
    lXRef.setConstant(0.1);
    lXRef = lView.solve(lBRef);
    // OSP
    lXOsp.setConstant(0.1);
    lBOsp.setZero();    // this will not be used as x will take the values that already has instead of the b values
    sim.x = &lXOsp[0];
    sim.b = &lBOsp[0];
    // sim.permute_x_vector(perm);
    sim.lsolve_with_permutation_in_place();

    sim.permute_x_vector(perm);
    BOOST_CHECK(CompareVectors(lXRef, lXOsp));
}

#endif
