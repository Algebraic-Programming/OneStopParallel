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

#ifdef EIGEN_FOUND

#define BOOST_TEST_MODULE SPTRSV_SSP
#include <boost/test/unit_test.hpp>

#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <filesystem>

#include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/auxiliary/sptrsv_simulator/sptrsv.hpp"

using namespace osp;

static constexpr double EPSILON = 1e-6;

template <typename Graph_t>
void validate_schedule_order(const BspScheduleCS<Graph_t>& schedule,
                             const BspInstance<Graph_t>& inst) {

    const size_t V = inst.numberOfVertices();
    std::vector<size_t> step_of(V, std::numeric_limits<size_t>::max());

    // 1) Extract step for each vertex via PUBLIC API
    for (size_t v = 0; v < V; ++v) {
        step_of[v] = schedule.assignedSuperstep(v);   // ✔️ official API
    }

    // 2) Validate parent before child
    const auto& dag = inst.getComputationalDag();
    for (size_t v = 0; v < V; ++v) {
        for (auto p : dag.parents(v)) {
            BOOST_CHECK_MESSAGE(
                step_of[p] <= step_of[v],
                "ERROR: parent " << p 
                << " (step " << step_of[p] << ") "
                << "appears AFTER child " << v
                << " (step " << step_of[v] << ")!"
            );
        }
    }
}


BOOST_AUTO_TEST_CASE(test_sptrsv_ssp_solver) {
    using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
    using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;

    // LOAD MATRIX
    std::filesystem::path cwd = std::filesystem::current_path();
    while (!cwd.empty() && cwd.filename() != "OneStopParallel")
        cwd = cwd.parent_path();

    SM_csr L_csr;
    std::string filename = (cwd / "data/mtx_tests/ErdosRenyi_2k_14k_A.mtx").string();
    BOOST_REQUIRE(Eigen::loadMarket(L_csr, filename));

    // BUILD INSTANCE + SCHEDULE
    SM_csc L_csc = L_csr;
    SparseMatrixImp<int32_t> graph;
    graph.setCSR(&L_csr);
    graph.setCSC(&L_csc);

    BspArchitecture<SparseMatrixImp<int32_t>> arch(16, 1, 500);
    BspInstance<SparseMatrixImp<int32_t>> instance(graph, arch);

    GrowLocalAutoCores<SparseMatrixImp<int32_t>> scheduler;
    BspScheduleCS<SparseMatrixImp<int32_t>> schedule(instance);
    BOOST_REQUIRE_EQUAL(scheduler.computeScheduleCS(schedule), RETURN_STATUS::OSP_SUCCESS);

    // 1) VALIDATE ORDER
    validate_schedule_order(schedule, instance);

    // 2) RUN OUR SSP SOLVER
    Sptrsv<int32_t> sim(instance);
    sim.setup_csr_no_permutation(schedule);

    // FILL CSR DATA
    const auto *Lptr = instance.getComputationalDag().getCSR();
    sim.val.assign(Lptr->valuePtr(), Lptr->valuePtr() + Lptr->nonZeros());
    sim.col_idx.assign(Lptr->innerIndexPtr(), Lptr->innerIndexPtr() + Lptr->nonZeros());
    sim.row_ptr.assign(Lptr->outerIndexPtr(),
                       Lptr->outerIndexPtr() + instance.numberOfVertices() + 1);

    // PREPARE X, B
    const auto n = instance.numberOfVertices();
    const Eigen::Index n_eig = static_cast<Eigen::Index>(n);
    Eigen::VectorXd b_ref = Eigen::VectorXd::Ones(n_eig);
    Eigen::VectorXd x_osp = Eigen::VectorXd::Zero(n_eig);
    sim.x = x_osp.data();
    sim.b = b_ref.data();

    // 3) RUN KERNEL
    sim.simulate_ssp_sptrsv_no_permutation();

    // 4) CHECK RESULT
    Eigen::VectorXd res = L_csc * x_osp - b_ref;
    double rel_error = res.norm() / b_ref.norm();
    std::cout << "SSP error = " << rel_error << "\n";
    BOOST_CHECK_MESSAGE(rel_error < EPSILON, "SSP FAILED: rel_error=" << rel_error);
}


#endif // EIGEN_FOUND
