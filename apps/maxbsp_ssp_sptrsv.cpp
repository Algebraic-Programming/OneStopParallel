/*
 * maxbsp_ssp_sptrsv.cpp
 * Demonstrates maxbsp scheduling with staleness=2, then runs SpTRSV with SSP kernel.
 */

#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "osp/auxiliary/sptrsv_simulator/sptrsv.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyVarianceSspScheduler.hpp"
#include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"
#include <chrono>

using namespace osp;

int main(int argc, char* argv[]) {
    // Accept matrix filename and iteration count as arguments
    std::string filename = "../data/mtx_tests/ErdosRenyi_2k_14k_A.mtx";
    int num_iterations = 1;
    if (argc > 1) {
        filename = argv[1];
    }
    if (argc > 2) {
        num_iterations = std::stoi(argv[2]);
    }

    // Load matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t> lCsr;
    bool matrixLoadSuccess = Eigen::loadMarket(lCsr, filename);
    if (!matrixLoadSuccess) {
        std::cerr << "Failed to read matrix from " << filename << std::endl;
        return 1;
    }
    std::cout << "Loaded matrix of size " << lCsr.rows() << " x " << lCsr.cols() << " with " << lCsr.nonZeros() << " non-zeros.\n";

    // Setup graph and architecture
    SparseMatrixImp<int32_t> graph;
    graph.SetCsr(&lCsr);
    Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t> lCsc = lCsr;
    graph.SetCsc(&lCsc);
    BspArchitecture<SparseMatrixImp<int32_t>> architecture(16, 1, 500); // 16 processors
    BspInstance<SparseMatrixImp<int32_t>> instance(graph, architecture);

    // Create SSP-aware schedule using GreedyVarianceSspScheduler (staleness=2)
    GreedyVarianceSspScheduler<SparseMatrixImp<int32_t>> ssp_scheduler;
    MaxBspSchedule<SparseMatrixImp<int32_t>> ssp_schedule(instance);
    ssp_scheduler.ComputeSchedule(ssp_schedule);

    // Setup SpTRSV kernel
    Sptrsv<int32_t> sptrsv_kernel(instance);
    sptrsv_kernel.SetupCsrNoPermutation(ssp_schedule);

    size_t n = static_cast<size_t>(lCsc.cols());

    // Benchmark SSP L-solve
    double ssp_total_time = 0.0;
    std::vector<double> ssp_result(n, 0.0);
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<double> x(n, 0.0);
        std::vector<double> b(n, 1.0);
        sptrsv_kernel.x_ = x.data();
        sptrsv_kernel.b_ = b.data();
        auto start = std::chrono::high_resolution_clock::now();
        sptrsv_kernel.SspLsolveStaleness2();
        auto end = std::chrono::high_resolution_clock::now();
        ssp_total_time += std::chrono::duration<double>(end - start).count();
        if (iter == 0) ssp_result = std::vector<double>(x.begin(), x.end());
    }
    double ssp_avg_time = ssp_total_time / num_iterations;

    // Benchmark serial L-solve
    double serial_total_time = 0.0;
    std::vector<double> serial_result(n, 0.0);
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<double> x_serial(n, 0.0);
        std::vector<double> b_serial(n, 1.0);
        sptrsv_kernel.x_ = x_serial.data();
        sptrsv_kernel.b_ = b_serial.data();
        auto start = std::chrono::high_resolution_clock::now();
        sptrsv_kernel.LsolveSerial();
        auto end = std::chrono::high_resolution_clock::now();
        serial_total_time += std::chrono::duration<double>(end - start).count();
        if (iter == 0) serial_result = std::vector<double>(x_serial.begin(), x_serial.end());
    }
    double serial_avg_time = serial_total_time / num_iterations;

    // Compare results
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(ssp_result[i] - serial_result[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max difference between SSP and serial L-solve: " << max_diff << std::endl;
    if (max_diff < 1e-10) {
        std::cout << "SSP L-solve matches serial L-solve!" << std::endl;
    } else {
        std::cout << "SSP L-solve does NOT match serial L-solve!" << std::endl;
    }
    std::cout << "Average SSP L-solve time (" << num_iterations << " runs): " << ssp_avg_time << " seconds" << std::endl;
    std::cout << "Average serial L-solve time (" << num_iterations << " runs): " << serial_avg_time << " seconds" << std::endl;
    std::cout << "MaxBSP with staleness=2 and SSP SpTRSV executed." << std::endl;
    return 0;
}
