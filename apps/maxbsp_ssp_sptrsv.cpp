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
    // Accept matrix filename and iteration count as arguments (threads via OMP_NUM_THREADS or optional arg)
    std::string filename = "../data/mtx_tests/ErdosRenyi_2k_14k_A.mtx";
    int num_iterations = 1;
    unsigned num_threads = 16U;
    if (argc > 1) {
        filename = argv[1];
    }
    if (argc > 2) {
        num_iterations = std::stoi(argv[2]);
    }
    if (const char *omp_env = std::getenv("OMP_NUM_THREADS")) {
        num_threads = static_cast<unsigned>(std::stoul(omp_env));
    } else if (argc > 3) {
        num_threads = static_cast<unsigned>(std::stoul(argv[3]));
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
    BspArchitecture<SparseMatrixImp<int32_t>> architecture(num_threads, 1, 500); // configurable processors
    BspInstance<SparseMatrixImp<int32_t>> instance(graph, architecture);

    // Create SSP-aware schedule using GreedyVarianceSspScheduler (staleness=2)
    GreedyVarianceSspScheduler<SparseMatrixImp<int32_t>> ssp_scheduler;
    MaxBspSchedule<SparseMatrixImp<int32_t>> ssp_schedule(instance);
    ssp_scheduler.ComputeSchedule(ssp_schedule);

    // Create a non-SSP schedule using GrowLocalAutoCores
    GrowLocalAutoCores<SparseMatrixImp<int32_t>> growlocal_scheduler;
    BspSchedule<SparseMatrixImp<int32_t>> growlocal_schedule(instance);
    growlocal_scheduler.ComputeSchedule(growlocal_schedule);

    // Setup SpTRSV kernel
    Sptrsv<int32_t> sptrsv_kernel(instance);

    size_t n = static_cast<size_t>(lCsc.cols());

    // Benchmark SSP L-solve
    double ssp_flat_total_time = 0.0;
    std::vector<double> ssp_flat_result(n, 0.0);
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<double> x(n, 0.0);
        std::vector<double> b(n, 1.0);
        sptrsv_kernel.SetupCsrNoPermutation(ssp_schedule);
        sptrsv_kernel.x_ = x.data();
        sptrsv_kernel.b_ = b.data();
        FlatCheckpointCounterBarrier barrier(num_threads);
        auto ops = Sptrsv<int32_t>::MakeBarrierOps(barrier);
        auto start = std::chrono::high_resolution_clock::now();
        sptrsv_kernel.SspLsolveStaleness2(ops);
        auto end = std::chrono::high_resolution_clock::now();
        ssp_flat_total_time += std::chrono::duration<double>(end - start).count();
        if (iter == 0) ssp_flat_result = std::vector<double>(x.begin(), x.end());
    }
    double ssp_flat_avg_time = ssp_flat_total_time / num_iterations;

    // Benchmark GrowLocalAutoCores schedule with non-SSP L-solve (no permutation)
    double growlocal_total_time = 0.0;
    std::vector<double> growlocal_result(n, 0.0);
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<double> x(n, 0.0);
        std::vector<double> b(n, 1.0);
        sptrsv_kernel.SetupCsrNoPermutation(growlocal_schedule);
        sptrsv_kernel.x_ = x.data();
        sptrsv_kernel.b_ = b.data();
        auto start = std::chrono::high_resolution_clock::now();
        sptrsv_kernel.LsolveNoPermutation();
        auto end = std::chrono::high_resolution_clock::now();
        growlocal_total_time += std::chrono::duration<double>(end - start).count();
        if (iter == 0) growlocal_result = std::vector<double>(x.begin(), x.end());
    }
    double growlocal_avg_time = growlocal_total_time / num_iterations;

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
    double max_diff_flat = 0.0;
    double frobNorm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(ssp_flat_result[i] - serial_result[i]);
        if (diff > max_diff_flat) max_diff_flat = diff;
        frobNorm += diff * diff;
    }
    frobNorm = std::sqrt(frobNorm);
    std::cout << "Frobenius norm of difference: " << frobNorm << std::endl;
    std::cout << "Max difference between SSP and serial L-solve: " << max_diff_flat << std::endl;
    if (frobNorm <= 1e-30 || max_diff_flat < 1e-10 * frobNorm) {
        std::cout << "SSP L-solve matches serial L-solve!" << std::endl;
    } else {
        std::cout << "SSP L-solve does NOT match serial L-solve!" << std::endl;
        std::cout << "Relative error: " << (max_diff_flat / frobNorm) << std::endl;
    }
    double max_diff_growlocal = 0.0;
    double frobNormGrowlocal = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(growlocal_result[i] - serial_result[i]);
        if (diff > max_diff_growlocal) max_diff_growlocal = diff;
        frobNormGrowlocal += diff * diff;
    }
    frobNormGrowlocal = std::sqrt(frobNormGrowlocal);
    std::cout << "Max difference between GrowLocalAutoCores and serial L-solve: " << max_diff_growlocal << std::endl;
    if (frobNormGrowlocal <= 1e-30 || max_diff_growlocal < 1e-10 * frobNormGrowlocal) {
        std::cout << "GrowLocalAutoCores L-solve matches serial L-solve!" << std::endl;
    } else {
        std::cout << "GrowLocalAutoCores L-solve does NOT match serial L-solve!" << std::endl;
        std::cout << "Relative error: " << (max_diff_growlocal / frobNormGrowlocal) << std::endl;
    }

    double max_diff_ssp_growlocal = 0.0;
    double frobNormSspGrowlocal = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(ssp_flat_result[i] - growlocal_result[i]);
        if (diff > max_diff_ssp_growlocal) max_diff_ssp_growlocal = diff;
        frobNormSspGrowlocal += diff * diff;
    }
    frobNormSspGrowlocal = std::sqrt(frobNormSspGrowlocal);
    std::cout << "Max difference between SSP and GrowLocalAutoCores L-solve: " << max_diff_ssp_growlocal
              << std::endl;
    if (frobNormSspGrowlocal <= 1e-30 || max_diff_ssp_growlocal < 1e-10 * frobNormSspGrowlocal) {
        std::cout << "SSP L-solve matches GrowLocalAutoCores L-solve!" << std::endl;
    } else {
        std::cout << "SSP L-solve does NOT match GrowLocalAutoCores L-solve!" << std::endl;
        std::cout << "Relative error: " << (max_diff_ssp_growlocal / frobNormSspGrowlocal) << std::endl;
    }

    std::cout << "Average SSP L-solve time (" << num_iterations << " runs): " << ssp_flat_avg_time
              << " seconds" << std::endl;
    std::cout << "Average GrowLocalAutoCores L-solve time (" << num_iterations << " runs): " << growlocal_avg_time
              << " seconds" << std::endl;
    std::cout << "Average serial L-solve time (" << num_iterations << " runs): " << serial_avg_time << " seconds" << std::endl;
    if (ssp_flat_avg_time > 0.0) {
        std::cout << "Speedup (serial/SSP): " << (serial_avg_time / ssp_flat_avg_time) << "x" << std::endl;
    }
    if (growlocal_avg_time > 0.0) {
        std::cout << "Speedup (serial/GrowLocalAutoCores): " << (serial_avg_time / growlocal_avg_time) << "x" << std::endl;
    }
    if (ssp_flat_avg_time > 0.0) {
        std::cout << "Speedup (GrowLocalAutoCores/SSP): " << (growlocal_avg_time / ssp_flat_avg_time) << "x" << std::endl;
    }
    std::cout << "MaxBSP staleness=2 SSP and GrowLocalAutoCores SpTRSV executed." << std::endl;
    return 0;
}
