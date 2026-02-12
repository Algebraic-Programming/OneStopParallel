/*
 * maxbsp_ssp_sptrsv.cpp
 * Demonstrates maxbsp scheduling with staleness=2, then runs SpTRSV with SSP kernel.
 */

#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

#include "osp/auxiliary/sptrsv_simulator/sptrsv.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyVarianceSspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalMaxBsp.hpp"
#include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"

using namespace osp;

#define EPSILON 1e-20

double L2NormalisedDiff(const std::vector<double> &v, const std::vector<double> &w) {
    assert(v.size() == w.size());
    double l2diff = 0.0;
    double frobNorm = 0.0;
    for (std::size_t i = 0U; i < v.size(); ++i) {
        const double absdiff = std::abs(v[i] - w[i]);
        l2diff += absdiff * absdiff;

        const double vAbs = std::abs(v[i]);
        const double wAbs = std::abs(w[i]);

        frobNorm += ((vAbs * vAbs) + (wAbs * wAbs)) / 2.0;
    }
    l2diff = std::sqrt(l2diff);
    frobNorm = std::sqrt(frobNorm);
    const double ratio = l2diff / (frobNorm + EPSILON);
    return ratio;
}

double LInftyNormalisedDiff(const std::vector<double> &v, const std::vector<double> &w) {
    double diff = 0.0;
    for (std::size_t i = 0U; i < v.size(); ++i) {
        const double absdiff = std::abs(v[i] - w[i]);
        const double vAbs = std::abs(v[i]);
        const double wAbs = std::abs(w[i]);

        diff = std::max(diff, 2 * absdiff / (vAbs + wAbs + EPSILON));
    }
    return diff;
}

int main(int argc, char *argv[]) {
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
    BspArchitecture<SparseMatrixImp<int32_t>> architecture(num_threads, 1, 500);    // configurable processors
    BspInstance<SparseMatrixImp<int32_t>> instance(graph, architecture);

    // Create SSP-aware schedule using GreedyVarianceSspScheduler (staleness=2)
    GreedyVarianceSspScheduler<SparseMatrixImp<int32_t>> ssp_var_scheduler;
    MaxBspSchedule<SparseMatrixImp<int32_t>> ssp_var_schedule(instance);
    ssp_var_scheduler.ComputeSchedule(ssp_var_schedule);

    // Create SSP-aware schedule using GrowLocalMaxBsp (staleness=2)
    GrowLocalSSP<SparseMatrixImp<int32_t>> ssp_gl_scheduler;
    MaxBspSchedule<SparseMatrixImp<int32_t>> ssp_gl_schedule(instance);
    ssp_gl_scheduler.ComputeSchedule(ssp_gl_schedule);

    // Create a non-SSP schedule using GrowLocalAutoCores
    GrowLocalAutoCores<SparseMatrixImp<int32_t>> growlocal_scheduler;
    BspSchedule<SparseMatrixImp<int32_t>> growlocal_schedule(instance);
    growlocal_scheduler.ComputeSchedule(growlocal_schedule);

    // Setup SpTRSV kernel
    Sptrsv<int32_t> sptrsv_kernel(instance);

    size_t n = static_cast<size_t>(lCsc.cols());

    // Benchmark SSP Variance L-solve
    double ssp_var_flat_total_time = 0.0;
    std::vector<double> ssp_var_flat_result(n, 0.0);
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<double> x(n, 0.0);
        std::vector<double> b(n, 1.0);
        sptrsv_kernel.SetupCsrNoPermutation(ssp_var_schedule);
        sptrsv_kernel.x_ = x.data();
        sptrsv_kernel.b_ = b.data();
        FlatCheckpointCounterBarrier barrier(num_threads);
        auto ops = Sptrsv<int32_t>::MakeBarrierOps(barrier);
        auto start = std::chrono::high_resolution_clock::now();
        sptrsv_kernel.SspLsolveStaleness2(ops);
        auto end = std::chrono::high_resolution_clock::now();
        ssp_var_flat_total_time += std::chrono::duration<double>(end - start).count();
        if (iter == 0) {
            ssp_var_flat_result = std::vector<double>(x.begin(), x.end());
        }
    }
    double ssp_var_flat_avg_time = ssp_var_flat_total_time / num_iterations;

    // Benchmark SSP GrowLocal L-solve
    double ssp_gl_flat_total_time = 0.0;
    std::vector<double> ssp_gl_flat_result(n, 0.0);
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<double> x(n, 0.0);
        std::vector<double> b(n, 1.0);
        sptrsv_kernel.SetupCsrNoPermutation(ssp_gl_schedule);
        sptrsv_kernel.x_ = x.data();
        sptrsv_kernel.b_ = b.data();
        FlatCheckpointCounterBarrier barrier(num_threads);
        auto ops = Sptrsv<int32_t>::MakeBarrierOps(barrier);
        auto start = std::chrono::high_resolution_clock::now();
        sptrsv_kernel.SspLsolveStaleness2(ops);
        auto end = std::chrono::high_resolution_clock::now();
        ssp_gl_flat_total_time += std::chrono::duration<double>(end - start).count();
        if (iter == 0) {
            ssp_gl_flat_result = std::vector<double>(x.begin(), x.end());
        }
    }
    double ssp_gl_flat_avg_time = ssp_gl_flat_total_time / num_iterations;

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
        if (iter == 0) {
            growlocal_result = std::vector<double>(x.begin(), x.end());
        }
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
        if (iter == 0) {
            serial_result = std::vector<double>(x_serial.begin(), x_serial.end());
        }
    }
    double serial_avg_time = serial_total_time / num_iterations;

    // Compare results
    const double varDiff = LInftyNormalisedDiff(ssp_var_flat_result, serial_result);

    std::cout << "Max relative difference between SSP Variance and serial L-solve: " << varDiff << std::endl;
    if (varDiff < EPSILON) {
        std::cout << "SSP Variance L-solve matches serial L-solve!" << std::endl;
    } else {
        std::cout << "SSP Variance L-solve does NOT match serial L-solve!" << std::endl;
    }

    const double GLSSPDiff = LInftyNormalisedDiff(ssp_gl_flat_result, serial_result);

    std::cout << "Max relative difference between SSP GrowLocal and serial L-solve: " << GLSSPDiff << std::endl;
    if (GLSSPDiff < EPSILON) {
        std::cout << "SSP GrowLocal L-solve matches serial L-solve!" << std::endl;
    } else {
        std::cout << "SSP GrowLocal L-solve does NOT match serial L-solve!" << std::endl;
    }

    const double GLPDiff = LInftyNormalisedDiff(growlocal_result, serial_result);

    std::cout << "Max relative difference between GrowLocal and serial L-solve: " << GLPDiff << std::endl;
    if (GLPDiff < EPSILON) {
        std::cout << "GrowLocal L-solve matches serial L-solve!" << std::endl;
    } else {
        std::cout << "GrowLocal L-solve does NOT match serial L-solve!" << std::endl;
    }

    std::cout << "Average SSP Variance L-solve time (" << num_iterations << " runs): " << ssp_var_flat_avg_time << " seconds"
              << std::endl;
    std::cout << "Average SSP GrowLocal L-solve time (" << num_iterations << " runs): " << ssp_gl_flat_avg_time << " seconds"
              << std::endl;
    std::cout << "Average GrowLocalAutoCores L-solve time (" << num_iterations << " runs): " << growlocal_avg_time << " seconds"
              << std::endl;
    std::cout << "Average serial L-solve time (" << num_iterations << " runs): " << serial_avg_time << " seconds" << std::endl << std::endl;

    if (ssp_var_flat_avg_time > 0.0) {
        std::cout << "Speedup (serial/SSP Var): " << (serial_avg_time / ssp_var_flat_avg_time) << "x" << std::endl;
    }
    if (ssp_gl_flat_avg_time > 0.0) {
        std::cout << "Speedup (serial/SSP GL): " << (serial_avg_time / ssp_gl_flat_avg_time) << "x" << std::endl;
    }
    if (growlocal_avg_time > 0.0) {
        std::cout << "Speedup (serial/GrowLocalAutoCores): " << (serial_avg_time / growlocal_avg_time) << "x" << std::endl;
    }
    if (ssp_var_flat_avg_time > 0.0) {
        std::cout << "Speedup (GrowLocalAutoCores/SSP Var): " << (growlocal_avg_time / ssp_var_flat_avg_time) << "x" << std::endl;
    }
    if (ssp_gl_flat_avg_time > 0.0) {
        std::cout << "Speedup (GrowLocalAutoCores/SSP GL): " << (growlocal_avg_time / ssp_gl_flat_avg_time) << "x" << std::endl;
    }

    return 0;
}
