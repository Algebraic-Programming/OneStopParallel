/*
 * maxbsp_ssp_sptrsv.cpp
 * Benchmark for SpTRSV using:
 *   - variance_ssp
 *   - growlocal_ssp
 *   - growlocal
 *   - eigen_serial
 *
 * Outputs per-iteration runtime rows to CSV:
 * graph,Algorithm,processors,time to compute schedule,schedule supersteps,
 * schedule synchronization costs,staleness,runtime
 */

#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
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

namespace {

constexpr double EPSILON = 1e-12;
constexpr unsigned kDefaultStaleness = 2U;
constexpr int defaultSynchronisationCosts = 500;

constexpr int preMeasureIterations = 2;

enum class Algorithm {
    VarianceSsp,
    GrowLocalSsp,
    GrowLocal,
    Serial
};

struct Args {
    std::string inputPath;
    std::string outputCsv = "sptrsv_benchmark.csv";
    int iterations = 100;
    unsigned processors = 16U;
    std::set<Algorithm> algorithms;
};

struct CsvRow {
    std::string graph;
    std::string algorithm;
    unsigned processors;
    double scheduleTimeSeconds;
    unsigned supersteps;
    int SyncCosts;
    unsigned staleness;
    double runtimeSeconds;
    bool correctness;
};

struct SummaryKey {
    std::string graph;
    std::string algorithm;
    unsigned processors;
    unsigned staleness;

    bool operator<(const SummaryKey &other) const {
        if (graph != other.graph) {
            return graph < other.graph;
        }
        if (algorithm != other.algorithm) {
            return algorithm < other.algorithm;
        }
        if (processors != other.processors) {
            return processors < other.processors;
        }
        return staleness < other.staleness;
    }
};

struct SummaryAgg {
    double scheduleTimeSeconds = 0.0;
    unsigned supersteps = 0U;
    int SyncCosts = 0;
    double sumLogRuntime = 0.0;
    std::size_t samples = 0U;
    bool correctness = false;
};

std::string CsvEscape(const std::string &s) {
    if (s.find(',') == std::string::npos && s.find('"') == std::string::npos && s.find('\n') == std::string::npos
        && s.find('\r') == std::string::npos) {
        return s;
    }
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out += "\"";
    return out;
}

double LInftyNormalisedDiff(const std::vector<double> &v, const std::vector<double> &w) {
    double diff = 0.0;
    for (std::size_t i = 0U; i < v.size(); ++i) {
        const double absdiff = std::abs(v[i] - w[i]);
        const double vAbs = std::abs(v[i]);
        const double wAbs = std::abs(w[i]);
        diff = std::max(diff, 2.0 * absdiff / (vAbs + wAbs + EPSILON));
    }
    return diff;
}

void PrintUsage(const char *prog) {
    std::cout << "Usage:\n"
              << "  " << prog
              << " --input <file_or_directory> [--output <csv>] [--iterations <n>] [--processors <p>]\n"
              << "      [--variance-ssp] [--growlocal-ssp] [--growlocal] [--eigen-serial] [--all]\n\n"
              << "Examples:\n"
              << "  " << prog << " --input ../data/mtx_tests/ErdosRenyi_2k_14k_A.mtx --all\n"
              << "  " << prog
              << " --input ../data/mtx_tests --output bench.csv --iterations 100 --processors 16 --variance-ssp --growlocal-ssp --growlocal\n";
}

bool ParseArgs(int argc, char *argv[], Args &args) {
    if (const char *ompEnv = std::getenv("OMP_NUM_THREADS")) {
        args.processors = static_cast<unsigned>(std::stoul(ompEnv));
    }

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];

        const bool needsValue
            = (flag == "--input" || flag == "--output" || flag == "--iterations" || flag == "--processors");
        if (needsValue && i + 1 >= argc) {
            std::cerr << "Missing value for " << flag << "\n";
            return false;
        }

        if (flag == "--input") {
            args.inputPath = argv[++i];
        } else if (flag == "--output") {
            args.outputCsv = argv[++i];
        } else if (flag == "--iterations") {
            args.iterations = std::stoi(argv[++i]);
        } else if (flag == "--processors") {
            args.processors = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (flag == "--variance-ssp") {
            args.algorithms.insert(Algorithm::VarianceSsp);
        } else if (flag == "--growlocal-ssp") {
            args.algorithms.insert(Algorithm::GrowLocalSsp);
        } else if (flag == "--growlocal") {
            args.algorithms.insert(Algorithm::GrowLocal);
        } else if (flag == "--eigen-serial") {
            args.algorithms.insert(Algorithm::Serial);
        } else if (flag == "--all") {
            args.algorithms = {Algorithm::VarianceSsp, Algorithm::GrowLocalSsp, Algorithm::GrowLocal, Algorithm::Serial};
        } else if (flag == "--help" || flag == "-h") {
            PrintUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown option: " << flag << "\n";
            return false;
        }
    }

    if (args.inputPath.empty()) {
        std::cerr << "--input is required\n";
        return false;
    }
    if (args.iterations <= 0) {
        std::cerr << "--iterations must be > 0\n";
        return false;
    }
    if (args.processors == 0U) {
        std::cerr << "--processors must be > 0\n";
        return false;
    }
    if (args.algorithms.empty()) {
        std::cerr << "No algorithm selected. Use --all or explicit flags.\n";
        return false;
    }

    return true;
}

std::vector<std::filesystem::path> CollectInputGraphs(const std::string &inputPath) {
    std::vector<std::filesystem::path> inputs;
    std::filesystem::path p(inputPath);

    while (std::filesystem::exists(p) && std::filesystem::is_symlink(p)) {
        p = std::filesystem::read_symlink(p);
    }

    if (!std::filesystem::exists(p)) {
        throw std::runtime_error("Input path does not exist: " + inputPath);
    }

    if (std::filesystem::is_regular_file(p)) {
        if (p.extension() == ".mtx") {
            inputs.push_back(p);
        }
    } else if (std::filesystem::is_directory(p)) {
        for (const auto &entry : std::filesystem::recursive_directory_iterator(p)) {
            auto entryPath = entry.path();
            while (std::filesystem::exists(entryPath) && std::filesystem::is_symlink(entryPath)) {
                entryPath = std::filesystem::read_symlink(entryPath);
            }

            if (!std::filesystem::is_regular_file(entryPath)) {
                continue;
            }
            if (entryPath.extension() == ".mtx") {
                inputs.push_back(entryPath);
            }
        }
    }

    std::sort(inputs.begin(), inputs.end());
    return inputs;
}

void EnsureCsvHeader(std::ofstream &csv) {
    csv << "Graph,Algorithm,Processors,ScheduleTimeSeconds,ScheduleSupersteps,SynchronizationCosts,Staleness,RuntimeSeconds,Correctness\n";
}

void EnsureSummaryCsvHeader(std::ofstream &csv) {
    csv << "Graph,Algorithm,Processors,ScheduleTimeSeconds,ScheduleSupersteps,SynchronizationCosts,Staleness,"
           "RuntimeSamples,RuntimeGeometricMeanSeconds,Correctness\n";
}

void WriteCsvRow(std::ofstream &csv, const CsvRow &row) {
    csv << CsvEscape(row.graph) << "," << row.algorithm << "," << row.processors << "," << row.scheduleTimeSeconds << ","
    << row.supersteps << "," << row.SyncCosts << "," << row.staleness << "," << row.runtimeSeconds << "," << row.correctness << "\n";
}

std::string BuildSummaryCsvPath(const std::string &detailPath) {
    const std::filesystem::path p(detailPath);
    const std::string stem = p.stem().string();
    const std::string ext = p.has_extension() ? p.extension().string() : std::string(".csv");
    const std::filesystem::path summary = p.parent_path() / (stem + "_summary" + ext);
    return summary.string();
}

std::string FormatExperimentStartTimestampForFilename() {
    const std::time_t now = std::time(nullptr);
    std::tm localTm{};
#ifdef _WIN32
    localtime_s(&localTm, &now);
#else
    localtime_r(&now, &localTm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&localTm, "%d-%m-%Y_%H%M");
    return oss.str();
}

std::string BuildTimestampedCsvPath(const std::string &basePath, const std::string &timestamp) {
    const std::filesystem::path p(basePath);
    const std::string stem = p.stem().string();
    const std::string ext = p.has_extension() ? p.extension().string() : std::string(".csv");
    const std::filesystem::path out = p.parent_path() / (stem + "_" + timestamp + ext);
    return out.string();
}

int ComputeSyncCosts(const BspInstance<SparseMatrixImp<int32_t>> &instance) {
    return instance.GetArchitecture().SynchronisationCosts();
}

}    // namespace

int main(int argc, char *argv[]) {
    const std::string experimentStart = FormatExperimentStartTimestampForFilename();

    Args args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::vector<std::filesystem::path> graphFiles;
    try {
        graphFiles = CollectInputGraphs(args.inputPath);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (graphFiles.empty()) {
        std::cerr << "No .mtx files found at input path: " << args.inputPath << std::endl;
        return 1;
    }

    const std::string detailCsvPath = BuildTimestampedCsvPath(args.outputCsv, experimentStart);
    std::ofstream csv(detailCsvPath, std::ios::out | std::ios::trunc);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV output: " << detailCsvPath << std::endl;
        return 1;
    }
    EnsureCsvHeader(csv);

    const std::string summaryCsvPath = BuildSummaryCsvPath(detailCsvPath);
    std::ofstream summaryCsv(summaryCsvPath, std::ios::out | std::ios::trunc);
    if (!summaryCsv.is_open()) {
        std::cerr << "Failed to open summary CSV output: " << summaryCsvPath << std::endl;
        return 1;
    }
    EnsureSummaryCsvHeader(summaryCsv);

    std::cout << "Running benchmark on " << graphFiles.size() << " graph(s), iterations=" << args.iterations
              << ", processors=" << args.processors << std::endl;
    std::cout << "Experiment id timestamp: " << experimentStart << std::endl;

    std::vector<CsvRow> bufferedRows;
    bufferedRows.reserve(graphFiles.size() * args.algorithms.size() * static_cast<std::size_t>(args.iterations));
    typename std::vector<CsvRow>::difference_type writtenEntries = 0U;

    for (const auto &graphPath : graphFiles) {
        const std::string graphName = graphPath.filename().string();

        Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t> lCsr;
        if (!Eigen::loadMarket(lCsr, graphPath.string())) {
            std::cerr << "Failed to load matrix: " << graphPath << std::endl;
            continue;
        }

        Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t> lCsc = lCsr;

        SparseMatrixImp<int32_t> graph;
        graph.SetCsr(&lCsr);
        graph.SetCsc(&lCsc);

        BspArchitecture<SparseMatrixImp<int32_t>> architecture(args.processors, 1, defaultSynchronisationCosts);
        BspInstance<SparseMatrixImp<int32_t>> instance(graph, architecture);

        Sptrsv<int32_t> sptrsv(instance);
        const std::size_t n = static_cast<std::size_t>(lCsr.cols());

        std::vector<double> serialRefX(n, 0.0);
        std::vector<double> serialB(n, 1.0);
        sptrsv.x_ = serialRefX.data();
        sptrsv.b_ = serialB.data();
        sptrsv.LsolveSerial();

        std::cout << "Graph: " << graphName << " (" << lCsr.rows() << "x" << lCsr.cols() << ", nnz=" << lCsr.nonZeros() << ")\n";

        if (args.algorithms.count(Algorithm::VarianceSsp) > 0U) {
            GreedyVarianceSspScheduler<SparseMatrixImp<int32_t>> scheduler;
            MaxBspSchedule<SparseMatrixImp<int32_t>> schedule(instance);

            const auto t0 = std::chrono::high_resolution_clock::now();
            scheduler.ComputeSspSchedule(schedule, kDefaultStaleness);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double scheduleTime = std::chrono::duration<double>(t1 - t0).count();

            sptrsv.SetupCsrNoPermutation(schedule);
            const unsigned supersteps = schedule.NumberOfSupersteps();
            const int syncCosts = ComputeSyncCosts(instance);

            bool correct = false;
            for (int iter = 0; iter < args.iterations + preMeasureIterations; ++iter) {
                std::vector<double> x(n, 0.0);
                std::vector<double> b(n, 1.0);
                sptrsv.x_ = x.data();
                sptrsv.b_ = b.data();

                const auto s = std::chrono::high_resolution_clock::now();
                sptrsv.SspLsolveStaleness<kDefaultStaleness>();
                const auto e = std::chrono::high_resolution_clock::now();
                const double runtime = std::chrono::duration<double>(e - s).count();

                if (iter == 0) {
                    const double diff = LInftyNormalisedDiff(x, serialRefX);
                    correct = (diff < EPSILON);
                    std::cout << "  Variance_SSP first-run max relative diff vs serial: " << diff << std::endl;
                }

                if (iter >= preMeasureIterations) {
                    bufferedRows.emplace_back(CsvRow{graphName,
                                                     "Variance_SSP",
                                                     args.processors,
                                                     scheduleTime,
                                                     supersteps,
                                                     syncCosts,
                                                     kDefaultStaleness,
                                                     runtime,
                                                     correct});
                }
            }

            for (auto it = std::next(bufferedRows.cbegin(), writtenEntries); it != bufferedRows.cend(); ++it) {
                WriteCsvRow(csv, *it);
                ++writtenEntries;
            }
        }

        if (args.algorithms.count(Algorithm::GrowLocalSsp) > 0U) {
            GrowLocalSSP<SparseMatrixImp<int32_t>, kDefaultStaleness> scheduler;
            MaxBspSchedule<SparseMatrixImp<int32_t>> schedule(instance);

            const auto t0 = std::chrono::high_resolution_clock::now();
            scheduler.ComputeSchedule(schedule);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double scheduleTime = std::chrono::duration<double>(t1 - t0).count();

            sptrsv.SetupCsrNoPermutation(schedule);
            const unsigned supersteps = schedule.NumberOfSupersteps();
            const int syncCosts = ComputeSyncCosts(instance);

            bool correct = false;
            for (int iter = 0; iter < args.iterations + preMeasureIterations; ++iter) {
                std::vector<double> x(n, 0.0);
                std::vector<double> b(n, 1.0);
                sptrsv.x_ = x.data();
                sptrsv.b_ = b.data();

                const auto s = std::chrono::high_resolution_clock::now();
                sptrsv.SspLsolveStaleness<kDefaultStaleness>();
                const auto e = std::chrono::high_resolution_clock::now();
                const double runtime = std::chrono::duration<double>(e - s).count();

                if (iter == 0) {
                    const double diff = LInftyNormalisedDiff(x, serialRefX);
                    correct = (diff < EPSILON);
                    std::cout << "  Growlocal_SSP first-run max relative diff vs serial: " << diff << std::endl;
                }

                if (iter >= preMeasureIterations) {
                    bufferedRows.emplace_back(CsvRow{graphName,
                                                     "Growlocal_SSP",
                                                     args.processors,
                                                     scheduleTime,
                                                     supersteps,
                                                     syncCosts,
                                                     kDefaultStaleness,
                                                     runtime,
                                                     correct});
                }
            }

            for (auto it = std::next(bufferedRows.cbegin(), writtenEntries); it != bufferedRows.cend(); ++it) {
                WriteCsvRow(csv, *it);
                ++writtenEntries;
            }
        }

        if (args.algorithms.count(Algorithm::GrowLocal) > 0U) {
            GrowLocalAutoCores<SparseMatrixImp<int32_t>> scheduler;
            BspSchedule<SparseMatrixImp<int32_t>> schedule(instance);

            const auto t0 = std::chrono::high_resolution_clock::now();
            scheduler.ComputeSchedule(schedule);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double scheduleTime = std::chrono::duration<double>(t1 - t0).count();

            sptrsv.SetupCsrNoPermutation(schedule);
            const unsigned supersteps = schedule.NumberOfSupersteps();
            const int syncCosts = ComputeSyncCosts(instance);

            bool correct;
            for (int iter = 0; iter < args.iterations + preMeasureIterations; ++iter) {
                std::vector<double> x(n, 0.0);
                std::vector<double> b(n, 1.0);
                sptrsv.x_ = x.data();
                sptrsv.b_ = b.data();

                const auto s = std::chrono::high_resolution_clock::now();
                sptrsv.LsolveNoPermutation();
                const auto e = std::chrono::high_resolution_clock::now();
                const double runtime = std::chrono::duration<double>(e - s).count();

                if (iter == 0) {
                    const double diff = LInftyNormalisedDiff(x, serialRefX);
                    correct = (diff < EPSILON);
                    std::cout << "  Growlocal first-run max relative diff vs serial: " << diff << std::endl;
                }

                if (iter >= preMeasureIterations) {
                    bufferedRows.emplace_back(CsvRow{graphName,
                                                     "Growlocal",
                                                     args.processors,
                                                     scheduleTime,
                                                     supersteps,
                                                     syncCosts,
                                                     1U,
                                                     runtime,
                                                     correct});
                }
            }

            for (auto it = std::next(bufferedRows.cbegin(), writtenEntries); it != bufferedRows.cend(); ++it) {
                WriteCsvRow(csv, *it);
                ++writtenEntries;
            }
        }

        if (args.algorithms.count(Algorithm::Serial) > 0U) {
            for (int iter = 0; iter < args.iterations + preMeasureIterations; ++iter) {
                std::vector<double> x(n, 0.0);
                std::vector<double> b(n, 1.0);
                sptrsv.x_ = x.data();
                sptrsv.b_ = b.data();

                const auto s = std::chrono::high_resolution_clock::now();
                sptrsv.LsolveSerial();
                const auto e = std::chrono::high_resolution_clock::now();
                const double runtime = std::chrono::duration<double>(e - s).count();

                if (iter >= preMeasureIterations) {
                    bufferedRows.emplace_back(CsvRow{graphName,
                                                     "Serial",
                                                     1U,
                                                     0.0,
                                                     1U,
                                                     0,
                                                     1U,
                                                     runtime,
                                                     true});
                }
            }

            for (auto it = std::next(bufferedRows.cbegin(), writtenEntries); it != bufferedRows.cend(); ++it) {
                WriteCsvRow(csv, *it);
                ++writtenEntries;
            }
        }
    }

    std::map<SummaryKey, SummaryAgg> summary;
    constexpr double kMinRuntime = 1e-15;
    for (const CsvRow &row : bufferedRows) {
        SummaryKey key{row.graph, row.algorithm, row.processors, row.staleness};
        SummaryAgg &agg = summary[key];
        if (agg.samples == 0U) {
            agg.scheduleTimeSeconds = row.scheduleTimeSeconds;
            agg.supersteps = row.supersteps;
            agg.SyncCosts = row.SyncCosts;
            agg.correctness = row.correctness;
        }
        agg.sumLogRuntime += std::log(std::max(row.runtimeSeconds, kMinRuntime));
        ++agg.samples;
    }

    for (const auto &[key, agg] : summary) {
        const double geomean = std::exp(agg.sumLogRuntime / static_cast<double>(agg.samples));
        summaryCsv << CsvEscape(key.graph) << "," << key.algorithm << "," << key.processors << "," << agg.scheduleTimeSeconds
               << "," << agg.supersteps << "," << agg.SyncCosts << "," << key.staleness
                   << "," << agg.samples << "," << geomean << "," << agg.correctness << "\n";
    }

    std::cout << "Benchmark complete. CSV written to: " << detailCsvPath << std::endl;
    std::cout << "Summary CSV written to: " << summaryCsvPath << std::endl;
    return 0;
}
