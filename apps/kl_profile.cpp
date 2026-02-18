/*
Standalone profiling app for the KL local search on large SPAA graph instances.
Run with perf or vtune:
    perf record -g ./kl_profile
    perf report
*/

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/model/cost/TotalCommunicationCost.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

using graph = ComputationalDagEdgeIdxVectorImplDefIntT;

void AddMemWeights(graph &dag) {
    int memWeight = 1;
    int commWeight = 7;

    for (const auto &v : dag.Vertices()) {
        dag.SetVertexWorkWeight(v, static_cast<VMemwT<graph>>(memWeight++ % 10 + 2));
        dag.SetVertexMemWeight(v, static_cast<VMemwT<graph>>(memWeight++ % 10 + 2));
        dag.SetVertexCommWeight(v, static_cast<VCommwT<graph>>(commWeight++ % 10 + 2));
    }
}

std::filesystem::path FindProjectRoot() {
    std::filesystem::path cwd = std::filesystem::current_path();
    while (!cwd.empty() && cwd.filename() != "OneStopParallel") {
        cwd = cwd.parent_path();
    }
    if (cwd.empty()) {
        std::cerr << "Error: could not find OneStopParallel project root from " << std::filesystem::current_path() << std::endl;
        std::exit(1);
    }
    return cwd;
}

void RunOnInstance(const std::filesystem::path &graphPath) {
    std::cout << "\n=== " << graphPath.filename() << " ===" << std::endl;

    BspInstance<graph> instance;
    bool ok = file_reader::ReadComputationalDagHyperdagFormatDB(graphPath.string(), instance.GetComputationalDag());
    if (!ok) {
        std::cerr << "Failed to read: " << graphPath << std::endl;
        return;
    }

    instance.GetArchitecture().SetSynchronisationCosts(500);
    instance.GetArchitecture().SetCommunicationCosts(5);
    instance.GetArchitecture().SetNumberOfProcessors(4);

    std::vector<std::vector<int>> send_cost = {
        {0, 1, 4, 4},
        {1, 0, 4, 4},
        {4, 4, 0, 1},
        {4, 4, 1, 0}
    };
    instance.GetArchitecture().SetSendCosts(send_cost);

    AddMemWeights(instance.GetComputationalDag());

    // Compute initial greedy schedule
    GreedyBspScheduler<graph> greedy;
    BspSchedule<graph> schedule(instance);
    greedy.ComputeSchedule(schedule);
    schedule.UpdateNumberOfSupersteps();

    std::cout << "Initial schedule: cost=" << TotalCommunicationCost<graph>()(schedule)
              << " supersteps=" << schedule.NumberOfSupersteps() << std::endl;

    // Run KL local search (this is what we want to profile)
    KlTotalCommImprover<graph, NoLocalSearchMemoryConstraint, 1, true> kl(42);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto status = kl.ImproveSchedule(schedule);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "KL finished in " << ms << " ms, status=" << (status == ReturnStatus::OSP_SUCCESS ? "SUCCESS" : "BEST_FOUND")
              << std::endl;
    std::cout << "Final schedule: cost=" << TotalCommunicationCost<graph>()(schedule)
              << " supersteps=" << schedule.NumberOfSupersteps() << std::endl;
}

int main(int argc, char *argv[]) {
    std::filesystem::path root = FindProjectRoot();

    std::vector<std::string> defaultGraphs = {"data/spaa/large/instance_exp_N50_K12_nzP0d15.hdag",
                                              "data/spaa/large/instance_CG_N24_K22_nzP0d2.hdag",
                                              "data/spaa/large/instance_kNN_N45_K15_nzP0d16.hdag",
                                              "data/spaa/large/instance_spmv_N120_nzP0d18.hdag"};

    std::vector<std::string> graphs;
    if (argc > 1) {
        // Allow passing specific graph files as arguments
        for (int i = 1; i < argc; ++i) {
            graphs.emplace_back(argv[i]);
        }
    } else {
        graphs = defaultGraphs;
    }

    auto total_t0 = std::chrono::high_resolution_clock::now();

    for (const auto &g : graphs) {
        std::filesystem::path graphPath = root / g;
        if (!std::filesystem::exists(graphPath)) {
            std::cerr << "Graph file not found: " << graphPath << std::endl;
            continue;
        }
        RunOnInstance(graphPath);
    }

    auto total_t1 = std::chrono::high_resolution_clock::now();
    auto total_s = std::chrono::duration_cast<std::chrono::seconds>(total_t1 - total_t0).count();
    std::cout << "\nTotal time: " << total_s << " s" << std::endl;

    return 0;
}
