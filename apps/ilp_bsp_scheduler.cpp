

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "auxiliary/misc.hpp"
#include "bsp/scheduler/IlpSchedulers/CoptFullScheduler.hpp"
#include "graph_algorithms/directed_graph_path_util.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"
#include "io/DotFileWriter.hpp"
#include "io/arch_file_reader.hpp"
#include "io/bsp_schedule_file_writer.hpp"
#include "io/dot_graph_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"
#include "io/mtx_graph_file_reader.hpp"

using namespace osp;

using ComputationalDag = boost_graph_int_t;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <machine_file> <max_number_step> <optional:recomp>"
                  << std::endl;
        return 1;
    }

    std::string filename_graph = argv[1];
    std::string name_graph = filename_graph.substr(0, filename_graph.rfind("."));

    std::cout << name_graph << std::endl;

    std::string filename_machine = argv[2];
    std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
    name_machine = name_machine.substr(0, name_machine.rfind("."));

    int step_int = std::stoi(argv[3]);
    if (step_int < 1) {
        std::cerr << "Argument max_number_step must be a positive integer: " << step_int << std::endl;
        return 1;
    }

    bool recomp = false;

    if (argc > 4 && std::string(argv[4]) == "recomp") {
        recomp = true;
    } else if (argc > 4) {
        std::cerr << "Unknown argument: " << argv[4] << ". Expected 'recomp' for recomputation." << std::endl;
        return 1;
    }

    unsigned steps = static_cast<unsigned>(step_int);

    BspInstance<ComputationalDag> instance;
    ComputationalDag &graph = instance.getComputationalDag();
    bool status_graph = false;

    if (filename_graph.substr(filename_graph.rfind(".") + 1) == "hdag") {

        status_graph = file_reader::readComputationalDagHyperdagFormat(filename_graph, graph);

    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {

        status_graph = file_reader::readComputationalDagMartixMarketFormat(filename_graph, graph);

    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {

        status_graph = file_reader::readComputationalDagDotFormat(filename_graph, graph);

    } else {
        std::cout << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                  << " ...assuming hyperDag format." << std::endl;
        status_graph = file_reader::readComputationalDagHyperdagFormat(filename_graph, graph);
    }

    bool status_arch = file_reader::readBspArchitecture(filename_machine, instance.getArchitecture());

    if (!status_graph || !status_arch) {

        std::cout << "Reading files failed." << std::endl;
        return 1;
    }

    // for (const auto &vertex : graph.vertices()) {

    //     graph.set_vertex_work_weight(vertex, graph.vertex_work_weight(vertex) * 80);
    // }

    CoptFullScheduler<ComputationalDag> scheduler;
    scheduler.setMaxNumberOfSupersteps(steps);

    if (recomp) {

        BspScheduleRecomp<ComputationalDag> schedule(instance);

        auto status_schedule = scheduler.computeScheduleRecomp(schedule);

        if (status_schedule == SUCCESS || status_schedule == BEST_FOUND) {

            DotFileWriter dot_writer;
            dot_writer.write_schedule_recomp(name_graph + "_" + name_machine + "_maxS_" + std::to_string(steps) + "_" +
                                                 scheduler.getScheduleName() + "_recomp_schedule.dot",
                                             schedule);

            std::cout << "Recomp Schedule computed with costs: " << schedule.computeCosts() << std::endl;

        } else {
            std::cout << "Computing schedule failed." << std::endl;
            return 1;
        }

    } else {

        BspSchedule<ComputationalDag> schedule(instance);

        auto status_schedule = scheduler.computeSchedule(schedule);

        if (status_schedule == SUCCESS || status_schedule == BEST_FOUND) {

            DotFileWriter dot_writer;
            dot_writer.write_schedule(name_graph + "_" + name_machine + "_maxS_" + std::to_string(steps) + "_" +
                                          scheduler.getScheduleName() + "_schedule.dot",
                                      schedule);

            std::cout << "Schedule computed with costs: " << schedule.computeCosts() << std::endl;

        } else {
            std::cout << "Computing schedule failed." << std::endl;
            return 1;
        }
    }
    return 0;
}