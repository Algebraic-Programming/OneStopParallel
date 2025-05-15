#define BOOST_TEST_MODULE BSP_MEM_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "coarser/StepByStep/StepByStepCoarser.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "io/hdag_graph_file_reader.hpp"

#include "graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(StepByStepCoarser_test) {

    using graph = boost_graph;
    StepByStepCoarser<boost_graph> test;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    graph DAG;

    bool status = file_reader::readComputationalDagHyperdagFormat(
        (cwd / "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag").string(), DAG);

    BOOST_CHECK(status);

    StepByStepCoarser<graph> coarser;

    coarser.setTargetNumberOfNodes(static_cast<unsigned>(DAG.num_vertices())/2);

    graph coarsened_dag1, coarsened_dag2;
    std::vector<std::vector<vertex_idx_t<graph>>> old_vertex_ids;
    std::vector<vertex_idx_t<graph>> new_vertex_id;

    coarser.coarseDag(DAG, coarsened_dag1, old_vertex_ids, new_vertex_id);

    coarser.coarsenForPebbling(DAG, coarsened_dag2, old_vertex_ids, new_vertex_id);

};