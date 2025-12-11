/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#define BOOST_TEST_MODULE kl
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_base.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

template <typename Graph_t>
void add_mem_weights(Graph_t &dag) {
    int mem_weight = 1;
    int comm_weight = 1;

    for (const auto &v : dag.vertices()) {
        dag.set_vertex_mem_weight(v, static_cast<v_memw_t<Graph_t>>(mem_weight++ % 3 + 1));
        dag.set_vertex_comm_weight(v, static_cast<v_commw_t<Graph_t>>(comm_weight++ % 3 + 1));
    }
}

BOOST_AUTO_TEST_CASE(kl_local_memconst) {
    std::vector<std::string> filenames_graph = test_graphs();

    using graph = computational_dag_edge_idx_vector_impl_def_int_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<graph, local_memory_constraint<graph>> test_scheduler;

    for (auto &filename_graph : filenames_graph) {
        std::cout << filename_graph << std::endl;
        BspInstance<graph> instance;

        bool status_graph
            = file_reader::readComputationalDagHyperdagFormatDB((cwd / filename_graph).string(), instance.getComputationalDag());
        instance.getArchitecture().setSynchronisationCosts(10);
        instance.getArchitecture().setCommunicationCosts(5);
        instance.getArchitecture().setNumberOfProcessors(4);
        instance.getArchitecture().setMemoryConstraintType(MEMORY_CONSTRAINT_TYPE::LOCAL);
        instance.getArchitecture().setSynchronisationCosts(0);

        const std::vector<int> bounds_to_test = {10, 20};

        add_mem_weights(instance.getComputationalDag());

        if (!status_graph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        for (const auto &bound : bounds_to_test) {
            instance.getArchitecture().setMemoryBound(bound);

            BspSchedule<graph> schedule(instance);
            const auto result = test_scheduler.computeSchedule(schedule);

            BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
            BOOST_CHECK(schedule.satisfiesMemoryConstraints());

            kl_total_comm_improver_local_mem_constr<graph> kl;

            auto status = kl.improveSchedule(schedule);

            BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
            BOOST_CHECK(schedule.satisfiesMemoryConstraints());
        }
    }
}
