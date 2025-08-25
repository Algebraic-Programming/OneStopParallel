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

#define BOOST_TEST_MODULE wavefront_divider
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <iostream>

#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "osp/dag_divider/wavefront_divider/ScanWavefrontDivider.hpp"
#include "osp/dag_divider/wavefront_divider/RecursiveWavefrontDivider.hpp"
#include "osp/dag_divider/WavefrontComponentScheduler.hpp"
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"

#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"


using namespace osp;

std::vector<std::string> test_graphs_dot() { return {"data/dot/smpl_dot_graph_1.dot"}; }

std::vector<std::string> tiny_spaa_graphs() {
    return {
        "data/spaa/tiny/instance_bicgstab.hdag", "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.hdag",
                 "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.hdag",
                 "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag"
        //         "data/spaa/tiny/instance_exp_N4_K2_nzP0d5.hdag",
        //         "data/spaa/tiny/instance_exp_N5_K3_nzP0d4.hdag",
        //         "data/spaa/tiny/instance_exp_N6_K4_nzP0d25.hdag",
        //         "data/spaa/tiny/instance_k-means.hdag",
        //         "data/spaa/tiny/instance_k-NN_3_gyro_m.hdag",
        //         "data/spaa/tiny/instance_kNN_N4_K3_nzP0d5.hdag",
        //         "data/spaa/tiny/instance_kNN_N5_K3_nzP0d3.hdag",
        //         "data/spaa/tiny/instance_kNN_N6_K4_nzP0d2.hdag",
        //         "data/spaa/tiny/instance_pregel.hdag",
        //         "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag",
        //         "data/spaa/tiny/instance_spmv_N7_nzP0d35.hdag",
        //         "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag",
    };
}

template<typename Graph_t>
bool check_vertex_maps(const std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> &maps, const Graph_t &dag) {

    std::unordered_set<vertex_idx_t<Graph_t>> all_vertices;
    for (const auto &step : maps) {
        for (const auto &subgraph : step) {

            for (const auto &vertex : subgraph)
                all_vertices.insert(vertex);
        }
    }

    return all_vertices.size() == dag.num_vertices();
}

BOOST_AUTO_TEST_CASE(wavefront_component_divider) {

    std::vector<std::string> filenames_graph = test_graphs_dot();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    using graph_t = computational_dag_edge_idx_vector_impl_def_t;

    for (auto &filename_graph : filenames_graph) {

        BspInstance<graph_t> instance;
        auto &graph = instance.getComputationalDag();

        auto status_graph = file_reader::readComputationalDagDotFormat((cwd / filename_graph).string(), graph);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filename_graph << std::endl;
        }

        ScanWavefrontDivider<graph_t> wavefront;       
        auto maps = wavefront.divide(graph);

        if (!maps.empty()) {

            BOOST_CHECK(check_vertex_maps(maps, graph));
        }
    }
}

BOOST_AUTO_TEST_CASE(wavefront_component_parallelism_divider) {

    std::vector<std::string> filenames_graph = tiny_spaa_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    using graph_t = computational_dag_edge_idx_vector_impl_def_t;

    for (auto &filename_graph : filenames_graph) {

        BspInstance<graph_t> instance;
        auto &graph = instance.getComputationalDag();

        auto status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(), graph);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filename_graph << std::endl;
        }

        ScanWavefrontDivider<graph_t> wavefront;
        wavefront.set_metric(SequenceMetric::AVAILABLE_PARALLELISM);
        wavefront.use_variance_splitter(1.0,1.0,1);

        auto maps = wavefront.divide(graph);

        if (!maps.empty()) {

            BOOST_CHECK(check_vertex_maps(maps, graph));
        }
    }
}
