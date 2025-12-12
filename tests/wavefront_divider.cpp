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

#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "osp/dag_divider/WavefrontComponentScheduler.hpp"
#include "osp/dag_divider/wavefront_divider/RecursiveWavefrontDivider.hpp"
#include "osp/dag_divider/wavefront_divider/ScanWavefrontDivider.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_utils.hpp"

using namespace osp;

std::vector<std::string> TestGraphsDot() { return {"data/dot/smpl_dot_graph_1.dot"}; }

std::vector<std::string> TinySpaaGraphs() {
    return {"data/spaa/tiny/instance_bicgstab.hdag",
            "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.hdag",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.hdag",
            "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag"};
}

template <typename GraphT>
bool CheckVertexMaps(const std::vector<std::vector<std::vector<vertex_idx_t<GraphT>>>> &maps, const GraphT &dag) {
    std::unordered_set<vertex_idx_t<GraphT>> allVertices;
    for (const auto &step : maps) {
        for (const auto &subgraph : step) {
            for (const auto &vertex : subgraph) {
                allVertices.insert(vertex);
            }
        }
    }

    return allVertices.size() == dag.NumVertices();
}

BOOST_AUTO_TEST_CASE(WavefrontComponentDivider) {
    std::vector<std::string> filenamesGraph = TestGraphsDot();

    const auto projectRoot = GetProjectRoot();

    using GraphT = computational_dag_edge_idx_vector_impl_def_t;

    for (auto &filenameGraph : filenamesGraph) {
        BspInstance<GraphT> instance;
        auto &graph = instance.getComputationalDag();

        auto statusGraph = file_reader::readComputationalDagDotFormat((projectRoot / filenameGraph).string(), graph);

        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filenameGraph << std::endl;
        }

        ScanWavefrontDivider<GraphT> wavefront;
        auto maps = wavefront.divide(graph);

        if (!maps.empty()) {
            BOOST_CHECK(CheckVertexMaps(maps, graph));
        }
    }
}

BOOST_AUTO_TEST_CASE(WavefrontComponentParallelismDivider) {
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();

    const auto projectRoot = GetProjectRoot();

    using GraphT = computational_dag_edge_idx_vector_impl_def_t;

    for (auto &filenameGraph : filenamesGraph) {
        BspInstance<GraphT> instance;
        auto &graph = instance.getComputationalDag();

        auto statusGraph = file_reader::readComputationalDagHyperdagFormatDB((projectRoot / filenameGraph).string(), graph);

        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        } else {
            std::cout << "File read:" << filenameGraph << std::endl;
        }

        ScanWavefrontDivider<GraphT> wavefront;
        wavefront.set_metric(SequenceMetric::AVAILABLE_PARALLELISM);
        wavefront.use_variance_splitter(1.0, 1.0, 1);

        auto maps = wavefront.divide(graph);

        if (!maps.empty()) {
            BOOST_CHECK(CheckVertexMaps(maps, graph));
        }
    }
}
