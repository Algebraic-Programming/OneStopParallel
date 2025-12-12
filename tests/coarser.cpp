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

#define BOOST_TEST_MODULE COARSER_TEST
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/CoarseAndSchedule.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/coarser/BspScheduleCoarser.hpp"
#include "osp/coarser/Sarkar/Sarkar.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"
#include "osp/coarser/SquashA/SquashA.hpp"
#include "osp/coarser/SquashA/SquashAMul.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/coarser/funnel/FunnelBfs.hpp"
#include "osp/coarser/hdagg/hdagg_coarser.hpp"
#include "osp/coarser/top_order/top_order_coarser.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph_edge_desc.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

using VertexType = VertexIdxT<computational_dag_edge_idx_vector_impl_def_t>;

bool CheckVertexMap(std::vector<std::vector<VertexType>> &map, std::size_t size) {
    std::unordered_set<VertexType> vertices;

    for (auto &v : map) {
        for (auto &v2 : v) {
            if (vertices.find(v2) != vertices.end()) {
                return false;
            }
            vertices.insert(v2);
        }
    }

    return vertices.size() == size;
}

template <typename ComputationalDag>
bool CheckVertexMapConstraints(std::vector<std::vector<VertexType>> &map,
                               ComputationalDag &dag,
                               VTypeT<ComputationalDag> sizeThreshold,
                               VMemwT<ComputationalDag> memoryThreshold,
                               VWorkwT<ComputationalDag> workThreshold,
                               VCommwT<ComputationalDag> communicationThreshold) {
    std::unordered_set<VertexType> vertices;

    for (auto &superNode : map) {
        VMemwT<ComputationalDag> memory = 0;
        VWorkwT<ComputationalDag> work = 0;
        VCommwT<ComputationalDag> communication = 0;

        if (superNode.size() > sizeThreshold) {
            return false;
        }

        if (superNode.size() == 0) {
            return false;
        }

        for (auto &v : superNode) {
            memory += dag.VertexMemWeight(v);
            work += dag.VertexWorkWeight(v);
            communication += dag.VertexCommWeight(v);

            if (dag.VertexType(v) != dag.VertexType(superNode[0])) {
                return false;
            }
        }

        if (memory > memoryThreshold || work > workThreshold || communication > communicationThreshold) {
            return false;
        }
    }
    return true;
}

BOOST_AUTO_TEST_CASE(CoarserHdaggTest) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
        nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));

        std::cout << std::endl << "Graph: " << nameGraph << std::endl;

        using GraphT = computational_dag_edge_idx_vector_impl_def_t;

        BspInstance<GraphT> instance;

        bool statusGraph = file_reader::readGraph((cwd / filenameGraph).string(), instance.GetComputationalDag());

        bool statusArchitecture
            = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

        if (!statusGraph || !statusArchitecture) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<GraphT> coarseInstance;
        coarseInstance.GetArchitecture() = instance.GetArchitecture();
        std::vector<std::vector<VertexType>> vertexMap;
        std::vector<VertexType> reverseVertexMap;

        hdagg_coarser<GraphT, GraphT> coarser;

        BOOST_CHECK_EQUAL(coarser.getCoarserName(), "hdagg_coarser");

        coarser.coarsenDag(instance.GetComputationalDag(), coarseInstance.GetComputationalDag(), reverseVertexMap);

        vertexMap = coarser_util::invert_vertex_contraction_map<GraphT, GraphT>(reverseVertexMap);

        BOOST_CHECK(CheckVertexMap(vertexMap, instance.GetComputationalDag().NumVertices()));

        GreedyBspScheduler<GraphT> scheduler;
        BspSchedule<GraphT> schedule(coarseInstance);

        const auto statusSched = scheduler.ComputeSchedule(schedule);

        BOOST_CHECK(statusSched == ReturnStatus::OSP_SUCCESS);
        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

        BspSchedule<GraphT> scheduleOut(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertexMap, scheduleOut), true);
        BOOST_CHECK(scheduleOut.SatisfiesPrecedenceConstraints());

        CoarseAndSchedule<GraphT, GraphT> coarseAndSchedule(coarser, scheduler);
        BspSchedule<GraphT> schedule2(instance);

        const auto status = coarseAndSchedule.ComputeSchedule(schedule2);
        BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
        BOOST_CHECK(schedule2.SatisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(CoarserHdaggTestDiffGraphImpl) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
        nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));

        std::cout << std::endl << "Graph: " << nameGraph << std::endl;

        using GraphT1 = computational_dag_edge_idx_vector_impl_def_t;
        using GraphT2 = computational_dag_vector_impl_def_t;

        BspInstance<GraphT1> instance;

        bool statusGraph = file_reader::readGraph((cwd / filenameGraph).string(), instance.GetComputationalDag());

        bool statusArchitecture
            = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

        if (!statusGraph || !statusArchitecture) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<GraphT2> coarseInstance;
        BspArchitecture<GraphT2> architectureT2(instance.GetArchitecture());
        coarseInstance.GetArchitecture() = architectureT2;
        std::vector<std::vector<VertexType>> vertexMap;
        std::vector<VertexType> reverseVertexMap;

        hdagg_coarser<GraphT1, GraphT2> coarser;

        coarser.coarsenDag(instance.GetComputationalDag(), coarseInstance.GetComputationalDag(), reverseVertexMap);

        vertexMap = coarser_util::invert_vertex_contraction_map<GraphT1, GraphT2>(reverseVertexMap);

        BOOST_CHECK(CheckVertexMap(vertexMap, instance.GetComputationalDag().NumVertices()));

        GreedyBspScheduler<GraphT2> scheduler;
        BspSchedule<GraphT2> schedule(coarseInstance);

        auto statusSched = scheduler.ComputeSchedule(schedule);

        BOOST_CHECK(statusSched == ReturnStatus::OSP_SUCCESS);
        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

        BspSchedule<GraphT1> scheduleOut(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertexMap, scheduleOut), true);
        BOOST_CHECK(scheduleOut.SatisfiesPrecedenceConstraints());

        CoarseAndSchedule<GraphT1, GraphT2> coarseAndSchedule(coarser, scheduler);
        BspSchedule<GraphT1> schedule2(instance);

        auto status = coarseAndSchedule.ComputeSchedule(schedule2);
        BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
        BOOST_CHECK(schedule2.SatisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(CoarserBspscheduleTest) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
        nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));

        std::cout << std::endl << "Graph: " << nameGraph << std::endl;

        using GraphT = computational_dag_edge_idx_vector_impl_def_t;

        BspInstance<GraphT> instance;

        bool statusGraph = file_reader::readGraph((cwd / filenameGraph).string(), instance.GetComputationalDag());

        bool statusArchitecture
            = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

        if (!statusGraph || !statusArchitecture) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<GraphT> coarseInstance;
        coarseInstance.GetArchitecture() = instance.GetArchitecture();
        std::vector<std::vector<VertexType>> vertexMap;
        std::vector<VertexType> reverseVertexMap;

        GreedyBspScheduler<GraphT> scheduler;
        BspSchedule<GraphT> scheduleOrig(instance);

        const auto statusSchedOrig = scheduler.ComputeSchedule(scheduleOrig);

        BOOST_CHECK(statusSchedOrig == ReturnStatus::OSP_SUCCESS);
        BOOST_CHECK(scheduleOrig.SatisfiesPrecedenceConstraints());

        BspScheduleCoarser<GraphT, GraphT> coarser(scheduleOrig);

        coarser.coarsenDag(instance.GetComputationalDag(), coarseInstance.GetComputationalDag(), reverseVertexMap);

        vertexMap = coarser_util::invert_vertex_contraction_map<GraphT, GraphT>(reverseVertexMap);

        BOOST_CHECK(CheckVertexMap(vertexMap, instance.GetComputationalDag().NumVertices()));

        BspSchedule<GraphT> schedule(coarseInstance);

        const auto statusSched = scheduler.ComputeSchedule(schedule);

        BOOST_CHECK(statusSched == ReturnStatus::OSP_SUCCESS);
        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

        BspSchedule<GraphT> scheduleOut(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertexMap, scheduleOut), true);
        BOOST_CHECK(scheduleOut.SatisfiesPrecedenceConstraints());

        CoarseAndSchedule<GraphT, GraphT> coarseAndSchedule(coarser, scheduler);
        BspSchedule<GraphT> schedule2(instance);

        const auto status = coarseAndSchedule.ComputeSchedule(schedule2);
        BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
        BOOST_CHECK(schedule2.SatisfiesPrecedenceConstraints());
    }
}

template <typename GraphT>
void TestCoarserSameGraph(Coarser<GraphT, GraphT> &coarser) {
    // BOOST_AUTO_TEST_CASE(coarser_bspschedule_test) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
        nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));

        std::cout << std::endl << "Graph: " << nameGraph << std::endl;

        BspInstance<GraphT> instance;

        bool statusGraph = file_reader::readGraph((cwd / filenameGraph).string(), instance.GetComputationalDag());

        bool statusArchitecture
            = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

        if (!statusGraph || !statusArchitecture) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<GraphT> coarseInstance;
        coarseInstance.GetArchitecture() = instance.GetArchitecture();
        std::vector<std::vector<VertexType>> vertexMap;
        std::vector<VertexType> reverseVertexMap;

        GreedyBspScheduler<GraphT> scheduler;

        bool coarseSuccess
            = coarser.coarsenDag(instance.GetComputationalDag(), coarseInstance.GetComputationalDag(), reverseVertexMap);
        BOOST_CHECK(coarseSuccess);

        vertexMap = coarser_util::invert_vertex_contraction_map<GraphT, GraphT>(reverseVertexMap);

        BOOST_CHECK(CheckVertexMap(vertexMap, instance.GetComputationalDag().NumVertices()));

        BspSchedule<GraphT> schedule(coarseInstance);

        const auto statusSched = scheduler.ComputeSchedule(schedule);

        BOOST_CHECK(statusSched == ReturnStatus::OSP_SUCCESS);
        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

        BspSchedule<GraphT> scheduleOut(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertexMap, scheduleOut), true);
        BOOST_CHECK(scheduleOut.SatisfiesPrecedenceConstraints());

        CoarseAndSchedule<GraphT, GraphT> coarseAndSchedule(coarser, scheduler);
        BspSchedule<GraphT> schedule2(instance);

        const auto status = coarseAndSchedule.ComputeSchedule(schedule2);
        BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
        BOOST_CHECK(schedule2.SatisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(CoarserFunndelBfsTest) {
    using GraphT = computational_dag_edge_idx_vector_impl_def_t;
    FunnelBfs<GraphT, GraphT> coarser;

    TestCoarserSameGraph<GraphT>(coarser);

    FunnelBfs<GraphT, GraphT>::FunnelBfs_parameters params{std::numeric_limits<VWorkwT<GraphT>>::max(),
                                                           std::numeric_limits<VMemwT<GraphT>>::max(),
                                                           std::numeric_limits<unsigned>::max(),
                                                           false,
                                                           true};

    FunnelBfs<GraphT, GraphT> coarserParams(params);

    TestCoarserSameGraph<GraphT>(coarserParams);

    params.max_depth = 2;
    FunnelBfs<GraphT, GraphT> coarserParams2(params);

    TestCoarserSameGraph<GraphT>(coarserParams2);
}

BOOST_AUTO_TEST_CASE(CoarserTopSortTest) {
    using GraphT = computational_dag_edge_idx_vector_impl_def_t;
    top_order_coarser<GraphT, GraphT, GetTopOrder> coarser;

    TestCoarserSameGraph<GraphT>(coarser);

    top_order_coarser<GraphT, GraphT, GetTopOrderMaxChildren> coarser2;

    TestCoarserSameGraph<GraphT>(coarser2);

    top_order_coarser<GraphT, GraphT, GetTopOrderGorder> coarser3;

    TestCoarserSameGraph<GraphT>(coarser3);
}

BOOST_AUTO_TEST_CASE(SquashATest) {
    using GraphT = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SquashAParams::Parameters params;
    params.mode = SquashAParams::Mode::EDGE_WEIGHT;
    params.use_structured_poset = false;

    SquashA<GraphT, GraphT> coarser(params);

    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SquashAParams::Mode::TRIANGLES;
    params.use_structured_poset = true;
    params.use_top_poset = true;
    coarser.setParams(params);

    TestCoarserSameGraph<GraphT>(coarser);

    params.use_top_poset = false;
    coarser.setParams(params);

    TestCoarserSameGraph<GraphT>(coarser);
}

BOOST_AUTO_TEST_CASE(CoarserSquashATestDiffGraphImplCsg) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
        nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));

        std::cout << std::endl << "Graph: " << nameGraph << std::endl;

        using GraphT1 = computational_dag_edge_idx_vector_impl_def_t;
        using GraphT2 = CSG;

        BspInstance<GraphT1> instance;

        bool statusGraph = file_reader::readGraph((cwd / filenameGraph).string(), instance.GetComputationalDag());

        bool statusArchitecture
            = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

        if (!statusGraph || !statusArchitecture) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<GraphT2> coarseInstance;
        BspArchitecture<GraphT2> architectureT2(instance.GetArchitecture());
        coarseInstance.GetArchitecture() = architectureT2;
        std::vector<std::vector<VertexType>> vertexMap;
        std::vector<VertexType> reverseVertexMap;

        SquashAParams::Parameters params;
        params.mode = SquashAParams::Mode::EDGE_WEIGHT;
        params.use_structured_poset = false;

        SquashA<GraphT1, GraphT2> coarser(params);

        coarser.coarsenDag(instance.GetComputationalDag(), coarseInstance.GetComputationalDag(), reverseVertexMap);

        vertexMap = coarser_util::invert_vertex_contraction_map<GraphT1, GraphT2>(reverseVertexMap);

        BOOST_CHECK(CheckVertexMap(vertexMap, instance.GetComputationalDag().NumVertices()));

        GreedyBspScheduler<GraphT2> scheduler;
        BspSchedule<GraphT2> schedule(coarseInstance);

        auto statusSched = scheduler.ComputeSchedule(schedule);

        BOOST_CHECK(statusSched == ReturnStatus::OSP_SUCCESS);
        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

        BspSchedule<GraphT1> scheduleOut(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertexMap, scheduleOut), true);
        BOOST_CHECK(scheduleOut.SatisfiesPrecedenceConstraints());

        CoarseAndSchedule<GraphT1, GraphT2> coarseAndSchedule(coarser, scheduler);
        BspSchedule<GraphT1> schedule2(instance);

        auto status = coarseAndSchedule.ComputeSchedule(schedule2);
        BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
        BOOST_CHECK(schedule2.SatisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(CoarserSquashATestDiffGraphImplCsge) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
        nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));

        std::cout << std::endl << "Graph: " << nameGraph << std::endl;

        using GraphT1 = computational_dag_edge_idx_vector_impl_def_t;
        using GraphT2 = CSGE;

        BspInstance<GraphT1> instance;

        bool statusGraph = file_reader::readGraph((cwd / filenameGraph).string(), instance.GetComputationalDag());

        bool statusArchitecture
            = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

        if (!statusGraph || !statusArchitecture) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        BspInstance<GraphT2> coarseInstance;
        BspArchitecture<GraphT2> architectureT2(instance.GetArchitecture());
        coarseInstance.GetArchitecture() = architectureT2;
        std::vector<std::vector<VertexType>> vertexMap;
        std::vector<VertexType> reverseVertexMap;

        SquashAParams::Parameters params;
        params.mode = SquashAParams::Mode::EDGE_WEIGHT;
        params.use_structured_poset = false;

        SquashA<GraphT1, GraphT2> coarser(params);

        coarser.coarsenDag(instance.GetComputationalDag(), coarseInstance.GetComputationalDag(), reverseVertexMap);

        vertexMap = coarser_util::invert_vertex_contraction_map<GraphT1, GraphT2>(reverseVertexMap);

        BOOST_CHECK(CheckVertexMap(vertexMap, instance.GetComputationalDag().NumVertices()));

        GreedyBspScheduler<GraphT2> scheduler;
        BspSchedule<GraphT2> schedule(coarseInstance);

        auto statusSched = scheduler.ComputeSchedule(schedule);

        BOOST_CHECK(statusSched == ReturnStatus::OSP_SUCCESS);
        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

        BspSchedule<GraphT1> scheduleOut(instance);

        BOOST_CHECK_EQUAL(coarser_util::pull_back_schedule(schedule, vertexMap, scheduleOut), true);
        BOOST_CHECK(scheduleOut.SatisfiesPrecedenceConstraints());

        CoarseAndSchedule<GraphT1, GraphT2> coarseAndSchedule(coarser, scheduler);
        BspSchedule<GraphT1> schedule2(instance);

        auto status = coarseAndSchedule.ComputeSchedule(schedule2);
        BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
        BOOST_CHECK(schedule2.SatisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(SarkarTest) {
    using GraphT = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SarkarParams::Parameters<VWorkwT<GraphT>> params;
    params.mode = SarkarParams::Mode::LINES;
    params.commCost = 100;
    params.useTopPoset = true;

    Sarkar<GraphT, GraphT> Coarser(params);

    TestCoarserSameGraph<GraphT>(coarser);

    params.useTopPoset = false;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::FAN_IN_FULL;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::FAN_IN_PARTIAL;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::FAN_OUT_FULL;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::FAN_OUT_PARTIAL;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::LEVEL_EVEN;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::LEVEL_ODD;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::FAN_IN_BUFFER;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::FAN_OUT_BUFFER;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);

    params.mode = SarkarParams::Mode::HOMOGENEOUS_BUFFER;
    coarser.setParameters(params);
    TestCoarserSameGraph<GraphT>(coarser);
}

BOOST_AUTO_TEST_CASE(SarkarMlTest) {
    using GraphT = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SarkarParams::MulParameters<VWorkwT<GraphT>> params;
    params.commCostVec = {100};

    SarkarMul<GraphT, GraphT> coarser;
    coarser.setParameters(params);

    TestCoarserSameGraph<GraphT>(coarser);
}

BOOST_AUTO_TEST_CASE(SarkarMlBufferMergeTest) {
    using GraphT = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SarkarParams::MulParameters<VWorkwT<GraphT>> params;
    params.commCostVec = {1, 2, 10, 50, 100};
    params.buffer_merge_mode = SarkarParams::BufferMergeMode::FULL;

    SarkarMul<GraphT, GraphT> coarser;
    coarser.setParameters(params);

    TestCoarserSameGraph<GraphT>(coarser);
}

BOOST_AUTO_TEST_CASE(SquashAmlTest) {
    using GraphT = computational_dag_edge_idx_vector_impl_def_t;
    // using graph_t = computational_dag_vector_impl_def_t;

    SquashAMul<GraphT, GraphT> coarser;

    TestCoarserSameGraph<GraphT>(coarser);
}
