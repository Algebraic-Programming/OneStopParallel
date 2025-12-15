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

#define BOOST_TEST_MODULE AbstractWavefrontSchedulerTest
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/AbstractWavefrontScheduler.hpp"
#include "osp/dag_divider/IsomorphicWavefrontComponentScheduler.hpp"
#include "osp/dag_divider/WavefrontComponentScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using GraphT = osp::ComputationalDagEdgeIdxVectorImplDefT;

template <typename GraphT, typename ConstrGraphT>
class ConcreteWavefrontScheduler : public osp::AbstractWavefrontScheduler<GraphT, ConstrGraphT> {
  public:
    ConcreteWavefrontScheduler(osp::IDagDivider<GraphT> &div, osp::Scheduler<ConstrGraphT> &sched)
        : osp::AbstractWavefrontScheduler<GraphT, ConstrGraphT>(div, sched) {}

    // Expose the protected method for testing with the new signature
    bool TestDistributeProcessors(unsigned totalProcessors,
                                  const std::vector<double> &workWeights,
                                  std::vector<unsigned> &allocation) const {
        return this->DistributeProcessors(totalProcessors, workWeights, allocation);
    }

    // Dummy implementation for the pure virtual method
    osp::ReturnStatus ComputeSchedule(osp::BspSchedule<GraphT> &) override { return osp::ReturnStatus::OSP_SUCCESS; }

    std::string GetScheduleName() const override { return "ConcreteScheduler"; }
};

// Mock dependencies for the test
struct MockDivider : public osp::IDagDivider<GraphT> {
    std::vector<std::vector<std::vector<GraphT::VertexIdx>>> divide(const GraphT &) override { return {}; }
};

struct MockScheduler : public osp::Scheduler<GraphT> {
    osp::ReturnStatus ComputeSchedule(osp::BspSchedule<GraphT> &) override { return osp::ReturnStatus::OSP_SUCCESS; }

    std::string GetScheduleName() const override { return "Mock"; }
};

BOOST_AUTO_TEST_SUITE(abstract_wavefront_scheduler_test_suite)

BOOST_AUTO_TEST_CASE(DistributeProcessorsTest) {
    MockDivider mockDivider;
    MockScheduler mockScheduler;
    ConcreteWavefrontScheduler<GraphT, GraphT> scheduler(mockDivider, mockScheduler);

    std::vector<unsigned> allocation;
    bool starvationHit;

    // Test 1: Proportional distribution with anti-starvation (Abundance)
    std::vector<double> work1 = {100.0, 200.0, 700.0};
    starvationHit = scheduler.TestDistributeProcessors(10, work1, allocation);
    std::vector<unsigned> expected1 = {1, 2, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected1.begin(), expected1.end());
    BOOST_CHECK(!starvationHit);

    // Test 2: Proportional with remainders and anti-starvation (Abundance)
    std::vector<double> work2 = {10.0, 10.0, 10.0, 70.0};
    starvationHit = scheduler.TestDistributeProcessors(10, work2, allocation);
    std::vector<unsigned> expected2 = {1, 1, 1, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected2.begin(), expected2.end());
    BOOST_CHECK(!starvationHit);

    // Test 3: Scarcity case (fewer processors than components)
    std::vector<double> work3 = {50.0, 100.0, 20.0, 80.0};
    starvationHit = scheduler.TestDistributeProcessors(2, work3, allocation);
    std::vector<unsigned> expected3 = {0, 1, 0, 1};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected3.begin(), expected3.end());
    BOOST_CHECK(starvationHit);

    // Test 4: More processors than components, with remainders (Abundance)
    std::vector<double> work4 = {10, 90};
    starvationHit = scheduler.TestDistributeProcessors(12, work4, allocation);
    std::vector<unsigned> expected4 = {1, 11};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected4.begin(), expected4.end());
    BOOST_CHECK(!starvationHit);

    // Test 5: Edge case - zero processors
    std::vector<double> work5 = {100.0, 200.0};
    starvationHit = scheduler.TestDistributeProcessors(0, work5, allocation);
    std::vector<unsigned> expected5 = {0, 0};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected5.begin(), expected5.end());
    BOOST_CHECK(!starvationHit);

    // Test 6: Edge case - zero work
    std::vector<double> work6 = {0.0, 0.0, 0.0};
    starvationHit = scheduler.TestDistributeProcessors(10, work6, allocation);
    std::vector<unsigned> expected6 = {0, 0, 0};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected6.begin(), expected6.end());
    BOOST_CHECK(!starvationHit);

    // Test 7: Inactive components (work is zero)
    std::vector<double> work7 = {100.0, 0.0, 300.0, 0.0};
    starvationHit = scheduler.TestDistributeProcessors(8, work7, allocation);
    std::vector<unsigned> expected7 = {2, 0, 6, 0};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected7.begin(), expected7.end());
    BOOST_CHECK(!starvationHit);

    // Test 8: Scarcity with equal work
    std::vector<double> work8 = {100.0, 100.0, 100.0, 100.0};
    starvationHit = scheduler.TestDistributeProcessors(3, work8, allocation);
    // Expect processors to be given to the first components due to stable sort
    std::vector<unsigned> expected8 = {0, 1, 1, 1};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected8.begin(), expected8.end());
    BOOST_CHECK(starvationHit);

    // Test 9: Scarcity with one dominant component
    std::vector<double> work9 = {10.0, 10.0, 1000.0};
    starvationHit = scheduler.TestDistributeProcessors(2, work9, allocation);
    // Both processors should go to the largest component
    std::vector<unsigned> expected9 = {0, 1, 1};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected9.begin(), expected9.end());
    BOOST_CHECK(starvationHit);

    // Test 10: Scarcity with a single processor
    std::vector<double> work10 = {10.0, 50.0, 20.0};
    starvationHit = scheduler.TestDistributeProcessors(1, work10, allocation);
    // The single processor should go to the component with the most work
    std::vector<unsigned> expected10 = {0, 1, 0};
    BOOST_CHECK_EQUAL_COLLECTIONS(allocation.begin(), allocation.end(), expected10.begin(), expected10.end());
    BOOST_CHECK(starvationHit);
}

BOOST_AUTO_TEST_SUITE_END()

// Mock implementations for dependencies
using GraphT = osp::ComputationalDagEdgeIdxVectorImplDefT;
using VertexType = GraphT::VertexIdx;

// A mock divider that returns a predictable set of sections.
struct MockDivider2 : public osp::IDagDivider<GraphT> {
    std::vector<std::vector<std::vector<VertexType>>> sectionsToReturn_;

    std::vector<std::vector<std::vector<VertexType>>> divide(const GraphT &) override { return sectionsToReturn_; }
};

// A mock sub-scheduler that returns a simple, predictable schedule.
struct MockSubScheduler : public osp::Scheduler<GraphT> {
    osp::ReturnStatus ComputeSchedule(osp::BspSchedule<GraphT> &schedule) override {
        // Assign all tasks to the first processor in a single superstep
        for (VertexType v = 0; v < schedule.GetInstance().GetComputationalDag().NumVertices(); ++v) {
            schedule.SetAssignedProcessor(v, 0);
            schedule.SetAssignedSuperstep(v, 0);
        }
        schedule.SetNumberOfSupersteps(1);
        return osp::ReturnStatus::OSP_SUCCESS;
    }

    std::string GetScheduleName() const override { return "MockSubScheduler"; }
};

struct TestFixture {
    GraphT dag_;
    osp::BspArchitecture<GraphT> arch_;
    MockDivider2 mockDivider_;
    MockSubScheduler mockSubScheduler_;

    TestFixture() {
        // A simple DAG: v0 -> v1, v2 -> v3
        // Two components that will be in the same wavefront set.
        dag_.add_vertex(10, 1, 1);    // v0
        dag_.add_vertex(20, 1, 1);    // v1
        dag_.add_vertex(30, 1, 1);    // v2
        dag_.add_vertex(40, 1, 1);    // v3
        dag_.add_edge(0, 1);
        dag_.add_edge(2, 3);

        // An architecture with 10 processors of one type
        arch_.setNumberOfProcessors(10);
    }
};

BOOST_FIXTURE_TEST_SUITE(wavefront_component_scheduler_test_suite, TestFixture)

BOOST_AUTO_TEST_CASE(BasicSchedulingTest) {
    // Setup the mock divider to return one section with our two components
    mockDivider_.sectionsToReturn_ = {{{0, 1}}, {{2, 3}}};

    osp::WavefrontComponentScheduler<GraphT, GraphT> scheduler(mockDivider_, mockSubScheduler_);
    osp::BspInstance<GraphT> instance(dag_, arch_);
    osp::BspSchedule<GraphT> schedule(instance);

    auto status = scheduler.ComputeSchedule(schedule);
    BOOST_CHECK_EQUAL(status, osp::ReturnStatus::OSP_SUCCESS);

    BOOST_CHECK_EQUAL(schedule.AssignedProcessor(0), 0);
    BOOST_CHECK_EQUAL(schedule.AssignedProcessor(1), 0);
    BOOST_CHECK_EQUAL(schedule.AssignedProcessor(2), 0);
    BOOST_CHECK_EQUAL(schedule.AssignedProcessor(3), 0);

    BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(0), 0);
    BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(1), 0);
    BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(2), 1);
    BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(3), 1);

    BOOST_CHECK_EQUAL(schedule.NumberOfSupersteps(), 2);
}

BOOST_AUTO_TEST_CASE(MultipleSectionsTest) {
    // Setup the mock divider to return two separate sections
    mockDivider_.sectionsToReturn_ = {
        {{0}, {1}},
        {{2}, {3}}
    };

    osp::WavefrontComponentScheduler<GraphT, GraphT> scheduler(mockDivider_, mockSubScheduler_);
    osp::BspInstance<GraphT> instance(dag_, arch_);
    osp::BspSchedule<GraphT> schedule(instance);

    auto status = scheduler.ComputeSchedule(schedule);
    BOOST_CHECK_EQUAL(status, osp::ReturnStatus::OSP_SUCCESS);

    BOOST_CHECK_EQUAL(schedule.AssignedProcessor(0), 0);
    BOOST_CHECK_EQUAL(schedule.AssignedProcessor(1), 3);
    BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(0), 0);
    BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(1), 0);

    BOOST_CHECK_EQUAL(schedule.AssignedProcessor(2), 0);
    BOOST_CHECK_EQUAL(schedule.AssignedProcessor(3), 4);
    BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(2), 1);
    BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(3), 1);

    BOOST_CHECK_EQUAL(schedule.NumberOfSupersteps(), 2);
}

BOOST_AUTO_TEST_CASE(StarvationReturnsErrorTest) {
    // Use an architecture with only 1 processor
    osp::BspArchitecture<GraphT> scarceArch;
    scarceArch.setNumberOfProcessors(1);

    // Setup the mock divider to return one section with two components
    mockDivider_.sectionsToReturn_ = {
        {{0}, {1}},
        {{2, 3}}
    };

    osp::WavefrontComponentScheduler<GraphT, GraphT> scheduler(mockDivider_, mockSubScheduler_);
    osp::BspInstance<GraphT> instance(dag_, scarceArch);
    osp::BspSchedule<GraphT> schedule(instance);

    // With 2 components and only 1 processor, the starvation case should be hit.
    auto status = scheduler.ComputeSchedule(schedule);
    BOOST_CHECK_EQUAL(status, osp::ReturnStatus::ERROR);
}

BOOST_AUTO_TEST_SUITE_END()

// struct TestFixture_2 {
//     graph_t dag;
//     osp::BspArchitecture<graph_t> arch;
//     MockDivider_2 mock_divider;
//     MockSubScheduler mock_sub_scheduler;

//     TestFixture_2() {
//         // A DAG with two isomorphic components {0,1} and {2,3}, and one unique one {4,5}
//         dag.add_vertex(10, 1, 1); dag.add_vertex(20, 1, 1); // v0, v1
//         dag.add_vertex(10, 1, 1); dag.add_vertex(20, 1, 1); // v2, v3
//         dag.add_vertex(50, 1, 1); dag.add_vertex(50, 1, 1); // v4, v5
//         dag.add_edge(0, 1); dag.add_edge(2, 3); dag.add_edge(4, 5);
//     }
// };

// BOOST_FIXTURE_TEST_SUITE(IsomorphicWavefrontComponentSchedulerTestSuite, TestFixture_2)

// BOOST_AUTO_TEST_CASE(AbundanceSchedulingTest) {

//     arch.setNumberOfProcessors(6);
//     mock_divider.sections_to_return = {{{0, 1}, {2, 3}, {4, 5}}};

//     osp::IsomorphicWavefrontComponentScheduler<graph_t, graph_t> scheduler(mock_divider, mock_sub_scheduler);
//     osp::BspInstance<graph_t> instance(dag, arch);
//     osp::BspSchedule<graph_t> schedule(instance);

//     auto status = scheduler.ComputeSchedule(schedule);
//     BOOST_CHECK_EQUAL(status, osp::ReturnStatus::OSP_SUCCESS);

//     // Member 1 of iso group {0,1} gets 1 proc (global proc 0)
//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(0), 0);
//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(1), 0);
//     BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(0), 0);

//     // Member 2 of iso group {2,3} gets 1 proc (global proc 1)
//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(2), 1);
//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(3), 1);
//     BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(2), 0);

//     // Unique group {4,5} gets 4 procs (global procs 2,3,4,5), sub-schedule uses first one.
//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(4), 2);
//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(5), 2);
//     BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(4), 0);

//     BOOST_CHECK_EQUAL(schedule.NumberOfSupersteps(), 1);
// }

// BOOST_AUTO_TEST_CASE(IndivisibleScarcitySchedulingTest) {
//     // 2 isomorphic components, 1 unique. 3 processors available.
//     arch.setNumberOfProcessors(3);
//     mock_divider.sections_to_return = {{{0, 1}, {2, 3}, {4, 5}}};

//     osp::IsomorphicWavefrontComponentScheduler<graph_t, graph_t> scheduler(mock_divider, mock_sub_scheduler);
//     osp::BspInstance<graph_t> instance(dag, arch);
//     osp::BspSchedule<graph_t> schedule(instance);

//     auto status = scheduler.ComputeSchedule(schedule);
//     BOOST_CHECK_EQUAL(status, osp::ReturnStatus::OSP_SUCCESS);

//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(0), 0);
//     BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(0), 0);
//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(2), 0);
//     BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(2), 1); // Sequential

//     // Unique group scheduled on its 2 processors (global procs 1, 2)
//     BOOST_CHECK_EQUAL(schedule.AssignedProcessor(4), 1);
//     BOOST_CHECK_EQUAL(schedule.AssignedSuperstep(4), 0);

//     BOOST_CHECK_EQUAL(schedule.NumberOfSupersteps(), 2);
// }

// BOOST_AUTO_TEST_CASE(StarvationReturnsErrorTest) {
//     // IsomorphismGroups will find 2 groups: {{0,1}, {2,3}} and {{4,5}}.
//     // With only 1 processor, this is a starvation scenario.
//     arch.setNumberOfProcessors(1);
//     mock_divider.sections_to_return = {{{0, 1}, {2, 3}, {4, 5}}};

//     osp::IsomorphicWavefrontComponentScheduler<graph_t, graph_t> scheduler(mock_divider, mock_sub_scheduler);
//     osp::BspInstance<graph_t> instance(dag, arch);
//     osp::BspSchedule<graph_t> schedule(instance);

//     // With 2 active groups and only 1 processor, starvation is hit.
//     auto status = scheduler.ComputeSchedule(schedule);
//     BOOST_CHECK_EQUAL(status, osp::ReturnStatus::ERROR);
// }

// BOOST_AUTO_TEST_SUITE_END()
