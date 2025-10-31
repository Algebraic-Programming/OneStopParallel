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

#define BOOST_TEST_MODULE TrimmedGroupSchedulerTest
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/isomorphism_divider/TrimmedGroupScheduler.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

using graph_t = computational_dag_vector_impl_def_t;

// Mock SubScheduler for TrimmedGroupScheduler tests
template <typename Constr_Graph_t>
class MockSubScheduler : public Scheduler<Constr_Graph_t> {
public:
    // This mock scheduler assigns all nodes to local processor 0 and superstep 0.
    // This simplifies verification of the TrimmedGroupScheduler's mapping logic.
    RETURN_STATUS computeSchedule(BspSchedule<Constr_Graph_t> &schedule) override {
        for (vertex_idx_t<Constr_Graph_t> v = 0; v < schedule.getInstance().getComputationalDag().num_vertices(); ++v) {
            schedule.setAssignedProcessor(v, 0);
            schedule.setAssignedSuperstep(v, 0);
        }
        schedule.setNumberOfSupersteps(1);
        return RETURN_STATUS::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return "MockSubScheduler"; }
};

struct TrimmedGroupSchedulerFixture {
    graph_t dag;
    BspArchitecture<graph_t> arch;
    BspInstance<graph_t> instance;
    MockSubScheduler<graph_t> mock_sub_scheduler;

    TrimmedGroupSchedulerFixture() : instance(dag, arch) {
        // Default architecture: 1 processor type, 100 memory bound
        arch.setCommunicationCosts(1);
        arch.setSynchronisationCosts(1);
        instance.setAllOnesCompatibilityMatrix(); // All node types compatible with all processor types
    }
};

BOOST_FIXTURE_TEST_SUITE(TrimmedGroupSchedulerTestSuite, TrimmedGroupSchedulerFixture)

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    // Graph is empty by default
    arch.setNumberOfProcessors(4);
    instance.setArchitecture(arch);

    TrimmedGroupScheduler<graph_t> scheduler(mock_sub_scheduler, 1);
    BspSchedule<graph_t> schedule(instance);

    auto status = scheduler.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(status, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), 0);
}

BOOST_AUTO_TEST_CASE(SingleComponentSingleProcessorTypeTest) {
    // Graph: 0-1-2 (single component)
    dag.add_vertex(1, 1, 1, 0); // 0
    dag.add_vertex(1, 1, 1, 0); // 1
    dag.add_vertex(1, 1, 1, 0); // 2
    dag.add_edge(0, 1);
    dag.add_edge(1, 2);
    instance.getComputationalDag() = dag;

    // Architecture: 4 processors of type 0
    arch.setProcessorsWithTypes({0, 0, 0, 0});
    instance.setArchitecture(arch);

    // min_non_zero_procs_ = 1 (all 4 processors assigned to this single component group)
    TrimmedGroupScheduler<graph_t> scheduler(mock_sub_scheduler, 1);
    BspSchedule<graph_t> schedule(instance);

    auto status = scheduler.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(status, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), 1);

    // MockSubScheduler assigns to local proc 0.
    // TrimmedGroupScheduler should map this to global proc 0.
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(0), 0);
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(1), 0);
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(2), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(0), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(1), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(2), 0);
}

BOOST_AUTO_TEST_CASE(MultipleComponentsSingleProcessorTypeEvenDistributionTest) {
    // Graph: 0-1 (component 0), 2-3 (component 1)
    dag.add_vertex(1, 1, 1, 0); // 0
    dag.add_vertex(1, 1, 1, 0); // 1
    dag.add_vertex(1, 1, 1, 0); // 2
    dag.add_vertex(1, 1, 1, 0); // 3
    dag.add_edge(0, 1);
    dag.add_edge(2, 3);
    instance.getComputationalDag() = dag;

    // Architecture: 4 processors of type 0
    arch.setProcessorsWithTypes({0, 0, 0, 0});
    instance.setArchitecture(arch);

    // min_non_zero_procs_ = 2 (2 component groups, each gets 2 processors)
    TrimmedGroupScheduler<graph_t> scheduler(mock_sub_scheduler, 2);
    BspSchedule<graph_t> schedule(instance);

    auto status = scheduler.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(status, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), 1);

    // Component 0 (vertices 0,1) assigned to global processors 0,1. Mock scheduler uses local 0.
    // Global proc for group 0: offset 0 + local 0 = 0.
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(0), 0);
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(1), 0);

    // Component 1 (vertices 2,3) assigned to global processors 2,3. Mock scheduler uses local 0.
    // Global proc for group 1: offset 2 + local 0 = 2.
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(2), 2);
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(3), 2);

    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(0), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(1), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(2), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(3), 0);
}

BOOST_AUTO_TEST_CASE(MultipleComponentsSingleProcessorTypeUnevenDistributionTest) {
    // Graph: 0 (component 0), 1 (component 1), 2 (component 2) - all isolated
    dag.add_vertex(1, 1, 1, 0); // 0
    dag.add_vertex(1, 1, 1, 0); // 1
    dag.add_vertex(1, 1, 1, 0); // 2
    instance.getComputationalDag() = dag;

    // Architecture: 6 processors of type 0
    arch.setProcessorsWithTypes({0, 0, 0, 0, 0, 0});
    instance.setArchitecture(arch);

    // min_non_zero_procs_ = 2 (3 components, 2 groups)
    // base_count = 3 / 2 = 1, remainder = 3 % 2 = 1
    // Group 0 gets 2 components (0, 1)
    // Group 1 gets 1 component (2)
    // sub_proc_counts for type 0: 6 / 2 = 3
    TrimmedGroupScheduler<graph_t> scheduler(mock_sub_scheduler, 2);
    BspSchedule<graph_t> schedule(instance);

    auto status = scheduler.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(status, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), 1);

    // Group 0 (components 0, 1) maps to global procs 0,1,2. Mock scheduler uses local 0.
    // Global proc for group 0: offset 0 + local 0 = 0.
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(0), 0);
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(1), 0);

    // Group 1 (component 2) maps to global procs 3,4,5. Mock scheduler uses local 0.
    // Global proc for group 1: offset 3 + local 0 = 3.
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(2), 3);

    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(0), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(1), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(2), 0);
}

BOOST_AUTO_TEST_CASE(MultipleComponentsHeterogeneousArchitectureTest) {
    // Graph: 0 (type 0), 1 (type 1) - isolated nodes
    dag.add_vertex(1, 1, 1, 0); // 0 (component 0, type 0)
    dag.add_vertex(1, 1, 1, 1); // 1 (component 1, type 1)
    instance.getComputationalDag() = dag;

    // Architecture: 2 processors of type 0 (global 0,1), 2 processors of type 1 (global 2,3)
    arch.setProcessorsWithTypes({0, 0, 1, 1});
    instance.setArchitecture(arch);
    instance.setDiagonalCompatibilityMatrix(2); // Node type 0 compatible with proc type 0, etc.

    // min_non_zero_procs_ = 2 (2 components, 2 groups)
    // sub_proc_counts for type 0: 2 / 2 = 1
    // sub_proc_counts for type 1: 2 / 2 = 1
    TrimmedGroupScheduler<graph_t> scheduler(mock_sub_scheduler, 2);
    BspSchedule<graph_t> schedule(instance);

    auto status = scheduler.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(status, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), 1);

    BOOST_CHECK_EQUAL(schedule.assignedProcessor(0), 0);
    BOOST_CHECK_EQUAL(schedule.assignedProcessor(1), 1);

    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(0), 0);
    BOOST_CHECK_EQUAL(schedule.assignedSuperstep(1), 0);
}

BOOST_AUTO_TEST_SUITE_END()
