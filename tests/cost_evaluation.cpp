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

#define BOOST_TEST_MODULE CostEvaluation
#include <boost/test/unit_test.hpp>

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/cost/BufferedSendingCost.hpp"
#include "osp/bsp/model/cost/LazyCommunicationCost.hpp"
#include "osp/bsp/model/cost/TotalCommunicationCost.hpp"
#include "osp/bsp/model/cost/TotalLambdaCommunicationCost.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(TestCostModelsSimpleDag) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;

    BspInstance<Graph> instance;
    instance.SetNumberOfProcessors(2);
    instance.SetCommunicationCosts(10);
    instance.SetSynchronisationCosts(5);

    auto &dag = instance.GetComputationalDag();
    dag.AddVertex(10, 1, 0);
    dag.AddVertex(20, 2, 0);
    dag.AddVertex(30, 3, 0);
    dag.AddVertex(40, 4, 0);
    dag.AddVertex(50, 5, 0);
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);
    dag.AddEdge(1, 4);
    dag.AddEdge(2, 3);
    dag.AddEdge(3, 4);

    BspSchedule<Graph> schedule(instance);

    schedule.SetAssignedProcessor(0, 0);
    schedule.SetAssignedSuperstep(0, 0);
    schedule.SetAssignedProcessor(1, 0);
    schedule.SetAssignedSuperstep(1, 1);
    schedule.SetAssignedProcessor(2, 1);
    schedule.SetAssignedSuperstep(2, 1);
    schedule.SetAssignedProcessor(3, 1);
    schedule.SetAssignedSuperstep(3, 2);
    schedule.SetAssignedProcessor(4, 1);
    schedule.SetAssignedSuperstep(4, 3);
    schedule.UpdateNumberOfSupersteps();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(schedule.NumberOfSupersteps(), 4);

    // Work cost (BSP model) = sum of max work per superstep across processors
    // SS0: max(P0=10, P1=0) = 10
    // SS1: max(P0=20, P1=30) = 30
    // SS2: max(P0=0, P1=40) = 40
    // SS3: max(P0=0, P1=50) = 50
    // Total work = 10 + 30 + 40 + 50 = 130
    BOOST_CHECK_EQUAL(schedule.computeWorkCosts(), 130);

    // LazyCommunicationCost
    // Sends/receives at step_needed - staleness (staleness=1)
    // Node 0→{P1}: step_needed=1, send/rec at SS0, vol=1*1*g=10
    // Node 1→{P1}: step_needed=3, send/rec at SS2, vol=2*1*g=20
    // Max comm per step: SS0=10, SS1=0, SS2=20, SS3=0
    // Comm = 10 + 20 = 30
    // Syncs = 2 * L = 2 * 5 = 10 (only steps with comm)
    // Total = 30 + 10 + 130 = 170
    BOOST_CHECK_EQUAL(LazyCommunicationCost<Graph>()(schedule), 170);

    // BufferedSendingCost
    // Send at producer step, receive at step_needed - staleness
    // Node 0 (SS0): send to P1, vol=1*1*g=10 at SS0, rec at SS0
    // Node 1 (SS1): send to P1, vol=2*1*g=20 at SS1, rec at SS2
    // Send volumes: SS0[P0]=10, SS1[P0]=20, SS2[P0]=0, SS3[P0]=0
    // Recv volumes: SS0[P1]=10, SS1[P1]=0, SS2[P1]=20, SS3[P1]=0
    // Max comm per step: SS0=10, SS1=20, SS2=20, SS3=0
    // Comm = 10 + 20 + 20 = 50
    // Syncs = 3 * L = 3 * 5 = 15 (all steps with comm)
    // Total = 50 + 15 + 130 = 195
    BOOST_CHECK_EQUAL(BufferedSendingCost<Graph>()(schedule), 195);

    // TotalCommunicationCost
    // Sum of cross-processor edge comm weights * g / P
    // Cross edges: 0→2 (cw=1), 1→4 (cw=2)
    // Total cross comm weight = (1 + 2) * 1 = 3
    // Comm cost = 3 * 10 / 2 = 15
    // Work = 130
    // Sync = 3 * 5 = 15 (number_of_supersteps - 1)
    // Total = 15 + 130 + 15 = 160
    BOOST_CHECK_EQUAL(TotalCommunicationCost<Graph>()(schedule), 160);

    // TotalLambdaCommunicationCost
    // For each node, sum comm_weight * sendCosts over unique target processors
    // Then multiply total by (1/P) * g
    // Node 0 (P0, cw=1): target_procs={P0,P1} → 1*(0+1) = 1
    // Node 1 (P0, cw=2): target_procs={P1} → 2*1 = 2
    // Node 2 (P1, cw=3): target_procs={P1} → 3*0 = 0
    // Node 3 (P1, cw=4): target_procs={P1} → 4*0 = 0
    // comm_costs = 1+2+0+0 = 3, comm_cost = 3 * (1/2) * 10 = 15
    // Work = 130, Sync = 3 * 5 = 15
    // Total = 15 + 130 + 15 = 160
    BOOST_CHECK_EQUAL(TotalLambdaCommunicationCost<Graph>()(schedule), 160);
}
