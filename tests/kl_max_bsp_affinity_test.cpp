
#define BOOST_TEST_MODULE kl_max_bsp_affinity
#include <boost/test/unit_test.hpp>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_max_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;
using graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using kl_active_schedule_t = KlActiveSchedule<graph, double, NoLocalSearchMemoryConstraint>;

/**
 * Helper to validate comm datastructures by comparing with freshly computed ones
 */
template <typename Graph>
bool validate_comm_datastructures(const MaxCommDatastructure<Graph, double, kl_active_schedule_t> &comm_ds_incremental,
                                  kl_active_schedule_t &active_sched,
                                  const BspInstance<Graph> &instance,
                                  const std::string &context) {
    // 1. Clone Schedule
    BspSchedule<Graph> current_schedule(instance);
    active_sched.WriteSchedule(current_schedule);

    // 2. Fresh Computation
    kl_active_schedule_t kl_sched_fresh;
    kl_sched_fresh.Initialize(current_schedule);

    MaxCommDatastructure<Graph, double, kl_active_schedule_t> comm_ds_fresh;
    comm_ds_fresh.Initialize(kl_sched_fresh);

    // Compute for all steps
    unsigned max_step = current_schedule.NumberOfSupersteps();
    comm_ds_fresh.ComputeCommDatastructures(0, max_step > 0 ? max_step - 1 : 0);

    bool all_match = true;

    // 3. Validate Comm Costs
    for (unsigned step = 0; step < max_step; ++step) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            auto send_inc = comm_ds_incremental.StepProcSend(step, p);
            auto send_fresh = comm_ds_fresh.StepProcSend(step, p);
            auto recv_inc = comm_ds_incremental.StepProcReceive(step, p);
            auto recv_fresh = comm_ds_fresh.StepProcReceive(step, p);

            if (std::abs(send_inc - send_fresh) > 1e-6 || std::abs(recv_inc - recv_fresh) > 1e-6) {
                all_match = false;
                std::cout << "  MISMATCH at step " << step << " proc " << p << ":" << std::endl;
                std::cout << "    Incremental: send=" << send_inc << ", recv=" << recv_inc << std::endl;
                std::cout << "    Fresh:       send=" << send_fresh << ", recv=" << recv_fresh << std::endl;
            }
        }
    }

    // 4. Validate Lambda Maps
    for (const auto v : instance.Vertices()) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            unsigned count_inc = 0;
            if (comm_ds_incremental.nodeLambdaMap_.HasProcEntry(v, p)) {
                count_inc = comm_ds_incremental.nodeLambdaMap_.GetProcEntry(v, p);
            }

            unsigned count_fresh = 0;
            if (comm_ds_fresh.nodeLambdaMap_.HasProcEntry(v, p)) {
                count_fresh = comm_ds_fresh.nodeLambdaMap_.GetProcEntry(v, p);
            }

            if (count_inc != count_fresh) {
                all_match = false;
                std::cout << "  LAMBDA MISMATCH at node " << v << " proc " << p << ":" << std::endl;
                std::cout << "    Incremental: " << count_inc << std::endl;
                std::cout << "    Fresh:       " << count_fresh << std::endl;
            }
        }
    }

    return all_match;
}

BOOST_AUTO_TEST_CASE(test_max_cost_logic) {
    graph dag;
    // v0 -> v1
    const auto v0 = dag.AddVertex(10, 1, 1);    // Work 10
    const auto v1 = dag.AddVertex(10, 1, 1);    // Work 10
    dag.AddEdge(v0, v1, 5);                     // Comm 5

    BspArchitecture<graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    // Initial: v0 on P0 (step 0), v1 on P1 (step 1)
    // Comm from P0 to P1 in step 0 is 5.
    // Work in step 1 is 10.
    // Cost = Work[0] + max(Work[1], Comm[0]) = 10 + max(10, 5) = 20.
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    using comm_cost_t = KlMaxBspCommCostFunction<graph, double, NoLocalSearchMemoryConstraint>;
    using kl_improver_test = KlImproverTest<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.SetupSchedule(schedule);

    // Verify initial cost
    double initial_cost = kl.GetCommCostF().ComputeScheduleCostTest();
    BOOST_CHECK_CLOSE(initial_cost, 20.0, 0.00001);

    // Move v1 to P0 (same proc as v0).
    // Comm becomes 0.
    // Cost = 10 + max(10, 0) = 20.
    // Gain should be 0 because Work[1] dominates Comm[0].

    // We simulate this by checking affinity.
    // Affinity for v1 to move to P0 should be 0 (no improvement).

    kl.InsertGainHeapTest({v1});
    kl.RunInnerIterationTest();

    double after_cost = kl.GetCommCostF().ComputeScheduleCostTest();
    BOOST_CHECK_CLOSE(after_cost, 20.0, 0.00001);
}

BOOST_AUTO_TEST_CASE(test_staleness_penalty) {
    graph dag;
    // v0 -> v1
    const auto v0 = dag.AddVertex(10, 1, 1);
    const auto v1 = dag.AddVertex(10, 1, 1);
    dag.AddEdge(v0, v1, 5);

    BspArchitecture<graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    // Initial: v0 on P0 (step 0), v1 on P1 (step 1).
    // Staleness 2 requires step(v1) >= step(v0) + 2 if different procs.
    // Here step(v1) = 1, step(v0) = 0. Diff = 1 < 2.
    // This should be a penalty.

    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    using comm_cost_t = KlMaxBspCommCostFunction<graph, double, NoLocalSearchMemoryConstraint>;
    using kl_improver_test = KlImproverTest<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.SetupSchedule(schedule);

    // The current implementation might not capture this penalty correctly yet,
    // or it might be captured as a violation.
    // If we move v1 to step 2, the penalty should disappear.

    // Let's check if there are violations.
    // The active schedule data should track violations.
    // But kl_max_bsp_comm_cost_function defines staleness = 2.

    // We can check if moving v1 to P0 (same proc) removes the penalty/violation.

    kl.InsertGainHeapTest({v1});
    kl.RunInnerIterationTest();

    // If logic is correct, moving v1 to P0 should be preferred if it resolves a violation.
    // Or moving v1 to step 2.
}
