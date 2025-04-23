#define BOOST_TEST_MODULE SSP_Scheduler_Class
#include <boost/test/unit_test.hpp>

#include "model/SspSchedule.hpp"

BOOST_AUTO_TEST_CASE(Schedule_1)
{
    BspArchitecture architecture(2, 2, 3);

    std::vector<std::vector<int>> out = {
        {1, 2, 5},
        {2},
        {},
        {2, 4, 5},
        {5},
        {}
    };
    std::vector<int> work(6,1);
    std::vector<int> comm(6,1);

    ComputationalDag graph(out, work, comm); 

    BspInstance bsp_inst(graph, architecture);

    std::vector<unsigned> proc_assign = {0, 0, 0, 1, 1, 1};
    std::vector<unsigned> step_assign = {0, 1, 2, 0, 1, 2};

    SspSchedule sched(bsp_inst, proc_assign, step_assign, 2);

    BOOST_CHECK_EQUAL(sched.getStaleness(), 2);
    BOOST_CHECK_EQUAL(sched.getMaxStaleness(), 2);
    BOOST_CHECK_EQUAL(sched.satisfiesPrecedenceConstraints(), true);
    BOOST_CHECK_EQUAL(sched.satisfiesPrecedenceConstraints(1), true);
    BOOST_CHECK_EQUAL(sched.satisfiesPrecedenceConstraints(3), false);

    sched.setStaleness(4);
    BOOST_CHECK_EQUAL(sched.getStaleness(), 4);
    BOOST_CHECK_EQUAL(sched.satisfiesPrecedenceConstraints(), false);
}