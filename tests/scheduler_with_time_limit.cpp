#define BOOST_TEST_MODULE scheduler_with_time_limit
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

#include "DummyScheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "structures/dag.hpp"

BOOST_AUTO_TEST_CASE(RunWithTimeLimit) {
    BspInstance instance;
    DummyScheduler scheduler(2);
    auto result = scheduler.computeScheduleWithTimeLimit(instance);
    BOOST_CHECK_EQUAL(result.first, TIMEOUT);
}
