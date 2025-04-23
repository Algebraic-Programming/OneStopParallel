#define BOOST_TEST_MODULE Minimal_matching
#include <boost/test/unit_test.hpp>

#include "scheduler/Minimal_matching/Hungarian_algorithm.hpp"

BOOST_AUTO_TEST_CASE(Hungarian_Algorithm) {
    std::vector<std::vector<long long unsigned>> costs( { {8, 4, 7}, {5, 2, 3}, {9, 4, 8} } );

    std::vector<unsigned> task_alloc = min_perfect_matching_for_complete_bipartite(costs);

    BOOST_CHECK_EQUAL( task_alloc[0] , 0 );
    BOOST_CHECK_EQUAL( task_alloc[1] , 2 );
    BOOST_CHECK_EQUAL( task_alloc[2] , 1 );


    std::vector<std::vector<long long unsigned>> savings( { {10, 1, 2}, {2, 12, 3}, {2, 4, 18} } );

    task_alloc = max_perfect_matching_for_complete_bipartite(savings);

    BOOST_CHECK_EQUAL( task_alloc[0] , 0 );
    BOOST_CHECK_EQUAL( task_alloc[1] , 1 );
    BOOST_CHECK_EQUAL( task_alloc[2] , 2 );
}