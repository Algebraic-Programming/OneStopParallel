#define BOOST_TEST_MODULE Hash_Pair
#include <boost/test/unit_test.hpp>

#include "auxiliary/auxiliary.hpp"

BOOST_AUTO_TEST_CASE(Hash_Pair) {
    std::pair<int, int> p1({0,0});
    std::pair<int, int> p2({1,1});
    std::pair<int, int> p3({1,2});
    std::pair<int, int> p4({2,1});
    std::pair<int, int> p5({1,3});
    std::pair<int, int> p6({2,6});
    std::pair<int, int> p7 = p6;

    pair_hash hasher;


    BOOST_CHECK( hasher(p7) == hasher(p6) );

    // Can technically fail, but should not
    BOOST_CHECK( hasher(p1) != hasher(p2) );
    BOOST_CHECK( hasher(p3) != hasher(p4) );
    BOOST_CHECK( hasher(p2) != hasher(p3) );
    BOOST_CHECK( hasher(p2) != hasher(p5) );
    BOOST_CHECK( hasher(p4) != hasher(p6) );
}