#define BOOST_TEST_MODULE Sets
#include <boost/test/unit_test.hpp>

#include "auxiliary/auxiliary.hpp"

#include <numeric>
#include <unordered_set>
#include <vector>

BOOST_AUTO_TEST_CASE(SetIntersection) {
    std::unordered_set<int> a({3, 6, 2, 6, 7});
    std::unordered_set<int> b({6, 6});
    std::unordered_set<int> c({6});
    std::unordered_set<int> d({1, 5});
    std::unordered_set<int> e({});
    std::unordered_set<int> f({1, 2, 3});
    std::unordered_set<int> g({3, 6, 2, 7});
    std::unordered_set<int> h({2, 3});
    std::unordered_set<int> i({3, 2});
    std::unordered_set<int> j({1});

    BOOST_CHECK(get_intersection(a, b) == c);
    BOOST_CHECK(get_intersection(b, a) == c);
    BOOST_CHECK(get_intersection(c, a) == c);
    BOOST_CHECK(get_intersection(g, g) == g);
    BOOST_CHECK(get_intersection(a, g) == g);
    BOOST_CHECK(get_intersection(a, a) == g);
    BOOST_CHECK(get_intersection(a, f) == i);
    BOOST_CHECK(get_intersection(a, e) == e);
    BOOST_CHECK(get_intersection(d, f) == j);
}

BOOST_AUTO_TEST_CASE(SetIntersectionLarge) {
    std::vector<int> iota_0_to_10k(10'000);
    std::iota(iota_0_to_10k.begin(), iota_0_to_10k.end(), 0);

    std::vector<int> iota_10k_to_20k(10'000);
    std::iota(iota_10k_to_20k.begin(), iota_10k_to_20k.end(), 10'000);

    std::unordered_set<int> iota_0_to_10k_set(iota_0_to_10k.begin(), iota_0_to_10k.end());

    { // Intersection of [0,10k] and [10k,20k]  -->  []
        std::unordered_set<int> iota_10k_to_20k_set(iota_10k_to_20k.begin(), iota_10k_to_20k.end());
        BOOST_CHECK(get_intersection(iota_0_to_10k_set, iota_10k_to_20k_set).empty());
    }

    { // Intersection of [0,10k] and [0k,10k]  -->  [0k,10k]
        BOOST_CHECK(get_intersection(iota_0_to_10k_set, iota_0_to_10k_set) == iota_0_to_10k_set);
    }

    { // Intersection of [0,10k] and [5k,10k]  -->  [5k,10k]
        std::vector<int> iota_5k_to_10k(5'000);
        std::iota(iota_5k_to_10k.begin(), iota_5k_to_10k.end(), 5'000);
        std::unordered_set<int> iota_5k_to_10k_set(iota_5k_to_10k.begin(), iota_5k_to_10k.end());

        BOOST_CHECK(get_intersection(iota_0_to_10k_set, iota_5k_to_10k_set) == iota_5k_to_10k_set);
    }
}

BOOST_AUTO_TEST_CASE(SetUnions) {
    std::unordered_set<int> a({3, 6, 2, 6, 7});
    std::unordered_set<int> b({6, 6});
    std::unordered_set<int> c({6});
    std::unordered_set<int> d({1, 5});
    std::unordered_set<int> e({});
    std::unordered_set<int> f({1, 2, 3});
    std::unordered_set<int> g({3, 6, 2, 7});
    std::unordered_set<int> h({2, 3});
    std::unordered_set<int> i({3, 2});
    std::unordered_set<int> j({1});
    std::unordered_set<int> k({1, 2, 3, 6, 7});
    std::unordered_set<int> l({1, 2, 3, 5});

    BOOST_CHECK(get_union(a, b) == g);
    BOOST_CHECK(get_union(b, a) == a);
    BOOST_CHECK(get_union(c, a) == g);
    BOOST_CHECK(get_union(g, g) == g);
    BOOST_CHECK(get_union(a, g) == g);
    BOOST_CHECK(get_union(a, a) == g);
    BOOST_CHECK(get_union(a, f) == k);
    BOOST_CHECK(get_union(a, e) == a);
    BOOST_CHECK(get_union(d, f) == l);
}

BOOST_AUTO_TEST_CASE(SetUnionLarge) {
    std::vector<int> iota_0_to_10k(10'000);
    std::iota(iota_0_to_10k.begin(), iota_0_to_10k.end(), 0);

    std::vector<int> iota_10k_to_20k(10'000);
    std::iota(iota_10k_to_20k.begin(), iota_10k_to_20k.end(), 10'000);

    std::unordered_set<int> iota_0_to_10k_set(iota_0_to_10k.begin(), iota_0_to_10k.end());

    { // Union of [0,10k] and [10k,20k]  -->  [0k,20k]
        std::unordered_set<int> iota_10k_to_20k_set(iota_10k_to_20k.begin(), iota_10k_to_20k.end());
        std::unordered_set<int> expected_union(iota_0_to_10k.begin(), iota_0_to_10k.end());
        expected_union.insert(iota_10k_to_20k.begin(), iota_10k_to_20k.end());
        BOOST_CHECK(get_union(iota_0_to_10k_set, iota_10k_to_20k_set) == expected_union);
    }

    { // Union of [0,10k] and [0k,10k]  -->  [0k,10k]
        BOOST_CHECK(get_union(iota_0_to_10k_set, iota_0_to_10k_set) == iota_0_to_10k_set);
    }

    { // Union of [0,10k] and [5k,15k]  -->  [0k,15k]
        std::vector<int> iota_5k_to_15k(10'000);
        std::iota(iota_5k_to_15k.begin(), iota_5k_to_15k.end(), 5'000);
        std::unordered_set<int> iota_5k_to_15k_set(iota_5k_to_15k.begin(), iota_5k_to_15k.end());
        std::unordered_set<int> expected_union(iota_0_to_10k.begin(), iota_0_to_10k.end());
        expected_union.insert(iota_5k_to_15k.begin(), iota_5k_to_15k.end());
        BOOST_CHECK(get_union(iota_0_to_10k_set, iota_5k_to_15k_set) == expected_union);
    }
}
