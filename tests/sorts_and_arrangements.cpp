#define BOOST_TEST_MODULE Sorts_and_Arrangements
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "auxiliary/auxiliary.hpp"

BOOST_AUTO_TEST_CASE(Sorts_and_Arrangements1) {
    std::vector<int> a({4, 7, 2, -2, 4});
    std::vector<size_t> neg_test1({4, 7, 2, 8, 4});
    std::vector<size_t> neg_test2({8, 2, 4, 4, 7});
    std::vector<int> b = a;
    std::vector<int> a_sort({-2, 2, 4, 4, 7});
    std::vector<size_t> a_re1({3, 2, 0, 4, 1});
    std::vector<size_t> a_re2({3, 2, 4, 0, 1});

    std::vector<size_t> re = sort_and_sorting_arrangement(a);
    BOOST_CHECK(re == a_re1 || re == a_re2);
    BOOST_CHECK(a == a_sort);

    BOOST_CHECK(check_vector_is_rearrangement_of_0_to_N(re));
    BOOST_CHECK(check_vector_is_rearrangement_of_0_to_N(a_re1));
    BOOST_CHECK(!check_vector_is_rearrangement_of_0_to_N(neg_test1));
    BOOST_CHECK(!check_vector_is_rearrangement_of_0_to_N(neg_test2));

    std::cout << "b: ";
    for (auto &i : b) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    sort_like_arrangement(b, re);

    std::cout << "re: ";
    for (auto &i : re) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    std::cout << "b: ";
    for (auto &i : b) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    BOOST_CHECK(a == b);

    std::cout << "a: ";
    for (auto &i : a) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(Sorts_and_Arrangements2) {
    std::vector<std::string> a({"aa", "z", "b", "trace", "racket"});
    std::vector<size_t> c({16, 901, 2, 8, 29});
    std::vector<size_t> b = c;
    std::vector<std::string> a_sort({"b", "trace", "aa", "racket", "z"});
    std::vector<size_t> c_re({2, 3, 0, 4, 1});

    BOOST_CHECK(check_vector_is_rearrangement_of_0_to_N(c_re));
    BOOST_CHECK(!check_vector_is_rearrangement_of_0_to_N(c));

    BOOST_CHECK(sorting_arrangement(c) == c_re);
    BOOST_CHECK(c == b);

    sort_like(a, c);

    BOOST_CHECK(a == a_sort);
    BOOST_CHECK(c == b);
}

BOOST_AUTO_TEST_CASE(Sorts_and_Arrangements3) {
    std::vector<int> id({0, 1, 2, 3, 4, 5, 6});
    std::vector<int> v = id;
    std::vector<int> perm_a({0, 2, 1, 3, 4, 5, 6});
    std::vector<int> perm_b({0, 2, 1, 4, 5, 6, 3});
    std::vector<int> perm_c({1, 2, 0, 3, 4, 5, 6});

    sort_like(v, id);
    BOOST_CHECK(v == id);
    sort_like(v, perm_a);
    BOOST_CHECK(v == perm_a);
    sort_like(v, perm_a);
    BOOST_CHECK(v == id);

    sort_like(v, perm_b);
    BOOST_CHECK(v != perm_b);
    sort_like(v, perm_b);
    BOOST_CHECK(v != id);
    sort_like(v, perm_b);
    BOOST_CHECK(v == perm_b);
    sort_like(v, perm_b);
    BOOST_CHECK(v == id);

    sort_like(v, perm_c);
    BOOST_CHECK(v != perm_c);
    sort_like(v, perm_c);
    BOOST_CHECK(v == perm_c);
    sort_like(v, perm_c);
    BOOST_CHECK(v == id);
}
