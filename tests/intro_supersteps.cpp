#define BOOST_TEST_MODULE Intro_Supersteps
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

#include "refine/new_superstep.hpp"

BOOST_AUTO_TEST_CASE(weight_bal_cut) {

    const DAG graph(
        {// In edges
         {6, 2, 5},
         {6, 9},
         {6, 3, 5},
         {},
         {8},
         {9},
         {9},
         {0},
         {},
         {}},
        {// Out edges
         {7},
         {},
         {0},
         {2},
         {},
         {2, 0},
         {1, 2, 0},
         {},
         {4},
         {6, 1, 5}},
        {1, 1, 1, 1, 2, 3, 2, 1, 1, 1}, {1, 1, 1, 1, 2, 3, 2, 1, 1, 1});
    const SubDAG graph_sub = graph.toSubDAG();

    const DAG graph_empty;
    const SubDAG graph_empty_sub = graph_empty.toSubDAG();

    const std::vector<int> biases = {-20, -2, -1, 0, 2, 5, 8, 18};

    for (const auto &bias : biases) {
        std::pair<std::unordered_set<int>, std::unordered_set<int>> cuts = dag_weight_bal_cut(graph_sub, bias);
        for (int i = 0; i < graph_sub.n; i++) {
            BOOST_CHECK((cuts.first.find(i) != cuts.first.end()) || (cuts.second.find(i) != cuts.second.end()));
            BOOST_CHECK(!((cuts.first.find(i) != cuts.first.end()) && (cuts.second.find(i) != cuts.second.end())));
        }
        for (int i = 0; i < graph_sub.n; i++) {
            for (const auto &j : graph_sub.Out[i]) {
                if (cuts.first.find(graph_sub.sub_to_super.at(j)) != cuts.first.end()) {
                    BOOST_CHECK(cuts.first.find(graph_sub.sub_to_super.at(i)) != cuts.first.end());
                }
                if (cuts.second.find(graph_sub.sub_to_super.at(i)) != cuts.second.end()) {
                    BOOST_CHECK(cuts.second.find(graph_sub.sub_to_super.at(j)) != cuts.second.end());
                }
            }
        }

        // std::cout << "Top: ";
        // for (auto& node: cuts.first) {
        //     std::cout << node << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "Bot: ";
        // for (auto& node: cuts.second) {
        //     std::cout << node << " ";
        // }
        // std::cout << std::endl;
    }

    std::pair<std::unordered_set<int>, std::unordered_set<int>> cuts = dag_weight_bal_cut(graph_empty_sub, 0);
    BOOST_CHECK(cuts.first == std::unordered_set<int>({}));
    BOOST_CHECK(cuts.second == std::unordered_set<int>({}));
};

BOOST_AUTO_TEST_CASE(top_shaves) {

    DAG graph;

    graph.n = 10;
    // Side-question: If n is equal to the size of the vectors, why do we need to store it?
    // Because it is redundant!
    graph.In = {{6, 2, 5}, {6, 9}, {6, 3, 5}, {}, {8}, {9}, {9}, {0}, {}, {}};
    graph.Out = {{7}, {}, {0}, {2}, {}, {2, 0}, {1, 2, 0}, {}, {4}, {6, 1, 5}};
    graph.workW = {1, 1, 1, 1, 2, 3, 2, 1, 1, 1};
    graph.commW = graph.workW;

    SubDAG graph_sub = graph.toSubDAG();

    DAG graph_empty;

    graph_empty.n = 0;
    // Side-question: If n is equal to the size of the vectors, why do we need to store it?
    // Because it is redundant!
    graph_empty.In = {};
    graph_empty.Out = {};
    graph_empty.workW = {};
    graph_empty.commW = graph_empty.workW;

    SubDAG graph_empty_sub = graph_empty.toSubDAG();

    std::pair<std::unordered_set<int>, std::unordered_set<int>> cutties = top_shave_few_sources(graph_sub);
    for (int i = 0; i < graph_sub.n; i++) {
        BOOST_CHECK((cutties.first.find(i) != cutties.first.end()) || (cutties.second.find(i) != cutties.second.end()));
        BOOST_CHECK(
            !((cutties.first.find(i) != cutties.first.end()) && (cutties.second.find(i) != cutties.second.end())));
    }
    for (int i = 0; i < graph_sub.n; i++) {
        for (auto &j : graph_sub.Out[i]) {
            if (cutties.first.find(graph_sub.sub_to_super.at(j)) != cutties.first.end()) {
                BOOST_CHECK(cutties.first.find(graph_sub.sub_to_super.at(i)) != cutties.first.end());
            }
            if (cutties.second.find(graph_sub.sub_to_super.at(i)) != cutties.second.end()) {
                BOOST_CHECK(cutties.second.find(graph_sub.sub_to_super.at(j)) != cutties.second.end());
            }
        }
    }

    // std::cout << "Top: ";
    // for (auto& node: cutties.first) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Bot: ";
    // for (auto& node: cutties.second) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;

    std::pair<std::unordered_set<int>, std::unordered_set<int>> cuts = top_shave_few_sources(graph_empty_sub);
    BOOST_CHECK(cuts.first == std::unordered_set<int>({}));
    BOOST_CHECK(cuts.second == std::unordered_set<int>({}));

    DAG graph2;

    graph2.n = 7;
    // Side-question: If n is equal to the size of the vectors, why do we need to store it?
    // Because it is redundant!
    graph2.In = {
        {}, {0}, {0}, {1}, {1, 2}, {2}, {3, 5},
    };
    graph2.Out = {
        {1, 2}, {3, 4}, {4, 5}, {6}, {}, {6}, {},
    };
    graph2.workW = {1, 1, 1, 1, 1, 1, 3};
    graph2.commW = graph2.workW;

    SubDAG graph_sub2 = graph2.toSubDAG();

    cutties = top_shave_few_sources(graph_sub2);
    for (int i = 0; i < graph_sub2.n; i++) {
        BOOST_CHECK((cutties.first.find(i) != cutties.first.end()) || (cutties.second.find(i) != cutties.second.end()));
        BOOST_CHECK(
            !((cutties.first.find(i) != cutties.first.end()) && (cutties.second.find(i) != cutties.second.end())));
    }
    for (int i = 0; i < graph_sub2.n; i++) {
        for (auto &j : graph_sub2.Out[i]) {
            if (cutties.first.find(graph_sub2.sub_to_super.at(j)) != cutties.first.end()) {
                BOOST_CHECK(cutties.first.find(graph_sub2.sub_to_super.at(i)) != cutties.first.end());
            }
            if (cutties.second.find(graph_sub2.sub_to_super.at(i)) != cutties.second.end()) {
                BOOST_CHECK(cutties.second.find(graph_sub2.sub_to_super.at(j)) != cutties.second.end());
            }
        }
    }

    std::unordered_set<int> graph2_top_allocation({0, 1, 2});
    BOOST_CHECK(cutties.first == graph2_top_allocation);

    // std::cout << "Top: ";
    // for (auto& node: cutties.first) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Bot: ";
    // for (auto& node: cutties.second) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;
};

BOOST_AUTO_TEST_CASE(bottom_shaves) {

    DAG graph;

    graph.n = 10;
    // Side-question: If n is equal to the size of the vectors, why do we need to store it?
    // Because it is redundant!
    graph.In = {{6, 2, 5}, {6, 9}, {6, 3, 5}, {}, {8}, {9}, {9}, {0}, {}, {}};
    graph.Out = {{7}, {}, {0}, {2}, {}, {2, 0}, {1, 2, 0}, {}, {4}, {6, 1, 5}};
    graph.workW = {1, 1, 1, 1, 2, 3, 2, 1, 1, 1};
    graph.commW = graph.workW;

    SubDAG graph_sub = graph.toSubDAG();

    DAG graph_empty;

    graph_empty.n = 0;
    // Side-question: If n is equal to the size of the vectors, why do we need to store it?
    // Because it is redundant!
    graph_empty.In = {};
    graph_empty.Out = {};
    graph_empty.workW = {};
    graph_empty.commW = graph_empty.workW;

    SubDAG graph_empty_sub = graph_empty.toSubDAG();

    std::pair<std::unordered_set<int>, std::unordered_set<int>> cutties = bottom_shave_few_sinks(graph_sub);
    for (int i = 0; i < graph_sub.n; i++) {
        BOOST_CHECK((cutties.first.find(i) != cutties.first.end()) || (cutties.second.find(i) != cutties.second.end()));
        BOOST_CHECK(
            !((cutties.first.find(i) != cutties.first.end()) && (cutties.second.find(i) != cutties.second.end())));
    }
    for (int i = 0; i < graph_sub.n; i++) {
        for (auto &j : graph_sub.Out[i]) {
            if (cutties.first.find(graph_sub.sub_to_super.at(j)) != cutties.first.end()) {
                BOOST_CHECK(cutties.first.find(graph_sub.sub_to_super.at(i)) != cutties.first.end());
            }
            if (cutties.second.find(graph_sub.sub_to_super.at(i)) != cutties.second.end()) {
                BOOST_CHECK(cutties.second.find(graph_sub.sub_to_super.at(j)) != cutties.second.end());
            }
        }
    }

    // std::cout << "Top: ";
    // for (auto& node: cutties.first) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Bot: ";
    // for (auto& node: cutties.second) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;

    std::pair<std::unordered_set<int>, std::unordered_set<int>> cuts = bottom_shave_few_sinks(graph_empty_sub);
    BOOST_CHECK(cuts.first == std::unordered_set<int>({}));
    BOOST_CHECK(cuts.second == std::unordered_set<int>({}));

    DAG graph2;

    graph2.n = 7;
    // Side-question: If n is equal to the size of the vectors, why do we need to store it?
    // Because it is redundant!
    graph2.Out = {
        {}, {0}, {0}, {1}, {1, 2}, {2}, {3, 5},
    };
    graph2.In = {
        {1, 2}, {3, 4}, {4, 5}, {6}, {}, {6}, {},
    };
    graph2.workW = {1, 1, 1, 1, 1, 1, 3};
    graph2.commW = graph2.workW;

    SubDAG graph_sub2 = graph2.toSubDAG();

    cutties = bottom_shave_few_sinks(graph_sub2);
    // std::pair<std::unordered_set<int>, std::unordered_set<int>> cutties = bottom_shave_few_sinks(graph_sub2);
    for (int i = 0; i < graph_sub2.n; i++) {
        BOOST_CHECK((cutties.first.find(i) != cutties.first.end()) || (cutties.second.find(i) != cutties.second.end()));
        BOOST_CHECK(
            !((cutties.first.find(i) != cutties.first.end()) && (cutties.second.find(i) != cutties.second.end())));
    }
    for (int i = 0; i < graph_sub2.n; i++) {
        for (auto &j : graph_sub2.Out[i]) {
            if (cutties.first.find(graph_sub2.sub_to_super.at(j)) != cutties.first.end()) {
                BOOST_CHECK(cutties.first.find(graph_sub2.sub_to_super.at(i)) != cutties.first.end());
            }
            if (cutties.second.find(graph_sub2.sub_to_super.at(i)) != cutties.second.end()) {
                BOOST_CHECK(cutties.second.find(graph_sub2.sub_to_super.at(j)) != cutties.second.end());
            }
        }
    }

    std::unordered_set<int> graph2_top_allocation({0, 1, 2});
    BOOST_CHECK(cutties.second == graph2_top_allocation);

    // std::cout << "Top: ";
    // for (auto& node: cutties.first) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Bot: ";
    // for (auto& node: cutties.second) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;
};