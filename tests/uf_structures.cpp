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


#define BOOST_TEST_MODULE Union_Find
#include <boost/test/unit_test.hpp>

#include "osp/auxiliary/datastructures/union_find.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace osp;

BOOST_AUTO_TEST_CASE(Union_find_structure1) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    Union_Find_Universe<std::string, unsigned, int, int> test_universe(names);

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(test_universe.find_origin_by_name(name), name);
    }

    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 6);

    BOOST_CHECK_THROW(test_universe.add_object("a"), std::runtime_error);
    BOOST_CHECK_THROW(test_universe.add_object("e"), std::runtime_error);

    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 6);

    test_universe.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 5);

    test_universe.join_by_name("b", "c");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 4);
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("c"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("b"), test_universe.find_origin_by_name("c"));

    test_universe.join_by_name("d", "b");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 3);
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("d"), test_universe.find_origin_by_name("b"));

    test_universe.join_by_name("a", "c");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 3);
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("c"));

    test_universe.join_by_name("a", "d");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 3);
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("d"));

    test_universe.join_by_name("e", "f");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 2);
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("e"), test_universe.find_origin_by_name("f"));
    BOOST_CHECK_NE(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("f"));

    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("b"), test_universe.find_origin_by_name("c"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("c"), test_universe.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("c"), test_universe.find_origin_by_name("b"));

    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("e"), test_universe.find_origin_by_name("f"));

    BOOST_CHECK_NE(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("f"));
}

BOOST_AUTO_TEST_CASE(Union_find_structure2) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f", "g", "h", "i"});
    Union_Find_Universe<std::string, unsigned, int, int> test_universe;

    for (auto &name : names) {
        test_universe.add_object(name);
    }

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(test_universe.find_origin_by_name(name), name);
    }

    BOOST_CHECK_THROW(test_universe.add_object("c"), std::runtime_error);
    BOOST_CHECK_THROW(test_universe.add_object("i"), std::runtime_error);

    for (auto &name : names) {
        BOOST_CHECK_EQUAL(test_universe.find_origin_by_name(name), name);
    }

    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 9);

    test_universe.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 8);
    test_universe.join_by_name("b", "c");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 7);
    test_universe.join_by_name("c", "d");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 6);
    test_universe.join_by_name("d", "e");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 5);
    test_universe.join_by_name("e", "f");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 4);

    test_universe.join_by_name("c", "f");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 4);

    test_universe.join_by_name("g", "h");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 3);
    test_universe.join_by_name("h", "i");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 2);

    test_universe.join_by_name("b", "h");
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 1);

    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("b"), test_universe.find_origin_by_name("c"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("c"), test_universe.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("h"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("c"), test_universe.find_origin_by_name("i"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("f"), test_universe.find_origin_by_name("g"));
}

BOOST_AUTO_TEST_CASE(Union_find_weight_structure) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    std::vector<unsigned> weights({1, 2, 1, 3, 1, 1});

    Union_Find_Universe<std::string, unsigned, unsigned, unsigned> test_universe(names, weights, weights);

    for (size_t i = 0; i < names.size(); i++) {
        BOOST_CHECK_EQUAL(test_universe.find_origin_by_name(names[i]), names[i]);
        BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name(names[i]), weights[i]);
        BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name(names[i]), weights[i]);
    }

    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 6);

    BOOST_CHECK_THROW(test_universe.add_object("a"), std::runtime_error);
    BOOST_CHECK_THROW(test_universe.add_object("e"), std::runtime_error);

    test_universe.join_by_name("a", "b");
    test_universe.join_by_name("b", "c");
    test_universe.join_by_name("d", "b");
    test_universe.join_by_name("a", "c");
    test_universe.join_by_name("a", "d");

    test_universe.join_by_name("e", "f");

    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("b"), test_universe.find_origin_by_name("c"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("c"), test_universe.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("d"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("c"), test_universe.find_origin_by_name("b"));

    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("e"), test_universe.find_origin_by_name("f"));

    BOOST_CHECK_NE(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("f"));

    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 7);
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("b"), 7);
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("e"), 2);

    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 7);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("e"), 2);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("b"), 7);

    std::vector<std::pair<std::vector<std::string>, unsigned>> components_n_weights =
        test_universe.get_connected_components_and_weights();
    unsigned total_comp_weights = 0;
    unsigned total_elements = 0;
    for (auto &[comp, wt] : components_n_weights) {
        total_comp_weights += wt;
        total_elements += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            BOOST_CHECK(std::any_of(names.cbegin(), names.cend(),
                                    [name](std::string other_name) { return name == other_name; }));
        }
    }

    std::vector<std::tuple<std::vector<std::string>, unsigned, unsigned>> components_n_weights_n_memory =
        test_universe.get_connected_components_weights_and_memories();
    unsigned total_comp_weights_2 = 0;
    unsigned total_comp_memory = 0;
    unsigned total_elements_2 = 0;
    for (const auto &[comp, wt, mem] : components_n_weights_n_memory) {
        total_comp_weights_2 += wt;
        total_comp_memory += mem;
        total_elements_2 += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            BOOST_CHECK(std::any_of(names.cbegin(), names.cend(),
                                    [name](std::string other_name) { return name == other_name; }));
        }
    }

    unsigned total_weight = 0;
    for (auto &wt : weights) {
        total_weight += wt;
    }

    BOOST_CHECK_EQUAL(total_elements, names.size());
    BOOST_CHECK_EQUAL(total_elements_2, names.size());
    BOOST_CHECK_EQUAL(total_weight, total_comp_weights);
    BOOST_CHECK_EQUAL(total_weight, total_comp_weights_2);
    BOOST_CHECK_EQUAL(total_weight, total_comp_memory);

    for (auto &name : names) {
        BOOST_CHECK(std::any_of(components_n_weights.cbegin(), components_n_weights.cend(),
                                [name](std::pair<std::vector<std::string>, unsigned> comp_pair) {
                                    return std::any_of(comp_pair.first.cbegin(), comp_pair.first.cend(),
                                                       [name](std::string other_name) { return name == other_name; });
                                }));
    }
}

BOOST_AUTO_TEST_CASE(Union_find_structure_weight_comp_count) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    std::vector<unsigned> weights({1, 2, 1, 3, 1, 1});
    Union_Find_Universe<std::string, unsigned, unsigned, unsigned> test_universe;

    for (size_t i = 0; i < names.size(); i++) {
        test_universe.add_object(names[i], weights[i], weights[i]);
    }

    for (size_t i = 0; i < names.size(); i++) {
        BOOST_CHECK_EQUAL(test_universe.find_origin_by_name(names[i]), names[i]);
        BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name(names[i]), weights[i]);
        BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name(names[i]), weights[i]);
    }

    test_universe.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 3);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 3);
    test_universe.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 3);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 3);
    test_universe.join_by_name("b", "a");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 3);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 3);

    test_universe.join_by_name("a", "c");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("c"), 4);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("c"), 4);
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("b"), test_universe.find_origin_by_name("c"));

    test_universe.join_by_name("d", "e");
    test_universe.join_by_name("e", "f");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("f"), 5);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("f"), 5);
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("e"), test_universe.find_origin_by_name("f"));
    BOOST_CHECK_NE(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("e"));
    BOOST_CHECK_NE(test_universe.find_origin_by_name("b"), test_universe.find_origin_by_name("d"));

    std::vector<std::pair<std::vector<std::string>, unsigned>> comp_n_weights =
        test_universe.get_connected_components_and_weights();
    BOOST_CHECK(comp_n_weights.size() == 2);
    BOOST_CHECK(comp_n_weights.size() == test_universe.get_number_of_connected_components());
    BOOST_CHECK(comp_n_weights[0].first.size() == 3);
    BOOST_CHECK(comp_n_weights[1].first.size() == 3);
    BOOST_CHECK((comp_n_weights[0].second == 4 && comp_n_weights[1].second == 5) ||
                (comp_n_weights[0].second == 5 && comp_n_weights[1].second == 4));

    std::vector<std::tuple<std::vector<std::string>, unsigned, unsigned>> comp_n_weight_n_memory =
        test_universe.get_connected_components_weights_and_memories();
    BOOST_CHECK(comp_n_weight_n_memory.size() == 2);
    BOOST_CHECK(comp_n_weight_n_memory.size() == test_universe.get_number_of_connected_components());
    BOOST_CHECK(std::get<0>(comp_n_weight_n_memory[0]).size() == 3);
    BOOST_CHECK(std::get<0>(comp_n_weight_n_memory[1]).size() == 3);
    BOOST_CHECK((std::get<1>(comp_n_weight_n_memory[0]) == 4 && std::get<1>(comp_n_weight_n_memory[1]) == 5) ||
                (std::get<1>(comp_n_weight_n_memory[0]) == 5 && std::get<1>(comp_n_weight_n_memory[1]) == 4));
    BOOST_CHECK((std::get<2>(comp_n_weight_n_memory[0]) == 4 && std::get<2>(comp_n_weight_n_memory[1]) == 5) ||
                (std::get<2>(comp_n_weight_n_memory[0]) == 5 && std::get<2>(comp_n_weight_n_memory[1]) == 4));

    std::vector<std::pair<std::vector<std::string>, unsigned>> components_n_weights =
        test_universe.get_connected_components_and_weights();
    unsigned total_comp_weights = 0;
    unsigned total_elements = 0;
    for (auto &[comp, wt] : components_n_weights) {
        total_comp_weights += wt;
        total_elements += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            BOOST_CHECK(std::any_of(names.cbegin(), names.cend(),
                                    [name](std::string other_name) { return name == other_name; }));
        }
    }

    unsigned total_weight = 0;
    for (auto &wt : weights) {
        total_weight += wt;
    }

    BOOST_CHECK_EQUAL(total_elements, names.size());
    BOOST_CHECK_EQUAL(total_weight, total_comp_weights);
    for (auto &name : names) {
        BOOST_CHECK(std::any_of(components_n_weights.cbegin(), components_n_weights.cend(),
                                [name](std::pair<std::vector<std::string>, unsigned> comp_pair) {
                                    return std::any_of(comp_pair.first.cbegin(), comp_pair.first.cend(),
                                                       [name](std::string other_name) { return name == other_name; });
                                }));
    }
}

BOOST_AUTO_TEST_CASE(Union_find_structure_weight_chains_comp_count) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f", "g", "h", "i"});
    std::vector<unsigned> weights({1, 1, 1, 1, 1, 1, 1, 1, 1});
    Union_Find_Universe<std::string, unsigned, unsigned, unsigned> test_universe(names, weights, weights);

    test_universe.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 2);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 2);
    test_universe.join_by_name("b", "c");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 3);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 3);
    test_universe.join_by_name("c", "d");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("c"), 4);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("c"), 4);
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("e"), 1);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("e"), 1);
    test_universe.join_by_name("d", "e");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("e"), 5);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("e"), 5);
    test_universe.join_by_name("a", "b");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 5);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 5);
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("e"), 5);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("e"), 5);
    test_universe.join_by_name("e", "f");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 6);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 6);

    test_universe.join_by_name("c", "f");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 6);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 6);
    BOOST_CHECK_EQUAL(test_universe.get_number_of_connected_components(), 4);

    test_universe.join_by_name("g", "h");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("g"), 2);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("g"), 2);

    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("i"), "i");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("i"), 1);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("i"), 1);

    test_universe.join_by_name("h", "i");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("i"), 3);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("i"), 3);

    test_universe.join_by_name("b", "h");
    BOOST_CHECK_EQUAL(test_universe.get_weight_of_component_by_name("a"), 9);
    BOOST_CHECK_EQUAL(test_universe.get_memory_of_component_by_name("a"), 9);

    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("a"), test_universe.find_origin_by_name("b"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("d"), test_universe.find_origin_by_name("i"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("e"), test_universe.find_origin_by_name("h"));
    BOOST_CHECK_EQUAL(test_universe.find_origin_by_name("b"), test_universe.find_origin_by_name("i"));

    std::vector<std::pair<std::vector<std::string>, unsigned>> components_n_weights =
        test_universe.get_connected_components_and_weights();
    unsigned total_comp_weights = 0;
    unsigned total_elements = 0;
    for (auto &[comp, wt] : components_n_weights) {
        total_comp_weights += wt;
        total_elements += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            BOOST_CHECK(std::any_of(names.cbegin(), names.cend(),
                                    [name](std::string other_name) { return name == other_name; }));
        }
    }

    unsigned total_weight = 0;
    for (auto &wt : weights) {
        total_weight += wt;
    }

    BOOST_CHECK_EQUAL(total_elements, names.size());
    BOOST_CHECK_EQUAL(total_weight, total_comp_weights);
    for (auto &name : names) {
        BOOST_CHECK(std::any_of(components_n_weights.cbegin(), components_n_weights.cend(),
                                [name](std::pair<std::vector<std::string>, unsigned> comp_pair) {
                                    return std::any_of(comp_pair.first.cbegin(), comp_pair.first.cend(),
                                                       [name](std::string other_name) { return name == other_name; });
                                }));
    }
}
