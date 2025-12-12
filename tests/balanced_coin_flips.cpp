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

#define BOOST_TEST_MODULE Balanced_Coin_Flips
#include "osp/auxiliary/Balanced_Coin_Flips.hpp"

#include <bitset>
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace osp;

bool ThueMorseGen(long unsigned int n) {
    // std::bitset<sizeof(n)*CHAR_BIT> bits(n);
    unsigned long int binSum = 0;
    while (n != 0) {
        binSum += n % 2;
        n /= 2;
    }
    return bool(binSum % 2);    // (bits.count()%2);
}

BOOST_AUTO_TEST_CASE(RandomBiasedCoin) {
    std::cout << "True: " << true << " False: " << false << std::endl;
    Biased_Random coin;
    std::cout << "Biased Coin: ";
    for (int i = 0; i < 200; i++) {
        std::cout << coin.get_flip();
    }
    std::cout << std::endl << std::endl;
}

BOOST_AUTO_TEST_CASE(ThueMorse) {
    Thue_Morse_Sequence coin(0);

    std::vector<bool> beginning({0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1});
    std::vector<bool> generated;
    for (long unsigned i = 0; i < beginning.size(); i++) {
        bool next = coin.get_flip();
        generated.emplace_back(next);
        // std::cout << next;
    }
    // std::cout << std::endl;

    BOOST_CHECK(beginning == generated);

    Thue_Morse_Sequence testCoinInSeq(0);
    for (unsigned i = 0; i < 200; i++) {
        BOOST_CHECK_EQUAL(testCoinInSeq.get_flip(), ThueMorseGen(i));
        // std::cout << "hi " << i << std::endl;
    }

    for (int i = 0; i < 100; i++) {
        unsigned ind = static_cast<unsigned>(randInt(1048575));
        Thue_Morse_Sequence testCoinRandom(ind);
        BOOST_CHECK_EQUAL(testCoinRandom.get_flip(), ThueMorseGen(ind));
        // std::cout << "bye " << i << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(RepeaterCoin) {
    Repeat_Chance coin;
    std::cout << "Repeater Coin: ";
    for (int i = 0; i < 200; i++) {
        std::cout << coin.get_flip();
    }
    std::cout << std::endl << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomBiasedCoinWithSideBias11) {
    Biased_Random_with_side_bias coin({1, 1});
    int trueCount = 0;
    int falseCount = 0;
    std::cout << "Biased Coin with side bias 1:1 : ";
    for (int i = 0; i < 200; i++) {
        bool flip = coin.get_flip();
        if (flip) {
            trueCount++;
        } else {
            falseCount++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << trueCount << " False count: " << falseCount << std::endl;
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomBiasedCoinWithSideBias10) {
    Biased_Random_with_side_bias coin({1, 0});
    int trueCount = 0;
    int falseCount = 0;
    std::cout << "Biased Coin with side bias 1:0 : ";
    for (int i = 0; i < 200; i++) {
        bool flip = coin.get_flip();
        if (flip) {
            trueCount++;
        } else {
            falseCount++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << trueCount << " False count: " << falseCount << std::endl;
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomBiasedCoinWithSideBias01) {
    Biased_Random_with_side_bias coin({0, 1});
    int trueCount = 0;
    int falseCount = 0;
    std::cout << "Biased Coin with side bias 0:1 : ";
    for (int i = 0; i < 200; i++) {
        bool flip = coin.get_flip();
        if (flip) {
            trueCount++;
        } else {
            falseCount++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << trueCount << " False count: " << falseCount << std::endl;
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomBiasedCoinWithSideBias32) {
    Biased_Random_with_side_bias coin({3, 2});
    int trueCount = 0;
    int falseCount = 0;
    std::cout << "Biased Coin with side bias 3:2 : ";
    for (int i = 0; i < 200; i++) {
        bool flip = coin.get_flip();
        if (flip) {
            trueCount++;
        } else {
            falseCount++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << trueCount << " False count: " << falseCount << std::endl;
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(RandomBiasedCoinWithSideBias31) {
    Biased_Random_with_side_bias coin({3, 1});
    int trueCount = 0;
    int falseCount = 0;
    std::cout << "Biased Coin with side bias 3:1 : ";
    for (int i = 0; i < 200; i++) {
        bool flip = coin.get_flip();
        if (flip) {
            trueCount++;
        } else {
            falseCount++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << trueCount << " False count: " << falseCount << std::endl;
    std::cout << std::endl;
}
