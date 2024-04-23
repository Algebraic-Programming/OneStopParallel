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

#pragma once

#include <vector>
#include <stdexcept>
#include <random>

#include "auxiliary/auxiliary.hpp"

enum CoinType { Thue_Morse, Biased_Randomly };

class BalancedCoinFlips {
    public:
        /// @brief Returns true/false in a pseudo-random balanced manner
        /// @return true/false
        virtual bool get_flip() = 0;
};

class Biased_Random : public BalancedCoinFlips {
    public:
        bool get_flip() override;
        Biased_Random() : true_bias(0) { };
    private:
        /// @brief Biases the coin towards true
        int true_bias;
};


/// @brief Generates the Thue Morse Sequence
/// @param shift Starting point in the sequence
class Thue_Morse_Sequence : public BalancedCoinFlips {
    public:
        bool get_flip() override;
        Thue_Morse_Sequence();
        Thue_Morse_Sequence(long unsigned int shift);
    private:
        long unsigned int next;
        std::vector<bool> sequence;
};

/// @brief Coin flip with 1/3 chance to return previous toss otherwise fair toss
class Repeat_Chance : public BalancedCoinFlips {
    public:
        bool get_flip() override;
        Repeat_Chance() {
            previous = (randInt(2) == 0 );
            };
    private:
        bool previous;
};


class Biased_Random_with_side_bias : public BalancedCoinFlips {
    public:
        bool get_flip() override;
        Biased_Random_with_side_bias( const std::pair<unsigned, unsigned> side_ratio_ = std::make_pair(1,1) ) : true_bias(0), side_ratio(side_ratio_) { };
    private:
        /// @brief Biases the coin towards true
        long long int true_bias;
        /// @brief ratio true : false
        const std::pair<unsigned, unsigned> side_ratio;
};