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

#include <random>
#include <stdexcept>
#include <vector>

#include "osp/auxiliary/misc.hpp"

namespace osp {

enum CoinType { Thue_Morse, Biased_Randomly };

class BalancedCoinFlips {
  public:
    /// @brief Returns true/false in a pseudo-random balanced manner
    /// @return true/false
    virtual bool get_flip() = 0;

    virtual ~BalancedCoinFlips() = default;
};

class Biased_Random : public BalancedCoinFlips {
  public:
    bool get_flip() override {
        int genuine_random_size = 3;
        int die_size = 2 * genuine_random_size + abs(true_bias);
        std::uniform_int_distribution<int> distrib(0, die_size - 1);
        int flip = distrib(gen);
        if (true_bias >= 0) {
            if (flip >= genuine_random_size) {
                true_bias--;
                return true;
            } else {
                true_bias++;
                return false;
            }
        } else {
            if (flip >= genuine_random_size) {
                true_bias++;
                return false;
            } else {
                true_bias--;
                return true;
            }
        }
        throw std::runtime_error("Coin landed on its side!");
    }

    Biased_Random(std::size_t seed = 1729U) : gen(seed), true_bias(0) {};

  private:
    /// @brief Random number generator
    std::mt19937 gen;
    /// @brief Biases the coin towards true
    int true_bias;
};

/// @brief Generates the Thue Morse Sequence
/// @param shift Starting point in the sequence
class Thue_Morse_Sequence : public BalancedCoinFlips {
  public:
    Thue_Morse_Sequence() {
        next = static_cast<long unsigned>(randInt(1024));
        sequence.emplace_back(false);
    }

    Thue_Morse_Sequence(long unsigned int shift) : next(shift) { sequence.emplace_back(false); }

    bool get_flip() override {
        for (long unsigned int i = sequence.size(); i <= next; i++) {
            if (i % 2 == 0) {
                sequence.emplace_back(sequence[i / 2]);
            } else {
                sequence.emplace_back(!sequence[i / 2]);
            }
        }
        return sequence[next++];
    }

  private:
    long unsigned int next;
    std::vector<bool> sequence;
};

/// @brief Coin flip with 1/3 chance to return previous toss otherwise fair toss
class Repeat_Chance : public BalancedCoinFlips {
  public:
    bool get_flip() override {
        if (randInt(3) > 0) { previous = (randInt(2) == 0); }
        return previous;
    }

    Repeat_Chance() { previous = (randInt(2) == 0); };

  private:
    bool previous;
};

class Biased_Random_with_side_bias : public BalancedCoinFlips {
  public:
    bool get_flip() override {
        unsigned genuine_random_size = 3;

        const long long abs_true_bias = std::abs(true_bias);
        if (abs_true_bias > std::numeric_limits<unsigned>::max()) { throw std::runtime_error("true_bias is too large!"); }

        unsigned die_size = (side_ratio.first + side_ratio.second) * genuine_random_size + static_cast<unsigned>(abs_true_bias);

        if (die_size > static_cast<unsigned>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("die_size is too large!");
        }

        unsigned flip = static_cast<unsigned>(randInt(static_cast<int>(die_size)));
        if (true_bias >= 0) {
            if (flip >= side_ratio.second * genuine_random_size) {
                true_bias -= side_ratio.second;
                return true;
            } else {
                true_bias += side_ratio.first;
                return false;
            }
        } else {
            if (flip >= side_ratio.first * genuine_random_size) {
                true_bias += side_ratio.first;
                return false;
            } else {
                true_bias -= side_ratio.second;
                return true;
            }
        }
        throw std::runtime_error("Coin landed on its side!");
    }

    Biased_Random_with_side_bias(const std::pair<unsigned, unsigned> side_ratio_ = std::make_pair(1, 1))
        : true_bias(0), side_ratio(side_ratio_) {};

  private:
    /// @brief Biases the coin towards true
    long long int true_bias;
    /// @brief ratio true : false
    const std::pair<unsigned, unsigned> side_ratio;
};

}    // namespace osp
