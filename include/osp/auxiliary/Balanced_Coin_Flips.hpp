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

enum CoinType { THUE_MORSE, BIASED_RANDOMLY };

class BalancedCoinFlips {
  public:
    /// @brief Returns true/false in a pseudo-random balanced manner
    /// @return true/false
    virtual bool GetFlip() = 0;

    virtual ~BalancedCoinFlips() = default;
};

class BiasedRandom : public BalancedCoinFlips {
  public:
    bool GetFlip() override {
        int genuineRandomSize = 3;
        int dieSize = 2 * genuineRandomSize + abs(trueBias_);
        std::uniform_int_distribution<int> distrib(0, dieSize - 1);
        int flip = distrib(gen_);
        if (trueBias_ >= 0) {
            if (flip >= genuineRandomSize) {
                trueBias_--;
                return true;
            } else {
                trueBias_++;
                return false;
            }
        } else {
            if (flip >= genuineRandomSize) {
                trueBias_++;
                return false;
            } else {
                trueBias_--;
                return true;
            }
        }
        throw std::runtime_error("Coin landed on its side!");
    }

    BiasedRandom(std::size_t seed = 1729U) : gen_(seed), trueBias_(0) {};

  private:
    /// @brief Random number generator
    std::mt19937 gen_;
    /// @brief Biases the coin towards true
    int trueBias_;
};

/// @brief Generates the Thue Morse Sequence
/// @param shift Starting point in the sequence
class ThueMorseSequence : public BalancedCoinFlips {
  public:
    ThueMorseSequence() {
        next_ = static_cast<long unsigned>(RandInt(1024));
        sequence_.emplace_back(false);
    }

    ThueMorseSequence(long unsigned int shift) : next_(shift) { sequence_.emplace_back(false); }

    bool GetFlip() override {
        for (long unsigned int i = sequence_.size(); i <= next_; i++) {
            if (i % 2 == 0) {
                sequence_.emplace_back(sequence_[i / 2]);
            } else {
                sequence_.emplace_back(!sequence_[i / 2]);
            }
        }
        return sequence_[next_++];
    }

  private:
    long unsigned int next_;
    std::vector<bool> sequence_;
};

/// @brief Coin flip with 1/3 chance to return previous toss otherwise fair toss
class RepeatChance : public BalancedCoinFlips {
  public:
    bool GetFlip() override {
        if (RandInt(3) > 0) {
            previous_ = (RandInt(2) == 0);
        }
        return previous_;
    }

    RepeatChance() { previous_ = (RandInt(2) == 0); };

  private:
    bool previous_;
};

class BiasedRandomWithSideBias : public BalancedCoinFlips {
  public:
    bool GetFlip() override {
        unsigned genuineRandomSize = 3;

        const long long absTrueBias = std::abs(trueBias_);
        if (absTrueBias > std::numeric_limits<unsigned>::max()) {
            throw std::runtime_error("true_bias is too large!");
        }

        unsigned dieSize = (sideRatio_.first + sideRatio_.second) * genuineRandomSize + static_cast<unsigned>(absTrueBias);

        if (dieSize > static_cast<unsigned>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("die_size is too large!");
        }

        unsigned flip = static_cast<unsigned>(RandInt(static_cast<int>(dieSize)));
        if (trueBias_ >= 0) {
            if (flip >= sideRatio_.second * genuineRandomSize) {
                trueBias_ -= sideRatio_.second;
                return true;
            } else {
                trueBias_ += sideRatio_.first;
                return false;
            }
        } else {
            if (flip >= sideRatio_.first * genuineRandomSize) {
                trueBias_ += sideRatio_.first;
                return false;
            } else {
                trueBias_ -= sideRatio_.second;
                return true;
            }
        }
        throw std::runtime_error("Coin landed on its side!");
    }

    BiasedRandomWithSideBias(const std::pair<unsigned, unsigned> sideRatio = std::make_pair(1, 1))
        : trueBias_(0), sideRatio_(sideRatio) {};

  private:
    /// @brief Biases the coin towards true
    long long int trueBias_;
    /// @brief ratio true : false
    const std::pair<unsigned, unsigned> sideRatio_;
};

}    // namespace osp
