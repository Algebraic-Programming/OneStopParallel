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

#include "auxiliary/Balanced_Coin_Flips.hpp"

bool osp::Biased_Random::get_flip() {
    int genuine_random_size = 3;
    int die_size = 2*genuine_random_size+abs(true_bias);
    int flip = randInt(die_size);
    if (true_bias >= 0) {
        if (flip >= genuine_random_size) {
            true_bias--;
            return true;
        }
        else {
            true_bias++;
            return false;
        }
    }
    else {
        if (flip >= genuine_random_size) {
            true_bias++;
            return false;
        }
        else {
            true_bias--;
            return true;
        }
    }
    throw std::runtime_error("Coin landed on its side!");
}



bool osp::Thue_Morse_Sequence::get_flip() {
    for (long unsigned int i = sequence.size(); i <= next; i++) {
        if (i % 2 == 0) {
            sequence.emplace_back( sequence[i/2] );
        }
        else {
            sequence.emplace_back( !sequence[i/2] );
        }
    }
    return sequence[next++];
}

bool osp::Repeat_Chance::get_flip() {
    if ( randInt(3) > 0 ) {
        previous = (randInt(2) == 0);
    }
    return previous;
}

bool osp::Biased_Random_with_side_bias::get_flip() {
    unsigned genuine_random_size = 3;
    unsigned die_size = (side_ratio.first+side_ratio.second)*genuine_random_size+abs(true_bias);
    unsigned flip = randInt(die_size);
    if (true_bias >= 0) {
        if (flip >= side_ratio.second * genuine_random_size) {
            true_bias -= side_ratio.second;
            return true;
        }
        else {
            true_bias += side_ratio.first;
            return false;
        }
    }
    else {
        if (flip >= side_ratio.first * genuine_random_size) {
            true_bias += side_ratio.first;
            return false;
        }
        else {
            true_bias -= side_ratio.second;
            return true;
        }
    }
    throw std::runtime_error("Coin landed on its side!");
}