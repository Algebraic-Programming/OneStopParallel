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

#include <cmath>
#include <stdexcept>

#include "model/BspArchitecture.hpp"
#include <auxiliary/auxiliary.hpp>

void BspArchitecture::setSendCosts(const std::vector<std::vector<unsigned int>> &vec) {

    if (vec.size() != number_processors) {
        throw std::invalid_argument("Invalid Argument");
    }

    isNuma = false;
    for (unsigned i = 0; i < number_processors; i++) {

        if (vec[i].size() != number_processors) {
            throw std::invalid_argument("Invalid Argument");
        }

        for (unsigned j = 0; j < number_processors; j++) {

            if (i == j) {
                if (vec[i][j] != 0)
                    throw std::invalid_argument("Invalid Argument, Diagonal elements should be 0");
            } else {
                send_costs[i][j] = vec[i][j];

                if ( number_processors > 1 && vec[i][j] != vec[0][1] ) {
                    isNuma = true;
                }
            }
        }
    }
}

void BspArchitecture::setNumberOfProcessors(unsigned int num_proc) {

    number_processors = num_proc;
    number_of_processor_types = 1;
    processor_type = std::vector<unsigned int>(number_processors, 0);
    send_costs =
        std::vector<std::vector<unsigned int>>(number_processors, std::vector<unsigned int>(number_processors, 1));
    for (unsigned i = 0; i < number_processors; i++) {
        send_costs[i][i] = 0;
    }
    memory_bound.resize(num_proc, memory_bound.back());

    isNuma = false;
}

void BspArchitecture::setProcessorsWithTypes(const std::vector<unsigned> & processor_types_) {

    number_processors = processor_types_.size();
    number_of_processor_types = 0;
    processor_type = processor_types_;
    send_costs =
        std::vector<std::vector<unsigned int>>(number_processors, std::vector<unsigned int>(number_processors, 1));
    for (unsigned i = 0; i < number_processors; i++) {
        send_costs[i][i] = 0;
    }
    memory_bound.resize(number_processors, memory_bound.back());

    isNuma = false;
    updateNumberOfProcessorTypes();
}

void BspArchitecture::set_processors_consequ_types(const std::vector<unsigned> &processor_type_count_, const std::vector<unsigned> &processor_type_memory_) {

    assert(processor_type_count_.size() == processor_type_memory_.size());

    number_of_processor_types = processor_type_count_.size();
    number_processors = std::accumulate(processor_type_count_.begin(), processor_type_count_.end(), 0);
    processor_type = std::vector<unsigned int>(number_processors, 0);
    memory_bound = std::vector<unsigned int>(number_processors, 0);

    unsigned offset = 0;
    for (unsigned i = 0; i < processor_type_count_.size(); i++) {
            
            for (unsigned j = 0; j < processor_type_count_[i]; j++) {
                processor_type[offset + j] = i;
                memory_bound[offset + j] = processor_type_memory_[i];
            }
            offset += processor_type_count_[i];
    }

    send_costs =
        std::vector<std::vector<unsigned int>>(number_processors, std::vector<unsigned int>(number_processors, 1));
    for (unsigned i = 0; i < number_processors; i++) {
        send_costs[i][i] = 0;
    }
    isNuma = false;
}


void BspArchitecture::SetUniformSendCost() {
    for (unsigned i = 0; i < number_processors; i++) {
        for (unsigned j = 0; j < number_processors; j++) {
            if (i == j) {
                send_costs[i][j] = 0;
            } else {
                send_costs[i][j] = 1;
            }
        }
    }
    isNuma = false;
}

void BspArchitecture::SetExpSendCost(const unsigned int base) {

    isNuma = true;

    unsigned maxPos = 1;
    const unsigned two = 2;
    for (; uintpow(two, maxPos + 1) <= number_processors - 1; ++maxPos) {
    }
    for (unsigned i = 0; i < number_processors; ++i)
        for (unsigned j = i + 1; j < number_processors; ++j)
            for (unsigned pos = maxPos; pos >= 0; --pos)
                if (((1 << pos) & i) != ((1 << pos) & j)) {
                    send_costs[i][j] = send_costs[j][i] = intpow(base, pos);
                    break;
                }
};

// compute average comm. coefficient between a pair of processors
unsigned BspArchitecture::computeCommAverage() const {
    double avg = 0;
    for (unsigned i = 0; i < number_processors; ++i)
        for (unsigned j = 0; j < number_processors; ++j)
            avg += send_costs[i][j];
    avg = avg * (double)communication_costs / (double)number_processors / (double)number_processors;
    return static_cast<unsigned>(round(avg));
};

bool BspArchitecture::are_send_cost_numa() {
    if (number_processors == 1) return false;

    const unsigned val = send_costs[0][1];
    for (unsigned p1 = 0; p1 < number_processors; p1++) {
        for (unsigned p2 = 0; p2 < number_processors; p2++) {
            if (p1 == p2) continue;
            if (send_costs[p1][p2] != val) return true;
        }
    }
    return false;
}

void BspArchitecture::print_architecture(std::ostream& os) const {

    os << "Architectur info:  number of processors: " << number_processors 
    << ", Number of processor types: " << number_of_processor_types 
    << ", Communication costs: " << communication_costs 
    << ", Synchronization costs: " << synchronisation_costs << std::endl;
    os << std::setw(17) << " Processor: ";
    for (unsigned i = 0; i < number_processors; i++) {
        os << std::right << std::setw(5) << i << " ";
    }
    os << std::endl;
    os << std::setw(17) << "Processor type: ";
    for (unsigned i = 0; i < number_processors; i++) {
        os << std::right << std::setw(5) << processor_type[i] << " ";
    }
    os << std::endl;
    os << std::setw(17) << "Memory bound: "; 
    for (unsigned i = 0; i < number_processors; i++) {
        os << std::right << std::setw(5) << memory_bound[i] << " ";
    }
    os << std::endl;
}

void BspArchitecture::updateNumberOfProcessorTypes() {
    number_of_processor_types = 0;
    for (unsigned p = 0; p < number_processors; p++) {
        if(processor_type[p] >= number_of_processor_types) {
            number_of_processor_types = processor_type[p] + 1;
        }
    }
}

unsigned BspArchitecture::maxMemoryBoundProcType(unsigned procType) const {
    unsigned max_mem = 0;
    for (unsigned proc = 0; proc < number_processors; proc++) {
        if (processor_type[proc] == procType) {
            max_mem = std::max(max_mem, memory_bound[proc]);
        }
    }
    return max_mem;
}