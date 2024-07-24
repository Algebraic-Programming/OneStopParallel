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

#include "algorithms/SubArchitectureSchedulers/SubArchitectures.hpp"

void SubArchitectureScheduler::setTimeLimitSeconds(unsigned int limit) {
    timeLimitSeconds = limit;
    if (scheduler) scheduler->setTimeLimitHours(limit);
}

void SubArchitectureScheduler::setTimeLimitHours(unsigned int limit) {
    timeLimitSeconds = limit * 3600;
    if (scheduler) scheduler->setTimeLimitHours(limit);
}

void SubArchitectureScheduler::min_symmetric_sub_sum( const std::vector<std::vector<unsigned>>& matrix,
                            const size_t size,
                            std::vector<unsigned>& current_processors,
                            std::vector<unsigned>& best_ans,
                            long unsigned& current_best) {
    if (current_processors.size() == size) {
        long unsigned sum = 0;
        for (const auto& p1 : current_processors) {
            for (const auto& p2: current_processors) {
                sum += matrix[p1][p2];
            }
        }
        if (sum < current_best) {
            current_best = sum;
            best_ans = current_processors;
        }
        return;
    }

    size_t new_iteration_start = 0;
    if (current_processors.size() == 0) {
        new_iteration_start = 0;
    } else {
        new_iteration_start = current_processors.back()+1;
    }
    for (size_t new_proc = new_iteration_start; new_proc < matrix.size(); new_proc++) {
        current_processors.push_back(new_proc);
        min_symmetric_sub_sum(matrix, size, current_processors, best_ans, current_best);
        current_processors.pop_back();
    }

    return;
}

std::vector<unsigned> SubArchitectureScheduler::min_symmetric_sub_sum(const std::vector<std::vector<unsigned>>& matrix, const size_t size) {
    assert(size <= matrix.size());
    assert( matrix.size() == 0 || std::all_of(matrix.begin(), matrix.end(), [matrix](const auto& line) { return line.size() == matrix[0].size(); }) );

    std::vector<unsigned> best_ans;
    best_ans.reserve(size);
    std::vector<unsigned> current_processors({});
    current_processors.reserve(size);
    long unsigned curr_best_costs = ULONG_MAX;
    
    min_symmetric_sub_sum(matrix, size, current_processors, best_ans, curr_best_costs);
    return best_ans;
}

std::pair<RETURN_STATUS, BspSchedule> SubArchitectureScheduler::computeSchedule_fixed_size(const BspInstance &instance, size_t size) {
    if ( size > instance.numberOfProcessors() ) {
        throw std::runtime_error("Number of processors must be at most the number of processors of the passed architecture.");
    }

    // Computing new Architecture
    const std::vector<unsigned> best_processors = min_symmetric_sub_sum(instance.getArchitecture().sendCostMatrix(), size);
    std::vector<std::vector<unsigned>> new_send_cost(size, std::vector<unsigned>(size,0));
    for (size_t i = 0; i < best_processors.size(); i++) {
        for (size_t j = 0; j < best_processors.size(); j++) {
            new_send_cost[i][j] = instance.getArchitecture().sendCosts(best_processors[i], best_processors[j]);
        }
    }
    BspArchitecture new_arch(size, instance.getArchitecture().communicationCosts(), instance.getArchitecture().synchronisationCosts(), new_send_cost);
    BspInstance new_inst(instance.getComputationalDag(), new_arch);

    auto [status, sched] = scheduler->computeSchedule(new_inst);
    BspSchedule new_schedule(instance);
    for (const auto& vert : instance.getComputationalDag().vertices()) {
        new_schedule.setAssignedSuperstep(vert, sched.assignedSuperstep(vert));
        new_schedule.setAssignedProcessor(vert, best_processors[sched.assignedProcessor(vert)] );
    }
    for (const auto& [triple, step] : sched.getCommunicationSchedule() ) {
        new_schedule.addCommunicationScheduleEntry( std::get<0>(triple), best_processors[std::get<1>(triple)], best_processors[std::get<2>(triple)], step);
    }

    return {status, new_schedule};
}


std::pair<RETURN_STATUS, BspSchedule> SubArchitectureScheduler::computeSchedule(const BspInstance &instance) {
    if ( num_processors > instance.numberOfProcessors() ) {
        throw std::runtime_error("Number of processors must be at most the number of processors of the passed architecture.");
    }

    if (num_processors > 0) {
        return computeSchedule_fixed_size(instance, num_processors);
    }
    
    RETURN_STATUS best_status = ERROR;
    BspSchedule best_out;
    unsigned best_costs = UINT_MAX;

    unsigned i = 1;
    while (i <= instance.numberOfProcessors()) {
        auto result = computeSchedule_fixed_size(instance, i);
        unsigned costs = result.second.computeCosts();
        if (costs < best_costs) {
            best_num_processors = i;
            best_out = result.second;
            best_status = result.first;
            best_costs = costs;
        }

        if (logarithmic && i < instance.numberOfProcessors()) {
            i = std::min(2*i, instance.numberOfProcessors());
        } else {
            i++;
        }
    }    

    return {best_status, best_out};
}


// BspArchitecture subarchitecture_min_total_comm_cost(const BspArchitecture& architecture, unsigned num_processors) {
//     

//     std::vector<std::vector<unsigned>> new_send_costs(num_processors, std::vector<unsigned>(num_processors));
//     std::vector<unsigned> best_procs = min_symmetric_sub_sum(architecture.sendCostMatrix(), num_processors);

//     for     
// }