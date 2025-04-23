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

#include "scheduler/GreedySchedulers/GreedyBspGrowLocalAutoCores.hpp"
#include <algorithm>
#include <stdexcept>

std::pair<RETURN_STATUS, BspSchedule> GreedyBspGrowLocalAutoCores::computeSchedule(const BspInstance &instance) {

    const unsigned N = instance.numberOfVertices();
    const unsigned P = instance.numberOfProcessors();
    const ComputationalDag& G = instance.getComputationalDag();

    std::vector<unsigned> node_to_proc(N, UINT_MAX), node_to_supstep(N, UINT_MAX);
    std::set<unsigned> ready;
    std::vector<std::set<unsigned>::iterator> place_in_ready(N);

    std::set<unsigned> allReady;
    std::vector<std::set<unsigned>> procReady(P);

    std::vector<unsigned> predec(N);

    for(unsigned node = 0; node < N; ++node) {
        predec[node] = G.numberOfParents(node);
    }

    for(unsigned node = 0; node < N; ++node) {
        if(predec[node] == 0) {
            place_in_ready[node] = ready.insert(node).first;
        }
    }

    std::vector<std::vector<unsigned>> new_assignments(P);
    std::vector<std::vector<unsigned>> best_new_assignments(P);

    std::vector<unsigned> new_ready;
    std::vector<unsigned> best_new_ready;

    const unsigned minWeightParallelCheck = params.syncCostMultiplierParallelCheck * instance.synchronisationCosts();
    const unsigned minSuperstepWeight = params.syncCostMultiplierMinSuperstepWeight * instance.synchronisationCosts();

    double desiredParallelism = static_cast<double>(P);

    unsigned supstep = 0, total_assigned = 0;
    while(total_assigned < N) {

        unsigned limit = params.minSuperstepSize;
        double best_score = 0;
        double best_parallelism = 0;

        bool continueSuperstepAttempts = true;

        while(continueSuperstepAttempts) {
            for (unsigned p = 0; p < P; p++) {
                new_assignments[p].clear();
            }
            new_ready.clear();

            for (unsigned p = 0; p < P; p++) {
                procReady[p].clear();
            }
            
            allReady = ready;

            unsigned new_total_assigned = 0;
            unsigned weight_limit = 0, total_weight_assigned = 0;

            // Processor 0
            while(new_assignments[0].size() < limit) {
                unsigned chosen_node = UINT_MAX;
                if(!procReady[0].empty()) {
                    chosen_node = *procReady[0].begin();
                    procReady[0].erase(procReady[0].begin());
                } else if(!allReady.empty()) {
                    chosen_node = *allReady.begin();
                    allReady.erase(allReady.begin());
                } else {
                    break;
                }

                new_assignments[0].push_back(chosen_node);
                node_to_proc[chosen_node] = 0;
                new_total_assigned++;
                weight_limit += G.nodeWorkWeight(chosen_node);

                for (const auto &succ : G.children(chosen_node)) {
                    if ( node_to_proc[succ] == UINT_MAX ) {
                        node_to_proc[succ] = 0;
                    } else if ( node_to_proc[succ] != 0 ) {
                        node_to_proc[succ] = P;
                    }


                    predec[succ]--;
                    if(predec[succ] == 0) {
                        new_ready.push_back(succ);

                        if( node_to_proc[succ] == 0 ) {
                            procReady[0].insert(succ);
                        }
                    }
                }
            }
            
            total_weight_assigned += weight_limit;



            // Processors 1 through P-1
            for(unsigned proc = 1; proc < P; ++proc) {
                unsigned current_weight_assigned = 0;
                while(current_weight_assigned < weight_limit) {
                    unsigned chosen_node = UINT_MAX;
                    if(!procReady[proc].empty()) {
                        chosen_node = *procReady[proc].begin();
                        procReady[proc].erase(procReady[proc].begin());
                    } else if(!allReady.empty()) {
                        chosen_node = *allReady.begin();
                        allReady.erase(allReady.begin());
                    } else break;

                    new_assignments[proc].push_back(chosen_node);
                    node_to_proc[chosen_node] = proc;
                    new_total_assigned++;
                    current_weight_assigned += G.nodeWorkWeight(chosen_node);

                    for (const auto &succ : G.children(chosen_node)) {
                        if ( node_to_proc[succ] == UINT_MAX ) {
                            node_to_proc[succ] = proc;
                        } else if ( node_to_proc[succ] != proc ) {
                            node_to_proc[succ] = P;
                        }
                        predec[succ]--;
                        if(predec[succ] == 0) {
                            new_ready.push_back(succ);

                            if( node_to_proc[succ] == proc ) {
                                procReady[proc].insert(succ);
                            }
                        }
                    }
                }
                
                weight_limit = std::max(weight_limit, current_weight_assigned);
                total_weight_assigned += current_weight_assigned;
            }

            bool accept_step = false;

            double score = static_cast<double>(total_weight_assigned) / static_cast<double>( weight_limit + instance.synchronisationCosts() );
            double parallelism = 0;
            if (weight_limit > 0) {
                parallelism = static_cast<double>(total_weight_assigned) / static_cast<double>(weight_limit);
            }

            if (score > 0.97 * best_score) {
                best_score = std::max(best_score, score);
                best_parallelism = parallelism;
                accept_step = true;
            } else {
                continueSuperstepAttempts = false;
            }

            if (weight_limit >= minWeightParallelCheck) {
                if (parallelism < std::max(2.0, 0.8 * desiredParallelism)) {
                    continueSuperstepAttempts = false;
                }
            }

            if (weight_limit <= minSuperstepWeight) {
                continueSuperstepAttempts = true;
                if(total_assigned + new_total_assigned == N) {
                    accept_step = true;
                    continueSuperstepAttempts = false;
                }
            }

            if(total_assigned + new_total_assigned == N) {
                continueSuperstepAttempts = false;
            }



            // undo proc assingments and predec decreases in any case
            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    node_to_proc[node] = UINT_MAX;
                }
            }

            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    for (const auto &succ : G.children(node)) {
                        predec[succ]++;
                    }
                }
            }

            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    for (const auto &succ : G.children(node)) {
                        node_to_proc[succ] = UINT_MAX;
                    }
                }
            }

            if(accept_step) {
                best_new_assignments.swap(new_assignments);
                best_new_ready.swap(new_ready);
            }

            limit++;
            limit += ( limit / 2 );
        }

        // apply best iteration
        for(const unsigned &node : best_new_ready) {
            place_in_ready[node] = ready.insert(node).first;
        }

        for(unsigned proc = 0; proc < P; ++proc) {
            for(const unsigned &node : best_new_assignments[proc]) {
                node_to_proc[node] = proc;
                node_to_supstep[node] = supstep;
                ready.erase(place_in_ready[node]);
                ++total_assigned;
                for (const auto &succ : G.children(node)) {
                    predec[succ]--;
                }
            }
        }

        desiredParallelism = (0.3 * desiredParallelism) + (0.6 * best_parallelism) + (0.1 * static_cast<double>(P)); // weights should sum up to one

        ++supstep;
    }

    BspSchedule schedule(instance, node_to_proc, node_to_supstep);
    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};