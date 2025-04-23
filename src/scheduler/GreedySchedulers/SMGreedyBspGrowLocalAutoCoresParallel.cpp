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

#include "scheduler/GreedySchedulers/SMGreedyBspGrowLocalAutoCoresParallel.hpp"

#define CACHE_LINE_SIZE 64
#define UNSIGNED_PADDING ((CACHE_LINE_SIZE + sizeof(unsigned) - 1) / sizeof(unsigned))
#define NUM_UNSIGNED_IN_CACHE_LINE ( CACHE_LINE_SIZE / sizeof(unsigned) )

// #define TIME_THREADS_GROW_LOCAL

#ifdef TIME_THREADS_GROW_LOCAL
    #include <chrono>
#endif


std::pair<RETURN_STATUS, SmSchedule> SMGreedyBspGrowLocalAutoCoresParallel::computeSmSchedule(const SmInstance &instance) const {
    unsigned numThreads = params.numThreads;
    if (numThreads == 0) {
        // numThreads = static_cast<unsigned>(std::sqrt( static_cast<double>((instance.numberOfVertices() / 1000000)))) + 1;
        numThreads = static_cast<unsigned>(std::log2( static_cast<double>((instance.numberOfVertices() / 1000)))) + 1;
    }
    numThreads = std::min(numThreads, params.maxNumThreads);
    if (numThreads == 0) {
        numThreads = 1;
    }

    return computeScheduleParallel(instance, numThreads);
};


void SMGreedyBspGrowLocalAutoCoresParallel::computePartialSchedule(const SmInstance &instance, unsigned *const node_to_proc, unsigned *const node_to_supstep,  const unsigned int startNode, const unsigned int endNode, unsigned &supstep) const {

#ifdef TIME_THREADS_GROW_LOCAL
    double startTime = omp_get_wtime();
#endif

    const unsigned N = endNode - startNode;
    const unsigned P = instance.numberOfProcessors();

    const SM_csr * const csr_mat = instance.getMatrix().getCSR();
    const SM_csc * const csc_mat = instance.getMatrix().getCSC();

    std::set<unsigned> ready;

    std::vector<unsigned> futureReady;
    std::vector<unsigned> best_futureReady;

    std::vector<std::set<unsigned>> procReady(P);
    std::vector<std::set<unsigned>> best_procReady(P);

    std::vector<unsigned> predec(N, 0);

    unsigned finalReverseNode = std::max(startNode, 1U); // Node 0 has no parents
    for(unsigned nodePos = std::max(endNode, 1U) - 1; nodePos >= finalReverseNode; nodePos--) { // endNode can be 0 -> startNode = 0 -> finalReverseNode = 1 -> empty loop as should be the case
        unsigned index = nodePos - startNode;

        for (unsigned i = (*csr_mat).outerIndexPtr()[nodePos + 1] - 2; i >= (*csr_mat).outerIndexPtr()[nodePos]; i--) {
            if ((*csr_mat).innerIndexPtr()[i] < startNode) {
                break;
            }
            predec[index]++;
        }
    }

    for(unsigned nodePos = startNode; nodePos < endNode; nodePos++) {
        unsigned index = nodePos - startNode;

        if (predec[index] == 0) {
            ready.insert(nodePos).first;
        }

    }

    std::vector<std::vector<unsigned>> new_assignments(P);
    std::vector<std::vector<unsigned>> best_new_assignments(P);

    const unsigned minWeightParallelCheck = params.syncCostMultiplierParallelCheck * instance.synchronisationCosts();
    const unsigned minSuperstepWeight = params.syncCostMultiplierMinSuperstepWeight * instance.synchronisationCosts();

    double desiredParallelism = static_cast<double>(P);

    unsigned total_assigned = 0;
    supstep = 0;

    while(total_assigned < N) {
        unsigned limit = params.minSuperstepSize;
        double best_score = 0;
        double best_parallelism = 0;

        std::set<unsigned>::iterator readyIter;
        std::set<unsigned>::iterator bestReadyIter;

        bool continueSuperstepAttempts = true;

        while(continueSuperstepAttempts) {
            for (unsigned p = 0; p < P; p++) {
                new_assignments[p].clear();
            }
            futureReady.clear();

            for (unsigned p = 0; p < P; p++) {
                procReady[p].clear();
            }
            
            readyIter = ready.begin();

            unsigned new_total_assigned = 0;
            unsigned weight_limit = 0, total_weight_assigned = 0;


            // Processor 0
            while(new_assignments[0].size() < limit) {
                unsigned chosen_node = UINT_MAX;
                if(!procReady[0].empty()) {
                    chosen_node = *procReady[0].begin();
                    procReady[0].erase(procReady[0].begin());
                } else if( readyIter != ready.end() ) {
                    chosen_node = *readyIter;
                    readyIter++;
                } else {
                    break;
                }

                new_assignments[0].push_back(chosen_node);
                node_to_proc[chosen_node] = 0;
                new_total_assigned++;
                weight_limit += (*csr_mat).outerIndexPtr()[chosen_node + 1] - (*csr_mat).outerIndexPtr()[chosen_node];

                for (unsigned i = (*csc_mat).outerIndexPtr()[chosen_node] + 1; i < (*csc_mat).outerIndexPtr()[chosen_node + 1]; i++) {
                    const unsigned succ = (*csc_mat).innerIndexPtr()[i];
                    if (succ >= endNode) {
                        break;
                    }

                    if ( node_to_proc[succ] == UINT_MAX ) {
                        node_to_proc[succ] = 0;
                    } else if ( node_to_proc[succ] != 0 ) {
                        node_to_proc[succ] = P;
                    }

                    const unsigned succIndex = succ - startNode;
                    predec[succIndex]--;
                    if(predec[succIndex] == 0) {
                        if( node_to_proc[succ] == 0 ) {
                            procReady[0].insert(succ);
                        } else {
                            futureReady.push_back(succ);
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
                    } else if( readyIter != ready.end() ) {
                        chosen_node = *readyIter;
                        readyIter++;
                    } else break;

                    new_assignments[proc].push_back(chosen_node);
                    node_to_proc[chosen_node] = proc;
                    new_total_assigned++;
                    current_weight_assigned += (*csr_mat).outerIndexPtr()[chosen_node + 1] - (*csr_mat).outerIndexPtr()[chosen_node];

                    for (unsigned i = (*csc_mat).outerIndexPtr()[chosen_node] + 1; i < (*csc_mat).outerIndexPtr()[chosen_node + 1]; i++) {
                        const unsigned succ = (*csc_mat).innerIndexPtr()[i];
                        if (succ >= endNode) {
                            break;
                        }

                        if ( node_to_proc[succ] == UINT_MAX ) {
                            node_to_proc[succ] = proc;
                        } else if ( node_to_proc[succ] != proc ) {
                            node_to_proc[succ] = P;
                        }

                        const unsigned succIndex = succ - startNode;
                        predec[succIndex]--;
                        if(predec[succIndex] == 0) {
                            if( node_to_proc[succ] == proc ) {
                                procReady[proc].insert(succ);
                            } else {
                                futureReady.push_back(succ);
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


            if (score > 0.97 * best_score) { // It is possible to make this less strict, i.e. score > 0.98 * best_score. The purpose of this would be to encourage larger supersteps.
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

            // undo proc assingments and predec increases in any case
            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    node_to_proc[node] = UINT_MAX;
                }
            }

            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    for (unsigned i = (*csc_mat).outerIndexPtr()[node] + 1; i < (*csc_mat).outerIndexPtr()[node + 1]; i++) {
                        const unsigned succ = (*csc_mat).innerIndexPtr()[i];
                        if (succ >= endNode) {
                            break;
                        }

                        predec[succ - startNode]++;
                    }
                }
            }

            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    for (unsigned i = (*csc_mat).outerIndexPtr()[node] + 1; i < (*csc_mat).outerIndexPtr()[node + 1]; i++) {
                        const unsigned succ = (*csc_mat).innerIndexPtr()[i];
                        if (succ >= endNode) {
                            break;
                        }
                        node_to_proc[succ] = UINT_MAX;
                    }
                }
            }

            if(accept_step) {
                best_new_assignments.swap(new_assignments);
                best_futureReady.swap(futureReady);
                best_procReady.swap(procReady);
                bestReadyIter = readyIter;
            }

            limit++;
            limit += ( limit / 2 );
        }

        // apply best iteration
        ready.erase(ready.begin(), bestReadyIter);
        ready.insert(best_futureReady.begin(), best_futureReady.end());
        for (unsigned proc = 0; proc < P; proc++) {
            ready.merge( best_procReady[proc] );
        }

        for(unsigned proc = 0; proc < P; ++proc) {
            for(const unsigned &node : best_new_assignments[proc]) {
                node_to_proc[node] = proc;
                node_to_supstep[node] = supstep;
                ++total_assigned;

                for (unsigned i = (*csc_mat).outerIndexPtr()[node] + 1; i < (*csc_mat).outerIndexPtr()[node + 1]; i++) {
                    const unsigned succ = (*csc_mat).innerIndexPtr()[i];
                    if (succ >= endNode) {
                        break;
                    }
                    predec[succ - startNode]--;
                }
            }
        }

        desiredParallelism = (0.3 * desiredParallelism) + (0.6 * best_parallelism) + (0.1 * static_cast<double>(P)); // weights should sum up to one

        ++supstep;
    }

#ifdef TIME_THREADS_GROW_LOCAL
    double endTime = omp_get_wtime();
    std::string padd = "";
    if (omp_get_thread_num() < 10) {
        padd = " ";
    }
    std::string outputString = "Thread: " + padd + std::to_string(omp_get_thread_num()) + "\t Time: " + std::to_string(endTime - startTime) + "\n";
    std::cout << outputString;
#endif

};

void SMGreedyBspGrowLocalAutoCoresParallel::incrementScheduleSupersteps(unsigned *const node_to_supstep, const unsigned startNode, const unsigned endNode, const unsigned incr) const {
    for (unsigned node = startNode; node < endNode; node++) {
        node_to_supstep[node] += incr;
    }
};

std::pair<RETURN_STATUS, SmSchedule> SMGreedyBspGrowLocalAutoCoresParallel::computeScheduleParallel(const SmInstance &instance, unsigned numThreads) const {

    const unsigned N = instance.numberOfVertices();
    const unsigned P = instance.numberOfProcessors();

    unsigned *const node_to_proc = static_cast<unsigned *>(aligned_alloc(CACHE_LINE_SIZE, ((N + NUM_UNSIGNED_IN_CACHE_LINE - 1) / NUM_UNSIGNED_IN_CACHE_LINE) * NUM_UNSIGNED_IN_CACHE_LINE * sizeof(unsigned) ));
    for (unsigned i = 0; i < N; i++) {
        node_to_proc[i] = UINT_MAX;
    }

    unsigned *const node_to_supstep = static_cast<unsigned *>(aligned_alloc(CACHE_LINE_SIZE, ((N + NUM_UNSIGNED_IN_CACHE_LINE - 1) / NUM_UNSIGNED_IN_CACHE_LINE) * NUM_UNSIGNED_IN_CACHE_LINE * sizeof(unsigned) ));
    for (unsigned i = 0; i < N; i++) {
        node_to_supstep[i] = UINT_MAX;
    }


    unsigned numNodesPerThread = N / numThreads;
    std::vector<unsigned> startNodes;
    startNodes.reserve(numThreads + 1);
    unsigned startNode = 0;
    for (unsigned thr = 0; thr < numThreads; thr++) {
        startNodes.push_back(startNode - (startNode % NUM_UNSIGNED_IN_CACHE_LINE));
        startNode += numNodesPerThread;
    }
    startNodes.push_back(N);

    std::vector<unsigned> superstepsThread(numThreads * UNSIGNED_PADDING, 0);
    std::vector<unsigned> supstepIncr(numThreads, 0);
    unsigned incr = 0;

    // for (unsigned i = 0; i < numThreads; i++) {
    //     std::cout << "Thread: " << i << " Address: " << &node_to_proc[startNodes[i]] << std::endl;
    // }

#pragma omp parallel num_threads(numThreads) default(none) shared(instance, node_to_proc, node_to_supstep, superstepsThread, supstepIncr, numThreads, startNodes, incr)
{
#pragma omp for schedule(static, 1)
    for (unsigned thr = 0; thr < numThreads; thr++) {
        computePartialSchedule(instance, node_to_proc, node_to_supstep, startNodes[thr], startNodes[thr + 1], superstepsThread[thr * UNSIGNED_PADDING]);
    }

#pragma omp master
{
    for (unsigned thr = 0; thr < numThreads; thr++) {
        supstepIncr[thr] = incr;
        incr += superstepsThread[thr * UNSIGNED_PADDING];
    }
    // the value of incr is now the number of supersteps
}

#pragma omp barrier

#pragma omp for schedule(static, 1)
    for (unsigned thr = 0; thr < numThreads; thr++) {
        incrementScheduleSupersteps(node_to_supstep, startNodes[thr], startNodes[thr + 1], supstepIncr[thr]);
    }
}

    SmSchedule schedule(instance, N, node_to_proc, node_to_supstep, incr);

    free(node_to_proc);
    free(node_to_supstep);

    return {SUCCESS, schedule};
};