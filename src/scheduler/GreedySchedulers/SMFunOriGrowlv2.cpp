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

#include "scheduler/GreedySchedulers/SMFunOriGrowlv2.hpp"

#define CACHE_LINE_SIZE 64

#define UNSIGNED_PADDING ((CACHE_LINE_SIZE + sizeof(unsigned) - 1) / sizeof(unsigned))
#define NUM_UNSIGNED_IN_CACHE_LINE ( CACHE_LINE_SIZE / sizeof(unsigned) )

#define BOOL_PADDING ((CACHE_LINE_SIZE + sizeof(bool) - 1) / sizeof(bool))
#define NUM_BOOL_IN_CACHE_LINE ( CACHE_LINE_SIZE / sizeof(bool) )

// #define TIME_THREADS_FUN_GROWL_V2

#define FUNORIGROWLV2_INCREASE_QUEUEING

#ifdef TIME_THREADS_FUN_GROWL_V2
    #include <chrono>
#endif


std::pair<RETURN_STATUS, SmSchedule> SMFunOriGrowlv2::computeSmSchedule(const SmInstance &instance) const {
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
}

void SMFunOriGrowlv2::computeTransitiveReduction(const SmInstance &instance, bool *const transEdgeMask, const unsigned int startNode, const unsigned int endNode) const {
#ifdef TIME_THREADS_FUN_GROWL_V2
    double startTime = omp_get_wtime();
#endif

    const SM_csr * const csr_mat = instance.getMatrix().getCSR();

    // Transitive edge reduction on CSR    
    unsigned finalReverseNode = std::max(startNode, 1U); // Node 0 has no parents
    for(unsigned vert = std::max(endNode, 1U) - 1; vert >= finalReverseNode; vert--) { // endNode can be 0 -> startNode = 0 -> finalReverseNode = 1 -> empty loop as should be the case
        for (unsigned lateIndex = (*csr_mat).outerIndexPtr()[vert + 1] - 2; lateIndex >= (*csr_mat).outerIndexPtr()[vert] + 1; lateIndex--) {
            
            const unsigned lateParent = (*csr_mat).innerIndexPtr()[lateIndex];

            unsigned earlyIndex = lateIndex - 1;
            unsigned grandIndex = (*csr_mat).outerIndexPtr()[lateParent + 1] - 2;

            while ( (earlyIndex >= (*csr_mat).outerIndexPtr()[vert]) && (grandIndex >= (*csr_mat).outerIndexPtr()[lateParent]) ) {
                
                const unsigned earlyParent = (*csr_mat).innerIndexPtr()[earlyIndex];
                const unsigned grandParent = (*csr_mat).innerIndexPtr()[grandIndex];
                
                if (earlyParent == grandParent) {
                    transEdgeMask[earlyIndex] = false;

                    earlyIndex--;
                    grandIndex--;
                } else if (earlyParent < grandParent) {
                    grandIndex--;
                } else {
                    earlyIndex--;
                }
            }
        }
    }


#ifdef TIME_THREADS_FUN_GROWL_V2
    double endTransRedTime = omp_get_wtime();
    std::string padd = "";
    if (omp_get_thread_num() < 10) {
        padd = " ";
    }
    std::string outputTransRedString = "Thread: " + padd + std::to_string(omp_get_thread_num()) + "\t Transitive Reduction Time: " + std::to_string(endTransRedTime - startTime) + "\n";
    std::cout << outputTransRedString;
#endif
}


void SMFunOriGrowlv2::computePartialSchedule(const SmInstance &instance, const bool *const transEdgeMask, unsigned *const node_to_proc, unsigned *const node_to_supstep,  const unsigned int startNode, const unsigned int endNode, unsigned &supstep) const {

#ifdef TIME_THREADS_FUN_GROWL_V2
    double startTime = omp_get_wtime();
#endif

    const unsigned N = endNode - startNode;
    if (N == 0) {
        return;
    }
    const unsigned P = instance.numberOfProcessors();

    const SM_csr * const csr_mat = instance.getMatrix().getCSR();
    const SM_csc * const csc_mat = instance.getMatrix().getCSC();

    unsigned finalReverseNode = std::max(startNode, 1U); // Node 0 has no parents

    // contracted graph reverse CSR with index starting at 1
    std::vector<unsigned> contractedWeights(1,0);
    std::vector<unsigned> contractedOuter(1,0);
    std::vector<unsigned> contractedInner(1,0);
    std::vector<unsigned> originalNode(1,0);
    std::map<unsigned, unsigned> contractedNode; // only for base nodes
    std::vector<unsigned> numContractedPrecedence(1,0);

    // Funnel
    std::vector<unsigned> graphNodeWeights(N);
    for (unsigned vert = startNode; vert < endNode; vert++) {
        graphNodeWeights[ vert - startNode ] = instance.getMatrix().nodeWorkWeight(vert);
    }
    auto median = graphNodeWeights.begin() + graphNodeWeights.size() / 2;
    std::nth_element(graphNodeWeights.begin(), median, graphNodeWeights.end());
    const unsigned maxWeightFunnel = static_cast<unsigned>( params.maxWeightMedianMultiplier * (static_cast<double>(graphNodeWeights[graphNodeWeights.size() / 2])) );

    std::vector<unsigned> nodeToPartAssignment(N, UINT_MAX);

    std::vector<unsigned> numReducedChildren(N, 0);
    for (unsigned vert = std::max(endNode, 1U) - 1; vert >= finalReverseNode; vert--) { // endNode can be 0 -> startNode = 0 -> finalReverseNode = 1 -> empty loop as should be the case
        for (unsigned parentIndex = (*csr_mat).outerIndexPtr()[vert + 1] - 2; parentIndex >= (*csr_mat).outerIndexPtr()[vert]; parentIndex--) {
            
            const unsigned parent = (*csr_mat).innerIndexPtr()[parentIndex];
            if (parent < startNode) break;

            if (transEdgeMask[parentIndex]) {
                numReducedChildren[ parent - startNode ]++;
            }
        }
    }

    std::vector<unsigned> undoDecrement;
    std::vector<unsigned> funnelQueue;
#ifdef FUNORIGROWLV2_INCREASE_QUEUEING
    std::vector<unsigned> queueReversingVec;
#endif

    const unsigned finalFunnelNode = std::max(startNode, 1U);
    for (unsigned baseVert = std::max(endNode, 1U) - 1; baseVert >= finalFunnelNode; baseVert--) {
        if (nodeToPartAssignment[ baseVert - startNode ] != UINT_MAX) continue;

        unsigned weightFunnel = 0;
        undoDecrement.clear();

        funnelQueue.clear();
        funnelQueue.push_back(baseVert);

        while (!funnelQueue.empty()) {
            const unsigned vert = funnelQueue.back();
            funnelQueue.pop_back();

            nodeToPartAssignment[ vert - startNode ] = baseVert;
            weightFunnel += (*csr_mat).outerIndexPtr()[vert + 1] - (*csr_mat).outerIndexPtr()[vert];

            if (vert == 0) break; // vertex 0 has no parents
            if (weightFunnel >= maxWeightFunnel) break;

            for (unsigned parentIndex = (*csr_mat).outerIndexPtr()[vert + 1] - 2; parentIndex >= (*csr_mat).outerIndexPtr()[vert]; parentIndex--) {
                if (!transEdgeMask[parentIndex]) continue;

                const unsigned parent = (*csr_mat).innerIndexPtr()[parentIndex];
                if (parent < startNode) break;

                const unsigned parentShifted = parent - startNode;
                undoDecrement.push_back(parentShifted);
                numReducedChildren[parentShifted]--;

                if (numReducedChildren[parentShifted] == 0) {
#ifdef FUNORIGROWLV2_INCREASE_QUEUEING
                    queueReversingVec.push_back(parent);
#else
                    funnelQueue.push_back(parent);
#endif
                }
            }
#ifdef FUNORIGROWLV2_INCREASE_QUEUEING
            for (auto rev_it = queueReversingVec.rbegin(); rev_it != queueReversingVec.rend(); rev_it++) {
                funnelQueue.push_back(*rev_it);
            }
            queueReversingVec.clear();
#endif
        }

        std::set<unsigned, std::greater<unsigned>> children;
        for (unsigned childIndex = (*csc_mat).outerIndexPtr()[baseVert] + 1; childIndex < (*csc_mat).outerIndexPtr()[baseVert + 1]; childIndex++) {
            const unsigned child = (*csc_mat).innerIndexPtr()[childIndex];
            if (child >= endNode) break;

            children.insert( nodeToPartAssignment[ child - startNode ] );
        }
        for (auto it = children.begin(); it != children.end(); it++) {
            const unsigned childContract = contractedNode.at(*it);
            contractedInner.push_back( childContract );
            numContractedPrecedence[childContract]++;
        }

        contractedNode.insert({baseVert, originalNode.size()});
        contractedOuter.push_back(contractedInner.size() - 1);
        contractedWeights.push_back(weightFunnel);
        originalNode.push_back(baseVert);
        numContractedPrecedence.push_back(0);

        for (const unsigned &ind : undoDecrement) {
            numReducedChildren[ind]++;
        }
    }

    if ( (endNode != 0) && (startNode == 0) && (nodeToPartAssignment[0] == UINT_MAX) ) { // if vertex 0 has not been assigned yet (was skipped earlier)
        nodeToPartAssignment[0] = 0;

        std::set<unsigned, std::greater<unsigned>> children;
        for (unsigned childIndex = (*csc_mat).outerIndexPtr()[0] + 1; childIndex < (*csc_mat).outerIndexPtr()[0 + 1]; childIndex++) {
            const unsigned child = (*csc_mat).innerIndexPtr()[childIndex];
            if (child >= endNode) break;

            children.insert( nodeToPartAssignment[ child - startNode ] );
        }
        for (auto it = children.begin(); it != children.end(); it++) {
            const unsigned childContract = contractedNode.at(*it);
            contractedInner.push_back( childContract );
            numContractedPrecedence[childContract]++;
        }

        contractedNode.insert({0, originalNode.size()});
        contractedOuter.push_back(contractedInner.size() - 1);
        contractedWeights.push_back(1);
        originalNode.push_back(0);
        numContractedPrecedence.push_back(0);
    }

    undoDecrement.clear();
    undoDecrement.shrink_to_fit();
    funnelQueue.clear();
    funnelQueue.shrink_to_fit();
#ifdef FUNORIGROWLV2_INCREASE_QUEUEING
    queueReversingVec.clear();
    queueReversingVec.shrink_to_fit();
#endif

#ifdef TIME_THREADS_FUN_GROWL_V2
    double endFunnelTime = omp_get_wtime();

    std::string padd = "";
    if (omp_get_thread_num() < 10) {
        padd = " ";
    }

    std::string outputFunnelString = "Thread: " + padd + std::to_string(omp_get_thread_num()) + "\t Funnel Time: " + std::to_string(endFunnelTime - startTime) + "\n";
    std::cout << outputFunnelString;

    std::string contractionComp = "Thread: " + padd + std::to_string(omp_get_thread_num()) + "\t Original Graph: " + std::to_string(N) + " Contracted Graph: " + std::to_string(originalNode.size() - 1) + "\n";
    std::cout << contractionComp;
#endif

    // Grow Local
    const unsigned numContractVertices = originalNode.size() - 1;
    std::vector<unsigned> contract_node_to_proc(numContractVertices + 1, UINT_MAX);
    std::vector<unsigned> contract_node_to_supstep(numContractVertices + 1, UINT_MAX);

    std::set<unsigned, std::greater<unsigned>> ready;
    
    std::vector<unsigned> futureReady;
    std::vector<unsigned> best_futureReady;

    std::vector<std::set<unsigned, std::greater<unsigned>>> procReady(P);
    std::vector<std::set<unsigned, std::greater<unsigned>>> best_procReady(P);

    for (unsigned node = numContractedPrecedence.size() - 1; node > 0; node--) {
        if (numContractedPrecedence[node] == 0) {
            ready.insert(node).first;
        }
    }



    std::vector<std::vector<unsigned>> new_assignments(P);
    std::vector<std::vector<unsigned>> best_new_assignments(P);


    const unsigned minWeightParallelCheck = params.syncCostMultiplierParallelCheck * instance.synchronisationCosts();
    const unsigned minSuperstepWeight = params.syncCostMultiplierMinSuperstepWeight * instance.synchronisationCosts();

    double desiredParallelism = static_cast<double>(P);

    unsigned total_assigned = 0;
    supstep = 0;

    while(total_assigned < numContractVertices) {
        unsigned limit = params.minSuperstepSize;
        double best_score = 0;
        double best_parallelism = 0;

        std::set<unsigned, std::greater<unsigned>>::iterator readyIter;
        std::set<unsigned, std::greater<unsigned>>::iterator bestReadyIter;

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
                contract_node_to_proc[chosen_node] = 0;
                new_total_assigned++;
                weight_limit += contractedWeights[chosen_node];

                for (unsigned i = contractedOuter[chosen_node]; i > contractedOuter[chosen_node - 1]; i--) {
                    const unsigned succ = contractedInner[i];

                    if ( contract_node_to_proc[succ] == UINT_MAX ) {
                        contract_node_to_proc[succ] = 0;
                    } else if ( contract_node_to_proc[succ] != 0 ) {
                        contract_node_to_proc[succ] = P;
                    }

                    numContractedPrecedence[succ]--;
                    if(numContractedPrecedence[succ] == 0) {
                        if( contract_node_to_proc[succ] == 0 ) {
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
                    } else if(readyIter != ready.end()) {
                        chosen_node = *readyIter;
                        readyIter++;
                    } else break;

                    new_assignments[proc].push_back(chosen_node);
                    contract_node_to_proc[chosen_node] = proc;
                    new_total_assigned++;
                    current_weight_assigned += contractedWeights[chosen_node];

                    for (unsigned i = contractedOuter[chosen_node]; i > contractedOuter[chosen_node - 1]; i--) {
                        const unsigned succ = contractedInner[i];

                        if ( contract_node_to_proc[succ] == UINT_MAX ) {
                            contract_node_to_proc[succ] = proc;
                        } else if ( contract_node_to_proc[succ] != proc ) {
                            contract_node_to_proc[succ] = P;
                        }

                        numContractedPrecedence[succ]--;
                        if(numContractedPrecedence[succ] == 0) {
                            if( contract_node_to_proc[succ] == proc ) {
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
                if(total_assigned + new_total_assigned == numContractVertices) {
                    accept_step = true;
                    continueSuperstepAttempts = false;
                }
            }

            if(total_assigned + new_total_assigned == numContractVertices) {
                continueSuperstepAttempts = false;
            }

            // undo proc assingments and predec increases in any case
            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    contract_node_to_proc[node] = UINT_MAX;
                }
            }

            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    for (unsigned i = contractedOuter[node]; i > contractedOuter[node - 1]; i--) {
                        const unsigned succ = contractedInner[i];
                        numContractedPrecedence[succ]++;
                    }
                }
            }

            for(unsigned proc = 0; proc < P; ++proc) {
                for(unsigned node : new_assignments[proc]) {
                    for (unsigned i = contractedOuter[node]; i > contractedOuter[node - 1]; i--) {
                        const unsigned succ = contractedInner[i];
                        contract_node_to_proc[succ] = UINT_MAX;
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
                contract_node_to_proc[node] = proc;
                contract_node_to_supstep[node] = supstep;
                ++total_assigned;

                for (unsigned i = contractedOuter[node]; i > contractedOuter[node - 1]; i--) {
                    const unsigned succ = contractedInner[i];
                    numContractedPrecedence[succ]--;
                }
            }
        }

        desiredParallelism = (0.3 * desiredParallelism) + (0.6 * best_parallelism) + (0.1 * static_cast<double>(P)); // weights should sum up to one

        ++supstep;
    }



    // Lifting of schedule
    for (unsigned i = 1; i < originalNode.size(); i++) {
        node_to_proc[ originalNode[i] ] = contract_node_to_proc[i];
    }

    for (unsigned vert = startNode; vert < endNode; vert++) {
        node_to_proc[vert] = node_to_proc[ nodeToPartAssignment[ vert - startNode ] ];
    }

    for (unsigned i = 1; i < originalNode.size(); i++) {
        node_to_supstep[ originalNode[i] ] = contract_node_to_supstep[i];
    }

    for (unsigned vert = startNode; vert < endNode; vert++) {
        node_to_supstep[vert] = node_to_supstep[ nodeToPartAssignment[ vert - startNode ] ];
    }


#ifdef TIME_THREADS_FUN_GROWL_V2
    double endGrowTime = omp_get_wtime();
    std::string outputGrowString = "Thread: " + padd + std::to_string(omp_get_thread_num()) + "\t Growl Time: " + std::to_string(endGrowTime - endFunnelTime) + "\n";
    std::cout << outputGrowString;

    std::string outputTotalString = "Thread: " + padd + std::to_string(omp_get_thread_num()) + "\t Total Time: " + std::to_string(endGrowTime - startTime) + "\n";
    std::cout << outputTotalString;
#endif

}


void SMFunOriGrowlv2::incrementScheduleSupersteps(unsigned *const node_to_supstep, const unsigned startNode, const unsigned endNode, const unsigned incr) const {
    for (unsigned node = startNode; node < endNode; node++) {
        node_to_supstep[node] += incr;
    }
}

std::pair<RETURN_STATUS, SmSchedule> SMFunOriGrowlv2::computeScheduleParallel(const SmInstance &instance, unsigned numThreads) const {

    const unsigned N = instance.numberOfVertices();
    const unsigned nnz = instance.getMatrix().getCSR()->nonZeros();

    const SM_csr * const csr_mat = instance.getMatrix().getCSR();

    bool *const transEdgeMask = static_cast<bool *>(aligned_alloc(CACHE_LINE_SIZE, ((nnz + NUM_BOOL_IN_CACHE_LINE - 1) / NUM_BOOL_IN_CACHE_LINE) * NUM_BOOL_IN_CACHE_LINE * sizeof(bool) ));
    for (unsigned i = 0; i < nnz; i++) {
        transEdgeMask[i] = true;
    }

    unsigned *const node_to_proc = static_cast<unsigned *>(aligned_alloc(CACHE_LINE_SIZE, ((N + NUM_UNSIGNED_IN_CACHE_LINE - 1) / NUM_UNSIGNED_IN_CACHE_LINE) * NUM_UNSIGNED_IN_CACHE_LINE * sizeof(unsigned) ));
    for (unsigned i = 0; i < N; i++) {
        node_to_proc[i] = UINT_MAX;
    }

    unsigned *const node_to_supstep = static_cast<unsigned *>(aligned_alloc(CACHE_LINE_SIZE, ((N + NUM_UNSIGNED_IN_CACHE_LINE - 1) / NUM_UNSIGNED_IN_CACHE_LINE) * NUM_UNSIGNED_IN_CACHE_LINE * sizeof(unsigned) ));
    for (unsigned i = 0; i < N; i++) {
        node_to_supstep[i] = UINT_MAX;
    }

    unsigned numTransRedThreads = omp_get_max_threads();
    numTransRedThreads = std::max(numTransRedThreads, 1U);
    if (N < 1000) {
        numTransRedThreads = 1;
    }

    unsigned numEdgesPerThreadTransRed = nnz / numTransRedThreads;
    std::vector<unsigned> startNodesTransRed;
    startNodesTransRed.reserve(numTransRedThreads + 1);
    unsigned startNodeTransRed = 0;
    const unsigned finalEdgeInd = csr_mat->outerIndexPtr()[N] - 1;

    for (unsigned thr = 0; thr < numTransRedThreads; thr++) {
        startNodesTransRed.push_back(startNodeTransRed);
        
        unsigned edgeInd = csr_mat->outerIndexPtr()[startNodeTransRed];
        edgeInd += numEdgesPerThreadTransRed;
        edgeInd = std::min(edgeInd, finalEdgeInd);
        startNodeTransRed = csr_mat->innerIndexPtr()[edgeInd];

        while ( (edgeInd < finalEdgeInd) && (startNodeTransRed < csr_mat->innerIndexPtr()[edgeInd + 1]) ) {
            edgeInd++;
            startNodeTransRed = csr_mat->innerIndexPtr()[edgeInd];
        }
    }
    startNodesTransRed.push_back(N);

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

unsigned threadNumParallelRegion = std::max(numThreads, numTransRedThreads);
#pragma omp parallel num_threads(threadNumParallelRegion) default(none) shared(instance, transEdgeMask, node_to_proc, node_to_supstep, superstepsThread, supstepIncr, numThreads, numTransRedThreads, startNodes, startNodesTransRed, incr)
{
#pragma omp for schedule(static, 1)
    for (unsigned thr = 0; thr < numTransRedThreads; thr++) {
        computeTransitiveReduction(instance, transEdgeMask, startNodesTransRed[thr], startNodesTransRed[thr + 1]); // The load balancing can probably be improved - right now balancing number of edges, but complexity is degree^2 per vertex
    }

#pragma omp for schedule(static, 1)
    for (unsigned thr = 0; thr < numThreads; thr++) {
        computePartialSchedule(instance, transEdgeMask, node_to_proc, node_to_supstep, startNodes[thr], startNodes[thr + 1], superstepsThread[thr * UNSIGNED_PADDING]);
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

    free(transEdgeMask);
    free(node_to_proc);
    free(node_to_supstep);

    return {SUCCESS, schedule};
}