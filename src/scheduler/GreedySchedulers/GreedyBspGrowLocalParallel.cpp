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

#include "scheduler/GreedySchedulers/GreedyBspGrowLocalParallel.hpp"
#include <algorithm>
#include <stdexcept>



std::pair<RETURN_STATUS, BspSchedule> GreedyBspGrowLocalParallel::computeSchedule(const BspInstance &instance) {
    unsigned numThreads = static_cast<unsigned>(std::sqrt( static_cast<double>((instance.numberOfVertices() / 1000000)))) + 1;
    
    return computeScheduleParallel(instance, numThreads);
};


void GreedyBspGrowLocalParallel::computePartialSchedule(const BspInstance &instance, std::vector<unsigned> &node_to_proc, std::vector<unsigned> &node_to_supstep, const std::vector<VertexType> &topOrder, const std::vector<size_t> &topOrdPos, const size_t startNode, const size_t endNode, unsigned &supstep) {

    const size_t N = endNode - startNode;
    const unsigned P = instance.numberOfProcessors();
    const ComputationalDag& G = instance.getComputationalDag();

    std::set<unsigned> ready;
    std::vector<std::set<unsigned>::iterator> place_in_ready(N);

    std::vector<unsigned> predec(N, 0);

    for(size_t nodePos = startNode; nodePos < endNode; nodePos++) {
        size_t index = nodePos - startNode;
        for (const VertexType &par : G.parents(topOrder[nodePos])) {
            if (topOrdPos[par] >= startNode && topOrdPos[par] < endNode) {
                predec[index]++;
            }
        }
        if (predec[index] == 0) {
            place_in_ready[index] = ready.insert(topOrder[nodePos]).first;
        }
    }

    unsigned total_assigned = 0;
    supstep = 0;

    while(total_assigned < N)
    {
        unsigned limit = minimum_superstep_size;
        double parallelization_rate = 0;

        // use two vectors alternatingly to avoid some copying
        unsigned parity = 0;
        std::vector<std::vector<std::vector<unsigned> > > new_assignments(2);
        std::vector<std::vector<unsigned> > new_ready(2);
        std::vector<std::vector<unsigned> > predec_decreased(2);

        while(true)
        {
            new_assignments[parity].clear();
            new_assignments[parity].resize(P);
            new_ready[parity].clear();
            predec_decreased[parity].clear();

            std::vector<std::set<unsigned> > procReady(P);
            std::set<unsigned> allReady = ready;

            unsigned new_total_assigned = 0;
            unsigned weight_limit = 0, total_weight_assigned = 0;

            for(unsigned proc = 0; proc < P; ++proc)
            {
                unsigned current_weight_assigned = 0;
                while((proc == 0 && new_assignments[parity][proc].size() < limit) || (proc > 0 && current_weight_assigned < weight_limit))
                {
                    unsigned chosen_node = UINT_MAX;
                    if(!procReady[proc].empty())
                    {
                        chosen_node = *procReady[proc].begin();
                        procReady[proc].erase(procReady[proc].begin());
                    }
                    else if(!allReady.empty())
                    {
                        chosen_node = *allReady.begin();
                        allReady.erase(allReady.begin());
                    }
                    else break;

                    if(proc > 0 && current_weight_assigned + G.nodeWorkWeight(chosen_node) > weight_limit)
                        break;

                    new_assignments[parity][proc].push_back(chosen_node);
                    node_to_proc[chosen_node] = proc;
                    ++new_total_assigned;
                    current_weight_assigned += G.nodeWorkWeight(chosen_node);

                    for (const auto &succ : G.children(chosen_node))
                    {
                        if (topOrdPos[succ] < startNode || topOrdPos[succ] >= endNode) {
                            continue;
                        }
                        size_t succIndex = topOrdPos[succ] - startNode;
                        predec[succIndex]--;
                        if(predec[succIndex] == 0) {
                            new_ready[parity].push_back(succ);

                            bool canAdd = true;
                            for(const auto &pred : G.parents(succ)) {
                                if (topOrdPos[pred] < startNode || topOrdPos[pred] >= endNode) {
                                    continue;
                                }
                                if(node_to_proc[pred] != proc && node_to_supstep[pred] == UINT_MAX) {
                                    canAdd = false;
                                    break;
                                }
                            }

                            if(canAdd) {
                                procReady[proc].insert(succ);
                            }
                        }
                        predec_decreased[parity].push_back(succ);
                    }
                }
                if(proc == 0)
                    weight_limit = current_weight_assigned;
                
                total_weight_assigned += current_weight_assigned;
            }

            bool accept_step = (limit == minimum_superstep_size);

            if(limit == minimum_superstep_size)
                //parallelization_rate = (double) new_total_assigned / ((double) P * (double)limit);
                parallelization_rate = (double) total_weight_assigned / ((double) P * (double)weight_limit);

            //if((double)new_total_assigned + 0.0001 >= (double)limit * (double) P * parallelization_rate * lower_limit_parallelization)
            //    accept_step = true;

            if((double)total_weight_assigned + 0.0001 >= (double)weight_limit * (double) P * parallelization_rate * lower_limit_parallelization)
                accept_step = true;
            
            if(limit > maximum_imbalanced_superstep_size && parallelization_rate < 0.9)
                accept_step = false;

            if(total_assigned + new_total_assigned == N)
            {
                accept_step = true;
                break;
            }

            // undo proc assingments and predec increases in any case
            for(unsigned proc = 0; proc < P; ++proc)
                for(unsigned node : new_assignments[parity][proc])
                    node_to_proc[node] = UINT_MAX;

            for(const unsigned &succ : predec_decreased[parity])
                predec[topOrdPos[succ] - startNode]++;

            parity = 1 - parity;

            if(accept_step)
                limit = (int) ceil((double)limit * grow_ratio);      
            else
                break;
        }

        // apply best iteration
        for(const unsigned &node : new_ready[parity])
            place_in_ready[topOrdPos[node] - startNode] = ready.insert(node).first;

        for(unsigned proc = 0; proc < P; ++proc)
            for(const unsigned &node : new_assignments[parity][proc])
            {
                node_to_proc[node] = proc;
                node_to_supstep[node] = supstep;
                ready.erase(place_in_ready[topOrdPos[node] - startNode]);
                ++total_assigned;
            }
        
        for(const unsigned &node : predec_decreased[parity])
            predec[topOrdPos[node] - startNode]--;

        ++supstep;
    }
};

void GreedyBspGrowLocalParallel::incrementScheduleSupersteps(std::vector<unsigned> &node_to_supstep, const std::vector<VertexType> &topOrder, const size_t startNode, const size_t endNode, unsigned incr) {
    for (size_t nodePos = startNode; nodePos < endNode; nodePos++) {
        node_to_supstep[ topOrder[nodePos] ] += incr;
    }
};


std::pair<RETURN_STATUS, BspSchedule> GreedyBspGrowLocalParallel::computeScheduleParallel(const BspInstance &instance, unsigned numThreads) {

    const unsigned N = instance.numberOfVertices();
    const unsigned P = instance.numberOfProcessors();
    const ComputationalDag& G = instance.getComputationalDag();

    std::vector<unsigned> node_to_proc(N, UINT_MAX), node_to_supstep(N, UINT_MAX);
    
    std::vector<VertexType> topOrder = G.GetTopOrder(ComputationalDag::TOP_SORT_ORDER::MINIMAL_NUMBER);
    std::vector<size_t> topOrdPos(N);
    for (size_t ind = 0; ind < topOrder.size(); ind++) {
        topOrdPos[ topOrder[ind] ] = ind;
    }

    unsigned numNodesPerThread = N / numThreads;
    std::vector<size_t> startNodes;
    startNodes.reserve(numThreads + 1);
    size_t startNode = 0;
    for (unsigned thr = 0; thr < numThreads; thr++) {
        startNodes.push_back(startNode);
        startNode += numNodesPerThread;
    }
    startNodes.push_back(N);

    std::vector<unsigned> superstepsThread(numThreads,0);

    std::vector<std::thread> scheduleThreads(numThreads);

    for (unsigned thr = 0; thr < numThreads; thr++) {
        scheduleThreads[thr] = std::thread(&GreedyBspGrowLocalParallel::computePartialSchedule, this, std::ref(instance), std::ref(node_to_proc), std::ref(node_to_supstep), std::ref(topOrder), std::ref(topOrdPos), startNodes[thr], startNodes[thr + 1], std::ref(superstepsThread[thr]));
    }

    for (unsigned thr = 0; thr < numThreads; thr++) {
        scheduleThreads[thr].join();
    }

    std::vector<unsigned> supstepIncr(numThreads, 0);
    unsigned incr = 0;
    for (unsigned thr = 0; thr < numThreads; thr++) {
        supstepIncr[thr] = incr;
        incr += superstepsThread[thr];
    }

    for (unsigned thr = 0; thr < numThreads; thr++) {
        scheduleThreads[thr] = std::thread(&GreedyBspGrowLocalParallel::incrementScheduleSupersteps, this, std::ref(node_to_supstep), std::ref(topOrder), startNodes[thr], startNodes[thr + 1], supstepIncr[thr]);
    }

    for (unsigned thr = 0; thr < numThreads; thr++) {
        scheduleThreads[thr].join();
    }

    BspSchedule schedule(instance, node_to_proc, node_to_supstep);
    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};