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

#include "scheduler/GreedySchedulers/GreedyBspGrowLocal.hpp"
#include <algorithm>
#include <stdexcept>

std::pair<RETURN_STATUS, BspSchedule> GreedyBspGrowLocal::computeSchedule(const BspInstance &instance) {

    const unsigned N = instance.numberOfVertices();
    const unsigned P = instance.numberOfProcessors();
    const ComputationalDag& G = instance.getComputationalDag();

    std::vector<unsigned> node_to_proc(N, UINT_MAX), node_to_supstep(N, UINT_MAX);
    std::set<unsigned> ready;
    std::vector<std::set<unsigned>::iterator> place_in_ready(N);

    std::vector<unsigned> predec(N, 0);

    for(unsigned node = 0; node < N; ++node)
        if(G.numberOfParents(node) == 0)
            place_in_ready[node] = ready.insert(node).first;

    unsigned supstep = 0, total_assigned = 0;
    while(total_assigned < N)
    {
        unsigned limit = minimum_superstep_size;
        double parallelization_rate = 0;

        // use two vectors alternatingly to avoid some copying
        unsigned parity = 0;
        std::vector<std::vector<std::vector<unsigned> > > new_assignments(2);
        std::vector<std::vector<unsigned> > new_ready(2);
        std::vector<std::vector<unsigned> > predec_increased(2);

        while(true)
        {
            new_assignments[parity].clear();
            new_assignments[parity].resize(P);
            new_ready[parity].clear();
            predec_increased[parity].clear();

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
                        ++predec[succ];
                        if(predec[succ]==G.numberOfParents(succ))
                        {
                            new_ready[parity].push_back(succ);

                            bool canAdd = true;
                            for(const auto &pred : G.parents(succ))
                                if(node_to_proc[pred] != proc && node_to_supstep[pred] == UINT_MAX)
                                {
                                    canAdd= false;
                                    break;
                                }

                            if(canAdd)
                                procReady[proc].insert(succ);
                        }
                        predec_increased[parity].push_back(succ);

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

            for(const unsigned &succ : predec_increased[parity])
                --predec[succ];

            parity = (parity + 1) % 2;

            if(accept_step)
                limit = (int) ceil((double)limit * grow_ratio);      
            else
                break;
        }

        // apply best iteration
        for(const unsigned &node : new_ready[parity])
            place_in_ready[node] = ready.insert(node).first;

        for(unsigned proc = 0; proc < P; ++proc)
            for(const unsigned &node : new_assignments[parity][proc])
            {
                node_to_proc[node] = proc;
                node_to_supstep[node] = supstep;
                ready.erase(place_in_ready[node]);
                ++total_assigned;
            }
        
        for(const unsigned &node : predec_increased[parity])
            ++predec[node];

        ++supstep;
    }

    BspSchedule schedule(instance, node_to_proc, node_to_supstep);
    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};


std::pair<RETURN_STATUS, BspSchedule> GreedyBspGrowLocal::computeSchedule_csr(const BspInstance_csr &instance_csr, const BspInstance &instance) {

    const unsigned N = instance_csr.numberOfVertices();
    const unsigned P = instance_csr.numberOfProcessors();
    const auto& G = instance_csr.getComputationalDag();

    std::vector<unsigned> node_to_proc(N, UINT_MAX), node_to_supstep(N, UINT_MAX);
    std::set<unsigned> ready;
    std::vector<std::set<unsigned>::iterator> place_in_ready(N);

    std::vector<unsigned> predec(N, 0);

    for(VertexType node = 0; node < N; ++node)
        if(boost::in_degree(node, G) == 0)
            place_in_ready[node] = ready.insert(node).first;

    unsigned supstep = 0, total_assigned = 0;
    while(total_assigned < N)
    {
        unsigned limit = minimum_superstep_size;
        double parallelization_rate = 0;

        // use two vectors alternatingly to avoid some copying
        unsigned parity = 0;
        std::vector<std::vector<std::vector<unsigned> > > new_assignments(2);
        std::vector<std::vector<unsigned> > new_ready(2);
        std::vector<std::vector<unsigned> > predec_increased(2);

        while(true)
        {
            new_assignments[parity].clear();
            new_assignments[parity].resize(P);
            new_ready[parity].clear();
            predec_increased[parity].clear();

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
                    
                    if(proc > 0 && current_weight_assigned + G[chosen_node].workWeight > weight_limit)
                        break;

                    new_assignments[parity][proc].push_back(chosen_node);
                    node_to_proc[chosen_node] = proc;
                    ++new_total_assigned;
                    current_weight_assigned += G[chosen_node].workWeight;

                for (const auto &out_edge : boost::make_iterator_range(boost::out_edges(static_cast<VertexType>(chosen_node), G))) {
                    const VertexType succ = boost::target(out_edge, G);
                    //for (const auto &succ : G.children(chosen_node))
                    //{
                        ++predec[succ];
                        if(predec[succ]== boost::in_degree(succ, G))
                        {
                            new_ready[parity].push_back(succ);

                            bool canAdd = true;

                            for (const auto &edge : boost::make_iterator_range(boost::in_edges(succ, G))) {
                                const VertexType pred = boost::source(edge, G);
                                // for(const auto &pred : G.parents(succ))
                                if(node_to_proc[pred] != proc && node_to_supstep[pred] == UINT_MAX)
                                {
                                    canAdd= false;
                                    break;
                                }
                            }
                            if(canAdd)
                                procReady[proc].insert(succ);
                        }
                        predec_increased[parity].push_back(succ);

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
                for(const unsigned &node : new_assignments[parity][proc])
                    node_to_proc[node] = UINT_MAX;

            for(const unsigned &succ : predec_increased[parity])
                --predec[succ];

            parity = (parity + 1) % 2;

            if(accept_step)
                limit = (int) ceil((double)limit * grow_ratio);      
            else
                break;
        }

        // apply best iteration
        for(const unsigned &node : new_ready[parity])
            place_in_ready[node] = ready.insert(node).first;

        for(unsigned proc = 0; proc < P; ++proc)
            for(const unsigned &node : new_assignments[parity][proc])
            {
                node_to_proc[node] = proc;
                node_to_supstep[node] = supstep;
                ready.erase(place_in_ready[node]);
                ++total_assigned;
            }
        
        for(const unsigned &node : predec_increased[parity])
            ++predec[node];

        ++supstep;
    }

    BspSchedule schedule(instance, node_to_proc, node_to_supstep);
    // schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};