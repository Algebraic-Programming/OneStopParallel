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

// This scheduler is based on the idea and early implementation by Aikaterini Karanasiou

#include <algorithm>
#include <stdexcept>

#include "algorithms/GreedySchedulers/GreedyLayers.hpp"

std::pair<RETURN_STATUS, BspSchedule> GreedyLayers::computeSchedule(const BspInstance &instance) {

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                        std::vector<unsigned>(instance.numberOfVertices()));

    std::deque<int> sources;
    for (const auto &v : G.sourceVertices())
        sources.push_back(v);

    std::vector<unsigned> nrPredecDone(N, 0);
    
    int stepIdx = 0;
    while(!sources.empty())
    {
        // assign current set of source nodes
        int next_proc = 0;
        if(stepIdx == 0)
        {
            // clustering in first superstep
            std::vector<std::vector<int> > clusters = FormClusters(sources, G);
            for(size_t i = 0; i < clusters.size(); ++i)
            {
                for(int node : clusters[i])
                {
                    schedule.setAssignedProcessor(node, next_proc);
                    schedule.setAssignedSuperstep(node, stepIdx);
                }
                next_proc = (next_proc + 1) % params_p;
            }
        }
        else
        {
            // weight-based round robin for all other supersteps
            std::set<intPair> sources_by_weight;
            for(int node : sources)
                sources_by_weight.emplace(G.nodeWorkWeight(node), node);
            
            for(intPair weight_and_node : sources_by_weight)
            {
                schedule.setAssignedProcessor(weight_and_node.b, next_proc);
                schedule.setAssignedSuperstep(weight_and_node.b, stepIdx);
                next_proc = (next_proc + 1) % params_p;
            }
        }

        // collect next layer of source nodes
        std::list<int> new_sources;
        for(int node : sources)
            for(const auto &succ : G.children(node))
                if((++nrPredecDone[succ]) == G.numberOfParents(succ))
                    new_sources.push_back(succ);

        // add rider nodes to current layer
        std::deque<int> sources_remaining;
        for(int node : new_sources)
        {
            std::set<int> pred_processors;
            for(const auto &pred : G.parents(node))
                pred_processors.insert(schedule.assignedProcessor(pred));
            
            if(pred_processors.size()==1)
            {
                schedule.setAssignedProcessor(node, *pred_processors.begin());
                schedule.setAssignedSuperstep(node, stepIdx);
                for(const auto &succ : G.children(node))
                    if((++nrPredecDone[succ]) == G.numberOfParents(succ))
                        new_sources.push_back(succ);
            }
            else
                sources_remaining.push_back(node);

        }
        sources = sources_remaining;
        ++stepIdx;
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
}


std::vector<std::vector<int> > GreedyLayers::FormClusters(const std::deque<int>& sources, const ComputationalDag& G) const
{
    std::map<int, std::vector<int> > In, Out; // bipartite relation between sources and immediate successors
    std::map<int, int > SuccDegree;
    std::map<int, int > clusterID;
    for(int node : sources)
    {
        clusterID[node] = -1;
        Out[node] = std::vector<int>();
        for(const auto &succ : G.children(node))
            if(SuccDegree.find(succ)==SuccDegree.end())
                SuccDegree[succ] = 1;
            else
                SuccDegree[succ] += 1;
    }
    for(int node : sources)
        for(const auto &succ : G.children(node))
        {
            if(SuccDegree[succ] == 1)
                continue;
            if(In.find(succ)==In.end())
                In[succ] = std::vector<int>();

            Out[node].push_back(succ);
            In[succ].push_back(node);
        }

    int nr_clusters = 0;
    std::set<intPair> successors_sorted;
    for(auto it = In.begin(); it != In.end(); ++it)
        successors_sorted.emplace(it->second.size(), it->first);
    
    for(intPair indegree_and_successor : successors_sorted)
    {
        int node = indegree_and_successor.b;
        std::set<int> pred_clusters;
        for(int pred : In[node])
            if(clusterID[pred]!=-1)
                pred_clusters.insert(clusterID[pred]);
        
        if(pred_clusters.size()==0)
        {
            for(int pred : In[node])
                clusterID[pred] = nr_clusters;
            ++nr_clusters;
        }
        else if(pred_clusters.size()==1)
            for(int pred : In[node])
                clusterID[pred] = *pred_clusters.begin();
    }

    std::vector<int> cluster_sizes(sources.size(), 0);
    for(int node : sources)
        if(clusterID[node]!=-1)
            ++cluster_sizes[clusterID[node]];

    for(auto it = In.begin(); it != In.end(); ++it)
    {
        // int succ = it->first;
        std::set<int> pred_clusters;
        for(int pred : it->second)
            if(clusterID[pred]!=-1)
                pred_clusters.insert(clusterID[pred]);

        if(pred_clusters.empty())
            std::cout<<"Error: this should not happen."<<std::endl;
        
        int best = *pred_clusters.begin();
        for(int clus : pred_clusters)
            if(cluster_sizes[clus] < cluster_sizes[best])
                best = clus;
        
        for(int pred : it->second)
            if(clusterID[pred]==-1)
            {
                clusterID[pred] = best;
                ++cluster_sizes[best];
            }
    }

    for(int node : sources)
        if(clusterID[node]==-1)
        {
            clusterID[node] = nr_clusters;
            ++nr_clusters;
        }
    
    std::vector<std::vector<int> > clusters(nr_clusters);
    for(int node : sources)
        clusters[clusterID[node]].push_back(node);

    return clusters;
}