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


#include "scheduler/PebblingILP/AuxiliaryForPartialILP/PebblingPartialILP.hpp"
#include <stdexcept>


std::pair<RETURN_STATUS, BspMemSchedule> PebblingPartialILP::computePebbling(const BspInstance &instance){

    if(!BspMemSchedule::hasValidSolution(instance))
        return {ERROR, BspMemSchedule()};
    
    // STEP 1: divide DAG acyclicly with partitioning ILP

    AcyclicDagDivider dag_divider;
    dag_divider.setMinAndMaxSize({minPartitionSize, maxPartitionSize});
    std::vector<unsigned> assignment_to_parts = dag_divider.computePartitioning(instance);
    unsigned nr_parts = *std::max_element(assignment_to_parts.begin(), assignment_to_parts.end()) + 1;

    // TODO remove source nodes before this?
    ComputationalDag contracted_dag = contractByPartition(instance, assignment_to_parts);

    // STEP 2: develop high-level multischedule on parts

    BspInstance contracted_instance(contracted_dag, instance.getArchitecture(), instance.getNodeProcessorCompatibilityMatrix());

    SubproblemMultiScheduling multi_scheduler;
    std::vector<std::set<unsigned> > processors_to_parts_and_types = multi_scheduler.computeMultiSchedule(contracted_instance).second;

    std::vector<std::set<unsigned> > processors_to_parts(nr_parts);
    for(unsigned part = 0; part < nr_parts; ++part)
        for(unsigned type = 0; type < instance.getComputationalDag().getNumberOfNodeTypes(); ++type)
            if(part_and_nodetype_to_new_index.find({part, type}) != part_and_nodetype_to_new_index.end())
            {
                unsigned new_index = part_and_nodetype_to_new_index[{part, type}];
                for(unsigned proc : processors_to_parts_and_types[new_index])
                    processors_to_parts[part].insert(proc);
            }

    // AUX: check for isomorphism

    // create set of nodes & external sources for all parts, and the nodes that need to have blue pebble at the end
    std::vector<std::set<unsigned> > nodes_in_part(nr_parts), extra_sources(nr_parts), needs_blue_at_end(nr_parts);
    std::vector<std::map<unsigned, unsigned> > original_node_id(nr_parts);
    std::vector<std::map<unsigned, unsigned> > original_proc_id(nr_parts);
    for(unsigned node = 0; node < instance.numberOfVertices(); ++node)
    {
        if(instance.getComputationalDag().numberOfParents(node) > 0)
            nodes_in_part[assignment_to_parts[node]].insert(node);
        else
            extra_sources[assignment_to_parts[node]].insert(node);
        for (const auto &pred : instance.getComputationalDag().parents(node))
            if(assignment_to_parts[node] != assignment_to_parts[pred])
                extra_sources[assignment_to_parts[node]].insert(pred);

        for (const auto &succ : instance.getComputationalDag().children(node))
            if(assignment_to_parts[node] != assignment_to_parts[succ])
                needs_blue_at_end[assignment_to_parts[node]].insert(node);
        
        if(instance.getComputationalDag().numberOfChildren(node) == 0)
            needs_blue_at_end[assignment_to_parts[node]].insert(node);
    }

    std::vector<ComputationalDag> subDags;
    for(unsigned part = 0; part < nr_parts; ++part)
    {
        subDags.push_back(instance.getComputationalDag().createInducedSubgraph(nodes_in_part[part], extra_sources[part]));
        
        // set source nodes to a new type, so that they are compatible with any processor
        unsigned artificial_type_for_sources = subDags.back().getNumberOfNodeTypes();
        for(unsigned node_idx = 0; node_idx < extra_sources[part].size(); ++node_idx)
            subDags.back().setNodeType(node_idx, artificial_type_for_sources);
        subDags.back().updateNumberOfNodeTypes();
    }
    
    std::vector<ComputationalDag> subDagsWithoutExternalSources = instance.getComputationalDag().createInducedSubgraphs(assignment_to_parts);

    std::vector<unsigned> isomorphicTo(nr_parts, UINT_MAX);

    std::cout<<"Number of parts: "<<nr_parts<<std::endl;

    for(unsigned part = 0; part < nr_parts; ++part)
        for(unsigned other_part = part + 1; other_part < nr_parts; ++other_part)
        {
            if(isomorphicTo[other_part] < UINT_MAX)
                continue;

            bool isomorphic = true;
            if(!subDags[part].checkOrderedIsomorphism(subDags[other_part]))
                continue;
            
            std::vector<unsigned> proc_assigned_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
            std::vector<unsigned> other_proc_assigned_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
            for(unsigned proc : processors_to_parts[part])
                ++proc_assigned_per_type[instance.getArchitecture().processorType(proc)];
            for(unsigned proc : processors_to_parts[other_part])
                ++other_proc_assigned_per_type[instance.getArchitecture().processorType(proc)];

            for(unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type)
                if(proc_assigned_per_type[proc_type] != other_proc_assigned_per_type[proc_type])
                    isomorphic = false;
            
            if(!isomorphic)
                continue;

            isomorphicTo[other_part] = part;
            std::cout<<"Part "<<other_part<<" is isomorphic to "<<part<<std::endl;
        }

    // PART 3: solve a small ILP for each part
    std::vector<std::set<unsigned> > in_fast_mem(instance.numberOfProcessors());
    std::vector<BspMemSchedule> pebbling(nr_parts);
    std::vector<BspArchitecture> subArch(nr_parts);
    std::vector<BspInstance> subInstance(nr_parts);

    // to handle the initial memory content for isomorphic parts
    std::vector<std::vector<std::set<unsigned> > > has_reds_in_beginning(nr_parts, std::vector<std::set<unsigned> >(instance.numberOfProcessors()));

    for(unsigned part = 0; part < nr_parts; ++part)
    {
        std::cout<<"part "<<part<<std::endl;

        // set up sub-DAG
        ComputationalDag& subDag = subDags[part];
        std::map<unsigned, unsigned> local_id;
        unsigned node_idx = 0;
        for(unsigned node : extra_sources[part])
        {
            local_id[node] = node_idx;
            original_node_id[part][node_idx] = node;
            ++node_idx;
        }
        for(unsigned node : nodes_in_part[part])
        {
            local_id[node] = node_idx;
            original_node_id[part][node_idx] = node;
            ++node_idx;
        }
        
        std::set<unsigned> needs_blue_at_end;
        for(unsigned node : nodes_in_part[part])
        {
            for (const auto &succ : instance.getComputationalDag().children(node))
                if(assignment_to_parts[node] != assignment_to_parts[succ])
                    needs_blue_at_end.insert(local_id[node]);
            
            if(instance.getComputationalDag().numberOfChildren(node) == 0)
                needs_blue_at_end.insert(local_id[node]);
        }

        // set up sub-architecture
        subArch[part].setNumberOfProcessors(processors_to_parts[part].size());
        unsigned proc_index = 0;
        for(unsigned proc : processors_to_parts[part])
        {
            subArch[part].setProcessorType(proc_index, instance.getArchitecture().processorType(proc));
            subArch[part].setMemoryBound(instance.getArchitecture().memoryBound(proc), proc_index);
            original_proc_id[part][proc_index] = proc;
            ++proc_index;
        }
        subArch[part].setCommunicationCosts(instance.getArchitecture().communicationCosts());
        subArch[part].setSynchronisationCosts(instance.getArchitecture().synchronisationCosts());
        // no NUMA parameters for now

        // skip if isomorphic to previous part
        if(isomorphicTo[part] < UINT_MAX)
        {
            pebbling[part] = pebbling[isomorphicTo[part]];
            has_reds_in_beginning[part] = has_reds_in_beginning[isomorphicTo[part]];
            continue;
        }

        // set node-processor compatibility matrix
        std::vector<std::vector<bool> > comp_matrix = instance.getNodeProcessorCompatibilityMatrix();
        comp_matrix.emplace_back(instance.getArchitecture().getNumberOfProcessorTypes(), true);
        subInstance[part] = BspInstance(subDag, subArch[part], comp_matrix);
        
        // currently we only allow the input laoding scenario - the case where this is false is unmaintained/untested
        bool need_to_load_inputs = true;

        // keep in fast memory what's relevant, remove the rest
        for(unsigned proc = 0; proc < processors_to_parts[part].size(); ++proc)
        {
            has_reds_in_beginning[part][proc].clear();
            std::set<unsigned> new_content_fast_mem;
            for(unsigned node : in_fast_mem[original_proc_id[part][proc]])
                if(local_id.find(node) != local_id.end())
                {
                    has_reds_in_beginning[part][proc].insert(local_id[node]);
                    new_content_fast_mem.insert(node);
                }

            in_fast_mem[original_proc_id[part][proc]] = new_content_fast_mem;
        }

        // heuristic solution for baseline
        BspMemSchedule heuristic_pebbling;
        GreedyBspFillupScheduler greedy_scheduler;
        BspSchedule bsp_herustic = greedy_scheduler.computeSchedule(subInstance[part]).second;
        
        std::set<unsigned> extra_source_ids;
        for(unsigned idx = 0; idx < extra_sources[part].size(); ++idx)
            extra_source_ids.insert(idx);

        heuristic_pebbling.setNeedToLoadInputs(true);
        heuristic_pebbling.SetExternalSources(extra_source_ids);
        heuristic_pebbling.SetNeedsBlueAtEnd(needs_blue_at_end);
        heuristic_pebbling.SetHasRedInBeginning(has_reds_in_beginning[part]);
        heuristic_pebbling.ConvertFromBsp(bsp_herustic, BspMemSchedule::CACHE_EVICTION_STRATEGY::FORESIGHT);      

        heuristic_pebbling.removeEvictStepsFromEnd();
        pebbling[part] = heuristic_pebbling;
        unsigned heuristicCost = asynchronous ? heuristic_pebbling.computeAsynchronousCost() : heuristic_pebbling.computeCost();

        if(!heuristic_pebbling.isValid())
            std::cout<<"ERROR: Pebbling heuristic INVALID!"<<std::endl;

        // solution with subILP
        MultiProcessorPebbling mpp;
        mpp.setVerbose(verbose);
        mpp.setTimeLimitSeconds(time_seconds_for_subILPs);
        mpp.setMaxTime(2*maxPartitionSize); // just a heuristic choice, does not guarantee feasibility!
        mpp.setNeedsBlueAtEnd(needs_blue_at_end);
        mpp.setNeedToLoadInputs(need_to_load_inputs);
        mpp.setHasRedInBeginning(has_reds_in_beginning[part]);

        std::pair<RETURN_STATUS, BspMemSchedule> ILP_result;
        ILP_result = mpp.computePebblingWithInitialSolution(subInstance[part], heuristic_pebbling, asynchronous);
        if(ILP_result.first == RETURN_STATUS::SUCCESS || ILP_result.first == RETURN_STATUS::BEST_FOUND)
        {
            BspMemSchedule pebblingILP = ILP_result.second;
            if(!pebblingILP.isValid())
                std::cout<<"ERROR: Pebbling ILP INVALID!"<<std::endl;

            pebblingILP.removeEvictStepsFromEnd();
            unsigned ILP_cost = asynchronous ? pebblingILP.computeAsynchronousCost() : pebblingILP.computeCost();
            if(ILP_cost < heuristicCost)
            {
                pebbling[part] = pebblingILP;
                std::cout<<"ILP chosen instead of greedy. ("<<ILP_cost<<" < "<<heuristicCost<<")"<<std::endl;
            }
            else
                std::cout<<"Greedy chosen instead of ILP. ("<<heuristicCost<<" < "<<ILP_cost<<")"<<std::endl;
            
            // save fast memory content for next subproblem
            std::vector<std::set<unsigned> > fast_mem_content_at_end = pebbling[part].getMemContentAtEnd();
            for(unsigned proc = 0; proc < processors_to_parts[part].size(); ++proc)
            {
                in_fast_mem[original_proc_id[part][proc]].clear();
                for(unsigned node : fast_mem_content_at_end[proc])
                    in_fast_mem[original_proc_id[part][proc]].insert(original_node_id[part][node]);
            }
        }
        else
            std::cout<<"ILP found no solution; using greedy instead (cost = "<<heuristicCost<<")."<<std::endl;
    }

    // AUX: assemble final schedule from subschedules
    BspMemSchedule final_pebbling;
    final_pebbling.CreateFromPartialPebblings(instance, pebbling, processors_to_parts, original_node_id, original_proc_id, has_reds_in_beginning);
    final_pebbling.cleanSchedule();
    return {final_pebbling.isValid() ? SUCCESS : ERROR, final_pebbling};

}

ComputationalDag PebblingPartialILP::contractByPartition(const BspInstance &instance, const std::vector<unsigned> &node_to_part_assignment)
{
    const auto &G = instance.getComputationalDag();

    part_and_nodetype_to_new_index.clear();

    unsigned nr_new_nodes = 0;
    for(unsigned node = 0; node < instance.numberOfVertices(); ++node)
    {
        if(part_and_nodetype_to_new_index.find({node_to_part_assignment[node], G.nodeType(node)}) == part_and_nodetype_to_new_index.end())
        {
            part_and_nodetype_to_new_index[{node_to_part_assignment[node], G.nodeType(node)}] = nr_new_nodes;
            ++nr_new_nodes;
        }
    }

    ComputationalDag contracted(nr_new_nodes);
    std::set<std::pair<unsigned, unsigned> > edges;

    for(unsigned node = 0; node < instance.numberOfVertices(); ++node)
    {
        unsigned node_new_index = part_and_nodetype_to_new_index[{node_to_part_assignment[node], G.nodeType(node)}];
        for (const auto &succ : instance.getComputationalDag().children(node))
            if(node_to_part_assignment[node] != node_to_part_assignment[succ])
                edges.emplace(node_new_index, part_and_nodetype_to_new_index[{node_to_part_assignment[succ], G.nodeType(succ)}]);

        contracted.setNodeWorkWeight(node_new_index, contracted.nodeWorkWeight(node_new_index) + G.nodeWorkWeight(node));
        contracted.setNodeCommunicationWeight(node_new_index, contracted.nodeCommunicationWeight(node_new_index) + G.nodeCommunicationWeight(node));
        contracted.setNodeMemoryWeight(node_new_index, contracted.nodeMemoryWeight(node_new_index) + G.nodeMemoryWeight(node));
        contracted.setNodeType(node_new_index, G.nodeType(node));
    }

    for(auto edge : edges)
        contracted.addEdge(edge.first, edge.second, 1);

    return contracted;
}

BspSchedule PebblingPartialILP::computeGreedyBaselineForSubDag(const BspInstance &subInstance, const ComputationalDag& subDagWithoutExternalSources, const std::set<unsigned>& nodes_in_part, unsigned nr_extra_sources) const
{
    // We need to run the bsp scheduling heuristic without the external source nodes, since some of the
    // external source nodes might not have a type-compatible processor at all in the given subinstance.
    // We then explicitly add the external source nodes to the bspmemschedule in the conversion.

    BspInstance restricted_instance(subDagWithoutExternalSources, subInstance.getArchitecture(), subInstance.getNodeProcessorCompatibilityMatrix());

    GreedyBspFillupScheduler greedy;
    BspSchedule schedule_without_external_sources = greedy.computeSchedule(restricted_instance).second;
    BspSchedule schedule_with_external_sources(subInstance);

    std::map<unsigned, unsigned> new_id_without_sources, new_id_with_sources;
    unsigned node_idx = 0;
    for(unsigned node : nodes_in_part)
    {
        new_id_with_sources[node] = node_idx + nr_extra_sources;
        new_id_without_sources[node] = node_idx;
        ++node_idx;
    }

    for(unsigned node : nodes_in_part)
    {
        schedule_with_external_sources.setAssignedProcessor(new_id_with_sources[node], schedule_without_external_sources.assignedProcessor(new_id_without_sources[node]));
        schedule_with_external_sources.setAssignedSuperstep(new_id_with_sources[node], schedule_without_external_sources.assignedSuperstep(new_id_without_sources[node]));
    }

    return schedule_with_external_sources;
}

std::pair<RETURN_STATUS, BspSchedule> PebblingPartialILP::computeSchedule(const BspInstance &instance) {
    return {ERROR, BspSchedule()};
}
