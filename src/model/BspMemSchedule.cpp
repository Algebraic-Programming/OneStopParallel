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

#include "model/BspMemSchedule.hpp"

void BspMemSchedule::updateNumberOfSupersteps(unsigned new_number_of_supersteps) {

    number_of_supersteps = new_number_of_supersteps;

    compute_steps_for_proc_superstep.clear();
    compute_steps_for_proc_superstep.resize(instance->numberOfProcessors(), std::vector<std::vector<compute_step> >(number_of_supersteps));

    nodes_evicted_in_comm.clear();
    nodes_evicted_in_comm.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    nodes_sent_down.clear();
    nodes_sent_down.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    nodes_sent_up.clear();
    nodes_sent_up.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));
}

unsigned BspMemSchedule::computeCost() const {

    int total_costs = 0;
    for(unsigned step = 0; step < number_of_supersteps; ++step)
    {
        // compute phase
        int max_work = INT_MIN;
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
        {
            int work = 0;
            for(const auto& computeStep : compute_steps_for_proc_superstep[proc][step])
                work += instance->getComputationalDag().nodeWorkWeight(computeStep.node);

            if(work > max_work)
                max_work = work;
        }
        total_costs += max_work;

        // communication phase
        int max_send_up = INT_MIN;
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
        {
            int send_up = 0;
            for(unsigned node : nodes_sent_up[proc][step])
                send_up += instance->getComputationalDag().nodeCommunicationWeight(node) * (int)instance->getArchitecture().communicationCosts();

            if(send_up > max_send_up)
                max_send_up = send_up;
        }
        total_costs += max_send_up;

        total_costs += (int)instance->getArchitecture().synchronisationCosts();

        int max_send_down = INT_MIN;
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
        {
            int send_down = 0;
            for(unsigned node : nodes_sent_down[proc][step])
                send_down += instance->getComputationalDag().nodeCommunicationWeight(node) * (int)instance->getArchitecture().communicationCosts();

            if(send_down > max_send_down)
                max_send_down = send_down;
        }
        total_costs += max_send_down;

    }

    return total_costs;
}

unsigned BspMemSchedule::computeAsynchronousCost() const {

    std::vector<unsigned> current_time_at_processor(instance->getArchitecture().numberOfProcessors(), 0);
    std::vector<unsigned> time_when_node_gets_blue(instance->getComputationalDag().numberOfVertices(), UINT_MAX);
    if(need_to_load_inputs)
        for(unsigned node = 0; node < instance->numberOfVertices(); ++node)
            if(instance->getComputationalDag().numberOfParents(node) == 0)
                time_when_node_gets_blue[node] = 0;

    for(unsigned step = 0; step < number_of_supersteps; ++step)
    {
        // compute phase
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
            for(const auto& computeStep : compute_steps_for_proc_superstep[proc][step])
                current_time_at_processor[proc] += instance->getComputationalDag().nodeWorkWeight(computeStep.node);

        // communication phase - send up
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
            for(unsigned node : nodes_sent_up[proc][step])
            {
                current_time_at_processor[proc] += instance->getComputationalDag().nodeCommunicationWeight(node) * (int)instance->getArchitecture().communicationCosts();
                if(time_when_node_gets_blue[node] > current_time_at_processor[proc])
                    time_when_node_gets_blue[node] = current_time_at_processor[proc];
            }

        // communication phase - send down
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
            for(unsigned node : nodes_sent_down[proc][step])
            {
                if(current_time_at_processor[proc] < time_when_node_gets_blue[node])
                    current_time_at_processor[proc] = time_when_node_gets_blue[node];
                current_time_at_processor[proc] += instance->getComputationalDag().nodeCommunicationWeight(node) * (int)instance->getArchitecture().communicationCosts();
            }

    }

    unsigned makespan = 0;
    for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
        if(current_time_at_processor[proc] > makespan)
            makespan = current_time_at_processor[proc];

    return makespan;
}

void BspMemSchedule::cleanSchedule() {

    if(!isValid())
        return;

    // NOTE - this function removes unnecessary steps in most cases, but not all (some require e.g. multiple iterations)

    std::vector<std::vector<std::deque<bool> > > needed(instance->numberOfVertices(), std::vector<std::deque<bool> >(instance->numberOfProcessors()));
    std::vector<std::vector<bool > > keep_false(instance->numberOfVertices(), std::vector<bool >(instance->numberOfProcessors(), false));
    std::vector<std::vector<bool > > has_red_after_cleaning(instance->numberOfVertices(), std::vector<bool >(instance->numberOfProcessors(), false));
    
    std::vector<bool> ever_needed_as_blue(instance->numberOfVertices(), false);
    if(needs_blue_at_end.empty())
    {
        for(unsigned node = 0; node < instance->numberOfVertices(); ++node)
            if(instance->getComputationalDag().numberOfChildren(node) == 0)
                ever_needed_as_blue[node] = true;
    }
    else
    {
        for(unsigned node : needs_blue_at_end)
            ever_needed_as_blue[node] = true;
    }

    for(unsigned step = 0; step < number_of_supersteps; ++step)
        for(unsigned proc = 0; proc < instance->numberOfProcessors(); ++proc)     
            for(unsigned node : nodes_sent_down[proc][step])
                ever_needed_as_blue[node] = true;

    if(!has_red_in_beginning.empty())
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned node : has_red_in_beginning[proc])
                has_red_after_cleaning[node][proc] = true;
    
    for(unsigned step = 0; step < number_of_supersteps; ++step)
    {
        // compute phase
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
            for(const auto& computeStep : compute_steps_for_proc_superstep[proc][step])
            {
                unsigned node = computeStep.node;
                needed[node][proc].emplace_back(false);
                keep_false[node][proc] = has_red_after_cleaning[node][proc];
                for(unsigned pred : instance->getComputationalDag().parents(node))
                {
                    has_red_after_cleaning[pred][proc] = true;
                    if(!keep_false[pred][proc])
                        needed[pred][proc].back() = true;
                }
                for(unsigned to_evict : computeStep.nodes_evicted_after)
                    has_red_after_cleaning[to_evict][proc] = false;
            }

        // send up phase
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
            for(unsigned node : nodes_sent_up[proc][step])
                if(ever_needed_as_blue[node])
                {
                    has_red_after_cleaning[node][proc] = true;
                    if(!keep_false[node][proc])
                        needed[node][proc].back() = true;
                }

        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
            for(unsigned node : nodes_evicted_in_comm[proc][step])
                has_red_after_cleaning[node][proc] = false;

        //send down phase
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)     
            for(unsigned node : nodes_sent_down[proc][step])
            {
                needed[node][proc].emplace_back(false);
                keep_false[node][proc] = has_red_after_cleaning[node][proc];
            }
    }

    std::vector<std::vector<std::vector<compute_step> > > new_compute_steps_for_proc_superstep(instance->numberOfProcessors(), std::vector<std::vector<compute_step> >(number_of_supersteps));
    std::vector<std::vector<std::vector<unsigned> > > new_nodes_evicted_in_comm(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));
    std::vector<std::vector<std::vector<unsigned> > > new_nodes_sent_down(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));
    std::vector<std::vector<std::vector<unsigned> > > new_nodes_sent_up(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    std::vector<std::vector<bool> > has_red(instance->numberOfVertices(), std::vector<bool>(instance->numberOfProcessors(), false));
    if(!has_red_in_beginning.empty())
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned node : has_red_in_beginning[proc])
                has_red[node][proc] = true;
    
    std::vector<bool> has_blue(instance->numberOfVertices());
    std::vector<int> time_when_node_gets_blue(instance->getComputationalDag().numberOfVertices(), INT_MAX);
    if(need_to_load_inputs)
        for(unsigned node = 0; node < instance->numberOfVertices(); ++node)
            if(instance->getComputationalDag().numberOfParents(node) == 0)
            {
                has_blue[node] = true;
                time_when_node_gets_blue[node] = 0;
            }

    std::vector<int> current_time_at_processor(instance->getArchitecture().numberOfProcessors(), 0);

    for(unsigned superstep = 0; superstep < number_of_supersteps; ++superstep)
    {
        // compute phase
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
        {
            std::vector<bool> step_remains(compute_steps_for_proc_superstep[proc][superstep].size(), false);
            std::vector<std::vector<unsigned> > new_evict_after(compute_steps_for_proc_superstep[proc][superstep].size());
            
            unsigned new_stepIndex = 0;
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;

                if(needed[node][proc].front())
                {
                    new_compute_steps_for_proc_superstep[proc][superstep].emplace_back(node, std::vector<unsigned>());
                    step_remains[stepIndex] = true;
                    has_red[node][proc] = true;
                    ++new_stepIndex;
                    current_time_at_processor[proc] += instance->getComputationalDag().nodeWorkWeight(node);
                }

                needed[node][proc].pop_front();

                for(unsigned to_evict : compute_steps_for_proc_superstep[proc][superstep][stepIndex].nodes_evicted_after)
                {
                    if(has_red[to_evict][proc])
                        new_evict_after[stepIndex].push_back(to_evict);
                    has_red[to_evict][proc] = false;
                }
            }

            // go backwards to fix cache eviction steps
            std::vector<unsigned> to_evict;
            for(int stepIndex = compute_steps_for_proc_superstep[proc][superstep].size() - 1; stepIndex >= 0; --stepIndex)
            {
                for(unsigned node : new_evict_after[stepIndex])
                    to_evict.push_back(node);

                if(step_remains[stepIndex])
                {
                    new_compute_steps_for_proc_superstep[proc][superstep][new_stepIndex-1].nodes_evicted_after = to_evict;
                    to_evict.clear();
                    --new_stepIndex;
                }
            }
            if(!to_evict.empty() && superstep>=1)
                for(unsigned node : to_evict)
                {
                    auto itr = std::find(new_nodes_sent_down[proc][superstep-1].begin(), new_nodes_sent_down[proc][superstep-1].end(), node);
                    if(itr == new_nodes_sent_down[proc][superstep-1].end())
                        new_nodes_evicted_in_comm[proc][superstep-1].push_back(node);
                    else
                        new_nodes_sent_down[proc][superstep-1].erase(itr);
                }
        }
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
        {
            // send up phase
            for(unsigned node : nodes_sent_up[proc][superstep])
            {
                if(!ever_needed_as_blue[node])
                    continue;

                int new_time_at_processor = current_time_at_processor[proc] + instance->getComputationalDag().nodeCommunicationWeight(node) * (int)instance->getArchitecture().communicationCosts();

                // only copy send up step if it is not obsolete in at least one of the two cases (sync or async schedule)
                if(!has_blue[node] || new_time_at_processor < time_when_node_gets_blue[node])
                {
                    new_nodes_sent_up[proc][superstep].push_back(node);
                    has_blue[node] = true;
                    current_time_at_processor[proc] = new_time_at_processor;
                    if(time_when_node_gets_blue[node] > new_time_at_processor)
                        time_when_node_gets_blue[node] = new_time_at_processor;
                }
            }
        }

        // comm phase evict
        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
            for(unsigned node : nodes_evicted_in_comm[proc][superstep])
                if(has_red[node][proc])
                {
                    new_nodes_evicted_in_comm[proc][superstep].push_back(node);
                    has_red[node][proc] = false;
                }

        for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
        {
            //send down phase     
            for(unsigned node : nodes_sent_down[proc][superstep])
            {
                if(needed[node][proc].front())
                {
                    new_nodes_sent_down[proc][superstep].push_back(node);
                    has_red[node][proc] = true;
                    if(current_time_at_processor[proc] < time_when_node_gets_blue[node])
                        current_time_at_processor[proc] = time_when_node_gets_blue[node];
                    current_time_at_processor[proc] += instance->getComputationalDag().nodeCommunicationWeight(node) * (int)instance->getArchitecture().communicationCosts();
                }
                needed[node][proc].pop_front();
            }

        }
    }

    compute_steps_for_proc_superstep = new_compute_steps_for_proc_superstep;
    nodes_evicted_in_comm = new_nodes_evicted_in_comm;
    nodes_sent_down = new_nodes_sent_down;
    nodes_sent_up = new_nodes_sent_up;
}

void BspMemSchedule::ConvertFromBsp(const BspSchedule &schedule, CACHE_EVICTION_STRATEGY evict_rule)
{
    instance = &schedule.getInstance();

    // check if conversion possible at all
    if(!hasValidSolution(schedule.getInstance(), external_sources))
    {
        std::cout<<"Conversion failed."<<std::endl;
        return;
    }

    // split supersteps
    SplitSupersteps(schedule);

    // track memory
    SetMemoryMovement(evict_rule);   
}

bool BspMemSchedule::hasValidSolution(const BspInstance &instance, const std::set<unsigned>& external_sources)
{
    std::vector<unsigned> memory_required = minimumMemoryRequiredPerNodeType(instance);
    std::vector<bool> has_enough_memory(instance.getComputationalDag().getNumberOfNodeTypes(), true);
    for(unsigned node = 0; node < instance.numberOfVertices(); ++node)
        if(external_sources.find(node) == external_sources.end())
            has_enough_memory[instance.getComputationalDag().nodeType(node)] = false;

    for(unsigned node_type = 0; node_type < instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type)
        for(unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc)
            if(instance.isCompatibleType(node_type, instance.getArchitecture().processorType(proc)) &&
                instance.getArchitecture().memoryBound(proc) >= memory_required[node_type])
            {
                has_enough_memory[node_type] = true;
                break;
            }

    for(unsigned node_type = 0; node_type < instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type)
        if(!has_enough_memory[node_type])
        {
            std::cout<<"No valid solution exists. Minimum memory required for node type "<<node_type<<" is "<<memory_required[node_type]<<std::endl;
            return false;
        }
    return true;
}

void BspMemSchedule::SplitSupersteps(const BspSchedule &schedule)
{
    // get DFS topological order in each superstep
    std::vector<std::vector<std::vector<unsigned> > > top_orders = computeTopOrdersDFS(schedule);

    std::vector<unsigned> top_order_idx(instance->getComputationalDag().numberOfVertices(), 0);
    for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        for(unsigned step=0; step<schedule.numberOfSupersteps(); ++step)
            for(unsigned idx =0; idx < top_orders[proc][step].size(); ++idx)
                top_order_idx[top_orders[proc][step][idx]] = idx;

    // split supersteps as needed
    std::vector<unsigned> new_superstep_ID(instance->getComputationalDag().numberOfVertices());
    unsigned superstep_index = 0;
    for(unsigned step=0; step<schedule.numberOfSupersteps(); ++step)
    {
        unsigned max_segments_in_superstep = 0;
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            if(top_orders[proc][step].empty())
                continue;

            // the superstep will be split into smaller segments
            std::vector<std::pair<unsigned, unsigned> > segments;
            unsigned start_idx = 0;
            while(start_idx < top_orders[proc][step].size())
            {
                // binary search for largest segment that still statisfies mem constraint
                bool doubling_phase = true;
                unsigned end_lower_bound = start_idx, end_upper_bound = top_orders[proc][step].size()-1;
                while(end_lower_bound < end_upper_bound)
                {
                    unsigned end_current;
                    
                    if(doubling_phase)
                    {
                        if(end_lower_bound == start_idx)
                            end_current = start_idx + 1;
                        else
                            end_current = std::min(start_idx + 2* (end_lower_bound - start_idx),
                                                (unsigned) top_orders[proc][step].size()-1);
                    }
                    else
                        end_current = end_lower_bound + (end_upper_bound - end_lower_bound + 1) / 2;

                    // check if this segment is valid
                    bool valid = true;

                    std::map<int, bool> neededAfter;
                    for(unsigned idx = start_idx; idx <= end_current; ++idx)
                    {
                        unsigned node = top_orders[proc][step][idx];
                        neededAfter[node] = false;
                        if(needs_blue_at_end.empty())
                            neededAfter[node] = (instance->getComputationalDag().numberOfChildren(node) == 0);
                        else
                            neededAfter[node] = (needs_blue_at_end.find(node) != needs_blue_at_end.end());
                        for(unsigned succ : instance->getComputationalDag().children(node))
                        {
                            if(schedule.assignedSuperstep(succ)>step)
                                neededAfter[node] = true;
                            if(schedule.assignedSuperstep(succ) == step && top_order_idx[succ] <= end_current)
                                neededAfter[node] = true;
                        }

                    }

                    std::map<unsigned, unsigned> lastUsedBy;
                    std::set<unsigned> values_needed;
                    for(unsigned idx = start_idx; idx <= end_current; ++idx)
                    {
                        unsigned node = top_orders[proc][step][idx];
                        for(unsigned pred : instance->getComputationalDag().parents(node))
                        {
                            if(schedule.assignedSuperstep(pred)<step || (schedule.assignedSuperstep(pred)==step && !neededAfter[pred]))
                                lastUsedBy[pred] = node;
                            if(schedule.assignedSuperstep(pred)<step || (schedule.assignedSuperstep(pred)==step && top_order_idx[pred] < start_idx)
                                || (need_to_load_inputs && instance->getComputationalDag().numberOfParents(pred)==0) 
                                || external_sources.find(pred) != external_sources.end() )
                                values_needed.insert(pred);
                        }
                    }

                    unsigned mem_needed = 0;
                    for(unsigned node : values_needed)
                        mem_needed += instance->getComputationalDag().nodeMemoryWeight(node);


                    for(unsigned idx = start_idx; idx <= end_current; ++idx)
                    {
                        unsigned node = top_orders[proc][step][idx];

                        if(need_to_load_inputs && instance->getComputationalDag().numberOfParents(node) == 0)
                            continue;

                        mem_needed += instance->getComputationalDag().nodeMemoryWeight(node);
                        if(mem_needed > instance->getArchitecture().memoryBound(proc))
                        {
                            valid = false;
                            break;
                        }

                        for(unsigned pred : instance->getComputationalDag().parents(node))
                            if(lastUsedBy[pred] == node)
                                mem_needed -= instance->getComputationalDag().nodeMemoryWeight(pred);
                    }

                    if(valid)
                    {
                        end_lower_bound = end_current;
                        if(end_current == top_orders[proc][step].size()-1)
                        {
                            doubling_phase = false;
                            end_upper_bound = end_current;
                        }
                    }
                    else
                    {
                        doubling_phase = false;
                        end_upper_bound = end_current - 1;
                    }

                }
                segments.emplace_back(start_idx, end_lower_bound);
                start_idx = end_lower_bound + 1;
            }

            unsigned step_idx = 0;
            for(auto segment : segments)
            {
                for(unsigned idx = segment.first; idx <= segment.second; ++idx)
                    new_superstep_ID[top_orders[proc][step][idx]] = superstep_index + step_idx;

                ++step_idx;
            }

            if(step_idx>max_segments_in_superstep)
                max_segments_in_superstep = step_idx;
        }
        superstep_index += max_segments_in_superstep;
    }

    std::vector<unsigned> reindex_to_shrink(superstep_index);
    std::vector<bool> has_compute(superstep_index, false);
    for(unsigned node = 0; node < instance->numberOfVertices(); ++node)
        if(!need_to_load_inputs || instance->getComputationalDag().numberOfParents(node) > 0)
            has_compute[new_superstep_ID[node]] = true;
    
    unsigned current_index = 0;
    for(unsigned superstep = 0; superstep < superstep_index; ++superstep)
        if(has_compute[superstep])
        {
            reindex_to_shrink[superstep] = current_index;
            ++current_index;
        }

    unsigned offset = need_to_load_inputs ? 1 : 0;
    updateNumberOfSupersteps(current_index+offset);
    std::cout<<schedule.numberOfSupersteps()<<" -> "<<number_of_supersteps<<std::endl;

    // TODO: might not need offset for first step when beginning with red pebbles

    for(unsigned step=0; step<schedule.numberOfSupersteps(); ++step)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned node : top_orders[proc][step])
                if(!need_to_load_inputs || instance->getComputationalDag().numberOfParents(node) > 0)
                    compute_steps_for_proc_superstep[proc][reindex_to_shrink[new_superstep_ID[node]]+offset].emplace_back(node);
    
}

void BspMemSchedule::SetMemoryMovement(CACHE_EVICTION_STRATEGY evict_rule)
{
    const unsigned N = instance->getComputationalDag().numberOfVertices();

    std::vector<unsigned> mem_used(instance->numberOfProcessors(), 0);
    std::vector<std::set<unsigned> > in_mem(instance->numberOfProcessors());

    std::vector<bool> in_slow_mem(N, false);
    if(need_to_load_inputs)
        for(unsigned node=0; node<N; ++node)
            if(instance->getComputationalDag().numberOfParents(node) == 0)
                in_slow_mem[node] = true;

    std::vector<std::set<std::pair<std::pair<unsigned, unsigned>, unsigned>> > evictable(instance->numberOfProcessors());
    std::vector<std::set<unsigned> > non_evictable(instance->numberOfProcessors());
        
    // iterator to its position in "evictable" - for efficient delete
    std::vector<std::vector<std::set<std::pair<std::pair<unsigned, unsigned>, unsigned> >::iterator > > place_in_evictable(N,
        std::vector<std::set<std::pair<std::pair<unsigned, unsigned>, unsigned> >::iterator>(instance->numberOfProcessors()));
    for(unsigned node=0; node<N; ++node)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            place_in_evictable[node][proc] = evictable[proc].end();

    // utility for LRU eviction strategy
    std::vector<std::vector<unsigned> > node_last_used_on_proc;
    if(evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED)
        node_last_used_on_proc.resize(N, std::vector<unsigned>(instance->numberOfProcessors(), 0));
    std::vector<unsigned> total_step_count_on_proc(instance->numberOfProcessors(), 0);

    // select a representative compute step for each node, in case of being computed multiple times
    // (NOTE - the conversion assumes that there is enough fast memory to keep each value until the end of
    // its representative step, if the value in question is ever needed on another processor/superster
    // without being recomputed there - otherwise, it would be even hard to decide whether a solution exists)
    std::vector<unsigned> selected_processor(N);
    std::vector<std::pair<unsigned, unsigned> > selected_step(N, std::make_pair(number_of_supersteps, 0));
    for(unsigned superstep=0; superstep<number_of_supersteps; ++superstep)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                if(selected_step[node].first > superstep || (selected_step[node].first == superstep && selected_step[node].second < stepIndex))
                {
                    selected_processor[node] = proc;
                    selected_step[node] = std::make_pair(superstep, stepIndex);
                }
            }

    // check if the node needs to be kept until the end of its representative superstep
    std::vector<bool> must_be_preserved(N, false);
    std::vector<bool> computed_in_current_superstep(N, false);
    for(unsigned superstep=0; superstep<number_of_supersteps; ++superstep)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                computed_in_current_superstep[node] = true;
                for(unsigned pred : instance->getComputationalDag().parents(node))
                    if(!computed_in_current_superstep[pred])
                        must_be_preserved[pred] = true;
            }
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
                computed_in_current_superstep[compute_steps_for_proc_superstep[proc][superstep][stepIndex].node] = false;
        }
    if(needs_blue_at_end.empty())
    {
        for(unsigned node = 0; node < N; ++node)
            if(instance->getComputationalDag().numberOfChildren(node) == 0)
                must_be_preserved[node] = true;
    }
    else
    {
        for(unsigned node : needs_blue_at_end)
            must_be_preserved[node] = true;
    }

    // superstep-step pairs where a node is required (on a given proc) - opening a separate queue after each time it's recomputed
    std::vector<std::vector<std::deque<std::deque<std::pair<unsigned, unsigned> > > > > node_used_at_proc_lists(N, std::vector<std::deque<std::deque<std::pair<unsigned, unsigned> > > >(instance->numberOfProcessors(), std::deque<std::deque<std::pair<unsigned, unsigned> > >(1)));
    for(unsigned superstep=0; superstep<number_of_supersteps; ++superstep)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                for(unsigned pred : instance->getComputationalDag().parents(node))
                    node_used_at_proc_lists[pred][proc].back().emplace_back(superstep, stepIndex);
                
                node_used_at_proc_lists[node][proc].emplace_back();
            }

    // set up initial content of fast memories
    if(!has_red_in_beginning.empty())
    {
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            in_mem = has_red_in_beginning;
            for(unsigned node : in_mem[proc])
            {
                mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(node);

                std::pair<unsigned, unsigned> prio;
                if(evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT)
                    prio = node_used_at_proc_lists[node][proc].front().front();
                else if(evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED)
                    prio = std::make_pair(UINT_MAX - node_last_used_on_proc[node][proc], node);
                else if(evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID)
                    prio = std::make_pair(node, 0);

                place_in_evictable[node][proc] = evictable[proc].emplace(prio, node).first;
            }
        }
    }
    
    // iterate through schedule
    for(unsigned superstep=0; superstep<number_of_supersteps; ++superstep)
    {
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            if(compute_steps_for_proc_superstep[proc][superstep].empty())
                continue;

            // before compute phase, evict data in comm phase of previous superstep
            std::set<unsigned> new_values_needed;
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                computed_in_current_superstep[node] = true;
                for(unsigned pred : instance->getComputationalDag().parents(node))
                    if(!computed_in_current_superstep[pred])
                    {
                        non_evictable[proc].insert(pred);

                        if(place_in_evictable[pred][proc] != evictable[proc].end())
                        {
                            evictable[proc].erase(place_in_evictable[pred][proc]);
                            place_in_evictable[pred][proc] = evictable[proc].end();
                        }

                        if(in_mem[proc].find(pred) == in_mem[proc].end())
                            new_values_needed.insert(pred);
                    }
            }
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
                computed_in_current_superstep[compute_steps_for_proc_superstep[proc][superstep][stepIndex].node] = false;
            
            for(unsigned node : new_values_needed)
            {
                in_mem[proc].insert(node);
                mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
                nodes_sent_down[proc][superstep-1].push_back(node);
                if(!in_slow_mem[node])
                {
                    in_slow_mem[node] = true;
                    nodes_sent_up[selected_processor[node]][selected_step[node].first].push_back(node);
                }
            }

            unsigned first_node_weight = instance->getComputationalDag().nodeMemoryWeight(compute_steps_for_proc_superstep[proc][superstep][0].node);

            while(mem_used[proc] + first_node_weight > instance->getArchitecture().memoryBound(proc)) // no sliding pebbles for now
            {
                if(evictable[proc].empty())
                {
                    std::cout<<"ERROR: Cannot create valid memory movement for these superstep lists."<<std::endl;
                    return;
                }
                unsigned evicted = (--evictable[proc].end())->second;
                evictable[proc].erase(--evictable[proc].end());
                place_in_evictable[evicted][proc] = evictable[proc].end();

                mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(evicted);
                in_mem[proc].erase(evicted);

                nodes_evicted_in_comm[proc][superstep-1].push_back(evicted);
            }

            // indicates if the node will be needed after (and thus cannot be deleted during) this compute phase
            std::map<unsigned, bool> needed_after;            

            // during compute phase
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                unsigned node_weight = instance->getComputationalDag().nodeMemoryWeight(node);

                if(stepIndex > 0)
                {
                    //evict nodes to make space
                    while(mem_used[proc] + node_weight > instance->getArchitecture().memoryBound(proc))
                    {
                        if(evictable[proc].empty())
                        {
                            std::cout<<"ERROR: Cannot create valid memory movement for these superstep lists."<<std::endl;
                            return;
                        }
                        unsigned evicted = (--evictable[proc].end())->second;
                        evictable[proc].erase(--evictable[proc].end());
                        place_in_evictable[evicted][proc] = evictable[proc].end();

                        mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(evicted);
                        in_mem[proc].erase(evicted);              

                        compute_steps_for_proc_superstep[proc][superstep][stepIndex-1].nodes_evicted_after.push_back(evicted);
                    }
                }

                in_mem[proc].insert(node);
                mem_used[proc] += node_weight;

                non_evictable[proc].insert(node);

                if(evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED) // update usage times for LRU strategy
                {
                    ++total_step_count_on_proc[proc];
                    node_last_used_on_proc[node][proc] = total_step_count_on_proc[proc];
                    for(unsigned pred : instance->getComputationalDag().parents(node))
                        node_last_used_on_proc[pred][proc] = total_step_count_on_proc[proc];
                }

                if(selected_processor[node] == proc && selected_step[node] == std::make_pair(superstep, stepIndex) && must_be_preserved[node])
                    needed_after[node] = true;
                else
                    needed_after[node] = false;

                node_used_at_proc_lists[node][proc].pop_front();
                
                for(unsigned pred : instance->getComputationalDag().parents(node))
                {
                    node_used_at_proc_lists[pred][proc].front().pop_front();

                    if(needed_after[pred])
                        continue;

                    // autoevict
                    if(node_used_at_proc_lists[pred][proc].front().empty())
                    {
                        in_mem[proc].erase(pred);
                        non_evictable[proc].erase(pred);
                        mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(pred);
                        compute_steps_for_proc_superstep[proc][superstep][stepIndex].nodes_evicted_after.push_back(pred);            
                    }
                    else if(node_used_at_proc_lists[pred][proc].front().front().first > superstep)
                    {
                        non_evictable[proc].erase(pred);

                        std::pair<unsigned, unsigned> prio;
                        if(evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT)
                            prio = node_used_at_proc_lists[pred][proc].front().front();
                        else if(evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED)
                            prio = std::make_pair(UINT_MAX - node_last_used_on_proc[pred][proc], node);
                        else if(evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID)
                            prio = std::make_pair(pred, 0);

                        place_in_evictable[pred][proc] = evictable[proc].emplace(prio, pred).first;
                    }
                }
                
            }

            // after compute phase
            for(unsigned node : non_evictable[proc])
            {
                if(node_used_at_proc_lists[node][proc].front().empty())
                {
                    mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(node);
                    in_mem[proc].erase(node);
                    nodes_evicted_in_comm[proc][superstep].push_back(node);
                    if((instance->getComputationalDag().numberOfChildren(node) == 0 || needs_blue_at_end.find(node) != needs_blue_at_end.end())
                        && !in_slow_mem[node])
                    {
                        in_slow_mem[node] = true;
                        nodes_sent_up[proc][superstep].push_back(node);
                    }
                }
                else
                {
                    std::pair<unsigned, unsigned> prio;
                    if(evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT)
                        prio = node_used_at_proc_lists[node][proc].front().front();
                    else if(evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED)
                        prio = std::make_pair(UINT_MAX - node_last_used_on_proc[node][proc], node);
                    else if(evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID)
                        prio = std::make_pair(node, 0);

                    place_in_evictable[node][proc] = evictable[proc].emplace(prio, node).first;

                    if(needs_blue_at_end.find(node) != needs_blue_at_end.end() && !in_slow_mem[node])
                    {
                        in_slow_mem[node] = true;
                        nodes_sent_up[proc][superstep].push_back(node);
                    }
                }
            }
            non_evictable[proc].clear();
        }
    }

}

void BspMemSchedule::ResetToForesight()
{
    nodes_evicted_in_comm.clear();
    nodes_evicted_in_comm.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    nodes_sent_down.clear();
    nodes_sent_down.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    nodes_sent_up.clear();
    nodes_sent_up.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    SetMemoryMovement(CACHE_EVICTION_STRATEGY::FORESIGHT);
}

bool BspMemSchedule::isValid() const
{
    std::vector<unsigned> mem_used(instance->numberOfProcessors(), 0);
    std::vector<std::vector<unsigned> > in_fast_mem(instance->getComputationalDag().numberOfVertices(),
         std::vector<unsigned>(instance->numberOfProcessors(), false));
    std::vector<unsigned> in_slow_mem(instance->getComputationalDag().numberOfVertices(), false);

    if(need_to_load_inputs)
        for(unsigned node = 0; node < instance->numberOfVertices(); ++node)
            if(instance->getComputationalDag().numberOfParents(node) == 0)
                in_slow_mem[node] = true;
    
    if(!has_red_in_beginning.empty())
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned node : has_red_in_beginning[proc])
            {
                mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
                in_fast_mem[node][proc] = true;
            }

    for(unsigned step=0; step<number_of_supersteps; ++step)
    {
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            // computation phase
            for(const auto& computeStep : compute_steps_for_proc_superstep[proc][step])
            {                
                if(!instance->isCompatible(computeStep.node, proc))
                    return false;

                for(unsigned pred : instance->getComputationalDag().parents(computeStep.node))
                    if(!in_fast_mem[pred][proc])
                        return false;

                if(need_to_load_inputs && instance->getComputationalDag().numberOfParents(computeStep.node) == 0)
                    return false;
                
                if(!in_fast_mem[computeStep.node][proc])
                {            
                    in_fast_mem[computeStep.node][proc] = true;
                    mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(computeStep.node);
                }

                if(mem_used[proc] > instance->getArchitecture().memoryBound(proc))
                    return false;

                for(unsigned to_remove : computeStep.nodes_evicted_after)
                {
                    if(!in_fast_mem[to_remove][proc])
                        return false;

                    in_fast_mem[to_remove][proc] = false;
                    mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(to_remove);

                }
            }

            //communication phase - sendup and eviction
            for(unsigned node : nodes_sent_up[proc][step])
            {
                if(!in_fast_mem[node][proc])
                    return false;
                
                in_slow_mem[node] = true;
            }
            for(unsigned node : nodes_evicted_in_comm[proc][step])
            {
                if(!in_fast_mem[node][proc])
                    return false;

                in_fast_mem[node][proc] = false;
                mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(node);
            }
        }

        // communication phase - senddown
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            for(unsigned node : nodes_sent_down[proc][step])
            {
                if(!in_slow_mem[node])
                    return false;

                if(!in_fast_mem[node][proc])
                {
                    in_fast_mem[node][proc] = true;
                    mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
                }
            }
        }

        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            if(mem_used[proc] > instance->getArchitecture().memoryBound(proc))
                return false;
    }

    if(needs_blue_at_end.empty())
    {
        for(unsigned node = 0; node < instance->numberOfVertices(); ++node)
            if(instance->getComputationalDag().numberOfChildren(node) == 0 && !in_slow_mem[node])
                return false;
    }
    else
    {
        for(unsigned node : needs_blue_at_end)
            if(!in_slow_mem[node])
                return false;
    }

    return true;
}

std::vector<unsigned> BspMemSchedule::minimumMemoryRequiredPerNodeType(const BspInstance& instance, const std::set<unsigned>& external_sources)
{
    std::vector<unsigned> max_needed(instance.getComputationalDag().getNumberOfNodeTypes(), 0);
    for(unsigned node=0; node<instance.getComputationalDag().numberOfVertices(); ++node)
    {
        if(external_sources.find(node) != external_sources.end())
            continue;

        unsigned needed = instance.getComputationalDag().nodeMemoryWeight(node);
        const unsigned type = instance.getComputationalDag().nodeType(node);
        for(unsigned pred : instance.getComputationalDag().parents(node))
            needed += instance.getComputationalDag().nodeMemoryWeight(pred);
        
        if(needed>max_needed[type])
            max_needed[type]=needed;
    }
    return max_needed;
}

std::vector<std::vector<std::vector<unsigned> > > BspMemSchedule::computeTopOrdersDFS(const BspSchedule &schedule) const
{
    unsigned n = schedule.getInstance().getComputationalDag().numberOfVertices();
    unsigned num_procs = schedule.getInstance().numberOfProcessors();
    unsigned num_supsteps = schedule.numberOfSupersteps();

    std::vector<std::vector<std::vector<unsigned> > > top_orders(num_procs, std::vector<std::vector<unsigned> >(num_supsteps));

    std::vector<std::vector<std::deque<unsigned> > > Q(num_procs, std::vector<std::deque<unsigned> >(num_supsteps));
    std::vector<std::vector<std::vector<unsigned> > > nodesUpdated(num_procs, std::vector<std::vector<unsigned> >(num_supsteps));
    std::vector<int> nr_pred(n);
    std::vector<int> pred_done(n, 0);
    for(unsigned node=0; node<n; ++node)
    {
        int predecessors = 0;
        for(unsigned pred : schedule.getInstance().getComputationalDag().parents(node))
            if(external_sources.find(pred) == external_sources.end()
            && schedule.assignedProcessor(node)==schedule.assignedProcessor(pred)
            && schedule.assignedSuperstep(node)==schedule.assignedSuperstep(pred))
                ++predecessors;
        nr_pred[node] = predecessors;
        if(predecessors==0 && external_sources.find(node) == external_sources.end())
            Q[schedule.assignedProcessor(node)][schedule.assignedSuperstep(node)].push_back(node);
    }
    for(unsigned proc=0; proc<num_procs; ++proc)
        for(unsigned step=0; step<num_supsteps; ++step)
        {
            while(!Q[proc][step].empty())
            {
                int node = Q[proc][step].front();
                Q[proc][step].pop_front();
                top_orders[proc][step].push_back(node);
                for(unsigned succ : schedule.getInstance().getComputationalDag().children(node))
                    if(schedule.assignedProcessor(node)==schedule.assignedProcessor(succ)
                    && schedule.assignedSuperstep(node)==schedule.assignedSuperstep(succ))
                    {
                        ++pred_done[succ];
                        if(pred_done[succ]==nr_pred[succ])
                            Q[proc][step].push_front(succ);
                    }
            }
        }

    return top_orders;
}

 void BspMemSchedule::getDataForMultiprocessorPebbling(std::vector<std::vector<std::vector<unsigned> > >& computeSteps,
                                          std::vector<std::vector<std::vector<unsigned> > >& sendUpSteps,
                                          std::vector<std::vector<std::vector<unsigned> > >& sendDownSteps,
                                          std::vector<std::vector<std::vector<unsigned> > >& nodesEvictedAfterStep) const
{
    computeSteps.clear();
    computeSteps.resize(instance->numberOfProcessors());
    sendUpSteps.clear();
    sendUpSteps.resize(instance->numberOfProcessors());
    sendDownSteps.clear();
    sendDownSteps.resize(instance->numberOfProcessors());
    nodesEvictedAfterStep.clear();
    nodesEvictedAfterStep.resize(instance->numberOfProcessors());

    std::vector<unsigned> mem_used(instance->numberOfProcessors(), 0);
    std::vector<std::set<unsigned> > in_mem(instance->numberOfProcessors());
    if(!has_red_in_beginning.empty())
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned node : has_red_in_beginning[proc])
            {
                in_mem[proc].insert(node);
                mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
            }

    unsigned step = 0;

    for(unsigned superstep=0; superstep<number_of_supersteps; ++superstep)
    {
        std::vector<unsigned> step_on_proc(instance->numberOfProcessors(), step);
        bool any_compute = false;
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            if(!compute_steps_for_proc_superstep[proc][superstep].empty())
                any_compute = true;
        
        if(any_compute)
            for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back();
            }

        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            std::vector<unsigned> evict_list;
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                if(mem_used[proc] + instance->getComputationalDag().nodeMemoryWeight(node) > instance->getArchitecture().memoryBound(proc))
                {
                    //open new step
                    nodesEvictedAfterStep[proc][step_on_proc[proc]] = evict_list;
                    ++step_on_proc[proc];
                    for(unsigned to_evict : evict_list)
                        mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(to_evict);
                    
                    evict_list.clear();
                    computeSteps[proc].emplace_back();
                    sendUpSteps[proc].emplace_back();
                    sendDownSteps[proc].emplace_back();
                    nodesEvictedAfterStep[proc].emplace_back();
                }

                computeSteps[proc][step_on_proc[proc]].emplace_back(node);
                mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
                for(unsigned to_evict : compute_steps_for_proc_superstep[proc][superstep][stepIndex].nodes_evicted_after)
                    evict_list.emplace_back(to_evict);
                
            }

            if(!evict_list.empty())
            {
                nodesEvictedAfterStep[proc][step_on_proc[proc]] = evict_list;
                for(unsigned to_evict : evict_list)
                    mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(to_evict);
            }
            
        }
        if(any_compute)
            for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
                ++step_on_proc[proc];

        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            step = std::max(step, step_on_proc[proc]);
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(; step_on_proc[proc]<step; ++step_on_proc[proc])
            {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back();
            }
        
        bool any_send_up = false;
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            if(!nodes_sent_up[proc][superstep].empty() || !nodes_evicted_in_comm[proc][superstep].empty())
                any_send_up = true;
        
        if(any_send_up)
        {
            for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back(nodes_sent_up[proc][superstep]);
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back(nodes_evicted_in_comm[proc][superstep]);
                for(unsigned to_evict : nodes_evicted_in_comm[proc][superstep])
                    mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(to_evict);
                ++step_on_proc[proc];
            }
            ++step;
        }

        bool any_send_down = false;
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            if(!nodes_sent_down[proc][superstep].empty())
                any_send_down = true;

        if(any_send_down)
        {
            for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back(nodes_sent_down[proc][superstep]);
                for(unsigned send_down : nodes_sent_down[proc][superstep])
                    mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(send_down);
                nodesEvictedAfterStep[proc].emplace_back();
                ++step_on_proc[proc];
            }
            ++step;
        }

    }
}

std::vector<std::set<unsigned> > BspMemSchedule::getMemContentAtEnd() const
{
    std::vector<std::set<unsigned> > mem_content(instance->numberOfProcessors());
    if(!has_red_in_beginning.empty())
        mem_content = has_red_in_beginning;

    for(unsigned step=0; step<number_of_supersteps; ++step)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            // computation phase
            for(const auto& computeStep : compute_steps_for_proc_superstep[proc][step])
            {
                mem_content[proc].insert(computeStep.node);
                for(unsigned to_remove : computeStep.nodes_evicted_after)
                    mem_content[proc].erase(to_remove);
            }

            //communication phase - eviction
            for(unsigned node : nodes_evicted_in_comm[proc][step])
                mem_content[proc].erase(node);

            // communication phase - senddown
            for(unsigned node : nodes_sent_down[proc][step])
                mem_content[proc].insert(node);
        }

    return mem_content;
}

void BspMemSchedule::removeEvictStepsFromEnd()
{
    std::vector<unsigned> mem_used(instance->numberOfProcessors(), 0);
    std::vector<unsigned> bottleneck(instance->numberOfProcessors(), 0);
    std::vector<std::set<unsigned> > fast_mem_end = getMemContentAtEnd();
    for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
    {
        for(unsigned node : fast_mem_end[proc])
            mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(node);

        bottleneck[proc] = instance->getArchitecture().memoryBound(proc) - mem_used[proc];
    }

    for(unsigned step=number_of_supersteps; step>0;)
    {
        --step;

        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            // communication phase - senddown
            for(unsigned node : nodes_sent_down[proc][step])
                mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(node);

            //communication phase - eviction
            std::vector<unsigned> remaining;
            for(unsigned node : nodes_evicted_in_comm[proc][step])
            {
                mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
                if((unsigned) instance->getComputationalDag().nodeMemoryWeight(node) <= bottleneck[proc]
                    && fast_mem_end[proc].find(node) == fast_mem_end[proc].end())
                {
                    fast_mem_end[proc].insert(node);
                    bottleneck[proc] -= instance->getComputationalDag().nodeMemoryWeight(node);
                }
                else
                    remaining.push_back(node);
            }
            nodes_evicted_in_comm[proc][step] = remaining;
            bottleneck[proc] = std::min(bottleneck[proc], instance->getArchitecture().memoryBound(proc) - mem_used[proc]);

            // computation phase
            for(unsigned stepIndex = compute_steps_for_proc_superstep[proc][step].size(); stepIndex > 0;)
            {
                --stepIndex;
                auto &computeStep = compute_steps_for_proc_superstep[proc][step][stepIndex];

                std::vector<unsigned> remaining;
                for(unsigned to_remove : computeStep.nodes_evicted_after)
                {
                    mem_used[proc] += instance->getComputationalDag().nodeMemoryWeight(to_remove);
                    if( (unsigned) instance->getComputationalDag().nodeMemoryWeight(to_remove) <= bottleneck[proc]
                        && fast_mem_end[proc].find(to_remove) == fast_mem_end[proc].end())
                    {
                        fast_mem_end[proc].insert(to_remove);
                        bottleneck[proc] -= instance->getComputationalDag().nodeMemoryWeight(to_remove);
                    }
                    else
                        remaining.push_back(to_remove);
                }
                computeStep.nodes_evicted_after = remaining;
                bottleneck[proc] = std::min(bottleneck[proc], instance->getArchitecture().memoryBound(proc) - mem_used[proc]);
                
                mem_used[proc] -= instance->getComputationalDag().nodeMemoryWeight(computeStep.node);
            }
        }
    }

    if(!isValid())
        std::cout<<"ERROR: eviction removal process created an invalid schedule."<<std::endl;
}

void BspMemSchedule::CreateFromPartialPebblings(const BspInstance &bsp_instance, 
                                                const std::vector<BspMemSchedule>& pebblings,
                                                const std::vector<std::set<unsigned> >& processors_to_parts,
                                                const std::vector<std::map<unsigned, unsigned> >& original_node_id,
                                                const std::vector<std::map<unsigned, unsigned> >& original_proc_id,
                                                const std::vector<std::vector<std::set<unsigned> > >& has_reds_in_beginning)
{
    instance = &bsp_instance;

    unsigned nr_parts = processors_to_parts.size();

    std::vector<std::set<unsigned> > in_mem(instance->numberOfProcessors());
    std::vector<std::tuple<unsigned, unsigned, unsigned> > force_evicts;

    compute_steps_for_proc_superstep.clear();
    nodes_sent_up.clear();
    nodes_sent_down.clear();
    nodes_evicted_in_comm.clear();
    compute_steps_for_proc_superstep.resize(instance->numberOfProcessors());
    nodes_sent_up.resize(instance->numberOfProcessors());
    nodes_sent_down.resize(instance->numberOfProcessors());
    nodes_evicted_in_comm.resize(instance->numberOfProcessors());

    std::vector<unsigned> supstep_idx(instance->numberOfProcessors(), 0);

    std::vector<unsigned> gets_blue_in_superstep(instance->numberOfVertices(), UINT_MAX);
    for(unsigned node = 0; node < instance->numberOfVertices(); ++node)
        if(instance->getComputationalDag().numberOfParents(node) == 0)
            gets_blue_in_superstep[node] = 0;

    for(unsigned part = 0; part < nr_parts; ++part)
    {
        unsigned starting_step_index = 0;

        // find dependencies on previous subschedules
        for(unsigned node = 0; node < pebblings[part].instance->numberOfVertices(); ++node)
            if(pebblings[part].instance->getComputationalDag().numberOfParents(node) == 0)
                starting_step_index = std::max(starting_step_index, gets_blue_in_superstep[original_node_id[part].at(node)]);

        // sync starting points for the subset of processors
        for(unsigned proc : processors_to_parts[part])
            starting_step_index = std::max(starting_step_index, supstep_idx[proc]);
        for(unsigned proc : processors_to_parts[part])
            while(supstep_idx[proc] < starting_step_index)
            {
                compute_steps_for_proc_superstep[proc].emplace_back();
                nodes_sent_up[proc].emplace_back();
                nodes_sent_down[proc].emplace_back();
                nodes_evicted_in_comm[proc].emplace_back();
                ++supstep_idx[proc];
            }
        
        // check and update according to initial states of red pebbles
        for(unsigned proc = 0; proc < processors_to_parts[part].size(); ++proc)
        {
            unsigned proc_id = original_proc_id[part].at(proc);
            std::set<unsigned> needed_in_red, add_before, remove_before;
            for(unsigned node : has_reds_in_beginning[part][proc])
            {
                unsigned node_id = original_node_id[part].at(node);
                needed_in_red.insert(node_id);
                if(in_mem[proc_id].find(node_id) == in_mem[proc_id].end())
                    add_before.insert(node_id);
            }
            for(unsigned node : in_mem[proc_id])
                if(needed_in_red.find(node) == needed_in_red.end())
                    remove_before.insert(node);

            if((!add_before.empty() || !remove_before.empty()) && supstep_idx[proc_id] == 0)
            {
                // this code is added just in case - this shouldn't happen in normal schedules
                compute_steps_for_proc_superstep[proc_id].emplace_back();
                nodes_sent_up[proc_id].emplace_back();
                nodes_sent_down[proc_id].emplace_back();
                nodes_evicted_in_comm[proc_id].emplace_back();
                ++supstep_idx[proc_id];
            }

            for(unsigned node : add_before)
            {
                in_mem[proc_id].insert(node);
                nodes_sent_down[proc_id].back().push_back(node);
            }
            for(unsigned node : remove_before)
            {
                in_mem[proc_id].erase(node);
                nodes_evicted_in_comm[proc_id].back().push_back(node);
                force_evicts.push_back(std::make_tuple(node, proc_id, nodes_evicted_in_comm[proc_id].size()-1));
            } 
        }
        
        for(unsigned supstep = 0; supstep < pebblings[part].numberOfSupersteps(); ++supstep)
            for(unsigned proc = 0; proc < processors_to_parts[part].size(); ++proc)
            {
                unsigned proc_id = original_proc_id[part].at(proc);
                compute_steps_for_proc_superstep[proc_id].emplace_back();
                nodes_sent_up[proc_id].emplace_back();
                nodes_sent_down[proc_id].emplace_back();
                nodes_evicted_in_comm[proc_id].emplace_back();

                // copy schedule with translated indeces
                for(const compute_step& computeStep : pebblings[part].GetComputeStepsForProcSuperstep(proc, supstep))
                {
                    compute_steps_for_proc_superstep[proc_id].back().emplace_back();
                    compute_steps_for_proc_superstep[proc_id].back().back().node = original_node_id[part].at(computeStep.node);
                    in_mem[proc_id].insert(original_node_id[part].at(computeStep.node));
                    
                    for(unsigned local_id : computeStep.nodes_evicted_after)
                    {
                        compute_steps_for_proc_superstep[proc_id].back().back().nodes_evicted_after.push_back(original_node_id[part].at(local_id));
                        in_mem[proc_id].erase(original_node_id[part].at(local_id));
                   }
                }
                for(unsigned node : pebblings[part].GetNodesSentUp(proc, supstep))
                {
                    unsigned node_id = original_node_id[part].at(node);
                    nodes_sent_up[proc_id].back().push_back(node_id);
                    gets_blue_in_superstep[node_id] = std::min(gets_blue_in_superstep[node_id], supstep_idx[proc_id]);
                }
                for(unsigned node : pebblings[part].GetNodesEvictedInComm(proc, supstep))
                {
                    nodes_evicted_in_comm[proc_id].back().push_back(original_node_id[part].at(node));
                    in_mem[proc_id].erase(original_node_id[part].at(node));
                }
                for(unsigned node : pebblings[part].GetNodesSentDown(proc, supstep))
                {
                    nodes_sent_down[proc_id].back().push_back(original_node_id[part].at(node));
                    in_mem[proc_id].insert(original_node_id[part].at(node));
                }

                ++supstep_idx[proc_id];
            }    
    }

    // padding supersteps in the end
    unsigned max_step_index = 0;
    for(unsigned proc = 0; proc < instance->numberOfProcessors(); ++proc)
        max_step_index = std::max(max_step_index, supstep_idx[proc]);
    for(unsigned proc = 0; proc < instance->numberOfProcessors(); ++proc)
        while(supstep_idx[proc] < max_step_index)
        {
            compute_steps_for_proc_superstep[proc].emplace_back();
            nodes_sent_up[proc].emplace_back();
            nodes_sent_down[proc].emplace_back();
            nodes_evicted_in_comm[proc].emplace_back();
            ++supstep_idx[proc];
        }
    number_of_supersteps = max_step_index;
    need_to_load_inputs = true;

    FixForceEvicts(*instance, force_evicts);
    TryToMergeSupersteps(*instance);
}

void BspMemSchedule::FixForceEvicts(const BspInstance &bsp_instance, const std::vector<std::tuple<unsigned, unsigned, unsigned> > force_evict_node_proc_step)
{
    // Some values were evicted only because they weren't present in the next part - see if we can undo those evictions
    for(auto force_evict : force_evict_node_proc_step)
    {
        unsigned node = std::get<0>(force_evict);
        unsigned proc = std::get<1>(force_evict);
        unsigned superstep = std::get<2>(force_evict);

        bool next_in_comp = false;
        bool next_in_comm = false;
        std::pair<unsigned, unsigned> where;

        for(unsigned find_supstep = superstep + 1; find_supstep < numberOfSupersteps(); ++find_supstep)
        {
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][find_supstep].size(); ++stepIndex)
                if(compute_steps_for_proc_superstep[proc][find_supstep][stepIndex].node == node)
                {
                    next_in_comp = true;
                    where = std::make_pair(find_supstep, stepIndex);
                    break;
                }
            if(next_in_comp)
                break;
            for(unsigned send_down : nodes_sent_down[proc][find_supstep])
                if(send_down == node)
                {
                    next_in_comm = true;
                    where = std::make_pair(find_supstep, 0);
                    break;
                }
            if(next_in_comm)
                break;
        }

        // check new schedule for validity
        if(!next_in_comp && !next_in_comm)
            continue;
        
        bool eraseit=false;
        BspMemSchedule test_schedule = *this;
        for(auto itr = test_schedule.nodes_evicted_in_comm[proc][superstep].begin(); itr != test_schedule.nodes_evicted_in_comm[proc][superstep].end(); ++itr)
            if(*itr == node)
            {
                test_schedule.nodes_evicted_in_comm[proc][superstep].erase(itr);
                eraseit = true;
                break;
            }

        if(next_in_comp)
        {            
            for(auto itr = test_schedule.compute_steps_for_proc_superstep[proc][where.first].begin(); itr != test_schedule.compute_steps_for_proc_superstep[proc][where.first].end(); ++itr)
                if(itr->node == node)
                {
                    if(where.second > 0)
                    {
                        auto previous_step = itr;
                        --previous_step;
                        for(unsigned to_evict : itr->nodes_evicted_after)
                            previous_step->nodes_evicted_after.push_back(to_evict);
                    }
                    else
                    {
                        for(unsigned to_evict : itr->nodes_evicted_after)
                            test_schedule.nodes_evicted_in_comm[proc][where.first-1].push_back(to_evict);
                    }
                    test_schedule.compute_steps_for_proc_superstep[proc][where.first].erase(itr);
                    break;
                }

            if(test_schedule.isValid())
            {
                nodes_evicted_in_comm[proc][superstep] = test_schedule.nodes_evicted_in_comm[proc][superstep];
                compute_steps_for_proc_superstep[proc][where.first] = test_schedule.compute_steps_for_proc_superstep[proc][where.first];
                nodes_evicted_in_comm[proc][where.first-1] = test_schedule.nodes_evicted_in_comm[proc][where.first-1];
            }
        }
        else if(next_in_comm)
        {
            for(auto itr = test_schedule.nodes_sent_down[proc][where.first].begin(); itr != test_schedule.nodes_sent_down[proc][where.first].end(); ++itr)
                if(*itr == node)
                {
                    test_schedule.nodes_sent_down[proc][where.first].erase(itr);
                    break;
                }
                
            if(test_schedule.isValid())
            {
                nodes_evicted_in_comm[proc][superstep] = test_schedule.nodes_evicted_in_comm[proc][superstep];
                nodes_sent_down[proc][where.first] = test_schedule.nodes_sent_down[proc][where.first];
            }
        }
    }
}

void BspMemSchedule::TryToMergeSupersteps(const BspInstance &bsp_instance)
{
    std::vector<bool> is_removed(number_of_supersteps, false);

    for(unsigned step = 1; step < number_of_supersteps; ++step)
    {
        if(is_removed[step])
            continue;

        unsigned prev_step = step - 1;
        while(is_removed[prev_step])
            --prev_step;

        for(unsigned next_step = step + 1; next_step < number_of_supersteps; ++next_step)
        {
            // Try to merge step and next_step
            BspMemSchedule test_schedule = *this;

            for(unsigned proc = 0; proc < instance->numberOfProcessors(); ++proc)
            {
                test_schedule.compute_steps_for_proc_superstep[proc][step].insert(
                        test_schedule.compute_steps_for_proc_superstep[proc][step].end(),
                        test_schedule.compute_steps_for_proc_superstep[proc][next_step].begin(),
                        test_schedule.compute_steps_for_proc_superstep[proc][next_step].end());
                test_schedule.compute_steps_for_proc_superstep[proc][next_step].clear();
                
                test_schedule.nodes_sent_up[proc][step].insert(
                        test_schedule.nodes_sent_up[proc][step].end(),
                        test_schedule.nodes_sent_up[proc][next_step].begin(),
                        test_schedule.nodes_sent_up[proc][next_step].end());
                test_schedule.nodes_sent_up[proc][next_step].clear();

                test_schedule.nodes_sent_down[proc][prev_step].insert(
                        test_schedule.nodes_sent_down[proc][prev_step].end(),
                        test_schedule.nodes_sent_down[proc][step].begin(),
                        test_schedule.nodes_sent_down[proc][step].end());
                test_schedule.nodes_sent_down[proc][step].clear();

                test_schedule.nodes_evicted_in_comm[proc][step].insert(
                        test_schedule.nodes_evicted_in_comm[proc][step].end(),
                        test_schedule.nodes_evicted_in_comm[proc][next_step].begin(),
                        test_schedule.nodes_evicted_in_comm[proc][next_step].end());
                test_schedule.nodes_evicted_in_comm[proc][next_step].clear();

            }

            if(test_schedule.isValid())
            {
                is_removed[next_step] = true;
                for(unsigned proc = 0; proc < instance->numberOfProcessors(); ++proc)
                {
                    compute_steps_for_proc_superstep[proc][step] = test_schedule.compute_steps_for_proc_superstep[proc][step];
                    compute_steps_for_proc_superstep[proc][next_step].clear();
                    
                    nodes_sent_up[proc][step] = test_schedule.nodes_sent_up[proc][step];
                    nodes_sent_up[proc][next_step].clear();

                    nodes_sent_down[proc][prev_step] = test_schedule.nodes_sent_down[proc][prev_step];
                    nodes_sent_down[proc][step] = nodes_sent_down[proc][next_step];
                    nodes_sent_down[proc][next_step].clear();

                    nodes_evicted_in_comm[proc][step] = test_schedule.nodes_evicted_in_comm[proc][step];
                    nodes_evicted_in_comm[proc][next_step].clear();
                }
            }
            else
                break;
        }
    }

    unsigned new_nr_supersteps = 0;
    for(unsigned step = 0; step < number_of_supersteps; ++step)
        if(!is_removed[step])
            ++new_nr_supersteps;
    
    if(new_nr_supersteps == number_of_supersteps)
        return;

    BspMemSchedule shortened_schedule = *this;
    shortened_schedule.updateNumberOfSupersteps(new_nr_supersteps);

    unsigned new_index = 0;
    for(unsigned step = 0; step < number_of_supersteps; ++step)
    {
        if(is_removed[step])
            continue;

        for(unsigned proc = 0; proc < instance->numberOfProcessors(); ++proc)
        {
            shortened_schedule.compute_steps_for_proc_superstep[proc][new_index] = compute_steps_for_proc_superstep[proc][step];
            shortened_schedule.nodes_sent_up[proc][new_index] = nodes_sent_up[proc][step];
            shortened_schedule.nodes_sent_down[proc][new_index] = nodes_sent_down[proc][step];
            shortened_schedule.nodes_evicted_in_comm[proc][new_index] = nodes_evicted_in_comm[proc][step];
        }

        ++new_index;
    }
    
    *this = shortened_schedule;

    if(!isValid())
        std::cout<<"ERROR: schedule is not valid after superstep merging."<<std::endl;

}

BspMemSchedule BspMemSchedule::ExpandMemSchedule(const BspInstance& original_instance, const std::vector<unsigned> mapping_to_coarse) const
{
    std::map<unsigned, std::set<unsigned> > original_vertices_for_coarse_ID;
    for(unsigned node = 0; node < original_instance.numberOfVertices(); ++node)
        original_vertices_for_coarse_ID[mapping_to_coarse[node]].insert(node);

    BspMemSchedule fine_schedule;
    fine_schedule.instance = &original_instance;
    fine_schedule.updateNumberOfSupersteps(number_of_supersteps);

    for(unsigned step=0; step<number_of_supersteps; ++step)
    {
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            // computation phase
            for(const auto& computeStep : compute_steps_for_proc_superstep[proc][step])
            {
                unsigned node = computeStep.node;
                for(unsigned original_node : original_vertices_for_coarse_ID[node])
                    fine_schedule.compute_steps_for_proc_superstep[proc][step].emplace_back(original_node);

                for(unsigned to_remove : computeStep.nodes_evicted_after)
                    for(unsigned original_node : original_vertices_for_coarse_ID[to_remove])
                        fine_schedule.compute_steps_for_proc_superstep[proc][step].back().nodes_evicted_after.push_back(original_node);
            }

            //communication phase
            for(unsigned node : nodes_sent_up[proc][step])
                for(unsigned original_node : original_vertices_for_coarse_ID[node])
                    fine_schedule.nodes_sent_up[proc][step].push_back(original_node);
            
            for(unsigned node : nodes_evicted_in_comm[proc][step])
                for(unsigned original_node : original_vertices_for_coarse_ID[node])
                    fine_schedule.nodes_evicted_in_comm[proc][step].push_back(original_node);

            for(unsigned node : nodes_sent_down[proc][step])
                for(unsigned original_node : original_vertices_for_coarse_ID[node])
                    fine_schedule.nodes_sent_down[proc][step].push_back(original_node);
        }
    }

    fine_schedule.cleanSchedule();
    return fine_schedule;
}

BspSchedule BspMemSchedule::ConvertToBsp() const
{
    std::vector<unsigned> node_to_proc(instance->numberOfVertices(), UINT_MAX), node_to_supstep(instance->numberOfVertices(), UINT_MAX);

    for(unsigned step=0; step<number_of_supersteps; ++step)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(const auto& computeStep : compute_steps_for_proc_superstep[proc][step])
            {
                const unsigned& node = computeStep.node;             
                if(node_to_proc[node] == UINT_MAX)
                {
                    node_to_proc[node] = proc;
                    node_to_supstep[node] = step;
                }
            }
    if(need_to_load_inputs)
        for(unsigned node = 0; node < instance->numberOfVertices(); ++node)
            if(instance->getComputationalDag().numberOfParents(node) == 0)
            {
                unsigned min_superstep = UINT_MAX, proc_chosen = 0;
                for(unsigned succ : instance->getComputationalDag().children(node))
                    if(node_to_supstep[succ] < min_superstep)
                    {
                        min_superstep = node_to_supstep[succ];
                        proc_chosen = node_to_proc[succ];
                    }
                node_to_supstep[node] = min_superstep;
                node_to_proc[node] = proc_chosen;
            }

    BspSchedule schedule(*instance, node_to_proc, node_to_supstep);
    if(schedule.satisfiesPrecedenceConstraints() && schedule.satisfiesNodeTypeConstraints())
    {
        schedule.setAutoCommunicationSchedule();
        return schedule;
    }
    else
    {
        std::cout<<"ERROR: no direct conversion to Bsp schedule exists, using dummy schedule instead."<<std::endl;
        return BspSchedule(*instance);
    }
}