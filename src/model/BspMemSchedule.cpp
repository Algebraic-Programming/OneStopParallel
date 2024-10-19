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

    std::vector<int> current_time_at_processor(instance->getArchitecture().numberOfProcessors(), 0);
    std::vector<int> time_when_node_gets_blue(instance->getComputationalDag().numberOfVertices(), INT_MAX);

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

    int makespan = 0;
    for(unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); ++proc)
        if(current_time_at_processor[proc] > makespan)
            makespan = current_time_at_processor[proc];

    return makespan;
}

void BspMemSchedule::ConvertFromBsp(const BspSchedule &schedule, CACHE_EVICTION_STRATEGY evict_rule)
{
    // check if conversion possible at all
    unsigned memory_required = minimumMemoryRequired(*instance);
    if(memory_limit < memory_required)
    {
        std::cout<<"Conversion failed. Minimum memory required is "<<memory_required<<std::endl;
        return;
    }

    // split supersteps
    SplitSupersteps(schedule);

    // track memory
    SetMemoryMovement(evict_rule);    
}

void BspMemSchedule::SplitSupersteps(const BspSchedule &schedule)
{
    // get DFS topological order in each superstep
    std::vector<std::vector<std::vector<unsigned> > > top_orders = computeTopOrdersDFS(schedule);

    std::vector<unsigned> top_order_idx(instance->getComputationalDag().numberOfVertices());
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
                unsigned end_lower_bound = start_idx, end_upper_bound = top_orders[proc][step].size()-1;
                while(end_lower_bound < end_upper_bound)
                {
                    unsigned end_current = end_lower_bound + (end_upper_bound - end_lower_bound + 1) / 2;

                    // check if this segment is valid
                    bool valid = true;

                    std::map<int, bool> neededAfter;
                    for(unsigned idx = start_idx; idx <= end_current; ++idx)
                    {
                        unsigned node = top_orders[proc][step][idx];
                        neededAfter[node] = false;
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
                            if(schedule.assignedSuperstep(pred)<step || (schedule.assignedSuperstep(pred)==step && top_order_idx[pred] < start_idx))
                                values_needed.insert(pred);
                        }
                    }

                    unsigned mem_needed = 0;
                    for(unsigned node : values_needed)
                        mem_needed += instance->getComputationalDag().nodeWorkWeight(node);
        

                    for(unsigned idx = start_idx; idx <= end_current; ++idx)
                    {
                        unsigned node = top_orders[proc][step][idx];

                        mem_needed += instance->getComputationalDag().nodeWorkWeight(node);
                        if(mem_needed > memory_limit)
                        {
                            valid = false;
                            break;
                        }

                        for(unsigned pred : instance->getComputationalDag().parents(node))
                            if(lastUsedBy[pred] == node)
                                mem_needed -= instance->getComputationalDag().nodeWorkWeight(pred);
                    }

                    if(valid)
                        end_lower_bound = end_current;
                    else
                        end_upper_bound = end_current - 1;

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
    
    updateNumberOfSupersteps(superstep_index+1);
    std::cout<<schedule.numberOfSupersteps()<<" -> "<<number_of_supersteps<<std::endl;

    for(unsigned step=0; step<schedule.numberOfSupersteps(); ++step)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned node : top_orders[proc][step])
                compute_steps_for_proc_superstep[proc][new_superstep_ID[node]].emplace_back(node);
}

void BspMemSchedule::SetMemoryMovement(CACHE_EVICTION_STRATEGY evict_rule)
{
    // Note - this currently uses work weights instead of memory weights, to allow testing with our current inputs

    const unsigned N = instance->getComputationalDag().numberOfVertices();

    std::vector<unsigned> mem_used(instance->numberOfProcessors(), 0);
    std::vector<std::set<unsigned> > in_mem(instance->numberOfProcessors());
    std::vector<std::set<std::pair<std::pair<unsigned, unsigned>, unsigned>> > evictable(instance->numberOfProcessors());
    std::vector<std::set<unsigned> > non_evictable(instance->numberOfProcessors());

    std::vector<bool> in_slow_mem(N, false);
    
    // superstep-step pairs where a node is required (on a given proc)
    std::vector<std::vector<std::deque<std::pair<unsigned, unsigned> > > > used_at(N, std::vector<std::deque<std::pair<unsigned, unsigned> > >(instance->numberOfProcessors()));
    
    // iterator to its position in "evictable" - for efficient delete
    std::vector<std::vector<std::set<std::pair<std::pair<unsigned, unsigned>, unsigned> >::iterator > > place_in_evictable(N,
        std::vector<std::set<std::pair<std::pair<unsigned, unsigned>, unsigned> >::iterator>(instance->numberOfProcessors()));

    // utility for LRU eviction strategy
    std::vector<std::vector<unsigned> > node_last_used_on_proc;
    if(evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED)
        node_last_used_on_proc.resize(N, std::vector<unsigned>(instance->numberOfProcessors(), 0));
    std::vector<unsigned> total_step_count_on_proc(instance->numberOfProcessors(), 0);

    // NOTE - conversion currently assumes that input schedule has no recomputation -> TODO for later
    std::vector<unsigned> assigned_processor(N);
    std::vector<unsigned> assigned_superstep(N);
    for(unsigned superstep=0; superstep<number_of_supersteps; ++superstep)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                assigned_processor[node] = proc;
                assigned_superstep[node] = superstep;
            }

    for(unsigned superstep=0; superstep<number_of_supersteps; ++superstep)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                for(unsigned pred : instance->getComputationalDag().parents(node))
                    used_at[pred][proc].emplace_back(superstep, stepIndex);
            }
    
    for(unsigned node=0; node<N; ++node)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            place_in_evictable[node][proc] = evictable[proc].end();

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
                for(unsigned pred : instance->getComputationalDag().parents(node))
                    if(assigned_superstep[pred] < superstep)
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
            
            for(unsigned node : new_values_needed)
            {
                in_mem[proc].insert(node);
                mem_used[proc] += instance->getComputationalDag().nodeWorkWeight(node);
                nodes_sent_down[proc][superstep-1].push_back(node);
                if(!in_slow_mem[node])
                {
                    in_slow_mem[node] = true;
                    nodes_sent_up[assigned_processor[node]][assigned_superstep[node]].push_back(node);
                }
            }

            unsigned first_node_weight = instance->getComputationalDag().nodeWorkWeight(compute_steps_for_proc_superstep[proc][superstep][0].node);

            while(mem_used[proc] + first_node_weight > memory_limit) // no sliding pebbles for now
            {
                unsigned evicted = (--evictable[proc].end())->second;
                evictable[proc].erase(--evictable[proc].end());
                place_in_evictable[evicted][proc] = evictable[proc].end();

                mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(evicted);
                in_mem[proc].erase(evicted);

                nodes_evicted_in_comm[proc][superstep-1].push_back(evicted);
                if(!in_slow_mem[evicted])
                {
                    in_slow_mem[evicted] = true;
                    nodes_sent_up[assigned_processor[evicted]][assigned_superstep[evicted]].push_back(evicted);
                }
            }

            // check if the new nodes will be needed after (and thus cannot be deleted during) this compute phase
            std::map<unsigned, bool> needed_after;
            for(const auto& computeStep : compute_steps_for_proc_superstep[proc][superstep])
            {
                needed_after[computeStep.node] = false;
                for(unsigned succ : instance->getComputationalDag().children(computeStep.node))
                    if(assigned_superstep[succ] > superstep)
                        needed_after[computeStep.node] = true;
            }
            

            // during compute phase
            for(unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][superstep].size(); ++stepIndex)
            {
                unsigned node = compute_steps_for_proc_superstep[proc][superstep][stepIndex].node;
                unsigned node_weight = instance->getComputationalDag().nodeWorkWeight(node);

                if(stepIndex > 0)
                {
                    //evict nodes to make space
                    while(mem_used[proc] + node_weight > memory_limit)
                    {
                        unsigned evicted = (--evictable[proc].end())->second;
                        evictable[proc].erase(--evictable[proc].end());
                        place_in_evictable[evicted][proc] = evictable[proc].end();

                        mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(evicted);
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
                
                for(unsigned pred : instance->getComputationalDag().parents(node))
                {
                    used_at[pred][proc].pop_front();

                    // autoevict
                    if(used_at[pred][proc].empty() && (assigned_superstep[pred]<superstep || !needed_after[pred]))
                    {
                        in_mem[proc].erase(pred);
                        non_evictable[proc].erase(pred);
                        mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(pred);
                        compute_steps_for_proc_superstep[proc][superstep][stepIndex].nodes_evicted_after.push_back(pred);
                    }
                    else if(assigned_superstep[pred]<superstep && used_at[pred][proc].front().first > superstep)
                    {
                        non_evictable[proc].erase(pred);

                        std::pair<unsigned, unsigned> prio;
                        if(evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT)
                            prio = used_at[pred][proc].front();
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
                if(used_at[node][proc].empty())
                {
                    mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(node);
                    in_mem[proc].erase(node);
                    nodes_evicted_in_comm[proc][superstep].push_back(node);
                    in_slow_mem[node] = true;
                    nodes_sent_up[proc][superstep].push_back(node);
                }
                else
                {
                    std::pair<unsigned, unsigned> prio;
                    if(evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT)
                        prio = used_at[node][proc].front();
                    else if(evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED)
                        prio = std::make_pair(UINT_MAX - node_last_used_on_proc[node][proc], node);
                    else if(evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID)
                        prio = std::make_pair(node, 0);

                    place_in_evictable[node][proc] = evictable[proc].emplace(prio, node).first;
                }
            }
            non_evictable[proc].clear();
        }
    }

}

bool BspMemSchedule::isValid()
{
    std::vector<unsigned> mem_used(instance->numberOfProcessors(), 0);
    std::vector<std::vector<unsigned> > in_fast_mem(instance->getComputationalDag().numberOfVertices(),
         std::vector<unsigned>(instance->numberOfProcessors(), false));
    std::vector<unsigned> in_slow_mem(instance->getComputationalDag().numberOfVertices(), false);

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
                
                in_fast_mem[computeStep.node][proc] = true;
                mem_used[proc] += instance->getComputationalDag().nodeWorkWeight(computeStep.node);

                if(mem_used[proc] > memory_limit)
                        return false;

                for(unsigned to_remove : computeStep.nodes_evicted_after)
                {
                    if(!in_fast_mem[to_remove][proc])
                        return false;

                    in_fast_mem[to_remove][proc] = false;
                    mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(to_remove);
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
                mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(node);
            }
        }

        // communication phase - senddown
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            for(unsigned node : nodes_sent_down[proc][step])
            {
                if(!in_slow_mem[node])
                    return false;

                in_fast_mem[node][proc] = true;
                mem_used[proc] += instance->getComputationalDag().nodeWorkWeight(node);
            }
        }

        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            if(mem_used[proc] > memory_limit)
                return false;
    }

    return true;
}

unsigned BspMemSchedule::minimumMemoryRequired(const BspInstance& instance)
{
    unsigned max_needed = 0;
    for(unsigned node=0; node<instance.getComputationalDag().numberOfVertices(); ++node)
    {
        // TODO change to memory weight once we have those!
        unsigned needed = instance.getComputationalDag().nodeWorkWeight(node);
        for(unsigned pred : instance.getComputationalDag().parents(node))
            needed += instance.getComputationalDag().nodeWorkWeight(pred);
        
        if(needed>max_needed)
            max_needed=needed;
    }
    return max_needed;
}

std::vector<std::vector<std::vector<unsigned> > > BspMemSchedule::computeTopOrdersDFS(const BspSchedule &schedule)
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
            if(schedule.assignedProcessor(node)==schedule.assignedProcessor(pred)
            && schedule.assignedSuperstep(node)==schedule.assignedSuperstep(pred))
                ++predecessors;
        nr_pred[node] = predecessors;
        if(predecessors==0)
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