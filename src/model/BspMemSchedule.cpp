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
#include "model/SetSchedule.hpp"

void BspMemSchedule::updateNumberOfSupersteps() {

    number_of_supersteps = 0;

    for (unsigned i = 0; i < instance->numberOfVertices(); ++i) {

        if (node_to_superstep_assignment[i] >= number_of_supersteps) {
            number_of_supersteps = node_to_superstep_assignment[i] + 1;
        }
    }

    top_order_for_proc_superstep.clear();
    top_order_for_proc_superstep.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    nodes_evicted_in_comm.clear();
    nodes_evicted_in_comm.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    nodes_sent_down.clear();
    nodes_sent_down.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));

    nodes_sent_up.clear();
    nodes_sent_up.resize(instance->numberOfProcessors(), std::vector<std::vector<unsigned> >(number_of_supersteps));
}

void BspMemSchedule::setAssignedSuperstep(unsigned node, unsigned superstep) {

    if (node < instance->numberOfVertices()) {
        node_to_superstep_assignment[node] = superstep;

        if (superstep >= number_of_supersteps) {
            number_of_supersteps = superstep + 1;
        }

    } else {
        throw std::invalid_argument("Invalid Argument while assigning node to superstep: index out of range.");
    }
}

void BspMemSchedule::setAssignedProcessor(unsigned node, unsigned processor) {

    if (node < instance->numberOfVertices() && processor < instance->numberOfProcessors()) {
        node_to_processor_assignment[node] = processor;
    } else {
        // std::cout << "node " << node << " num nodes " << instance->numberOfVertices() << "  processor " << processor
        //          << " num proc " << instance->numberOfProcessors() << std::endl;
        throw std::invalid_argument("Invalid Argument while assigning node to processor");
    }
}

/*void BspMemSchedule::addCommunicationScheduleEntry(unsigned node, unsigned from_proc, unsigned to_proc, unsigned step) {
    addCommunicationScheduleEntry(std::make_tuple(node, from_proc, to_proc), step);
}

void BspMemSchedule::addCommunicationScheduleEntry(KeyTriple key, unsigned step) {

    if (step >= number_of_supersteps)
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: step out of range.");

    if (get<0>(key) >= instance->numberOfVertices())
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: node out of range.");

    if (get<1>(key) >= instance->numberOfProcessors())
        throw std::invalid_argument(
            "Invalid Argument while adding communication schedule entry: from processor out of range.");

    if (get<2>(key) >= instance->numberOfProcessors())
        throw std::invalid_argument(
            "Invalid Argument while adding communication schedule entry: to processor out of range.");

    commSchedule[key] = step;
}*/

void BspMemSchedule::setAssignedSupersteps(const std::vector<unsigned> &vec) {

    if (vec.size() == instance->numberOfVertices()) {
        for (unsigned i = 0; i < instance->numberOfVertices(); ++i) {

            if (vec[i] >= number_of_supersteps) {
                number_of_supersteps = vec[i] + 1;
            }

            node_to_superstep_assignment[i] = vec[i];
        }
    } else {
        throw std::invalid_argument(
            "Invalid Argument while assigning supersteps: size does not match number of nodes.");
    }
}

void BspMemSchedule::setAssignedProcessors(const std::vector<unsigned> &vec) {

    if (vec.size() == instance->numberOfVertices()) {
        for (unsigned i = 0; i < instance->numberOfVertices(); ++i) {

            if (vec[i] >= instance->numberOfProcessors()) {
                throw std::invalid_argument(
                    "Invalid Argument while assigning processors: processor index out of range.");
            }

            node_to_processor_assignment[i] = vec[i];
        }
    } else {
        throw std::invalid_argument(
            "Invalid Argument while assigning processors: size does not match number of nodes.");
    }
}

std::vector<unsigned> BspMemSchedule::getAssignedNodeVector(unsigned processor) const {

    std::vector<unsigned> vec;

    for (unsigned i = 0; i < instance->numberOfVertices(); i++) {

        if (node_to_processor_assignment[i] == processor) {
            vec.push_back(i);
        }
    }

    return vec;
}

std::vector<unsigned int> BspMemSchedule::getAssignedNodeVector(unsigned processor, unsigned superstep) const {
    std::vector<unsigned int> vec;

    for (unsigned int i = 0; i < instance->numberOfVertices(); i++) {

        if (node_to_processor_assignment[i] == processor && node_to_superstep_assignment[i] == superstep) {
            vec.push_back(i);
        }
    }

    return vec;
}

bool BspMemSchedule::satisfiesPrecedenceConstraints() const {

    if (node_to_processor_assignment.size() != instance->numberOfVertices() ||
        node_to_superstep_assignment.size() != instance->numberOfVertices()) {
        return false;
    }

    // bool comm_edge_found = false;

    for (const auto &ep : instance->getComputationalDag().edges()) {
        const unsigned &source = instance->getComputationalDag().source(ep);
        const unsigned &target = instance->getComputationalDag().target(ep);

        const int different_processors =
            (node_to_processor_assignment[source] == node_to_processor_assignment[target]) ? 0 : 1;

        if (node_to_superstep_assignment[source] + different_processors > node_to_superstep_assignment[target]) {
            // std::cout << "This is not a valid scheduling (problems with nodes " << source << " and " << target <<
            // ")."
            //           << std::endl; // todo should be removed
            return false;
        }
    }

    return true;
};


/*void BspMemSchedule::setCommunicationSchedule(const std::map<KeyTriple, unsigned int> &cs) {
    if (checkCommScheduleValidity(cs)) {
        commSchedule.clear();
        commSchedule = std::map<KeyTriple, unsigned int>(cs);

    } else {
        throw std::invalid_argument("Given communication schedule is not valid for instance");
    }
}*/

unsigned BspMemSchedule::computeWorkCosts() const {

    assert(satisfiesPrecedenceConstraints());

    std::vector<std::vector<unsigned>> work = std::vector<std::vector<unsigned>>(
        number_of_supersteps, std::vector<unsigned>(instance->numberOfProcessors(), 0));

    for (unsigned node = 0; node < instance->numberOfVertices(); node++) {
        work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]] +=
            instance->getComputationalDag().nodeWorkWeight(node);
    }

    unsigned total_costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        unsigned max_work = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (max_work < work[step][proc]) {
                max_work = work[step][proc];
            }
        }

        total_costs += max_work;
    }

    return total_costs;
}


BspMemSchedule::BspMemSchedule(const BspSchedule &schedule, unsigned Mem_limit, bool clever_evict)
    : instance(&schedule.getInstance()), memory_limit(Mem_limit)
{
    // check if conversion possible at all
    unsigned memory_required = minimumMemoryRequired(*instance);
    if(Mem_limit < memory_required)
    {
        std::cout<<"Conversion failed. Minimum memory required is "<<memory_required<<std::endl;
        return;
    }

    std::cout<<"Minimum memory required: "<<memory_required<<std::endl;
    // resize vectors
    nodes_evicted_after.clear();
    nodes_evicted_after.resize(instance->getComputationalDag().numberOfVertices());

    // initialize to original schedule
    node_to_processor_assignment = schedule.assignedProcessors();
    node_to_superstep_assignment = schedule.assignedSupersteps();

    // split supersteps
    SplitSupersteps(schedule);

    // track memory
    SetMemoryMovement(clever_evict);    
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
                            if(assignedSuperstep(succ)>step)
                                neededAfter[node] = true;
                            if(assignedSuperstep(succ) == step && top_order_idx[succ] <= end_current)
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
                            if(assignedSuperstep(pred)<step || (assignedSuperstep(pred)==step && !neededAfter[pred]))
                                lastUsedBy[pred] = node;
                            if(assignedSuperstep(pred)<step || (assignedSuperstep(pred)==step && top_order_idx[pred] < start_idx))
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

    for(unsigned node=0; node < instance->getComputationalDag().numberOfVertices(); ++node)
        setAssignedSuperstep(node, new_superstep_ID[node]);
    
    updateNumberOfSupersteps();
    std::cout<<schedule.numberOfSupersteps()<<" -> "<<number_of_supersteps<<std::endl;

    for(unsigned step=0; step<schedule.numberOfSupersteps(); ++step)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned node : top_orders[proc][step])
                top_order_for_proc_superstep[proc][assignedSuperstep(node)].push_back(node);
}

void BspMemSchedule::SetMemoryMovement(bool clever_evict)
{
    std::vector<unsigned> mem_used(instance->numberOfProcessors(), 0);
    std::vector<std::set<unsigned> > in_mem(instance->numberOfProcessors());
    std::vector<std::set<std::pair<unsigned, unsigned>> > evictable(instance->numberOfProcessors());
    std::vector<std::set<unsigned> > non_evictable(instance->numberOfProcessors());

    std::vector<bool> in_slow_mem(instance->getComputationalDag().numberOfVertices(), false);

    // last point where each node is required on each proc - autoevict afterwards
    std::vector<std::vector<unsigned> > last_used_at(instance->getComputationalDag().numberOfVertices(),
        std::vector<unsigned>(instance->numberOfProcessors(), instance->getComputationalDag().numberOfVertices()));
    
    // supersteps where each node is required on a given proc
    std::vector<std::vector<std::deque<unsigned> > > used_at(instance->getComputationalDag().numberOfVertices(),
        std::vector<std::deque<unsigned> >(instance->numberOfProcessors()));
    
    // iterator to place in evictable
    std::vector<std::vector<std::set<std::pair<unsigned, unsigned> >::iterator > > place_in_evictable(instance->getComputationalDag().numberOfVertices(),
        std::vector<std::set<std::pair<unsigned, unsigned> >::iterator>(instance->numberOfProcessors()));

    for(unsigned step=0; step<number_of_supersteps; ++step)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            for(unsigned node : top_order_for_proc_superstep[proc][step])
            {
                for(unsigned pred : instance->getComputationalDag().parents(node))
                    last_used_at[pred][proc] = node;
                for(unsigned succ : instance->getComputationalDag().children(node))
                    used_at[node][proc].push_back(assignedSuperstep(succ));
            }
    
    for(unsigned node=0; node<instance->getComputationalDag().numberOfVertices(); ++node)
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
            place_in_evictable[node][proc] = evictable[proc].end();

    // iterate through schedule
    for(unsigned step=0; step<number_of_supersteps; ++step)
    {
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            if(top_order_for_proc_superstep[proc][step].empty())
                continue;

            // before compute phase
            std::map<unsigned, unsigned> last_used_in_this_superstep;
            std::set<unsigned> new_values_needed;
            for(unsigned node : top_order_for_proc_superstep[proc][step])
                for(unsigned pred : instance->getComputationalDag().parents(node))
                    if(assignedSuperstep(pred) < step)
                    {
                        non_evictable[proc].insert(pred);

                        if(place_in_evictable[pred][proc] != evictable[proc].end())
                        {
                            evictable[proc].erase(place_in_evictable[pred][proc]);
                            place_in_evictable[pred][proc] = evictable[proc].end();
                        }

                        if(in_mem[proc].find(pred) == in_mem[proc].end())
                            new_values_needed.insert(pred);

                        last_used_in_this_superstep[pred] = node;
                    }
            
            for(unsigned node : new_values_needed)
            {
                in_mem[proc].insert(node);
                mem_used[proc] += instance->getComputationalDag().nodeWorkWeight(node);
                nodes_sent_down[proc][step-1].push_back(node);
                if(!in_slow_mem[node])
                {
                    in_slow_mem[node] = true;
                    nodes_sent_up[assignedProcessor(node)][assignedSuperstep(node)].push_back(node);
                }
            }

            unsigned first_node_weight = instance->getComputationalDag().nodeWorkWeight(top_order_for_proc_superstep[proc][step][0]);

            while(mem_used[proc] + first_node_weight > memory_limit)
            {
                unsigned evicted = (--evictable[proc].end())->second;
                evictable[proc].erase(--evictable[proc].end());
                place_in_evictable[evicted][proc] = evictable[proc].end();

                mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(evicted);
                in_mem[proc].erase(evicted);

                nodes_evicted_in_comm[proc][step-1].push_back(evicted);
                if(!in_slow_mem[evicted])
                {
                    in_slow_mem[evicted] = true;
                    nodes_sent_up[assignedProcessor(evicted)][assignedSuperstep(evicted)].push_back(evicted);
                }
            }

            // check if new nodes will be needed after this phase
            std::map<unsigned, bool> needed_after;
            for(unsigned node : top_order_for_proc_superstep[proc][step])
            {
                needed_after[node] = false;
                for(unsigned succ : instance->getComputationalDag().children(node))
                    if(assignedSuperstep(succ) > step)
                        needed_after[node] = true;
                if(instance->getComputationalDag().numberOfChildren(node) == 0)
                    needed_after[node] = true;
            }

            // during compute phase
            for(unsigned idx = 0; idx < top_order_for_proc_superstep[proc][step].size(); ++idx)
            {
                unsigned node = top_order_for_proc_superstep[proc][step][idx];
                unsigned node_weight = instance->getComputationalDag().nodeWorkWeight(node);

                if(idx > 0)
                {
                    //evict nodes to make space
                    while(mem_used[proc] + node_weight > memory_limit)
                    {
                        unsigned evicted = (--evictable[proc].end())->second;
                        evictable[proc].erase(--evictable[proc].end());
                        place_in_evictable[evicted][proc] = evictable[proc].end();

                        mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(evicted);
                        in_mem[proc].erase(evicted);

                        nodes_evicted_after[top_order_for_proc_superstep[proc][step][idx-1]].push_back(evicted);
                    }
                }

                in_mem[proc].insert(node);
                mem_used[proc] += node_weight;

                non_evictable[proc].insert(node);
                
                for(unsigned pred : instance->getComputationalDag().parents(node))
                {
                    used_at[pred][proc].pop_front();

                    // autoevict
                    if(last_used_at[pred][proc] == node && (assignedSuperstep(pred)<step || !needed_after[pred]))
                    {
                        in_mem[proc].erase(pred);
                        non_evictable[proc].erase(pred);
                        mem_used[proc] -= instance->getComputationalDag().nodeWorkWeight(pred);
                        nodes_evicted_after[node].push_back(pred);
                    }
                    else if(assignedSuperstep(pred)<step && last_used_in_this_superstep[pred] == node)
                    {
                        non_evictable[proc].erase(pred);

                        unsigned prio = clever_evict ? used_at[pred][proc].front() : 2000000-pred;
                        place_in_evictable[pred][proc] = evictable[proc].emplace(prio, pred).first;
                    }
                }
                
            }


            // after compute phase
            for(unsigned node : non_evictable[proc])
            {
                unsigned prio = clever_evict ? used_at[node][proc].front() : 2000000-node;
                place_in_evictable[node][proc] = evictable[proc].emplace(prio, node).first;
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
            for(unsigned node : top_order_for_proc_superstep[proc][step])
            {
                for(unsigned pred : instance->getComputationalDag().parents(node))
                    if(!in_fast_mem[pred][proc])
                        return false;
                
                in_fast_mem[node][proc] = true;
                mem_used[proc] += instance->getComputationalDag().nodeWorkWeight(node);

                if(mem_used[proc] > memory_limit)
                    return false;

                for(unsigned to_remove : nodes_evicted_after[node])
                {
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

unsigned BspMemSchedule::getCost()
{
    unsigned cost = 0;

    for(unsigned step=0; step<number_of_supersteps; ++step)
    {
        unsigned max_compute = 0;
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            unsigned compute_cost = 0;
            for(unsigned node : top_order_for_proc_superstep[proc][step])
                compute_cost += instance->getComputationalDag().nodeWorkWeight(node);
            
            if(compute_cost > max_compute)
                max_compute = compute_cost;
        }
        cost += max_compute;

        unsigned max_up = 0, max_down = 0;
        for(unsigned proc=0; proc<instance->numberOfProcessors(); ++proc)
        {
            unsigned cost_up = 0, cost_down = 0;
            for(unsigned node : nodes_sent_up[proc][step])
                cost_up += instance->getComputationalDag().nodeWorkWeight(node);
            for(unsigned node : nodes_sent_down[proc][step])
                cost_down += instance->getComputationalDag().nodeWorkWeight(node);
            
            if(cost_up > max_up)
                max_up = cost_up;
            if(cost_down > max_down)
                max_down = cost_down;
        }
        cost += (max_up + max_down) * instance->communicationCosts();

        if(max_up + max_down > 0)
            cost += instance->synchronisationCosts();

    }

    return cost;
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