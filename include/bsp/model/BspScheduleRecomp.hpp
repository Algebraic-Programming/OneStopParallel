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

#pragma once

#include "bsp/model/BspScheduleCS.hpp"
#include "concepts/computational_dag_concept.hpp"

namespace osp {

template<typename Graph_t>
class BspScheduleRecomp {

  public:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = v_workw_t<Graph_t>;

    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;

    static_assert(is_computational_dag_v<Graph_t>, "BspScheduleRecomp can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t> >, "BspScheduleRecomp requires work and comm. weights to have the same type."); 

  private:

    const BspInstance<Graph_t> *instance;

    unsigned int number_of_supersteps = 0;

    std::vector<std::vector<std::pair<unsigned, unsigned>>> node_to_processor_and_supertep_assignment;

    std::map<KeyTriple, unsigned> commSchedule;

  public:

    BspScheduleRecomp() = default;

    BspScheduleRecomp(const BspInstance<Graph_t> &inst) : instance(&inst)
                            {node_to_processor_and_supertep_assignment.resize(inst.numberOfVertices());}

    BspScheduleRecomp(const BspScheduleCS<Graph_t> &schedule);
    BspScheduleRecomp(const BspSchedule<Graph_t> &schedule) : BspScheduleRecomp<Graph_t>(BspScheduleCS<Graph_t>(schedule)) {}

    virtual ~BspScheduleRecomp() = default;

    const BspInstance<Graph_t> &getInstance() const { return *instance; }

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    unsigned numberOfSupersteps() const { return number_of_supersteps; }
    void setNumberOfSupersteps(unsigned number_of_supersteps_) { number_of_supersteps = number_of_supersteps_; }

    std::vector<std::pair<unsigned, unsigned>>& assignments(vertex_idx node) {
        return node_to_processor_and_supertep_assignment[node];
    }

    const std::vector<std::pair<unsigned, unsigned>>& assignments(vertex_idx node) const {
        return node_to_processor_and_supertep_assignment[node];
    }


    /**
     * @brief Sets the communication schedule for the schedule.
     *
     * @param cs The communication schedule to set.
     */
    void setCommunicationSchedule(const std::map<KeyTriple, unsigned int> &cs);

    /**
     * @brief Adds an entry to the communication schedule.
     *
     * @param key The key for the communication schedule entry.
     * @param step The superstep for the communication schedule entry.
     */
    void addCommunicationScheduleEntry(KeyTriple key, unsigned step);

    /**
     * @brief Adds an entry to the communication schedule.
     *
     * @param node The node resp. its data which is sent.
     * @param from_proc The processor from which the data is sent.
     * @param to_proc The processor to which the data is sent.
     * @param step The superstep in which the data is sent.
     */
    void addCommunicationScheduleEntry(unsigned node, unsigned from_proc, unsigned to_proc, unsigned step);

    /**
     * @brief Returns the communication schedule for the schedule.
     *
     * @return The communication schedule for the schedule.
     */
    const std::map<KeyTriple, unsigned int> &getCommunicationSchedule() const { return commSchedule; }

    std::map<KeyTriple, unsigned int> &getCommunicationSchedule() { return commSchedule; }

    cost_type computeWorkCosts() const;

    cost_type computeCosts() const;

    /**
     * @brief Returns true if the schedule is valid, i.e. if every time we compute a node, all its parents are already available
     * on the given processor, and every time we send a value, it is also already available on the given source processor.
     *
     * @return True if the schedule is valid, false otherwise.
     */
    bool satisfiesConstraints() const;

    vertex_idx getTotalAssignments() const;

    void mergeSupersteps();   

};

template<typename Graph_t>
BspScheduleRecomp<Graph_t>::BspScheduleRecomp(const BspScheduleCS<Graph_t> &schedule) : instance(&schedule.getInstance())
{
    node_to_processor_and_supertep_assignment.clear();
    node_to_processor_and_supertep_assignment.resize(instance->numberOfVertices());
    number_of_supersteps = schedule.numberOfSupersteps();

    for(vertex_idx node = 0; node < instance->numberOfVertices(); ++node)
        node_to_processor_and_supertep_assignment[node].emplace_back(schedule.assignedProcessor(node), schedule.assignedSuperstep(node));

    commSchedule = schedule.getCommunicationSchedule();
}

template<typename Graph_t>
void BspScheduleRecomp<Graph_t>::addCommunicationScheduleEntry(unsigned node, unsigned from_proc, unsigned to_proc,
                                                      unsigned step) {
    addCommunicationScheduleEntry(std::make_tuple(node, from_proc, to_proc), step);
}

template<typename Graph_t>
void BspScheduleRecomp<Graph_t>::addCommunicationScheduleEntry(KeyTriple key, unsigned step) {

    if (step >= number_of_supersteps)
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: step out of range.");

    if (std::get<0>(key) >= instance->numberOfVertices())
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: node out of range.");

    if (std::get<1>(key) >= instance->numberOfProcessors())
        throw std::invalid_argument(
            "Invalid Argument while adding communication schedule entry: from processor out of range.");

    if (std::get<2>(key) >= instance->numberOfProcessors())
        throw std::invalid_argument(
            "Invalid Argument while adding communication schedule entry: to processor out of range.");

    commSchedule[key] = step;
}

template<typename Graph_t>
bool BspScheduleRecomp<Graph_t>::satisfiesConstraints() const {

    // find first availability
    
    std::vector<std::vector<unsigned> > node_first_available_on_proc(instance->numberOfVertices(),
                                                                    std::vector<unsigned>(instance->numberOfProcessors(), std::numeric_limits<unsigned>::max()));

    for(vertex_idx node = 0; node < instance->numberOfVertices(); ++node)
        for(const std::pair<unsigned, unsigned>& compute_step : node_to_processor_and_supertep_assignment[node])
            node_first_available_on_proc[node][compute_step.first] =
                std::min(node_first_available_on_proc[node][compute_step.first], compute_step.second);

    for (auto const &[key, val] : commSchedule) {

        const vertex_idx& node = std::get<0>(key);
        const unsigned& to_proc = std::get<2>(key);

        node_first_available_on_proc[node][to_proc] = std::min(node_first_available_on_proc[node][to_proc], val + 1);
    }

    // check validity

    for(vertex_idx node = 0; node < instance->numberOfVertices(); ++node)
        for(vertex_idx pred : instance->getComputationalDag().parents(node))
            for(const std::pair<unsigned, unsigned>& compute_step : node_to_processor_and_supertep_assignment[node])
                if(node_first_available_on_proc[pred][compute_step.first] > compute_step.second){
                    // std::cout << "Not a valid schedule: parent " << pred << " of node "<< node <<
                    //" not yet available on processor " << compute_step.first << " in superstep "<< compute_step.second <<"." << std::endl;
                    return false;
                }

    for (auto const &[key, val] : commSchedule) {

        const vertex_idx& node = std::get<0>(key);
        const unsigned& from_proc = std::get<1>(key);

        if (node_first_available_on_proc[node][from_proc] > val) {
            // std::cout << "Not a valid schedule: node " << node << " not yet available for sending from processor "
            // << from_proc << " in superstep "<< val <<"." << std::endl;
            return false;
        }
    }
    
    return true;
};

template<typename Graph_t>
v_workw_t<Graph_t> BspScheduleRecomp<Graph_t>::computeWorkCosts() const {

    assert(satisfiesConstraints());

    std::vector<std::vector<cost_type>> step_proc_work(number_of_supersteps,
                                                      std::vector<cost_type>(instance->numberOfProcessors(), 0));

    for (vertex_idx node = 0; node < instance->numberOfVertices(); node++) {

        for (const std::pair<unsigned, unsigned>& processor_superstep : node_to_processor_and_supertep_assignment[node]) {
            step_proc_work[processor_superstep.second][processor_superstep.first] +=
                instance->getComputationalDag().vertex_work_weight(node);
        }
    }

    cost_type total_costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        cost_type max_work = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (max_work < step_proc_work[step][proc]) {
                max_work = step_proc_work[step][proc];
            }
        }

        total_costs += max_work;
    }

    return total_costs;
}

template<typename Graph_t>
v_workw_t<Graph_t> BspScheduleRecomp<Graph_t>::computeCosts() const {
    
    assert(satisfiesConstraints());

    std::vector<std::vector<cost_type>> rec(number_of_supersteps,
                                            std::vector<cost_type>(instance->numberOfProcessors(), 0));
    std::vector<std::vector<cost_type>> send(number_of_supersteps,
                                            std::vector<cost_type>(instance->numberOfProcessors(), 0));

    for (auto const &[key, val] : commSchedule) {

        send[val][std::get<1>(key)] += instance->sendCosts(std::get<1>(key), std::get<2>(key)) *
                                    instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
        rec[val][std::get<2>(key)] += instance->sendCosts(std::get<1>(key), std::get<2>(key)) *
                                    instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
    }

    cost_type total_costs = 0;
    for (unsigned step = 0; step < number_of_supersteps; step++) {

        cost_type max_comm = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            if (max_comm < send[step][proc])
                max_comm = send[step][proc];
            if (max_comm < rec[step][proc])
                max_comm = rec[step][proc];

        }

        if (max_comm > 0) {
            total_costs += instance->synchronisationCosts() + max_comm * instance->communicationCosts();
        }
    }

    total_costs += computeWorkCosts();

    return total_costs;
        
}

template<typename Graph_t>
vertex_idx_t<Graph_t> BspScheduleRecomp<Graph_t>::getTotalAssignments() const {
    vertex_idx total = 0;
    for (vertex_idx node = 0; node < instance->numberOfVertices(); ++node) {
        total += node_to_processor_and_supertep_assignment[node].size();
    }
    return total;
}

template<typename Graph_t>
void BspScheduleRecomp<Graph_t>::mergeSupersteps()
{
    std::vector<unsigned> new_step_idx(number_of_supersteps);
    std::vector<bool> comm_phase_empty(number_of_supersteps, true);

    for (auto const &[key, val] : commSchedule)
        comm_phase_empty[val] = false;

    unsigned current_step_idx = 0;
    for(unsigned step = 0; step < number_of_supersteps; ++step)
    {
        new_step_idx[step] = current_step_idx;
        if(!comm_phase_empty[step] || step == number_of_supersteps - 1)
            ++current_step_idx;
    }
    for(vertex_idx node = 0; node < instance->numberOfVertices(); ++node)
    {
        std::vector<std::pair<unsigned, unsigned>> new_assignment;
        for(const std::pair<unsigned, unsigned>& entry : node_to_processor_and_supertep_assignment[node])
            new_assignment.emplace_back(entry.first, new_step_idx[entry.second]);
        node_to_processor_and_supertep_assignment[node] = new_assignment; 
    }
    for (auto &[key, step] : commSchedule)
        step = new_step_idx[step];

    number_of_supersteps = current_step_idx;
}

} // namespace osp