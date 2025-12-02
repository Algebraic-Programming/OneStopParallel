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

#include "osp/pebbling/pebblers/pebblingILP/COPTEnv.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/bsp/model/BspInstance.hpp"

namespace osp{

template<typename Graph_t>
class AcyclicPartitioningILP {

    static_assert(is_computational_dag_v<Graph_t>, "PebblingSchedule can only be used with computational DAGs."); 

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using commweight_type = v_commw_t<Graph_t>;

    Model model;

    bool write_solutions_found;
    bool ignore_sources_for_constraint = true;

    class WriteSolutionCallback : public CallbackBase {

      private:
        unsigned counter;
        unsigned max_number_solution;

        double best_obj;

      public:
        WriteSolutionCallback()
            : counter(0), max_number_solution(500), best_obj(COPT_INFINITY), write_solutions_path_cb(""),
              solution_file_prefix_cb("") {}

        std::string write_solutions_path_cb;
        std::string solution_file_prefix_cb;

        void callback() override;

    };

    WriteSolutionCallback solution_callback;

    unsigned numberOfParts = 0;

    std::vector<bool> is_original_source;

    unsigned time_limit_seconds;

  protected:
    std::vector<VarArray> node_in_partition;
    std::vector<VarArray> hyperedge_intersects_partition;

    unsigned minPartitionSize = 500, maxPartitionSize = 1400;

    std::vector<unsigned> returnAssignment(const BspInstance<Graph_t> &instance);

    void setupVariablesConstraintsObjective(const BspInstance<Graph_t> &instance);

    void solveILP();

  public:
    AcyclicPartitioningILP()
        : model(COPTEnv::getInstance().CreateModel("AsyncPart")), write_solutions_found(false) {}

    virtual ~AcyclicPartitioningILP() = default;

    RETURN_STATUS computePartitioning(const BspInstance<Graph_t> &instance, std::vector<unsigned>& partitioning);

    /**
     * @brief Enables writing intermediate solutions.
     *
     * This function enables the writing of intermediate solutions. The
     * `path` parameter specifies the path where the solutions will be
     * written, and the `file_prefix` parameter specifies the prefix
     * that will be used for the solution files.
     *
     * @param path The path where the solutions will be written.
     * @param file_prefix The prefix that will be used for the solution files.
     */
    inline void enableWriteIntermediateSol(std::string path, std::string file_prefix) {
        write_solutions_found = true;
        solution_callback.write_solutions_path_cb = path;
        solution_callback.solution_file_prefix_cb = file_prefix;
    }

    /**
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void disableWriteIntermediateSol() { write_solutions_found = false; }

    /**
     * @brief Get the best gap found by the solver.
     *
     * @return The best gap found by the solver.
     */
    inline double bestGap() { return model.GetDblAttr(COPT_DBLATTR_BESTGAP); }

    /**
     * @brief Get the best objective value found by the solver.
     *
     * @return The best objective value found by the solver.
     */
    inline double bestObjective() { return model.GetDblAttr(COPT_DBLATTR_BESTOBJ); }

    /**
     * @brief Get the best bound found by the solver.
     *
     * @return The best bound found by the solver.
     */
    inline double bestBound() { return model.GetDblAttr(COPT_DBLATTR_BESTBND); }
  
    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const { return "AcyclicPartitioningILP"; }

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> getMinAndMaxSize() const { return std::make_pair(minPartitionSize, maxPartitionSize); }
    inline void setMinAndMaxSize(const std::pair<unsigned, unsigned> min_and_max) {minPartitionSize = min_and_max.first; maxPartitionSize = min_and_max.second; }

    inline unsigned getNumberOfParts() const { return numberOfParts; }
    inline void setNumberOfParts(const unsigned number_of_parts) {numberOfParts = number_of_parts; }
    inline void setIgnoreSourceForConstraint(const bool ignore_) {ignore_sources_for_constraint = ignore_; }
    inline void setIsOriginalSource(const std::vector<bool>& is_original_source_) {is_original_source = is_original_source_; }
    void setTimeLimitSeconds(unsigned time_limit_seconds_) { time_limit_seconds = time_limit_seconds_; }
};

template<typename Graph_t>
void AcyclicPartitioningILP<Graph_t>::solveILP() {

    model.SetIntParam(COPT_INTPARAM_LOGTOCONSOLE, 0);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, time_limit_seconds);
    model.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model.SetIntParam(COPT_INTPARAM_STRONGBRANCHING, 1);
    model.SetIntParam(COPT_INTPARAM_LPMETHOD, 1);
    model.SetIntParam(COPT_INTPARAM_ROUNDINGHEURLEVEL, 1);

    model.SetIntParam(COPT_INTPARAM_SUBMIPHEURLEVEL, 1);
    // model.SetIntParam(COPT_INTPARAM_PRESOLVE, 1);
    // model.SetIntParam(COPT_INTPARAM_CUTLEVEL, 0);
    model.SetIntParam(COPT_INTPARAM_TREECUTLEVEL, 2);
    // model.SetIntParam(COPT_INTPARAM_DIVINGHEURLEVEL, 2);

    model.Solve();
}

template<typename Graph_t>
RETURN_STATUS AcyclicPartitioningILP<Graph_t>::computePartitioning(const BspInstance<Graph_t> &instance, std::vector<unsigned>& partitioning)
{
    partitioning.clear();

    if(numberOfParts == 0)
    {
        numberOfParts = static_cast<unsigned>(std::floor(static_cast<double>(instance.numberOfVertices())  / static_cast<double>(minPartitionSize)));
        std::cout<<"ILP nr parts: "<<numberOfParts<<std::endl;
    }

    setupVariablesConstraintsObjective(instance);

    solveILP();

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {

        partitioning = returnAssignment(instance);
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {

        partitioning.resize(instance.numberOfVertices(), UINT_MAX);
        return RETURN_STATUS::ERROR;

    } else {

        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {

            partitioning = returnAssignment(instance);
            return RETURN_STATUS::OSP_SUCCESS;

        } else {
            partitioning.resize(instance.numberOfVertices(), UINT_MAX);
            return RETURN_STATUS::ERROR;
        }
    }
}

template<typename Graph_t>
void AcyclicPartitioningILP<Graph_t>::setupVariablesConstraintsObjective(const BspInstance<Graph_t> &instance) {

    // Variables

    node_in_partition = std::vector<VarArray>(instance.numberOfVertices());

    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++)
        node_in_partition[node] = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "node_in_partition");

    
    std::map<vertex_idx, unsigned> node_to_hyperedge_index;
    unsigned numberOfHyperedges = 0;
    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++)
        if(instance.getComputationalDag().out_degree(node) > 0)
        {
            node_to_hyperedge_index[node] = numberOfHyperedges;
            ++numberOfHyperedges;
        }

    hyperedge_intersects_partition = std::vector<VarArray>(numberOfHyperedges);

    for (unsigned hyperedge = 0; hyperedge < numberOfHyperedges; hyperedge++)
        hyperedge_intersects_partition[hyperedge] = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "hyperedge_intersects_partition");

    // Constraints

    // each node assigned to exactly one partition
    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++) {

        Expr expr;
        for (unsigned part = 0; part < numberOfParts; part++) {

            expr += node_in_partition[node][static_cast<int>(part)];
        }
        model.AddConstr(expr == 1);
    }

    // hyperedge indicators match node variables
    for (unsigned part = 0; part < numberOfParts; part++)
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++)
        {
            if(instance.getComputationalDag().out_degree(node) == 0)
                continue;

            model.AddConstr(hyperedge_intersects_partition[node_to_hyperedge_index[node]][static_cast<int>(part)] >= node_in_partition[node][static_cast<int>(part)]);
            for (const auto &succ : instance.getComputationalDag().children(node))
                model.AddConstr(hyperedge_intersects_partition[node_to_hyperedge_index[node]][static_cast<int>(part)] >= node_in_partition[succ][static_cast<int>(part)]);
        }
    
    // partition size constraints
    for (unsigned part = 0; part < numberOfParts; part++)
    {
        Expr expr;
        for (vertex_idx node = 0; node < instance.numberOfVertices(); node++)
            if(!ignore_sources_for_constraint || is_original_source.empty() || !is_original_source[node])
                expr += node_in_partition[node][static_cast<int>(part)];

        model.AddConstr(expr <= maxPartitionSize);
        model.AddConstr(expr >= minPartitionSize);
    }

    // acyclicity constraints
    for (unsigned from_part = 0; from_part < numberOfParts; from_part++)
        for (unsigned to_part = 0; to_part < from_part; to_part++)
            for (vertex_idx node = 0; node < instance.numberOfVertices(); node++)
                for (const auto &succ : instance.getComputationalDag().children(node))
                    model.AddConstr(node_in_partition[node][static_cast<int>(from_part)] + node_in_partition[succ][static_cast<int>(to_part)] <= 1);
    

    // set objective
    Expr expr;
    for (vertex_idx node = 0; node < instance.numberOfVertices(); node++)
        if(instance.getComputationalDag().out_degree(node) > 0)
        {
            expr -= instance.getComputationalDag().vertex_comm_weight(node);
            for (unsigned part = 0; part < numberOfParts; part++)
                expr += instance.getComputationalDag().vertex_comm_weight(node) * hyperedge_intersects_partition[node_to_hyperedge_index[node]][static_cast<int>(part)];
        }

    model.SetObjective(expr, COPT_MINIMIZE);
             
};

template<typename Graph_t>
void AcyclicPartitioningILP<Graph_t>::WriteSolutionCallback::callback() {

    if (Where() == COPT_CBCONTEXT_MIPSOL && counter < max_number_solution && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {

        try {

            if (GetDblInfo(COPT_CBINFO_BESTOBJ) < best_obj && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {

                best_obj = GetDblInfo(COPT_CBINFO_BESTOBJ);
                counter++;
            }

        } catch (const std::exception &e) {
        }
    }
};

template<typename Graph_t>
std::vector<unsigned> AcyclicPartitioningILP<Graph_t>::returnAssignment(const BspInstance<Graph_t> &instance)
{
    std::vector<unsigned> node_to_partition(instance.numberOfVertices(), UINT_MAX);

    std::set<unsigned> nonempty_partition_ids;
    for (unsigned node = 0; node < instance.numberOfVertices(); node++)
        for(unsigned part = 0; part < numberOfParts; part++)
            if(node_in_partition[node][static_cast<int>(part)].Get(COPT_DBLINFO_VALUE) >= .99)
            {
                node_to_partition[node] = part;
                nonempty_partition_ids.insert(part);
            }

    for(unsigned chosen_partition : node_to_partition)
        if(chosen_partition == UINT_MAX)
            std::cout<<"Error: partitioning returned by ILP seems incomplete!"<<std::endl;
    
    unsigned current_index = 0;
    std::map<unsigned, unsigned> new_index;
    for(unsigned part_index : nonempty_partition_ids)
    {
        new_index[part_index] = current_index;
        ++current_index;
    }

    for(vertex_idx node = 0; node < instance.numberOfVertices(); node++)
        node_to_partition[node] = new_index[node_to_partition[node]];

    std::cout<<"Acyclic partitioning ILP best solution value: "<<model.GetDblAttr(COPT_DBLATTR_BESTOBJ)<<", best lower bound: "<<model.GetDblAttr(COPT_DBLATTR_BESTBND)<<std::endl;

    return node_to_partition;
}

}