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

#include "COPTEnv.hpp"
#include "bsp/scheduler/LocalSearchSchedulers/KernighanLin/kl_total_comm.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "file_interactions/BspScheduleWriter.hpp"

class TotalCommunicationScheduler : public Scheduler {

  private:
    Model model;

    bool use_memory_constraint;
    bool ignore_workload_balance;

    bool use_initial_schedule;
    const BspSchedule *initial_schedule;

    bool write_solutions_found;
    bool use_lk_heuristic_callback;

    class WriteSolutionCallback : public CallbackBase {

      private:
        unsigned counter;
        unsigned max_number_solution;

        double best_obj;

      public:
        WriteSolutionCallback()
            : counter(0), max_number_solution(100), best_obj(COPT_INFINITY), write_solutions_path_cb(""),
              solution_file_prefix_cb(""), instance_ptr(0), node_to_processor_superstep_var_ptr() {}

        std::string write_solutions_path_cb;
        std::string solution_file_prefix_cb;
        const BspInstance *instance_ptr;

        std::vector<std::vector<VarArray>> *node_to_processor_superstep_var_ptr;

        void callback() override;
        BspSchedule constructBspScheduleFromCallback();
    };

    class LKHeuristicCallback : public CallbackBase {

      private:
        kl_total_comm lk_heuristic;

        double best_obj;

      public:

        LKHeuristicCallback()
            : lk_heuristic(), best_obj(COPT_INFINITY), num_step(0), instance_ptr(0), max_work_superstep_var_ptr(0),
              superstep_used_var_ptr(0), node_to_processor_superstep_var_ptr(0), edge_vars_ptr(0) {}
              
        unsigned num_step;
        const BspInstance *instance_ptr;

        VarArray *max_work_superstep_var_ptr;
        VarArray *superstep_used_var_ptr;
        std::vector<std::vector<VarArray>> *node_to_processor_superstep_var_ptr;
        std::vector<std::vector<VarArray>> *edge_vars_ptr;

        void callback() override;
        BspSchedule constructBspScheduleFromCallback();
        void feedImprovedSchedule(const BspSchedule &s);
    };

    WriteSolutionCallback solution_callback;
    LKHeuristicCallback heuristic_callback;

  protected:
    unsigned int max_number_supersteps;

    VarArray superstep_used_var;
    std::vector<std::vector<VarArray>> node_to_processor_superstep_var;
    std::vector<std::vector<VarArray>> edge_vars;
    VarArray max_work_superstep_var;

    BspSchedule constructBspScheduleFromSolution(const BspInstance &instance, bool cleanup_ = false);

    void loadInitialSchedule();

    void setupVariablesConstraintsObjective(const BspInstance &instance);

  public:
    TotalCommunicationScheduler(unsigned steps = 5)
        : Scheduler(), model(COPTEnv::getInstance().CreateModel("BspSchedule")), use_memory_constraint(false), ignore_workload_balance(false), use_initial_schedule(false),
          initial_schedule(0), write_solutions_found(false), use_lk_heuristic_callback(true), solution_callback(), heuristic_callback(),
          max_number_supersteps(steps) {

        heuristic_callback.max_work_superstep_var_ptr = &max_work_superstep_var;
        heuristic_callback.superstep_used_var_ptr = &superstep_used_var;
        heuristic_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
        heuristic_callback.edge_vars_ptr = &edge_vars;

        solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }


    TotalCommunicationScheduler(const BspSchedule &schedule)
        : Scheduler(), model(COPTEnv::getInstance().CreateModel("BspSchedule")), use_memory_constraint(false), ignore_workload_balance(false), use_initial_schedule(true),
          initial_schedule(&schedule), write_solutions_found(false), use_lk_heuristic_callback(true), solution_callback(), heuristic_callback(),
          max_number_supersteps(schedule.numberOfSupersteps()) {

        heuristic_callback.max_work_superstep_var_ptr = &max_work_superstep_var;
        heuristic_callback.superstep_used_var_ptr = &superstep_used_var;
        heuristic_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
        heuristic_callback.edge_vars_ptr = &edge_vars;
        
        solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

    virtual ~TotalCommunicationScheduler() = default;

    /**
     * @brief Compute the schedule for the given BspInstance using the COPT solver.
     *
     * @param instance the BspInstance for which to compute the schedule
     *
     * @return a pair containing the return status and the computed BspSchedule
     *
     * @throws std::invalid_argument if the instance parameters do not
     *         agree with those of the initial schedule's instance
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    /**
     * @brief Sets the provided schedule as the initial solution for the ILP.
     *
     * This function sets the provided schedule as the initial solution for the ILP.
     * The maximum number of allowed supersteps is set to the number of supersteps
     * in the provided schedule.
     *
     * @param schedule The provided schedule.
     */
    inline void setInitialSolutionFromBspSchedule(const BspSchedule &schedule) {

        initial_schedule = &schedule;

        max_number_supersteps = schedule.numberOfSupersteps();

        use_initial_schedule = true;
    }


    /**
     * @brief Sets the maximum number of supersteps allowed.
     *
     * This function sets the maximum number of supersteps allowed
     * for the computation of the BSP schedule. If an initial
     * solution is used, the maximum number of supersteps must be
     * greater or equal to the number of supersteps in the initial
     * solution.
     *
     * @param max The maximum number of supersteps allowed.
     *
     * @throws std::invalid_argument If the maximum number of
     *         supersteps is less than the number of supersteps in
     *         the initial solution.
     */
    void setMaxNumberOfSupersteps(unsigned max) {
        if (use_initial_schedule && max < initial_schedule->numberOfSupersteps()) {

            throw std::invalid_argument("Invalid Argument while setting "
                                        "max number of supersteps to a value "
                                        "which is less than the number of "
                                        "supersteps of the initial schedule!");
        }
        max_number_supersteps = max;
    }

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
     * @brief Set the use of memory constraint.
     *
     * This function sets the use of memory constraint. If the memory
     * constraint is enabled, the solver will use a memory constraint
     * to limit total memory of nodes assigend to a processor in a superstep.
     *
     * @param use True if the memory constraint should be used, false otherwise.
     */
    inline void setUseMemoryConstraint(bool use) { use_memory_constraint = use; }

    /**
     * @brief Set the use of workload balance constraint.
     *
     * This function sets the use of workload balance constraint. If the
     * workload balance constraint is enabled, the solver will use a workload
     * balance constraint to limit the difference of total work of nodes
     * assigned to a processor in a superstep.
     *
     * @param use True if the workload balance constraint should be used, false otherwise.
     */
    inline void setIgnoreWorkloadBalance(bool use) { ignore_workload_balance = use; }


    /**
     * @brief Set the use of LK heuristic callback.
     *
     * This function sets the use of LK heuristic callback. If the LK heuristic
     * callback is enabled, the solver will use the LK heuristic on any feasible solution
     * that is found to improve it.
     * 
     *
     * @param use True if the LK heuristic callback should be used, false otherwise.
     */
    inline void setUseLkHeuristicCallback(bool use) { use_lk_heuristic_callback = use; }

    /**
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void disableWriteIntermediateSol() { write_solutions_found = false; }

    /**
     * @brief Get the maximum number of supersteps.
     *
     * @return The maximum number of supersteps.
     */
    inline unsigned getMaxNumberOfSupersteps() const { return max_number_supersteps; }

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
    virtual std::string getScheduleName() const override { return "TotalCommIlp"; }
};
