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
#include "file_interactions/BspScheduleRecompWriter.hpp"
#include "file_interactions/BspScheduleWriter.hpp"
#include "model/BspScheduleRecomp.hpp"
#include "model/VectorSchedule.hpp"
#include "scheduler/Scheduler.hpp"

/**
 * @class CoptFullScheduler
 * @brief A class that represents a scheduler using the COPT solver for computing the schedule of a BSP instance.
 *
 * The `CoptFullScheduler` class is a subclass of the `Scheduler` class and provides an implementation of the
 * `computeSchedule` method using the COPT solver. It uses an ILP (Integer Linear Programming) formulation to find an
 * optimal schedule for a given BSP instance. The scheduler supports various options such as setting an initial
 * solution, setting the maximum number of supersteps, enabling/disabling writing intermediate solutions, and setting
 * communication constraints.
 *
 * The COPT solver is used to solve the ILP formulation and find the optimal schedule. It provides methods to set up the
 * ILP model, define variables and constraints, and solve the ILP problem. The scheduler constructs a `Model` object
 * from the COPT library to represent the ILP model and uses various callbacks to define the objective function,
 * constraints, and solution handling.
 *
 * To compute the schedule, the `computeSchedule` method is called with a `BspInstance` object representing the BSP
 * instance for which the schedule needs to be computed. The method returns a pair containing the return status and the
 * computed `BspSchedule`.
 *
 * The `CoptFullScheduler` class also provides methods to set the initial solution, set the maximum number of
 * supersteps, enable/disable writing intermediate solutions, and get information about the best gap, objective value,
 * and bound found by the solver.
 */
class CoptFullScheduler : public Scheduler {

  private:
   
    bool allow_recomputation;
    bool use_memory_constraint;
    bool use_initial_schedule;
    const BspSchedule *initial_schedule;

    bool write_solutions_found;
    std::string write_solutions_path;
    std::string solution_file_prefix;

    class WriteSolutionCallback : public CallbackBase {

      private:
        unsigned counter;
        unsigned max_number_solution;

        double best_obj;

      public:
        WriteSolutionCallback()
            : counter(0), max_number_solution(500), best_obj(COPT_INFINITY), allow_recomputation_cb(false),
              write_solutions_path_cb(""), solution_file_prefix_cb(""), instance_ptr(),
              node_to_processor_superstep_var_ptr(), comm_processor_to_processor_superstep_node_var_ptr() {}

        bool allow_recomputation_cb;
        std::string write_solutions_path_cb;
        std::string solution_file_prefix_cb;
        const BspInstance *instance_ptr;

        std::vector<std::vector<VarArray>> *node_to_processor_superstep_var_ptr;
        std::vector<std::vector<std::vector<VarArray>>> *comm_processor_to_processor_superstep_node_var_ptr;

        void callback() override;
        BspSchedule constructBspScheduleFromCallback();
        BspScheduleRecomp constructBspScheduleRecompFromCallback();
    };

    // WriteSolutionCallback solution_callback;

  protected:
    unsigned int max_number_supersteps;

    VarArray superstep_used_var;
    std::vector<std::vector<VarArray>> node_to_processor_superstep_var;
    std::vector<std::vector<std::vector<VarArray>>> comm_processor_to_processor_superstep_node_var;

    VarArray max_comm_superstep_var;
    VarArray max_work_superstep_var;

    BspSchedule constructBspScheduleFromSolution(const BspInstance &instance, bool cleanup_ = false);

    BspScheduleRecomp constructBspScheduleRecompFromSolution(const BspInstance &instance, bool cleanup_ = false);

    void loadInitialSchedule(Model &model);

    void setupVariablesConstraintsObjective(const BspInstance &instance, Model &model);

  public:
    CoptFullScheduler(unsigned steps = 5)
        : Scheduler(), allow_recomputation(false),
          use_memory_constraint(false), use_initial_schedule(false), initial_schedule(0), write_solutions_found(false),
          max_number_supersteps(steps) {

        // solution_callback.comm_processor_to_processor_superstep_node_var_ptr =
        //     &comm_processor_to_processor_superstep_node_var;
        // solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

    CoptFullScheduler(const BspSchedule &schedule)
        : Scheduler(), allow_recomputation(false),
          use_memory_constraint(false), use_initial_schedule(true), initial_schedule(&schedule),
          write_solutions_found(false), max_number_supersteps(schedule.numberOfSupersteps()) {

        // solution_callback.comm_processor_to_processor_superstep_node_var_ptr =
        //     &comm_processor_to_processor_superstep_node_var;
        // solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

    virtual ~CoptFullScheduler() = default;

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

    virtual std::pair<RETURN_STATUS, BspScheduleRecomp> computeScheduleRecomp(const BspInstance &instance);

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
        write_solutions_path = path;
        solution_file_prefix = file_prefix;
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
    //inline double bestGap() { return model.GetDblAttr(COPT_DBLATTR_BESTGAP); }

    /**
     * @brief Get the best objective value found by the solver.
     *
     * @return The best objective value found by the solver.
     */
    //inline double bestObjective() { return model.GetDblAttr(COPT_DBLATTR_BESTOBJ); }

    /**
     * @brief Get the best bound found by the solver.
     *
     * @return The best bound found by the solver.
     */
    //inline double bestBound() { return model.GetDblAttr(COPT_DBLATTR_BESTBND); }

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "FullIlp"; }
};
