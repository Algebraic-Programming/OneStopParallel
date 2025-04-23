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

#include "scheduler/IlpSchedulers/COPTEnv.hpp"
#include "callbackbase.h"
#include "coptcpp_pch.h"
#include "scheduler/Scheduler.hpp"
#include "model/BspMemSchedule.hpp"
#include "PebblingStrategy.hpp"


class MultiProcessorPebbling : public Scheduler {

  private:
    Model model;

    bool write_solutions_found;

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

  protected:
    std::vector<std::vector<VarArray>> compute;
    std::vector<std::vector<VarArray>> send_up;
    std::vector<std::vector<VarArray>> send_down;
    std::vector<std::vector<VarArray>> has_red;
    std::vector<VarArray> has_blue;

    std::vector<std::vector<std::vector<bool>>> compute_exists;
    std::vector<std::vector<std::vector<bool>>> send_up_exists;
    std::vector<std::vector<std::vector<bool>>> send_down_exists;
    std::vector<std::vector<bool>> has_blue_exists;

    VarArray comp_phase;
    VarArray comm_phase;
    VarArray send_up_phase;
    VarArray send_down_phase;

    VarArray comm_phase_ends;
    VarArray comp_phase_ends;

    unsigned max_time = 0;

    // problem settings
    bool slidingPebbles = false;
    bool mergeSteps = true;
    bool synchronous = true;
    bool up_and_down_cost_summed = true;
    bool allows_recomputation = true;
    bool restrict_step_types = false;
    unsigned compute_steps_per_cycle = 3;
    bool need_to_load_inputs = true;
    std::set<unsigned> needs_blue_at_end;
    std::vector<std::set<unsigned> > has_red_in_beginning;
    bool verbose = false;

    PebblingStrategy constructStrategyFromSolution(const BspInstance &instance);

    BspMemSchedule constructMemScheduleFromSolution(const BspInstance &instance);

    void setInitialSolution(const BspInstance &instance,
                            const std::vector<std::vector<std::vector<unsigned> > >& computeSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& sendUpSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& sendDownSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& nodesEvictedAfterStep);

    unsigned computeMaxTimeForInitialSolution(const BspInstance &instance,
                            const std::vector<std::vector<std::vector<unsigned> > >& computeSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& sendUpSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& sendDownSteps,
                            const std::vector<std::vector<std::vector<unsigned> > >& nodesEvictedAfterStep) const;

    void setupBaseVariablesConstraints(const BspInstance &instance);

    void setupSyncPhaseVariablesConstraints(const BspInstance &instance);
    void setupSyncObjective(const BspInstance &instance);

    void setupAsyncVariablesConstraintsObjective(const BspInstance &instance);
    void setupBspVariablesConstraintsObjective(const BspInstance &instance);

    void solveILP(const BspInstance &instance);

  public:
    MultiProcessorPebbling()
        : Scheduler(), model(COPTEnv::getInstance().CreateModel("MPP")), write_solutions_found(false), max_time(0) {}

    virtual ~MultiProcessorPebbling() = default;

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;
    virtual std::pair<RETURN_STATUS, BspMemSchedule> computeSynchPebbling(const BspInstance &instance);

    virtual std::pair<RETURN_STATUS, BspMemSchedule> computePebbling(const BspInstance &instance, bool use_async = false);

    virtual  std::pair<RETURN_STATUS, BspMemSchedule> computePebblingWithInitialSolution(const BspInstance &instance, const BspMemSchedule& initial_solution, bool use_async = false);

    BspSchedule convertToBspSchedule(const PebblingStrategy &instance);

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
    virtual std::string getScheduleName() const override { return "MultiProcessorPebbling"; }

    // getters and setters for problem parameters
    inline bool allowsSlidingPebbles() const { return slidingPebbles; }
    inline bool allowsMergingSteps() const { return mergeSteps; }
    inline bool isUpAndDownCostSummed() const { return up_and_down_cost_summed; }
    inline bool allowsRecomputation() const { return allows_recomputation; }
    inline bool hasRestrictedStepTypes() const { return restrict_step_types; }
    inline bool needsToLoadInputs() const { return need_to_load_inputs; }
    inline unsigned getComputeStepsPerCycle() const { return compute_steps_per_cycle; }
    inline unsigned getMaxTime() const { return max_time; }

    inline void setSlidingPebbles (const bool slidingPebbles_) {slidingPebbles = slidingPebbles_; }
    inline void setMergingSteps (const bool mergeSteps_) {mergeSteps = mergeSteps_; }
    inline void setUpAndDownCostSummed (const bool is_summed_) {up_and_down_cost_summed = is_summed_; }
    inline void setRecomputation(const bool allow_recompute_) { allows_recomputation = allow_recompute_; }
    inline void setRestrictStepTypes(const bool restrict_) { restrict_step_types = restrict_; if(restrict_){mergeSteps = true;} }
    inline void setNeedToLoadInputs(const bool load_inputs_) { need_to_load_inputs = load_inputs_;}
    inline void setComputeStepsPerCycle (const unsigned steps_per_cycle_) {compute_steps_per_cycle = steps_per_cycle_; }
    inline void setMaxTime (const unsigned max_time_) {max_time = max_time_; }
    inline void setNeedsBlueAtEnd (const std::set<unsigned>& needs_blue_) {needs_blue_at_end = needs_blue_; }
    inline void setHasRedInBeginning (const std::vector<std::set<unsigned> >& has_red_) {has_red_in_beginning = has_red_; }
    inline void setVerbose (const bool verbose_) {verbose = verbose_; }

    bool hasEmptyStep(const BspInstance &instance);
};
