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
#include "scheduler/Scheduler.hpp"
#include "model/BspInstance.hpp"

class AcyclicPartitioningILP : public Scheduler {

  private:
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

  protected:
    std::vector<VarArray> node_in_partition;
    std::vector<VarArray> hyperedge_intersects_partition;

    unsigned minPartitionSize = 500, maxPartitionSize = 1400;

    std::vector<unsigned> returnAssignment(const BspInstance &instance);

    void setupVariablesConstraintsObjective(const BspInstance &instance);

    void solveILP(const BspInstance &instance);

  public:
    AcyclicPartitioningILP()
        : model(COPTEnv::getInstance().CreateModel("AsyncPart")), write_solutions_found(false) {}

    virtual ~AcyclicPartitioningILP() = default;

    std::pair<RETURN_STATUS, std::vector<unsigned> > computePartitioning(const BspInstance &instance);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

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
    virtual std::string getScheduleName() const override { return "AcyclicPartitioningILP"; }

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> getMinAndMaxSize() const { return std::make_pair(minPartitionSize, maxPartitionSize); }
    inline void setMinAndMaxSize(const std::pair<unsigned, unsigned> min_and_max) {minPartitionSize = min_and_max.first; maxPartitionSize = min_and_max.second; }

    inline unsigned getNumberOfParts() const { return numberOfParts; }
    inline void setNumberOfParts(const unsigned number_of_parts) {numberOfParts = number_of_parts; }
    inline void setIgnoreSourceForConstraint(const bool ignore_) {ignore_sources_for_constraint = ignore_; }
    inline void setIsOriginalSource(const std::vector<bool>& is_original_source_) {is_original_source = is_original_source_; }
};
