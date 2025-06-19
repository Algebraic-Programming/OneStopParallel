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

#include "AbstractTestSuiteRunner.hpp"
#include "bsp/model/BspSchedule.hpp"

#include "auxiliary/run_algorithm.hpp" 
#include "io/bsp_schedule_file_writer.hpp" 

#include "util/BspScheduleStatistics/BasicBspStatsModule.hpp"
#include "util/BspScheduleStatistics/CommStatsModule.hpp"
#include "util/BspScheduleStatistics/DetailedCommStatsModule.hpp" // If used


namespace osp {

template <typename concrete_graph_t>
class BspScheduleTestSuiteRunner : public AbstractTestSuiteRunner<BspSchedule<concrete_graph_t>, concrete_graph_t> {
private:

    using base = AbstractTestSuiteRunner<BspSchedule<concrete_graph_t>, concrete_graph_t>;
    
    bool use_memory_constraint_for_bsp; 

protected:
    BspSchedule<concrete_graph_t> compute_target_object_impl(
        const BspInstance<concrete_graph_t>& instance, 
        const pt::ptree& algo_config, 
        RETURN_STATUS& status, 
        long long& computation_time_ms) override 
    {
        BspSchedule<concrete_graph_t> schedule(instance); 
        
        const auto start_time = std::chrono::high_resolution_clock::now();
        // The run_algorithm function in test_suite_bsp_schedulers.cpp was:
        // auto return_status = run_algorithm<graph_t>(parser, algorithm.second, schedule);
        // It needs to be adapted if it used more from parser than just algo_config or if it needs time_limit.
        // The run_algorithm in OneStopParallel.cpp is:
        // return_status = run_bsp_scheduler(parser, algorithm.second, schedule);
        // We need to ensure the correct run_algorithm is linked and its signature matches.
        // Assuming the one from test_suite_bsp_schedulers context:
        status = ::osp::run_algorithm<concrete_graph_t>(this->parser, algo_config, schedule, use_memory_constraint_for_bsp);
        const auto finish_time = std::chrono::high_resolution_clock::now();
        computation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();
        
        return schedule;
    }

    void create_and_register_statistic_modules(const std::string& module_name, 
                                               const pt::ptree& /*module_config*/) override {
        if (module_name == "BasicBspStats") {
            this->active_stats_modules.push_back(std::make_unique<BasicBspStatsModule>());
        } else if (module_name == "DetailedCommStats") { 
            // Assuming CommStatsModule is identified by "DetailedCommStats" in config
            this->active_stats_modules.push_back(std::make_unique<CommStatsModule>());
        } else if (module_name == "CommStats") { // If config uses this name
             this->active_stats_modules.push_back(std::make_unique<CommStatsModule>());
        }
    }
    
    void write_target_object_hook(const BspSchedule<concrete_graph_t>& schedule,
                             const std::string& graph_name,
                             const std::string& machine_name,
                             const std::string& algo_name) const override {
        std::string file_path = this->output_target_object_dir_path + graph_name + "_" + machine_name + "_" + algo_name + "_schedule.txt";
        file_writer::write_txt(file_path, schedule);
    }

    bool parse_common_config() override {
        
        if (! base::parse_common_config())
            return false;

        try {
 use_memory_constraint_for_bsp = this->parser.global_params.get_child("useMemoryConstraint").get_value_optional<bool>().value_or(false);
        } catch (const std::exception &e) {
            this->log_stream << "Error parsing BSP schedule runner specific config: " << e.what() << std::endl;
            std::cerr << "Error parsing BSP schedule runner specific config: " << e.what() << std::endl;
            return false;
        }
        

       
        
        return true;
    }


public:
    BspScheduleTestSuiteRunner(int argc, char *argv[], const std::string &main_config_path)
        : AbstractTestSuiteRunner<BspSchedule<concrete_graph_t>, concrete_graph_t>(argc, argv, main_config_path) {
            


            


        }
};

} // namespace osp
