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
#include "WavefrontDivider.hpp"
#include "WavefrontComponentDivider.hpp"
#include "IsomorphismGroups.hpp"
#include "scheduler/Scheduler.hpp"

class WavefrontComponentScheduler : public Scheduler {

    bool set_num_proc_crit_path = false;

    IWavefrontDivider *divider;

    Scheduler *scheduler;

    bool check_isomorphism_groups = true;

    BspArchitecture setup_sub_architecture(const BspArchitecture &original, const ComputationalDag &sub_dag,
                                    const double subgraph_work_weight, const double total_step_work);


    std::pair<RETURN_STATUS, BspSchedule> computeSchedule_with_isomorphism_groups(const BspInstance &instance);
    std::pair<RETURN_STATUS, BspSchedule> computeSchedule_without_isomorphism_groups(const BspInstance &instance);

  public:
  
    WavefrontComponentScheduler(IWavefrontDivider &div, Scheduler &scheduler) : divider(&div), scheduler(&scheduler) {}

    void set_check_isomorphism_groups(bool check) { check_isomorphism_groups = check; }

    std::string getScheduleName() const override { return "WavefrontComponentScheduler"; }

    std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;
};
