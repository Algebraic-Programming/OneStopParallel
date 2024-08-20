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

#include "scheduler/InstanceContractor.hpp"

/**
 * @brief Acyclic graph contractor from (Zarebavani, Behrooz, et al. "HDagg: hybrid aggregation of loop-carried dependence iterations in sparse matrix computations." 2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2022.)
 * 
 */
class HDaggCoarser : public InstanceContractor {
    private:

    protected:
        RETURN_STATUS run_contractions() override;

    public:
        HDaggCoarser() : InstanceContractor() { }
        HDaggCoarser(Scheduler* sched_) : HDaggCoarser(sched_, nullptr) { }
        HDaggCoarser(Scheduler* sched_, ImprovementScheduler* improver_) : InstanceContractor(sched_, improver_) { }
        HDaggCoarser(unsigned timelimit, Scheduler* sched_) : HDaggCoarser(timelimit, sched_, nullptr) { }
        HDaggCoarser(unsigned timelimit, Scheduler* sched_, ImprovementScheduler* improver_) : InstanceContractor(timelimit, sched_, improver_) { }
        virtual ~HDaggCoarser() = default;

        std::string getCoarserName() const override { return "HDaggCoarser"; }
};