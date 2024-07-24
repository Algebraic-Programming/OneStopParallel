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

#include <algorithm>
#include <cassert>
#include <limits.h>
#include <stdexcept>
#include <vector>

#include "algorithms/Scheduler.hpp"
#include "model/BspArchitecture.hpp"

class SubArchitectureScheduler : public Scheduler {
    protected:
        Scheduler* scheduler;
        unsigned num_processors;
        bool logarithmic;
        unsigned best_num_processors;

    public:
        SubArchitectureScheduler( Scheduler* scheduler_, unsigned num_processors_ = 0, bool logarithmic_ = true ) : Scheduler(scheduler_->getTimeLimitSeconds()+10), scheduler(scheduler_), num_processors(num_processors_), logarithmic(logarithmic_), best_num_processors(0) { };

        void setNumberofProcessors(unsigned proc) { num_processors = proc; }
        virtual void setTimeLimitSeconds(unsigned int limit) override;
        virtual void setTimeLimitHours(unsigned int limit) override;

        static void min_symmetric_sub_sum( const std::vector<std::vector<unsigned>>& matrix,
                                    const size_t size,
                                    std::vector<unsigned>& current_processors,
                                    std::vector<unsigned>& best_ans,
                                    long unsigned& current_best);
        static std::vector<unsigned> min_symmetric_sub_sum(const std::vector<std::vector<unsigned>>& matrix, const size_t size);

        std::pair<RETURN_STATUS, BspSchedule> computeSchedule_fixed_size(const BspInstance &instance, size_t size);
        std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

        std::string getScheduleName() const override {
            std::string out = scheduler->getScheduleName() + "-Subarch" + std::to_string(num_processors);
            if (num_processors == 0) out += ("best" + std::to_string(best_num_processors));
            return out;
        }
};