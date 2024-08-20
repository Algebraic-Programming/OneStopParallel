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

#include "model/IBspSchedule.hpp"

typedef std::tuple<unsigned int, unsigned int, unsigned int> KeyTriple;

class ICommunicationScheduler {

  public:

    virtual ~ICommunicationScheduler() = default;

    virtual std::map<KeyTriple, unsigned> computeCommunicationSchedule(const IBspSchedule &sched) = 0;
};

class EagerCommunicationScheduler : ICommunicationScheduler {

  public:
    EagerCommunicationScheduler() {}
    virtual ~EagerCommunicationScheduler() = default;

    virtual std::map<KeyTriple, unsigned> computeCommunicationSchedule(const IBspSchedule &sched) override;
};

class LazyCommunicationScheduler : ICommunicationScheduler {


  public:
    LazyCommunicationScheduler() {}
    virtual ~LazyCommunicationScheduler() = default;

    virtual std::map<KeyTriple, unsigned> computeCommunicationSchedule(const IBspSchedule &sched) override;
};
