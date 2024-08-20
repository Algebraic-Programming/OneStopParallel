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

#include "Scheduler.hpp"
#include "file_interactions/FileReader.hpp"
#include <string>
#include <iostream>

class FileReadScheduler : public Scheduler {

  private:
    std::string filename_schedule;

  public:
    FileReadScheduler() = default;
    FileReadScheduler(const std::string file_name) : Scheduler(), filename_schedule(file_name) {}
    virtual ~FileReadScheduler() {}

    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override {
        std::pair<bool, BspSchedule> read = FileReader::readBspSchdeuleTxtFormat(instance, filename_schedule);

        if (read.first) {
            read.second.setAutoCommunicationSchedule();
            return std::make_pair(SUCCESS, read.second);
        } else {
            return std::make_pair(ERROR, BspSchedule());
        }
    };

    virtual std::string getScheduleName() const { return "FileReadScheduler_" + filename_schedule; }
};