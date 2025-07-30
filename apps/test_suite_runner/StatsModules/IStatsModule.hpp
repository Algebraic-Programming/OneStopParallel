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

#include <boost/property_tree/ptree.hpp>
#include <fstream>
#include <string>
#include <vector>
 #include <map>
// #include "osp/bsp/model/BspSchedule.hpp" // TargetObject will be passed, no specific include here

// Forward declarations to avoid circular dependencies
namespace osp { // Ensure this is within the osp namespace

namespace pt = boost::property_tree;

template<typename TargetObjectType>
class IStatisticModule { // Changed from Graph_t_ to TargetObjectType
  public:
    virtual ~IStatisticModule() = default;

    // Returns a list of column headers this module provides.
    virtual std::vector<std::string> get_metric_headers() const = 0;

    // Called for each generated target_object.
    // Returns a map of {header_name: value_string}.
    virtual std::map<std::string, std::string> record_statistics( 
                                   const TargetObjectType &target_object, // Changed parameter
                                   std::ofstream &log_stream) const = 0;
};

} // namespace osp
