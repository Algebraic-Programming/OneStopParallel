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

#include "algorithms/Coarsers/CacheLineGluer.hpp"

RETURN_STATUS CacheLineGluer::run_contractions() {
    std::vector<std::unordered_set<VertexType>> partition;
    unsigned counter = cacheline_shift;
    for ( VertexType node : dag_history.back()->getComputationalDag().vertices() ) {
        if (counter == 0) {
            partition.push_back({node});
        } else {
            partition.back().emplace(node);
        }

        counter++;
        counter %= cacheline_size;
    }
    
    add_contraction(partition);
    return SUCCESS;
}
