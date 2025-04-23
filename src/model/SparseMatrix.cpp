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

@author Christos Matzoros, Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
*/

#include "model/SparseMatrix.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<unsigned int> SparseMatrix::sourceVertices() const {
    std::vector<unsigned int> sources;
    auto rows = (*L_csr_p).rows();
    for (auto row = 0; row < rows; ++row){
        bool found = false;
        for (SM_csr::InnerIterator it((*L_csr_p), row); it; ++it){
            auto index = static_cast<unsigned int>(it.index());
            if( row != index ){
                found = true;
                break;
            }
        }
        if (!found)
            sources.push_back(static_cast<unsigned int>(row));
    }

    return sources;
}

std::vector<unsigned int> SparseMatrix::sinkVertices() const {
    std::vector<unsigned int> sink_vertices;
    auto cols = (*L_csc_p).cols();
    for (auto col = 0; col < cols; ++col){
        bool found = false;
        for (SM_csc::InnerIterator it((*L_csc_p), col); it; ++it){
            auto index = static_cast<unsigned int>(it.index());
            if( col != index ){
                found = true;
                break;
            }
        }
        if (!found)
            sink_vertices.push_back(static_cast<unsigned int>(col));
    }

    return sink_vertices;
}
