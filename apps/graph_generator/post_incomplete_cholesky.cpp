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
#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/OrderingMethods>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> \n" << std::endl;
        return 1;
    }

    std::string filename_graph = argv[1];

    std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
    name_graph = name_graph.substr(0, name_graph.find_last_of("."));

    std::cout << "Graph: " << name_graph << std::endl;
    
    using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>; // Compressed Sparse Column format
    using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>; // Compressed Sparse Row format

    SM_csc L_csc; // Initialize a sparse matrix in CSC format

    Eigen::loadMarket(L_csc, filename_graph);

    SM_csr L_csr = L_csc;   // Reformat the sparse matrix from CSC to CSR format

    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<int>> ichol(L_csc);

    SM_csc LChol_csc = ichol.matrixL();

    Eigen::saveMarket(LChol_csc, filename_graph.substr(0, filename_graph.find_last_of(".")) + "_postChol.mtx", Eigen::UpLoType::Symmetric);

    return 0;
}