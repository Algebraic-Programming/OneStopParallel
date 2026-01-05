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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>
#include <filesystem>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> \n" << std::endl;
        return 1;
    }

    std::string filenameGraph = argv[1];

    std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1);
    nameGraph = nameGraph.substr(0, nameGraph.find_last_of("."));

    std::cout << "Graph: " << nameGraph << std::endl;

    using SmCsc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;    // Compressed Sparse Column format
    using SmCsr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;    // Compressed Sparse Row format

    SmCsc lCsc;    // Initialize a sparse matrix in CSC format

    Eigen::loadMarket(lCsc, filenameGraph);

    SmCsr lCsr = lCsc;    // Reformat the sparse matrix from CSC to CSR format

    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<int>> ichol(lCsc);

    SmCsc lCholCsc = ichol.matrixL();

    Eigen::saveMarket(
        lCholCsc, filenameGraph.substr(0, filenameGraph.find_last_of(".")) + "_postChol.mtx", Eigen::UpLoType::Symmetric);

    return 0;
}
