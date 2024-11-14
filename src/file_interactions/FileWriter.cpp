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

#include "file_interactions/FileWriter.hpp"

void FileWriter::writeBspArchitecture(const BspArchitecture &architecture, std::string filename) {
    std::ofstream os(filename);
    if (!os.is_open())
        std::cout << "Unable to write/open output architecture file.\n";
    else {
        writeBspArchitecture(architecture, os);
    }
}

// write machine parameters to file
void FileWriter::writeBspArchitecture(const BspArchitecture &architecture, std::ofstream &outfile) {
    outfile << architecture.numberOfProcessors() << " " << architecture.communicationCosts() << " "
            << architecture.synchronisationCosts() << std::endl;

    if (architecture.isNumaArchitecture())
        for (unsigned i = 0; i < architecture.numberOfProcessors(); ++i)
            for (unsigned j = 0; j < architecture.numberOfProcessors(); ++j)
                outfile << i << " " << j << " " << architecture.sendCosts(i, j) << std::endl;
};

void FileWriter::writeComputationalDagHyperDagFormat(const ComputationalDag &dag, std::string filename) {
    std::ofstream os(filename);
    if (!os.is_open())
        std::cout << "Unable to write/open output ComputationalDagHyperDagFormat file.\n";
    else {
        writeComputationalDagHyperDagFormat(dag, os);
      }
}

void FileWriter::writeComputationalDagHyperDagFormat(const ComputationalDag &dag, std::ofstream &outfile) {

    unsigned sinks = 0, pins = 0;
    for (const auto &node : dag.vertices()) {
        if (dag.isSink(node))
            ++sinks;
        else
            pins += 1 + dag.numberOfChildren(node);
    }

    outfile << dag.numberOfVertices() - sinks << " " << dag.numberOfVertices() << " " << pins << "\n";

    unsigned edgeIndex = 0;
    for (const auto &node : dag.vertices()) {
        if (not dag.isSink(node)) {

            outfile << edgeIndex << " " << node << "\n";

            for (const auto &child : dag.children(node)) {
                outfile << edgeIndex << " " << child << "\n";
            }
            ++edgeIndex;
        }
    }

    for (const auto &node : dag.vertices()) {
        outfile << node << " " << dag.nodeWorkWeight(node) << " " << dag.nodeCommunicationWeight(node) << " " << dag.nodeMemoryWeight(node) << " " << dag.nodeType(node) << "\n";
    }
}

void FileWriter::writePermutationFile(std::vector<size_t> perm, std::string filename) {
       
    std::ofstream os(filename);
    if (!os.is_open())
        std::cout << "Unable to write/open output permutation file.\n";
    else {
        writePermutationFile(perm, os);
    }
}

void FileWriter::writePermutationFile(std::vector<size_t> perm, std::ofstream &outfile) {

    std::vector<size_t> inverse_permutation(perm.size());
    for (size_t i = 0; i < perm.size(); ++i)
        inverse_permutation[perm[i]] = i;

    for (size_t i = 0; i < perm.size(); ++i)
        outfile << inverse_permutation[i] << std::endl;
}
