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

#include <boost/algorithm/string.hpp>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>
// #include <libs/graph/src/read_graphviz_new.cpp>

#include "model/BspInstance_csr.hpp"
#include "model/BspSchedule.hpp"
#include "model/BspScheduleRecomp.hpp"

#include "structures/bsp.hpp"
#include "structures/dag.hpp"

namespace FileReader {

bool readProblem(const std::string &filename, DAG &G, BSPproblem &params, bool NoNUMA = true);

std::pair<bool, BspInstance> readBspInstance(const std::string &filename);

std::pair<bool, ComputationalDag> readComputationalDagMetisFormat(std::ifstream &infile);

std::pair<bool, ComputationalDag> readComputationalDagMetisFormat(const std::string &filename);

std::pair<bool, BspArchitecture> readBspArchitecture(const std::string &filename);

std::pair<bool, BspArchitecture> readBspArchitecture(std::ifstream &infile);

std::pair<bool, ComputationalDag> readComputationalDagHyperdagFormat(const std::string &filename);

std::pair<bool, ComputationalDag> readComputationalDagHyperdagFormat(std::ifstream &infile);

std::pair<bool, ComputationalDag>
readComputationalDagMartixMarketFormat(const std::string &filename,
                                       std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &mtx);

std::pair<bool, ComputationalDag>
readComputationalDagMartixMarketFormat(std::ifstream &infile,
                                       std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &mtx);

std::pair<bool, ComputationalDag> readComputationalDagMartixMarketFormat(const std::string &filename);

std::pair<bool, ComputationalDag> readComputationalDagMartixMarketFormat(std::ifstream &infile);

std::pair<bool, csr_graph> readComputationalDagMartixMarketFormat_csr(const std::string &filename);

std::pair<bool, csr_graph> readComputationalDagMartixMarketFormat_csr(std::ifstream &infile);

std::pair<bool, BspArchitecture> readBspArchitecture(const std::string &filename);

std::pair<bool, BspArchitecture> readBspArchitecture(std::ifstream &infile);

std::pair<bool, ComputationalDag> readComputationalDagDotFormat(std::ifstream &infile);

std::pair<bool, ComputationalDag> readComputationalDagDotFormat(const std::string &filename);

std::pair<bool, BspSchedule> readBspSchdeuleTxtFormat(const BspInstance &instance, const std::string &filename);

std::pair<bool, BspSchedule> readBspSchdeuleTxtFormat(const BspInstance &instance, std::ifstream &infile);

/**
 * Reads a BspSchedule AND Instance in Dot format from a file. The parameter BspInstance is set as the instance of the
 * schedule. The ComputationalDag of the intance is supposed to be empty. Vertices are added as specified in the Dot
 * file.
 *
 *
 */
std::tuple<bool, BspSchedule> readBspScheduleDotFormat(const std::string &filename, BspInstance &instance);

/**
 * Reads a BspSchedule AND Instance in Dot format from a file. The parameter BspInstance is set as the instance of the
 * schedule. The ComputationalDag of the intance is supposed to be empty. Vertices are added as specified in the Dot
 * file.
 *
 *
 */
std::tuple<bool, BspSchedule> readBspScheduleDotFormat(std::ifstream &infile, BspInstance &instance);

/**
 * Reads a BspSchedule in Dot format from a file. Does not read an Instance form the DOT file. An appropriate instance
 * is meant to be passed as an agument and is set as the BspInstance of the schedule.
 *
 */
std::pair<bool, BspScheduleRecomp> extractBspScheduleRecomp(const std::string &filename, const BspInstance &instance);

/**
 * Reads a BspSchedule in Dot format from a file. Does not read an Instance form the DOT file. An appropriate instance
 * is meant to be passed as an agument and is set as the BspInstance of the schedule.
 *
 */
std::pair<bool, BspScheduleRecomp> extractBspScheduleRecomp(std::ifstream &infile, const BspInstance &instance);

} // namespace FileReader
