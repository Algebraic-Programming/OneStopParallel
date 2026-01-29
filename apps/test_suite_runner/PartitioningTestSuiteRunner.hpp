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

#include "AbstractTestSuiteRunner.hpp"
#include "StringToScheduler/run_partitioner.hpp"
#include "osp/auxiliary/io/mtx_hypergraph_file_reader.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/partitioning/model/partitioning.hpp"
#include "osp/partitioning/model/partitioning_replication.hpp"

namespace osp {

class PartitioningStatsModule : public IStatisticModule<Partitioning<HypergraphDefT>> {
  public:
    std::vector<std::string> GetMetricHeaders() const override { return {"Cost", "CutNet"}; }

    std::map<std::string, std::string> RecordStatistics(const Partitioning<HypergraphDefT> &partitioning,
                                                        std::ofstream & /*log_stream*/) const override {
        std::map<std::string, std::string> stats;
        stats["Cost"] = std::to_string(partitioning.ComputeConnectivityCost());
        stats["CutNet"] = std::to_string(partitioning.ComputeCutNetCost());
        return stats;
    }
};

template <typename GraphType>
class PartitioningTestSuiteRunner : public AbstractTestSuiteRunner<Partitioning<HypergraphDefT>, GraphType> {
  private:
  protected:
    ReturnStatus ComputeTargetObjectImpl(const BspInstance<GraphType> &instance,
                                         std::unique_ptr<Partitioning<HypergraphDefT>> &targetObject,
                                         const pt::ptree &algoConfig,
                                         long long &computationTimeMs) override {
        return ReturnStatus::ERROR;    // unused
    }

    void CreateAndRegisterStatisticModules(const std::string &moduleName) override {
        if (moduleName == "PartitioningStats") {
            this->activeStatsModules_.push_back(std::make_unique<PartitioningStatsModule>());
        }
    }

  public:
    PartitioningTestSuiteRunner() : AbstractTestSuiteRunner<Partitioning<HypergraphDefT>, GraphType>() {}

    int virtual Run(int argc, char *argv[]) override;
};

template <typename GraphType>
int PartitioningTestSuiteRunner<GraphType>::Run(int argc, char *argv[]) {
    using HypergraphT = HypergraphDefT;
    try {
        this->parser_.ParseArgs(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return 1;
    }

    if (!this->ParseCommonConfig()) {
        return 1;
    }

    this->SetupLogFile();

    CreateAndRegisterStatisticModules("PartitioningStats");
    this->SetupStatisticsFile();

    for (const auto &machineEntry : std::filesystem::recursive_directory_iterator(this->machineDirPath_)) {
        if (std::filesystem::is_directory(machineEntry)) {
            this->logStream_ << "Skipping directory " << machineEntry.path().string() << std::endl;
            continue;
        }
        std::string filenameMachine = machineEntry.path().string();
        std::string nameMachine = filenameMachine.substr(filenameMachine.rfind('/') + 1);
        if (nameMachine.rfind('.') != std::string::npos) {
            nameMachine = nameMachine.substr(0, nameMachine.rfind('.'));
        }

        // Temporary hack. Until there is no separate file format for partitioning problem parameters, we abuse
        // bsp arch files: 1st number is number of parts, 2nd is imbalance allowed (percentage), rest is ignored
        BspArchitecture<GraphType> arch;
        if (!file_reader::ReadBspArchitecture(filenameMachine, arch)) {
            this->logStream_ << "Reading architecture file " << filenameMachine << " failed." << std::endl;
            continue;
        }
        this->logStream_ << "Start Machine: " + filenameMachine + "\n";
        std::cout << "Start Machine: " + filenameMachine + "\n";

        for (const auto &graphEntry : std::filesystem::recursive_directory_iterator(this->graphDirPath_)) {
            if (std::filesystem::is_directory(graphEntry)) {
                this->logStream_ << "Skipping directory " << graphEntry.path().string() << std::endl;
                continue;
            }
            std::string filenameGraph = graphEntry.path().string();
            std::string nameGraph = filenameGraph.substr(filenameGraph.rfind('/') + 1);
            if (nameGraph.rfind('.') != std::string::npos) {
                nameGraph = nameGraph.substr(0, nameGraph.rfind('.'));
            }
            this->logStream_ << "Start Hypergraph: " + filenameGraph + "\n";
            std::cout << "Start Hypergraph: " + filenameGraph + "\n";

            bool graphStatus = false;
            GraphType dag;

            std::string fileEnding = filenameGraph.substr(filenameGraph.rfind(".") + 1);

            PartitioningProblem<HypergraphT>
                instance;    //(ConvertFromCdagAsHyperdag<HypergraphT, GraphType>(dag), arch.NumberOfProcessors());
            instance.SetNumberOfPartitions(arch.NumberOfProcessors());
            instance.SetMaxWorkWeightViaImbalanceFactor(static_cast<double>(arch.CommunicationCosts()) / 100.0);

            if (fileEnding == "mtx") {
                graphStatus = file_reader::ReadHypergraphMartixMarketFormat(
                    filenameGraph, instance.GetHypergraph(), MatrixToHypergraphFormat::FINE_GRAINED);

            } else if (fileEnding == "mtx2") {
                graphStatus = file_reader::ReadHypergraphMartixMarketFormat(
                    filenameGraph, instance.GetHypergraph(), MatrixToHypergraphFormat::ROW_NET);

            } else {
                graphStatus = file_reader::ReadGraph(filenameGraph, dag);
                instance.SetHypergraph(ConvertFromCdagAsHyperdag<HypergraphT, GraphType>(dag));
            }

            if (!graphStatus) {
                this->logStream_ << "Reading graph file " << filenameGraph << " failed." << std::endl;
                continue;
            }

            for (auto &algorithmConfigPair : this->parser_.scheduler_) {
                const pt::ptree &algoConfig = algorithmConfigPair.second;

                std::string currentAlgoName = algoConfig.get_child("name").get_value<std::string>();
                this->logStream_ << "Start Algorithm " + currentAlgoName + "\n";
                std::cout << "Start Algorithm " + currentAlgoName + "\n";

                long long computationTimeMs;
                const auto startTime = std::chrono::high_resolution_clock::now();

                std::pair<HypergraphT::VertexCommWeightType, HypergraphT::VertexCommWeightType> cost;
                ReturnStatus execStatus = RunPartitioner(this->parser_, algoConfig, instance, cost);

                const auto finishTime = std::chrono::high_resolution_clock::now();
                computationTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count();

                if (execStatus != ReturnStatus::OSP_SUCCESS && execStatus != ReturnStatus::BEST_FOUND) {
                    if (execStatus == ReturnStatus::ERROR) {
                        this->logStream_ << "Error computing with " << currentAlgoName << "." << std::endl;
                    } else if (execStatus == ReturnStatus::TIMEOUT) {
                        this->logStream_ << "Partitioner " << currentAlgoName << " timed out." << std::endl;
                    }
                    continue;
                }

                // currently not writing output to file

                if (this->statsOutStream_.is_open()) {
                    std::map<std::string, std::string> currentRowValues;
                    currentRowValues["Graph"] = nameGraph;
                    currentRowValues["Machine"] = nameMachine;
                    currentRowValues["Algorithm"] = currentAlgoName;
                    currentRowValues["TimeToCompute(ms)"] = std::to_string(computationTimeMs);
                    currentRowValues["Cost"] = std::to_string(cost.first);
                    currentRowValues["CutNet"] = std::to_string(cost.second);

                    for (size_t i = 0; i < this->allCsvHeaders_.size(); ++i) {
                        this->statsOutStream_ << currentRowValues[this->allCsvHeaders_[i]]
                                              << (i == this->allCsvHeaders_.size() - 1 ? "" : ",");
                    }
                    this->statsOutStream_ << "\n";
                }
            }
        }
    }
    return 0;
}

}    // namespace osp
