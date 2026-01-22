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

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ConfigParser.hpp"
#include "StatsModules/IStatsModule.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/return_status.hpp"
#include "osp/bsp/model/BspInstance.hpp"

// #define EIGEN_FOUND 1

#ifdef EIGEN_FOUND
#    include <Eigen/Sparse>
#    include <unsupported/Eigen/SparseExtra>

#    include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"
#endif

namespace osp {

namespace pt = boost::property_tree;

template <typename TargetObjectType, typename GraphType>
class AbstractTestSuiteRunner {
  protected:
    std::string executableDir_;
    ConfigParser parser_;
    std::ofstream logStream_;
    std::ofstream statsOutStream_;
    std::vector<std::string> allCsvHeaders_;
    std::vector<std::unique_ptr<IStatisticModule<TargetObjectType>>> activeStatsModules_;

    std::string graphDirPath_, machineDirPath_, outputTargetObjectDirPath_, logFilePath_, statisticsOutputFilePath_;
    bool writeTargetObjectToFile_ = false;
    unsigned timeLimitSeconds_ = 0;

    virtual std::filesystem::path GetExecutablePath() const { return std::filesystem::canonical("/proc/self/exe"); }

    virtual bool ParseCommonConfig() {
        try {
            executableDir_ = GetExecutablePath().remove_filename().string();
            timeLimitSeconds_ = parser_.globalParams_.get_child("timeLimit").get_value<unsigned>();
            writeTargetObjectToFile_ = parser_.globalParams_.get_child("outputSchedule").get_value_optional<bool>().value_or(false);

            graphDirPath_ = parser_.globalParams_.get_child("graphDirectory").get_value<std::string>();
            if (graphDirPath_.substr(0, 1) != "/") {
                graphDirPath_ = executableDir_ + graphDirPath_;
            }

            machineDirPath_ = parser_.globalParams_.get_child("archDirectory").get_value<std::string>();
            if (machineDirPath_.substr(0, 1) != "/") {
                machineDirPath_ = executableDir_ + machineDirPath_;
            }

            if (writeTargetObjectToFile_) {
                outputTargetObjectDirPath_ = parser_.globalParams_.get_child("scheduleDirectory").get_value<std::string>();
                if (outputTargetObjectDirPath_.substr(0, 1) != "/") {
                    outputTargetObjectDirPath_ = executableDir_ + outputTargetObjectDirPath_;
                }
                if (!outputTargetObjectDirPath_.empty() && !std::filesystem::exists(outputTargetObjectDirPath_)) {
                    std::filesystem::create_directories(outputTargetObjectDirPath_);
                }
            }

            logFilePath_ = parser_.globalParams_.get_child("outputLogFile").get_value<std::string>();
            if (logFilePath_.substr(0, 1) != "/") {
                logFilePath_ = executableDir_ + logFilePath_;
            }

            statisticsOutputFilePath_ = parser_.globalParams_.get_child("outputStatsFile").get_value<std::string>();
            if (statisticsOutputFilePath_.substr(0, 1) != "/") {
                statisticsOutputFilePath_ = executableDir_ + statisticsOutputFilePath_;
            }

            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error, invalid common config: " << e.what() << std::endl;
            return false;
        }
    }

    virtual void SetupLogFile() {
        logStream_.open(logFilePath_, std::ios_base::app);
        if (!logStream_.is_open()) {
            std::cerr << "Error: Could not open log file: " << logFilePath_ << std::endl;
        }
    }

    virtual void SetupStatisticsFile() {
        allCsvHeaders_ = {"Graph", "Machine", "Algorithm", "TimeToCompute(ms)"};

        std::set<std::string> uniqueModuleMetricHeaders;
        for (const auto &mod : activeStatsModules_) {
            for (const auto &header : mod->GetMetricHeaders()) {
                auto pair = uniqueModuleMetricHeaders.insert(header);

                if (!pair.second) {
                    logStream_ << "Warning: Duplicate metric header '" << header
                               << "' found across statistic modules. Using the first one encountered." << std::endl;
                }
            }
        }

        allCsvHeaders_.insert(allCsvHeaders_.end(), uniqueModuleMetricHeaders.begin(), uniqueModuleMetricHeaders.end());

        std::filesystem::path statsP(statisticsOutputFilePath_);
        if (statsP.has_parent_path() && !std::filesystem::exists(statsP.parent_path())) {
            std::filesystem::create_directories(statsP.parent_path());
        }

        bool fileExistsAndHasHeader = false;
        std::ifstream statsFileCheck(statisticsOutputFilePath_);
        if (statsFileCheck.is_open()) {
            std::string firstLineInFile;
            getline(statsFileCheck, firstLineInFile);
            std::string expectedHeaderLine;
            for (size_t i = 0; i < allCsvHeaders_.size(); ++i) {
                expectedHeaderLine += allCsvHeaders_[i] + (i == allCsvHeaders_.size() - 1 ? "" : ",");
            }
            if (firstLineInFile == expectedHeaderLine) {
                fileExistsAndHasHeader = true;
            }
            statsFileCheck.close();
        }

        statsOutStream_.open(statisticsOutputFilePath_, std::ios_base::app);
        if (!statsOutStream_.is_open()) {
            logStream_ << "CRITICAL ERROR: Could not open statistics output file: " << statisticsOutputFilePath_ << std::endl;
            std::cerr << "CRITICAL ERROR: Could not open statistics output file: " << statisticsOutputFilePath_ << std::endl;
        } else if (!fileExistsAndHasHeader) {
            for (size_t i = 0; i < allCsvHeaders_.size(); ++i) {
                statsOutStream_ << allCsvHeaders_[i] << (i == allCsvHeaders_.size() - 1 ? "" : ",");
            }
            statsOutStream_ << "\n";
            logStream_ << "Initialized statistics file " << statisticsOutputFilePath_ << " with header." << std::endl;
        }
    }

    virtual ReturnStatus ComputeTargetObjectImpl(const BspInstance<GraphType> &instance,
                                                 std::unique_ptr<TargetObjectType> &targetObject,
                                                 const pt::ptree &algoConfig,
                                                 long long &computationTimeMs)
        = 0;

    virtual void CreateAndRegisterStatisticModules(const std::string &moduleName) = 0;

    virtual void WriteTargetObjectHook(const TargetObjectType &, const std::string &, const std::string &, const std::string &) {
    }    // default in case TargetObjectType cannot be written to file

  public:
    AbstractTestSuiteRunner() {}

    virtual ~AbstractTestSuiteRunner() {
        if (logStream_.is_open()) {
            logStream_.close();
        }
        if (statsOutStream_.is_open()) {
            statsOutStream_.close();
        }
    }

    int virtual Run(int argc, char *argv[]) {
        try {
            parser_.ParseArgs(argc, argv);
        } catch (const std::exception &e) {
            std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
            return 1;
        }

        if (!ParseCommonConfig()) {
            return 1;
        }

        SetupLogFile();

        std::vector<std::string> activeModuleNamesFromConfig;
        try {
            for (const auto &item : parser_.globalParams_.get_child("activeStatisticModules")) {
                activeModuleNamesFromConfig.push_back(item.second.get_value<std::string>());
            }
        } catch (const pt::ptree_bad_path &e) {
            logStream_ << "Warning: 'activeStatisticModules' not found. No statistics modules will be run. " << e.what()
                       << std::endl;
        }

        for (const std::string &moduleName : activeModuleNamesFromConfig) {
            CreateAndRegisterStatisticModules(moduleName);
        }

        if (activeStatsModules_.empty()) {
            logStream_ << "No active statistic modules configured or loaded." << std::endl;
        }

        SetupStatisticsFile();

        for (const auto &machineEntry : std::filesystem::recursive_directory_iterator(machineDirPath_)) {
            if (std::filesystem::is_directory(machineEntry)) {
                logStream_ << "Skipping directory " << machineEntry.path().string() << std::endl;
                continue;
            }
            std::string filenameMachine = machineEntry.path().string();
            std::string nameMachine = filenameMachine.substr(filenameMachine.rfind('/') + 1);
            if (nameMachine.rfind('.') != std::string::npos) {
                nameMachine = nameMachine.substr(0, nameMachine.rfind('.'));
            }

            BspArchitecture<GraphType> arch;
            if (!file_reader::ReadBspArchitecture(filenameMachine, arch)) {
                logStream_ << "Reading architecture file " << filenameMachine << " failed." << std::endl;
                continue;
            }
            logStream_ << "Start Machine: " + filenameMachine + "\n";

            for (const auto &graphEntry : std::filesystem::recursive_directory_iterator(graphDirPath_)) {
                if (std::filesystem::is_directory(graphEntry)) {
                    logStream_ << "Skipping directory " << graphEntry.path().string() << std::endl;
                    continue;
                }
                std::string filenameGraph = graphEntry.path().string();
                std::string nameGraph = filenameGraph.substr(filenameGraph.rfind('/') + 1);
                if (nameGraph.rfind('.') != std::string::npos) {
                    nameGraph = nameGraph.substr(0, nameGraph.rfind('.'));
                }
                logStream_ << "Start Graph: " + filenameGraph + "\n";

                BspInstance<GraphType> bspInstance;
                bspInstance.GetArchitecture() = arch;
                bool graphStatus = false;
                std::string ext;
                if (filenameGraph.rfind('.') != std::string::npos) {
                    ext = filenameGraph.substr(filenameGraph.rfind('.') + 1);
                }

#ifdef EIGEN_FOUND

                using SmCsrInt32 = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>;
                using SmCscInt32 = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;
                using SmCsrInt64 = Eigen::SparseMatrix<double, Eigen::RowMajor, int64_t>;
                using SmCscInt64 = Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t>;
                SmCsrInt32 lCsrInt32;
                SmCsrInt64 lCsrInt64;
                SmCscInt32 lCscInt32{};
                SmCscInt64 lCscInt64{};

                if constexpr (std::is_same_v<GraphType, SparseMatrixGraphInt32T>
                              || std::is_same_v<GraphType, SparseMatrixGraphInt64T>) {
                    if (ext != "mtx") {
                        logStream_ << "Error: Only .mtx file is accepted for SpTRSV" << std::endl;
                        return 0;
                    }

                    if constexpr (std::is_same_v<GraphType, SparseMatrixGraphInt32T>) {
                        graphStatus = Eigen::loadMarket(lCsrInt32, filenameGraph);
                        if (!graphStatus) {
                            std::cerr << "Failed to read matrix from " << filenameGraph << std::endl;
                            return -1;
                        }

                        bspInstance.GetComputationalDag().SetCsr(&lCsrInt32);
                        lCscInt32 = lCsrInt32;
                        bspInstance.GetComputationalDag().SetCsc(&lCscInt32);
                    } else {
                        graphStatus = Eigen::loadMarket(lCsrInt64, filenameGraph);
                        if (!graphStatus) {
                            std::cerr << "Failed to read matrix from " << filenameGraph << std::endl;
                            return -1;
                        }

                        bspInstance.GetComputationalDag().SetCsr(&lCsrInt64);
                        lCscInt64 = lCsrInt64;
                        bspInstance.GetComputationalDag().SetCsc(&lCscInt64);
                    }
                } else {
#endif
                    graphStatus = file_reader::ReadGraph(filenameGraph, bspInstance.GetComputationalDag());

#ifdef EIGEN_FOUND
                }
#endif
                if (!graphStatus) {
                    logStream_ << "Reading graph file " << filenameGraph << " failed." << std::endl;
                    continue;
                }

                for (auto &algorithmConfigPair : parser_.scheduler_) {
                    const pt::ptree &algoConfig = algorithmConfigPair.second;

                    std::string currentAlgoName = algoConfig.get_child("name").get_value<std::string>();
                    logStream_ << "Start Algorithm " + currentAlgoName + "\n";

                    long long computationTimeMs;
                    std::unique_ptr<TargetObjectType> targetObject;

                    ReturnStatus execStatus = ComputeTargetObjectImpl(bspInstance, targetObject, algoConfig, computationTimeMs);

                    if (execStatus != ReturnStatus::OSP_SUCCESS && execStatus != ReturnStatus::BEST_FOUND) {
                        if (execStatus == ReturnStatus::ERROR) {
                            logStream_ << "Error computing with " << currentAlgoName << "." << std::endl;
                        } else if (execStatus == ReturnStatus::TIMEOUT) {
                            logStream_ << "Scheduler " << currentAlgoName << " timed out." << std::endl;
                        }
                        continue;
                    }

                    if (writeTargetObjectToFile_) {
                        try {
                            WriteTargetObjectHook(*targetObject, nameGraph, nameMachine, currentAlgoName);
                        } catch (const std::exception &e) {
                            logStream_ << "Writing target object file for " << nameGraph << ", " << nameMachine << ", "
                                       << currentAlgoName << " has failed: " << e.what() << std::endl;
                        }
                    }

                    if (statsOutStream_.is_open()) {
                        std::map<std::string, std::string> currentRowValues;
                        currentRowValues["Graph"] = nameGraph;
                        currentRowValues["Machine"] = nameMachine;
                        currentRowValues["Algorithm"] = currentAlgoName;
                        currentRowValues["TimeToCompute(ms)"] = std::to_string(computationTimeMs);

                        for (auto &statModule : activeStatsModules_) {
                            auto moduleMetrics = statModule->RecordStatistics(*targetObject, logStream_);
                            currentRowValues.insert(moduleMetrics.begin(), moduleMetrics.end());
                        }

                        for (size_t i = 0; i < allCsvHeaders_.size(); ++i) {
                            statsOutStream_ << currentRowValues[allCsvHeaders_[i]] << (i == allCsvHeaders_.size() - 1 ? "" : ",");
                        }
                        statsOutStream_ << "\n";
                    }
                }
            }
        }
        return 0;
    }
};

}    // namespace osp
