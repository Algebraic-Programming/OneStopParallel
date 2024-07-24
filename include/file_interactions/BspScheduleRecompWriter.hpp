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

#include "model/BspScheduleRecomp.hpp"
#include "model/ComputationalDag.hpp"
#include <boost/graph/graphviz.hpp>

#include <fstream>
#include <string>

class BspScheduleRecompWriter {
  private:
    const BspScheduleRecomp &schedule;

  public:
    struct EdgeWriterSchedule_DOT {
        const GraphType &graph;

        EdgeWriterSchedule_DOT(const GraphType &graph_) : graph(graph_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "["
                << "comm_weight=\"" << graph[i].communicationWeight << "\";"
                << "]";
        }
    };

    struct VertexWriterSchedule_DOT {
        const BspScheduleRecomp &schedule;

        VertexWriterSchedule_DOT(const BspScheduleRecomp &schedule_) : schedule(schedule_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "["
                << "work_weight=\"" << schedule.getInstance().getComputationalDag().nodeWorkWeight(i) << "\";"
                << "comm_weight=\"" << schedule.getInstance().getComputationalDag().nodeCommunicationWeight(i) << "\";"
                << "mem_weight=\"" << schedule.getInstance().getComputationalDag().nodeMemoryWeight(i) << "\";"
                << "proc=\"(";
                for (size_t j = 0; j < schedule.assignedProcessors(i).size() - 1; ++j) {
                    out <<  schedule.assignedProcessors(i)[j] << ",";
                }
                out << schedule.assignedProcessors(i)[schedule.assignedProcessors(i).size() - 1] << ")\";"
                << "superstep=\"(";
                for (size_t j = 0; j < schedule.assignedSupersteps(i).size() - 1; ++j) {
                    out <<  schedule.assignedSupersteps(i)[j] << ",";
                }
                out << schedule.assignedSupersteps(i)[schedule.assignedSupersteps(i).size() - 1] << ")\";";

            bool found = false;

            for (const auto &[key, val] : schedule.getCommunicationSchedule()) {

                if (get<0>(key) == i) {

                    if (!found) {
                        out << "cs=\"[";
                        found = true;
                    } else {
                        out << ";";
                    }

                    out << "(" << get<1>(key) << "," << get<2>(key) << "," << val << ")";
                }
            }

            if (found) {
                out << "]\";";
            }

            out << "]";
        }
    };

    struct VertexWriterDuplicateSchedule_DOT {
        const BspSchedule &schedule;

        VertexWriterDuplicateSchedule_DOT(const BspSchedule &schedule_) : schedule(schedule_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "["
                << "label=\"" << i << "\";"
                << "work_weight=\"" << schedule.getInstance().getComputationalDag().nodeWorkWeight(i) << "\";"
                << "comm_weight=\"" << schedule.getInstance().getComputationalDag().nodeCommunicationWeight(i) << "\";"
                << "mem_weight=\"" << schedule.getInstance().getComputationalDag().nodeMemoryWeight(i) << "\";"
                << "proc=\"" << schedule.assignedProcessor(i) << "\";"
                << "superstep=\"" << schedule.assignedSuperstep(i) << "\";";

            out << "]";
        }
    };



    /**
     * Constructs a BspScheduleWriter object with the given BspSchdule.
     *
     * @param schdule_
     */
    BspScheduleRecompWriter(const BspScheduleRecomp &schedule_) : schedule(schedule_) {}

    /**
     * Writes the BspSchedule to the specified output stream in DOT format.
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param os The output stream to write the DOT representation of the computational DAG.
     */
    template<class VertexWriterType = VertexWriterSchedule_DOT, class EdgeWriterType = EdgeWriterSchedule_DOT>
    void write_dot(std::ostream &os) const;

    /**
     * Writes the BspSchedule to the specified file in DOT format.
     *
     * @note The file will be overwritten if it already exists.
     * @note The file will be created if it does not exist.
     * @note This function is an alias for `write_dot(std::ostream &os)`
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param filename The name of the file to write the DOT representation of the computational DAG.
     */
    template<class VertexWriterType = VertexWriterSchedule_DOT, class EdgeWriterType = EdgeWriterSchedule_DOT>
    void write_dot(const std::string &filename) const {
        std::ofstream os(filename);
        write_dot(os);
    }


    template<class VertexWriterType = VertexWriterDuplicateSchedule_DOT, class EdgeWriterType = EdgeWriterSchedule_DOT>
    void write_dot_duplicate_vertices(std::ostream &os) const;


};

template<class VertexWriterType, class EdgeWriterType>
void BspScheduleRecompWriter::write_dot(std::ostream &os) const {
    const auto &g = schedule.getInstance().getComputationalDag().getGraph();
    boost::write_graphviz(os, g, VertexWriterType(schedule), EdgeWriterType(g));
}


template<class VertexWriterType, class EdgeWriterType>
void BspScheduleRecompWriter::write_dot_duplicate_vertices(std::ostream &os) const {
    
    
    const auto &g = schedule.getInstance().getComputationalDag().getGraph();

    
    boost::write_graphviz(os, g, VertexWriterType(schedule), EdgeWriterType(g));
}