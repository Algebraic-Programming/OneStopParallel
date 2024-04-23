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

#include <iostream>

#include "structures/schedule.hpp"

// MAIN FUNCTIONS FOR SCHEDULES (R/W, CHECKS)
bool Schedule_Base::readDataFromFile(const std::string &filename, std::vector<int> &processor_assignment,
                                     std::vector<int> &time_assignment, bool NoNUMA) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input schedule file.\n";
        return false;
    }

    G.read(infile);
    params.read(infile, NoNUMA);

    int N = G.n;
    processor_assignment.clear();
    processor_assignment.resize(N);
    time_assignment.clear();
    time_assignment.resize(N);
    std::string line;

    // read schedule
    for (int i = 0; i < N; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return false;
        }

        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int node, inProc, inStep;
        sscanf(line.c_str(), "%d %d %d", &node, &inProc, &inStep);

        if (node < 0 || inProc < 0 || inStep < 0 || node >= N || inProc >= params.p) {
            std::cout << "Incorrect input file format (index out of range for one of "
                         "the schedule entries)."
                      << std::endl;
            return false;
        }

        processor_assignment[node] = inProc;
        time_assignment[node] = inStep;
    }
    infile.close();
    return true;
};

// reading schedule from file
bool Schedule::read(const std::string &filename, bool NoNUMA) {
    if (!readDataFromFile(filename, proc, supstep, NoNUMA))
        return false;

    CreateSupStepLists();
    if (!IsValid())
        return false;

    return true;
};

bool ClassicalSchedule::read(const std::string &filename, bool NoNUMA) {
    if (!readDataFromFile(filename, proc, time, NoNUMA))
        return false;

    if (!IsValid())
        return false;

    return true;
};

// auxiliary for classical schedule validity check
bool ClassicalSchedule::CheckJobOverlap() const {

    std::vector<std::vector<intPair>> jobs(params.p);

    for (int i = 0; i < G.n; ++i)
        jobs[proc[i]].emplace_back(time[i], time[i] + G.workW[i]);

    for (int i = 0; i < params.p; ++i)
        if (!isDisjoint(jobs[i])) {
            std::cout << "This is not a valid scheduling (jobs overlap at processor " << i << ")." << std::endl;
            return false;
        }

    return true;
};

// check if a classical (non-BSP) schedule is valid (assuming a commdelay model
// with work weights)
bool ClassicalSchedule::IsValid() const {
    if (!CheckJobOverlap())
        return false;

    const int delay = params.g;
    for (int i = 0; i < G.n; ++i)
        for (const int succ : G.Out[i]) {
            const int diff = (proc[i] == proc[succ]) ? 0 : delay * G.commW[i];
            if (time[i] + G.workW[i] + diff > time[succ]) {
                std::cout << "This is not a valid scheduling (problems with nodes " << i << " and " << succ << ")."
                          << std::endl;
                return false;
            }
        }

    return true;
};

Schedule ClassicalSchedule::ConvertToBSP(const std::vector<std::deque<int>> &procAssignmentLists) const {
    Schedule s;
    s.G = G;
    s.params = params;
    s.proc = proc;
    const int N = G.n;
    s.supstep.clear();
    s.supstep.resize(N);

    int superStepIdx = 0, totalNodesDone = 0;
    std::vector<bool> processed(N, false);
    std::vector<std::deque<int>::const_iterator> done(params.p), limit(params.p);
    for (int j = 0; j < params.p; ++j)
        done[j] = procAssignmentLists[j].begin();

    while (totalNodesDone < N) {
        // create next superstep
        int timeLimit = INT_MAX;
        for (int j = 0; j < params.p; ++j) {
            for (limit[j] = done[j]; limit[j] != procAssignmentLists[j].end(); ++limit[j]) {
                const int node = *limit[j];
                bool cut = false;
                for (const int source : G.In[node])
                    if (!processed[source] && proc[source] != proc[node])
                        cut = true;

                if (cut)
                    break;
            }
            if (limit[j] != procAssignmentLists[j].end() && time[*limit[j]] < timeLimit)
                timeLimit = time[*limit[j]];
        }

        for (int j = 0; j < params.p; ++j)
            for (; done[j] != limit[j] &&
                   (time[*done[j]] < timeLimit || (time[*done[j]] == timeLimit && G.workW[*done[j]] == 0));
                 ++done[j]) {
                processed[*done[j]] = true;
                s.supstep[*done[j]] = superStepIdx;
                ++totalNodesDone;
            }

        ++superStepIdx;
    }

    s.CreateSupStepLists();
    return s;
};

Schedule ClassicalSchedule::ConvertToBSP() const { return ConvertToBSP(getProcAssignmentLists()); }

std::vector<std::deque<int>> ClassicalSchedule::getProcAssignmentLists() const {
    std::vector<std::deque<int>> assignments(params.p);
    std::vector<std::map<intPair, int>> unsortedAssignments(params.p);

    // a secondary sort according to topological order is necessary for the
    // annoying case of 0-weight nodes
    std::vector<int> topOrder = G.GetTopOrder();
    std::vector<int> topOrderPos(G.n);

    for (int i = 0; i < G.n; ++i) {
        topOrderPos[topOrder[i]] = i;
        unsortedAssignments[proc[i]][intPair(time[i], topOrderPos[i])] = i;
    }

    for (int j = 0; j < params.p; ++j)
        for (auto itr = unsortedAssignments[j].begin(); itr != unsortedAssignments[j].end(); ++itr)
            assignments[j].push_back(itr->second);

    return assignments;
}

// check if BSP schedule is valid
bool Schedule::IsValid() const {
    for (int i = 0; i < G.n; ++i) {
        for (const int succ : G.Out[i]) {
            const int diff = (proc[i] == proc[succ]) ? 0 : 1;
            if (supstep[i] + diff > supstep[succ]) {
                std::cout << "This is not a valid scheduling (problems with nodes " << i << " and " << succ << ")."
                          << std::endl;
                return false;
            }

            if (!commSchedule.empty() && proc[i] != proc[succ])
                if (commSchedule[i][proc[succ]] < supstep[i] || commSchedule[i][proc[succ]] >= supstep[succ]) {
                    std::cout << "This is not a valid scheduling (problems with nodes " << i << " and " << succ << ")."
                              << std::endl;
                    return false;
                }
        }
    }

    return true;
};

// write BSP (problem and) schedule to file
bool Schedule::WriteToFile(const std::string &filename, const bool NoNUMA) const {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cout << "Unable to write/open output schedule file.\n";
        return false;
    };

    G.write(outfile);

    params.write(outfile, NoNUMA);

    for (int i = 0; i < G.n; ++i)
        outfile << i << " " << proc[i] << " " << supstep[i] << std::endl;

    if (!commSchedule.empty()) {
        int countLines = 0;
        for (int i = 0; i < G.n; ++i)
            for (int j = 0; j < params.p; ++j)
                if (commSchedule[i][j] >= 0)
                    ++countLines;

        outfile << countLines << std::endl;
        for (int i = 0; i < G.n; ++i)
            for (int j = 0; j < params.p; ++j)
                if (commSchedule[i][j] >= 0)
                    outfile << i << " " << proc[i] << " " << j << " " << commSchedule[i][j] << std::endl;
    }

    outfile.close();
    return true;
};

// COST CALCULATIONS

// compute classical schedule makespan
int ClassicalSchedule::GetCost() const {
    int mx = 0;
    for (int i = 0; i < G.n; ++i)
        if (time[i] + G.workW[i] > mx)
            mx = time[i] + G.workW[i];

    return mx;
};

// compute BSP schedule cost
int Schedule::GetCost() const {
    // IsValid();

    const auto nrSupSteps = supsteplists.size();
    const int N = G.n;
    int cost = 0;

    for (size_t i = 0; i < nrSupSteps; ++i) {
        int maxWork = 0;
        for (int j = 0; j < params.p; ++j) {
            int work = 0;
            for (const int node : supsteplists[i][j])
                work += G.workW[node];

            if (work > maxWork)
                maxWork = work;
        }

        cost += maxWork;
    }

    if (commSchedule.empty()) // lazy data sending - default comm schedule
    {

        std::vector<std::vector<bool>> present(N, std::vector<bool>(params.p, false));
        for (size_t i = 1; i < nrSupSteps; ++i) {
            std::vector<int> send(params.p, 0), receive(params.p, 0);

            for (int j = 0; j < params.p; ++j) {
                for (const int target : supsteplists[i][j]) {
                    for (int l = 0; l < G.In[target].size(); ++l) {
                        const int source = G.In[target][l];
                        if (proc[target] != proc[source] && !present[source][proc[target]]) {
                            present[source][proc[target]] = true;
                            send[proc[source]] += G.commW[source] * params.sendCost[proc[source]][proc[target]];
                            receive[proc[target]] += G.commW[source] * params.sendCost[proc[source]][proc[target]];
                        }
                    }
                }
            }

            int mx = 0;
            for (int j = 0; j < params.p; ++j) {
                if (send[j] > mx)
                    mx = send[j];
                if (receive[j] > mx)
                    mx = receive[j];
            }

            const int latency = mx > 0 ? params.L : 0;
            cost += params.g * mx + latency;
        }
    } else // we have a specifically defined comm schedule
    {
        std::vector<std::vector<int>> send(nrSupSteps - 1, std::vector<int>(params.p, 0)),
            receive(nrSupSteps - 1, std::vector<int>(params.p, 0));
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < params.p; ++j)
                if (commSchedule[i][j] >= 0) {

                    send[commSchedule[i][j]][proc[i]] += G.commW[i] * params.sendCost[proc[i]][j];
                    receive[commSchedule[i][j]][j] += G.commW[i] * params.sendCost[proc[i]][j];
                }

        for (size_t i = 1; i < nrSupSteps; ++i) {
            int mx = 0;
            for (int j = 0; j < params.p; ++j) {
                if (send[i - 1][j] > mx)
                    mx = send[i - 1][j];
                if (receive[i - 1][j] > mx)
                    mx = receive[i - 1][j];
            }

            const int latency = mx > 0 ? params.L : 0;
            cost += params.g * mx + latency;
        }
    }

    return cost;
};

// create superstep lists (for convenience) for a BSP schedule
void Schedule::CreateSupStepLists() {
    const int N = G.n;
    int nrSupSteps = 0;
    for (int i = 0; i < N; ++i)
        if (supstep[i] >= nrSupSteps)
            nrSupSteps = supstep[i] + 1;

    supsteplists.clear();
    supsteplists.resize(nrSupSteps, std::vector<std::list<int>>(params.p));

    std::vector<std::vector<int>> timer(nrSupSteps, std::vector<int>(params.p, 0));

    const std::vector<int> topOrder = G.GetTopOrder();
    for (int i = 0; i < N; ++i) {
        int node = topOrder[i];
        supsteplists[supstep[node]][proc[node]].push_back(node);
    }
};

// Combine subsequent supersteps whenever there is no communication inbetween
void Schedule::RemoveNeedlessSupSteps() {
    int step = 0;
    if (commSchedule.empty()) // lazy data sending - default comm schedule
    {
        auto nextBreak = supsteplists.size();
        for (size_t i = 0; i < supsteplists.size(); ++i) {
            if (nextBreak == i) {
                ++step;
                nextBreak = supsteplists.size();
            }
            for (int j = 0; j < params.p; ++j)
                for (const int node : supsteplists[i][j]) {
                    supstep[node] = step;
                    for (const int succ : G.Out[node])
                        if (proc[node] != proc[succ] && supstep[succ] < nextBreak)
                            nextBreak = supstep[succ];
                }
        }
    } else // concrete comm schedule
    {
        std::vector<bool> emptyStep(supsteplists.size(), true);
        for (int i = 0; i < G.n; ++i)
            for (int j = 0; j < params.p; ++j)
                if (commSchedule[i][j] >= 0)
                    emptyStep[commSchedule[i][j]] = false;

        std::vector<int> newIdx(supsteplists.size());
        for (size_t i = 0; i < supsteplists.size(); ++i) {
            newIdx[i] = step;
            if (!emptyStep[i])
                ++step;
        }
        for (int i = 0; i < G.n; ++i) {
            supstep[i] = newIdx[supstep[i]];
            for (int j = 0; j < params.p; ++j)
                if (commSchedule[i][j] >= 0)
                    commSchedule[i][j] = newIdx[commSchedule[i][j]];
        }
    }

    // update data structures
    CreateSupStepLists();
};

BspSchedule Schedule::ConvertToNewSchedule(const BspInstance &instance) const {
    BspSchedule new_bsp(instance);
    for (int node = 0; node < G.n; ++node) {
        new_bsp.setAssignedProcessor(node, proc[node]);
        new_bsp.setAssignedSuperstep(node, supstep[node]);
    }
    if (!commSchedule.empty()) {

        for (int node = 0; node < G.n; ++node)
            for (int processor = 0; processor < params.p; ++processor)
                if (commSchedule[node][processor] >= 0)

                    new_bsp.addCommunicationScheduleEntry(node, proc[node], processor, commSchedule[node][processor]);

    } else {

        new_bsp.setAutoCommunicationSchedule();
    }

    return new_bsp;
}
void Schedule::ConvertFromNewSchedule(const BspSchedule &new_bsp) {
    // G.ConvertFromNewDAG(new_bsp.getInstance().getComputationalDag());
    G = DAG(new_bsp.getInstance().getComputationalDag());
    params.ConvertFromNewBspParam(new_bsp.getInstance().getArchitecture());
    proc.clear();
    proc.resize(G.n);
    supstep.clear();
    supstep.resize(G.n);
    for (int node = 0; node < G.n; ++node) {
        proc[node] = new_bsp.assignedProcessor(node);
        supstep[node] = new_bsp.assignedSuperstep(node);
    }
    supsteplists.clear();

    CreateSupStepLists();
}
