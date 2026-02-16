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

#include "osp/bsp/model/BspScheduleRecomp.hpp"

namespace osp {

/**
 * @brief The GreedyReccomputer class applies a greedy algorithm to replace some of the communciation steps in
 * a BspSchedule by recomputation steps if this decreases the cost.
 */
template <typename GraphT>
class GreedyRecomputer {
    static_assert(isComputationalDagV<GraphT>, "GreedyRecomputer can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;
    using CostType = VWorkwT<GraphT>;
    using KeyTriple = std::tuple<VertexIdxT<GraphT>, unsigned int, unsigned int>;

    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "GreedyRecomputer requires work and comm. weights to have the same type.");

    // auxiliary data to handle schedule efficiently
    std::vector<std::vector<CostType>> workCost_, sendCost_, recCost_;
    std::vector<std::vector<unsigned>> firstPresent_;
    std::vector<std::vector<std::multiset<unsigned> > > neededOnProc_;
    std::vector<std::vector<std::vector<VertexIdx> > > nodesPerProcAndStep_;
    std::vector<CostType> maxWork_, maxComm_;
    std::vector<std::set<KeyTriple> > commSteps_;

    void RefreshAuxData(const BspScheduleRecomp<GraphT> &schedule);

    // elementary operations to edit schedule - add/remove step, and update data structures
    void AddCommStep(const BspScheduleRecomp<GraphT> &schedule, const KeyTriple &newComm, const unsigned step);
    void RemoveCommStep(const BspScheduleRecomp<GraphT> &schedule, const KeyTriple &removedComm, const unsigned step);
    void AddRecomputeStep(BspScheduleRecomp<GraphT> &schedule, const VertexIdx node, const unsigned proc, const unsigned step);

    // DIFFERENT TECHNIQUES TO IMPROVE SCHEDULE BY INTRODUCING RECOMPUTATION
    // (return values show whether there were any succesful improvement steps)

    // Replace single comm. steps by recomp, if it is better
    bool GreedyImprove(BspScheduleRecomp<GraphT> &schedule);

    // Merge consecutive supersteps using recomp, if it is better
    bool MergeEntireSupersteps(BspScheduleRecomp<GraphT> &schedule);

    // Copy all the (necessary) nodes from one processor to another in a superstep, if it is better
    bool RecomputeEntireSupersteps(BspScheduleRecomp<GraphT> &schedule);

    // Remove multiple comm steps from the same superstep at once, attempting to escape local minima
    bool BatchRemoveSteps(BspScheduleRecomp<GraphT> &schedule);

  public:
    /**
     * @brief Default destructor for GreedyRecomputer.
     */
    virtual ~GreedyRecomputer() = default;

    ReturnStatus ComputeRecompScheduleBasic(BspScheduleCS<GraphT> &initialSchedule, BspScheduleRecomp<GraphT> &recompSchedule);

    ReturnStatus ComputeRecompScheduleAdvanced(BspScheduleCS<GraphT> &initialSchedule, BspScheduleRecomp<GraphT> &recompSchedule);
};

template <typename GraphT>
ReturnStatus GreedyRecomputer<GraphT>::ComputeRecompScheduleBasic(BspScheduleCS<GraphT> &initialSchedule, BspScheduleRecomp<GraphT> &recompSchedule)
{
    recompSchedule = BspScheduleRecomp<GraphT>(initialSchedule);
    GreedyImprove(recompSchedule);
    recompSchedule.MergeSupersteps();
    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT>
ReturnStatus GreedyRecomputer<GraphT>::ComputeRecompScheduleAdvanced(BspScheduleCS<GraphT> &initialSchedule, BspScheduleRecomp<GraphT> &recompSchedule)
{
    recompSchedule = BspScheduleRecomp<GraphT>(initialSchedule);
    bool keepsImproving = true;
    while (keepsImproving)
    {
      keepsImproving = BatchRemoveSteps(recompSchedule); // no need for greedyImprove if we use this more general one
      recompSchedule.MergeSupersteps();

      keepsImproving = MergeEntireSupersteps(recompSchedule) || keepsImproving;
      recompSchedule.CleanSchedule();
      recompSchedule.MergeSupersteps();

      keepsImproving = RecomputeEntireSupersteps(recompSchedule) || keepsImproving;
      recompSchedule.MergeSupersteps();

      // add further methods, if desired
    }

    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT>
bool GreedyRecomputer<GraphT>::GreedyImprove(BspScheduleRecomp<GraphT> &schedule)
{
    const VertexIdx N = schedule.GetInstance().NumberOfVertices();
    const unsigned P = schedule.GetInstance().NumberOfProcessors();
    const unsigned S = schedule.NumberOfSupersteps();
    const GraphT &G = schedule.GetInstance().GetComputationalDag();

    bool improved = false;

    // Initialize required data structures
    RefreshAuxData(schedule);

    std::vector<std::vector<unsigned>> firstComputable(N, std::vector<unsigned>(P, 0U));
    for (VertexIdx node = 0; node < N; ++node) {
      for (const VertexIdx &pred : G.Parents(node)) {
        for (unsigned proc = 0; proc < P; ++proc) {
          firstComputable[node][proc] = std::max(firstComputable[node][proc], firstPresent_[pred][proc]);
        }
      }
    }

    // Find improvement steps
    bool stillImproved = true;
    while (stillImproved) {
      stillImproved = false;

      for (unsigned step = 0; step < S; ++step) {
        std::vector<KeyTriple> toErase;
        for (const KeyTriple &entry : commSteps_[step]) {
          const VertexIdx &node = std::get<0>(entry);
          const unsigned &fromProc = std::get<1>(entry);
          const unsigned &toProc = std::get<2>(entry);

          // check how much comm cost we save by removing comm schedule entry
          CostType commInduced = G.VertexCommWeight(node)
                                   * schedule.GetInstance().GetArchitecture().CommunicationCosts(fromProc, toProc);

          CostType newMaxComm = 0;
          for (unsigned proc = 0; proc < P; ++proc) {
            if (proc == fromProc) {
              newMaxComm = std::max(newMaxComm, sendCost_[proc][step] - commInduced);
            } else {
              newMaxComm = std::max(newMaxComm, sendCost_[proc][step]);
            }
            if (proc == toProc) {
              newMaxComm = std::max(newMaxComm, recCost_[proc][step] - commInduced);
            } else {
              newMaxComm = std::max(newMaxComm, recCost_[proc][step]);
            }
          }
          if (newMaxComm == maxComm_[step]) {
            continue;
          }

          if (!schedule.GetInstance().IsCompatible(node, toProc)) {
            continue;
          }

          CostType decrease = maxComm_[step] - newMaxComm;
          if (maxComm_[step] > 0 && newMaxComm == 0) {
            decrease += schedule.GetInstance().GetArchitecture().SynchronisationCosts();
          }

          // check how much it would increase the work cost instead
          unsigned bestStep = S;
          CostType smallestIncrease = std::numeric_limits<CostType>::max();
          for (unsigned compStep = firstComputable[node][toProc]; compStep <= *neededOnProc_[node][toProc].begin(); ++compStep) {
            CostType increase = workCost_[toProc][compStep] + G.VertexWorkWeight(node) > maxWork_[compStep]
                                  ? workCost_[toProc][compStep] + G.VertexWorkWeight(node) - maxWork_[compStep]
                                  : 0;

            if (increase < smallestIncrease) {
              bestStep = compStep;
              smallestIncrease = increase;
            }
          }

          // check if this modification is beneficial
          if (bestStep == S || smallestIncrease > decrease) {
            continue;
          }

          // execute the modification
          toErase.emplace_back(entry);
          AddRecomputeStep(schedule, node, toProc, bestStep);
          improved = true;

          sendCost_[fromProc][step] -= commInduced;
          recCost_[toProc][step] -= commInduced;
          maxComm_[step] = newMaxComm;

          maxWork_[bestStep] += smallestIncrease;

          // update movability bounds
          neededOnProc_[node][fromProc].erase(neededOnProc_[node][fromProc].lower_bound(step));

          firstPresent_[node][toProc] = bestStep;
          for (const VertexIdx &succ : G.Children(node)) {
            firstComputable[succ][toProc] = 0U;
            for (const VertexIdx &pred : G.Parents(succ)) {
              firstComputable[succ][toProc] = std::max(firstComputable[succ][toProc], firstPresent_[pred][toProc]);
            }
          }

          stillImproved = true;
        }
        for (const KeyTriple &entry : toErase) {
          commSteps_[step].erase(entry);
        }
      }
    }

    schedule.GetCommunicationSchedule().clear();
    for (unsigned step = 0; step < S; ++step) {
      for (const KeyTriple &entry : commSteps_[step]) {
        schedule.AddCommunicationScheduleEntry(entry, step);
      }
    }

    return improved;
}

template <typename GraphT>
bool GreedyRecomputer<GraphT>::MergeEntireSupersteps(BspScheduleRecomp<GraphT> &schedule)
{
  bool improved = false;
  RefreshAuxData(schedule);
  std::vector<bool> stepRemoved(schedule.NumberOfSupersteps(), false);

  const GraphT &G = schedule.GetInstance().GetComputationalDag();

  unsigned previousStep = 0;
  for (unsigned step = 0; step < schedule.NumberOfSupersteps() - 1; ++step) {
    if (stepRemoved[step]) {
      continue;
    }

    for (unsigned nextStep = step + 1; nextStep < schedule.NumberOfSupersteps(); ++nextStep) {

      // TRY TO MERGE step AND nextStep
      std::set<KeyTriple> newCommStepsBefore, newCommStepsAfter;
      std::set<std::pair<VertexIdx, unsigned> > newWorkSteps;

      std::vector<std::set<VertexIdx> > mustReplicate(schedule.GetInstance().NumberOfProcessors());

      for (const KeyTriple &entry : commSteps_[step]) {
        const VertexIdx &node = std::get<0>(entry);
        const unsigned &fromProc = std::get<1>(entry);
        const unsigned &toProc = std::get<2>(entry);

        bool used = false;
        if (neededOnProc_[node][toProc].empty() || *neededOnProc_[node][toProc].begin() > nextStep) {
          newCommStepsAfter.insert(entry);
          continue;
        }

        if (step > 0 && firstPresent_[node][fromProc] <= previousStep) {
          newCommStepsBefore.insert(entry);
        } else {
          mustReplicate[toProc].insert(node);
        }
      }
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        for (const VertexIdx node : nodesPerProcAndStep_[proc][nextStep]) {
          newWorkSteps.emplace(node, proc);
        }

        while (!mustReplicate[proc].empty()) {
          const VertexIdx node = *mustReplicate[proc].begin();
          mustReplicate[proc].erase(mustReplicate[proc].begin());
          if (newWorkSteps.find(std::make_pair(node, proc)) != newWorkSteps.end()) {
            continue;
          }
          newWorkSteps.emplace(node, proc);
          for (const VertexIdx &pred : G.Parents(node)) {
            if (firstPresent_[pred][proc] <= step) {
              continue;
            }

            unsigned sendFromProcBefore = std::numeric_limits<unsigned>::max();
            for (unsigned procOffset = 0; procOffset < schedule.GetInstance().NumberOfProcessors(); ++procOffset) {
              unsigned fromProc = (proc + procOffset) % schedule.GetInstance().NumberOfProcessors();
              if (step > 0 && firstPresent_[pred][fromProc] <= previousStep) {
                sendFromProcBefore = fromProc;
                break;
              }
            }
            if (sendFromProcBefore < std::numeric_limits<unsigned>::max()) {
              newCommStepsBefore.emplace(pred, sendFromProcBefore, proc);
            } else {
              mustReplicate[proc].insert(pred);
            }
          }
        }
      }

      // now that newWorkSteps is finalized, check types
      bool typesIncompatible = false;
      for (const std::pair<VertexIdx, unsigned> &nodeAndProc : newWorkSteps) {
        if (!schedule.GetInstance().IsCompatible(nodeAndProc.first, nodeAndProc.second)) {
            typesIncompatible = true;
            break;
        }
      }
      if (typesIncompatible) {
        break;
      }

      // EVALUATE COST
      int costChange = 0;

      // work cost in merged step
      std::vector<CostType> newWorkCost(schedule.GetInstance().NumberOfProcessors());
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        newWorkCost[proc] = workCost_[proc][step];
      }

      for (const std::pair<VertexIdx, unsigned> &newCompute : newWorkSteps) {
        newWorkCost[newCompute.second] += G.VertexWorkWeight(newCompute.first);
      }

      CostType newMax = 0;
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        newMax = std::max(newMax, newWorkCost[proc]);
      }

      costChange += static_cast<int>(newMax) - static_cast<int>(maxWork_[step] + maxWork_[nextStep]);

      // comm cost before merged step
      std::vector<CostType> newSendCost(schedule.GetInstance().NumberOfProcessors()), newRecCost(schedule.GetInstance().NumberOfProcessors());
      if (step > 0) {
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          newSendCost[proc] = sendCost_[proc][previousStep];
          newRecCost[proc] = recCost_[proc][previousStep];
        }
        for (const KeyTriple &newComm : newCommStepsBefore) {
          CostType commCost = G.VertexCommWeight(std::get<0>(newComm)) *
                                      schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(newComm), std::get<2>(newComm));
          newSendCost[std::get<1>(newComm)] += commCost;
          newRecCost[std::get<2>(newComm)] += commCost;
        }

        newMax = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          newMax = std::max(newMax, newSendCost[proc]);
          newMax = std::max(newMax, newRecCost[proc]);
        }
        costChange += static_cast<int>(newMax) - static_cast<int>(maxComm_[previousStep]);

        CostType oldSync = (maxComm_[previousStep] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;
        CostType newSync = (newMax > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;

        costChange += static_cast<int>(newSync) - static_cast<int>(oldSync);
      }

      // comm cost after merged step
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        newSendCost[proc] = sendCost_[proc][nextStep];
        newRecCost[proc] = recCost_[proc][nextStep];
      }
      for (const KeyTriple &newComm : newCommStepsAfter) {
        CostType commCost = G.VertexCommWeight(std::get<0>(newComm))
                              * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(newComm), std::get<2>(newComm));
        newSendCost[std::get<1>(newComm)] += commCost;
        newRecCost[std::get<2>(newComm)] += commCost;
      }

      newMax = 0;
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        newMax = std::max(newMax, newSendCost[proc]);
        newMax = std::max(newMax, newRecCost[proc]);
      }
      costChange += static_cast<int>(newMax) - static_cast<int>(maxComm_[step] + maxComm_[nextStep]);

      CostType oldSync = ((maxComm_[step] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0)
                            + ((maxComm_[nextStep] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0);
      CostType newSync = (newMax > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;

      costChange += static_cast<int>(newSync) - static_cast<int>(oldSync);

      if (costChange < 0)
      {
        // MERGE STEPS - change schedule and update data structures

        // update assignments and compute data
        for (const std::pair<VertexIdx, unsigned> &nodeAndProc : newWorkSteps) {
          AddRecomputeStep(schedule, nodeAndProc.first, nodeAndProc.second, step);
        }
        maxWork_[step] = 0;
        maxWork_[nextStep] = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          maxWork_[step] = std::max(maxWork_[step], workCost_[proc][step]);
          workCost_[proc][nextStep] = 0;
          for (const VertexIdx node : nodesPerProcAndStep_[proc][nextStep]) {
            auto &assignments = schedule.Assignments(node);
            for (auto itr = assignments.begin(); itr != assignments.end(); ++itr) {
              if (*itr == std::make_pair(proc, nextStep)) {
                assignments.erase(itr);
                break;
              }
            }
            for (const VertexIdx &pred : G.Parents(node)) {
              neededOnProc_[pred][proc].erase(neededOnProc_[pred][proc].lower_bound(nextStep));
            }
          }
          nodesPerProcAndStep_[proc][nextStep].clear();
        }

        // update comm and its data in step (imported mostly from nextStep)
        for (const KeyTriple &entry : commSteps_[step]) {
          neededOnProc_[std::get<0>(entry)][std::get<1>(entry)].erase(neededOnProc_[std::get<0>(entry)][std::get<1>(entry)].lower_bound(step));
        }

        for (const KeyTriple &entry : commSteps_[nextStep]) {
          neededOnProc_[std::get<0>(entry)][std::get<1>(entry)].erase(neededOnProc_[std::get<0>(entry)][std::get<1>(entry)].lower_bound(nextStep));
        }

        commSteps_[step].clear();
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          sendCost_[proc][step] = 0;
          recCost_[proc][step] = 0;
          sendCost_[proc][nextStep] = 0;
          recCost_[proc][nextStep] = 0;
        }
        std::set<KeyTriple> commNextSteps = commSteps_[nextStep];
        commSteps_[nextStep].clear();
        for (const KeyTriple &newComm : commNextSteps) {
          AddCommStep(schedule, newComm, step);
        }

        for (const KeyTriple &newComm : newCommStepsAfter) {
          AddCommStep(schedule, newComm, step);
        }

        maxComm_[nextStep] = 0;

        maxComm_[step] = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          maxComm_[step] = std::max(maxComm_[step], sendCost_[proc][step]);
          maxComm_[step] = std::max(maxComm_[step], recCost_[proc][step]);
        }

        // update comm and its data in step-1
        if (step > 0) {
          for (const KeyTriple &newComm : newCommStepsBefore) {
            AddCommStep(schedule, newComm, previousStep);
          }

          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            maxComm_[previousStep] = std::max(maxComm_[previousStep], sendCost_[proc][previousStep]);
            maxComm_[previousStep] = std::max(maxComm_[previousStep], recCost_[proc][previousStep]);
          }
        }

        stepRemoved[nextStep] = true;
        improved = true;
      } else {
        break;
      }
    }
    previousStep = step;
  }

  schedule.GetCommunicationSchedule().clear();
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (const KeyTriple &entry : commSteps_[step]) {
      schedule.AddCommunicationScheduleEntry(entry, step);
    }
  }

  return improved;
}

template <typename GraphT>
bool GreedyRecomputer<GraphT>::RecomputeEntireSupersteps(BspScheduleRecomp<GraphT> &schedule)
{
  bool improved = false;
  RefreshAuxData(schedule);

  const GraphT &G = schedule.GetInstance().GetComputationalDag();

  std::map<std::pair<VertexIdx, unsigned>, std::vector<std::pair<unsigned, unsigned>>> commStepPerNodeAndReceiver;
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (const KeyTriple &entry : commSteps_[step]) {
      commStepPerNodeAndReceiver[std::make_pair(std::get<0>(entry), std::get<2>(entry))].emplace_back(std::get<1>(entry), step);
    }
  }

  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (unsigned fromProc = 0; fromProc < schedule.GetInstance().NumberOfProcessors(); ++fromProc) {
      for (unsigned toProc = 0; toProc < schedule.GetInstance().NumberOfProcessors(); ++toProc) {
        if (fromProc == toProc) {
          continue;
        }

        // ATTEMPT TO REPLICATE all the necessary nodes of (fromProc, step) on (toProc, step)

        // collect the nodes that would be useful to replicate (not present before, not unnecessary)
        std::set<KeyTriple> newCommStepsBefore, removedCommStepsAfter;
        std::set<VertexIdx> mustReplicate;

        for (const VertexIdx node : nodesPerProcAndStep_[fromProc][step])
        {
          if (firstPresent_[node][toProc] <= step) {
            continue;
          }
          mustReplicate.insert(node);
        }

        std::map<VertexIdx, unsigned> internalOutDegree;
        for (const VertexIdx node : mustReplicate) {
          internalOutDegree[node] = 0;
        }
        for (const VertexIdx node : mustReplicate) {
          for (const VertexIdx &pred : G.Parents(node)) {
            if (mustReplicate.find(pred) == mustReplicate.end()) {
              continue;
            }
            internalOutDegree[pred] += 1;
          }
        }

        std::set<VertexIdx> checkIfDisposable;
        for (const VertexIdx node : mustReplicate) {
          if (internalOutDegree.at(node) == 0) {
            checkIfDisposable.insert(node);
          }
        }

        while (!checkIfDisposable.empty()) {
          const VertexIdx node = *checkIfDisposable.begin();
          checkIfDisposable.erase(checkIfDisposable.begin());
          if (neededOnProc_[node][toProc].empty()) {
            mustReplicate.erase(node);
            for (const VertexIdx &pred : G.Parents(node)) {
              if (mustReplicate.find(pred) == mustReplicate.end()) {
                continue;
              }
              if ((--internalOutDegree[pred]) == 0) {
                checkIfDisposable.insert(pred);
              }
            }
          }
        }

        // now that mustReplicate is finalized, check types
        bool typesIncompatible = false;
        for (const VertexIdx node : mustReplicate) {
          if (!schedule.GetInstance().IsCompatible(node, toProc)) {
              typesIncompatible = true;
              break;
            }
        }
        if (typesIncompatible) {
          continue;
        }

        // collect new comm steps - before
        for (const VertexIdx node : mustReplicate) {
          for (const VertexIdx &pred : G.Parents(node)) {
            if (firstPresent_[pred][toProc] <= step || mustReplicate.find(pred) != mustReplicate.end()) {
              continue;
            }

            unsigned sendFromProcBefore = fromProc;
            for (unsigned procOffset = 0; procOffset < schedule.GetInstance().NumberOfProcessors(); ++procOffset) {
              unsigned sendFromCandidate = (fromProc + procOffset) % schedule.GetInstance().NumberOfProcessors();
              if (step > 0 && firstPresent_[pred][sendFromCandidate] <= step - 1) {
                sendFromProcBefore = sendFromCandidate;
                break;
              }
            }
            if (sendFromProcBefore < std::numeric_limits<unsigned>::max()) {
              newCommStepsBefore.emplace(pred, sendFromProcBefore, toProc);
            } else {
              std::cout<<"ERROR: parent of replicated node not present anywhere."<<std::endl;
            }
          }
        }

        // collect comm steps to remove afterwards
        for (const VertexIdx node : mustReplicate) {
          for (const std::pair<unsigned, unsigned> &entry : commStepPerNodeAndReceiver[std::make_pair(node, toProc)]) {
            removedCommStepsAfter.emplace(node, entry.first, entry.second);
          }
        }

        // EVALUATE COST

        int costChange = 0;

        // work cost
        CostType newWorkCost = workCost_[toProc][step];
        for (const VertexIdx node : mustReplicate) {
          newWorkCost += G.VertexWorkWeight(node);
        }
        CostType newMax = std::max(maxWork_[step], newWorkCost);

        costChange += static_cast<int>(newMax) - static_cast<int>(maxWork_[step]);

        // comm cost before merged step
        if (step > 0) {
          std::vector<CostType> newSendCost(schedule.GetInstance().NumberOfProcessors());
          CostType newRecCost = recCost_[toProc][step-1];
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            newSendCost[proc] = sendCost_[proc][step-1];
          }
          for (const KeyTriple &newComm : newCommStepsBefore)
          {
            CostType commCost = G.VertexCommWeight(std::get<0>(newComm))
                                  * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(newComm), std::get<2>(newComm));
            newSendCost[std::get<1>(newComm)] += commCost;
            newRecCost += commCost;
          }

          newMax = std::max(maxComm_[step - 1], newRecCost);
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            newMax = std::max(newMax, newSendCost[proc]);
          }
          costChange += static_cast<int>(newMax) - static_cast<int>(maxComm_[step - 1]);

          CostType oldSync = (maxComm_[step - 1] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;
          CostType newSync = (newMax > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;

          costChange += static_cast<int>(newSync) - static_cast<int>(oldSync);
        }

        // comm cost after merged step
        std::map<unsigned, std::map<unsigned, CostType>> changedStepsSent;
        std::map<unsigned, CostType> changedStepsRec;
        for (const KeyTriple &newComm : removedCommStepsAfter) {
          CostType commCost = G.VertexCommWeight(std::get<0>(newComm))
                                * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(newComm), toProc);
          if (changedStepsSent[std::get<2>(newComm)].find(std::get<1>(newComm)) == changedStepsSent[std::get<2>(newComm)].end()) {
            changedStepsSent[std::get<2>(newComm)][std::get<1>(newComm)] = commCost;
          } else {
            changedStepsSent[std::get<2>(newComm)][std::get<1>(newComm)] += commCost;
          }
          if (changedStepsRec.find(std::get<2>(newComm)) == changedStepsRec.end()) {
            changedStepsRec[std::get<2>(newComm)] = commCost;
          } else {
            changedStepsRec[std::get<2>(newComm)] += commCost;
          }
        }
        for (const auto &changingStep : changedStepsRec) {
          unsigned stepChanged = changingStep.first;

          std::vector<CostType> newSendCost(schedule.GetInstance().NumberOfProcessors());
          CostType newRecCost = recCost_[toProc][stepChanged] - changingStep.second;
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            newSendCost[proc] = sendCost_[proc][stepChanged];
          }
          for (const auto &procAndChange : changedStepsSent[stepChanged]) {
            newSendCost[procAndChange.first] -= procAndChange.second;
          }

          newMax = 0;
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            newMax = std::max(newMax, newSendCost[proc]);
          }
          for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
            if (proc == toProc) {
              newMax = std::max(newMax, newRecCost);
            } else {
              newMax = std::max(newMax, recCost_[proc][stepChanged]);
            }
          }
          costChange += static_cast<int>(newMax) - static_cast<int>(maxComm_[stepChanged]);

          CostType oldSync = (maxComm_[stepChanged] > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;
          CostType newSync = (newMax > 0) ? schedule.GetInstance().GetArchitecture().SynchronisationCosts() : 0;

          costChange += static_cast<int>(newSync) - static_cast<int>(oldSync);
        }

        if (costChange < 0) {
          // REPLICATE STEP IF BENEFICIAL - change schedule and update data structures

          // update assignments and compute data
          for (const VertexIdx node : mustReplicate) {
            AddRecomputeStep(schedule, node, toProc, step);
            auto itr = commStepPerNodeAndReceiver.find(std::make_pair(node, toProc));
            if (itr != commStepPerNodeAndReceiver.end()) {
              commStepPerNodeAndReceiver.erase(itr);
            }
          }
          maxWork_[step] = std::max(maxWork_[step], workCost_[toProc][step]);

          // update comm and its data in step-1
          if (step > 0) {
            for (const KeyTriple &newComm : newCommStepsBefore) {
              AddCommStep(schedule, newComm, step - 1);
            }

            for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
              maxComm_[step - 1] = std::max(maxComm_[step - 1], sendCost_[proc][step - 1]);
              maxComm_[step - 1] = std::max(maxComm_[step - 1], recCost_[proc][step - 1]);
            }
          }

          // update comm and its data in later steps
          for (const KeyTriple &newComm : removedCommStepsAfter) {
            unsigned changingStep = std::get<2>(newComm);
            RemoveCommStep(schedule, KeyTriple(std::get<0>(newComm), std::get<1>(newComm), toProc), changingStep);
          }
          for (const auto &stepAndChange : changedStepsRec) {
            unsigned changingStep = stepAndChange.first;
            maxComm_[changingStep] = 0;
            for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
              maxComm_[changingStep] = std::max(maxComm_[changingStep], sendCost_[proc][changingStep]);
              maxComm_[changingStep] = std::max(maxComm_[changingStep], recCost_[proc][changingStep]);
            }
          }

          improved = true;
        }
      }
    }
  }

  schedule.GetCommunicationSchedule().clear();
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (const KeyTriple &entry : commSteps_[step]) {
      schedule.AddCommunicationScheduleEntry(entry, step);
    }
  }

  return improved;
}

template <typename GraphT>
bool GreedyRecomputer<GraphT>::BatchRemoveSteps(BspScheduleRecomp<GraphT> &schedule)
{
  bool improved = false;
  const GraphT &G = schedule.GetInstance().GetComputationalDag();

  // Initialize required data structures
  RefreshAuxData(schedule);

  std::vector<std::vector<unsigned>> firstComputable(schedule.GetInstance().NumberOfVertices(), std::vector<unsigned>(schedule.GetInstance().NumberOfProcessors(), 0U));
  for (VertexIdx node = 0; node < schedule.GetInstance().NumberOfVertices(); ++node) {
    for (const VertexIdx &pred : G.Parents(node)) {
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        firstComputable[node][proc] = std::max(firstComputable[node][proc], firstPresent_[pred][proc]);
      }
    }
  }

  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {

    bool canReduce = (maxComm_[step] > 0);
    while (canReduce) {

      // find processors where send/rec costs equals the maximum (so we want to remove comm steps)
      canReduce = false;
      std::vector<bool> sendSaturated(schedule.GetInstance().NumberOfProcessors(), false), recSaturated(schedule.GetInstance().NumberOfProcessors(), false);
      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        if (sendCost_[proc][step] == maxComm_[step]) {
          sendSaturated[proc] = true;
        }
        if (recCost_[proc][step] == maxComm_[step]) {
          recSaturated[proc] = true;
        }
      }

      // initialize required variables
      std::map<std::pair<unsigned, unsigned>, CostType> workIncreased;
      std::set<KeyTriple> removedCommSteps, addedComputeSteps;
      std::vector<std::set<KeyTriple> > sendCommSteps(schedule.GetInstance().NumberOfProcessors()),
                                        recCommSteps(schedule.GetInstance().NumberOfProcessors());
      for (const KeyTriple &commStep : commSteps_[step]) {
        sendCommSteps[std::get<1>(commStep)].insert(commStep);
        recCommSteps[std::get<2>(commStep)].insert(commStep);
      }
      bool skipStep = false;
      CostType workIncrease = 0;
      CostType commDecrease = std::numeric_limits<CostType>::max();

      for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
        for (unsigned sendOrRec = 0; sendOrRec < 2; ++sendOrRec) {

          std::set<KeyTriple> *currentCommSteps;
          if (sendOrRec == 0) {
            if (!sendSaturated[proc]) {
              continue;
            }
            currentCommSteps = &sendCommSteps[proc];
          } else {
            if (!recSaturated[proc]) {
              continue;
            }
            currentCommSteps = &recCommSteps[proc];
          }

          KeyTriple bestCommStep;
          unsigned bestStepTarget = std::numeric_limits<unsigned>::max();
          CostType smallestIncrease = std::numeric_limits<CostType>::max();
          for (const KeyTriple &commStep : *currentCommSteps) {
            const VertexIdx node = std::get<0>(commStep);
            const unsigned fromProc = std::get<1>(commStep);
            const unsigned toProc = std::get<2>(commStep);
            if (G.VertexCommWeight(node) == 0) {
              continue;
            }
            if (!schedule.GetInstance().IsCompatible(node, toProc)) {
              continue;
            }

            for (unsigned compStep = firstComputable[node][toProc]; compStep <= *neededOnProc_[node][toProc].begin(); ++compStep) {
              auto itr = workIncreased.find(std::make_pair(toProc, compStep));
              CostType assignedExtra = (itr != workIncreased.end()) ? itr->second : 0;
              CostType increase = 0;
              if (workCost_[toProc][compStep] + assignedExtra + G.VertexWorkWeight(node) > maxWork_[compStep]) {
                increase = workCost_[toProc][compStep] + assignedExtra + G.VertexWorkWeight(node) - maxWork_[compStep];
              }
              if (increase < smallestIncrease) {
                smallestIncrease = increase;
                bestStepTarget = compStep;
                bestCommStep = commStep;
              }
            }
          }

          // save this if this is the cheapest way to move away a comm step
          if (smallestIncrease < std::numeric_limits<CostType>::max()) {
            const VertexIdx node = std::get<0>(bestCommStep);
            const unsigned fromProc = std::get<1>(bestCommStep);
            const unsigned toProc = std::get<2>(bestCommStep);
            addedComputeSteps.emplace(node, toProc, bestStepTarget);
            auto itr = workIncreased.find(std::make_pair(toProc, bestStepTarget));
            if (itr == workIncreased.end()) {
              workIncreased[std::make_pair(toProc, bestStepTarget)] = G.VertexWorkWeight(node);
            } else {
              itr->second += G.VertexWorkWeight(node);
            }

            sendSaturated[fromProc] = false;
            recSaturated[toProc] = false;

            removedCommSteps.insert(bestCommStep);
            workIncrease += smallestIncrease;
            CostType commCost = schedule.GetInstance().GetComputationalDag().VertexCommWeight(node)
                        * schedule.GetInstance().GetArchitecture().CommunicationCosts(fromProc, toProc);
            commDecrease = std::min(commDecrease, commCost);

          } else {
            skipStep = true;
          }
        }
        if (skipStep) {
          // weird edge case if all comm steps have weight 0 (can be removed?)
          break;
        }
      }
      if (skipStep) {
        continue;
      }

      if (maxComm_[step] > 0 && commSteps_[step].size() == removedCommSteps.size()) {
        commDecrease += schedule.GetInstance().GetArchitecture().SynchronisationCosts();
      }
      // execute step if total work cost increase < total comm cost decrease
      if (commDecrease > workIncrease) {
        for (const KeyTriple &newComp : addedComputeSteps) {
          const VertexIdx node = std::get<0>(newComp);
          const unsigned proc = std::get<1>(newComp);
          const unsigned newStep = std::get<2>(newComp);
          AddRecomputeStep(schedule, node, proc, newStep);
          firstPresent_[node][proc] = newStep;
          maxWork_[newStep] = std::max(maxWork_[newStep], workCost_[proc][newStep]);

          for (const VertexIdx &succ : G.Children(node)) {
            firstComputable[succ][proc] = 0U;
            for (const VertexIdx &pred : G.Parents(succ)) {
              firstComputable[succ][proc] = std::max(firstComputable[succ][proc], firstPresent_[pred][proc]);
            }
          }
        }
        for (const KeyTriple &removedComm : removedCommSteps) {
          RemoveCommStep(schedule, removedComm, step);
        }
        maxComm_[step] = 0;
        for (unsigned proc = 0; proc < schedule.GetInstance().NumberOfProcessors(); ++proc) {
          maxComm_[step] = std::max(maxComm_[step], sendCost_[proc][step]);
          maxComm_[step] = std::max(maxComm_[step], recCost_[proc][step]);
        }

        canReduce = true;
        improved = true;
      }
    }
  }

  schedule.GetCommunicationSchedule().clear();
  for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
    for (const KeyTriple &entry : commSteps_[step]) {
      schedule.AddCommunicationScheduleEntry(entry, step);
    }
  }

  return improved;
}

template <typename GraphT>
void GreedyRecomputer<GraphT>::RefreshAuxData(const BspScheduleRecomp<GraphT> &schedule)
{
    const VertexIdx N = schedule.GetInstance().NumberOfVertices();
    const unsigned P = schedule.GetInstance().NumberOfProcessors();
    const unsigned S = schedule.NumberOfSupersteps();
    const GraphT &G = schedule.GetInstance().GetComputationalDag();

    workCost_.clear();
    sendCost_.clear();
    recCost_.clear();

    workCost_.resize(P, std::vector<CostType>(S, 0));
    sendCost_.resize(P, std::vector<CostType>(S, 0)),
    recCost_.resize(P, std::vector<CostType>(S, 0));

    firstPresent_.clear();
    firstPresent_.resize(N, std::vector<unsigned>(P, std::numeric_limits<unsigned>::max()));

    nodesPerProcAndStep_.clear();
    nodesPerProcAndStep_.resize(P, std::vector<std::vector<VertexIdx> >(S));

    neededOnProc_.clear();
    neededOnProc_.resize(N, std::vector<std::multiset<unsigned> >(P, {S}));

    maxWork_.clear();
    maxComm_.clear();
    maxWork_.resize(S, 0);
    maxComm_.resize(S, 0);

    commSteps_.clear();
    commSteps_.resize(S);

    for (VertexIdx node = 0; node < N; ++node) {
      for (const std::pair<unsigned, unsigned> &procAndStep : schedule.Assignments(node)) {
        const unsigned &proc = procAndStep.first;
        const unsigned &step = procAndStep.second;
        nodesPerProcAndStep_[proc][step].push_back(node);
        workCost_[proc][step] += G.VertexWorkWeight(node);
        firstPresent_[node][proc] = std::min(firstPresent_[node][proc], step);
        for (VertexIdx pred : G.Parents(node)) {
          neededOnProc_[pred][proc].insert(step);
        }
      }
    }
    for (const std::pair<KeyTriple, unsigned> item : schedule.GetCommunicationSchedule()) {
      const VertexIdx &node = std::get<0>(item.first);
      const unsigned &fromProc = std::get<1>(item.first);
      const unsigned &toProc = std::get<2>(item.first);
      const unsigned &step = item.second;
      sendCost_[fromProc][step] += G.VertexCommWeight(node)
                                    * schedule.GetInstance().GetArchitecture().CommunicationCosts(fromProc, toProc);
      recCost_[toProc][step] += G.VertexCommWeight(node)
                                    * schedule.GetInstance().GetArchitecture().CommunicationCosts(fromProc, toProc);

      commSteps_[step].emplace(item.first);
      neededOnProc_[node][fromProc].insert(step);
      firstPresent_[node][toProc] = std::min(firstPresent_[node][toProc], step + 1);
    }
    for (unsigned step = 0; step < S; ++step) {
      for (unsigned proc = 0; proc < P; ++proc) {
        maxWork_[step] = std::max(maxWork_[step], workCost_[proc][step]);
        maxComm_[step] = std::max(maxComm_[step], sendCost_[proc][step]);
        maxComm_[step] = std::max(maxComm_[step], recCost_[proc][step]);
      }
    }
}

template <typename GraphT>
void GreedyRecomputer<GraphT>::AddRecomputeStep(BspScheduleRecomp<GraphT> &schedule, const VertexIdx node, const unsigned proc, const unsigned step)
{
  schedule.Assignments(node).emplace_back(proc, step);
  nodesPerProcAndStep_[proc][step].push_back(node);
  workCost_[proc][step] += schedule.GetInstance().GetComputationalDag().VertexWorkWeight(node);
  firstPresent_[node][proc] = std::min(firstPresent_[node][proc], step);
  for (const VertexIdx &pred : schedule.GetInstance().GetComputationalDag().Parents(node)) {
    neededOnProc_[pred][proc].insert(step);
  }
}

template <typename GraphT>
void GreedyRecomputer<GraphT>::AddCommStep(const BspScheduleRecomp<GraphT> &schedule, const KeyTriple &newComm, const unsigned step)
{
  commSteps_[step].insert(newComm);
  CostType commCost = schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(newComm))
                        * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(newComm), std::get<2>(newComm));
  sendCost_[std::get<1>(newComm)][step] += commCost;
  recCost_[std::get<2>(newComm)][step] += commCost;
  neededOnProc_[std::get<0>(newComm)][std::get<1>(newComm)].insert(step);
  unsigned &firstPres = firstPresent_[std::get<0>(newComm)][std::get<2>(newComm)];
  if (firstPres > step + 1 && firstPres <= schedule.NumberOfSupersteps()) {
    auto itr = commSteps_[firstPres - 1].find(newComm);
    if (itr != commSteps_[firstPres - 1].end()) {
      commSteps_[firstPres - 1].erase(itr);
      sendCost_[std::get<1>(newComm)][firstPres - 1] -= commCost;
      recCost_[std::get<2>(newComm)][firstPres - 1] -= commCost;
      neededOnProc_[std::get<0>(newComm)][std::get<1>(newComm)].erase(neededOnProc_[std::get<0>(newComm)][std::get<1>(newComm)].lower_bound(firstPres - 1));
    }
  }
  firstPres = std::min(firstPres, step + 1);
}

template <typename GraphT>
void GreedyRecomputer<GraphT>::RemoveCommStep(const BspScheduleRecomp<GraphT> &schedule, const KeyTriple &removedComm, const unsigned step)
{
  neededOnProc_[std::get<0>(removedComm)][std::get<1>(removedComm)].erase(neededOnProc_[std::get<0>(removedComm)][std::get<1>(removedComm)].lower_bound(step));

  CostType commCost = schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(removedComm))
                        * schedule.GetInstance().GetArchitecture().CommunicationCosts(std::get<1>(removedComm), std::get<2>(removedComm));

  auto itr = commSteps_[step].find(removedComm);
  if (itr != commSteps_[step].end()) {
    commSteps_[step].erase(itr);
  }
  sendCost_[std::get<1>(removedComm)][step] -= commCost;
  recCost_[std::get<2>(removedComm)][step] -= commCost;
}

} // namespace osp