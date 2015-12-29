/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#include "RdcForce.h"
#include "internal/RdcForceImpl.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

RdcForce::RdcForce() : numRestraints(0) {
}

void RdcForce::updateParametersInContext(Context& context) {
    dynamic_cast<RdcForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

int RdcForce::getNumExperiments() const {
    return experiments.size();
}

int RdcForce::getNumRestraints(int experiment) const {
    return experiments[experiment].rdcIndices.size();
}


int RdcForce::getNumTotalRestraints() const {
    return rdcRestraints.size();
}

int RdcForce::addRdcRestraint(int particle1, int particle2, float kappa, float dObs, float tolerance,
        float force_const, float weight) {
    rdcRestraints.push_back(
            RdcRestraintInfo(particle1, particle2, kappa, dObs, tolerance, force_const, weight, numRestraints));
    numRestraints++;
    return numRestraints - 1;
}
void RdcForce::updateRdcRestraint(int index, int particle1, int particle2, float kappa, float dObs,
        float tolerance, float force_const, float weight) {
    int oldGlobal = rdcRestraints[index].globalIndex;
    rdcRestraints[index] =
        RdcRestraintInfo(particle1, particle2, kappa, dObs, tolerance, force_const, weight, oldGlobal);
}

int RdcForce::addExperiment(std::vector<int> rdcIndices) {
    experiments.push_back(ExperimentInfo(rdcIndices));
    return experiments.size() - 1;
}


ForceImpl* RdcForce::createImpl() const {
    return new RdcForceImpl(*this);
}

void RdcForce::getExperimentInfo(int index, std::vector<int>& indices) const {
    indices = experiments[index].rdcIndices;
}

void RdcForce::getRdcRestraintInfo(int index, int& particle1, int& particle2, float& kappa,
        float& dObs, float& tolerance, float& force_const, float& weight, int& globalIndex) const {
    const RdcRestraintInfo& rest = rdcRestraints[index];
    particle1 = rest.atom1;
    particle2 = rest.atom2;
    kappa = rest.kappa;
    dObs = rest.dObs;
    tolerance = rest.tolerance;
    force_const = rest.force_const;
    weight = rest.weight;
    globalIndex = rest.globalIndex;
}

