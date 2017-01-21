/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#include "internal/MeldForceImpl.h"
#include "meldKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Platform.h"
#include <vector>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

MeldForceImpl::MeldForceImpl(const MeldForce& owner) : owner(owner) {
}

MeldForceImpl::~MeldForceImpl() {
}

void MeldForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcMeldForceKernel::Name(), context);
    kernel.getAs<CalcMeldForceKernel>().initialize(context.getSystem(), owner);
}

double MeldForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcMeldForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> MeldForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcMeldForceKernel::Name());
    return names;
}

void MeldForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcMeldForceKernel>().copyParametersToContext(context, owner);
}
