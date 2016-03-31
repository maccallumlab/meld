/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#include "internal/RdcForceImpl.h"
#include "meldKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Platform.h"
#include <vector>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

RdcForceImpl::RdcForceImpl(const RdcForce& owner) : owner(owner) {
}

RdcForceImpl::~RdcForceImpl() {
}

void RdcForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcRdcForceKernel::Name(), context);
    kernel.getAs<CalcRdcForceKernel>().initialize(context.getSystem(), owner);
}

double RdcForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcRdcForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> RdcForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcRdcForceKernel::Name());
    return names;
}

void RdcForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcRdcForceKernel>().copyParametersToContext(context, owner);
}
