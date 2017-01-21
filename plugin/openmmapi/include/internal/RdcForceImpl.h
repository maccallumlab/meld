/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#ifndef OPENMM_RDC_FORCE_IMPL_H_
#define OPENMM_RDC_FORCE_IMPL_H_

#include "RdcForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <vector>
#include <string>

namespace MeldPlugin {

class RdcForceImpl : public OpenMM::ForceImpl {
public:
    RdcForceImpl(const RdcForce& owner);

    ~RdcForceImpl();

    void initialize(OpenMM::ContextImpl& context);

    const RdcForce& getOwner() const {
        return owner;
    }

    void updateContextState(OpenMM::ContextImpl& context) {
        // This force field doesn't update the state directly.
    }

    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);

    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }

    std::vector<std::string> getKernelNames();

    void updateParametersInContext(OpenMM::ContextImpl& context);

private:
    const RdcForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace MeldPlugin

#endif /*OPENMM_RdcFORCE_IMPL_H_*/
