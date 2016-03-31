/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#ifndef MELD_OPENMM_KERNELS_H_
#define MELD_OPENMM_KERNELS_H_

#include "MeldForce.h"
#include "RdcForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/System.h"
#include "openmm/Platform.h"

#include <set>
#include <string>
#include <vector>

namespace MeldPlugin {


class CalcMeldForceKernel : public OpenMM::KernelImpl {

public:
    static std::string Name() {
        return "CalcMeldForce";
    }

    CalcMeldForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MeldForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const MeldForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const MeldForce& force) = 0;
};

class CalcRdcForceKernel : public OpenMM::KernelImpl {

public:
    static std::string Name() {
        return "CalcRdcForce";
    }

    CalcRdcForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MeldForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const RdcForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const RdcForce& force) = 0;
};

} // namespace MeldPlugin

#endif /*MELD_OPENMM_KERNELS_H*/
