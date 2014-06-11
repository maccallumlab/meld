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

namespace OpenMM {


class CalcMeldForceKernel : public KernelImpl {

public:
    static std::string Name() {
        return "CalcMeldForce";
    }

    CalcMeldForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MeldForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MeldForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    virtual void copyParametersToContext(ContextImpl& context, const MeldForce& force) = 0;
};

class CalcRdcForceKernel : public KernelImpl {

public:
    static std::string Name() {
        return "CalcRdcForce";
    }

    CalcRdcForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MeldForce this kernel will be used for
     */
    virtual void initialize(const System& system, const RdcForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    virtual void copyParametersToContext(ContextImpl& context, const RdcForce& force) = 0;
};

} // namespace OpenMM

#endif /*MELD_OPENMM_KERNELS_H*/
