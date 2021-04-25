#ifndef MELD_OPENMM_REFERENCEKERNELS_H_
#define MELD_OPENMM_REFERENCEKERNELS_H_

#include "meldKernels.h"
#include "openmm/Platform.h"
#include <vector>

namespace MeldPlugin {

class ReferenceCalcMeldForceKernel : public CalcMeldForceKernel {
public:
    ReferenceCalcMeldForceKernel(std::string name, const OpenMM::Platform& platform) : CalcMeldForceKernel(name, platform) {
    }

    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the ExampleForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const MeldForce& force);

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the ExampleForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const MeldForce& force);
};
}
#endif /*MELD_OPENMM_REFERENCEKERNELS_H*/
