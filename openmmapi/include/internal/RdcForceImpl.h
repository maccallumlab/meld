#ifndef OPENMM_RDC_FORCE_IMPL_H_
#define OPENMM_RDC_FORCE_IMPL_H_

#include "RdcForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <vector>
#include <string>

namespace OpenMM {

class RdcForceImpl : public ForceImpl {
public:
    RdcForceImpl(const RdcForce& owner);

    ~RdcForceImpl();

    void initialize(ContextImpl& context);

    const RdcForce& getOwner() const {
        return owner;
    }

    void updateContextState(ContextImpl& context) {
        // This force field doesn't update the state directly.
    }

    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);

    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }

    std::vector<std::string> getKernelNames();

    void updateParametersInContext(ContextImpl& context);

private:
    const RdcForce& owner;
    Kernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_RdcFORCE_IMPL_H_*/
