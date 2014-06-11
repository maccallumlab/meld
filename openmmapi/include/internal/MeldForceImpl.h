#ifndef OPENMM_MELD_FORCE_IMPL_H_
#define OPENMM_MELD_FORCE_IMPL_H_

#include "MeldForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <vector>
#include <string>

namespace OpenMM {

class MeldForceImpl : public ForceImpl {
public:
    MeldForceImpl(const MeldForce& owner);

    ~MeldForceImpl();

    void initialize(ContextImpl& context);

    const MeldForce& getOwner() const {
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
    const MeldForce& owner;
    Kernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_MELD_FORCE_IMPL_H_*/
