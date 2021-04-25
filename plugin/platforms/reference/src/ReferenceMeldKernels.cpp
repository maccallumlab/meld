#include "ReferenceMeldKernels.h"
#include "MeldForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

void ReferenceCalcMeldForceKernel::initialize(const System& system, const MeldForce& force) {
}

double ReferenceCalcMeldForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void ReferenceCalcMeldForceKernel::copyParametersToContext(ContextImpl& context, const MeldForce& force) {
}