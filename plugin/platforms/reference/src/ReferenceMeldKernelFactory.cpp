#include "ReferenceMeldKernelFactory.h"
#include "ReferenceMeldKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace MeldPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceMeldKernelFactory* factory = new ReferenceMeldKernelFactory();
            platform.registerKernelFactory(CalcMeldForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerMeldReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceMeldKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcMeldForceKernel::Name())
        return new ReferenceCalcMeldForceKernel(name, platform, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}