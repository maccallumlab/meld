/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#include "MeldCudaKernelFactory.h"
#include "MeldCudaKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/windowsExport.h"

using namespace MeldPlugin;
using namespace OpenMM;

extern "C" void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        MeldCudaKernelFactory* factory = new MeldCudaKernelFactory();
        platform.registerKernelFactory(CalcMeldForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcRdcForceKernel::Name(), factory);
    }
    catch (...) {
        // Ignore.  The CUDA platform isn't available.
    }
}

extern "C" OPENMM_EXPORT void registerMeldCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* MeldCudaKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
    CudaContext& cu = *data.contexts[0];

    if (name == CalcMeldForceKernel::Name())
        return new CudaCalcMeldForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcRdcForceKernel::Name())
        return new CudaCalcRdcForceKernel(name, platform, cu, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
