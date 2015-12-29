/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#ifndef MELD_OPENMM_CUDAKERNELFACTORY_H_
#define MELD_OPENMM_CUDAKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

class MeldCudaKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*MELD_OPENMM_CUDAKERNELFACTORY_H_*/
