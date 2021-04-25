/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#ifndef OPENMM_REFERENCEMELDKERNELFACTORY_H_
#define OPENMM_REFERENCEMELDKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

class ReferenceMeldKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_REFERENCEMELDKERNELFACTORY_H_*/