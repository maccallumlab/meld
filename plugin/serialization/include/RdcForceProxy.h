/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#ifndef OPENMM_RDC_FORCE_PROXY_H_
#define OPENMM_RDC_FORCE_PROXY_H_

#include "internal/windowsExportMeld.h"
#include "openmm/serialization/SerializationProxy.h"

namespace OpenMM {

/**
 * This is a proxy for serializing RdcForce objects.
 */

class OPENMM_EXPORT_MELD RdcForceProxy : public SerializationProxy {
public:
    RdcForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

} // namespace OpenMM

#endif /*OPENMM_RDC_FORCE_PROXY_H_*/
