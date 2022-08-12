/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#ifndef OPENMM_MELD_FORCE_IMPL_H_
#define OPENMM_MELD_FORCE_IMPL_H_

#include "MeldForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <vector>
#include <string>

namespace MeldPlugin
{

class MeldForceImpl : public OpenMM::ForceImpl
{
  public:
    MeldForceImpl(const MeldForce &owner);

    ~MeldForceImpl();

    void initialize(OpenMM::ContextImpl &context);

    const MeldForce &getOwner() const
    {
        return owner;
    }

    void updateContextState(OpenMM::ContextImpl &context)
    {
        // This force field doesn't update the state directly.
    }

    double calcForcesAndEnergy(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy, int groups);

    std::map<std::string, double> getDefaultParameters()
    {
        int nAlignments = owner.getNumRDCAlignments();
        std::map<std::string, double> globalParams;

        for (int i = 0; i < nAlignments; i++)
        {
            std::string base = "rdc_" + std::to_string(i);
            globalParams[base + "_s1"] = 0.0;
            globalParams[base + "_s2"] = 0.0;
            globalParams[base + "_s3"] = 0.0;
            globalParams[base + "_s4"] = 0.0;
            globalParams[base + "_s5"] = 0.0;
        }
        return globalParams;
    }

    std::vector<std::string> getKernelNames();

    void updateParametersInContext(OpenMM::ContextImpl &context);

    std::vector<std::pair<int, int>> getBondedParticles() const
    {
        return owner.getBondedParticles();
    }

  private:
    const MeldForce &owner;
    OpenMM::Kernel kernel;
};

} // namespace MeldPlugin

#endif /*OPENMM_MELD_FORCE_IMPL_H_*/
