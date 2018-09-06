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
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }

    std::vector<std::string> getKernelNames();

    void updateParametersInContext(OpenMM::ContextImpl &context);

    std::vector<std::pair<int, int>> getBondedParticles() const
    {
        std::vector<std::pair<int, int>> bonds;
        for (int i = 0; i < owner.getNumDistRestraints(); i++)
        {
            int atom1, atom2;
            float r1, r2, r3, r4;
            float forceConstant;
            int globalIndex;
            owner.getDistanceRestraintParams(i, atom1, atom2, r1, r2, r3, r4, forceConstant, globalIndex);
            bonds.push_back(std::make_pair(atom1, atom2));
        }

        for (int i = 0; i < owner.getNumHyperbolicDistRestraints(); i++)
        {
            int atom1, atom2;
            float r1, r2, r3, r4;
            float forceConstant, asymptote;
            int globalIndex;
            owner.getHyperbolicDistanceRestraintParams(i, atom1, atom2, r1, r2, r3, r4, forceConstant, asymptote, globalIndex);
            bonds.push_back(std::make_pair(atom1, atom2));
        }

        for (int i = 0; i < owner.getNumTorsionRestraints(); i++)
        {
            int atom1, atom2, atom3, atom4;
            float phi, deltaPhi, forceConstant;
            owner.getHyperbolicDistanceRestraintParams(i, atom1, atom2, atom3, atom4, phi, deltaPhi, forceConstant, globalIndex);
            bonds.push_back(std::make_pair(atom1, atom2));
            bonds.push_back(std::make_pair(atom2, atom3));
            bonds.push_back(std::make_pair(atom3, atom4));
        }

        for (int i = 0; i < owner.getNumGMMRestraints(); i++)
        {
            int nPairs, nComponents;
            float scale;
            std::vector<double> atomIndices, weights, means, precisionOnDiagonal, precisionOffDiagonal;
            float scaleFactor;
            int globalIndex;
            owner.getGMMRestraintParams(i, nPairs, nComponents, scale, atomIndices, weights, means, precisionOnDiagonal, precisionOffDiagonal, globalIndex);
            for (int j = 0; j < nPairs; j++)
            {
                double atom1 = atomIndices[2 * j];
                double atom2 = atomIndices[(2 * j) + 1];
                bonds.push_back(std::make_pair(atom1, atom2));
            }
        }

        for (int i = 0; i < owner.getNumDistProfileRestraints(); i++)
        {
            int index, atom1, atom2, nBins;
            float rMin, rMax, scaleFactor;
            std::vector<double> a0, a1, a2, a3;
            int globalIndex;
            owner.getDistProfileRestraintParams(index, atom1, atom2, rMin, rMax, nBins, a0, a1, a2, a3, scaleFactor, globalIndex);
            bonds.push_back(std::make_pair(atom1, atom2));
        }

        for (int i = 0; i < owner.getNumTorsionProfileRestraints(); i++)
        {
            int index, atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins;
            std::vector<double> a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
            float scaleFactor;
            int globalIndex;
            owner.getTorsProfileRestraintParams(index, atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, scaleFactor, globalIndex);
            bonds.push_back(std::make_pair(atom1, atom2));
            bonds.push_back(std::make_pair(atom2, atom3));
            bonds.push_back(std::make_pair(atom3, atom4));
            bonds.push_back(std::make_pair(atom5, atom6));
            bonds.push_back(std::make_pair(atom6, atom7));
            bonds.push_back(std::make_pair(atom7, atom8));
        }
    }

  private:
    const MeldForce &owner;
    OpenMM::Kernel kernel;
};

} // namespace MeldPlugin

#endif /*OPENMM_MELD_FORCE_IMPL_H_*/
