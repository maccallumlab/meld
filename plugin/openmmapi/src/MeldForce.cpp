/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#include "MeldForce.h"
#include "internal/MeldForceImpl.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

MeldForce::MeldForce() : n_restraints(0), isDirty(true), n_rdc_alignments(0), rdcScaleFactor(0)
{
}

MeldForce::MeldForce(int numRDCAlignments, float rdcScaleFactor) :
    n_restraints(0), isDirty(true), n_rdc_alignments(numRDCAlignments),
    rdcScaleFactor(rdcScaleFactor)
{
}

std::vector<std::pair<int, int> > MeldForce::getBondedParticles() const
{
    std::vector<std::pair<int, int> > bonds;

    for (int i = 0; i < this->getNumRDCRestraints(); i++) {
        int atom1, atom2, alignment, globalIndex;
        float kappa, obs, tol, quad_cut, force_constant;
        this->getRDCRestraintParameters(i, atom1, atom2, alignment, kappa, obs, tol, quad_cut, force_constant, globalIndex);
        bonds.push_back(std::make_pair(atom1, atom2));
    }

    for (int i = 0; i < this->getNumDistRestraints(); i++)
    {
        int atom1, atom2;
        float r1, r2, r3, r4;
        float forceConstant;
        int globalIndex;
        this->getDistanceRestraintParams(i, atom1, atom2, r1, r2, r3, r4, forceConstant, globalIndex);
        bonds.push_back(std::make_pair(atom1, atom2));
    }

    for (int i = 0; i < this->getNumHyperbolicDistRestraints(); i++)
    {
        int atom1, atom2;
        float r1, r2, r3, r4;
        float forceConstant, asymptote;
        int globalIndex;
        this->getHyperbolicDistanceRestraintParams(i, atom1, atom2, r1, r2, r3, r4, forceConstant, asymptote, globalIndex);
        bonds.push_back(std::make_pair(atom1, atom2));
    }

    for (int i = 0; i < this->getNumTorsionRestraints(); i++)
    {
        int atom1, atom2, atom3, atom4, globalIndex;
        float phi, deltaPhi, forceConstant;
        this->getTorsionRestraintParams(i, atom1, atom2, atom3, atom4, phi,
                                        deltaPhi, forceConstant, globalIndex);
        bonds.push_back(std::make_pair(atom1, atom2));
        bonds.push_back(std::make_pair(atom2, atom3));
        bonds.push_back(std::make_pair(atom3, atom4));
    }

    for (int i = 0; i < this->getNumGMMRestraints(); i++)
    {
        int nPairs, nComponents;
        float scale;
        std::vector<int> atomIndices;
        std::vector<double> weights, means, precisionOnDiagonal, precisionOffDiagonal;
        float scaleFactor;
        int globalIndex;
        this->getGMMRestraintParams(i, nPairs, nComponents, scale, atomIndices, weights, means,
                                    precisionOnDiagonal, precisionOffDiagonal, globalIndex);
        for (int j = 0; j < nPairs; j++)
        {
            double atom1 = atomIndices[2 * j];
            double atom2 = atomIndices[(2 * j) + 1];
            bonds.push_back(std::make_pair(atom1, atom2));
        }
    }

    for (int i = 0; i < this->getNumDistProfileRestraints(); i++)
    {
        int atom1, atom2, nBins;
        float rMin, rMax, scaleFactor;
        std::vector<double> a0, a1, a2, a3;
        int globalIndex;
        this->getDistProfileRestraintParams(i, atom1, atom2, rMin, rMax, nBins, a0, a1, a2, a3, scaleFactor, globalIndex);
        bonds.push_back(std::make_pair(atom1, atom2));
    }

    for (int i = 0; i < this->getNumTorsProfileRestraints(); i++)
    {
        int atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins;
        std::vector<double> a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
        float scaleFactor;
        int globalIndex;
        this->getTorsProfileRestraintParams(i, atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, scaleFactor, globalIndex);
        bonds.push_back(std::make_pair(atom1, atom2));
        bonds.push_back(std::make_pair(atom2, atom3));
        bonds.push_back(std::make_pair(atom3, atom4));
        bonds.push_back(std::make_pair(atom5, atom6));
        bonds.push_back(std::make_pair(atom6, atom7));
        bonds.push_back(std::make_pair(atom7, atom8));
    }
    return bonds;
}

bool MeldForce::containsParticle(int particle) const
{
    std::set<int>::const_iterator loc = meldParticleSet.find(particle);
    if (loc == meldParticleSet.end())
    {
        return false;
    }
    return true;
}

bool MeldForce::usesPeriodicBoundaryConditions() const
{
    return false;
}

void MeldForce::updateMeldParticleSet()
{
    if (isDirty)
    {
        meldParticleSet.clear();

        for (const auto &r : rdcRestraints)
        {
            meldParticleSet.insert(r.particle1);
            meldParticleSet.insert(r.particle2);
        }

        for (const auto &r : distanceRestraints)
        {
            meldParticleSet.insert(r.particle1);
            meldParticleSet.insert(r.particle2);
        }

        for (const auto &r : hyperbolicDistanceRestraints)
        {
            meldParticleSet.insert(r.particle1);
            meldParticleSet.insert(r.particle2);
        }

        for (const auto &r : distProfileRestraints)
        {
            meldParticleSet.insert(r.atom1);
            meldParticleSet.insert(r.atom2);
        }

        for (const auto &r : torsions)
        {
            meldParticleSet.insert(r.atom1);
            meldParticleSet.insert(r.atom2);
            meldParticleSet.insert(r.atom3);
            meldParticleSet.insert(r.atom4);
        }

        for (const auto &r : torsProfileRestraints)
        {
            meldParticleSet.insert(r.atom1);
            meldParticleSet.insert(r.atom2);
            meldParticleSet.insert(r.atom3);
            meldParticleSet.insert(r.atom4);
            meldParticleSet.insert(r.atom5);
            meldParticleSet.insert(r.atom6);
            meldParticleSet.insert(r.atom7);
            meldParticleSet.insert(r.atom8);
        }
        for (const auto &r : gmmRestraints)
        {
            for (const auto &atom : r.atomIndices)
            {
                meldParticleSet.insert(atom);
            }
        }
        isDirty = false;
    }
}

void MeldForce::updateParametersInContext(Context &context)
{
    updateMeldParticleSet();
    dynamic_cast<MeldForceImpl &>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

int MeldForce::getNumRDCAlignments() const
{
    return n_rdc_alignments;
}

int MeldForce::getNumRDCRestraints() const
{
    return rdcRestraints.size();
}

float MeldForce::getRDCScaleFactor() const
{
    return rdcScaleFactor;
}

int MeldForce::getNumDistRestraints() const
{
    return distanceRestraints.size();
}

int MeldForce::getNumHyperbolicDistRestraints() const
{
    return hyperbolicDistanceRestraints.size();
}

int MeldForce::getNumTorsionRestraints() const
{
    return torsions.size();
}

int MeldForce::getNumDistProfileRestraints() const
{
    return distProfileRestraints.size();
}

int MeldForce::getNumDistProfileRestParams() const
{
    int total = 0;
    for (std::vector<DistProfileRestraintInfo>::const_iterator iter = distProfileRestraints.begin();
         iter != distProfileRestraints.end(); ++iter)
    {
        total += iter->nBins;
    }
    return total;
}

int MeldForce::getNumTorsProfileRestraints() const
{
    return torsProfileRestraints.size();
}

int MeldForce::getNumTorsProfileRestParams() const
{
    int total = 0;
    for (std::vector<TorsProfileRestraintInfo>::const_iterator iter = torsProfileRestraints.begin();
         iter != torsProfileRestraints.end(); ++iter)
    {
        total += iter->nBins * iter->nBins;
    }
    return total;
}

int MeldForce::getNumGMMRestraints() const
{
    return gmmRestraints.size();
}

int MeldForce::getNumGridPotentials() const {
    return gridPotentials.size();
}

int MeldForce::getNumGridPotentialRestraints() const {
    return gridPotentialRestraints.size();
}

int MeldForce::getNumTotalRestraints() const {
    return rdcRestraints.size() + distanceRestraints.size() + hyperbolicDistanceRestraints.size() + torsions.size() +
           distProfileRestraints.size() + torsProfileRestraints.size() + gmmRestraints.size() +
           gridPotentialRestraints.size();
}

int MeldForce::getNumGroups() const
{
    return groups.size();
}

int MeldForce::getNumCollections() const
{
    return collections.size();
}

int MeldForce::addRDCRestraint(int particle1, int particle2, int alignment,
                               float kappa, float obs, float tol, float quad_cut, float force_constant)
{
    meldParticleSet.insert(particle1);
    meldParticleSet.insert(particle2);
    rdcRestraints.push_back(
        RDCRestraintInfo(particle1, particle2, alignment, kappa, obs, tol, quad_cut, force_constant, n_restraints)
    );
    n_restraints++;
    return n_restraints - 1;
}


void MeldForce::modifyRDCRestraint(int index, int particle1, int particle2, int alignment,
                                   float kappa, float obs, float tol, float quad_cut, float force_constant)
{
    int oldGlobal = rdcRestraints[index].global_index;

    if (rdcRestraints[index].particle1 != particle1)
        isDirty = true;
    if (rdcRestraints[index].particle2 != particle2)
        isDirty = true;

    rdcRestraints.at(index) = 
        RDCRestraintInfo(particle1, particle2, alignment, kappa, obs, tol, quad_cut, force_constant, oldGlobal);    
}

int MeldForce::addDistanceRestraint(int particle1, int particle2, float r1, float r2,
                                    float r3, float r4, float force_constant)
{
    meldParticleSet.insert(particle1);
    meldParticleSet.insert(particle2);
    distanceRestraints.push_back(
        DistanceRestraintInfo(particle1, particle2, r1, r2, r3, r4, force_constant, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyDistanceRestraint(int index, int particle1, int particle2, float r1, float r2,
                                        float r3, float r4, float force_constant)
{
    int oldGlobal = distanceRestraints[index].global_index;

    if (distanceRestraints[index].particle1 != particle1)
        isDirty = true;
    if (distanceRestraints[index].particle2 != particle2)
        isDirty = true;

    distanceRestraints.at(index) =
        DistanceRestraintInfo(particle1, particle2, r1, r2, r3, r4, force_constant, oldGlobal);
}

int MeldForce::addGMMRestraint(int nPairs, int nComponents, float scale,
                               std::vector<int> atomIndices,
                               std::vector<double> weights,
                               std::vector<double> means,
                               std::vector<double> precisionOnDiagonal,
                               std::vector<double> precisionOffDiagonal)
{
    // sanity checks
    if (nPairs > 32)
    {
        throw OpenMMException("nPairs must be <= 32.");
    }
    if (nComponents > 32)
    {
        throw OpenMMException("nComponents must be <= 32.");
    }
    if (atomIndices.size() != 2 * nPairs)
    {
        throw OpenMMException("atomIndices.size() must be 2*nPairs.");
    }
    if (weights.size() != nComponents)
    {
        throw OpenMMException("weights.size() must be nComponents.");
    }
    if (means.size() != nComponents * nPairs)
    {
        throw OpenMMException("means.size() must be nComponents*nPairs.");
    }
    if (precisionOnDiagonal.size() != nComponents * nPairs)
    {
        throw OpenMMException("precisionOnDiagonal.size() must be nComponents*nPairs.");
    }
    if (precisionOffDiagonal.size() != nComponents * nPairs * (nPairs - 1) / 2)
    {
        throw OpenMMException("precisionOffDiagonal.size() must be nComponents*nPairs*(nPairs-1)/2.");
    }
    float weightSum = 0.0;
    for (auto &w : weights)
    {
        weightSum += w;
    }
    if (fabs(weightSum - 1.0) > 1e-5)
    {
        throw OpenMMException("weights must sum to 1.0");
    }

    // update the list of particles involved in this meld force
    for (const auto &atom : atomIndices)
    {
        meldParticleSet.insert(atom);
    }

    // store the parameters
    gmmRestraints.push_back(GMMRestraintInfo(nPairs, nComponents, n_restraints, scale,
                                             atomIndices, weights, means,
                                             precisionOnDiagonal, precisionOffDiagonal));
    n_restraints++;

    return n_restraints - 1;
}

void MeldForce::modifyGMMRestraint(int index, int nPairs, int nComponents, float scale,
                                   std::vector<int> atomIndices,
                                   std::vector<double> weights,
                                   std::vector<double> means,
                                   std::vector<double> precisionOnDiagonal,
                                   std::vector<double> precisionOffDiagonal)
{
    int oldGlobal = gmmRestraints[index].globalIndex;

    if (gmmRestraints[index].atomIndices != atomIndices)
    {
        isDirty = true;
    }

    // sanity checks
    if (nPairs > 32)
    {
        throw OpenMMException("nPairs must be <= 32.");
    }
    if (nComponents > 32)
    {
        throw OpenMMException("nComponents must be <= 32.");
    }
    if (atomIndices.size() != 2 * nPairs)
    {
        throw OpenMMException("atomIndices.size() must be 2*nPairs.");
    }
    if (weights.size() != nComponents)
    {
        throw OpenMMException("weights.size() must be nComponents.");
    }
    if (means.size() != nComponents * nPairs)
    {
        throw OpenMMException("means.size() must be nComponents*nPairs.");
    }
    if (precisionOnDiagonal.size() != nComponents * nPairs)
    {
        throw OpenMMException("precisionOnDiagonal.size() must be nComponents*nPairs.");
    }
    if (precisionOffDiagonal.size() != nComponents * nPairs * (nPairs - 1) / 2)
    {
        throw OpenMMException("precisionOffDiagonal.size() must be nComponents*nPairs*(nPairs-1)/2.");
    }
    float weightSum = 0.0;
    for (auto &w : weights)
    {
        weightSum += w;
    }
    if (fabs(weightSum - 1.0) > 1e-5)
    {
        throw OpenMMException("weights must sum to 1.0");
    }
    if (gmmRestraints[index].nPairs != nPairs)
    {
        throw OpenMMException("Cannot change nPairs after a gmm restraint is created.");
    }
    if (gmmRestraints[index].nComponents != nComponents)
    {
        throw OpenMMException("Cannot change nComponents after a gmm restraint is created.");
    }

    gmmRestraints.at(index) = GMMRestraintInfo(nPairs, nComponents, oldGlobal, scale,
                                               atomIndices, weights, means,
                                               precisionOnDiagonal, precisionOffDiagonal);
}

int MeldForce::addHyperbolicDistanceRestraint(int particle1, int particle2, float r1, float r2,
                                              float r3, float r4, float force_constant, float asymptote)
{
    meldParticleSet.insert(particle1);
    meldParticleSet.insert(particle2);
    hyperbolicDistanceRestraints.push_back(
        HyperbolicDistanceRestraintInfo(particle1, particle2, r1, r2, r3, r4, force_constant, asymptote, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyHyperbolicDistanceRestraint(int index, int particle1, int particle2, float r1, float r2,
                                                  float r3, float r4, float force_constant, float asymptote)
{
    int oldGlobal = hyperbolicDistanceRestraints[index].global_index;

    if (hyperbolicDistanceRestraints[index].particle1 != particle1)
        isDirty = true;
    if (hyperbolicDistanceRestraints[index].particle2 != particle2)
        isDirty = true;

    hyperbolicDistanceRestraints.at(index) =
        HyperbolicDistanceRestraintInfo(particle1, particle2, r1, r2, r3, r4, force_constant, asymptote, oldGlobal);
}

int MeldForce::addTorsionRestraint(int atom1, int atom2, int atom3, int atom4,
                                   float phi, float deltaPhi, float forceConstant)
{
    meldParticleSet.insert(atom1);
    meldParticleSet.insert(atom2);
    meldParticleSet.insert(atom3);
    meldParticleSet.insert(atom4);
    torsions.push_back(
        TorsionRestraintInfo(atom1, atom2, atom3, atom4, phi, deltaPhi, forceConstant, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyTorsionRestraint(int index, int atom1, int atom2, int atom3, int atom4,
                                       float phi, float deltaPhi, float forceConstant)
{
    int oldIndex = torsions[index].globalIndex;

    if (torsions[index].atom1 != atom1)
        isDirty = true;
    if (torsions[index].atom2 != atom2)
        isDirty = true;
    if (torsions[index].atom3 != atom3)
        isDirty = true;
    if (torsions[index].atom4 != atom4)
        isDirty = true;

    torsions.at(index) =
        TorsionRestraintInfo(atom1, atom2, atom3, atom4, phi, deltaPhi, forceConstant, oldIndex);
}

int MeldForce::addDistProfileRestraint(int atom1, int atom2, float rMin, float rMax,
                                       int nBins, std::vector<double> a0, std::vector<double> a1, std::vector<double> a2,
                                       std::vector<double> a3, float scaleFactor)
{
    meldParticleSet.insert(atom1);
    meldParticleSet.insert(atom2);
    distProfileRestraints.push_back(
        DistProfileRestraintInfo(atom1, atom2, rMin, rMax, nBins, a0, a1, a2, a3, scaleFactor, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyDistProfileRestraint(int index, int atom1, int atom2, float rMin, float rMax,
                                           int nBins, std::vector<double> a0, std::vector<double> a1, std::vector<double> a2,
                                           std::vector<double> a3, float scaleFactor)
{
    int oldIndex = distProfileRestraints[index].globalIndex;

    if (distProfileRestraints[index].atom1 != atom1)
        isDirty = true;
    if (distProfileRestraints[index].atom2 != atom2)
        isDirty = true;

    distProfileRestraints.at(index) =
        DistProfileRestraintInfo(atom1, atom2, rMin, rMax, nBins, a0, a1, a2, a3, scaleFactor, oldIndex);
}

int MeldForce::addTorsProfileRestraint(int atom1, int atom2, int atom3, int atom4,
                                       int atom5, int atom6, int atom7, int atom8, int nBins,
                                       std::vector<double> a0, std::vector<double> a1, std::vector<double> a2,
                                       std::vector<double> a3, std::vector<double> a4, std::vector<double> a5,
                                       std::vector<double> a6, std::vector<double> a7, std::vector<double> a8,
                                       std::vector<double> a9, std::vector<double> a10, std::vector<double> a11,
                                       std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
                                       std::vector<double> a15, float scaleFactor)
{
    meldParticleSet.insert(atom1);
    meldParticleSet.insert(atom2);
    meldParticleSet.insert(atom3);
    meldParticleSet.insert(atom4);
    meldParticleSet.insert(atom5);
    meldParticleSet.insert(atom6);
    meldParticleSet.insert(atom7);
    meldParticleSet.insert(atom8);
    torsProfileRestraints.push_back(
        TorsProfileRestraintInfo(atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, a0, a1, a2, a3,
                                 a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, scaleFactor, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyTorsProfileRestraint(int index, int atom1, int atom2, int atom3, int atom4,
                                           int atom5, int atom6, int atom7, int atom8, int nBins,
                                           std::vector<double> a0, std::vector<double> a1, std::vector<double> a2,
                                           std::vector<double> a3, std::vector<double> a4, std::vector<double> a5,
                                           std::vector<double> a6, std::vector<double> a7, std::vector<double> a8,
                                           std::vector<double> a9, std::vector<double> a10, std::vector<double> a11,
                                           std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
                                           std::vector<double> a15, float scaleFactor)
{
    int oldIndex = torsProfileRestraints[index].globalIndex;

    if (torsProfileRestraints[index].atom1 != atom1)
        isDirty = true;
    if (torsProfileRestraints[index].atom2 != atom2)
        isDirty = true;
    if (torsProfileRestraints[index].atom3 != atom3)
        isDirty = true;
    if (torsProfileRestraints[index].atom4 != atom4)
        isDirty = true;
    if (torsProfileRestraints[index].atom5 != atom5)
        isDirty = true;
    if (torsProfileRestraints[index].atom6 != atom6)
        isDirty = true;
    if (torsProfileRestraints[index].atom7 != atom7)
        isDirty = true;
    if (torsProfileRestraints[index].atom8 != atom8)
        isDirty = true;

    torsProfileRestraints.at(index) =
        TorsProfileRestraintInfo(atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, a0, a1,
                                 a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                                 scaleFactor, oldIndex);
}

int MeldForce::addGroup(std::vector<int> restraint_indices, int n_active)
{
    if (n_active < 0)
    {
        n_active = restraint_indices.size();
    }
    groups.push_back(GroupInfo(restraint_indices, n_active));
    return groups.size() - 1;
}

void MeldForce::modifyGroupNumActive(int index, int n_active)
{
    auto oldIndices = groups[index].restraint_indices;
    groups.at(index) = GroupInfo(oldIndices, n_active);
}

int MeldForce::addCollection(std::vector<int> group_indices, int n_active)
{
    if (n_active < 0)
    {
        n_active = group_indices.size();
    }
    collections.push_back(CollectionInfo(group_indices, n_active));
    return collections.size() - 1;
}

void MeldForce::modifyCollectionNumActive(int index, int n_active)
{
    auto oldIndices = collections[index].group_indices;
    collections.at(index) = CollectionInfo(oldIndices, n_active);
}

void MeldForce::addGridPotential(
    std::vector<double> potential,
    float originx,
    float originy,
    float originz,
    float gridx,
    float gridy,
    float gridz,
    int nx,
    int ny,
    int nz,
    int density_index) {
    gridPotentials.push_back(GridPotentialInfo(potential,originx,originy,originz,gridx,gridy,gridz,nx,ny,nz,density_index));
}


void MeldForce::modifyGridPotential(
    int index, 
    std::vector<double> potential,
    float originx,
    float originy,
    float originz,
    float gridx,
    float gridy,
    float gridz,
    int nx,
    int ny,
    int nz) {
    int oldIndices = gridPotentials[index].density_index;
    gridPotentials.at(index) = GridPotentialInfo(potential,originx,originy,originz,gridx,gridy,gridz,nx,ny,nz,oldIndices);

}

int MeldForce::addGridPotentialRestraint(std::vector<int> particle, std::vector<double> mu,
        std::vector<double> gridpos_x, std::vector<double> gridpos_y, std::vector<double> gridpos_z) {
    for (int i = 0; i < particle.size();i++){
        meldParticleSet.insert(particle[i]);
    }
    
    gridPotentialRestraints.push_back(
        GridPotentialRestraintInfo(particle, mu, gridpos_x, gridpos_y, gridpos_z, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyGridPotentialRestraint(int index, std::vector<int> particle, std::vector<double> mu,       
        std::vector<double> gridpos_x, std::vector<double> gridpos_y, std::vector<double> gridpos_z) {
    int oldGlobal = gridPotentialRestraints[index].globalIndex; 
    gridPotentialRestraints.at(index) = GridPotentialRestraintInfo(particle, mu, gridpos_x, gridpos_y, gridpos_z,oldGlobal);
}

ForceImpl* MeldForce::createImpl() const {
    return new MeldForceImpl(*this);
}

void MeldForce::getGridPotentialParams(int index, std::vector<double>& potential,float& originx,float& originy,float& originz,
            float& gridx,float& gridy,float& gridz, int& nx, int& ny, int& nz) const {
    const GridPotentialInfo& density=gridPotentials[index];
    potential= density.potential;
    originx = density.originx;
    originy = density.originy;
    originz = density.originz;
    gridx = density.gridx;
    gridy = density.gridy;
    gridz = density.gridz;
    nx = density.nx;
    ny = density.ny;
    nz = density.nz;
}

void MeldForce::getGridPotentialRestraintParams(int index, std::vector<int>& atom, 
                            std::vector<double>& mu, 
                            std::vector<double>& gridpos_x,
                            std::vector<double>& gridpos_y,
                            std::vector<double>& gridpos_z,
                            int& globalIndex) const 
{
    const GridPotentialRestraintInfo& rest = gridPotentialRestraints[index];
    atom = rest.particle;
    mu = rest.mu;
    gridpos_x = rest.gridpos_x;
    gridpos_y = rest.gridpos_y;
    gridpos_z = rest.gridpos_z;
    globalIndex = rest.globalIndex;
}

void MeldForce::getRDCRestraintParameters(int index, int& particle1, int& particle2, int& alignment,
                                          float& kappa, float& obs, float& tol, float& quad_cut, float& force_constant,
                                          int& globalIndex) const
{
    const RDCRestraintInfo &rest = rdcRestraints[index];
    particle1 = rest.particle1;
    particle2 = rest.particle2;
    alignment = rest.alignment;
    kappa = rest.kappa;
    obs = rest.obs;
    tol = rest.tol;
    quad_cut = rest.quad_cut;
    force_constant = rest.force_constant;
    globalIndex = rest.global_index;
}

void MeldForce::getDistanceRestraintParams(int index, int &atom1, int &atom2, float &r1, float &r2, float &r3,
                                           float &r4, float &forceConstant, int &globalIndex) const
{
    const DistanceRestraintInfo &rest = distanceRestraints[index];
    atom1 = rest.particle1;
    atom2 = rest.particle2;
    r1 = rest.r1;
    r2 = rest.r2;
    r3 = rest.r3;
    r4 = rest.r4;
    forceConstant = rest.force_constant;
    globalIndex = rest.global_index;
}

void MeldForce::getHyperbolicDistanceRestraintParams(int index, int &atom1, int &atom2, float &r1, float &r2, float &r3,
                                                     float &r4, float &forceConstant, float &asymptote, int &globalIndex) const
{
    const HyperbolicDistanceRestraintInfo &rest = hyperbolicDistanceRestraints[index];
    atom1 = rest.particle1;
    atom2 = rest.particle2;
    r1 = rest.r1;
    r2 = rest.r2;
    r3 = rest.r3;
    r4 = rest.r4;
    forceConstant = rest.force_constant;
    asymptote = rest.asymptote;
    globalIndex = rest.global_index;
}

void MeldForce::getTorsionRestraintParams(int index, int &atom1, int &atom2, int &atom3, int &atom4, float &phi,
                                          float &deltaPhi, float &forceConstant, int &globalIndex) const
{
    const TorsionRestraintInfo &rest = torsions[index];
    atom1 = rest.atom1;
    atom2 = rest.atom2;
    atom3 = rest.atom3;
    atom4 = rest.atom4;
    phi = rest.phi;
    deltaPhi = rest.deltaPhi;
    forceConstant = rest.forceConstant;
    globalIndex = rest.globalIndex;
}

void MeldForce::getDistProfileRestraintParams(int index, int &atom1, int &atom2, float &rMin, float &rMax,
                                              int &nBins, std::vector<double> &a0, std::vector<double> &a1, std::vector<double> &a2,
                                              std::vector<double> &a3, float &scaleFactor, int &globalIndex) const
{

    const DistProfileRestraintInfo &rest = distProfileRestraints[index];

    atom1 = rest.atom1;
    atom2 = rest.atom2;
    rMin = rest.rMin;
    rMax = rest.rMax;
    nBins = rest.nBins;
    a0 = rest.a0;
    a1 = rest.a1;
    a2 = rest.a2;
    a3 = rest.a3;
    scaleFactor = rest.scaleFactor;
    globalIndex = rest.globalIndex;
}

void MeldForce::getTorsProfileRestraintParams(int index, int &atom1, int &atom2, int &atom3, int &atom4,
                                              int &atom5, int &atom6, int &atom7, int &atom8, int &nBins,
                                              std::vector<double> &a0, std::vector<double> &a1, std::vector<double> &a2,
                                              std::vector<double> &a3, std::vector<double> &a4, std::vector<double> &a5,
                                              std::vector<double> &a6, std::vector<double> &a7, std::vector<double> &a8,
                                              std::vector<double> &a9, std::vector<double> &a10, std::vector<double> &a11,
                                              std::vector<double> &a12, std::vector<double> &a13, std::vector<double> &a14,
                                              std::vector<double> &a15, float &scaleFactor, int &globalIndex) const
{
    const TorsProfileRestraintInfo &rest = torsProfileRestraints[index];

    atom1 = rest.atom1;
    atom2 = rest.atom2;
    atom3 = rest.atom3;
    atom4 = rest.atom4;
    atom5 = rest.atom5;
    atom6 = rest.atom6;
    atom7 = rest.atom7;
    atom8 = rest.atom8;
    nBins = rest.nBins;
    a0 = rest.a0;
    a1 = rest.a1;
    a2 = rest.a2;
    a3 = rest.a3;
    a4 = rest.a4;
    a5 = rest.a5;
    a6 = rest.a6;
    a7 = rest.a7;
    a8 = rest.a8;
    a9 = rest.a9;
    a10 = rest.a10;
    a11 = rest.a11;
    a12 = rest.a12;
    a13 = rest.a13;
    a14 = rest.a14;
    a15 = rest.a15;
    scaleFactor = rest.scaleFactor;
    globalIndex = rest.globalIndex;
}

void MeldForce::getGMMRestraintParams(int index, int &nPairs, int &nComponents, float &scale,
                                      std::vector<int> &atomIndices,
                                      std::vector<double> &weights,
                                      std::vector<double> &means,
                                      std::vector<double> &precisionOnDiagonal,
                                      std::vector<double> &precisionOffDiagonal,
                                      int &globalIndex) const
{
    const GMMRestraintInfo &rest = gmmRestraints[index];
    nPairs = rest.nPairs;
    nComponents = rest.nComponents;
    scale = rest.scale;
    atomIndices = rest.atomIndices;
    weights = rest.weights;
    means = rest.means;
    precisionOnDiagonal = rest.precisionOnDiagonal;
    precisionOffDiagonal = rest.precisionOffDiagonal;
    globalIndex = rest.globalIndex;
}

void MeldForce::getGroupParams(int index, std::vector<int> &indices, int &numActive) const
{
    const GroupInfo &grp = groups[index];
    indices = grp.restraint_indices;
    numActive = grp.n_active;
}

void MeldForce::getCollectionParams(int index, std::vector<int> &indices, int &numActive) const
{
    const CollectionInfo &col = collections[index];
    indices = col.group_indices;
    numActive = col.n_active;
}
