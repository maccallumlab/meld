/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#include "MeldForce.h"
#include "internal/MeldForceImpl.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

MeldForce::MeldForce() : n_restraints(0) {
}

bool MeldForce::containsParticle(int particle) const {
    std::set<int>::const_iterator loc=meldParticleSet.find(particle);
    if(loc==meldParticleSet.end()) {
      return false;
    }
    return true;
}

void MeldForce::updateMeldParticleSet() {
    meldParticleSet.clear();

    for(std::vector<DistanceRestraintInfo>::iterator it=distanceRestraints.begin(); it!=distanceRestraints.end(); ++it) {
        meldParticleSet.insert(it->particle1);
        meldParticleSet.insert(it->particle2);
    }

    for(std::vector<HyperbolicDistanceRestraintInfo>::iterator it=hyperbolicDistanceRestraints.begin(); it!=hyperbolicDistanceRestraints.end(); ++it) {
        meldParticleSet.insert(it->particle1);
        meldParticleSet.insert(it->particle2);
    }

    for(std::vector<DistProfileRestraintInfo>::iterator it=distProfileRestraints.begin(); it!=distProfileRestraints.end(); ++it) {
        meldParticleSet.insert(it->atom1);
        meldParticleSet.insert(it->atom2);
    }

    for(std::vector<TorsionRestraintInfo>::iterator it=torsions.begin(); it!=torsions.end(); ++it) {
        meldParticleSet.insert(it->atom1);
        meldParticleSet.insert(it->atom2);
        meldParticleSet.insert(it->atom3);
        meldParticleSet.insert(it->atom4);
    }

    for(std::vector<TorsProfileRestraintInfo>::iterator it=torsProfileRestraints.begin(); it!=torsProfileRestraints.end(); ++it) {
        meldParticleSet.insert(it->atom1);
        meldParticleSet.insert(it->atom2);
        meldParticleSet.insert(it->atom3);
        meldParticleSet.insert(it->atom4);
        meldParticleSet.insert(it->atom5);
        meldParticleSet.insert(it->atom6);
        meldParticleSet.insert(it->atom7);
        meldParticleSet.insert(it->atom8);
    }
}

void MeldForce::updateParametersInContext(Context& context) {
    dynamic_cast<MeldForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

int MeldForce::getNumDistRestraints() const {
    return distanceRestraints.size();
}

int MeldForce::getNumHyperbolicDistRestraints() const {
    return hyperbolicDistanceRestraints.size();
}

int MeldForce::getNumTorsionRestraints() const {
    return torsions.size();
}

int MeldForce::getNumDistProfileRestraints() const {
    return distProfileRestraints.size();
}

int MeldForce::getNumDistProfileRestParams() const {
    int total = 0;
    for(std::vector<DistProfileRestraintInfo>::const_iterator iter=distProfileRestraints.begin();
            iter != distProfileRestraints.end(); ++iter) {
        total += iter->nBins;
    }
    return total;
}

int MeldForce::getNumTorsProfileRestraints() const {
    return torsProfileRestraints.size();
}

int MeldForce::getNumTorsProfileRestParams() const {
    int total = 0;
    for(std::vector<TorsProfileRestraintInfo>::const_iterator iter=torsProfileRestraints.begin();
            iter != torsProfileRestraints.end(); ++iter) {
        total += iter->nBins * iter->nBins;
    }
    return total;
}


int MeldForce::getNumTotalRestraints() const {
    return distanceRestraints.size() + hyperbolicDistanceRestraints.size() + torsions.size() +
        distProfileRestraints.size() + torsProfileRestraints.size();
}


int MeldForce::getNumGroups() const {
    return groups.size();
}


int MeldForce::getNumCollections() const {
    return collections.size();
}


int MeldForce::addDistanceRestraint(int particle1, int particle2, float r1, float r2,
                                    float r3, float r4, float force_constant) {
    meldParticleSet.insert(particle1);
    meldParticleSet.insert(particle2);
    distanceRestraints.push_back(
            DistanceRestraintInfo(particle1, particle2, r1, r2, r3, r4, force_constant, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyDistanceRestraint(int index, int particle1, int particle2, float r1, float r2,
                                        float r3, float r4, float force_constant) {
    int oldGlobal = distanceRestraints[index].global_index;

    bool updateParticles = false;
    if(distanceRestraints[index].particle1!=particle1)
        updateParticles=true;
    if(distanceRestraints[index].particle2!=particle2)
        updateParticles=true;

    distanceRestraints[index] =
            DistanceRestraintInfo(particle1, particle2, r1, r2, r3, r4, force_constant, oldGlobal);

    if(updateParticles)
        updateMeldParticleSet();
}

int MeldForce::addHyperbolicDistanceRestraint(int particle1, int particle2, float r1, float r2,
                                    float r3, float r4, float force_constant, float asymptote) {
    meldParticleSet.insert(particle1);
    meldParticleSet.insert(particle2);
    hyperbolicDistanceRestraints.push_back(
            HyperbolicDistanceRestraintInfo(particle1, particle2, r1, r2, r3, r4, force_constant, asymptote, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyHyperbolicDistanceRestraint(int index, int particle1, int particle2, float r1, float r2,
                                        float r3, float r4, float force_constant, float asymptote) {
    int oldGlobal = hyperbolicDistanceRestraints[index].global_index;

    bool updateParticles=false;
    if(hyperbolicDistanceRestraints[index].particle1 != particle1)
        updateParticles=true;
    if(hyperbolicDistanceRestraints[index].particle2 != particle2)
        updateParticles=true;

    hyperbolicDistanceRestraints[index] =
            HyperbolicDistanceRestraintInfo(particle1, particle2, r1, r2, r3, r4, force_constant, asymptote, oldGlobal);

    if(updateParticles)
        updateMeldParticleSet();
}

int MeldForce::addTorsionRestraint(int atom1, int atom2, int atom3, int atom4,
                                   float phi, float deltaPhi, float forceConstant) {
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
                                       float phi, float deltaPhi, float forceConstant) {
    int oldIndex = torsions[index].globalIndex;

    bool updateParticles=false;
    if(torsions[index].atom1 != atom1)
        updateParticles=true;
    if(torsions[index].atom2 != atom2)
        updateParticles=true;
    if(torsions[index].atom3 != atom3)
        updateParticles=true;
    if(torsions[index].atom4 != atom4)
        updateParticles=true;

    torsions[index] =
            TorsionRestraintInfo(atom1, atom2, atom3, atom4, phi, deltaPhi, forceConstant, oldIndex);

    if(updateParticles)
        updateMeldParticleSet();
}

int MeldForce::addDistProfileRestraint(int atom1, int atom2, float rMin, float rMax,
        int nBins, std::vector<double> a0, std::vector<double> a1, std::vector<double> a2,
        std::vector<double> a3, float scaleFactor) {
    meldParticleSet.insert(atom1);
    meldParticleSet.insert(atom2);
    distProfileRestraints.push_back(
            DistProfileRestraintInfo(atom1, atom2, rMin, rMax, nBins, a0, a1, a2, a3, scaleFactor, n_restraints));
    n_restraints++;
    return n_restraints - 1;
}

void MeldForce::modifyDistProfileRestraint(int index, int atom1, int atom2, float rMin, float rMax,
        int nBins, std::vector<double> a0, std::vector<double> a1, std::vector<double> a2,
        std::vector<double> a3, float scaleFactor) {
    int oldIndex = distProfileRestraints[index].globalIndex;

    bool updateParticles=false;
    if(distProfileRestraints[index].atom1 != atom1)
        updateParticles=true;
    if(distProfileRestraints[index].atom2 != atom2)
        updateParticles=true;

    distProfileRestraints[index] =
        DistProfileRestraintInfo(atom1, atom2, rMin, rMax, nBins, a0, a1, a2, a3, scaleFactor, oldIndex);

    if(updateParticles)
        updateMeldParticleSet();
}

int MeldForce::addTorsProfileRestraint(int atom1, int atom2, int atom3, int atom4,
        int atom5, int atom6, int atom7, int atom8, int nBins,
        std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
        std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
        std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
        std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
        std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
        std::vector<double> a15, float scaleFactor) {
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
        std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
        std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
        std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
        std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
        std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
        std::vector<double> a15, float scaleFactor) {
    int oldIndex = torsProfileRestraints[index].globalIndex;

    bool updateParticles=false;
    if(torsProfileRestraints[index].atom1 != atom1)
        updateParticles=true;
    if(torsProfileRestraints[index].atom2 != atom2)
        updateParticles=true;
    if(torsProfileRestraints[index].atom3 != atom3)
        updateParticles=true;
    if(torsProfileRestraints[index].atom4 != atom4)
        updateParticles=true;
    if(torsProfileRestraints[index].atom5 != atom5)
        updateParticles=true;
    if(torsProfileRestraints[index].atom6 != atom6)
        updateParticles=true;
    if(torsProfileRestraints[index].atom7 != atom7)
        updateParticles=true;
    if(torsProfileRestraints[index].atom8 != atom8)
        updateParticles=true;

    torsProfileRestraints[index] =
        TorsProfileRestraintInfo(atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, a0, a1,
                a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                scaleFactor, oldIndex);

    if(updateParticles)
        updateMeldParticleSet();
}


int MeldForce::addGroup(std::vector<int> restraint_indices, int n_active) {
    if (n_active < 0) {
        n_active = restraint_indices.size();
    }
    groups.push_back(GroupInfo(restraint_indices, n_active));
    return groups.size() - 1;
}


int MeldForce::addCollection(std::vector<int> group_indices, int n_active) {
    if (n_active < 0) {
        n_active = group_indices.size();
    }
    collections.push_back(CollectionInfo(group_indices, n_active));
    return collections.size() - 1;
}


ForceImpl* MeldForce::createImpl() const {
    return new MeldForceImpl(*this);
}


void MeldForce::getDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
            float& r4, float& forceConstant, int& globalIndex) const {
    const DistanceRestraintInfo& rest = distanceRestraints[index];
    atom1 = rest.particle1;
    atom2 = rest.particle2;
    r1 = rest.r1;
    r2 = rest.r2;
    r3 = rest.r3;
    r4 = rest.r4;
    forceConstant = rest.force_constant;
    globalIndex = rest.global_index;
}

void MeldForce::getHyperbolicDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
            float& r4, float& forceConstant, float& asymptote, int& globalIndex) const {
    const HyperbolicDistanceRestraintInfo& rest = hyperbolicDistanceRestraints[index];
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


void MeldForce::getTorsionRestraintParams(int index, int& atom1, int& atom2, int& atom3, int& atom4, float& phi,
                                          float& deltaPhi, float& forceConstant, int& globalIndex) const {
    const TorsionRestraintInfo& rest = torsions[index];
    atom1 = rest.atom1;
    atom2 = rest.atom2;
    atom3 = rest.atom3;
    atom4 = rest.atom4;
    phi = rest.phi;
    deltaPhi = rest.deltaPhi;
    forceConstant = rest.forceConstant;
    globalIndex = rest.globalIndex;
}


void MeldForce::getDistProfileRestraintParams(int index, int& atom1, int& atom2, float& rMin, float & rMax,
        int& nBins, std::vector<double>& a0, std::vector<double>& a1, std::vector<double>& a2,
        std::vector<double>& a3, float& scaleFactor, int& globalIndex) const {

    const DistProfileRestraintInfo& rest = distProfileRestraints[index];

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

void MeldForce::getTorsProfileRestraintParams(int index, int& atom1, int& atom2, int& atom3, int& atom4,
        int& atom5, int& atom6, int& atom7, int& atom8, int& nBins,
        std::vector<double>&  a0, std::vector<double>&  a1, std::vector<double>&  a2,
        std::vector<double>&  a3, std::vector<double>&  a4, std::vector<double>&  a5,
        std::vector<double>&  a6, std::vector<double>&  a7, std::vector<double>&  a8,
        std::vector<double>&  a9, std::vector<double>& a10, std::vector<double>& a11,
        std::vector<double>& a12, std::vector<double>& a13, std::vector<double>& a14,
        std::vector<double>& a15, float& scaleFactor, int& globalIndex) const {
    const TorsProfileRestraintInfo& rest = torsProfileRestraints[index];

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


void MeldForce::getGroupParams(int index, std::vector<int>& indices, int& numActive) const {
    const GroupInfo& grp = groups[index];
    indices = grp.restraint_indices;
    numActive = grp.n_active;
}


void MeldForce::getCollectionParams(int index, std::vector<int>& indices, int& numActive) const {
    const CollectionInfo& col = collections[index];
    indices = col.group_indices;
    numActive = col.n_active;
}
//Cong added
void MeldForce::setUsesPeriodicBoundaryConditions(bool periodic) {
  usePeriodic = periodic;
}

bool MeldForce::usesPeriodicBoundaryConditions() const {
  return usePeriodic;
}
//Cong added end
