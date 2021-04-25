#include "openmm/OpenMMException.h"
#include "openmm/reference/RealVec.h"
#include "MeldVecTypes.h"
#include <vector>
#include <algorithm>
#include <iostream>

using namespace OpenMM;
using namespace std;

float3 RealVecToFloat3(RealVec v)
{
    return float3(v[0], v[1], v[2]);
}

Vec3 Float3ToVec3(float3 v)
{
    return RealVec(
        get<0>(v),
        get<1>(v),
        get<2>(v));
}

void computeDistRest(
    vector<RealVec> &pos,
    vector<int2> &atomIndices,
    vector<float4> &distanceBounds,
    vector<float> &forceConstants,
    vector<int> &indexToGlobal,
    vector<float> &energies,
    vector<float3> &forceBuffer,
    int numRestraints)
{
    for (int index = 0; index < numRestraints; index++)
    {
        // get my global index
        const int globalIndex = indexToGlobal[index];

        // get the distances
        const float r1 = get<0>(distanceBounds[index]);
        const float r2 = get<1>(distanceBounds[index]);
        const float r3 = get<2>(distanceBounds[index]);
        const float r4 = get<3>(distanceBounds[index]);

        // get the force constant
        const float k = forceConstants[index];

        // get atom indices and compute distance
        int atomIndexA = get<0>(atomIndices[index]);
        int atomIndexB = get<1>(atomIndices[index]);
        RealVec delta = pos[atomIndexA] - pos[atomIndexB];
        float distSquared = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
        float r = SQRT(distSquared);

        // compute force and energy
        float energy = 0.0;
        float dEdR = 0.0;
        float diff = 0.0;
        float diff2 = 0.0;
        float3 f;

        if (r < r1)
        {
            energy = k * (r - r1) * (r1 - r2) + 0.5 * k * (r1 - r2) * (r1 - r2);
            dEdR = k * (r1 - r2);
        }
        else if (r < r2)
        {
            diff = r - r2;
            diff2 = diff * diff;
            energy = 0.5 * k * diff2;
            dEdR = k * diff;
        }
        else if (r < r3)
        {
            dEdR = 0.0;
            energy = 0.0;
        }
        else if (r < r4)
        {
            diff = r - r3;
            diff2 = diff * diff;
            energy = 0.5 * k * diff2;
            dEdR = k * diff;
        }
        else
        {
            energy = k * (r - r4) * (r4 - r3) + 0.5 * k * (r4 - r3) * (r4 - r3);
            dEdR = k * (r4 - r3);
        }

        // store force into local buffer
        if (r > 0)
        {
            f = float3(delta[0] * dEdR / r, delta[1] * dEdR / r, delta[2] * dEdR / r);
        }
        else
        {
            f = float3(0, 0, 0);
        }
        forceBuffer[index] = f;

        // store energy into global buffer
        energies[globalIndex] = energy;
    }
}

void computeHyperbolicDistRest(
    vector<RealVec> &pos,
    vector<int2> &hyperbolicDistanceRestAtomIndices,
    vector<float4> &hyperbolicDistanceRestRParams,
    vector<float4> &hyperbolicDistanceRestParams,
    vector<int> &hyperbolicDistanceRestGlobalIndices,
    vector<float> &restraintEnergies,
    vector<float3> &hyperbolicDistanceRestForces,
    int numHyperbolicDistRestraints)
{
    throw OpenMMException("Hyperbolic distance restraints not implemented for Reference platform.");
}

void computeTorsionAngle(const vector<RealVec> pos, int atom_i, int atom_j, int atom_k, int atom_l,
                         RealVec &r_ij, RealVec &r_kj, RealVec &r_kl, RealVec &m, RealVec &n,
                         float &len_r_kj, float &len_m, float &len_n, float &phi)
{
    // compute vectors
    r_ij = pos[atom_j] - pos[atom_i];
    r_kj = pos[atom_j] - pos[atom_k];
    r_kl = pos[atom_l] - pos[atom_k];

    // compute normal vectors
    m = r_ij.cross(r_kj);
    n = r_kj.cross(r_kl);

    // compute lengths
    len_r_kj = sqrt(r_kj.dot(r_kj));
    len_m = sqrt(m.dot(m));
    len_n = sqrt(n.dot(n));

    // compute angle phi
    float x = (m / len_m).dot(n / len_n);
    float y = (m / len_m).cross(r_kj / len_r_kj).dot(n / len_n);
    phi = atan2(y, x) * 180. / 3.141592654;
}

void computeTorsionForce(const float dEdPhi, const RealVec &r_ij, const RealVec &r_kj, const RealVec &r_kl,
                         const RealVec &m, const RealVec &n, const float len_r_kj, const float len_m, const float len_n,
                         RealVec &F_i, RealVec &F_j, RealVec &F_k, RealVec &F_l)
{
    F_i = -180. / 3.141592654 * dEdPhi * len_r_kj * m / (len_m * len_m);
    F_l = 180. / 3.141592654 * dEdPhi * len_r_kj * n / (len_n * len_n);
    F_j = -F_i + r_ij.dot(r_kj) / (len_r_kj * len_r_kj) * F_i - r_kl.dot(r_kj) / (len_r_kj * len_r_kj) * F_l;
    F_k = -F_l - r_ij.dot(r_kj) / (len_r_kj * len_r_kj) * F_i + r_kl.dot(r_kj) / (len_r_kj * len_r_kj) * F_l;
}

void computeTorsionRest(
    vector<RealVec> &pos,
    vector<int4> &atomIndices,
    vector<float3> &params,
    vector<int> &indexToGlobal,
    vector<float> &energies,
    vector<float3> &forceBuffer,
    int numRestraints)
{
    for (int index = 0; index < numRestraints; index++)
    {
        // get my global index
        int globalIndex = indexToGlobal[index];

        // get the atom indices
        auto indices = atomIndices[index];
        auto atom_i = get<0>(indices);
        auto atom_j = get<1>(indices);
        auto atom_k = get<2>(indices);
        auto atom_l = get<3>(indices);

        // compute the angle and related quantities
        RealVec r_ij, r_kj, r_kl;
        RealVec m, n;
        float len_r_kj;
        float len_m;
        float len_n;
        float phi;
        computeTorsionAngle(pos, atom_i, atom_j, atom_k, atom_l,
                            r_ij, r_kj, r_kl, m, n, len_r_kj, len_m, len_n, phi);

        // compute E and dE/dphi
        auto phiEquil = get<0>(params[index]);
        auto phiDelta = get<1>(params[index]);
        auto forceConst = get<2>(params[index]);

        auto phiDiff = phi - phiEquil;
        if (phiDiff < -180.)
        {
            phiDiff += 360.;
        }
        else if (phiDiff > 180.)
        {
            phiDiff -= 360.;
        }

        float energy = 0.0;
        float dEdPhi = 0.0;
        if (phiDiff < -phiDelta)
        {
            energy = 0.5 * forceConst * (phiDiff + phiDelta) * (phiDiff + phiDelta);
            dEdPhi = forceConst * (phiDiff + phiDelta);
        }
        else if (phiDiff > phiDelta)
        {
            energy = 0.5 * forceConst * (phiDiff - phiDelta) * (phiDiff - phiDelta);
            dEdPhi = forceConst * (phiDiff - phiDelta);
        }
        else
        {
            energy = 0.0;
            dEdPhi = 0.0;
        }

        energies[globalIndex] = energy;

        RealVec f_i;
        RealVec f_j;
        RealVec f_k;
        RealVec f_l;
        computeTorsionForce(dEdPhi, r_ij, r_kj, r_kl, m, n, len_r_kj, len_m, len_n,
                            f_i, f_j, f_k, f_l);

        forceBuffer[4 * index + 0] = RealVecToFloat3(f_i);
        forceBuffer[4 * index + 1] = RealVecToFloat3(f_j);
        forceBuffer[4 * index + 2] = RealVecToFloat3(f_k);
        forceBuffer[4 * index + 3] = RealVecToFloat3(f_l);
    }
}

void computeDistProfileRest(
    vector<RealVec> &pos,
    vector<int2> &distProfileRestAtomIndices,
    vector<float2> &distProfileRestDistRanges,
    vector<int> &distProfileRestNumBins,
    vector<float4> &distProfileRestParams,
    vector<int2> &distProfileRestParamBounds,
    vector<float> &distProfileRestScaleFactor,
    vector<int> &distProfileRestGlobalIndices,
    vector<float> &restraintEnergies,
    vector<float3> &distProfileRestForces,
    int numDistProfileRestraints)
{
    throw OpenMMException("Distance profile restraints not implemented for Reference platform.");
}

void computeTorsProfileRest(
    vector<RealVec> &pos,
    vector<int4> &torsProfileRestAtomIndices0,
    vector<int4> &torsProfileRestAtomIndices1,
    vector<int> &torsProfileRestNumBins,
    vector<float4> &torsProfileRestParams0,
    vector<float4> &torsProfileRestParams1,
    vector<float4> &torsProfileRestParams2,
    vector<float4> &torsProfileRestParams3,
    vector<int2> &torsProfileRestParamBounds,
    vector<float> &torsProfileRestScaleFactor,
    vector<int> &torsProfileRestGlobalIndices,
    vector<float> &restraintEnergies,
    vector<float3> &torsProfileRestForces,
    int numTorsProfileRestraints)
{
    throw OpenMMException("Torsion profile restraints not implemented for Reference platform.");
}

void computeGMMRest(
    vector<RealVec> &pos,
    int numGMMRestraints,
    vector<int4> &gmmParams,
    vector<int2> &gmmOffsets,
    vector<int> &gmmAtomIndices,
    vector<float> &gmmData,
    vector<float> &restraintEnergies,
    vector<float3> &gmmForces)
{
    throw OpenMMException("GMM restraints not implemented for Reference platform.");
}

void evaluateAndActivate(
    int numGroups,
    vector<int> &groupNumActive,
    vector<int2> &groupBounds,
    vector<int> &groupRestraintIndices,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    vector<float> &groupEnergies)
{
    for (int i = 0; i < numGroups; i++)
    {
        auto start = get<0>(groupBounds[i]);
        auto end = get<1>(groupBounds[i]);
        auto length = end - start;

        // gather the restraint energies into a vector
        auto energies = vector<float>(length, 0);
        for (int j = 0; j < length; j++)
        {
            energies[j] = restraintEnergies[groupRestraintIndices[start + j]];
        }

        // find the nth largest
        sort(energies.begin(), energies.end());
        auto cutoff = energies[groupNumActive[i] - 1];

        // activate all with energy <= cutoff
        float totalEnergy = 0.0;
        for (int j = 0; j < length; j++)
        {
            if (energies[j] <= cutoff)
            {
                restraintActive[groupRestraintIndices[start + j]] = true;
                totalEnergy += energies[j];
            }
            else
            {
                restraintActive[groupRestraintIndices[start + j]] = false;
            }
        }
        // store the energy of the group
        groupEnergies[i] = totalEnergy;
    }
}

void evaluateAndActivateCollections(
    int numCollections,
    vector<int> &collectionNumActive,
    vector<int2> &collectionBounds,
    vector<int> &collectionGroupIndices,
    vector<float> &groupEnergies,
    vector<bool> &groupActive)
{
    // loop over collections
    for (int i = 0; i < numCollections; i++)
    {
        auto start = get<0>(collectionBounds[i]);
        auto end = get<1>(collectionBounds[i]);
        auto length = end - start;

        // gather group energies
        auto energies = vector<float>(length, 0);
        for (int j = 0; j < length; j++)
        {
            energies[j] = groupEnergies[collectionGroupIndices[start + j]];
        }

        // find the nth largest
        sort(energies.begin(), energies.end());
        auto cutoff = energies[collectionNumActive[i] - 1];

        // activate groups with energy <= cutoff
        for (int j = 0; j < length; j++)
        {
            if (energies[j] <= cutoff)
            {
                groupActive[collectionGroupIndices[start + j]] = true;
            }
            else
            {
                groupActive[collectionGroupIndices[start + j]] = false;
            }
        }
    }
}

void applyGroups(
    vector<bool> &groupActive,
    vector<bool> &restraintActive,
    vector<int2> &groupBounds,
    int numGroups)
{
    for (int i = 0; i < numGroups; i++)
    {
        auto start = get<0>(groupBounds[i]);
        auto end = get<1>(groupBounds[i]);
        auto active = groupActive[i];

        for (int j = start; j < end; j++)
        {
            restraintActive[j] = restraintActive[j] && active;
        }
    }
}

float applyDistRest(
    vector<RealVec> &force,
    vector<int2> &distanceRestAtomIndices,
    vector<int> &distanceRestGlobalIndices,
    vector<float3> &distanceRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numDistRestraints)
{
    float totalEnergy = 0.0;
    for (int i = 0; i < numDistRestraints; i++)
    {
        auto index = distanceRestGlobalIndices[i];
        if (restraintActive[index])
        {
            totalEnergy += restraintEnergies[index];

            auto fx = get<0>(distanceRestForces[i]);
            auto fy = get<1>(distanceRestForces[i]);
            auto fz = get<2>(distanceRestForces[i]);
            auto atom1 = get<0>(distanceRestAtomIndices[i]);
            auto atom2 = get<1>(distanceRestAtomIndices[i]);
            force[atom1] += Vec3(-fx, -fy, -fz);
            force[atom2] += Vec3(fx, fy, fz);
        }
    }
    return totalEnergy;
}

float applyHyperbolicDistRest(
    vector<RealVec> &force,
    vector<int2> &hyperbolicDistanceRestAtomIndices,
    vector<int> &hyperbolicDistanceRestGlobalIndices,
    vector<float3> &hyperbolicDistanceRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numHyperbolicDistRestraints)
{
    throw OpenMMException("Hyperbolic distance restraints not implemented for Reference platform.");
}

float applyTorsionRest(
    vector<RealVec> &force,
    vector<int4> &torsionRestAtomIndices,
    vector<int> &torsionRestGlobalIndices,
    vector<float3> &torsionRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numTorsionRestraints)
{
    float totalEnergy = 0.0;
    for (int i = 0; i < numTorsionRestraints; i++)
    {
        auto index = torsionRestGlobalIndices[i];
        if (restraintActive[index])
        {
            totalEnergy += restraintEnergies[index];

            auto f1 = Float3ToVec3(torsionRestForces[4 * index + 0]);
            auto f2 = Float3ToVec3(torsionRestForces[4 * index + 1]);
            auto f3 = Float3ToVec3(torsionRestForces[4 * index + 2]);
            auto f4 = Float3ToVec3(torsionRestForces[4 * index + 3]);
            auto atom1 = get<0>(torsionRestAtomIndices[i]);
            auto atom2 = get<1>(torsionRestAtomIndices[i]);
            auto atom3 = get<2>(torsionRestAtomIndices[i]);
            auto atom4 = get<3>(torsionRestAtomIndices[i]);
            force[atom1] += f1;
            force[atom2] += f2;
            force[atom3] += f3;
            force[atom4] += f4;
        }
    }
    return totalEnergy;
}

float applyDistProfileRest(
    vector<RealVec> &force,
    vector<int2> &distProfileRestAtomIndices,
    vector<int> &distProfileRestGlobalIndices,
    vector<float3> &distProfileRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numDistProfileRestraints)
{
    throw OpenMMException("Distance profile restraints not implemented for Reference platform.");
}

float applyTorsProfileRest(
    vector<RealVec> &force,
    vector<int4> &torsProfileRestAtomIndices0,
    vector<int4> &torsProfileRestAtomIndices1,
    vector<int> &torsProfileRestGlobalIndices,
    vector<float3> &torsProfileRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numTorsProfileRestraints)
{
    throw OpenMMException("Torsion profile restraints not implemented for Reference platform.");
}

float applyGMMRest(
    vector<RealVec> &force,
    int numGMMRestraints,
    vector<int4> &gmmParams,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    vector<int2> &gmmOffsets,
    vector<int> &gmmAtomIndices,
    vector<float3> &gmmForces)
{
    throw OpenMMException("GMM restraints not implemented for Reference platform.");
}
