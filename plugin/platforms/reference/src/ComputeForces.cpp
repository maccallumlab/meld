#include "openmm/OpenMMException.h"
#include "openmm/reference/RealVec.h"
#include "MeldVecTypes.h"
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <limits>

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

void computeRDCRest(
    vector<RealVec>& pos,
    vector<int2>& atomIndices,
    vector<float2>& params1,
    vector<float3>& params2,
    float scaleFactor,
    vector<int>& alignments,
    vector<float>& tensorComponents,
    vector<int>& globalIndices,
    vector<float>& energies,
    vector<float3>& forceBuffer,
    vector<float>& derivBuffer,
    int numRestraints)
{
    for (int index = 0; index < numRestraints; index++)
    {
        // Unpack parameters
        int globalIndex = globalIndices[index];
        int atom1 = get<0>(atomIndices[index]);
        int atom2 = get<1>(atomIndices[index]);
        float kappa = get<0>(params1[index]);
        float obs = get<1>(params1[index]);
        float tol = get<0>(params2[index]);
        float quadCut = get<1>(params2[index]);
        float forceConstant = get<2>(params2[index]);

        // Unpack the alignment tensor components for this restraint
        int alignment = alignments[index];
        float s1 = tensorComponents[alignment * 5 + 0];
        float s2 = tensorComponents[alignment * 5 + 1];
        float s3 = tensorComponents[alignment * 5 + 2];
        float s4 = tensorComponents[alignment * 5 + 3];
        float s5 = tensorComponents[alignment * 5 + 4];

        if (atom1 == -1)
        {
            // If the first index is -1, this restraint
            // is marked as being not mapped.  We set the force to
            // zero. We set the energy to the maximum float value,
            // so that this restraint will not be selected during
            // sorting when the groups are evaluated. Later,
            // when we apply restraints, this restraint will be
            // applied with an energy of zero should it be selected.
            forceBuffer[index] = float3(0, 0, 0);
            energies[globalIndex] = std::numeric_limits<float>::max();
            derivBuffer[5 * index + 0] = 0;
            derivBuffer[5 * index + 1] = 0;
            derivBuffer[5 * index + 2] = 0;
            derivBuffer[5 * index + 3] = 0;
            derivBuffer[5 * index + 4] = 0;
        }
        else
        {
            RealVec delta = pos[atom1] - pos[atom2];
            float x = delta[0];
            float y = delta[1];
            float z = delta[2];
            float x2 = x * x;
            float y2 = y * y;
            float z2 = z * z;
            float r2 = x2 + y2 + z2;
            float r = SQRT(r2);
            float r5 = r2 * r2 * r;
            float r7 = r5 * r2;

            float pre = kappa / r5;
            float c1 = scaleFactor * (x2 - y2);
            float c2 = scaleFactor * (2 * z2 - x2 - y2);
            float c3 = scaleFactor * (x * y);
            float c4 = scaleFactor * (x * z);
            float c5 = scaleFactor * (y * z);

            float calc = pre * (s1 * c1 + s2 * c2 + s3 * c3 + s4 * c4 + s5 * c5);
            float dcalc_ds1 = pre * c1;
            float dcalc_ds2 = pre * c2;
            float dcalc_ds3 = pre * c3;
            float dcalc_ds4 = pre * c4;
            float dcalc_ds5 = pre * c5;
            float dcalc_dx = -kappa * x / r7 * (s1 * c1 + s2 * c2 + s3 * c3 + s4 * c4 + s5 * c5) +
                             pre * (
                                 2 * scaleFactor * s1 * x -
                                 2 * scaleFactor * s2 * x +
                                 scaleFactor * s3 * y +
                                 scaleFactor * s4 * z
                             );
            float dcalc_dy = -kappa * y / r7 * (s1 * c1 + s2 * c2 + s3 * c3 + s4 * c4 + s5 * c5) +
                             pre * (
                                 -2 * scaleFactor * s1 * y -
                                 2 * scaleFactor * s2 * y +
                                 scaleFactor * s3 * x +
                                 scaleFactor * s5 * z
                             );
            float dcalc_dz = -kappa * z / r7 * (s1 * c1 + s2 * c2 + s3 * c3 + s4 * c4 + s5 * c5) +
                             pre * (
                                 2 * scaleFactor * s2 * z +
                                 scaleFactor * s4 * x +
                                 scaleFactor * s5 * y
                             );

            float energy, dx, dy, dz, ds1, ds2, ds3, ds4, ds5;
            if (calc < obs - tol - quadCut)
            {
                energy = -forceConstant * quadCut * (calc - obs + tol + quadCut) +
                         0.5 * forceConstant * quadCut * quadCut;
                dx = -forceConstant * quadCut * dcalc_dx;
                dy = -forceConstant * quadCut * dcalc_dy;
                dz = -forceConstant * quadCut * dcalc_dz;
                ds1 = -forceConstant * quadCut * dcalc_ds1;
                ds2 = -forceConstant * quadCut * dcalc_ds2;
                ds3 = -forceConstant * quadCut * dcalc_ds3;
                ds4 = -forceConstant * quadCut * dcalc_ds4;
                ds5 = -forceConstant * quadCut * dcalc_ds5;
            }
            else if (calc < obs - tol)
            {
                energy = 0.5 * forceConstant * (calc - obs + tol) * (calc - obs + tol);
                dx = forceConstant * (calc - obs + tol) * dcalc_dx;
                dy = forceConstant * (calc - obs + tol) * dcalc_dy;
                dz = forceConstant * (calc - obs + tol) * dcalc_dz;
                ds1 = forceConstant * (calc - obs + tol) * dcalc_ds1;
                ds2 = forceConstant * (calc - obs + tol) * dcalc_ds2;
                ds3 = forceConstant * (calc - obs + tol) * dcalc_ds3;
                ds4 = forceConstant * (calc - obs + tol) * dcalc_ds4;
                ds5 = forceConstant * (calc - obs + tol) * dcalc_ds5;
            }
            else if (calc < obs + tol)
            {
                energy = 0;
                dx = 0;
                dy = 0;
                dz = 0;
                ds1 = 0;
                ds2 = 0;
                ds3 = 0;
                ds4 = 0;
                ds5 = 0;
            }
            else if (calc < obs + tol + quadCut)
            {
                energy = 0.5 * forceConstant * (calc - obs - tol) * (calc - obs - tol);
                dx = forceConstant * (calc - obs - tol) * dcalc_dx;
                dy = forceConstant * (calc - obs - tol) * dcalc_dy;
                dz = forceConstant * (calc - obs - tol) * dcalc_dz;
                ds1 = forceConstant * (calc - obs - tol) * dcalc_ds1;
                ds2 = forceConstant * (calc - obs - tol) * dcalc_ds2;
                ds3 = forceConstant * (calc - obs - tol) * dcalc_ds3;
                ds4 = forceConstant * (calc - obs - tol) * dcalc_ds4;
                ds5 = forceConstant * (calc - obs - tol) * dcalc_ds5;
            }
            else
            {
                energy = forceConstant * quadCut * (calc - obs - tol - quadCut) +
                         0.5 * forceConstant * quadCut * quadCut;
                dx = forceConstant * quadCut * dcalc_dx;
                dy = forceConstant * quadCut * dcalc_dy;
                dz = forceConstant * quadCut * dcalc_dz;
                ds1 = forceConstant * quadCut * dcalc_ds1;
                ds2 = forceConstant * quadCut * dcalc_ds2;
                ds3 = forceConstant * quadCut * dcalc_ds3;
                ds4 = forceConstant * quadCut * dcalc_ds4;
                ds5 = forceConstant * quadCut * dcalc_ds5;
            }

            forceBuffer[index] = float3(dx, dy, dz);
            energies[globalIndex] = energy;
            derivBuffer[5 * index + 0] = ds1;
            derivBuffer[5 * index + 1] = ds2;
            derivBuffer[5 * index + 2] = ds3;
            derivBuffer[5 * index + 3] = ds4;
            derivBuffer[5 * index + 4] = ds5;
        }
    }
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

        if (atomIndexA == -1)
        {
            // If the first index is -1, this restraint
            // is marked as being not mapped.  We set the force to
            // zero. We set the energy to the maximum float value,
            // so that this restraint will not be selected during
            // sorting when the groups are evaluated. Later,
            // when we apply restraints, this restraint will be
            // applied with an energy of zero should it be selected.
            forceBuffer[index] = float3(0, 0, 0);
            energies[globalIndex] = std::numeric_limits<float>::max();
        }
        else
        {
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

//  void computeGridPotentialRest(
//                             vector<RealVec> &pos,
//                             vector<int> &atomIndices, 
//                             vector<float> &potentials,
//                             vector<float> &grid_x,
//                             vector<float> &grid_y,
//                             vector<float> &grid_z,
//                             vector<float> &weights,
//                             vector<int> &nxyz,
//                             vector<int> &densityIndices,
//                             vector<int> &indexToGlobal,
//                             int numRestraints,
//                             vector<float> &energies,    
//                             vector<float3> &forceBuffer)
// {
//     int grid_xmax = nxyz[0];
//     int grid_ymax = nxyz[1];
//     int grid_zmax = nxyz[2];
//     int grid_total = grid_xmax * grid_ymax * grid_zmax;
//     for (int index = 0; index < numRestraints ; index++) {
//         int globalIndex = indexToGlobal[index];
//         int grids_index = densityIndices[index];
//         int atomIndex = atomIndices[index];
//         float atom_weight = weights[index];
//         float3 atom_pos = RealVecToFloat3(pos[atomIndex]);
//         float atom_posx = get<0>(atom_pos);
//         float atom_posy = get<1>(atom_pos);
//         float atom_posz = get<2>(atom_pos);
//         cout << "atom_posz: " << atom_posz << endl;
//         cout << "grid_x[1]" << grid_x[1] << endl;
//         // check the atom is in which grid
//         int grid_xnum = floor((atom_posx-grid_x[0])/(grid_x[1]-grid_x[0])) + 1; 
//         int grid_ynum = floor((atom_posy-grid_y[0])/(grid_y[1]-grid_y[0])) + 1;
//         int grid_znum = floor((atom_posz-grid_z[0])/(grid_z[1]-grid_z[0])) + 1;
//         cout << "grid_znum: " << grid_znum << endl;

//         // scale atom position with grid length
//         float grid_x_pos = (grid_x[grid_xnum]-atom_posx)/(grid_x[1]-grid_x[0]); 
//         float grid_y_pos = (grid_y[grid_ynum]-atom_posy)/(grid_y[1]-grid_y[0]);
//         float grid_z_pos = (grid_z[grid_znum]-atom_posz)/(grid_z[1]-grid_z[0]);
//         float grid_xpos = (atom_posx-grid_x[grid_xnum-1])/(grid_x[1]-grid_x[0]);
//         float grid_ypos = (atom_posy-grid_y[grid_ynum-1])/(grid_y[1]-grid_y[0]);
//         float grid_zpos = (atom_posz-grid_z[grid_znum-1])/(grid_z[1]-grid_z[0]);
//         cout << "grid_zpos: " << grid_zpos << endl;

//         float energy = 0;
//         float f_x = 0;
//         float f_y = 0;
//         float f_z = 0;        
//         float v_000 = potentials[grid_total*grids_index+(grid_znum-1) * grid_ymax * grid_xmax + (grid_ynum-1) * grid_xmax + grid_xnum -1];
//         float v_100 = potentials[grid_total*grids_index+(grid_znum-1) * grid_ymax * grid_xmax + (grid_ynum-1) * grid_xmax + grid_xnum];
//         float v_010 = potentials[grid_total*grids_index+(grid_znum-1) * grid_ymax * grid_xmax + grid_ynum * grid_xmax + grid_xnum -1];
//         float v_001 = potentials[grid_total*grids_index+grid_znum * grid_ymax * grid_xmax + (grid_ynum-1) * grid_xmax + grid_xnum -1];
//         float v_101 = potentials[grid_total*grids_index+grid_znum * grid_ymax * grid_xmax + (grid_ynum-1) * grid_xmax + grid_xnum];
//         float v_011 = potentials[grid_total*grids_index+grid_znum * grid_ymax * grid_xmax + grid_ynum * grid_xmax + grid_xnum -1];
//         float v_110 = potentials[grid_total*grids_index+(grid_znum-1) * grid_ymax * grid_xmax + grid_ynum * grid_xmax + grid_xnum ];
//         float v_111 = potentials[grid_total*grids_index+grid_znum * grid_ymax * grid_xmax + grid_ynum * grid_xmax + grid_xnum];
//         cout << "v_111: " << v_111 << endl;
//         energy += atom_weight * (v_000 * grid_x_pos * grid_y_pos * grid_z_pos              
//                 + v_100 * grid_xpos * grid_y_pos * grid_z_pos 
//                 + v_010 * grid_x_pos * grid_ypos * grid_z_pos   
//                 + v_001 * grid_x_pos * grid_y_pos * grid_zpos
//                 + v_101 * grid_xpos * grid_y_pos * grid_zpos 
//                 + v_011 * grid_x_pos * grid_ypos * grid_zpos
//                 + v_110 * grid_xpos * grid_ypos * grid_z_pos           
//                 + v_111 * grid_xpos * grid_ypos * grid_zpos)  ;

//         f_x += -1 * atom_weight * ((v_100 - v_000) * grid_y_pos * grid_z_pos 
//                 + (v_110 - v_010) * grid_ypos * grid_z_pos
//                 + (v_101 - v_001) * grid_y_pos * grid_zpos
//                 + (v_111 - v_011) * grid_ypos * grid_zpos)/(grid_x[1]-grid_x[0])   ;
                
//         f_y += -1 * atom_weight * ((v_010 - v_000) * grid_x_pos * grid_z_pos 
//                 + (v_110 - v_100) * grid_xpos * grid_z_pos
//                 + (v_011 - v_001) * grid_x_pos * grid_zpos
//                 + (v_111 - v_101) * grid_xpos * grid_zpos)/(grid_x[1]-grid_x[0])  ;

//         f_z += -1 * atom_weight * ((v_001 - v_000) * grid_x_pos * grid_y_pos 
//                 + (v_101 - v_100) * grid_xpos * grid_y_pos
//                 + (v_011 - v_010) * grid_x_pos * grid_ypos
//                 + (v_111 - v_110) * grid_xpos * grid_ypos)/(grid_x[1]-grid_x[0])   ;
//         cout << "fxyz: " << f_x << " " << f_y << " " << f_z << endl;
//         forceBuffer[index] = float3(f_x,f_y,f_z);
//         energies[globalIndex] += energy;
//     }
// }

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
        auto sorted_energies = vector<float>(length, 0);
        for (int j = 0; j < length; j++)
        {
            energies[j] = restraintEnergies[groupRestraintIndices[start + j]];
            sorted_energies[j] = energies[j];
        }

        // find the nth largest
        sort(sorted_energies.begin(), sorted_energies.end());
        auto cutoff = sorted_energies[groupNumActive[i] - 1];

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
        auto sorted_energies = vector<float>(length, 0);
        for (int j = 0; j < length; j++)
        {
            energies[j] = groupEnergies[collectionGroupIndices[start + j]];
            sorted_energies[j] = energies[j];
        }

        // find the nth largest
        sort(sorted_energies.begin(), sorted_energies.end());
        auto cutoff = sorted_energies[collectionNumActive[i] - 1];

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

float applyRDCRest(
    vector<RealVec>& force,
    vector<int2>& rdcRestAtomIndices,
    vector<int>& rdcAlignments,
    vector<int>& rdcRestGlobalIndices,
    vector<float3>& rdcRestForces,
    vector<float>& rdcRestDerivs,
    vector<float>& restraintEnergies,
    vector<bool>& restraintActive,
    std::map<std::string, double>& derivMap,
    int numRDCRestraints)
{
    float totalEnergy = 0.0;
    for (int i = 0; i < numRDCRestraints; i++)
    {
        int index = rdcRestGlobalIndices[i];
        if (restraintActive[index])
        {
            if (get<0>(rdcRestAtomIndices[i]) == -1)
            {
                // Do nothing. This restraint is marked as being
                // not mapped, so it contributes no energy or force.
            }
            else
            {
                totalEnergy += restraintEnergies[index];

                float fx = get<0>(rdcRestForces[i]);
                float fy = get<1>(rdcRestForces[i]);
                float fz = get<2>(rdcRestForces[i]);
                float ds1 = rdcRestDerivs[5 * i + 0];
                float ds2 = rdcRestDerivs[5 * i + 1];
                float ds3 = rdcRestDerivs[5 * i + 2];
                float ds4 = rdcRestDerivs[5 * i + 3];
                float ds5 = rdcRestDerivs[5 * i + 4];
                int atom1 = get<0>(rdcRestAtomIndices[i]);
                int atom2 = get<1>(rdcRestAtomIndices[i]);

                force[atom1] += Vec3(-fx, -fy, -fz);
                force[atom2] += Vec3(fx, fy, fz);

                std::string base = "rdc_" + std::to_string(rdcAlignments[i]);
                derivMap[base + "_s1"] += ds1;
                derivMap[base + "_s2"] += ds2;
                derivMap[base + "_s3"] += ds3;
                derivMap[base + "_s4"] += ds4;
                derivMap[base + "_s5"] += ds5;
            }
        }
    }
    return totalEnergy;
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
            if (get<0>(distanceRestAtomIndices[i]) == -1)
            {
                // Do nothing. This restraint is marked as being
                // not mapped, so it contributes no energy or force.
            }
            else
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

float applyGridPotentialRest(vector<RealVec> &force,
                             vector<int> &atomIndices,
                             vector<int> &globalIndices,
                             vector<float> &globalEnergies,
                             vector<bool> &globalActive,
                             vector<float3> &restForces,
                             int numRestraints) {
    float totalEnergy = 0.0;
    for (int restraintIndex=0; restraintIndex<numRestraints; restraintIndex++) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            totalEnergy += globalEnergies[globalIndex];
            int index = atomIndices[restraintIndex];
            auto fx = get<0>(restForces[restraintIndex]);
            auto fy = get<1>(restForces[restraintIndex]);
            auto fz = get<2>(restForces[restraintIndex]);
            force[index] += Vec3(fx, fy, fz);
            }
    }

    return totalEnergy;
}
