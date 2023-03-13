/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#include <cub/cub.cuh>
#include <cfloat>

__device__ void computeTorsionAngle(const real4* __restrict__ posq, int atom_i, int atom_j, int atom_k, int atom_l,
        float3& r_ij, float3& r_kj, float3& r_kl, float3& m, float3& n,
        float& len_r_kj, float& len_m, float& len_n, float& phi) {
    // compute vectors
    r_ij = trimTo3(posq[atom_j] - posq[atom_i]);
    r_kj = trimTo3(posq[atom_j] - posq[atom_k]);
    r_kl = trimTo3(posq[atom_l] - posq[atom_k]);

    // compute normal vectors
    m = cross(r_ij, r_kj);
    n = cross(r_kj, r_kl);

    // compute lengths
    len_r_kj = sqrt(dot(r_kj, r_kj));
    len_m = sqrt(dot(m, m));
    len_n = sqrt(dot(n, n));

    // compute angle phi
    float x = dot(m / len_m, n / len_n);
    float y = dot(cross(m / len_m, r_kj / len_r_kj), n / len_n);
    phi = atan2(y, x) * 180. / 3.141592654;
}


__device__ void computeTorsionForce(const float dEdPhi, const float3& r_ij, const float3& r_kj, const float3& r_kl,
        const float3& m, const float3& n, const float len_r_kj, const float len_m, const float len_n,
        float3& F_i, float3& F_j, float3& F_k, float3& F_l) {
    F_i = -180. / 3.141592654 * dEdPhi * len_r_kj * m / (len_m * len_m);
    F_l = 180. / 3.141592654 * dEdPhi * len_r_kj * n / (len_n * len_n);
    F_j = -F_i + dot(r_ij, r_kj) / (len_r_kj * len_r_kj) * F_i - dot(r_kl, r_kj) / (len_r_kj * len_r_kj) * F_l;
    F_k = -F_l - dot(r_ij, r_kj) / (len_r_kj * len_r_kj) * F_i + dot(r_kl, r_kj) / (len_r_kj * len_r_kj) * F_l;
}


extern "C" __global__ void computeRDCRest(
    const real4* __restrict__ posq,
    const int2* __restrict__ atomIndices,
    const float2* __restrict__ params1,
    const float3* __restrict__ params2,
    const float scaleFactor,
    const int* __restrict__ alignments,
    const float* __restrict__ tensorComponents,
    const int* __restrict__ globalIndices,
    float* __restrict__ energies,
    float3* __restrict__ forceBuffer,
    float* __restrict__ derivBuffer,
    int numRestraints
) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) {
        // Unpack parameters
        int globalIndex = globalIndices[index];
        int atom1 = atomIndices[index].x;
        int atom2 = atomIndices[index].y;
        float kappa = params1[index].x;
        float obs = params1[index].y;
        float tol = params2[index].x;
        float quadCut = params2[index].y;
        float forceConstant = params2[index].z;

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
            float3 f;
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
            forceBuffer[index] = f;
            energies[globalIndex] = FLT_MAX;
            derivBuffer[5 * index + 0] = 0.0;
            derivBuffer[5 * index + 1] = 0.0;
            derivBuffer[5 * index + 2] = 0.0;
            derivBuffer[5 * index + 3] = 0.0;
            derivBuffer[5 * index + 4] = 0.0;
        }
        else
        {
            real4 delta = posq[atom1] - posq[atom2];
            float x = delta.x;
            float y = delta.y;
            float z = delta.z;
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

            float3 f;
            f.x = dx;
            f.y = dy;
            f.z = dz;
            forceBuffer[index] = f;
            energies[globalIndex] = energy;
            derivBuffer[5 * index + 0] = ds1;
            derivBuffer[5 * index + 1] = ds2;
            derivBuffer[5 * index + 2] = ds3;
            derivBuffer[5 * index + 3] = ds4;
            derivBuffer[5 * index + 4] = ds5;
        }
    }
}


extern "C" __global__ void computeDistRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int2* __restrict__ atomIndices,       // pair of atom indices
                            const float4* __restrict__ distanceBounds,  // r1, r2, r3, r4
                            const float* __restrict__ forceConstants,   // k
                            int* __restrict__ indexToGlobal,            // array of indices into global arrays
                            float* __restrict__ energies,               // global array of restraint energies
                            float3* __restrict__ forceBuffer,           // temporary buffer to hold the force
                            const int numRestraints) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) {
        // get my global index
        const int globalIndex = indexToGlobal[index];

        // get the distances
        const float r1 = distanceBounds[index].x;
        const float r2 = distanceBounds[index].y;
        const float r3 = distanceBounds[index].z;
        const float r4 = distanceBounds[index].w;

        // get the force constant
        const float k = forceConstants[index];

        // get atom indices and compute distance
        int atomIndexA = atomIndices[index].x;
        int atomIndexB = atomIndices[index].y;

        if (atomIndexA == -1) {
            // If the first index is -1, this restraint
            // is marked as being not mapped.  We set the force to
            // zero. We set the energy to FLT_MAX, so that this
            // restraint will not be selected during sorting when
            // the groups are evaluated. Later, when we apply
            // restraints, this restraint will be applied with
            // an energy of zero should it be selected.
            float3 f;
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
            forceBuffer[index] = f;
            energies[globalIndex] = FLT_MAX;
        } else {
            real4 delta = posq[atomIndexA] - posq[atomIndexB];
            real distSquared = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
            real r = SQRT(distSquared);

            // compute force and energy
            float energy = 0.0;
            float dEdR = 0.0;
            float diff = 0.0;
            float diff2 = 0.0;
            float3 f;

            if(r < r1) {
                energy = k * (r - r1) * (r1 - r2) + 0.5 * k * (r1 - r2) * (r1 - r2);
                dEdR = k * (r1 - r2);
            }
            else if(r < r2) {
                diff = r - r2;
                diff2 = diff * diff;
                energy = 0.5 * k * diff2;
                dEdR = k * diff;
            }
            else if(r < r3) {
                dEdR = 0.0;
                energy = 0.0;
            }
            else if(r < r4) {
                diff = r - r3;
                diff2 = diff * diff;
                energy = 0.5 * k * diff2;
                dEdR = k * diff;
            }
            else {
                energy = k * (r - r4) * (r4 - r3) + 0.5 * k * (r4 - r3) * (r4 - r3);
                dEdR = k * (r4 - r3);
            }

            // store force into local buffer
            if (r > 0) {
                f.x = delta.x * dEdR / r;
                f.y = delta.y * dEdR / r;
                f.z = delta.z * dEdR / r;
            } else {
                f.x = 0.0;
                f.y = 0.0;
                f.z = 0.0;
            }
            forceBuffer[index] = f;

            // store energy into global buffer
            energies[globalIndex] = energy;
        }
    }
}


extern "C" __global__ void computeHyperbolicDistRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int2* __restrict__ atomIndices,       // pair of atom indices
                            const float4* __restrict__ distanceBounds,  // r1, r2, r3, r4
                            const float4* __restrict__ params,          // k1, k2, a, b
                            int* __restrict__ indexToGlobal,            // array of indices into global arrays
                            float* __restrict__ energies,               // global array of restraint energies
                            float3* __restrict__ forceBuffer,           // temporary buffer to hold the force
                            const int numRestraints) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) {
        // get my global index
        const int globalIndex = indexToGlobal[index];

        // get the distances
        const float r1 = distanceBounds[index].x;
        const float r2 = distanceBounds[index].y;
        const float r3 = distanceBounds[index].z;
        const float r4 = distanceBounds[index].w;

        // get the parameters
        const float k1 = params[index].x;
        const float k2 = params[index].y;
        const float a = params[index].z;
        const float b = params[index].w;

        // get atom indices and compute distance
        int atomIndexA = atomIndices[index].x;
        int atomIndexB = atomIndices[index].y;
        real4 delta = posq[atomIndexA] - posq[atomIndexB];
        real distSquared = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real r = SQRT(distSquared);

        // compute force and energy
        float energy = 0.0;
        float dEdR = 0.0;
        float diff = 0.0;
        float diff2 = 0.0;
        float3 f;

        if(r < r1) {
            energy = k1 * (r - r1) * (r1 - r2) + 0.5 * k1 * (r1 - r2) * (r1 - r2);
            dEdR = k1 * (r1 - r2);
        }
        else if(r < r2) {
            diff = r - r2;
            diff2 = diff * diff;
            energy = 0.5 * k1 * diff2;
            dEdR = k1 * diff;
        }
        else if(r < r3) {
            dEdR = 0.0;
            energy = 0.0;
        }
        else if(r < r4) {
            diff = r - r3;
            diff2 = diff * diff;
            energy = 0.5 * k2 * diff2;
            dEdR = k2 * diff;
        }
        else {
            energy = 0.5 * k2 * (b / (r - r3) + a);
            dEdR = -0.5 * b * k2 / (r - r3) / (r - r3);
        }

        // store force into local buffer
        if (r > 0) {
            f.x = delta.x * dEdR / r;
            f.y = delta.y * dEdR / r;
            f.z = delta.z * dEdR / r;
        } else {
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
        }
        forceBuffer[index] = f;

        // store energy into global buffer
        energies[globalIndex] = energy;
    }
}


extern "C" __global__ void computeTorsionRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int4* __restrict__ atomIndices,       // indices of atom_{i,j,k,l}
                            const float3* __restrict__ params,          // phi, deltaPhi, forceConstant
                            int* __restrict__ indexToGlobal,            // array of indices into global arrays
                            float* __restrict__ energies,               // global array of restraint energies
                            float3* __restrict__ forceBuffer,           // temporary buffer to hold the force
                                                                        // forceBuffer[index*4] -> atom_i
                                                                        // forceBuffer[index*4 + 3] -> atom_l
                            const int numRestraints) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=gridDim.x*blockDim.x) {
        // get my global index
        int globalIndex = indexToGlobal[index];

        // get the atom indices
        int4 indices = atomIndices[index];
        int atom_i = indices.x;
        int atom_j = indices.y;
        int atom_k = indices.z;
        int atom_l = indices.w;

        // compute the angle and related quantities
        float3 r_ij, r_kj, r_kl;
        float3 m, n;
        float len_r_kj;
        float len_m;
        float len_n;
        float phi;
        computeTorsionAngle(posq, atom_i, atom_j, atom_k, atom_l,
                r_ij, r_kj, r_kl, m, n, len_r_kj, len_m, len_n,  phi);

        // compute E and dE/dphi
        float phiEquil = params[index].x;
        float phiDelta = params[index].y;
        float forceConst = params[index].z;

        float phiDiff = phi - phiEquil;
        if (phiDiff < -180.) {
            phiDiff += 360.;
        } else if (phiDiff > 180.) {
            phiDiff -= 360.;
        }

        float energy = 0.0;
        float dEdPhi = 0.0;
        if (phiDiff < -phiDelta) {
            energy = 0.5 * forceConst * (phiDiff + phiDelta) * (phiDiff + phiDelta);
            dEdPhi = forceConst * (phiDiff + phiDelta);
        }
        else if(phiDiff > phiDelta) {
            energy = 0.5 * forceConst * (phiDiff - phiDelta) * (phiDiff - phiDelta);
            dEdPhi = forceConst * (phiDiff - phiDelta);
        }
        else{
            energy = 0.0;
            dEdPhi = 0.0;
        }

        energies[globalIndex] = energy;

        computeTorsionForce(dEdPhi, r_ij, r_kj, r_kl, m, n, len_r_kj, len_m, len_n,
                forceBuffer[4 * index + 0], forceBuffer[4 * index + 1],
                forceBuffer[4 * index + 2], forceBuffer[4 * index + 3]);
    }
}


extern "C" __global__ void computeDistProfileRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int2* __restrict__ atomIndices,       // pair of atom indices
                            const float2* __restrict__ distRanges,      // upper and lower bounds of spline
                            const int* __restrict__ nBins,              // number of bins
                            const float4* __restrict__ splineParams,    // a0, a1, a2, a3
                            const int2* __restrict__ paramBounds,       // upper and lower bounds for each spline
                            const float* __restrict__ scaleFactor,      // scale factor for energies and forces
                            const int* __restrict__ indexToGlobal,      // index of this restraint in the global array
                            float* __restrict__ restraintEnergies,      // global energy of each restraint
                            float3* __restrict__ restraintForce,        // cache the forces for application later
                            const int numRestraints ) {

    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) {
        // get my global index
        int globalIndex = indexToGlobal[index];

        // get atom indices and compute distance
        int atomIndexA = atomIndices[index].x;
        int atomIndexB = atomIndices[index].y;

        real4 delta = posq[atomIndexA] - posq[atomIndexB];
        real distSquared = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real r = SQRT(distSquared);

        // compute bin
        int bin = (int)( floor((r - distRanges[index].x) / (distRanges[index].y - distRanges[index].x) * nBins[index]) );

        // compute the force and energy
        float energy = 0.0;
        float dEdR = 0.0;
        float binWidth = (distRanges[index].y - distRanges[index].x) / nBins[index];
        if (bin < 0){
            energy = scaleFactor[index] * splineParams[paramBounds[index].x].x;
        }
        else if (bin >= nBins[index]) {
            energy = scaleFactor[index] * (splineParams[paramBounds[index].y - 1].x +
                                           splineParams[paramBounds[index].y - 1].y +
                                           splineParams[paramBounds[index].y - 1].z +
                                           splineParams[paramBounds[index].y - 1].w);
        }
        else {
            float t = (r - bin * binWidth + distRanges[index].x) / binWidth;
            float a0 = splineParams[ paramBounds[index].x + bin ].x;
            float a1 = splineParams[ paramBounds[index].x + bin ].y;
            float a2 = splineParams[ paramBounds[index].x + bin ].z;
            float a3 = splineParams[ paramBounds[index].x + bin ].w;
            energy = scaleFactor[index] * (a0 + a1 * t + a2 * t * t + a3 * t * t * t);
            dEdR = scaleFactor[index] * (a1 + 2.0 * a2 * t + 3.0 * a3 * t * t) / binWidth;
        }

        // store force into local buffer
        float3 f;
        f.x = delta.x * dEdR / r;
        f.y = delta.y * dEdR / r;
        f.z = delta.z * dEdR / r;
        restraintForce[index] = f;

        // store energy into global buffer
        restraintEnergies[globalIndex] = energy;
    }
}


extern "C" __global__ void computeTorsProfileRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int4* __restrict__ atomIndices0,      // i,j,k,l for torsion 0
                            const int4* __restrict__ atomIndices1,      // i,j,k,l for torsion 1
                            const int* __restrict__ nBins,              // number of bins
                            const float4* __restrict__ params0,         // a0 - a3
                            const float4* __restrict__ params1,         // a4 - a7
                            const float4* __restrict__ params2,         // a8 - a11
                            const float4* __restrict__ params3,         // a12 - a15
                            const int2* __restrict__ paramBounds,       // upper and lower bounds for each spline
                            const float* __restrict__ scaleFactor,      // scale factor for energies and forces
                            const int* __restrict__ indexToGlobal,      // index of this restraint in the global array
                            float* __restrict__ restraintEnergies,      // global energy of each restraint
                            float3* __restrict__ forceBuffer,        // cache the forces for application later
                            const int numRestraints ) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=gridDim.x*blockDim.x) {
        // get my global index
        int globalIndex = indexToGlobal[index];

        // compute phi
        int phi_atom_i = atomIndices0[index].x;
        int phi_atom_j = atomIndices0[index].y;
        int phi_atom_k = atomIndices0[index].z;
        int phi_atom_l = atomIndices0[index].w;
        float3 phi_r_ij, phi_r_kj, phi_r_kl;
        float3 phi_m, phi_n;
        float phi_len_r_kj;
        float phi_len_m;
        float phi_len_n;
        float phi;
        computeTorsionAngle(posq, phi_atom_i, phi_atom_j, phi_atom_k, phi_atom_l,
                phi_r_ij, phi_r_kj, phi_r_kl, phi_m, phi_n, phi_len_r_kj, phi_len_m, phi_len_n, phi);

        // compute psi
        int psi_atom_i = atomIndices1[index].x;
        int psi_atom_j = atomIndices1[index].y;
        int psi_atom_k = atomIndices1[index].z;
        int psi_atom_l = atomIndices1[index].w;
        float3 psi_r_ij, psi_r_kj, psi_r_kl;
        float3 psi_m, psi_n;
        float psi_len_r_kj;
        float psi_len_m;
        float psi_len_n;
        float psi;
        computeTorsionAngle(posq, psi_atom_i, psi_atom_j, psi_atom_k, psi_atom_l,
                psi_r_ij, psi_r_kj, psi_r_kl, psi_m, psi_n, psi_len_r_kj, psi_len_m, psi_len_n, psi);

        // compute bin indices
        int i = (int)(floor((phi + 180.)/360. * nBins[index]));
        int j = (int)(floor((psi + 180.)/360. * nBins[index]));

        if (i >= nBins[index]) {
            i = 0;
            phi -= 360.;
        }
        if (i < 0) {
            i = nBins[index] - 1;
            phi += 360.;
        }

        if (j >= nBins[index]) {
            j = 0;
            psi -= 360.;
        }
        if (j < 0) {
            j = nBins[index] - 1;
            psi += 360.;
        }

        float delta = 360. / nBins[index];
        float u = (phi - i * delta + 180.) / delta;
        float v = (psi - j * delta + 180.) / delta;

        int pi = paramBounds[index].x + i * nBins[index] + j;

        float energy = params0[pi].x         + params0[pi].y * v       + params0[pi].z * v*v       + params0[pi].w * v*v*v +
                       params1[pi].x * u     + params1[pi].y * u*v     + params1[pi].z * u*v*v     + params1[pi].w * u*v*v*v +
                       params2[pi].x * u*u   + params2[pi].y * u*u*v   + params2[pi].z * u*u*v*v   + params2[pi].w * u*u*v*v*v +
                       params3[pi].x * u*u*u + params3[pi].y * u*u*u*v + params3[pi].z * u*u*u*v*v + params3[pi].w * u*u*u*v*v*v;
        energy = energy * scaleFactor[index];

        float dEdPhi = params1[pi].x         + params1[pi].y * v     + params1[pi].z * v*v     + params1[pi].w * v*v*v +
                       params2[pi].x * 2*u   + params2[pi].y * 2*u*v   + params2[pi].z * 2*u*v*v   + params2[pi].w * 2*u*v*v*v +
                       params3[pi].x * 3*u*u + params3[pi].y * 3*u*u*v + params3[pi].z * 3*u*u*v*v + params3[pi].w * 3*u*u*v*v*v;
        dEdPhi = dEdPhi * scaleFactor[index] / delta;

        float dEdPsi = params0[pi].y         + params0[pi].z * 2*v       + params0[pi].w * 3*v*v +
                       params1[pi].y * u     + params1[pi].z * u*2*v     + params1[pi].w * u*3*v*v +
                       params2[pi].y * u*u   + params2[pi].z * u*u*2*v   + params2[pi].w * u*u*3*v*v +
                       params3[pi].y * u*u*u + params3[pi].z * u*u*u*2*v + params3[pi].w * u*u*u*3*v*v;
        dEdPsi = dEdPsi * scaleFactor[index] / delta;

        restraintEnergies[globalIndex] = energy;

        computeTorsionForce(dEdPhi, phi_r_ij, phi_r_kj, phi_r_kl, phi_m, phi_n, phi_len_r_kj, phi_len_m, phi_len_n,
                forceBuffer[8 * index + 0], forceBuffer[8 * index + 1],
                forceBuffer[8 * index + 2], forceBuffer[8 * index + 3]);
        computeTorsionForce(dEdPsi, psi_r_ij, psi_r_kj, psi_r_kl, psi_m, psi_n, psi_len_r_kj, psi_len_m, psi_len_n,
                forceBuffer[8 * index + 4], forceBuffer[8 * index + 5],
                forceBuffer[8 * index + 6], forceBuffer[8 * index + 7]);
    }
}


extern "C" __global__ void computeGMMRest(
                            const real4* __restrict__ posq,             // positions and charges
                            const int numRestraints,                    // number of restraints
                            const int4* __restrict__ params,            // nPairs, nComponents, globalIndices
                            const int2* __restrict__ offsets,           // atomBlockOffset, dataBlockOffset
                            const int* __restrict__ atomIndices,        // atom indices
                            const float* __restrict__ data,             // weights, means, diags, offdiags
                            float* __restrict__ energies,               // global array of restraint energies
                            float3* __restrict__ forceBuffer) {         // temporary buffer to hold the force
    extern __shared__ volatile char scratch[];


    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    float* distances = (float*)&scratch[0];
    float* probabilities = (float*)&scratch[16*32*sizeof(float)];

    distances[tid] = 0.0;
    probabilities[tid] = 0.0;

    for (int index=16*blockIdx.x + warp; index<numRestraints; index+=16*gridDim.x) {
        int nPairs = params[index].x;
        int nComponents = params[index].y;
        int globalIndex = params[index].z;
        float scale = (float)(params[index].w) * 1e-6;

        int atomBlockOffset = offsets[index].x;
        int dataBlockOffset = offsets[index].y;

        // compute my distance
        if (lane < nPairs) {
            int atomIndex1 = atomIndices[atomBlockOffset + 2 * lane];
            int atomIndex2 = atomIndices[atomBlockOffset + 2 * lane + 1];

            real4 delta = posq[atomIndex1] - posq[atomIndex2];
            real distSquared = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
            real r = SQRT(distSquared);
            distances[tid] = r;
        }
        __syncthreads();


        float minsum = 9e99;

        // compute my probability
        const int blockSize = 1 + 2*nPairs + nPairs*(nPairs-1)/2;
        if (lane < nComponents) {
            // compute offsets into data array
            const float weight = data[dataBlockOffset + lane*blockSize];
            const float* means = &data[dataBlockOffset + lane*blockSize + 1];
            const float* diags = &data[dataBlockOffset + lane*blockSize + nPairs + 1];
            const float* offdiags = &data[dataBlockOffset + lane*blockSize + 2*nPairs + 1];
            float sum = 0;

            // do the diagonal part
            for (int i=0; i<nPairs; i++) {
                float mean = means[i];
                float diag = diags[i] * scale;
                float dist = distances[32 * warp + i];
                sum += (mean - dist) * (mean - dist) * diag;
            }

            // do the off diagonal part
            int count = 0;
            for (int i=0; i<nPairs; i++) {
                for (int j=i+1; j<nPairs; j++) {
                    float meani = means[i];
                    float meanj = means[j];
                    float coeff = 2 * offdiags[count] * scale;
                    float disti = distances[32 * warp + i];
                    float distj = distances[32 * warp + j];
                    sum += (disti - meani) * (distj - meanj) * coeff;
                    count++;
                }
            }
            probabilities[tid] = sum;

            __syncthreads();

            for (int i=0; i<nComponents; i++) {
                if(probabilities[32 * warp + i] < minsum) {
                    minsum = probabilities[32 * warp + i];
                }
            }
            __syncthreads();

            probabilities[tid] = weight * exp(-0.5 * (sum - minsum));
            __syncthreads();
        }

        // compute and store forces
        float totalProb = 0;
        for (int i=0; i<nComponents; i++) {
            totalProb += probabilities[32 * warp + i];
        }

        if (lane < nPairs) {
            float dEdr = 0;

            // compute diagonal part of force
            for (int i=0; i<nComponents; i++) {
                float distance = distances[32 * warp + lane];
                float mean = data[dataBlockOffset + i*blockSize + lane + 1];
                float diag = data[dataBlockOffset + i*blockSize + nPairs + lane + 1];
                dEdr += 2.48 * probabilities[32 * warp + i] / totalProb * (distance - mean) * diag * scale;
            }

            // compute off diagonal part of force
            for (int i=0; i<nComponents; i++) {
                for (int k=0; k<nPairs; k++) {
                    if (k != lane) {
                        float r = distances[32 * warp + k];
                        float mu = data[dataBlockOffset + i*blockSize + k + 1];
                        int coeffIndex = 0;
                        if (k > lane) {
                            coeffIndex = nPairs*(nPairs-1)/2 - (nPairs-lane)*((nPairs-lane)-1)/2 + k - lane - 1;
                        } else {
                            coeffIndex = nPairs*(nPairs-1)/2 - (nPairs-k)*((nPairs-k)-1)/2 + lane - k - 1;
                        }
                        float coeff = data[dataBlockOffset + i*blockSize + 1 + 2*nPairs + coeffIndex];
                        dEdr += 2.48 * probabilities[32 * warp + i] / totalProb * (r - mu) * coeff * scale;
                    }
                }
            }
            int atomIndex1 = atomIndices[atomBlockOffset + 2 * lane];
            int atomIndex2 = atomIndices[atomBlockOffset + 2 * lane + 1];
            real4 delta = posq[atomIndex1] - posq[atomIndex2];
            float4 f = dEdr * delta / distances[32 * warp + lane];
            forceBuffer[atomBlockOffset + lane].x = f.x;
            forceBuffer[atomBlockOffset + lane].y = f.y;
            forceBuffer[atomBlockOffset + lane].z = f.z;
        }

        // compute and store the energy
        if (lane == 0) {
            float energy = -2.48 * (log(totalProb) - 0.5 * minsum);
            energies[globalIndex] = energy;
        }
    }
}


extern "C" __global__ void computeGridPotentialRest(
                            const real4* __restrict__ posq, 
                            const int* __restrict__ atomIndices, 
                            const float* __restrict__ grid_x,
                            const float* __restrict__ grid_y,
                            const float* __restrict__ grid_z,
                            const float* __restrict__ mu,
                            const float* __restrict__ emap_weights,
                            const int* __restrict__ emapAtomList, 
                            int* __restrict__ indexToGlobal,
                            float* __restrict__ energies, 
                            float3* __restrict__ forceBuffer,
                            const int numRestraints,
                            const int3 numEmapGrids,
                            const int numEmapAtoms) {
        // set all emap restraints to 0 in the beginning
        for (int res=blockIdx.y*blockDim.y+threadIdx.y; res < numRestraints; res+=blockDim.y*gridDim.y) {
            int globalIndex = indexToGlobal[res];
            energies[globalIndex] = 0;
            __syncthreads();
        }
        // calculate force for each atom in all atom sets
        for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numEmapAtoms; index+=blockDim.x*gridDim.x) {
            int index_global;
            int mu_index;
            // determine atom is in which atom sets, then get globalIndex for store energy later
            for (int atom_list=blockIdx.y*blockDim.y+threadIdx.y; atom_list < numRestraints; atom_list+=blockDim.y*gridDim.y) {
                const int globalIndex = indexToGlobal[atom_list];
                if ((index - emapAtomList[atom_list] >= 0) && (index - emapAtomList[atom_list+1] < 0)) {
                    index_global = globalIndex;
                    mu_index = atom_list;
                }
                __syncthreads();
            }
            int atomIndex = atomIndices[index];
            float emap_weight = emap_weights[index]; // mass of the atom
            float3 f = make_float3(0,0,0);
            float3 atom_pos = trimTo3(posq[atomIndex]);
            int grid_xmax = numEmapGrids.x;
            int grid_ymax = numEmapGrids.y;
            int grid_zmax = numEmapGrids.z;
            int mu_num = grid_xmax*grid_ymax*grid_zmax; // size of map dimension to determine the atom corresponds to which grid potential set (mu) 
                                                        // if the input has multiple maps assuming all maps have the 
                                                        // same dimension but the potential on each grid could be different.
                                                        // mu shape (1,num_maps*numEmapGrids.x*numEmapGrids.y*numEmapGrids.z)
            // check the atom is in which grid
            int grid_xnum = floor((atom_pos.x-grid_x[0])/(grid_x[1]-grid_x[0])) + 1; 
            int grid_ynum = floor((atom_pos.y-grid_y[0])/(grid_y[1]-grid_y[0])) + 1;
            int grid_znum = floor((atom_pos.z-grid_z[0])/(grid_z[1]-grid_z[0])) + 1;
            // scale atom position with grid length
            float grid_x_pos = (grid_x[grid_xnum]-atom_pos.x)/(grid_x[1]-grid_x[0]); 
            float grid_y_pos = (grid_y[grid_ynum]-atom_pos.y)/(grid_y[1]-grid_y[0]);
            float grid_z_pos = (grid_z[grid_znum]-atom_pos.z)/(grid_z[1]-grid_z[0]);
            float grid_xpos = (atom_pos.x-grid_x[grid_xnum-1])/(grid_x[1]-grid_x[0]);
            float grid_ypos = (atom_pos.y-grid_y[grid_ynum-1])/(grid_y[1]-grid_y[0]);
            float grid_zpos = (atom_pos.z-grid_z[grid_znum-1])/(grid_z[1]-grid_z[0]);
            float energy = 0;
            float f_x = 0;
            float f_y = 0;
            float f_z = 0;
            // linear interpolation
            // get potential at 8 grids around the atom
            float v_000 = mu[mu_num*mu_index+(grid_znum-1) * grid_ymax * grid_xmax + (grid_ynum-1) * grid_xmax + grid_xnum -1];
            float v_100 = mu[mu_num*mu_index+(grid_znum-1) * grid_ymax * grid_xmax + (grid_ynum-1) * grid_xmax + grid_xnum];
            float v_010 = mu[mu_num*mu_index+(grid_znum-1) * grid_ymax * grid_xmax + grid_ynum * grid_xmax + grid_xnum -1];
            float v_001 = mu[mu_num*mu_index+grid_znum * grid_ymax * grid_xmax + (grid_ynum-1) * grid_xmax + grid_xnum -1];
            float v_101 = mu[mu_num*mu_index+grid_znum * grid_ymax * grid_xmax + (grid_ynum-1) * grid_xmax + grid_xnum];
            float v_011 = mu[mu_num*mu_index+grid_znum * grid_ymax * grid_xmax + grid_ynum * grid_xmax + grid_xnum -1];
            float v_110 = mu[mu_num*mu_index+(grid_znum-1) * grid_ymax * grid_xmax + grid_ynum * grid_xmax + grid_xnum ];
            float v_111 = mu[mu_num*mu_index+grid_znum * grid_ymax * grid_xmax + grid_ynum * grid_xmax + grid_xnum];
            energy += emap_weight * (v_000 * grid_x_pos * grid_y_pos * grid_z_pos              
                    + v_100 * grid_xpos * grid_y_pos * grid_z_pos 
                    + v_010 * grid_x_pos * grid_ypos * grid_z_pos   
                    + v_001 * grid_x_pos * grid_y_pos * grid_zpos
                    + v_101 * grid_xpos * grid_y_pos * grid_zpos 
                    + v_011 * grid_x_pos * grid_ypos * grid_zpos
                    + v_110 * grid_xpos * grid_ypos * grid_z_pos           
                    + v_111 * grid_xpos * grid_ypos * grid_zpos)  ;
            f_x += -1 * emap_weight * ((v_100 - v_000) * grid_y_pos * grid_z_pos 
                    + (v_110 - v_010) * grid_ypos * grid_z_pos
                    + (v_101 - v_001) * grid_y_pos * grid_zpos
                    + (v_111 - v_011) * grid_ypos * grid_zpos)/(grid_x[1]-grid_x[0])   ;
                    
            f_y += -1 * emap_weight * ((v_010 - v_000) * grid_x_pos * grid_z_pos 
                    + (v_110 - v_100) * grid_xpos * grid_z_pos
                    + (v_011 - v_001) * grid_x_pos * grid_zpos
                    + (v_111 - v_101) * grid_xpos * grid_zpos)/(grid_x[1]-grid_x[0])  ;

            f_z += -1 * emap_weight * ((v_001 - v_000) * grid_x_pos * grid_y_pos 
                    + (v_101 - v_100) * grid_xpos * grid_y_pos
                    + (v_011 - v_010) * grid_x_pos * grid_ypos
                    + (v_111 - v_110) * grid_xpos * grid_ypos)/(grid_x[1]-grid_x[0])   ;

            forceBuffer[index] = make_float3(f_x,f_y,f_z);
            energies[index_global] += energy;
            __syncthreads();
        }

}



extern "C" __global__ void evaluateAndActivate(
        const int numGroups,
        const int* __restrict__ numActiveArray,
        const int2* __restrict__ boundsArray,
        const int* __restrict__ indexArray,
        const float* __restrict__ energyArray,
        float* __restrict__ activeArray,
        float* __restrict__ groupEnergyArray)
{
    // Setup type alias for collective operations
    typedef cub::BlockRadixSort<float, NGROUPTHREADS, RESTS_PER_THREAD> BlockRadixSortT;
    typedef cub::BlockReduce<float, NGROUPTHREADS> BlockReduceT;

    // Setup shared memory for sorting.
    __shared__ union {
        typename BlockRadixSortT::TempStorage sort;
        typename BlockReduceT::TempStorage reduce;
        float cutoff;
    } sharedScratch;

    // local storage for energies to be sorted
    float energyScratch[RESTS_PER_THREAD];

    for (int groupIndex=blockIdx.x; groupIndex<numGroups; groupIndex+=gridDim.x) {
        int numActive = numActiveArray[groupIndex];
        int start = boundsArray[groupIndex].x;
        int end = boundsArray[groupIndex].y;

        // Load energies into statically allocated scratch buffer
        for(int i=0; i<RESTS_PER_THREAD; i++) {
            int index = threadIdx.x * RESTS_PER_THREAD + start + i;
            if(index < end) {
                energyScratch[i] = energyArray[indexArray[index]];
            } else {
                energyScratch[i] = FLT_MAX;
            }
        }
        __syncthreads();

        // Sort the energies.
        BlockRadixSortT(sharedScratch.sort).Sort(energyScratch);
        __syncthreads();

        // find the nth largest energy and store in scratch
        int myMin = threadIdx.x * RESTS_PER_THREAD;
        int myMax = myMin + RESTS_PER_THREAD;
        if((numActive - 1) >= myMin) {
            if((numActive - 1) < myMax) {
                // only one thread will get here
                int offset = numActive - 1 - myMin;
                sharedScratch.cutoff = energyScratch[offset];
            }
        }
        __syncthreads();

        // Read the nth largest energy from shared memory.
        float cutoff = (volatile float)sharedScratch.cutoff;
        __syncthreads();

        // now we know the cutoff, so apply it to each group and
        // load each energy into a scratch buffer.
        for(int i=0; i<RESTS_PER_THREAD; i++) {
            int index = threadIdx.x * RESTS_PER_THREAD + start + i;
            if(index < end) {
                if (energyArray[indexArray[index]] <= cutoff) {
                    activeArray[indexArray[index]] = 1.0;
                    energyScratch[i] = energyArray[indexArray[index]];

                } else {
                    activeArray[indexArray[index]] = 0.0;
                    energyScratch[i] = 0.0;
                }
            } else {
                energyScratch[i] = 0.0;
            }
        }
        __syncthreads();

        // Now sum all of the energies to get the total energy
        // for the group.
        float totalEnergy = BlockReduceT(sharedScratch.reduce).Sum(energyScratch);
        if(threadIdx.x == 0) {
            groupEnergyArray[groupIndex] = totalEnergy;
        }
        __syncthreads();
    }
}


extern "C" __global__ void evaluateAndActivateCollections(
        const int numCollections,
        const int* __restrict__ numActiveArray,
        const int2* __restrict__ boundsArray,
        const int* __restrict__ indexArray,
        const float* __restrict__ energyArray,
        float* __restrict__ activeArray)
{
    // Setup type alias for sorting.
    typedef cub::BlockRadixSort<float, NCOLLTHREADS, GROUPS_PER_THREAD> BlockRadixSortT;

    // Setup shared memory for sorting.
    __shared__ union {
        typename BlockRadixSortT::TempStorage sort;
        float cutoff;
    } sharedScratch;

    // local storage for energies to be sorted
    float energyScratch[GROUPS_PER_THREAD];

    for (int collIndex=blockIdx.x; collIndex<numCollections; collIndex+=gridDim.x) {
        int numActive = numActiveArray[collIndex];
        int start = boundsArray[collIndex].x;
        int end = boundsArray[collIndex].y;

        // Load energies into statically allocated scratch buffer
        for(int i=0; i<GROUPS_PER_THREAD; i++) {
            int index = threadIdx.x * GROUPS_PER_THREAD + start + i;
            if(index < end) {
                energyScratch[i] = energyArray[indexArray[index]];
            } else {
                energyScratch[i] = FLT_MAX;
            }
        }
        __syncthreads();

        // Sort the energies.
        BlockRadixSortT(sharedScratch.sort).Sort(energyScratch);
        __syncthreads();

        // find the nth largest energy and store in scratch
        int myMin = threadIdx.x * GROUPS_PER_THREAD;
        int myMax = myMin + GROUPS_PER_THREAD;
        if((numActive - 1) >= myMin) {
            if((numActive - 1) < myMax) {
                // only one thread will get here
                int offset = numActive - 1 - myMin;
                sharedScratch.cutoff = energyScratch[offset];
            }
        }
        __syncthreads();

        // Read the nth largest energy from shared memory.
        float cutoff = (volatile float)sharedScratch.cutoff;

        // now we know the cutoff, so apply it to each group
        for (int i=start + threadIdx.x; i<end; i+=blockDim.x) {
            if (energyArray[indexArray[i]] <= cutoff) {
                activeArray[indexArray[i]] = 1.0;
            }
            else {
                activeArray[indexArray[i]] = 0.0;
            }
        }
        __syncthreads();
    }
}


extern "C" __global__ void applyGroups(
                            float* __restrict__ groupActive,
                            float* __restrict__ restraintActive,
                            const int2* __restrict__ bounds,
                            int numGroups) {
    for (int groupIndex=blockIdx.x; groupIndex<numGroups; groupIndex+=gridDim.x) {
        float active = groupActive[groupIndex];
        for (int i=bounds[groupIndex].x + threadIdx.x; i<bounds[groupIndex].y; i+=blockDim.x) {
            restraintActive[i] *= active;
        }
    }
}

extern "C" __global__ void applyRDCRest(
    unsigned long long* __restrict__ force,
    mixed* __restrict__ energyBuffer,
    mixed* __restrict__ derivBuffer,
    const int2* __restrict__ rdcRestAtomIndices,
    const int* __restrict__ rdcRestAlignments,
    const int* __restrict__ rdcRestGlobalIndices,
    const float3* __restrict__ rdcRestForces,
    const float* __restrict__ rdcRestDerivs,
    const int* __restrict__ derivIndices,
    const float* __restrict__ rdcRestEnergies,
    const float* __restrict__ restraintActive,
    int numRDCRestraints
) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<numRDCRestraints; i+=blockDim.x*gridDim.x)
    {
        int index = rdcRestGlobalIndices[i];
        if (restraintActive[index])
        {
            if (rdcRestAtomIndices[i].x == -1)
            {
                // Do nothing. This restraint is marked as being
                // not mapped, so it contributes no energy or force.
            }
            else {
                energyAccum += rdcRestEnergies[index];

                float fx = rdcRestForces[i].x;
                float fy = rdcRestForces[i].y;
                float fz = rdcRestForces[i].z;
                float ds1 = rdcRestDerivs[5 * i + 0];
                float ds2 = rdcRestDerivs[5 * i + 1];
                float ds3 = rdcRestDerivs[5 * i + 2];
                float ds4 = rdcRestDerivs[5 * i + 3];
                float ds5 = rdcRestDerivs[5 * i + 4];
                int atom1 = rdcRestAtomIndices[i].x;
                int atom2 = rdcRestAtomIndices[i].y;

                atomicAdd(&force[atom1], static_cast<unsigned long long>((long long) (-fx*0x100000000)));
                atomicAdd(&force[atom1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-fy*0x100000000)));
                atomicAdd(&force[atom1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-fz*0x100000000)));

                atomicAdd(&force[atom2], static_cast<unsigned long long>((long long) (fx*0x100000000)));
                atomicAdd(&force[atom2  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (fy*0x100000000)));
                atomicAdd(&force[atom2 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (fz*0x100000000)));

                // Update parameter derivatives
                int alignment = rdcRestAlignments[i];
                derivBuffer[threadIndex * NUM_DERIVS + derivIndices[5 * alignment + 0]] += ds1;
                derivBuffer[threadIndex * NUM_DERIVS + derivIndices[5 * alignment + 1]] += ds2;
                derivBuffer[threadIndex * NUM_DERIVS + derivIndices[5 * alignment + 2]] += ds3;
                derivBuffer[threadIndex * NUM_DERIVS + derivIndices[5 * alignment + 3]] += ds4;
                derivBuffer[threadIndex * NUM_DERIVS + derivIndices[5 * alignment + 4]] += ds5;
            }
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}

extern "C" __global__ void applyDistRest(
                                unsigned long long * __restrict__ force,
                                mixed* __restrict__ energyBuffer,
                                const int2* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numDistRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numDistRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int index1 = atomIndices[restraintIndex].x;
            int index2 = atomIndices[restraintIndex].y;
            if (index1 == -1) {
                // Do nothing. This restraint is marked as being
                // not mapped, so it contributes no energy or force.
            } else {
                energyAccum += globalEnergies[globalIndex];
                float3 f = restForces[restraintIndex];

                atomicAdd(&force[index1], static_cast<unsigned long long>((long long) (-f.x*0x100000000)));
                atomicAdd(&force[index1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.y*0x100000000)));
                atomicAdd(&force[index1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.z*0x100000000)));

                atomicAdd(&force[index2], static_cast<unsigned long long>((long long) (f.x*0x100000000)));
                atomicAdd(&force[index2  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.y*0x100000000)));
                atomicAdd(&force[index2 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.z*0x100000000)));
            }
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


extern "C" __global__ void applyHyperbolicDistRest(
                                unsigned long long * __restrict__ force,
                                mixed* __restrict__ energyBuffer,
                                const int2* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numDistRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numDistRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int index1 = atomIndices[restraintIndex].x;
            int index2 = atomIndices[restraintIndex].y;
            energyAccum += globalEnergies[globalIndex];
            float3 f = restForces[restraintIndex];

            atomicAdd(&force[index1], static_cast<unsigned long long>((long long) (-f.x*0x100000000)));
            atomicAdd(&force[index1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.y*0x100000000)));
            atomicAdd(&force[index1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.z*0x100000000)));

            atomicAdd(&force[index2], static_cast<unsigned long long>((long long) (f.x*0x100000000)));
            atomicAdd(&force[index2  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.y*0x100000000)));
            atomicAdd(&force[index2 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


extern "C" __global__ void applyTorsionRest(
                                unsigned long long * __restrict__ force,
                                mixed* __restrict__ energyBuffer,
                                const int4* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int atom_i = atomIndices[restraintIndex].x;
            int atom_j = atomIndices[restraintIndex].y;
            int atom_k = atomIndices[restraintIndex].z;
            int atom_l = atomIndices[restraintIndex].w;
            energyAccum += globalEnergies[globalIndex];

            // update forces
            float3 f_i = restForces[restraintIndex * 4 + 0];
            float3 f_j = restForces[restraintIndex * 4 + 1];
            float3 f_k = restForces[restraintIndex * 4 + 2];
            float3 f_l = restForces[restraintIndex * 4 + 3];

            atomicAdd(&force[atom_i],                        static_cast<unsigned long long>((long long) (f_i.x*0x100000000)));
            atomicAdd(&force[atom_i  + PADDED_NUM_ATOMS],    static_cast<unsigned long long>((long long) (f_i.y*0x100000000)));
            atomicAdd(&force[atom_i + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f_i.z*0x100000000)));

            atomicAdd(&force[atom_j],                        static_cast<unsigned long long>((long long) (f_j.x*0x100000000)));
            atomicAdd(&force[atom_j  + PADDED_NUM_ATOMS],    static_cast<unsigned long long>((long long) (f_j.y*0x100000000)));
            atomicAdd(&force[atom_j + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f_j.z*0x100000000)));

            atomicAdd(&force[atom_k],                        static_cast<unsigned long long>((long long) (f_k.x*0x100000000)));
            atomicAdd(&force[atom_k  + PADDED_NUM_ATOMS],    static_cast<unsigned long long>((long long) (f_k.y*0x100000000)));
            atomicAdd(&force[atom_k + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f_k.z*0x100000000)));

            atomicAdd(&force[atom_l],                        static_cast<unsigned long long>((long long) (f_l.x*0x100000000)));
            atomicAdd(&force[atom_l  + PADDED_NUM_ATOMS],    static_cast<unsigned long long>((long long) (f_l.y*0x100000000)));
            atomicAdd(&force[atom_l + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f_l.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


extern "C" __global__ void applyDistProfileRest(
                                unsigned long long * __restrict__ force,
                                mixed* __restrict__ energyBuffer,
                                const int2* __restrict__ atomIndices,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            int index1 = atomIndices[restraintIndex].x;
            int index2 = atomIndices[restraintIndex].y;
            energyAccum += globalEnergies[globalIndex];
            float3 f = restForces[restraintIndex];

            atomicAdd(&force[index1], static_cast<unsigned long long>((long long) (-f.x*0x100000000)));
            atomicAdd(&force[index1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.y*0x100000000)));
            atomicAdd(&force[index1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.z*0x100000000)));

            atomicAdd(&force[index2], static_cast<unsigned long long>((long long) (f.x*0x100000000)));
            atomicAdd(&force[index2  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.y*0x100000000)));
            atomicAdd(&force[index2 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}


extern "C" __global__ void applyTorsProfileRest(
                                unsigned long long * __restrict__ force,
                                mixed* __restrict__ energyBuffer,
                                const int4* __restrict__ atomIndices0,
                                const int4* __restrict__ atomIndices1,
                                const int* __restrict__ globalIndices,
                                const float3* __restrict__ restForces,
                                const float* __restrict__ globalEnergies,
                                const float* __restrict__ globalActive,
                                const int numRestraints) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;

    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            // update energy
            energyAccum += globalEnergies[globalIndex];

            // update phi
            int phi_atom_i = atomIndices0[restraintIndex].x;
            int phi_atom_j = atomIndices0[restraintIndex].y;
            int phi_atom_k = atomIndices0[restraintIndex].z;
            int phi_atom_l = atomIndices0[restraintIndex].w;

            // update forces
            float3 phi_f_i = restForces[restraintIndex * 8 + 0];
            float3 phi_f_j = restForces[restraintIndex * 8 + 1];
            float3 phi_f_k = restForces[restraintIndex * 8 + 2];
            float3 phi_f_l = restForces[restraintIndex * 8 + 3];

            atomicAdd(&force[phi_atom_i],                        static_cast<unsigned long long>((long long) (phi_f_i.x*0x100000000)));
            atomicAdd(&force[phi_atom_i + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (phi_f_i.y*0x100000000)));
            atomicAdd(&force[phi_atom_i + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (phi_f_i.z*0x100000000)));

            atomicAdd(&force[phi_atom_j],                        static_cast<unsigned long long>((long long) (phi_f_j.x*0x100000000)));
            atomicAdd(&force[phi_atom_j + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (phi_f_j.y*0x100000000)));
            atomicAdd(&force[phi_atom_j + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (phi_f_j.z*0x100000000)));

            atomicAdd(&force[phi_atom_k],                        static_cast<unsigned long long>((long long) (phi_f_k.x*0x100000000)));
            atomicAdd(&force[phi_atom_k + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (phi_f_k.y*0x100000000)));
            atomicAdd(&force[phi_atom_k + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (phi_f_k.z*0x100000000)));

            atomicAdd(&force[phi_atom_l],                        static_cast<unsigned long long>((long long) (phi_f_l.x*0x100000000)));
            atomicAdd(&force[phi_atom_l + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (phi_f_l.y*0x100000000)));
            atomicAdd(&force[phi_atom_l + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (phi_f_l.z*0x100000000)));

            // update psi
            int psi_atom_i = atomIndices1[restraintIndex].x;
            int psi_atom_j = atomIndices1[restraintIndex].y;
            int psi_atom_k = atomIndices1[restraintIndex].z;
            int psi_atom_l = atomIndices1[restraintIndex].w;

            // update forces
            float3 psi_f_i = restForces[restraintIndex * 8 + 4];
            float3 psi_f_j = restForces[restraintIndex * 8 + 5];
            float3 psi_f_k = restForces[restraintIndex * 8 + 6];
            float3 psi_f_l = restForces[restraintIndex * 8 + 7];

            atomicAdd(&force[psi_atom_i],                        static_cast<unsigned long long>((long long) (psi_f_i.x*0x100000000)));
            atomicAdd(&force[psi_atom_i + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (psi_f_i.y*0x100000000)));
            atomicAdd(&force[psi_atom_i + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (psi_f_i.z*0x100000000)));

            atomicAdd(&force[psi_atom_j],                        static_cast<unsigned long long>((long long) (psi_f_j.x*0x100000000)));
            atomicAdd(&force[psi_atom_j + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (psi_f_j.y*0x100000000)));
            atomicAdd(&force[psi_atom_j + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (psi_f_j.z*0x100000000)));

            atomicAdd(&force[psi_atom_k],                        static_cast<unsigned long long>((long long) (psi_f_k.x*0x100000000)));
            atomicAdd(&force[psi_atom_k + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (psi_f_k.y*0x100000000)));
            atomicAdd(&force[psi_atom_k + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (psi_f_k.z*0x100000000)));

            atomicAdd(&force[psi_atom_l],                        static_cast<unsigned long long>((long long) (psi_f_l.x*0x100000000)));
            atomicAdd(&force[psi_atom_l + PADDED_NUM_ATOMS],     static_cast<unsigned long long>((long long) (psi_f_l.y*0x100000000)));
            atomicAdd(&force[psi_atom_l + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (psi_f_l.z*0x100000000)));
        }
    }
    energyBuffer[threadIndex] += energyAccum;
}

extern "C" __global__ void applyGMMRest(unsigned long long * __restrict__ force,
                                        mixed* __restrict__ energyBuffer,
                                        const int numRestraints,
                                        const int4* __restrict params,
                                        const float* __restrict__ globalEnergies,
                                        const float* __restrict__ globalActive,
                                        const int2* __restrict__ offsets,
                                        const int* __restrict__ atomIndices,
                                        const float3* __restrict__ restForces) {

    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    for (int index=16*blockIdx.x + warp; index<numRestraints; index+=16*gridDim.x) {
        int nPairs = params[index].x;
        int globalIndex = params[index].z;
        int atomBlockOffset = offsets[index].x;

        if (globalActive[globalIndex]) {
            // add the forces
            if (lane < nPairs) {
                float3 f = restForces[atomBlockOffset + lane];
                int atomIndex1 = atomIndices[atomBlockOffset + 2 * lane];
                int atomIndex2 = atomIndices[atomBlockOffset + 2 * lane + 1];

                atomicAdd(&force[atomIndex1], static_cast<unsigned long long>((long long) (-f.x*0x100000000)));
                atomicAdd(&force[atomIndex1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.y*0x100000000)));
                atomicAdd(&force[atomIndex1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-f.z*0x100000000)));

                atomicAdd(&force[atomIndex2], static_cast<unsigned long long>((long long) (f.x*0x100000000)));
                atomicAdd(&force[atomIndex2  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.y*0x100000000)));
                atomicAdd(&force[atomIndex2 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.z*0x100000000)));
            }

            // add the energy
            if (lane == 0) {
                energyBuffer[tid] += globalEnergies[globalIndex];
            }
        }
    }
}


extern "C" __global__ void applyGridPotentialRest(unsigned long long * __restrict__ force,
                                         mixed* __restrict__ energyBuffer,
                                         const int* __restrict__ atomIndices,
                                         const int* __restrict__ emapAtomList, 
                                         const int* __restrict__ globalIndices,
                                         const float* __restrict__ globalEnergies,
                                         const float* __restrict__ globalActive,
                                         const float3* __restrict__ restForces,
                                         const int numEmapRestraints,
                                         const int numEmapAtoms) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float energyAccum = 0.0;
    // add force to each atom if the restraint it belongs to is active
    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numEmapAtoms; restraintIndex+=blockDim.x*gridDim.x) {
        int index1 = atomIndices[restraintIndex];
        float3 f = restForces[restraintIndex];
        for (int atom_list=blockIdx.y*blockDim.y+threadIdx.y; atom_list < numEmapRestraints; atom_list+=blockDim.y*gridDim.y) {
            const int globalIndex = globalIndices[atom_list];
            if ((restraintIndex - emapAtomList[atom_list] >= 0) && (restraintIndex - emapAtomList[atom_list+1] < 0) && (globalActive[globalIndex])) {
                atomicAdd(&force[index1], static_cast<unsigned long long>((long long) (f.x*0x100000000)));
                atomicAdd(&force[index1  + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.y*0x100000000)));
                atomicAdd(&force[index1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (f.z*0x100000000)));
            }
        }
    }
    for (int restraintIndex=blockIdx.x*blockDim.x+threadIdx.x; restraintIndex<numEmapRestraints; restraintIndex+=blockDim.x*gridDim.x) {
        int globalIndex = globalIndices[restraintIndex];
        if (globalActive[globalIndex]) {
            energyAccum += globalEnergies[globalIndex];
        }
    }
    energyBuffer[threadIndex] += energyAccum;
    float3 f = restForces[threadIndex];
}
