/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#define ELEM_SWAP(a,b) { int t=(a);(a)=(b);(b)=t; }

__device__ float quick_select_float(const float* energy, int *index, int nelems, int select) {
    int low, high, middle, ll, hh;

    low = 0;
    high = nelems - 1;

    for (;;) {
        if (high <= low) { /* One element only */
            return energy[index[select]];
        }

        if (high == low + 1) {  /* Two elements only */
            if (energy[index[low]] > energy[index[high]])
                ELEM_SWAP(index[low], index[high]);
            return energy[index[select]];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (energy[index[middle]] > energy[index[high]])    ELEM_SWAP(index[middle], index[high]);
        if (energy[index[low]]    > energy[index[high]])    ELEM_SWAP(index[low],    index[high]);
        if (energy[index[middle]] > energy[index[low]])     ELEM_SWAP(index[middle], index[low]);

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(index[middle], index[low+1]);

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (energy[index[low]] > energy[index[ll]]);
            do hh--; while (energy[index[hh]]  > energy[index[low]]);

            if (hh < ll)
                break;

            ELEM_SWAP(index[ll], index[hh]);
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(index[low], index[hh]);

        /* Re-set active partition */
        if (hh <= select)
            low = ll;
        if (hh >= select)
            high = hh - 1;
    }
}
#undef ELEM_SWAP


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


extern "C" __global__ void evaluateAndActivate(
        const int numGroups,
        const int* __restrict__ numActiveArray,
        const int2* __restrict__ boundsArray,
        const int* __restrict__ pristineIndexArray,
        int* __restrict__ tempIndexArray,
        const float* __restrict__ energyArray,
        float* __restrict__ activeArray,
        float* __restrict__ targetEnergyArray)
{
    // This kernel computes which restraints are active within each group.
    // It uses "warp-level" programming to do this, where each warp within
    // a threadblock computes the results for a single group. All threads
    // within each group are implicity synchronized at the hardware
    // level.

    // These are runtime parameters set tby the C++ code.
    const int groupsPerBlock = GROUPSPERBLOCK;
    const int maxGroupSize = MAXGROUPSIZE;

    // Because each warp is computing a separate interaction, we need to
    // keep track of which block we are acting on and our index within
    // that warp.
    const int groupOffsetInBlock = threadIdx.x / 32;
    const int threadOffsetInWarp = threadIdx.x % 32;

    // We store the energies and indices into scratch buffers. These scratch
    // buffers are also used for reductions within each warp.
    extern __shared__ volatile char scratch[];
    volatile float* warpScratchEnergy = (float*)&scratch[groupOffsetInBlock*maxGroupSize*(sizeof(float)+sizeof(int))];
    volatile int* warpScratchIndices = (int*)&scratch[groupOffsetInBlock*maxGroupSize*(sizeof(float)+sizeof(int)) +
                                                      maxGroupSize*sizeof(float)];
    volatile float* warpReductionBuffer = (float*)&scratch[groupOffsetInBlock*32*sizeof(float)];

    // each warp loads the energies and indices for a group
    for (int groupIndex=groupsPerBlock*blockIdx.x+groupOffsetInBlock; groupIndex<numGroups; groupIndex+=groupsPerBlock*gridDim.x) {
        const int numActive = numActiveArray[groupIndex];
        const int start = boundsArray[groupIndex].x;
        const int end = boundsArray[groupIndex].y;
        const int length = end - start;
        const bool applyAll = (numActive == length);

        // copy the energies to shared memory and setup indices
        if (!applyAll) {
            for(int i=threadOffsetInWarp; i<length; i+=32) {
                const float energy = energyArray[pristineIndexArray[i + start]];
                warpScratchIndices[i] = i;
                warpScratchEnergy[i] = energy;
            }
        }

        // now, we run the quick select algorithm.
        // this is not parallelized, so we only run it on one thread
        // per block.
        if (threadOffsetInWarp==0) {
            float energyCut = 0.0;
            if (!applyAll) {
                energyCut = quick_select_float((const float*)warpScratchEnergy, (int *)warpScratchIndices, length, numActive-1);
            }
            else {
                energyCut = 9.99e99;
            }
            warpScratchEnergy[0] = energyCut;
        }


        // now we're back on all threads again
        float energyCut = warpScratchEnergy[0];
        float thisActive = 0.0;
        float thisEnergy = 0.0;

        // we are going to start writing to warpReductionBuffer,
        // which may overlap with the warpScratch* buffers, so
        // we need to make sure that all threads are done first.
        __syncthreads();

        // reset the reduction buffers to zero
        warpReductionBuffer[threadOffsetInWarp] = 0.0;

        // sum up the energy for each restraint
        for(int i=threadOffsetInWarp+start; i<end; i+=32) {
            thisEnergy = energyArray[pristineIndexArray[i]];
            thisActive = (float)(thisEnergy <= energyCut);
            activeArray[pristineIndexArray[i]] = thisActive;
            warpReductionBuffer[threadOffsetInWarp] += thisActive * thisEnergy;
        }

        // now we do a parallel reduction within each warp
        int totalThreads = 32;
        int index2 = 0;
        while (totalThreads > 1) {
            int halfPoint = (totalThreads >> 1);
            if (threadOffsetInWarp < halfPoint) {
                index2 = threadOffsetInWarp + halfPoint;
                warpReductionBuffer[threadOffsetInWarp] += warpReductionBuffer[index2];
            }
            totalThreads = halfPoint;
        }

        // now store the energy for this group
        if (threadOffsetInWarp == 0) {
            targetEnergyArray[groupIndex] = warpReductionBuffer[0];
        }

        // make sure we're all done before we start again
        __syncthreads();
    }
}


__device__ void findMinMax(int length, const float* energyArray, float* minBuffer, float* maxBuffer) {
    const int tid = threadIdx.x;
    float energy;
    float min = 9.9e99;
    float max = -9.9e99;
    // Each thread computes the min and max for it's energies and stores them in the buffers
    for (int i=tid; i<length; i+=blockDim.x) {
        energy = energyArray[i];
        if (energy < min) {
            min = energy;
        }
        if (energy > max) {
            max = energy;
        }
    }
    minBuffer[tid] = min;
    maxBuffer[tid] = max;
    __syncthreads();

    // Now we do a parallel reduction
    int totalThreads = blockDim.x;
    int index2 = 0;
    float temp = 0;
    while (totalThreads > 1) {
        int halfPoint = (totalThreads >> 1);
        if (tid < halfPoint) {
            index2 = tid + halfPoint;
            temp = minBuffer[index2];
            if (temp < minBuffer[tid]) {
                minBuffer[tid] = temp;
            }
            temp = maxBuffer[index2];
            if (temp > maxBuffer[tid]) {
                maxBuffer[tid] = temp;
            }
        }
        __syncthreads();
        totalThreads = halfPoint;
    }
    __syncthreads();
}


extern "C" __global__ void evaluateAndActivateCollections(
        const int numCollections,
        const int* __restrict__ numActiveArray,
        const int2* __restrict__ boundsArray,
        const int* __restrict__ indexArray,
        const float* __restrict__ energyArray,
        float* __restrict__ activeArray,
        int * __restrict__ encounteredNaN)
{
    const float TOLERANCE = 1e-4;
    const int maxCollectionSize = MAXCOLLECTIONSIZE;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;  // which thread are we within this warp

    // shared memory:
    // energyBuffer: maxCollectionSize floats
    // min/max Buffer: gridDim.x floats
    // binCounts: gridDim.x ints
    extern __shared__ char collectionScratch[];
    float* energyBuffer = (float*)&collectionScratch[0];
    float* minBuffer = (float*)&collectionScratch[maxCollectionSize*sizeof(float)];
    float* maxBuffer = (float*)&collectionScratch[(maxCollectionSize+blockDim.x)*sizeof(float)];
    int* binCounts = (int*)&collectionScratch[(maxCollectionSize+2*blockDim.x)*sizeof(float)];
    int* bestBin = (int*)&(collectionScratch[(maxCollectionSize + 2 * blockDim.x) * sizeof(float) +
                                             blockDim.x * sizeof(int)]);

    for (int collIndex=blockIdx.x; collIndex<numCollections; collIndex+=gridDim.x) {
        // we need to find the value of the cutoff energy below, then we will
        // activate all groups with lower energy
        float energyCutoff = 0.0;

        int numActive = numActiveArray[collIndex];
        int start = boundsArray[collIndex].x;
        int end = boundsArray[collIndex].y;
        int length = end - start;

        // load the energy buffer for this collection
        for (int i=tid; i<length; i+=blockDim.x) {
            const float energy = energyArray[indexArray[start + i]];
            energyBuffer[i] = energy;
        }
        __syncthreads();

        findMinMax(length, energyBuffer, minBuffer, maxBuffer);
        float min = minBuffer[0];
        float max = maxBuffer[0];
        float delta = max - min;


        // All of the energies are the same, so they should all be active.
        // Note: we need to break out here in this case, as otherwise delta
        // will be zero and bad things will happen
        if (fabs(max-min) < TOLERANCE) {
            energyCutoff = max;
        } else {
            // Here we need to find the k'th highest energy. We do this using a recursive,
            // binning and counting strategy. We divide the interval (min, max) into blockDim.x
            // bins. We assign each energy to a bin, increment the count, and update
            // the min and max. Then, we find the bin that contains the k'th lowest energy. If
            // min==max for this bin, then we are done. Otherwise, we set the new (min, max) for
            // the bins and recompute, assigning energies less than min to bin 0.

            // loop until we break out at convergence
            for (;;) {

                // check to see if have encountered NaN, which will
                // result in an infinite loop
                if(tid==0) {
                    if (!isfinite(min) || !isfinite(max)) {
                        *encounteredNaN = 1;
                    }
                }
                // zero out the buffers
                binCounts[tid] = 0;
                minBuffer[tid] = 9.0e99;
                maxBuffer[tid] = 0.0;
                __syncthreads();

                // If we hit a NaN then abort early now that encounteredNaN is set.
                // This will cause an exception on the C++ side
                if (*encounteredNaN) {
                    return;
                }

                // loop over all energies
                for (int i=tid; i<length; i+=blockDim.x) {
                    float energy = energyBuffer[i];
                    // compute which bin this energy lies in
                    int index = float2int(floorf((blockDim.x-1) / delta * (energy - min)));

                    // we only count entries that lie within min and max
                    if ( (index >= 0) && (index < blockDim.x) ) {

                        // increment the counter using atomic function
                        atomicAdd(&binCounts[index], 1);
                        // update the min and max bounds for the bin using atomic functions
                        // note we need to cast to an integer, but floating point values
                        // still compare correctly when represented as integers
                        // this assumes that all energies are >0
                        atomicMin((unsigned int*)&minBuffer[index], __float_as_int(energy));
                        atomicMax((unsigned int*)&maxBuffer[index], __float_as_int(energy));
                    }
                }
                // make sure all threads are done
                __syncthreads();

                // now we need to do a cumulative sum, also known as an inclusive scan
                // we will do this using a fast three-phase parallel algorithm
                // this code assumes 1024 threads in 32 warps of 32 threads
                // it will require modification to work with arbitrary sizes

                // first, we do the cumulative sum within each warp
                // this works because the threads are all implicity synchronized
                if (lane >= 1) binCounts[tid] += binCounts[tid - 1];
                if (lane >= 2) binCounts[tid] += binCounts[tid - 2];
                if (lane >= 4) binCounts[tid] += binCounts[tid - 4];
                if (lane >= 8) binCounts[tid] += binCounts[tid - 8];
                if (lane >= 16) binCounts[tid] += binCounts[tid - 16];
                __syncthreads();

                // now we use a single thread to do a cumulative sum over the last elements of each
                // of the 32 warps
                if (warp == 0) {
                    if (lane >= 1) binCounts[32 * tid + 31] += binCounts[32 * (tid - 1) + 31];
                    if (lane >= 2) binCounts[32 * tid + 31] += binCounts[32 * (tid - 2) + 31];
                    if (lane >= 4) binCounts[32 * tid + 31] += binCounts[32 * (tid - 4) + 31];
                    if (lane >= 8) binCounts[32 * tid + 31] += binCounts[32 * (tid - 8) + 31];
                    if (lane >= 16) binCounts[32 * tid + 31] += binCounts[32 * (tid - 16) + 31];
                }
                __syncthreads();

                // new each warp adds the value of the 31st element of the previous warp
                // there is nothing to add for warp0, so we skip it
                // the last element of each warp already has this sum from the previous step,
                // so we skip it
                if (warp>0 && lane<31) {
                    binCounts[tid] += binCounts[32 * warp - 1];
                }
                __syncthreads();

                // now we need to find the bin containing the k'th highest value
                // we use a single warp, where each thread looks at a block of 32 entries
                // to find the smallest index where the cumulative sum is >= numActive
                // we set flag if we find one
                // this section uses implicit synchronization between threads in a single warp
                if (warp == 0) {
                    int counter = 0;
                    int flag = false;
                    for (counter=0; counter<32; counter++) {
                        if (binCounts[32 * tid + counter] >= numActive) {
                            flag = true;
                            break;
                        }
                    }
                    // now find the smallest bin that meets the criteria
                    if (tid == 0) {
                        *bestBin = 1025;
                    }
                    // if we found a value >= numActive, then update the minimum value
                    if (flag) {
                        atomicMin(bestBin, 32 * tid + counter);
                    }
                }
                __syncthreads();

                const float binMin = minBuffer[*bestBin];
                const float binMax = maxBuffer[*bestBin];

                //  if all energies in this bin are the same, then we are done
                if (fabs(binMin-binMax) < TOLERANCE) {
                    energyCutoff = binMax;
                    break;
                }

                // if this bin ends exactly on the k'th lowest energy, then we are done
                if (binCounts[*bestBin] == numActive) {
                    energyCutoff = binMax;
                    break;
                }

                // otherwise, the correct value lies somewhere within this bin
                // it will between binMin and binMax and we need to find the
                // binCounts[*bestBin] - numActive 'th element
                // we loop through again searching with these updated parameters
                min = binMin;
                max = binMax;
                delta = max - min;
                numActive = binCounts[*bestBin] - numActive;
                __syncthreads();
            }
        }

        // now we know the energyCutoff, so apply it to each group
        for (int i=tid; i<length; i+=blockDim.x) {
            if (energyBuffer[i] <= energyCutoff) {
                activeArray[indexArray[i + start]] = 1.0;
            }
            else {
                activeArray[indexArray[i + start]] = 0.0;
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


extern "C" __global__ void applyDistRest(
                                unsigned long long * __restrict__ force,
                                real* __restrict__ energyBuffer,
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


extern "C" __global__ void applyHyperbolicDistRest(
                                unsigned long long * __restrict__ force,
                                real* __restrict__ energyBuffer,
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
                                real* __restrict__ energyBuffer,
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
                                real* __restrict__ energyBuffer,
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
                                real* __restrict__ energyBuffer,
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

