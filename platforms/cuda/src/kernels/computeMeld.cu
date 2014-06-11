#define ELEM_SWAP(a,b) { int t=(a);(a)=(b);(b)=t; }
__device__
float quick_select_float(const float* energy, int *index, int nelems, int select) {
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
        int globalIndex = indexToGlobal[index];

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

        if(r < distanceBounds[index].x) {
            energy = forceConstants[index] * (r - distanceBounds[index].x) * (distanceBounds[index].x - distanceBounds[index].y) +
                0.5 * forceConstants[index] * (distanceBounds[index].x - distanceBounds[index].y) * (distanceBounds[index].x - distanceBounds[index].y);
            dEdR = forceConstants[index] * (distanceBounds[index].x - distanceBounds[index].y);
        }
        else if(r < distanceBounds[index].y) {
            diff = r - distanceBounds[index].y;
            diff2 = diff * diff;
            energy = 0.5 * forceConstants[index] * diff2;
            dEdR = forceConstants[index] * diff;
        }
        else if(r < distanceBounds[index].z) {
            dEdR = 0.0;
            energy = 0.0;
        }
        else if(r < distanceBounds[index].w) {
            diff = r - distanceBounds[index].z;
            diff2 = diff * diff;
            energy = 0.5 * forceConstants[index] * diff2;
            dEdR = forceConstants[index] * diff;
        }
        else {
            energy = forceConstants[index] * (r - distanceBounds[index].w) * (distanceBounds[index].w - distanceBounds[index].z) +
                0.5 * forceConstants[index] * (distanceBounds[index].w - distanceBounds[index].z) * (distanceBounds[index].w - distanceBounds[index].z);
            dEdR = forceConstants[index] * (distanceBounds[index].w - distanceBounds[index].z);
        }

        // store force into local buffer
        f.x = delta.x * dEdR / r;
        f.y = delta.y * dEdR / r;
        f.z = delta.z * dEdR / r;
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

        /*printf("%d %d %d %d %d %d %d %d %d %f %f %f %f %f\n", index,*/
            /*phi_atom_i, phi_atom_j, phi_atom_k, phi_atom_l,*/
            /*psi_atom_i, psi_atom_j, psi_atom_k, psi_atom_l,*/
            /*phi, psi, energy, dEdPhi, dEdPsi);*/

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
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numGroups; index+=blockDim.x*gridDim.x) {
        int numActive = numActiveArray[index];
        int start = boundsArray[index].x;
        int end = boundsArray[index].y;
        int length = end - start;
        bool applyAll = numActive == length;

        // copy to temp index
        if (!applyAll) {
            for (int i = start; i<end; ++i) {
                tempIndexArray[i] = pristineIndexArray[i];
            }
        }

        // find the the numActive'th energy
        float energyCut = 0.0;
        if (!applyAll) {
            int* chunk = tempIndexArray + start;
            energyCut = quick_select_float(energyArray, chunk, length, numActive-1);
        } else {
            energyCut = 9.0e999;
        }

        // activate all springs where energy <= energyCut
        float thisActive = 0.0;
        float thisEnergy = 0.0;
        float totalEnergy = 0.0;

        for (int i=start; i<end; ++i) {
            thisEnergy = energyArray[pristineIndexArray[i]];
            thisActive = (float)(thisEnergy <= energyCut);
            activeArray[pristineIndexArray[i]] = thisActive;
            totalEnergy += thisActive * thisEnergy;
        }

        // store the total energy
        targetEnergyArray[index] = totalEnergy;
    }
}


__device__ void findMinMax(int start, int end, const int* indexArray,
                           const float* energyArray, float* minBuffer, float* maxBuffer) {
    // minBuffer and maxBuffer are shared arrays of size BlockDim.x
    minBuffer[threadIdx.x] = 9e99;
    maxBuffer[threadIdx.x] = -9e99;

    float energy;

    // Each thread computes the min and max for it's energies and stores them in the buffers
    for (int i=start+threadIdx.x; i<end; i+=blockDim.x) {
        energy = energyArray[indexArray[i]];
        if (energy < minBuffer[threadIdx.x]) {
            minBuffer[threadIdx.x] = energy;
        }
        if (energy > maxBuffer[threadIdx.x]) {
            maxBuffer[threadIdx.x] = energy;
        }
    }
    __syncthreads();

    // Now we do a parallel reduction
    int totalThreads = blockDim.x;
    int index2 = 0;
    float temp = 0;
    while (totalThreads > 1) {
        int halfPoint = (totalThreads >> 1);
        if (threadIdx.x < halfPoint) {
            index2 = threadIdx.x + halfPoint;
            temp = minBuffer[index2];
            if (temp < minBuffer[threadIdx.x]) {
                minBuffer[threadIdx.x] = temp;
            }
            temp = maxBuffer[index2];
            if (temp > maxBuffer[threadIdx.x]) {
                maxBuffer[threadIdx.x] = temp;
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
        float* __restrict__ activeArray)
{
    // shared memory for buffers
    // will hold float buffers for min/max calculation
    // then will be used as int buffer to hold bucket counts
    __shared__ unsigned int buffer[2050];

    for (int collIndex=blockIdx.x; collIndex<numCollections; collIndex+=gridDim.x) {
        float* minBuffer = (float *)buffer;
        float* maxBuffer = (float *)buffer + 1024;

        int numActive = numActiveArray[collIndex];
        int start = boundsArray[collIndex].x;
        int end = boundsArray[collIndex].y;

        findMinMax(start, end, indexArray, energyArray, minBuffer, maxBuffer);
        float min = minBuffer[0];
        float max = maxBuffer[0];

        if (min == max) {
            // all of the energies are the same. They should all be active
            for (int i=start+threadIdx.x; i<end; i+=blockDim.x) {
                activeArray[indexArray[i]] = 1.0;
            }
            // we're done, so do another iteration of the loop
            continue;
        }

        int oneMoreIteration = 0;
        float delta = max - min;
        // loop forever until we've found the k'th lowest energy and activated
        // all restraints as appropriate
        for(;;) {
            unsigned int* bucketCount = buffer;
            unsigned int* bucketBuffer = buffer + 1025;
            // reset the bucket counts
            bucketCount[threadIdx.x] = 0;
            if (threadIdx.x==0) {
                bucketCount[1024] = 0;
            }
            __syncthreads();

            // update the bucket counts
            for (int i=start+threadIdx.x; i<end; i+=blockDim.x) {
                // assign energies to buckets
                int globalIndex = indexArray[i];
                float energy = energyArray[globalIndex];
                int index = float2int(floorf((blockDim.x-1) / delta * (energy - min)));
                if (index < 0) {
                    atomicAdd(bucketCount + 1024, 1);
                } else if ( (index >=0) && (index < blockDim.x) ) {
                    atomicAdd(bucketCount + index, 1);
                }

                // set active
                if (energy <= max) {
                    activeArray[globalIndex] = 1.0;
                } else {
                    activeArray[globalIndex] = 0.0;
                }
            }
            __syncthreads();

            // we're done, so break out of this loop and do another iteration fo the outer
            // loop over collections
            if (oneMoreIteration == 1) {
                __syncthreads();
                break;
            }

            // do cumulative sum of bucket counts
            bucketBuffer[threadIdx.x] = bucketCount[threadIdx.x];
            __syncthreads();
            for (int d=0; d<10; ++d) {
                int twoToD = (1 << d);
                if (threadIdx.x >= twoToD) {
                    bucketCount[threadIdx.x] = bucketBuffer[threadIdx.x] + bucketBuffer[threadIdx.x - twoToD];
                } else {
                    bucketCount[threadIdx.x] = bucketBuffer[threadIdx.x];
                }
                __syncthreads();
                unsigned int* temp = bucketCount;
                bucketCount = bucketBuffer;
                bucketBuffer = temp;
            }

            // find the kth bin
            int kthBin;
            for(kthBin = 0; kthBin < 1024; ++kthBin) {
                if ( (bucketBuffer[kthBin] + bucketCount[1024]) >= numActive) {
                    break;
                }
            }

            // figure out the bin count
            int binCount;
            if (kthBin==0) {
                binCount = bucketBuffer[kthBin];
            } else {
                binCount = bucketBuffer[kthBin] - bucketBuffer[kthBin-1];
            }

            // if there's only one item, we just need to make one more lap around
            if (binCount==1) {
                oneMoreIteration = 1;
            }

            // update max, min
            max = min + ((float)kthBin + 1.0) * delta / (blockDim.x - 1);
            min = min + (float)kthBin * delta / (blockDim.x - 1);
            delta = max - min;

            // if max==min, then everything left has the same energy, so just make one more lap around
            if (max==min) {
                min = 0;
                delta = max;
                oneMoreIteration = 1;
            }
        }
    }
}


extern "C" __global__ void applyGroups(
                            float* __restrict__ groupActive,
                            float* __restrict__ restraintActive,
                            const int2* __restrict__ bounds,
                            int numGroups) {
    for (int groupIndex=blockIdx.x*blockDim.x+threadIdx.x; groupIndex<numGroups; groupIndex+=blockDim.x*gridDim.x) {
        float active = groupActive[groupIndex];
        for (int i=bounds[groupIndex].x; i<bounds[groupIndex].y; ++i) {
            float oldActive = restraintActive[i];
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

