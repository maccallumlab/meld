/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


extern "C" __global__ void computeRdcPhase1(
                            const int numRestraints,
                            const float4* __restrict__ posQ,
                            const int3* __restrict__ atomExptIndices,
                            const float2* __restrict__ kappaCut,
                            float4* __restrict__ r,
                            float* __restrict__ lhs) {
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) {
        // compute the dipole vector and its norm
        float3 rVec = trimTo3(posQ[atomExptIndices[index].x] - posQ[atomExptIndices[index].y]);
        float norm = SQRT(rVec.x*rVec.x + rVec.y*rVec.y + rVec.z*rVec.z);

        // compute the direction cosines
        float x = rVec.x / norm;
        float y = rVec.y / norm;
        float z = rVec.z / norm;

        // store the direction cosines and norm for phase3
        r[index].x = x;
        r[index].y = y;
        r[index].z = z;
        r[index].w = norm;

        // compute lhs
        float invr3 = 1.0 / (norm * norm * norm);
        float k = 2.0 * kappaCut[index].x;
        lhs[5 * index + 0] = k * invr3 * (y*y - x*x);
        lhs[5 * index + 1] = k * invr3 * (z*z - x*x);
        lhs[5 * index + 2] = k * invr3 * (2.0 * x * y);
        lhs[5 * index + 3] = k * invr3 * (2.0 * x * z);
        lhs[5 * index + 4] = k * invr3 * (2.0 * y * z);
    }
}


extern "C" __global__ void computeRdcPhase3(
                            const int numRestraints,
                            const float4* __restrict__ posQ,
                            const int3* __restrict__ atomExptIndices,
                            const float2* __restrict__ kappaCut,
                            const float* __restrict__ S,
                            const float* __restrict__ rhs,
                            const float* __restrict__ tolerance,
                            const float* __restrict__ force_const,
                            const float4* __restrict__ r,
                            const float* __restrict__ lhs,
                            unsigned long long * __restrict__ force,
                            mixed* __restrict__ energyBuffer) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<numRestraints; index+=blockDim.x*gridDim.x) {
        // get our indices, direction cosines and other things
        int expt = atomExptIndices[index].z;
        float x = r[index].x;
        float x2 = x * x;
        float y = r[index].y;
        float y2 = y * y;
        float z = r[index].z;
        float z2 = z * z;
        float norm = r[index].w;
        float invr3 = 1.0 / (norm * norm * norm);
        float kappa = kappaCut[index].x;
        float cut = kappaCut[index].y;

        // get our alignment tensor
        float SYY = S[5 * expt];
        float SZZ = S[5 * expt + 1];
        float SXX = -SYY - SZZ;
        float SXY = S[5 * expt + 2];
        float SXZ = S[5 * expt + 3];
        float SYZ = S[5 * expt + 4];

        // compute the calculated coupling
        float dcalc = 2.0 * kappa * invr3 * (
                SXX * x2 +
                SYY * y2 +
                SZZ * z2 +
                2.0 * SXY * x * y +
                2.0 * SXZ * x * z +
                2.0 * SYZ * y * z );

        // compute the energy and forces
        float energy = 0.;
        float dobs = rhs[index];
        float tol = tolerance[index];
        float fc = force_const[index];
        float temp = 0.;

        float factor = -4.0  * kappa * invr3 / norm;
        float delta = 0;

        // computed splitting is too high
        if (dcalc > (dobs + tol)) {
            delta = dcalc - dobs - tol;
            if (delta < cut) {
                // within the quadratic region
                energy = 0.5 * fc * delta * delta;
                temp = fc * delta * factor;
            } else {
                // outside of the quadratic region
                energy = 0.5 * fc * cut * cut + fc * cut * (delta - cut);
                temp = fc * cut * factor;
            }
        }
        // computed splitting is too low
        else if (dcalc < (dobs - tol)) {
            delta = dobs - tol - dcalc;
            if (delta < cut) {
                // within the quadratic region
                energy = 0.5 * fc * delta * delta;
                temp = -fc * delta * factor;
            } else {
                // outside quadratic region
                energy = 0.5 * fc * cut * cut + fc * cut * (delta - cut);
                temp = -fc * cut * factor;
            }
        }
        // computed splitting is within tolerance
        else {
            energy = 0.;
            temp = 0.;
        }
        float fx = temp * (
                SXX * x * (2.5 * x2 - 1.0) +
                2.5 * SYY * y2 * x +
                2.5 * SZZ * z2 * x +
                SXY * y * (5.0 * x2 - 1.0) +
                SXZ * z * (5.0 * x2 - 1.0) +
                SYZ * 5.0 * x * y * z);

        float fy = temp * (
                2.5 * SXX * x2 * y +
                SYY * y * (2.5 * y2  - 1.0) +
                2.5 * SZZ * z2 * y +
                SXY * x * (5.0 * y2 - 1.0) +
                SXZ * 5.0 * x * y * z +
                SYZ * z * (5.0 * y2 - 1.0));

        float fz = temp * (
                2.5 * SXX * x2 * z +
                2.5 * SYY * y2 * z +
                SZZ * z * (2.5 * z2 - 1.0) +
                SXY * 5.0 * x * y * z +
                SXZ * x * (5.0 * z2 - 1.0) +
                SYZ * y * (5.0 * z2 - 1.0));
                
        // apply forces and energies
        int atom_i = atomExptIndices[index].x;
        int atom_j = atomExptIndices[index].y;
        energyBuffer[threadIndex] += energy;
        atomicAdd(&force[atom_i                       ], static_cast<unsigned long long>((long long) (-fx*0x100000000)));
        atomicAdd(&force[atom_i +     PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-fy*0x100000000)));
        atomicAdd(&force[atom_i + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (-fz*0x100000000)));
        atomicAdd(&force[atom_j                       ], static_cast<unsigned long long>((long long) (fx*0x100000000)));
        atomicAdd(&force[atom_j +     PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (fy*0x100000000)));
        atomicAdd(&force[atom_j + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (fz*0x100000000)));
    }
}
