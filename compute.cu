#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
#include "compute.h"

vector3 *d_accels;
double *d_mass;

__global__ void accelMatrix(vector3 *accels,  vector3 *pos, double *mass, int n){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i>=n || j>=n){
        return;
    }

    vector3 a = {0.0, 0.0, 0.0};

    if (i!=j) {
        vector3 dist;
        dist[0] = pos[j][0] - pos[i][0];
        dist[1] = pos[j][1] - pos[i][1];
        dist[2] = pos[j][2] - pos[i][2];

        double distSq = dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2];
        double distR = sqrt(distSq);

        double g = GRAV_CONSTANT * mass[j] / distSq;

        a[0] = g * dist[0] / distR;
        a[1] = g * dist[1] / distR;
        a[2] = g * dist[2] / distR;
    }

    accels[i*n + j][0] = a[0];
    accels[i*n + j][1] = a[1];
    accels[i*n + j][2] = a[2];

}

__global__ void sumVP(vector3 *accels,vector3 *pos, vector3 *vel, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i>=n){
        return;
    }

    vector3 v = {0.0, 0.0, 0.0};

    for (int j=0; j<n; j++){
        v[0] += accels[i*n + j][0];
        v[1] += accels[i*n + j][1];
        v[2] += accels[i*n + j][2];
    }

    vel[i][0] += v[0] * INTERVAL;
    vel[i][1] += v[1] * INTERVAL;
    vel[i][2] += v[2] * INTERVAL;

    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

void initDeviceMemory(){
    size_t accelSize = sizeof(vector3) * NUMENTITIES * NUMENTITIES;
    size_t massSize = sizeof(double) * NUMENTITIES;
    size_t vectorSize = sizeof(vector3) * NUMENTITIES;

    cudaMalloc(&d_hPos, vectorSize);
    cudaMalloc(&d_hVel, vectorSize);
    cudaMalloc(&d_accels, accelSize);
    cudaMalloc(&d_mass, massSize);

    cudaMemcpy(d_hPos, hPos, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, massSize, cudaMemcpyHostToDevice);

}


void freeDMemory(){
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_accels);
    cudaFree(d_mass);
}

void cpyToHost(){
    size_t vectorSize = sizeof(vector3) * NUMENTITIES;

    cudaMemcpy(hPos, d_hPos, vectorSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, vectorSize, cudaMemcpyDeviceToHost);
}



void compute(){
	dim3 threads(16,16);
    dim3 blocks((NUMENTITIES + threads.x - 1)/threads.x, (NUMENTITIES + threads.y - 1)/threads.y);

    accelMatrix<<<blocks, threads>>>(d_accels, d_hPos, d_mass, NUMENTITIES);
    cudaDeviceSynchronize();

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUMENTITIES + threadsPerBlock - 1) / threadsPerBlock;

    sumVP<<<blocksPerGrid, threadsPerBlock>>>(d_accels, d_hPos, d_hVel, NUMENTITIES);
    cudaDeviceSynchronize();
}
