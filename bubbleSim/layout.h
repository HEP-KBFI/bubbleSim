#pragma once
#include <cstdint>

#ifndef BSIM_REAL
  #define BSIM_REAL double
#endif

#ifdef __HIPCC__
  #define BSIM_HD __host__ __device__ inline
#else
  #define BSIM_HD inline
#endif


namespace bubblesim {
    using real  = BSIM_REAL;   // switchable: particle state, cell state, fields
    using accum = double;      // never switches: reductions, bubble state, time

    struct ParticleSoA {
        real*  x;    real*  y;    real*  z;             // 3 arrays, N doubles each
        real*  p_x;  real*  p_y;  real*  p_z;           // 3 arrays, N doubles each
        real*  E;    real*  m;
        int32_t* cell;                                  // was idxCollisionCell
        uint8_t* flags;                                 // was b_collide + b_inBubble
        uint32_t n;                                     // particle count
    };


    struct Bubble {
        accum radius, speed, gamma;
    };


    struct CollisionCell { /* … */ };

    // ---- scalars ----
    struct SimParams { /* … */ };   





}
