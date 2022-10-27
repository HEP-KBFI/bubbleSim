#pragma once
#include <CL/cl.hpp>

#include "base.h"

typedef struct Particle {
  cl_numType x;
  cl_numType y;
  cl_numType z;

  cl_numType p_x;
  cl_numType p_y;
  cl_numType p_z;

  cl_numType E;
  cl_numType m;

} Particle;

typedef struct Bubble {
  cl_numType radius;
  cl_numType radius2;           // Squared
  cl_numType radiusAfterStep2;  // (radius + speed * dt)^2

  cl_numType speed;

  cl_numType gamma;
  cl_numType gammaXspeed;  // gamma * speed
} Bubble;
