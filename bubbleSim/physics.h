#pragma once
#include "layout.h"

namespace sim {

struct Vec3 { real x, y, z; };

BSIM_HD void moveLinear(Particle& p, real v_x, real v_y, real v_z, real dt) {
  p.x = fma(v_x, dt, p.x);
  p.y = fma(v_y, dt, p.y);
  p.z = fma(v_z, dt, p.z);
}

BSIM_HD real radiusSquared(const Particle& p) {
  return fma(p.x, p.x, fma(p.y, p.y, p.z * p.z));
}

BSIM_HD real energyOf(const Particle& p) {
  return sqrt(fma(p.p_x, p.p_x, fma(p.p_y, p.p_y, fma(p.p_z, p.p_z, pow(p.m, 2)))));
}

BSIM_HD Vec3 wallNormal(const Particle& p, const Bubble& b, real X2,
                        real m_in, real m_out) {
  if (m_in > m_out)
    return {-p.x * b.gamma / sqrt(X2),
            -p.y * b.gamma / sqrt(X2),
            -p.z * b.gamma / sqrt(X2)};

  return { p.x * b.gamma / sqrt(X2),
           p.y * b.gamma / sqrt(X2),
           p.z * b.gamma / sqrt(X2)};
}

}  // namespace sim