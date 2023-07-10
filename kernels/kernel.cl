#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Bubble type (also defined in c++ code)
typedef struct Bubble {
  double radius;
  double radius2;           // Squared
  double radiusAfterStep2;  // (radius + speed * dt)^2
  double speed;
  double gamma;
  double gammaXspeed;  // gamma * speed
} Bubble;

// In development
typedef struct CollisionCell {
  double vX;
  double vY;
  double vZ;

  double x;
  double y;
  double z;
  double theta;

  double p_E;
  double p_x;
  double p_y;
  double p_z;

  char b_collide;
  double mass;
  unsigned int particle_count;
} CollisionCell;

/*
 * Given particle and it's velocity move it's location by time=dt.
 * Also this function already updates particle location.
 */

double moveLinear_new(double x, double v, double dt) {
  // x = x + v*dt
  return x + v * dt;
}

// TODO: Remove if clauses? Make into single expression. Also check if time1 <
// time2.
double calculateTimeToWall_new(double x, double y, double z, double E,
                               double pX, double pY, double pZ, Bubble bubble,
                               double t_dt) {
  // p_i*p^i / E^2 - V_b^2
  double a = fma(pX, pX, fma(pY, pY, fma(pZ, pZ, 0.))) / pow(E, 2) -
             bubble.speed * bubble.speed;
  // x_i* p^i / E - V_b^R_b
  double b = fma(x, pX / E,
                 fma(y, pY / E, fma(z, pZ / E, -bubble.radius * bubble.speed)));
  // x_i * x^i - R_b^2
  double c = fma(x, x, fma(y, y, fma(z, z, -bubble.radius * bubble.radius)));
  double d = fma(b, b, -a * c);

  // Solve [-b +- sqrt(b^2 - a*c)]/a
  // Select smallest positive solution.

  // No solutions -> Never collide
  if (d < 0) {
    return 0;
  }

  double time1, time2;
  // Calculate two solutions
  time1 = (-b - sqrt(d)) / a;
  time2 = (-b + sqrt(d)) / a;
  if ((0 < time1) && (time1 <= t_dt)) {
    return time1;
  } else if ((0 < time2) && (time2 <= t_dt)) {
    return time2;
  }
  // Solution might exist but not for current step -> no solution between 0 <
  // time < dt
  else {
    return 0.;
  }
}

// TODO: Return only one sign normal and do sign change in the kernel code or
// somewhere else. All kernels are calculated directed "outside".
void calculateNormal_new(double *t_n1, double *t_n2, double *t_n3, double x,
                         double y, double z, Bubble bubble, double t_X2) {
  // If M_in > M_out then normal is towards center of bubble
  *t_n1 = x * bubble.gamma / sqrt(t_X2);
  *t_n2 = y * bubble.gamma / sqrt(t_X2);
  *t_n3 = z * bubble.gamma / sqrt(t_X2);
}

double calculateDistanceFromCenter(double x, double y, double z) {
  return sqrt(fma(x, x, fma(y, y, z * z)));
}

double calculateDistanceSquaredFromCenter(double x, double y, double z) {
  return fma(x, x, fma(y, y, z * z));
}

double calculateParticleEnergy(double pX, double pY, double pZ, double mass) {
  return sqrt(fma(pX, pX, fma(pY, pY, fma(pZ, pZ, pow(mass, 2)))));
}

/*
============================================================================
============================================================================
                                    Step
============================================================================
============================================================================
*/
__kernel void particle_step_linear(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global double *particles_E,
    __global double *particles_pX, __global double *particles_pY,
    __global double *particles_pZ, __constant double *t_dt) {
  unsigned int gid = get_global_id(0);

  double E = particles_E[gid];
  double vX = particles_pX[gid] / E;
  double vY = particles_pY[gid] / E;
  double vZ = particles_pZ[gid] / E;
  double dt = t_dt[0];

  particles_X[gid] = moveLinear_new(particles_X[gid], vX, dt);
  particles_Y[gid] = moveLinear_new(particles_Y[gid], vY, dt);
  particles_Z[gid] = moveLinear_new(particles_Z[gid], vZ, dt);
}

__kernel void particle_step_with_bubble(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global double *particles_E,
    __global double *particles_pX, __global double *particles_pY,
    __global double *particles_pZ, __global double *particles_M,
    __global double *t_dP, __global char *t_interactedFalse,
    __global char *t_passedFalse, __global char *t_interactedTrue,
    __constant Bubble *t_bubble, __constant double *t_m_in,
    __constant double *t_m_out, __constant double *t_delta_m2,
    __constant double *t_dt) {
  unsigned int gid = get_global_id(0);
  /*
   * t_dP is a array where each element is "pressure" by particle respective to
   * it's index. t_dP is not actual pressure but actually energy change ΔE.
   * Afterwards ΔP = ΔE/Area
   */

  // Bubble parameters
  struct Bubble bubble = t_bubble[0];
  // Particle
  double x = particles_X[gid];
  double y = particles_Y[gid];
  double z = particles_Z[gid];
  double E = particles_E[gid];
  double pX = particles_pX[gid];
  double pY = particles_pY[gid];
  double pZ = particles_pZ[gid];
  double mass = particles_M[gid];

  double M_in = t_m_in[0];
  double M_out = t_m_out[0];
  double Delta_M2 = t_delta_m2[0];
  double dt = t_dt[0];

  // Particle velocity
  double vX = pX / E;
  double vY = pY / E;
  double vZ = pZ / E;

  // Particle radius (squared) from the center
  double X2 = calculateDistanceSquaredFromCenter(x, y, z);

  // Particle coordinates after dt assuming linear movement
  double x_Vdt = fma(vX, dt, x);  // x component
  double y_Vdt = fma(vY, dt, y);  // y component
  double z_Vdt = fma(vZ, dt, z);  // z component

  double X_dt2 = fma(x_Vdt, x_Vdt, fma(y_Vdt, y_Vdt, z_Vdt * z_Vdt));
  // Algorithm if mass inside the bubble is bigger
  if (M_in < M_out) {
    if (X2 < bubble.radius2) {
      if (X_dt2 < bubble.radiusAfterStep2) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dP[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall =
            calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear_new(x, vX, time_to_wall);
        y = moveLinear_new(y, vY, time_to_wall);
        z = moveLinear_new(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        if (np * np < Delta_M2) {
          // Update particle 4-momentum after collision
          pX = fma(np * 2., nX, pX);
          pY = fma(np * 2., nY, pY);
          pZ = fma(np * 2., nZ, pZ);
          // Calculate applied pressure
          t_dP[gid] = bubble.gamma * np * 2.;
          E = calculateParticleEnergy(pX, pY, pZ, M_in);

          // Particle only interacts
          t_interactedFalse[gid] += 1;
        } else {
          particles_M[gid] = M_out;

          // Calculate particle momentum after collision
          pX = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nX, pX);
          pY = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nY, pY);
          pZ = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nZ, pZ);
          // Calculate applied pressure
          t_dP[gid] =
              bubble.gamma * np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.)));
          E = calculateParticleEnergy(pX, pY, pZ, M_out);

          // Particle interacts and passes through
          t_interactedFalse[gid] += 1;
          t_passedFalse[gid] += 1;
        }
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // ========== Movement after the bubble interaction ==========
        x = moveLinear_new(x, vX, dt - time_to_wall);
        y = moveLinear_new(y, vY, dt - time_to_wall);
        z = moveLinear_new(z, vZ, dt - time_to_wall);
      }
    } else {
      if (X_dt2 > bubble.radiusAfterStep2) {
        x = moveLinear_new(x, vX, dt);
        y = moveLinear_new(y, vY, dt);
        z = moveLinear_new(z, vZ, dt);
        t_dP[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall =
            calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear_new(x, vX, time_to_wall);
        y = moveLinear_new(y, vY, time_to_wall);
        z = moveLinear_new(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);

        calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;

        particles_M[gid] = M_in;
        pX = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nX, pX);
        pY = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nY, pY);
        pZ = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nZ, pZ);

        // Calculate applied pressure
        t_dP[gid] =
            bubble.gamma * np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.)));
        E = calculateParticleEnergy(pX, pY, pZ, M_in);

        // Update particle velocity and move amount of (dt - timeToWall)
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // Movement after the bubble interaction
        x = moveLinear_new(x, vX, dt - time_to_wall);
        y = moveLinear_new(y, vY, dt - time_to_wall);
        z = moveLinear_new(z, vZ, dt - time_to_wall);
        // Particle interacts from the higher mass side
        t_interactedTrue[gid] += 1;
      }
    }
  } else {
    if (X2 > bubble.radius2) {
      if (X_dt2 > bubble.radiusAfterStep2) {
        x = moveLinear_new(x, vX, dt);
        y = moveLinear_new(y, vY, dt);
        z = moveLinear_new(z, vZ, dt);
        t_dP[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall =
            calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear_new(x, vX, time_to_wall);
        y = moveLinear_new(y, vY, time_to_wall);
        z = moveLinear_new(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
        // Change direction of the normal
        nX = -nX;
        nY = -nY;
        nZ = -nZ;

        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        if (np * np < Delta_M2) {
          // Update particle 4-momentum after collision
          pX = fma(np * 2., nX, pX);
          pY = fma(np * 2., nY, pY);
          pZ = fma(np * 2., nZ, pZ);
          // Calculate applied pressure
          t_dP[gid] = bubble.gamma * np * 2.;
          E = calculateParticleEnergy(pX, pY, pZ, M_in);

          // Particle only interacts
          t_interactedFalse[gid] += 1;
        } else {
          particles_M[gid] = M_in;

          // Calculate particle momentum after collision
          pX = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nX, pX);
          pY = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nY, pY);
          pZ = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nZ, pZ);
          // Calculate applied pressure
          t_dP[gid] =
              bubble.gamma * np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.)));
          E = calculateParticleEnergy(pX, pY, pZ, M_out);

          // Particle interacts and passes through
          t_interactedFalse[gid] += 1;
          t_passedFalse[gid] += 1;
        }
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // ========== Movement after the bubble interaction ==========
        x = moveLinear_new(x, vX, dt - time_to_wall);
        y = moveLinear_new(y, vY, dt - time_to_wall);
        z = moveLinear_new(z, vZ, dt - time_to_wall);
      }
    } else {
      if (X_dt2 < bubble.radiusAfterStep2) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dP[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall =
            calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear_new(x, vX, time_to_wall);
        y = moveLinear_new(y, vY, time_to_wall);
        z = moveLinear_new(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);

        calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;

        particles_M[gid] = M_out;
        pX = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nX, pX);
        pY = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nY, pY);
        pZ = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nZ, pZ);

        // Calculate applied pressure
        t_dP[gid] =
            bubble.gamma * np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.)));
        E = calculateParticleEnergy(pX, pY, pZ, M_in);

        // Update particle velocity and move amount of (dt - timeToWall)
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // Movement after the bubble interaction
        x = moveLinear_new(x, vX, dt - time_to_wall);
        y = moveLinear_new(y, vY, dt - time_to_wall);
        z = moveLinear_new(z, vZ, dt - time_to_wall);
        // Particle interacts from the higher mass side
        t_interactedTrue[gid] += 1;
      }
    }
  }

  particles_X[gid] = x;
  particles_Y[gid] = y;
  particles_Z[gid] = z;
  particles_pX[gid] = pX;
  particles_pY[gid] = pY;
  particles_pZ[gid] = pZ;
  particles_E[gid] = E;
}

__kernel void particle_step_with_bubble_inverted(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global double *particles_E,
    __global double *particles_pX, __global double *particles_pY,
    __global double *particles_pZ, __global double *particles_M,
    __global double *t_dP, __global char *t_interactedFalse,
    __global char *t_passedFalse, __global char *t_interactedTrue,
    __constant Bubble *t_bubble, __constant double *t_m_in,
    __constant double *t_m_out, __constant double *t_delta_m2,
    __constant double *t_dt) {
  unsigned int gid = get_global_id(0);
  /*
   * t_dP is a array where each element is "pressure" by particle respective to
   * it's index. t_dP is not actual pressure but actually energy change ΔE.
   * Afterwards ΔP = ΔE/Area
   */

  // Normal must be defined in the direction of mass decrease

  // Bubble parameters
  struct Bubble bubble = t_bubble[0];

  // Particle
  double x = particles_X[gid];
  double y = particles_Y[gid];
  double z = particles_Z[gid];
  double E = particles_E[gid];
  double pX = particles_pX[gid];
  double pY = particles_pY[gid];
  double pZ = particles_pZ[gid];
  double mass = particles_M[gid];

  double M_in = t_m_in[0];
  double M_out = t_m_out[0];
  double Delta_M2 = t_delta_m2[0];
  double dt = t_dt[0];

  // Particle velocity
  double vX = pX / E;
  double vY = pY / E;
  double vZ = pZ / E;

  // Particle radius (squared) from the center
  double X2 = calculateDistanceSquaredFromCenter(x, y, z);

  // Particle coordinates after dt assuming linear movement
  double x_Vdt = fma(vX, dt, x);  // x component
  double y_Vdt = fma(vY, dt, y);  // y component
  double z_Vdt = fma(vZ, dt, z);  // z component

  double X_dt2 = fma(x_Vdt, x_Vdt, fma(y_Vdt, y_Vdt, z_Vdt * z_Vdt));

  if (M_in < M_out) {
    if (X2 < bubble.radius2) {
      if (X_dt2 < bubble.radiusAfterStep2) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dP[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall =
            calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear_new(x, vX, time_to_wall);
        y = moveLinear_new(y, vY, time_to_wall);
        z = moveLinear_new(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
        nX = -nX;
        nY = -nY;
        nZ = -nZ;

        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        if (np * np < Delta_M2) {
          // Update particle 4-momentum after collision
          pX = fma(np * 2., nX, pX);
          pY = fma(np * 2., nY, pY);
          pZ = fma(np * 2., nZ, pZ);
          // Calculate applied pressure
          t_dP[gid] = bubble.gamma * np * 2.;
          E = calculateParticleEnergy(pX, pY, pZ, M_in);

          // Particle only interacts
          t_interactedFalse[gid] += 1;
        } else {
          particles_M[gid] = M_out;

          // Calculate particle momentum after collision
          pX = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nX, pX);
          pY = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nY, pY);
          pZ = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nZ, pZ);
          // Calculate applied pressure
          t_dP[gid] =
              bubble.gamma * np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.)));
          E = calculateParticleEnergy(pX, pY, pZ, M_out);

          // Particle interacts and passes through
          t_interactedFalse[gid] += 1;
          t_passedFalse[gid] += 1;
        }
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // ========== Movement after the bubble interaction ==========
        x = moveLinear_new(x, vX, dt - time_to_wall);
        y = moveLinear_new(y, vY, dt - time_to_wall);
        z = moveLinear_new(z, vZ, dt - time_to_wall);
      }
    } else {
      if (X_dt2 > bubble.radiusAfterStep2) {
        x = moveLinear_new(x, vX, dt);
        y = moveLinear_new(y, vY, dt);
        z = moveLinear_new(z, vZ, dt);
        t_dP[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall =
            calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear_new(x, vX, time_to_wall);
        y = moveLinear_new(y, vY, time_to_wall);
        z = moveLinear_new(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);

        calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
        nX = -nX;
        nY = -nY;
        nZ = -nZ;

        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;

        particles_M[gid] = M_in;
        pX = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nX, pX);
        pY = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nY, pY);
        pZ = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nZ, pZ);

        // Calculate applied pressure
        t_dP[gid] =
            bubble.gamma * np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.)));
        E = calculateParticleEnergy(pX, pY, pZ, M_in);

        // Update particle velocity and move amount of (dt - timeToWall)
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // Movement after the bubble interaction
        x = moveLinear_new(x, vX, dt - time_to_wall);
        y = moveLinear_new(y, vY, dt - time_to_wall);
        z = moveLinear_new(z, vZ, dt - time_to_wall);
        // Particle interacts from the higher mass side
        t_interactedTrue[gid] += 1;
      }
    }
  } else {
    if (X2 > bubble.radius2) {
      if (X_dt2 > bubble.radiusAfterStep2) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dP[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall =
            calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear_new(x, vX, time_to_wall);
        y = moveLinear_new(y, vY, time_to_wall);
        z = moveLinear_new(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        if (np * np < Delta_M2) {
          // Update particle 4-momentum after collision
          pX = fma(np * 2., nX, pX);
          pY = fma(np * 2., nY, pY);
          pZ = fma(np * 2., nZ, pZ);
          // Calculate applied pressure
          t_dP[gid] = bubble.gamma * np * 2.;
          E = calculateParticleEnergy(pX, pY, pZ, M_in);

          // Particle only interacts
          t_interactedFalse[gid] += 1;
        } else {
          particles_M[gid] = M_in;

          // Calculate particle momentum after collision
          pX = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nX, pX);
          pY = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nY, pY);
          pZ = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nZ, pZ);
          // Calculate applied pressure
          t_dP[gid] =
              bubble.gamma * np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.)));
          E = calculateParticleEnergy(pX, pY, pZ, M_out);

          // Particle interacts and passes through
          t_interactedFalse[gid] += 1;
          t_passedFalse[gid] += 1;
        }
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // ========== Movement after the bubble interaction ==========
        x = moveLinear_new(x, vX, dt - time_to_wall);
        y = moveLinear_new(y, vY, dt - time_to_wall);
        z = moveLinear_new(z, vZ, dt - time_to_wall);
      }
    } else {
      if (X_dt2 < bubble.radiusAfterStep2) {
        x = moveLinear_new(x, vX, dt);
        y = moveLinear_new(y, vY, dt);
        z = moveLinear_new(z, vZ, dt);
        t_dP[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall =
            calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear_new(x, vX, time_to_wall);
        y = moveLinear_new(y, vY, time_to_wall);
        z = moveLinear_new(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);

        calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;

        particles_M[gid] = M_out;
        pX = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nX, pX);
        pY = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nY, pY);
        pZ = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nZ, pZ);

        // Calculate applied pressure
        t_dP[gid] =
            bubble.gamma * np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.)));
        E = calculateParticleEnergy(pX, pY, pZ, M_in);

        // Update particle velocity and move amount of (dt - timeToWall)
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // Movement after the bubble interaction
        x = moveLinear_new(x, vX, dt - time_to_wall);
        y = moveLinear_new(y, vY, dt - time_to_wall);
        z = moveLinear_new(z, vZ, dt - time_to_wall);
        // Particle interacts from the higher mass side
        t_interactedTrue[gid] += 1;
      }
    }
  }

  particles_X[gid] = x;
  particles_Y[gid] = y;
  particles_Z[gid] = z;
  particles_pX[gid] = pX;
  particles_pY[gid] = pY;
  particles_pZ[gid] = pZ;
  particles_E[gid] = E;
}

__kernel void particles_with_false_bubble_step_reflect(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global double *particles_E,
    __global double *particles_pX, __global double *particles_pY,
    __global double *particles_pZ, __global double *t_dP,
    __global char *t_interactedFalse, __global char *t_passedFalse,
    __global char *t_interactedTrue, __constant Bubble *t_bubble,
    __constant double *t_m_in, __constant double *t_m_out,
    __constant double *t_delta_m2, __constant double *t_dt) {
  unsigned int gid = get_global_id(0);
  // dE - dP is not actual energy difference. dE = ΔE/R_b -> to avoid
  // singularities/noise near R_b ~ 0

  // Bubble parameters
  struct Bubble bubble = t_bubble[0];
  // Particle
  double x = particles_X[gid];
  double y = particles_Y[gid];
  double z = particles_Z[gid];
  double E = particles_E[gid];
  double pX = particles_pX[gid];
  double pY = particles_pY[gid];
  double pZ = particles_pZ[gid];

  double M_in = t_m_in[0];
  double M_out = t_m_out[0];
  double Delta_M2 = t_delta_m2[0];

  // Particle parameters
  double dt = t_dt[0];

  // Calculate particle velocity
  double vX = pX / E;
  double vY = pY / E;
  double vZ = pZ / E;

  // fma(a, b, c) = a * b + c
  double X2 = calculateDistanceSquaredFromCenter(x, y, z);

  // Calculate new coordinates if particle would move linearly
  // after time dt
  double x_Vdt = fma(vX, dt, x);
  double y_Vdt = fma(vY, dt, y);
  double z_Vdt = fma(vZ, dt, z);

  // Find new particle radius (from the center) squared
  double X_dt2 = fma(x_Vdt, x_Vdt, fma(y_Vdt, y_Vdt, z_Vdt * z_Vdt));

  // Placholders for normal, time to wall, n·p (n and p are 4-vectors)

  if (((X2 < bubble.radius2) && (M_in < M_out))) {
    // move particles that stay in
    if ((X_dt2 < bubble.radiusAfterStep2) && (M_in < M_out)) {
      x = x_Vdt;
      y = y_Vdt;
      z = z_Vdt;
      t_dP[gid] = 0.;
    }
    // reflect particles that would to get out
    else {
      double nX, nY, nZ;
      double time_to_wall, np;
      time_to_wall =
          calculateTimeToWall_new(x, y, z, E, pX, pY, pZ, bubble, dt);
      x = moveLinear_new(x, vX, time_to_wall);
      y = moveLinear_new(y, vY, time_to_wall);
      z = moveLinear_new(z, vZ, time_to_wall);
      X2 = calculateDistanceSquaredFromCenter(x, y, z);

      // Update particle's radius squared value to calculate normal vector

      // Relativistic reflection algorithm
      // Calculate normal at the location where particle and bubble interact
      calculateNormal_new(&nX, &nY, &nZ, x, y, z, bubble, X2);
      np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;

      pX = fma(np * 2., nX, pX);
      pY = fma(np * 2., nY, pY);
      pZ = fma(np * 2., nZ, pZ);

      t_dP[gid] = bubble.gamma * np * 2.;
      // Update particle energy
      E = calculateParticleEnergy(pX, pY, pZ, M_in);
      // Count particle collision with the bubble wall
      t_interactedFalse[gid] += 1;

      // Update velocity vector
      vX = pX / E;
      vY = pY / E;
      vZ = pZ / E;

      // ========== Movement after the bubble interaction ==========
      x = moveLinear_new(x, vX, time_to_wall);
      y = moveLinear_new(y, vY, time_to_wall);
      z = moveLinear_new(z, vZ, time_to_wall);
    }
  } else {
    x = INFINITY;
    y = INFINITY;
    z = INFINITY;
  }

  particles_X[gid] = x;
  particles_Y[gid] = y;
  particles_Z[gid] = z;
  particles_pX[gid] = pX;
  particles_pY[gid] = pY;
  particles_pZ[gid] = pZ;
  particles_E[gid] = E;
}

/*
============================================================================
============================================================================
                                  Collision
============================================================================
============================================================================
*/

__kernel void rotate_momentum(__global double *particles_E,
                              __global double *particles_pX,
                              __global double *particles_pY,
                              __global double *particles_pZ,
                              __global unsigned int *particles_collision_cell_index,
                              __global CollisionCell *t_cells) {
  unsigned int gid = get_global_id(0);

  // If in bubble then cell number is doubled and second half is in bubble
  // cells.
  if (particles_collision_cell_index[gid] != 0) {
    CollisionCell cell = t_cells[particles_collision_cell_index[gid]];

    double v2 = fma(cell.vX, cell.vX, fma(cell.vY, cell.vY, cell.vZ * cell.vZ));
    double gamma = 1 / sqrt(1 - v2);
    double gamma_minus_one = gamma - 1;

    double gamma_minus_one_divided_cell_v2 = gamma_minus_one / v2;

    double cos_theta = cos(cell.theta);
    double one_minus_cos_theta = 1 - cos_theta;
    double sin_theta = sin(cell.theta);

    double E = particles_E[gid];
    double pX_1 = particles_pX[gid];
    double pY_1 = particles_pY[gid];
    double pZ_1 = particles_pZ[gid];
    double pX_2, pY_2, pZ_2;

    if ((cell.particle_count > 1) && (cell.mass != 0) && (cell.b_collide)) {
      // Lorentz transformation (to COM frame)
      pX_2 = -E * gamma * cell.vX +
             pX_1 * (1 + cell.vX * cell.vX * gamma_minus_one_divided_cell_v2) +
             pY_1 * cell.vX * cell.vY * gamma_minus_one_divided_cell_v2 +
             pZ_1 * cell.vX * cell.vZ * gamma_minus_one_divided_cell_v2;

      pY_2 = -gamma * E * cell.vY +
             pY_1 * (1 + cell.vY * cell.vY * gamma_minus_one_divided_cell_v2) +
             pX_1 * cell.vX * cell.vY * gamma_minus_one_divided_cell_v2 +
             pZ_1 * cell.vY * cell.vZ * gamma_minus_one_divided_cell_v2;

      pZ_2 = -gamma * E * cell.vZ +
             pZ_1 * (1 + cell.vZ * cell.vZ * gamma_minus_one_divided_cell_v2) +
             pX_1 * cell.vX * cell.vZ * gamma_minus_one_divided_cell_v2 +
             pY_1 * cell.vY * cell.vZ * gamma_minus_one_divided_cell_v2;
      E = gamma * (E - pX_1 * cell.vX - pY_1 * cell.vY - pZ_1 * cell.vZ);

      // Rotate 3-momentum
      pX_1 =
          pX_2 * (cos_theta + pow(cell.x, 2) * one_minus_cos_theta) +
          pY_2 * (cell.x * cell.y * one_minus_cos_theta - cell.z * sin_theta) +
          pZ_2 * (cell.x * cell.z * one_minus_cos_theta + cell.y * sin_theta);
      pY_1 =
          pX_2 * (cell.x * cell.y * one_minus_cos_theta + cell.z * sin_theta) +
          pY_2 * (cos_theta + pow(cell.y, 2) * one_minus_cos_theta) +
          pZ_2 * (cell.y * cell.z * one_minus_cos_theta - cell.x * sin_theta);
      pZ_1 =
          pX_2 * (cell.x * cell.z * one_minus_cos_theta - cell.y * sin_theta) +
          pY_2 * (cell.y * cell.z * one_minus_cos_theta + cell.x * sin_theta) +
          pZ_2 * (cos_theta + pow(cell.z, 2) * one_minus_cos_theta);

      // Lorentz inverse transformation (to initial frame)
      pX_2 = gamma * E * cell.vX +
             pX_1 * (1 + cell.vX * cell.vX * gamma_minus_one_divided_cell_v2) +
             pY_1 * cell.vX * cell.vY * gamma_minus_one_divided_cell_v2 +
             pZ_1 * cell.vX * cell.vZ * gamma_minus_one_divided_cell_v2;

      pY_2 = gamma * E * cell.vY +
             pY_1 * (1 + cell.vY * cell.vY * gamma_minus_one_divided_cell_v2) +
             pX_1 * cell.vX * cell.vY * gamma_minus_one_divided_cell_v2 +
             pZ_1 * cell.vY * cell.vZ * gamma_minus_one_divided_cell_v2;

      pZ_2 = gamma * E * cell.vZ +
             pZ_1 * (1 + cell.vZ * cell.vZ * gamma_minus_one_divided_cell_v2) +
             pX_1 * cell.vX * cell.vZ * gamma_minus_one_divided_cell_v2 +
             pY_1 * cell.vY * cell.vZ * gamma_minus_one_divided_cell_v2;

      E = gamma * (E + pX_1 * cell.vX + pY_1 * cell.vY + pZ_1 * cell.vZ);
      particles_E[gid] = E;
      particles_pX[gid] = pX_2;
      particles_pY[gid] = pY_2;
      particles_pZ[gid] = pZ_2;
    }
  }
}

/*
============================================================================
============================================================================
                               Labeling
============================================================================
============================================================================
*/

__kernel void label_particles_position_by_coordinate(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global char *particles_bool_in_bubble,
    __global Bubble *t_bubble) {
  unsigned int gid = get_global_id(0);
  double x = particles_X[gid];
  double y = particles_Y[gid];
  double z = particles_Z[gid];

  // If R_b^2 > R_x^2 then particle is inside the bubble
  particles_bool_in_bubble[gid] =
      fma(x, x, fma(y, y, z * z)) < t_bubble[0].radius2;
}

__kernel void label_particles_position_by_mass(
    __global double *particles_M, __global char *particle_in_bubble,
    __global double *mass_in) {
  unsigned int gid = get_global_id(0);

  // If R_b^2 > R_x^2 then particle is inside the bubble
  particle_in_bubble[gid] = particles_M[gid] == mass_in[0];
}

__kernel void assign_particle_to_collision_cell(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global unsigned int *m_particle_collision_cell_index,
    __global const unsigned int *maxCellIndex,
    __global const double *cellLength, __global const double *cuboidShift) {
  unsigned int gid = get_global_id(0);

  // Find cell numbers
  int x_index = (int)((particles_X[gid] + cellLength[0] * maxCellIndex[0] / 2 +
                       cuboidShift[0]) /
                      cellLength[0]);
  int y_index = (int)((particles_Y[gid] + cellLength[0] * maxCellIndex[0] / 2 +
                       cuboidShift[1]) /
                      cellLength[0]);
  int z_index = (int)((particles_Z[gid] + cellLength[0] * maxCellIndex[0] / 2 +
                       cuboidShift[2]) /
                      cellLength[0]);
  // Idx = 0 -> if particle is outside of the cuboid cell structure
  // Convert cell number into 1D vector

  if ((x_index < 0) || (x_index >= maxCellIndex[0])) {
    m_particle_collision_cell_index[gid] = 0;
    //printf("X. %i", x_index);
  } else if ((y_index < 0) || (y_index >= maxCellIndex[0])) {
    m_particle_collision_cell_index[gid] = 0;
    //printf("Y. %i", y_index);
  } else if ((z_index < 0) || (z_index >= maxCellIndex[0])) {
    m_particle_collision_cell_index[gid] = 0;
    //printf("Z. %i", z_index);

  } else {
    m_particle_collision_cell_index[gid] =
        1 + x_index + y_index * maxCellIndex[0] +
        z_index * maxCellIndex[0] * maxCellIndex[0];
  }
}

__kernel void assign_particle_cell_index_two_phase(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global unsigned int *m_particle_collision_cell_index,
    __global char *particles_bool_in_bubble, __global const int *maxCellIndex,
    __global const double *cellLength,
    __global const double *cuboidShift  // Random particle location shift

) {
  unsigned int gid = get_global_id(0);

  // Find cell numbers
  int a = (int)((particles_X[gid] + cuboidShift[0]) / cellLength[0]);
  int b = (int)((particles_Y[gid] + cuboidShift[1]) / cellLength[1]);
  int c = (int)((particles_Z[gid] + cuboidShift[2]) / cellLength[2]);
  // Idx = 0 -> if particle is outside of the cuboid cell structure
  // Convert cell number into 1D vector. First half of the vector is for outisde
  // the bubble and second half is inside the bubble
  if ((a < 0) || (a >= maxCellIndex[0])) {
    m_particle_collision_cell_index[gid] = 0;
  } else if ((b < 0) || (b >= maxCellIndex[1])) {
    m_particle_collision_cell_index[gid] = 0;
  } else if ((c < 0) || (c >= maxCellIndex[2])) {
    m_particle_collision_cell_index[gid] = 0;
  } else {
    m_particle_collision_cell_index[gid] =
        1 + a + b * maxCellIndex[0] + c * maxCellIndex[0] * maxCellIndex[1] +
        particles_bool_in_bubble[gid] * maxCellIndex[0] * maxCellIndex[1] *
            maxCellIndex[2];
  }
}

/*
============================================================================
============================================================================
                             Boundaries
============================================================================
============================================================================
*/

__kernel void particle_boundary_momentum_reflect(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global double *particles_pX,
    __global double *particles_pY, __global double *particles_pZ,
    __global double *t_boundaryRadius  // [x_delta]
) {
  unsigned int gid = get_global_id(0);

  double boundaryRadius = t_boundaryRadius[0];
  double x = particles_X[gid];
  double y = particles_Y[gid];
  double z = particles_Z[gid];
  double pX = particles_pX[gid];
  double pY = particles_pY[gid];
  double pZ = particles_pZ[gid];

  // If Particle is inside the boundary leave value same. Otherwise change the
  // sign abs(x) < Boundary -> leave momentum abs(x) > Boundary and x <
  // -Boundary -> Set momentum positive abs(x) > Boundary and x > Boundary ->
  // Set momentum negative
  pX = (boundaryRadius > fabs(x)) * pX +  // If inside, leave momentum alone
       ((boundaryRadius < fabs(x)) && (-boundaryRadius > x)) *
           fabs(pX) -  // particle too -, change to +
       ((boundaryRadius < fabs(x)) && (boundaryRadius < x)) *
           fabs(pX);  // particle too +, change to -
  pY = (boundaryRadius > fabs(y)) * pY +
       ((boundaryRadius < fabs(y)) && (-boundaryRadius > y)) * fabs(pY) -
       ((boundaryRadius < fabs(y)) && (boundaryRadius < y)) * fabs(pY);
  pZ = (boundaryRadius > fabs(z)) * pZ +
       ((boundaryRadius < fabs(z)) && (-boundaryRadius > z)) * fabs(pZ) -
       ((boundaryRadius < fabs(z)) && (boundaryRadius < z)) * fabs(pZ);

  particles_pX[gid] = pX;
  particles_pY[gid] = pY;
  particles_pZ[gid] = pZ;
}

__kernel void particle_boundary_check(__global double *particles_X,
                                      __global double *particles_Y,
                                      __global double *particles_Z,
                                      __global double *t_boundaryRadius) {
  unsigned int gid = get_global_id(0);

  double x = particles_X[gid];
  double y = particles_Y[gid];
  double z = particles_Z[gid];
  double boundaryRadius = t_boundaryRadius[0];

  x = (boundaryRadius > fabs(x)) * x +
      (x > boundaryRadius) * (x - 2 * boundaryRadius) +
      (x < -boundaryRadius) * (x + 2 * boundaryRadius);
  y = (boundaryRadius > fabs(y)) * y +
      (y > boundaryRadius) * (y - 2 * boundaryRadius) +
      (y < -boundaryRadius) * (y + 2 * boundaryRadius);
  z = (boundaryRadius > fabs(z)) * z +
      (z > boundaryRadius) * (z - 2 * boundaryRadius) +
      (z < -boundaryRadius) * (z + 2 * boundaryRadius);

  particles_X[gid] = x;
  particles_Y[gid] = y;
  particles_Z[gid] = z;
}