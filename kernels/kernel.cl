#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
void __attribute__((always_inline))
atomic_add_d(volatile global double *addr, const double val) {
  union {
    ulong u64;
    double f64;
  } next, expected, current;
  current.f64 = *addr;
  do {
    next.f64 =
        (expected.f64 = current.f64) + val;  // ...*val for atomic_mul_d()
    current.u64 =
        atom_cmpxchg((volatile global ulong *)addr, expected.u64, next.u64);
  } while (current.u64 != expected.u64);
}

// Bubble type (also defined in c++ code)
typedef struct Bubble {
  double radius;
  double radius2;  // Squared
  double speed;
  double gamma;
  double gammaXspeed;  // gamma * speed
} Bubble;

ulong xorshift64(ulong state) {
  ulong random_number = state;
  random_number ^= random_number >> 12;
  random_number ^= random_number << 25;
  random_number ^= random_number >> 27;
  return random_number * (ulong)2685821657736338717;
}

/*
 * Given particle and it's velocity move it's location by time=dt.
 * Also this function already updates particle location.
 */

double moveLinear(double x, double v, double dt) {
  // x = x + v*dt
  return x + v * dt;
}

// TODO: Remove if clauses? Make into single expression. Also check if time1 <
// time2.
double calculateTimeToWall(double x, double y, double z, double E, double pX,
                           double pY, double pZ, Bubble bubble, double t_dt) {
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
  if ((0 < time1) && (fabs(time1) <= t_dt)) {
    return time1;
  } else if ((0 < time2) && (fabs(time2) <= t_dt)) {
    return time2;
  }
  // Solution might exist but not for current step -> no solution between 0 <
  // time < dt
  else {
    return 0.;
  }
}

double calculateTimeToWall_DEBUG(double x, double y, double z, double E,
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
  printf("Time1: %.20f, Time2: %.20f", time1, time2);
  if ((0 < time1) && (time1 <= t_dt)) {
    printf(", Selected: %.20f\n", time1);
    return time1;
  } else if ((0 < time2) && (time2 <= t_dt)) {
    printf(", Selected: %.20f\n", time2);
    return time2;
  }
  // Solution might exist but not for current step -> no solution between 0 <
  // time < dt
  else {
    printf(", Selected: %.20f\n", 0.);
    return 0.;
  }
}

// TODO: Return only one sign normal and do sign change in the kernel code or
// somewhere else. All kernels are calculated directed "outside".
void calculateNormal(double *t_n1, double *t_n2, double *t_n3, double x,
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
  return sqrt(fma(pX, pX, fma(pY, pY, fma(pZ, pZ, pow(mass, 2.)))));
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
  size_t gid = get_global_id(0);

  double E = particles_E[gid];
  double vX = particles_pX[gid] / E;
  double vY = particles_pY[gid] / E;
  double vZ = particles_pZ[gid] / E;
  double dt = t_dt[0];

  particles_X[gid] = moveLinear(particles_X[gid], vX, dt);
  particles_Y[gid] = moveLinear(particles_Y[gid], vY, dt);
  particles_Z[gid] = moveLinear(particles_Z[gid], vZ, dt);
}

__kernel void particle_step_with_bubble(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global double *particles_E,
    __global double *particles_pX, __global double *particles_pY,
    __global double *particles_pZ, __global double *particles_M,
    __global double *t_dE, __global char *t_interactedFalse,
    __global char *t_passedFalse, __global char *t_interactedTrue,
    __constant Bubble *t_bubble, __constant double *t_m_in,
    __constant double *t_m_out, __constant double *t_delta_m2,
    __constant double *t_dt) {
  size_t gid = get_global_id(0);
  /*
   * t_dE is a array where each element is "pressure" by particle respective to
   * it's index. t_dE is not actual pressure but actually energy change ΔE/V_b.
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

  double X_dt2 = calculateDistanceSquaredFromCenter(x_Vdt, y_Vdt, z_Vdt);
  // M in < M out -> False vacuum bubble inside
  if (M_in < M_out) {
    // Particle starts inside the bubble
    if (X2 < bubble.radius2) {
      // Particle stays inside the bubble
      if (X_dt2 < pow(bubble.radius + bubble.speed * dt, 2.)) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dE[gid] = 0.;
      }
      // Particle can get out
      else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear(x, vX, time_to_wall);
        y = moveLinear(y, vY, time_to_wall);
        z = moveLinear(z, vZ, time_to_wall);

        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);

        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        // Particle bounces back from the bubble wall
        if (np * np < Delta_M2) {
          pX = fma(np * 2., nX, pX);
          pY = fma(np * 2., nY, pY);
          pZ = fma(np * 2., nZ, pZ);
          t_dE[gid] = bubble.gamma * np * 2.;
          E = calculateParticleEnergy(pX, pY, pZ, M_in);
          t_interactedFalse[gid] += 1;
        }
        // Particle penetrates the bubble wall
        else {
          particles_M[gid] = M_out;
          pX = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nX, pX);
          pY = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nY, pY);
          pZ = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nZ, pZ);
          t_dE[gid] =
              bubble.gamma * np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.)));
          E = calculateParticleEnergy(pX, pY, pZ, M_out);
          t_interactedFalse[gid] += 1;
          t_passedFalse[gid] += 1;
        }
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        x = moveLinear(x, vX, dt - time_to_wall);
        y = moveLinear(y, vY, dt - time_to_wall);
        z = moveLinear(z, vZ, dt - time_to_wall);
      }
    }
    // Particle starts outside
    else {
      // Particle stays outsided
      if (X_dt2 > pow(bubble.radius + bubble.speed * dt, 2.)) {
        x = moveLinear(x, vX, dt);
        y = moveLinear(y, vY, dt);
        z = moveLinear(z, vZ, dt);
        t_dE[gid] = 0.;
      }
      // Particle penetrates the bubble wall
      else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear(x, vX, time_to_wall);
        y = moveLinear(y, vY, time_to_wall);
        z = moveLinear(z, vZ, time_to_wall);

        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        particles_M[gid] = M_in;
        pX = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nX, pX);
        pY = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nY, pY);
        pZ = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nZ, pZ);
        t_dE[gid] =
            bubble.gamma * np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.)));
        E = calculateParticleEnergy(pX, pY, pZ, M_in);
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        x = moveLinear(x, vX, dt - time_to_wall);
        y = moveLinear(y, vY, dt - time_to_wall);
        z = moveLinear(z, vZ, dt - time_to_wall);
        t_interactedTrue[gid] += 1;
      }
    }
  } else {
    // Particle starts outside the bubble
    if (X2 > bubble.radius2) {
      // Particle stays outside
      if (X_dt2 > pow(bubble.radius + bubble.speed * dt, 2.)) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dE[gid] = 0.;
      }
      // Particle can get out
      else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear(x, vX, time_to_wall);
        y = moveLinear(y, vY, time_to_wall);
        z = moveLinear(z, vZ, time_to_wall);

        // x = 0.672283;
        // y = 56.043979;
        // z = -18.621177;
        // bubble.speed = 0.100000;
        // bubble.gamma = 1.005038;
        // E = 0.342819;
        // pX = 0.073634;
        // pY = -0.105717;
        // pZ = 0.317532;
        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        // printf("Energy-momentum: %.6f\n", E*E - pX*pX - pY*pY - pZ*pZ);
        // printf("V: %.5f\n", bubble.speed);
        calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);
        // printf("n0: %.6f, nX: %.6f, nY: %.6f, nZ: %.6f\n",
        // bubble.speed*bubble.gamma, nX, nY, nZ);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        // Particle bounces back from the bubble wall
        if (np < 0) {
          printf("ERROR: np < 0, gid %i\n", gid);
        }

        if (np * np < Delta_M2) {
          // printf("Osake põrkub, np: %.6f\n", np);
          particles_M[gid] = M_out;
          // printf("X: %.6f, Y: %.6f, Z: %.6f\n", x, y, z);
          // printf("Initial: E: %.6f, pX: %.6f, pY: %.6f, pZ: %.6f\n", E, pX,
          // pY, pZ);

          pX = fma(np * 2., nX, pX);
          pY = fma(np * 2., nY, pY);
          pZ = fma(np * 2., nZ, pZ);
          t_dE[gid] = bubble.gamma * np * 2.;

          // printf("Energy change (particle): %.6f\n",
          // calculateParticleEnergy(pX, pY, pZ, M_out) - E); printf("Energy
          // change (bubble): %.6f\n", -t_dE[gid]*bubble.speed); printf("dP:
          // %.5f\n", t_dE[gid]);
          E = calculateParticleEnergy(pX, pY, pZ, M_out);
          // printf("End: E: %.6f, pX: %.6f, pY: %.6f, pZ: %.6f\n", E, pX, pY,
          // pZ); printf("dE: %.5f\n", t_dE[gid]*bubble.speed);

          // np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
          t_interactedFalse[gid] += 1;
        }
        // Particle penetrates the bubble wall
        else {
          // printf("Osake liigub sisse, np: %.6f\n", np);
          particles_M[gid] = M_in;
          // pX = fma((np - sqrt(np * np - Delta_M2)), nX, pX);
          // pY = fma((np - sqrt(np * np - Delta_M2)), nY, pY);
          // pZ = fma((np - sqrt(np * np - Delta_M2)), nZ, pZ);
          // t_dE[gid] = bubble.gamma * (np - sqrt(np*np - Delta_M2));
          // printf("Initial: E: %.6f, pX: %.6f, pY: %.6f, pZ: %.6f\n", E, pX,
          // pY, pZ);
          pX = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nX, pX);
          pY = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nY, pY);
          pZ = fma(np * (1. - sqrt(1. - Delta_M2 / pow(np, 2.))), nZ, pZ);
          t_dE[gid] =
              bubble.gamma * np * (1 - sqrt(1 - Delta_M2 / pow(np, 2.)));
          // printf("Energy change: %.6f\n", E - calculateParticleEnergy(pX, pY,
          // pZ, M_out)); printf("Energy change: %.6f\n",
          // -t_dE[gid]*bubble.speed); printf("dP: %.5f\n", t_dE[gid]);
          E = calculateParticleEnergy(pX, pY, pZ, M_in);
          // printf("End: E: %.6f, pX: %.6f, pY: %.6f, pZ: %.6f\n", E, pX, pY,
          // pZ); printf("dE: %.5f\n", t_dE[gid]*bubble.speed);
          t_interactedFalse[gid] += 1;
          t_passedFalse[gid] += 1;
        }
        // printf("New momentum: %.6f, %.6f, %.6f, %.6f\n", E, pX, pY, pZ);
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        x = moveLinear(x, vX, dt - time_to_wall);
        y = moveLinear(y, vY, dt - time_to_wall);
        z = moveLinear(z, vZ, dt - time_to_wall);
      }
    }
    // Particle starts inside the bubble
    else {
      // Particle stays inside
      if (X_dt2 < pow(bubble.radius + bubble.speed * dt, 2.)) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dE[gid] = 0.;
      }
      // Particle penetrates the bubble wall
      else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear(x, vX, time_to_wall);
        y = moveLinear(y, vY, time_to_wall);
        z = moveLinear(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        particles_M[gid] = M_out;
        // pX = fma((np - sqrt(pow(np, 2.) + Delta_M2)), nX, pX);
        // pY = fma((np - sqrt(pow(np, 2.) + Delta_M2)), nY, pY);
        // pZ = fma((np - sqrt(pow(np, 2.) + Delta_M2)), nZ, pZ);
        // t_dE[gid] = bubble.gamma * (np - sqrt(np*np. + Delta_M2));
        pX = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nX, pX);
        pY = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nY, pY);
        pZ = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nZ, pZ);
        t_dE[gid] =
            bubble.gamma * np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.)));
        E = calculateParticleEnergy(pX, pY, pZ, M_out);
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        x = moveLinear(x, vX, dt - time_to_wall);
        y = moveLinear(y, vY, dt - time_to_wall);
        z = moveLinear(z, vZ, dt - time_to_wall);
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
    __global double *t_dE, __global char *t_interactedFalse,
    __global char *t_passedFalse, __global char *t_interactedTrue,
    __constant Bubble *t_bubble, __constant double *t_m_in,
    __constant double *t_m_out, __constant double *t_delta_m2,
    __constant double *t_dt) {
  size_t gid = get_global_id(0);
  /*
   * t_dE is a array where each element is "pressure" by particle respective to
   * it's index. t_dE is not actual pressure but actually energy change ΔE.
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
      if (X_dt2 < pow(bubble.radius + bubble.speed * dt, 2.)) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dE[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear(x, vX, time_to_wall);
        y = moveLinear(y, vY, time_to_wall);
        z = moveLinear(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);
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
          t_dE[gid] = bubble.gamma * np * 2.;
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
          t_dE[gid] =
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
        x = moveLinear(x, vX, dt - time_to_wall);
        y = moveLinear(y, vY, dt - time_to_wall);
        z = moveLinear(z, vZ, dt - time_to_wall);
      }
    } else {
      if (X_dt2 > pow(bubble.radius + bubble.speed * dt, 2.)) {
        x = moveLinear(x, vX, dt);
        y = moveLinear(y, vY, dt);
        z = moveLinear(z, vZ, dt);
        t_dE[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear(x, vX, time_to_wall);
        y = moveLinear(y, vY, time_to_wall);
        z = moveLinear(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);

        calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);
        nX = -nX;
        nY = -nY;
        nZ = -nZ;

        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;

        particles_M[gid] = M_in;
        pX = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nX, pX);
        pY = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nY, pY);
        pZ = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nZ, pZ);

        // Calculate applied pressure
        t_dE[gid] =
            bubble.gamma * np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.)));
        E = calculateParticleEnergy(pX, pY, pZ, M_in);

        // Update particle velocity and move amount of (dt - timeToWall)
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // Movement after the bubble interaction
        x = moveLinear(x, vX, dt - time_to_wall);
        y = moveLinear(y, vY, dt - time_to_wall);
        z = moveLinear(z, vZ, dt - time_to_wall);
        // Particle interacts from the higher mass side
        t_interactedTrue[gid] += 1;
      }
    }
  } else {
    if (X2 > bubble.radius2) {
      if (X_dt2 > pow(bubble.radius + bubble.speed * dt, 2.)) {
        x = x_Vdt;
        y = y_Vdt;
        z = z_Vdt;
        t_dE[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear(x, vX, time_to_wall);
        y = moveLinear(y, vY, time_to_wall);
        z = moveLinear(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);
        calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;
        if (np * np < Delta_M2) {
          // Update particle 4-momentum after collision
          pX = fma(np * 2., nX, pX);
          pY = fma(np * 2., nY, pY);
          pZ = fma(np * 2., nZ, pZ);
          // Calculate applied pressure
          t_dE[gid] = bubble.gamma * np * 2.;
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
          t_dE[gid] =
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
        x = moveLinear(x, vX, dt - time_to_wall);
        y = moveLinear(y, vY, dt - time_to_wall);
        z = moveLinear(z, vZ, dt - time_to_wall);
      }
    } else {
      if (X_dt2 < pow(bubble.radius + bubble.speed * dt, 2.)) {
        x = moveLinear(x, vX, dt);
        y = moveLinear(y, vY, dt);
        z = moveLinear(z, vZ, dt);
        t_dE[gid] = 0.;
      } else {
        double nX, nY, nZ;
        double time_to_wall, np;
        time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
        x = moveLinear(x, vX, time_to_wall);
        y = moveLinear(y, vY, time_to_wall);
        z = moveLinear(z, vZ, time_to_wall);
        X2 = calculateDistanceSquaredFromCenter(x, y, z);

        calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);
        np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;

        particles_M[gid] = M_out;
        pX = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nX, pX);
        pY = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nY, pY);
        pZ = fma(np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.))), nZ, pZ);

        // Calculate applied pressure
        t_dE[gid] =
            bubble.gamma * np * (1. - sqrt(1. + Delta_M2 / pow(np, 2.)));
        E = calculateParticleEnergy(pX, pY, pZ, M_in);
        // Update particle velocity and move amount of (dt - timeToWall)
        vX = pX / E;
        vY = pY / E;
        vZ = pZ / E;
        // Movement after the bubble interaction
        x = moveLinear(x, vX, dt - time_to_wall);
        y = moveLinear(y, vY, dt - time_to_wall);
        z = moveLinear(z, vZ, dt - time_to_wall);
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

__kernel void particles_step_with_false_bubble_reflect(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z, __global double *particles_E,
    __global double *particles_pX, __global double *particles_pY,
    __global double *particles_pZ, __global double *t_dE,
    __global char *t_interactedFalse, __global char *t_passedFalse,
    __global char *t_interactedTrue, __constant Bubble *t_bubble,
    __constant double *t_m_in, __constant double *t_m_out,
    __constant double *t_delta_m2, __constant double *t_dt) {
  size_t gid = get_global_id(0);
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
    if ((X_dt2 < pow(bubble.radius + bubble.speed * dt, 2.)) &&
        (M_in < M_out)) {
      x = x_Vdt;
      y = y_Vdt;
      z = z_Vdt;
      t_dE[gid] = 0.;
    }
    // reflect particles that would to get out
    else {
      double nX, nY, nZ;
      double time_to_wall, np;
      time_to_wall = calculateTimeToWall(x, y, z, E, pX, pY, pZ, bubble, dt);
      x = moveLinear(x, vX, time_to_wall);
      y = moveLinear(y, vY, time_to_wall);
      z = moveLinear(z, vZ, time_to_wall);
      X2 = calculateDistanceSquaredFromCenter(x, y, z);

      // Update particle's radius squared value to calculate normal vector

      // Relativistic reflection algorithm
      // Calculate normal at the location where particle and bubble interact
      calculateNormal(&nX, &nY, &nZ, x, y, z, bubble, X2);
      np = bubble.speed * bubble.gamma * E - nX * pX - nY * pY - nZ * pZ;

      pX = fma(np * 2., nX, pX);
      pY = fma(np * 2., nY, pY);
      pZ = fma(np * 2., nZ, pZ);

      t_dE[gid] = bubble.gamma * np * 2.;
      // Update particle energy
      E = calculateParticleEnergy(pX, pY, pZ, M_in);
      // Count particle collision with the bubble wall
      t_interactedFalse[gid] += 1;

      // Update velocity vector
      vX = pX / E;
      vY = pY / E;
      vZ = pZ / E;

      // ========== Movement after the bubble interaction ==========
      x = moveLinear(x, vX, time_to_wall);
      y = moveLinear(y, vY, time_to_wall);
      z = moveLinear(z, vZ, time_to_wall);
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
__kernel void collision_cell_reset(
    __global double *cell_theta_axis, __global double *cell_phi_axis,
    __global double *cell_theta_rotation, __global double *cell_E,
    __global double *cell_pX, __global double *cell_pY,
    __global double *cell_pZ, __global char *cell_collide_boolean,
    __global double *cell_logE, __global uint *cell_particle_count) {
  // Loop over collision cells and reset it's values.
  size_t gid = get_global_id(0);
  cell_theta_axis[gid] = 0.;
  cell_phi_axis[gid] = 0.;
  cell_theta_rotation[gid] = 0.;
  cell_E[gid] = 0.;
  cell_pX[gid] = 0.;
  cell_pY[gid] = 0.;
  cell_pZ[gid] = 0.;
  cell_collide_boolean[gid] = (char)0;
  cell_logE[gid] = 0.;
  cell_particle_count[gid] = 0;
}

__kernel void collision_cell_sum_particles(
    __global double *particles_E, __global double *particles_pX,
    __global double *particles_pY, __global double *particles_pZ,
    __global unsigned int *particle_collision_cell_index,
    __global double *cell_E, __global double *cell_pX, __global double *cell_pY,
    __global double *cell_pZ, __global double *cell_logE,
    __global uint *cell_particle_count) {
  // Loop over particles and sum neccessary values
  size_t gid = get_global_id(0);
  uint cell_index = particle_collision_cell_index[gid];
  if (cell_index != 0) {
    atomic_add_d(&cell_E[cell_index], particles_E[gid]);
    atomic_add_d(&cell_pX[cell_index], particles_pX[gid]);
    atomic_add_d(&cell_pY[cell_index], particles_pY[gid]);
    atomic_add_d(&cell_pZ[cell_index], particles_pZ[gid]);
    atomic_add_d(&cell_logE[cell_index], log(particles_E[gid]));
    atomic_inc(&cell_particle_count[cell_index]);
  }
}

__kernel void collision_cell_generate_collisions(
    __global double *cell_theta_axis, __global double *cell_phi_axis,
    __global double *cell_theta_rotation, __global double *cell_E,
    __global double *cell_pX, __global double *cell_pY,
    __global double *cell_pZ, __global char *cell_collide_boolean,
    __global double *cell_logE, __global uint *cell_particle_count,
    __global ulong *seeds, __global double *no_collision_probability) {
  size_t gid = get_global_id(0);
  if (cell_particle_count[gid] < 2 || gid == 0) {
    cell_collide_boolean[gid] = (char)0;
  } else {
    double mass = pow(cell_E[gid], 2.) - pow(cell_pX[gid], 2.) -
                  pow(cell_pY[gid], 2.) - pow(cell_pZ[gid], 2.);
    if (mass != 0.) {
      double probability;
      // Maximum value for ulong: 18446744073709551616.0;
      // Calculating probability not to collide based on timestep dt and
      // thermalization constant tau. Probability = exp(-dt/tau) -> if dt=0 then
      // never collide
      seeds[gid] = xorshift64(seeds[gid]);
      probability = (double)seeds[gid] / 18446744073709551616.0;

      if (probability <= no_collision_probability[0]) {
        cell_collide_boolean[gid] = (char)0;
      } else {
        // Calculating probability to collide based on particles' energies
        // P(E) = (Sum[E_i]/N)^N/(Prod[E_i]) * constant
        seeds[gid] = xorshift64(seeds[gid]);
        probability = (double)seeds[gid] / 18446744073709551616.0;
        if (probability <= exp(-exp((double)cell_particle_count[gid] *
                                        (log(cell_E[gid]) -
                                         log(3.0 * cell_particle_count[gid])) -
                                    cell_logE[gid]))) {
          cell_collide_boolean[gid] = (char)0;
        } else {
          cell_collide_boolean[gid] = (char)1;
          if (cell_particle_count[gid] > 2) {
            printf("Miks oled suurem kui 2? %i\n", gid);
          }

          seeds[gid] = xorshift64(seeds[gid]);
          probability = (double)seeds[gid] / 18446744073709551616.0;
          cell_theta_rotation[gid] = 2. * M_PI * probability;

          seeds[gid] = xorshift64(seeds[gid]);
          probability = (double)seeds[gid] / 18446744073709551616.0;
          cell_phi_axis[gid] = acos(2. * probability - 1.);

          seeds[gid] = xorshift64(seeds[gid]);
          probability = (double)seeds[gid] / 18446744073709551616.0;
          cell_theta_axis[gid] = 2. * M_PI * probability;
        }
      }
    }
  }
}

__kernel void rotate_momentum(
    __global double *particles_E, __global double *particles_pX,
    __global double *particles_pY, __global double *particles_pZ,
    __global unsigned int *particle_collision_cell_index,
    __global double *cell_theta_axis, __global double *cell_phi_axis,
    __global double *cell_theta_rotation, __global double *cell_E,
    __global double *cell_pX, __global double *cell_pY,
    __global double *cell_pZ, __global char *cell_collide_boolean) {
  size_t gid = get_global_id(0);
  uint cell_idx = particle_collision_cell_index[gid];
  // if (!cell_collide_boolean[cell_idx]) {
  //   particle_collision_cell_index[gid] = 0;
  // } else
  if (cell_collide_boolean[cell_idx] && cell_idx != 0) {
    // === Collision cell variables
    double x = sin(cell_phi_axis[cell_idx]) * cos(cell_theta_axis[cell_idx]);
    double y = sin(cell_phi_axis[cell_idx]) * sin(cell_theta_axis[cell_idx]);
    double z = cos(cell_phi_axis[cell_idx]);

    double E_cell = cell_E[cell_idx];
    double vX = cell_pX[cell_idx] / E_cell;
    double vY = cell_pY[cell_idx] / E_cell;
    double vZ = cell_pZ[cell_idx] / E_cell;

    double v2 = fma(vX, vX, fma(vY, vY, vZ * vZ));  // v^2
    // printf("%.5f, %.5f, %.5f, %.5f, %.5f, %i\n", E_cell, cell_pX[gid],
    // cell_pY[gid], cell_pZ[gid], v2,
    // cell_collide_boolean[particle_collision_cell_index[gid]]);
    double gamma = 1 / sqrt(1 - v2);
    double gamma_minus_one = gamma - 1;
    double gamma_minus_one_divided_cell_v2 = gamma_minus_one / v2;

    double cos_theta = cos(cell_theta_rotation[cell_idx]);
    double one_minus_cos_theta = 1 - cos_theta;
    double sin_theta = sin(cell_theta_rotation[cell_idx]);
    // Cell mass = E^2 - p^2
    // If cell mass = 0 we can't rotate as no frame of refernce. E^2 - p^2 = 0
    // -> 1 - v^2 = 0
    if (1 - v2 != 0) {
      double E = particles_E[gid];
      double pX_1 = particles_pX[gid];
      double pY_1 = particles_pY[gid];
      double pZ_1 = particles_pZ[gid];
      double pX_2, pY_2, pZ_2;
      // Lorentz transformation (to COM frame)
      pX_2 = -E * gamma * vX +
             pX_1 * (1 + vX * vX * gamma_minus_one_divided_cell_v2) +
             pY_1 * vX * vY * gamma_minus_one_divided_cell_v2 +
             pZ_1 * vX * vZ * gamma_minus_one_divided_cell_v2;

      pY_2 = -gamma * E * vY +
             pY_1 * (1 + vY * vY * gamma_minus_one_divided_cell_v2) +
             pX_1 * vX * vY * gamma_minus_one_divided_cell_v2 +
             pZ_1 * vY * vZ * gamma_minus_one_divided_cell_v2;

      pZ_2 = -gamma * E * vZ +
             pZ_1 * (1 + vZ * vZ * gamma_minus_one_divided_cell_v2) +
             pX_1 * vX * vZ * gamma_minus_one_divided_cell_v2 +
             pY_1 * vY * vZ * gamma_minus_one_divided_cell_v2;
      E = gamma * (E - pX_1 * vX - pY_1 * vY - pZ_1 * vZ);
      // Rotate 3-momentum
      pX_1 = pX_2 * (cos_theta + pow(x, 2) * one_minus_cos_theta) +
             pY_2 * (x * y * one_minus_cos_theta - z * sin_theta) +
             pZ_2 * (x * z * one_minus_cos_theta + y * sin_theta);
      pY_1 = pX_2 * (x * y * one_minus_cos_theta + z * sin_theta) +
             pY_2 * (cos_theta + pow(y, 2) * one_minus_cos_theta) +
             pZ_2 * (y * z * one_minus_cos_theta - x * sin_theta);
      pZ_1 = pX_2 * (x * z * one_minus_cos_theta - y * sin_theta) +
             pY_2 * (y * z * one_minus_cos_theta + x * sin_theta) +
             pZ_2 * (cos_theta + pow(z, 2) * one_minus_cos_theta);
      // Lorentz inverse transformation (to initial frame)
      pX_2 = gamma * E * vX +
             pX_1 * (1 + vX * vX * gamma_minus_one_divided_cell_v2) +
             pY_1 * vX * vY * gamma_minus_one_divided_cell_v2 +
             pZ_1 * vX * vZ * gamma_minus_one_divided_cell_v2;
      pY_2 = gamma * E * vY +
             pY_1 * (1 + vY * vY * gamma_minus_one_divided_cell_v2) +
             pX_1 * vX * vY * gamma_minus_one_divided_cell_v2 +
             pZ_1 * vY * vZ * gamma_minus_one_divided_cell_v2;
      pZ_2 = gamma * E * vZ +
             pZ_1 * (1 + vZ * vZ * gamma_minus_one_divided_cell_v2) +
             pX_1 * vX * vZ * gamma_minus_one_divided_cell_v2 +
             pY_1 * vY * vZ * gamma_minus_one_divided_cell_v2;
      E = gamma * (E + pX_1 * vX + pY_1 * vY + pZ_1 * vZ);
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
  size_t gid = get_global_id(0);
  double x = particles_X[gid];
  double y = particles_Y[gid];
  double z = particles_Z[gid];
  // If R_b^2 > R_x^2 then particle is inside the bubble
  particles_bool_in_bubble[gid] =
      (char)(fma(x, x, fma(y, y, z * z)) < t_bubble[0].radius2);
}

__kernel void label_particles_position_by_mass(
    __global double *particles_M, __global char *particle_in_bubble,
    __global double *mass_in) {
  size_t gid = get_global_id(0);

  // If R_b^2 > R_x^2 then particle is inside the bubble
  particle_in_bubble[gid] = particles_M[gid] == mass_in[0];
}

// CollisionHack
__kernel void assign_particle_to_collision_cell(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z,
    __global unsigned int *m_particle_collision_cell_index,
    __global const unsigned int *maxCellIndexInAxis,
    __global const double *cellLength, __global const double *cuboidShift,
    __global uint *cell_particle_count, __global uint *cell_duplication) {
  size_t gid = get_global_id(0);
  unsigned int collision_cell_index;
  // Find cell numbers
  int x_index =
      (int)((particles_X[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
             cuboidShift[0]) /
            cellLength[0]);
  int y_index =
      (int)((particles_Y[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
             cuboidShift[1]) /
            cellLength[0]);
  int z_index =
      (int)((particles_Z[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
             cuboidShift[2]) /
            cellLength[0]);
  // Idx = 0 -> if particle is outside of the cuboid cell structure
  // Convert cell number into 1D vector
  if ((x_index < 0) || (x_index >= maxCellIndexInAxis[0])) {
    collision_cell_index = 0;
  } else if ((y_index < 0) || (y_index >= maxCellIndexInAxis[0])) {
    collision_cell_index = 0;
  } else if ((z_index < 0) || (z_index >= maxCellIndexInAxis[0])) {
    collision_cell_index = 0;
  } else {
    collision_cell_index =
        1 + x_index + y_index * maxCellIndexInAxis[0] +
        z_index * maxCellIndexInAxis[0] * maxCellIndexInAxis[0];
  }

  // HARD CODED VALUE: SHIFT_MAX = 50
  if (collision_cell_index != 0) {
    uint old_value = atomic_inc(&cell_particle_count[collision_cell_index]);
    uint shift_value = old_value / 2;
    if (shift_value < cell_duplication[0]) {
      collision_cell_index += shift_value * maxCellIndexInAxis[0] *
                              maxCellIndexInAxis[0] * maxCellIndexInAxis[0];
    } else {
      collision_cell_index = 0;
    }
  }

  m_particle_collision_cell_index[gid] = collision_cell_index;
}

// __kernel void assign_particle_to_collision_cell(
//     __global double *particles_X, __global double *particles_Y,
//     __global double *particles_Z,
//     __global unsigned int *m_particle_collision_cell_index,
//     __global const unsigned int *maxCellIndexInAxis,
//     __global const double *cellLength, __global const double *cuboidShift) {
//   size_t gid = get_global_id(0);
//   // Find cell numbers
//   int x_index =
//       (int)((particles_X[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
//              cuboidShift[0]) /
//             cellLength[0]);
//   int y_index =
//       (int)((particles_Y[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
//              cuboidShift[1]) /
//             cellLength[0]);
//   int z_index =
//       (int)((particles_Z[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
//              cuboidShift[2]) /
//             cellLength[0]);
//   // Idx = 0 -> if particle is outside of the cuboid cell structure
//   // Convert cell number into 1D vector
//   if ((x_index < 0) || (x_index >= maxCellIndexInAxis[0])) {
//     m_particle_collision_cell_index[gid] = 0;
//   } else if ((y_index < 0) || (y_index >= maxCellIndexInAxis[0])) {
//     m_particle_collision_cell_index[gid] = 0;
//   } else if ((z_index < 0) || (z_index >= maxCellIndexInAxis[0])) {
//     m_particle_collision_cell_index[gid] = 0;
//   } else {
//     m_particle_collision_cell_index[gid] =
//         1 + x_index + y_index * maxCellIndexInAxis[0] +
//         z_index * maxCellIndexInAxis[0] * maxCellIndexInAxis[0];
//   }
// }

// CollisionHack
__kernel void assign_particle_to_collision_cell_two_state(
    __global double *particles_X, __global double *particles_Y,
    __global double *particles_Z,
    __global unsigned int *m_particle_collision_cell_index,
    __global char *particles_bool_in_bubble,
    __global const int *maxCellIndexInAxis, __global const double *cellLength,
    __global const double *cuboidShift, __global uint *cell_particle_count,
    __global uint *cell_duplication) {
  size_t gid = get_global_id(0);
  unsigned int collision_cell_index;
  // Find cell numbers
  int x_index =
      (int)((particles_X[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
             cuboidShift[0]) /
            cellLength[0]);
  int y_index =
      (int)((particles_Y[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
             cuboidShift[1]) /
            cellLength[0]);
  int z_index =
      (int)((particles_Z[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
             cuboidShift[2]) /
            cellLength[0]);
  // Idx = 0 -> if particle is outside of the cuboid cell structure
  // Convert cell number into 1D vector. First half of the vector is for outisde
  // the bubble and second half is inside the bubble
  if ((x_index < 0) || (x_index >= maxCellIndexInAxis[0])) {
    collision_cell_index = 0;
  } else if ((y_index < 0) || (y_index >= maxCellIndexInAxis[0])) {
    collision_cell_index = 0;
  } else if ((z_index < 0) || (z_index >= maxCellIndexInAxis[0])) {
    collision_cell_index = 0;
  } else {
    collision_cell_index =
        1 + x_index + y_index * maxCellIndexInAxis[0] +
        z_index * maxCellIndexInAxis[0] * maxCellIndexInAxis[0] +
        particles_bool_in_bubble[gid] * maxCellIndexInAxis[0] *
            maxCellIndexInAxis[0] * maxCellIndexInAxis[0];
  }

  // HARD CODED VALUE: SHIFT_MAX = 50

  if (collision_cell_index != 0) {
    uint old_value = atomic_inc(&cell_particle_count[collision_cell_index]);
    uint shift_value = old_value / 2;
    if (shift_value < cell_duplication[0]) {
      collision_cell_index += shift_value * 2 * maxCellIndexInAxis[0] *
                              maxCellIndexInAxis[0] * maxCellIndexInAxis[0];
    } else {
      collision_cell_index = 0;
    }
  }

  m_particle_collision_cell_index[gid] = collision_cell_index;
}

// __kernel void assign_particle_to_collision_cell_two_state(
//     __global double *particles_X, __global double *particles_Y,
//     __global double *particles_Z,
//     __global unsigned int *m_particle_collision_cell_index,
//     __global char *particles_bool_in_bubble,
//     __global const int *maxCellIndexInAxis, __global const double
//     *cellLength,
//     __global const double *cuboidShift  // Random particle location shift

// ) {
//   size_t gid = get_global_id(0);

//   // Find cell numbers
//   int x_index =
//       (int)((particles_X[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
//              cuboidShift[0]) /
//             cellLength[0]);
//   int y_index =
//       (int)((particles_Y[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
//              cuboidShift[1]) /
//             cellLength[0]);
//   int z_index =
//       (int)((particles_Z[gid] + cellLength[0] * maxCellIndexInAxis[0] / 2. +
//              cuboidShift[2]) /
//             cellLength[0]);
//   // Idx = 0 -> if particle is outside of the cuboid cell structure
//   // Convert cell number into 1D vector. First half of the vector is for
//   outisde
//   // the bubble and second half is inside the bubble
//   if ((x_index < 0) || (x_index >= maxCellIndexInAxis[0])) {
//     m_particle_collision_cell_index[gid] = 0;
//   } else if ((y_index < 0) || (y_index >= maxCellIndexInAxis[0])) {
//     m_particle_collision_cell_index[gid] = 0;
//   } else if ((z_index < 0) || (z_index >= maxCellIndexInAxis[0])) {
//     m_particle_collision_cell_index[gid] = 0;
//   } else {
//     m_particle_collision_cell_index[gid] =
//         1 + x_index + y_index * maxCellIndexInAxis[0] +
//         z_index * maxCellIndexInAxis[0] * maxCellIndexInAxis[0] +
//         particles_bool_in_bubble[gid] * maxCellIndexInAxis[0] *
//             maxCellIndexInAxis[0] * maxCellIndexInAxis[0];
//   }
// }

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
  size_t gid = get_global_id(0);

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
  size_t gid = get_global_id(0);

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