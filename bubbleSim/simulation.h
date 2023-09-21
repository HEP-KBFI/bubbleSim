#pragma once
#define _USE_MATH_DEFINES

#include <random>

#include "base.h"
#include "collision.h"
#include "objects.h"
#include "opencl_kernels.h"
#include "simulation_parameters.h"

using my_clock = std::chrono::steady_clock;

class Simulation {
 public:
  Simulation() { m_seed = 0; }
  Simulation(int t_seed, numType t_max_dt, SimulationParameters& t_simulation_parameters, cl::Context& cl_context);

  void addInitialTotalEnergy(numType energy) {
    assert(m_time == 0.);
    m_initialTotalEnergy += energy;
    m_totalEnergy += energy;
  }
  /*
   * ================================================================
   * ================================================================
   *                        Simulation step
   * ================================================================
   * ================================================================
   */

  void step(PhaseBubble& bubble, numType t_dP);
  /*
   * Step with calculated dP value
   */

  void stepParticleBubble(ParticleCollection& particles, PhaseBubble& bubble,
                          OpenCLLoader& t_kernels);

  void stepParticleBubbleBoundary(ParticleCollection& particles,
                                  PhaseBubble& bubble, OpenCLLoader& t_kernels);

  void collide(ParticleCollection& particles, CollisionCellCollection& cells,
               numType t_dt, numType t_tau, RandomNumberGeneratorNumType& t_rng,
               OpenCLLoader& t_kernels);

  void collide2(ParticleCollection& particles, CollisionCellCollection& cells,
                numType t_dt, numType t_tau,
                RandomNumberGeneratorNumType& t_rng_numtype,
                RandomNumberGeneratorULong& t_rng_int, OpenCLLoader& t_kernels);

  void collide3(ParticleCollection& particles, CollisionCellCollection& cells,
                numType t_dt, numType t_tau,
                RandomNumberGeneratorNumType& t_rng_numtype,
                RandomNumberGeneratorULong& t_rng_int, OpenCLLoader& t_kernels);

  void stepParticleCollisionBoundary(
      ParticleCollection& particles, CollisionCellCollection& cells,
      RandomNumberGeneratorNumType& t_rng_numtype,
      RandomNumberGeneratorULong& t_rng_int,
      OpenCLLoader& t_kernels);

  void stepParticleBubbleCollisionBoundary(
      ParticleCollection& particles, PhaseBubble& bubble,
      CollisionCellCollection& cells,
      RandomNumberGeneratorNumType& t_rng_numtype,
      RandomNumberGeneratorULong& t_rng_int,
      OpenCLLoader& t_kernels);

  /*
   * ================================================================
   * ================================================================
   *                            Setters
   * ================================================================
   * ================================================================
   */

  void setInitialCompactness(numType t_initialCompacntess) {
    m_initialCompactness = t_initialCompacntess;
  }

  void set_dt(numType t_dt) {
    if (t_dt <= 0.) {
      std::cerr << "Given dt value is <= 0." << std::endl;
      std::terminate();
    }
    m_dt = t_dt;
  };

  void setTau(numType t_tau) {
    if (t_tau <= 0.) {
      std::cerr << "Given tau value is <= 0." << std::endl;
      std::terminate();
    }
    m_tau = t_tau;
  }

  /*
   * ================================================================
   * ================================================================
   *                            Getters
   * ================================================================
   * ================================================================
   */

  unsigned int getStep() { return m_step_count; }

  numType getTime() { return m_time; }

  numType get_dt() { return m_dt; }

  numType get_dt_currentStep() { return m_dt_current; }

  numType get_dP() { return m_dP; }

  numType getTau() { return m_tau; }

  size_t getParticleCount() { return m_particleCount; }

  numType getTotalEnergy() { return m_totalEnergy; }

  numType getInitialTotalEnergy() { return m_initialTotalEnergy; }

  numType getInitialCompactnes() { return m_initialCompactness; }

  numType returnCumulativeDP() { 
      numType dP = m_cumulative_dP;
    m_cumulative_dP = 0.;
      return dP; }

  SimulationParameters& getSimulationParameters() { return m_parameters; }

 private:
  int m_seed;

  SimulationParameters m_parameters;

  // Simulation time state
  numType m_time = 0.;
  u_int m_step_count = 0;

  // One step time length
  numType m_dt;
  numType m_dt_current;
  numType m_dt_max;
  numType m_tau;

  // Simulation values
  numType m_initialCompactness = 0.;
  numType m_totalEnergy = 0.;
  numType m_initialTotalEnergy = 0.;

  // Current simulation state values
  numType m_dP = 0.;
  numType m_cumulative_dP = 0.;

  size_t m_particleCount;
  /*
   * Step with given dP value
   */
};