#pragma once
#define _USE_MATH_DEFINES

#include <random>

#include "base.h"
#include "bubble.h"
#include "objects.h"
#include "openclwrapper.h"

class Simulation {
  /*
  False at the end of variable means False vaccum (lower mass)
  True at the end of variable means False vaccum (higher mass)
  */

  // Simulation parameters
  numType m_coupling;
  // Masses of particles in true and false vacuum
  numType m_massTrue, m_massFalse, m_massDelta2;
  // Temperatures in true and false vacuum
  numType m_temperatureTrue, m_temperatureFalse;
  // Particle counts total / true vacuum / false vacuum
  int m_particleCountTotal, m_particleCountTrue, m_particleCountFalse;

  // Initial simulation values
  numType m_initialTotalEnergy;

  // Sim time paramters:
  // Cumulative time
  numType m_time;
  // One step time length
  numType m_dt;
  numType m_dPressureStep;

  // Data objects for buffers
  std::vector<Particle> m_particles;
  std::vector<numType> m_dP;
  std::vector<int8_t> m_interactedFalse;
  std::vector<int8_t> m_passedFalse;
  // True vaccum interaction also means that the particle gets through
  std::vector<int8_t> m_interactedTrue;

  // Random number generator
  int m_seed;

 public:
  Simulation() {}
  Simulation(int t_seed, numType t_massTrue, numType t_massFalse,
             numType t_temperatureTrue, numType t_temperatureFalse,
             unsigned int t_particleCountTrue,
             unsigned int t_particleCountFalse, numType t_coupling);
  Simulation& operator=(const Simulation& t) { return *this; }

  void set_dt(numType t_dt);

  std::vector<Particle>& getRef_Particles() { return m_particles; }
  std::vector<numType>& getRef_dP() { return m_dP; }
  std::vector<int8_t>& getRef_InteractedFalse() { return m_interactedFalse; }
  std::vector<int8_t>& getRef_PassedFalse() { return m_passedFalse; }
  std::vector<int8_t>& getRef_InteractedTrue() { return m_interactedTrue; }
  numType& getRef_dt() { return m_dt; }
  numType& getRef_MassFalse() { return m_massFalse; }
  numType& getRef_MassTrue() { return m_massTrue; }
  numType& getRef_MassDelta2() { return m_massDelta2; }

  numType getMassFalse() { return m_massFalse; }
  numType getMassTrue() { return m_massTrue; }
  numType getTime() { return m_time; }
  numType get_dt() { return m_dt; }
  numType getdPressureStep() { return m_dPressureStep; }
  int getParticleCountTotal() { return m_particleCountTotal; }
  int getParticleCountTrueInitial() { return m_particleCountTrue; }
  int getParticleCountFalseInitial() { return m_particleCountFalse; }

  // Particle functions
  numType getParticleEnergy(u_int i) { return m_particles[i].E; }
  numType getParticleMass(u_int i) { return m_particles[i].m; }
  numType calculateParticleRadius(u_int i);
  numType calculateParticleMomentum(u_int i);
  numType calculateParticleEnergy(u_int i);

  // Calculate distributions
  numType calculateNumberDensity(numType t_mass, numType t_temperature,
                                 numType t_dp, numType t_pMax);
  numType calculateEnergyDensity(numType t_mass, numType t_temperature,
                                 numType t_dp, numType t_pMax);

  void add_to_total_initial_energy(numType energy) {
    m_initialTotalEnergy += energy;
  }
  numType getInitialTotalEnergy() { return m_initialTotalEnergy; }

  // Get values from the simulation
  numType countParticleNumberDensity(numType t_radius1);
  numType countParticleNumberDensity(numType t_radius1, numType t_radius2);
  numType countParticleEnergyDensity(numType t_radius1);
  numType countParticleEnergyDensity(numType t_radius1, numType t_radius2);
  numType countParticlesEnergy();
  numType countParticlesEnergy(numType t_radius1);
  numType countParticlesEnergy(numType t_radius1, numType t_radius2);

  void step(PhaseBubble& bubble, numType t_dP);
  void step(PhaseBubble& bubble, OpenCLWrapper& openCLWrapper);
  /*
  void step(PhaseBubble bubble, OpenCLWrapper openCLWrapper, std::string
  device);


  void stepCPU(PhaseBubble bubble);

  void stepGPU(PhaseBubble bubble, OpenCLWrapper openCLWrapper);
  */
  /*
          Runs one time on GPU or CPU.
  */
};