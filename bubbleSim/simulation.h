#pragma once
#define _USE_MATH_DEFINES

#include <random>

#include "base.h"
#include "collision.h"
#include "objects.h"
#include "opencl_kernels.h"
#include "timestep.h"

class Simulation {
 public:
  Simulation() {}
  Simulation(int t_seed, numType t_max_dt, cl::Context& cl_context);
  Simulation(int t_seed, numType t_max_dt, numType boundaryRadius,
             cl::Context& cl_context);

  numType getTime() { return m_time; }

  numType get_dt() { return m_dt; }

  numType get_dt_currentStep() { return m_step_dt; }

  numType get_dP() { return m_dP; }

  bool getCyclicBoundaryOn() { return m_cyclicBoundaryOn; }

  size_t getParticleCount() { return m_particleCount; }

  numType getTotalEnergy() { return m_totalEnergy; }

  numType getInitialTotalEnergy() { return m_initialTotalEnergy; }

  numType getInitialCompactnes() { return m_initialCompactness; }

  void setInitialCompactness(numType t_initialCompacntess) {
    m_initialCompactness = t_initialCompacntess;
  }

  void addInitialTotalEnergy(numType energy) {
    assert(m_time == 0.);
    m_initialTotalEnergy += energy;
    m_totalEnergy += energy;
  }

  void set_dt(numType t_dt) {
    if (t_dt <= 0.) {
      std::cerr << "Set dt value is <= 0." << std::endl;
      std::terminate();
    }
  };

  unsigned int getStep() { return m_step; }

  void step(PhaseBubble& bubble, numType t_dP);
  /*
   * Step with calculated dP value
   */
  void step(ParticleCollection& particles, PhaseBubble& bubble,
            cl::Kernel& t_bubbleInteractionKernel, cl::CommandQueue& cl_queue);

  void step(ParticleCollection& particles, CollisionCellCollection& cells,
            RandomNumberGenerator& generator_collision, int i,
            cl::Kernel& t_particleStepKernel,
            cl::Kernel& t_cellAssignmentKernel, cl::Kernel& t_rotationKernel,
            cl::Kernel& t_particleBounceKernel, cl::CommandQueue& cl_queue);

  void set_particle_step_buffers(ParticleCollection& t_particles,
                                 PhaseBubble& t_bubble,
                                 cl::Kernel& t_bubbleInteractionKernel);

  void set_particle_interaction_buffers(ParticleCollection& t_particles,
                                        CollisionCellCollection& cells,
                                        cl::Kernel& t_cellAssignmentKernel,
                                        cl::Kernel& t_momentumRotationKernel);

  void set_particle_step_buffers(ParticleCollection& t_particles,
                                 CollisionCellCollection& cells,
                                 cl::Kernel& t_particleStepKernel);

  void set_particle_bounce_buffers(ParticleCollection& t_particles,
                                   CollisionCellCollection& cells,
                                   cl::Kernel& t_particleStepKernel);

  cl::Buffer& get_dtBuffer() { return m_dtBuffer; }

  void read_dtBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_dtBuffer, CL_TRUE, 0, sizeof(numType),
                               &m_step_dt);
  }

  void write_dtBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_dtBuffer, CL_TRUE, 0, sizeof(numType),
                                &m_step_dt);
  }

  void writeAllBuffersToKernel(cl::CommandQueue& cl_queue) {
    write_dtBuffer(cl_queue);
  }

 private:
  int m_seed;

  // Simulation time state
  numType m_time = 0.;
  u_int m_step = 0;

  // One step time length
  numType m_dt;
  numType m_step_dt;
  TimestepAdapter m_timestepAdapter;
  cl::Buffer m_dtBuffer;

  bool m_cyclicBoundaryOn = false;
  numType m_cyclicBoundaryRadius;
  cl::Buffer m_cyclicBoundaryRadiusBuffer;

  // Simulation values
  numType m_initialCompactness = 0.;
  numType m_totalEnergy = 0.;
  numType m_initialTotalEnergy = 0.;

  // Current simulation state values
  numType m_dP = 0.;

  size_t m_particleCount;
  /*
   * Step with given dP value
   */
};