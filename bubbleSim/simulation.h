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
  Simulation() { m_seed = 0; }
  Simulation(int t_seed, numType t_max_dt, cl::Context& cl_context);
  Simulation(int t_seed, numType t_max_dt, numType boundaryRadius,
             cl::Context& cl_context);

  void addInitialTotalEnergy(numType energy) {
    assert(m_time == 0.);
    m_initialTotalEnergy += energy;
    m_totalEnergy += energy;
  }

  /*
   * ================================================================
   * ================================================================
   *                        Kernel setup
   * ================================================================
   * ================================================================
   */
  void setBuffersParticleStepLinear(ParticleCollection& t_particles,
                                    cl::Kernel& t_kernel);

  void setBuffersParticleStepWithBubble(ParticleCollection& t_particles,
                                        PhaseBubble& t_bubble,
                                        cl::Kernel& t_bubbleInteractionKernel);

  void setBuffersParticleStepWithBubbleInverted(ParticleCollection& t_particles,
                                                PhaseBubble& t_bubble,
                                                cl::Kernel& t_kernel);

  void setBuffersParticleStepWithBubbleOnlyReflect(
      ParticleCollection& t_particles, PhaseBubble& t_bubble,
      cl::Kernel& t_kernel);

  void setBuffersParticleBoundaryCheck(ParticleCollection& t_particles,
                                       cl::Kernel& t_kernel);

  void setBuffersParticleBoundaryMomentumReflect(
      ParticleCollection& t_particles, cl::Kernel& t_kernel);

  void setBuffersRotateMomentum(ParticleCollection& t_particles,
                                CollisionCellCollection& cells,
                                cl::Kernel& t_kernel);

  void setBuffersAssignParticleToCollisionCell(
      ParticleCollection& t_particles, CollisionCellCollection& cells,
      cl::Kernel& t_cellAssignmentKernel);

  void setBuffersLabelParticleInBubbleCoordinate(
      ParticleCollection& t_particles, PhaseBubble& t_bubble,
      cl::Kernel& t_kernel);

  void setBuffersLabelParticleInBubbleMass(ParticleCollection& t_particles,
                                           cl::Kernel& t_kernel);
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

  void step(ParticleCollection& particles, PhaseBubble& bubble,
            cl::Kernel& t_bubbleInteractionKernel, cl::CommandQueue& cl_queue);

  void step(ParticleCollection& particles, CollisionCellCollection& cells,
            RandomNumberGenerator& generator_collision, int i,
            cl::Kernel& t_particleStepKernel,
            cl::Kernel& t_cellAssignmentKernel, cl::Kernel& t_rotationKernel,
            cl::Kernel& t_particleBounceKernel, cl::CommandQueue& cl_queue);

  void step(ParticleCollection& particles, PhaseBubble& bubble,
            cl::Kernel& t_particle_step_kernel,
            cl::Kernel& t_particle_boundary_check_kernel,
            cl::CommandQueue& cl_queue);

  void step(ParticleCollection& particles, PhaseBubble& bubble,
            CollisionCellCollection& cells, RandomNumberGenerator& t_rng,
            cl::Kernel& t_particle_step_kernel,
            cl::Kernel& t_particle_boundary_check_kernel,
            cl::Kernel& t_assign_particle_to_collision_cell_kernel,
            cl::Kernel& t_rotate_momentum_kernel, cl::CommandQueue& cl_queue);

  void collide(ParticleCollection& particles, CollisionCellCollection& cells,
               RandomNumberGenerator& generator_collision,
               cl::Kernel& t_assign_particle_to_collision_cell_kernel,
               cl::Kernel& t_rotate_momentum_kernel,
               cl::CommandQueue& cl_queue);

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
      std::cerr << "Set dt value is <= 0." << std::endl;
      std::terminate();
    }
  };

  /*
   * ================================================================
   * ================================================================
   *                            Getters
   * ================================================================
   * ================================================================
   */

  cl::Buffer& get_dtBuffer() { return m_dtBuffer; }

  unsigned int getStep() { return m_step_count; }

  numType getTime() { return m_time; }

  numType get_dt() { return m_dt; }

  numType get_dt_currentStep() { return m_dt_current; }

  numType get_dP() { return m_dP; }

  bool getCyclicBoundaryOn() { return m_boundaryOn; }

  size_t getParticleCount() { return m_particleCount; }

  numType getTotalEnergy() { return m_totalEnergy; }

  numType getInitialTotalEnergy() { return m_initialTotalEnergy; }

  numType getInitialCompactnes() { return m_initialCompactness; }

  /*
   * ================================================================
   * ================================================================
   *                        Buffer writers
   * ================================================================
   * ================================================================
   */

  void writedtBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_dtBuffer, CL_TRUE, 0, sizeof(numType),
                                &m_dt_current);
  }

  void writeAllBuffersToKernel(cl::CommandQueue& cl_queue) {
    writedtBuffer(cl_queue);
  }

  /*
   * ================================================================
   * ================================================================
   *                        Buffer readers
   * ================================================================
   * ================================================================
   */

  void readdtBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_dtBuffer, CL_TRUE, 0, sizeof(numType),
                               &m_dt_current);
  }

 private:
  int m_seed;

  // Simulation time state
  numType m_time = 0.;
  u_int m_step_count = 0;

  // One step time length
  numType m_dt;
  numType m_dt_current;
  numType m_dt_max;
  TimestepAdapter m_timestepAdapter;
  cl::Buffer m_dtBuffer;

  bool m_boundaryOn = false;
  numType m_boundaryRadius;
  cl::Buffer m_boundaryRadiusBuffer;

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