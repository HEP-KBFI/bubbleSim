#pragma once
#define _USE_MATH_DEFINES

#include <random>

#include "base.h"
#include "objects.h"
#include "opencl_kernels.h"

class Simulation {
 public:
  Simulation(int t_seed, numType t_dt, cl::Context& cl_context);

  numType getTime() { return m_time; }

  numType get_dt() { return m_dt; }

  cl::Buffer& get_dtBuffer() { return m_dtBuffer; }

  void set_dt(numType t_dt) {
    if (t_dt <= 0.) {
      std::cerr << "Set dt value is <= 0." << std::endl;
      std::terminate();
    }
  };

  numType get_dP() { return m_dP; }

  void addTotalEnergy(numType energy) { m_totalEnergy += energy; }

  void step(PhaseBubble& bubble, numType t_dP);
  /*
   * Step with calculated dP value
   */
  void step(ParticleCollection& particles, PhaseBubble& bubble,
            cl::Kernel& t_bubbleInteractionKernel, cl::CommandQueue& cl_queue);

  void set_bubble_interaction_buffers(ParticleCollection& t_particles,
                                      PhaseBubble& t_bubble,
                                      cl::Kernel& t_bubbleInteractionKernel);

  void read_dtBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_dtBuffer, CL_TRUE, 0, sizeof(numType), &m_dt);
  }

  void write_dtBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_dtBuffer, CL_TRUE, 0, sizeof(numType), &m_dt);
  }

 private:
  // Sim time paramters:
  // Cumulative time
  int m_seed;
  numType m_time;
  // One step time length
  numType m_dt;
  cl::Buffer m_dtBuffer;

  // Simulation values
  numType m_totalEnergy;

  // Current simulation state values
  numType m_dP;

  /*
   * Step with given dP value
   */
};