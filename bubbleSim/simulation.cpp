#include "simulation.h"

Simulation::Simulation(int t_seed, numType t_dt, cl::Context cl_context) {
  int openCLerrNum;
  m_seed = t_seed;
  m_dt = t_dt;
  m_dP = 0.;
  m_dtBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(numType), &t_dt, &openCLerrNum);
}

void Simulation::set_bubble_interaction_buffers(
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    cl::Kernel& t_bubbleInteractionKernel) {
  int errNum;
  errNum =
      t_bubbleInteractionKernel.setArg(0, t_particles.getParticlesBuffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize particles buffer." << std::endl;
  }
  errNum = t_bubbleInteractionKernel.setArg(1, t_particles.get_dPBuffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize dP buffer." << std::endl;
  }
  errNum = t_bubbleInteractionKernel.setArg(
      2, t_particles.getInteractedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize InteractedFalse buffer." << std::endl;
  }
  errNum = t_bubbleInteractionKernel.setArg(
      3, t_particles.getPassedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize PassedFalse buffer." << std::endl;
  }
  errNum = t_bubbleInteractionKernel.setArg(
      4, t_particles.getInteractedBubbleTrueStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize InteractedTrue buffer." << std::endl;
  }
  errNum = t_bubbleInteractionKernel.setArg(5, t_bubble.getBubbleBuffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize R buffer." << std::endl;
  }
  errNum = t_bubbleInteractionKernel.setArg(6, m_dtBuffer);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize dt buffer." << std::endl;
  }
  errNum = t_bubbleInteractionKernel.setArg(7, t_particles.getMassInBuffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize mass_in buffer." << std::endl;
  }
  errNum = t_bubbleInteractionKernel.setArg(8, t_particles.getMassOutBuffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize mass_out buffer." << std::endl;
  }
  errNum =
      t_bubbleInteractionKernel.setArg(9, t_particles.getMassDelta2Buffer());
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize Dm2 buffer." << std::endl;
  }
}

void Simulation::step(ParticleCollection& particles, PhaseBubble& bubble,
                      cl::Kernel& t_bubbleInteractionKernel,
                      cl::CommandQueue& cl_queue) {
  m_time += m_dt;
  // Write new bubble parameters to buffer on device
  bubble.calculateRadiusAfterStep2(m_dt);
  bubble.writeBubbleBuffer(cl_queue);
  // Run kernel
  cl_queue.enqueueNDRangeKernel(t_bubbleInteractionKernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  // Read dP vector and sum total change
  // dP is change for particles -> -dP change for bubble
  particles.read_dPBuffer(cl_queue);
  m_dP = 0.;
  for (numType dPi : particles.get_dP()) {
    m_dP += dPi;
  }
  // Evolve bubble
  bubble.evolveWall(m_dt, -m_dP / bubble.calculateArea());
  // Is it needed to write bubble now too?
  bubble.writeBubbleBuffer(cl_queue);
}

void Simulation::step(PhaseBubble& bubble, numType t_dP) {
  m_time += m_dt;
  bubble.evolveWall(m_dt, m_dPressureStep);
}