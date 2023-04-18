#include "simulation.h"

Simulation::Simulation(int t_seed, numType t_max_dt, cl::Context& cl_context) {
  int openCLerrNum = 0;
  m_seed = t_seed;
  m_dt = t_max_dt;
  m_step_dt = t_max_dt;
  m_timestepAdapter = TimestepAdapter(t_max_dt, t_max_dt);
  m_dP = 0.;
  m_dtBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(numType), &m_step_dt, &openCLerrNum);
  m_cyclicBoundaryOn = false;
}

Simulation::Simulation(int t_seed, numType t_max_dt, numType t_boundaryRadius,
                       cl::Context& cl_context) {
  int openCLerrNum = 0;
  m_seed = t_seed;
  m_dt = t_max_dt;
  m_step_dt = t_max_dt;
  m_timestepAdapter = TimestepAdapter(t_max_dt, t_max_dt);
  m_dP = 0.;
  m_dtBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(numType), &m_step_dt, &openCLerrNum);
  m_cyclicBoundaryOn = true;
  m_cyclicBoundaryRadius = t_boundaryRadius;
  m_cyclicBoundaryRadiusBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_cyclicBoundaryRadius, &openCLerrNum);
}

// No bubble
void Simulation::set_particle_step_buffers(ParticleCollection& t_particles,
                                           CollisionCellCollection& cells,
                                           cl::Kernel& t_particleStepKernel) {
  t_particleStepKernel.setArg(0, t_particles.getParticlesBuffer());
  t_particleStepKernel.setArg(1, cells.getStructureRadiusBuffer());
  t_particleStepKernel.setArg(2, m_dtBuffer);
};

// With a bubble
void Simulation::set_particle_step_buffers(
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    cl::Kernel& t_bubbleInteractionKernel) {
  int errNum;
  errNum =
      t_bubbleInteractionKernel.setArg(0, t_particles.getParticlesBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize particles buffer." << std::endl;
    std::terminate();
  }
  errNum = t_bubbleInteractionKernel.setArg(1, t_particles.get_dPBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize dP buffer." << std::endl;
    std::terminate();
  }
  errNum = t_bubbleInteractionKernel.setArg(
      2, t_particles.getInteractedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize InteractedFalse buffer." << std::endl;
    std::terminate();
  }
  errNum = t_bubbleInteractionKernel.setArg(
      3, t_particles.getPassedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize PassedFalse buffer." << std::endl;
    std::terminate();
  }
  errNum = t_bubbleInteractionKernel.setArg(
      4, t_particles.getInteractedBubbleTrueStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize InteractedTrue buffer." << std::endl;
    std::terminate();
  }
  errNum = t_bubbleInteractionKernel.setArg(5, t_bubble.getBubbleBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Bubble buffer." << std::endl;
    std::terminate();
  }
  errNum = t_bubbleInteractionKernel.setArg(6, m_dtBuffer);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize dt buffer." << std::endl;
    std::terminate();
  }
  errNum = t_bubbleInteractionKernel.setArg(7, t_particles.getMassInBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize mass_in buffer." << std::endl;
    std::terminate();
  }
  errNum = t_bubbleInteractionKernel.setArg(8, t_particles.getMassOutBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize mass_out buffer." << std::endl;
    std::terminate();
  }
  errNum =
      t_bubbleInteractionKernel.setArg(9, t_particles.getMassDelta2Buffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Dm2 buffer." << std::endl;
    std::terminate();
  }

  if (m_cyclicBoundaryOn) {
    errNum = t_bubbleInteractionKernel.setArg(10, m_cyclicBoundaryRadiusBuffer);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Couldn't initialize cyclic boundary radius buffer."
                << std::endl;
      std::terminate();
    }
  }
}

void Simulation::set_particle_interaction_buffers(
    ParticleCollection& t_particles, CollisionCellCollection& cells,
    cl::Kernel& t_cellAssignmentKernel, cl::Kernel& t_momentumRotationKernel) {
  t_cellAssignmentKernel.setArg(0, t_particles.getParticlesBuffer());
  t_cellAssignmentKernel.setArg(1, cells.getCellCountInOneAxisBuffer());
  t_cellAssignmentKernel.setArg(2, cells.getCellLengthBuffer());
  t_cellAssignmentKernel.setArg(3, cells.getShiftVectorBuffer());

  t_momentumRotationKernel.setArg(0, t_particles.getParticlesBuffer());
  t_momentumRotationKernel.setArg(1, cells.getCellBuffer());
  t_momentumRotationKernel.setArg(2, cells.getCellCountBuffer());
}

void Simulation::set_particle_bounce_buffers(ParticleCollection& t_particles,
                                             CollisionCellCollection& cells,
                                             cl::Kernel& t_particleStepKernel) {
  t_particleStepKernel.setArg(0, t_particles.getParticlesBuffer());
  t_particleStepKernel.setArg(1, cells.getStructureRadiusBuffer());
};

void Simulation::step(ParticleCollection& particles, PhaseBubble& bubble,
                      cl::Kernel& t_bubbleInteractionKernel,
                      cl::CommandQueue& cl_queue) {
  /*
   * 1) Move particles and do collision with phase bubble (GPU)
   * (1.1 Bounce particles back from some distance?)
   * 2) Calculate dP and evolve bubble
   * 3) Assign particles to collision cells (GPU)
   * 4) Calculate COM frame for particles in each cell. Generate rotation axis
   * and rotation angle for the collision cells 5) Perform "collisions" ->
   * Rotate particle momentum (GPU)
   */

  // 1) Update timestep
  numType currentStepEnergy = 0.;
  numType bubbleStartSpeed = bubble.getSpeed();
  m_step_dt = m_timestepAdapter.getTimestep();
  write_dtBuffer(cl_queue);

  // 2)
  // Write new bubble parameters to buffer on device
  bubble.calculateRadiusAfterStep2(m_step_dt);
  bubble.writeBubbleBuffer(cl_queue);

  // 3) Run kernel
  cl_queue.enqueueNDRangeKernel(t_bubbleInteractionKernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  // 3) Calculate how much "energy" particle get and then convert it to energy
  // bubble gets
  particles.readParticlesBuffer(cl_queue);
  particles.read_dPBuffer(cl_queue);

  m_dP = 0.;
  numType currentTotalEnergy = 0.;
  for (u_int i = 0; i < particles.getParticleCountTotal(); i++) {
    m_dP += particles.get_dP()[i];
    currentTotalEnergy += particles.getParticleEnergy(i);
  }
  m_dP = -m_dP / bubble.calculateArea();

  // 4) Evolve bubble
  bubble.evolveWall(m_step_dt, m_dP);

  currentTotalEnergy += bubble.calculateEnergy();
  numType bubbleStepFinalSpeed = bubble.getSpeed();
  numType bubbleSpeedChange = std::abs(bubbleStepFinalSpeed - bubbleStartSpeed);

  if ((bubbleSpeedChange > 0.02) && (m_step > 20)) {
    std::cout << "Problem: "
              << std::abs(bubbleStepFinalSpeed - bubbleStartSpeed) * 10 << ", "
              << m_step_dt << ", " << bubbleStartSpeed << ", "
              << bubbleStepFinalSpeed << std::endl;
    particles.revertToLastStep(cl_queue);
    bubble.revertBubbleToLastStep(cl_queue);
    m_timestepAdapter.claculateNewTimeStep(
        bubbleSpeedChange, bubble.getRadius(), bubble.getInitialRadius(),
        0.0001, bubble.getSpeed());
    step(particles, bubble, t_bubbleInteractionKernel, cl_queue);
  } else {
    m_timestepAdapter.claculateNewTimeStep(
        bubbleSpeedChange, bubble.getRadius(), bubble.getInitialRadius(),
        0.0001, bubble.getSpeed());
  }

  // 5) Collisions
  m_time += m_step_dt;
  m_step += 1;
  particles.makeCopy();
  bubble.makeBubbleCopy();
}

void Simulation::step(PhaseBubble& bubble, numType t_dP) {
  m_time += m_dt;
  bubble.evolveWall(m_dt, t_dP);
}

void Simulation::step(ParticleCollection& particles,
                      CollisionCellCollection& cells,
                      RandomNumberGenerator& generator_collision, int i,
                      cl::Kernel& t_particleStepKernel,
                      cl::Kernel& t_cellAssignmentKernel,
                      cl::Kernel& t_rotationKernel,
                      cl::Kernel& t_particleBounceKernel,
                      cl::CommandQueue& cl_queue) {
  m_time += m_dt;
  // Move particles
  cl_queue.enqueueNDRangeKernel(t_particleStepKernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  // Generate shift vector
  if (i % 1 == 0) {
    cells.generateShiftVector(generator_collision);
    cells.writeShiftVectorBuffer(cl_queue);

    // Assign particles to collision cells
    cl_queue.enqueueNDRangeKernel(
        t_cellAssignmentKernel, cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
    // Update particle data on CPU
    particles.readParticlesBuffer(cl_queue);

    // Calculate COM and genrate rotation matrix for each cell

    cells.recalculate_cells(particles.getParticles(), generator_collision);

    // Update data on GPU
    particles.writeParticlesBuffer(cl_queue);
    cells.writeCollisionCellBuffer(cl_queue);
    // Update momentum
    cl_queue.enqueueNDRangeKernel(
        t_rotationKernel, cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
  }
  cl_queue.enqueueNDRangeKernel(t_particleBounceKernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
}
