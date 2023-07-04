#include "simulation.h"

Simulation::Simulation(int t_seed, numType t_max_dt, cl::Context& cl_context) {
  int openCLerrNum = 0;
  m_seed = t_seed;
  m_dt = t_max_dt;
  m_dt_current = t_max_dt;
  m_dt_max = t_max_dt;
  m_timestepAdapter = TimestepAdapter(t_max_dt, t_max_dt);
  m_dP = 0.;
  m_dtBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(numType), &m_dt_current, &openCLerrNum);
  m_boundaryOn = false;
}

Simulation::Simulation(int t_seed, numType t_max_dt, numType t_boundaryRadius,
                       cl::Context& cl_context) {
  int openCLerrNum = 0;
  m_seed = t_seed;
  m_dt = t_max_dt;
  m_dt_current = t_max_dt;
  m_dt_max = t_max_dt;
  m_timestepAdapter = TimestepAdapter(t_max_dt, t_max_dt);
  m_dP = 0.;
  m_dtBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(numType), &m_dt_current, &openCLerrNum);
  m_boundaryOn = true;
  m_boundaryRadius = t_boundaryRadius;
  m_boundaryRadiusBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_boundaryRadius, &openCLerrNum);
}

/*
 * ================================================================
 * ================================================================
 *                        Kernel setup
 * ================================================================
 * ================================================================
 */

// Just linear movement
void Simulation::setBuffersParticleStepLinear(
    ParticleCollection& t_particles, cl::Kernel& t_kernel) {
  int errNum;
  errNum = t_kernel.setArg(0, t_particles.getParticleXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(1, t_particles.getParticleYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(2, t_particles.getParticleZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(3, t_particles.getParticleEBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's energy buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(4, t_particles.getParticlepXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(5, t_particles.getParticlepYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(6, t_particles.getParticlepZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(7, m_dtBuffer);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize timestep buffer." << std::endl;
    std::terminate();
  }
};

// With a bubble
void Simulation::setBuffersParticleStepWithBubble(
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    cl::Kernel& t_kernel) {
  int errNum;
  errNum = t_kernel.setArg(0, t_particles.getParticleXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(1, t_particles.getParticleYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(2, t_particles.getParticleZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(3, t_particles.getParticleEBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's energy buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(4, t_particles.getParticlepXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(5, t_particles.getParticlepYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(6, t_particles.getParticlepZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(7, t_particles.getParticleMBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's mass buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(8, t_particles.getdPBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's pressure (dP) buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      t_kernel.setArg(9, t_particles.getInteractedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(10, t_particles.getPassedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) and passing buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      t_kernel.setArg(11, t_particles.getInteractedBubbleTrueStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "true vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(12, t_bubble.getBubbleBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Bubble buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(13, t_particles.getMassInBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass In buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(14, t_particles.getMassOutBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass Out buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(15, t_particles.getDeltaMassSquaredBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Delta Mass Squared buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(16, m_dtBuffer);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize timestep buffer." << std::endl;
    std::terminate();
  }
}

// Mass inside the false vacuum is bigger
void Simulation::setBuffersParticleStepWithBubbleInverted(
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    cl::Kernel& t_kernel) {
  Simulation::setBuffersParticleStepWithBubble(t_particles, t_bubble,
                                                    t_kernel);
}

// With a bubble but all particles reflect back from the bubble wall
void Simulation::setBuffersParticleStepWithBubbleOnlyReflect(
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    cl::Kernel& t_kernel) {
  int errNum;
  errNum = t_kernel.setArg(0, t_particles.getParticleXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(1, t_particles.getParticleYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(2, t_particles.getParticleZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(3, t_particles.getParticleEBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's energy buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(4, t_particles.getParticlepXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(5, t_particles.getParticlepYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(6, t_particles.getParticlepZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(7, t_particles.getdPBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's pressure (dP) buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      t_kernel.setArg(8, t_particles.getInteractedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(9, t_particles.getPassedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) and passing buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      t_kernel.setArg(10, t_particles.getInteractedBubbleTrueStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "true vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(11, t_bubble.getBubbleBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Bubble buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(12, t_particles.getMassInBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass In buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(13, t_particles.getMassOutBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass Out buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(14, t_particles.getDeltaMassSquaredBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Delta Mass Squared buffer." << std::endl;
    std::terminate();
  }
  errNum = t_kernel.setArg(15, m_dtBuffer);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize timestep buffer." << std::endl;
    std::terminate();
  }
}

void Simulation::setBuffersParticleBoundaryCheck(
    ParticleCollection& t_particles, cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_particles.getParticleXBuffer());
  t_kernel.setArg(1, t_particles.getParticleYBuffer());
  t_kernel.setArg(2, t_particles.getParticleZBuffer());
  t_kernel.setArg(3, m_boundaryRadiusBuffer);
};

void Simulation::setBuffersParticleBoundaryMomentumReflect(
    ParticleCollection& t_particles, cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_particles.getParticleXBuffer());
  t_kernel.setArg(1, t_particles.getParticleYBuffer());
  t_kernel.setArg(2, t_particles.getParticleZBuffer());
  t_kernel.setArg(3, t_particles.getParticlepXBuffer());
  t_kernel.setArg(4, t_particles.getParticlepYBuffer());
  t_kernel.setArg(5, t_particles.getParticlepZBuffer());
  t_kernel.setArg(6, m_boundaryRadiusBuffer);
}

void Simulation::setBuffersRotateMomentum(ParticleCollection& t_particles,
                                             CollisionCellCollection& t_cells,
                                             cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_particles.getParticleEBuffer());
  t_kernel.setArg(1, t_particles.getParticlepXBuffer());
  t_kernel.setArg(2, t_particles.getParticlepYBuffer());
  t_kernel.setArg(3, t_particles.getParticlepZBuffer());
  t_kernel.setArg(4, t_particles.getParticleCollisionCellIndexBuffer());
  t_kernel.setArg(5, t_cells.getCellBuffer());
}

void Simulation::setBuffersAssignParticleToCollisionCell(
    ParticleCollection& t_particles, CollisionCellCollection& t_cells,
    cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_particles.getParticleXBuffer());
  t_kernel.setArg(1, t_particles.getParticleYBuffer());
  t_kernel.setArg(2, t_particles.getParticleZBuffer());
  t_kernel.setArg(3, t_particles.getParticleCollisionCellIndexBuffer());
  t_kernel.setArg(4, t_cells.getCellCountInOneAxisBuffer());
  t_kernel.setArg(5, t_cells.getCellLengthBuffer());
  t_kernel.setArg(6, t_cells.getShiftVectorBuffer());
}

void Simulation::setBuffersLabelParticleInBubbleCoordinate(
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_particles.getParticleXBuffer());
  t_kernel.setArg(1, t_particles.getParticleYBuffer());
  t_kernel.setArg(2, t_particles.getParticleZBuffer());
  t_kernel.setArg(3, t_particles.getParticleInBubbleBuffer());
  t_kernel.setArg(4, t_bubble.getBubbleBuffer());
}

void Simulation::setBuffersLabelParticleInBubbleMass(
    ParticleCollection& t_particles, cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_particles.getParticleMBuffer());
  t_kernel.setArg(1, t_particles.getParticleInBubbleBuffer());
  t_kernel.setArg(2, t_particles.getMassInBuffer());
}

/*
 * ================================================================
 * ================================================================
 *                        Simulation step
 * ================================================================
 * ================================================================
 */

// Step with bubble kernel
void Simulation::step(ParticleCollection& particles, PhaseBubble& bubble,
                      cl::Kernel& t_particle_step_kernel,
                      cl::CommandQueue& cl_queue) {
  // 1) Update timestep
  // numType bubbleStartSpeed = bubble.getSpeed();
  m_timestepAdapter.calculateNewTimeStep(bubble);
  m_dt_current = m_timestepAdapter.getTimestep();

  // Move timestep to GPU
  writedtBuffer(cl_queue);
  // 2) Update data on GPU
  bubble.calculateRadiusAfterStep2(m_dt_current);
  bubble.writeBubbleBuffer(cl_queue);

  // 3) Run kernel (move particles on GPU)
  cl_queue.enqueueNDRangeKernel(t_particle_step_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  // 4 Read particle data from GPU after the step

  // TODO: Energy summation on the GPU
  // TODO: dP summation on the GPU
  particles.readParticleEBuffer(cl_queue);
  particles.readdPBuffer(cl_queue);

  // Calculate particle energy change -> pressure on the wall
  m_dP = 0.;
  numType currentTotalEnergy = 0.;
  for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    m_dP += particles.returnParticledP(i);
    currentTotalEnergy += particles.returnParticleE(i);
  }
  // Convert energy change to pressure
  m_dP = -m_dP / bubble.calculateArea();

  // 5) Evolve bubble
  bubble.evolveWall(m_dt_current, m_dP);
  currentTotalEnergy += bubble.calculateEnergy();

  /* In development: adaptive timestep ; step revert
  numType bubbleStepFinalSpeed = bubble.getSpeed();
  numType bubbleSpeedChange = std::abs(bubbleStepFinalSpeed - bubbleStartSpeed);
  if ((bubbleStartSpeed > 0) && (bubbleStartSpeed - bubbleStepFinalSpeed > 0)) {
    if (bubbleSpeedChange > 0.00005) {
      particles.revertToLastStep(cl_queue);
      bubble.revertBubbleToLastStep(cl_queue);
      m_timestepAdapter.calculateNewTimeStep();
      step(particles, bubble, t_bubbleInteractionKernel, cl_queue);
    }
  } else if (bubbleSpeedChange > 0.005) {
    particles.revertToLastStep(cl_queue);
    bubble.revertBubbleToLastStep(cl_queue);
    m_timestepAdapter.calculateNewTimeStep();
    step(particles, bubble, t_bubbleInteractionKernel, cl_queue);
  } else if (std::abs(bubbleStepFinalSpeed) > 1) {
    particles.revertToLastStep(cl_queue);
    bubble.revertBubbleToLastStep(cl_queue);
    m_timestepAdapter.calculateNewTimeStep();
    step(particles, bubble, t_bubbleInteractionKernel, cl_queue);
  } else {
    m_timestepAdapter.calculateNewTimeStep(bubble);
  }
  */

  // Update simulation info
  m_totalEnergy = currentTotalEnergy;
  m_time += m_dt_current;
  m_step_count += 1;

  // In development: step revert
  // particles.makeCopy();
  // bubble.makeBubbleCopy();
}

// Step with bubble kernel and boundary condition
void Simulation::step(ParticleCollection& particles, PhaseBubble& bubble,
    cl::Kernel& t_particle_step_kernel,
    cl::Kernel& t_particle_boundary_check_kernel,
    cl::CommandQueue& cl_queue) {
  Simulation::step(particles, bubble, t_particle_step_kernel, cl_queue);
  cl_queue.enqueueNDRangeKernel(t_particle_boundary_check_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
}

// Step for only calculating bubble
void Simulation::step(PhaseBubble& bubble, numType t_dP) {
  m_time += m_dt;
  bubble.evolveWall(m_dt, t_dP);
}

// Step with collisions
void Simulation::step(ParticleCollection& particles,
                      CollisionCellCollection& cells,
                      RandomNumberGenerator& generator_collision, int i,
                      cl::Kernel& t_particle_step_kernel,
                      cl::Kernel& t_assign_particle_to_collision_cell_kernel,
                      cl::Kernel& t_rotate_momentum_kernel,
                      cl::Kernel& t_particle_boundary_check_kernel,
                      cl::CommandQueue& cl_queue) {
  // Move particles

  cl_queue.enqueueNDRangeKernel(t_particle_step_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  particles.readParticleXBuffer(cl_queue);

  cl_queue.enqueueNDRangeKernel(t_particle_boundary_check_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  if (i % 1 == 0) {
    // Generate shift vector
    cells.generateShiftVector(generator_collision);
    cells.writeShiftVectorBuffer(cl_queue);
    // Assign particles to collision cells
    cl_queue.enqueueNDRangeKernel(
        t_assign_particle_to_collision_cell_kernel, cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
    // Update particle data on CPU
    particles.readParticleCoordinatesBuffer(cl_queue);
    particles.readParticleMomentumsBuffer(cl_queue);
    particles.readParticleEBuffer(cl_queue);
    particles.readParticleCollisionCellIndexBuffer(cl_queue);
    // Calculate COM and genrate rotation matrix for each cell
    cells.recalculate_cells(particles, generator_collision);
    // Update collision cell data on GPU
    particles.writeParticleCollisionCellIndexBuffer(cl_queue);
    cells.writeCollisionCellBuffer(cl_queue);

    // Rotate momentum
    cl_queue.enqueueNDRangeKernel(
        t_rotate_momentum_kernel, cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
    // particles.readParticlesBuffer(cl_queue);
    //  Update simulation time
    m_time += m_dt_current;
    m_step_count += 1;
  }
}
