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
void Simulation::setBuffersParticleStepLinear(ParticleCollection& t_particles,
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
  Simulation::setBuffersParticleStepWithBubble(t_particles, t_bubble, t_kernel);
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

void Simulation::setBuffersCollisionCellReset(CollisionCellCollection t_cells,
                                              cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_cells.getCellBuffer());
}

void Simulation::setBuffersCollisionCellCalculateSummation(
    ParticleCollection& t_particles, CollisionCellCollection& cells,
    cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_particles.getParticleEBuffer());
  t_kernel.setArg(1, t_particles.getParticlepXBuffer());
  t_kernel.setArg(2, t_particles.getParticlepYBuffer());
  t_kernel.setArg(3, t_particles.getParticlepZBuffer());
  t_kernel.setArg(4, t_particles.getParticleCollisionCellIndexBuffer());
  t_kernel.setArg(5, cells.getCellBuffer());
}

void Simulation::setBuffersCollisionCellCalculateGeneration(
    CollisionCellCollection& cells, cl::Kernel& t_kernel) {
  t_kernel.setArg(0, cells.getCellBuffer());
  t_kernel.setArg(1, cells.getSeedBuffer());
  t_kernel.setArg(2, cells.getNoCollisionProbabilityBuffer());
}

/*
 * ================================================================
 * ================================================================
 *                        Simulation step
 * ================================================================
 * ================================================================
 */

// Step with bubble kernel
/*
 * Simple bubble step algorithm with bubble
 *
 * // TODO: Energy summation on the GPU
 * // TODO: dP summation on the GPU
 * // TODO: Simulation revert option
 */
void Simulation::stepParticleBubble(ParticleCollection& particles,
                                    PhaseBubble& bubble,
                                    cl::Kernel& t_particle_step_kernel,
                                    cl::CommandQueue& cl_queue) {
  numType dE;
  // m_timestepAdapter.calculateNewTimeStep(bubble);
  m_dt_current = m_timestepAdapter.getTimestep();
  writedtBuffer(cl_queue);

  cl_queue.enqueueNDRangeKernel(t_particle_step_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));

  particles.readParticleEBuffer(cl_queue);
  particles.readdPBuffer(cl_queue);

  m_dP = 0.;
  numType currentTotalEnergy = 0.;
  for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    m_dP += particles.returnParticledP(i);
    currentTotalEnergy += particles.returnParticleE(i);
  }
  dE = -m_dP * bubble.getSpeed();
  /*if (m_dP != 0) {
    std::cout << "dP: " << m_dP << std::endl;

    PhaseBubble testBubble(55.456826, 0.23236236, 2.83371e-05, 0.000345447);
    std::cout << "Energy change (bubble): " << -m_dP * testBubble.getSpeed()
              << std::endl;
    testBubble.evolveWall(0.975251, m_dP / testBubble.calculateArea());
    std::cout << testBubble.getRadius() << ", "
              << testBubble.getSpeed() * 0.975251
              << std::endl;


    exit(0);
  }*/

  m_dP = m_dP / bubble.calculateArea();
  bubble.evolveWall(m_dt_current, m_dP);
  currentTotalEnergy += bubble.calculateEnergy();
  // bubble.evolveWall2(m_dt_current, dE);
  // currentTotalEnergy += bubble.getEnergy();

  // std::cout << "= = = = = = = = = =" << std::endl;
  // std::cout << "Eb: " << bubble.calculateEnergy() << ", dt: " << m_dt_current
  // << std::endl; std::cout << "Velocity: "
  //           << std::sqrt(1 - std::pow(4 * M_PI * bubble.getSigma() *
  //                             std::pow(bubble.getRadius(), 2.) /
  //                             (bubble.calculateEnergy() +
  //                              4 * M_PI / 3 *
  //                              std::pow(bubble.getRadius(), 3.) *
  //                                  bubble.getdV()), 2.))
  //           << std::endl;
  // std::cout << "dP: " << m_dP << std::endl;
  // std::cout << "dV - dP: " << bubble.getdV() - m_dP / m_dt_current
  //           << std::endl;
  // std::cout << "V_b (before): " << bubble.getSpeed() << std::endl;
  // std::cout << "V_b (after): " << bubble.getSpeed() << std::endl;
  // std::cout << "= = = = = = = = = =" << std::endl;

  bubble.writeBubbleBuffer(cl_queue);

  // currentTotalEnergy += bubble.calculateEnergy();

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

  m_totalEnergy = currentTotalEnergy;
  m_time += m_dt_current;
  m_step_count += 1;
  // In development: step revert
  // particles.makeCopy();
  // bubble.makeBubbleCopy();
}

// Step with bubble kernel and boundary condition
void Simulation::stepParticleBubbleBoundary(
    ParticleCollection& particles, PhaseBubble& bubble,
    cl::Kernel& t_particle_step_kernel,
    cl::Kernel& t_particle_boundary_check_kernel, cl::CommandQueue& cl_queue) {
  Simulation::stepParticleBubble(particles, bubble, t_particle_step_kernel,
                                 cl_queue);

  cl_queue.enqueueNDRangeKernel(t_particle_boundary_check_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
}

// Step for only calculating bubble
void Simulation::step(PhaseBubble& bubble, numType t_dP) {
  m_time += m_dt;
  bubble.evolveWall(m_dt, t_dP);
}

void Simulation::collide(ParticleCollection& particles,
                         CollisionCellCollection& cells, numType t_dt,
                         numType t_tau, RandomNumberGeneratorNumType& t_rng,
                         cl::Kernel& t_assign_particle_to_collision_cell_kernel,
                         cl::Kernel& t_rotate_momentum_kernel,
                         cl::CommandQueue& cl_queue) {
  auto start = std::chrono::high_resolution_clock::now();
  cells.generateShiftVector(t_rng);
  cells.writeShiftVectorBuffer(cl_queue);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Shift vector generation and write time: " << duration.count() << std::endl;
  // Assign particles to collision cells
  start = std::chrono::high_resolution_clock::now();
  cl_queue.enqueueNDRangeKernel(t_assign_particle_to_collision_cell_kernel,
                                cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Collision cell assignation time: " << duration.count()
            << std::endl;
  // Update particle data on CPU
  start = std::chrono::high_resolution_clock::now();
  particles.readParticleCoordinatesBuffer(cl_queue);
  particles.readParticleMomentumsBuffer(cl_queue);
  particles.readParticleEBuffer(cl_queue);
  particles.readParticleCollisionCellIndexBuffer(cl_queue);
  // Calculate COM and genrate rotation matrix for each cell
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Read particle data to CPU time: " << duration.count()
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  cells.recalculate_cells(particles, t_dt, t_tau, t_rng);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Recalculate cell time: " << duration.count()
            << std::endl;
  /*start = std::chrono::high_resolution_clock::now();
  cells.recalculate_cells2(particles);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Recalculate cell2 time: " << duration.count() << std::endl;*/
  start = std::chrono::high_resolution_clock::now();
  cells.recalculate_cells3(particles);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Recalculate cell3 time: " << duration.count() << std::endl;

  // Update collision cell data on GPU
  start = std::chrono::high_resolution_clock::now();
  particles.writeParticleCollisionCellIndexBuffer(cl_queue);
  cells.writeCollisionCellBuffer(cl_queue);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Write data to GPU time: " << duration.count() << std::endl;
  // Rotate momentum
  start = std::chrono::high_resolution_clock::now();
  cl_queue.enqueueNDRangeKernel(t_rotate_momentum_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Rotation time: " << duration.count() << std::endl;
}

void Simulation::collide2(
    ParticleCollection& particles, CollisionCellCollection& cells, numType t_dt,
    numType t_tau, RandomNumberGeneratorNumType& t_rng_numtype,
    RandomNumberGeneratorULong& t_rng_int,
    cl::Kernel& t_assign_particle_to_collision_cell_kernel,
    cl::Kernel& t_rotate_momentum_kernel,
    cl::Kernel& t_reset_collision_cell_kernel,
    cl::Kernel& t_generate_collision_cell_kernel, cl::CommandQueue& cl_queue) {
  cells.generateShiftVector(t_rng_numtype);
  cells.writeShiftVectorBuffer(cl_queue);
  cl_queue.enqueueNDRangeKernel(t_assign_particle_to_collision_cell_kernel,
                                cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  particles.readParticleCollisionCellIndexBuffer(cl_queue);

  cells.calculate_new_no_collision_probability(t_dt, t_tau);
  cells.writeNoCollisionProbabilityBuffer(cl_queue);
  while (cells.getSeed() == 0) {
    cells.generateSeed(t_rng_int);
  }
  cells.writeSeedBuffer(cl_queue);

  cl_queue.enqueueNDRangeKernel(t_reset_collision_cell_kernel, cl::NullRange,
                                cl::NDRange(cells.getCellCount()));
  cells.readCollisionCellBuffer(cl_queue);
  auto start = std::chrono::high_resolution_clock::now();

  cells.recalculate_cells2(particles);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "= Recalcualte cell time: " << duration.count()
            << std::endl;

  cells.writeCollisionCellBuffer(cl_queue);
  
  /*cl_queue.enqueueNDRangeKernel(t_sum_collision_cell_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));*/
  cl_queue.enqueueNDRangeKernel(t_generate_collision_cell_kernel, cl::NullRange,
                                cl::NDRange(cells.getCellCount()));
  cl_queue.enqueueNDRangeKernel(t_rotate_momentum_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
}


void collide3(ParticleCollection& particles, CollisionCellCollection& cells,
    numType t_dt, numType t_tau,
    RandomNumberGeneratorNumType& t_rng_numtype,
    RandomNumberGeneratorULong& t_rng_int,
    cl::Kernel& t_assign_particle_to_collision_cell_kernel,
    cl::Kernel& t_rotate_momentum_kernel,
    cl::Kernel& t_reset_collision_cell_kernel,
    cl::Kernel& t_generate_collision_cell_kernel,
    cl::CommandQueue& cl_queue) {

  // Generate shift vector
  /*cells.generateShiftVector(t_rng_numtype);
  cells.writeShiftVectorBuffer(cl_queue);
  */
  cl_queue.enqueueNDRangeKernel(t_assign_particle_to_collision_cell_kernel,
                                cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  particles.readParticleCollisionCellIndexBuffer(cl_queue);

  // Generate probabilty that collision does not happen
  cells.calculate_new_no_collision_probability(t_dt, t_tau);
  cells.writeNoCollisionProbabilityBuffer(cl_queue);
  // Generate new seed (seed!=0 as algo does not work in that case)
  while (cells.getSeed() == 0) {
    cells.generateSeed(t_rng_int);
  }
  cells.writeSeedBuffer(cl_queue);
  // Reset collision cells
  cl_queue.enqueueNDRangeKernel(t_reset_collision_cell_kernel, cl::NullRange,
                                cl::NDRange(cells.getCellCount()));
  cells.readCollisionCellBuffer(cl_queue);
  cells.recalculate_cells3(particles);
  cells.writeCollisionCellBuffer(cl_queue);
  cl_queue.enqueueNDRangeKernel(t_generate_collision_cell_kernel, cl::NullRange,
                                cl::NDRange(cells.getCellCount()));
  cl_queue.enqueueNDRangeKernel(t_rotate_momentum_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
}

void Simulation::stepParticleCollisionBoundary(
    ParticleCollection& particles, CollisionCellCollection& cells,
    RandomNumberGeneratorNumType& t_rng_numtype,
    RandomNumberGeneratorULong& t_rng_int, cl::Kernel& t_particle_step_kernel,
    cl::Kernel& t_particle_boundary_check_kernel,
    cl::Kernel& t_assign_particle_to_collision_cell_kernel,
    cl::Kernel& t_rotate_momentum_kernel,
    cl::Kernel& t_reset_collision_cell_kernel,
    cl::Kernel& t_generate_collision_cell_kernel, cl::CommandQueue& cl_queue) {
  // m_timestepAdapter.calculateNewTimeStep(bubble);
  m_dt_current = m_timestepAdapter.getTimestep();
  writedtBuffer(cl_queue);
  auto start = std::chrono::high_resolution_clock::now();
  cl_queue.enqueueNDRangeKernel(t_particle_step_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Step time: " << duration.count() << std::endl;
  start = std::chrono::high_resolution_clock::now();
  cl_queue.enqueueNDRangeKernel(t_particle_boundary_check_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
   stop = std::chrono::high_resolution_clock::now();
   duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Boundary check time: " << duration.count() << std::endl;
   start = std::chrono::high_resolution_clock::now();
  collide(particles, cells, m_dt_current, getTau(), t_rng_numtype,
          t_assign_particle_to_collision_cell_kernel, t_rotate_momentum_kernel,
          cl_queue);

  /*collide2(particles, cells, m_dt_current, getTau(), t_rng_numtype, t_rng_int,
           t_assign_particle_to_collision_cell_kernel, t_rotate_momentum_kernel,
           t_reset_collision_cell_kernel, t_generate_collision_cell_kernel,
           cl_queue);*/
   stop = std::chrono::high_resolution_clock::now();
   duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
   std::cout << "Collision time: " << duration.count() << std::endl;
   start = std::chrono::high_resolution_clock::now();
  particles.readParticleEBuffer(cl_queue);
   stop = std::chrono::high_resolution_clock::now();
   duration =
       std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
   std::cout << "Particle Energy read time: " << duration.count() << std::endl;
   start = std::chrono::high_resolution_clock::now();
  numType currentTotalEnergy = 0.;
  for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    currentTotalEnergy += particles.returnParticleE(i);
  }
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Particle Energy count time: " << duration.count() << std::endl;
  std::cout << std::endl;
  m_totalEnergy = currentTotalEnergy;
  m_time += m_dt_current;
  m_step_count += 1;
}

void Simulation::stepParticleBubbleCollisionBoundary(
    ParticleCollection& particles, PhaseBubble& bubble,
    CollisionCellCollection& cells, RandomNumberGeneratorNumType& t_rng_numtype,
    RandomNumberGeneratorULong& t_rng_int, cl::Kernel& t_particle_step_kernel,
    cl::Kernel& t_particle_boundary_check_kernel,
    cl::Kernel& t_assign_particle_to_collision_cell_kernel,
    cl::Kernel& t_rotate_momentum_kernel,
    cl::Kernel& t_reset_collision_cell_kernel,
    cl::Kernel& t_generate_collision_cell_kernel, cl::CommandQueue& cl_queue) {
  numType dE;
  // m_timestepAdapter.calculateNewTimeStep(bubble);
  m_dt_current = m_timestepAdapter.getTimestep();
  writedtBuffer(cl_queue);

  cl_queue.enqueueNDRangeKernel(t_particle_step_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  cl_queue.enqueueNDRangeKernel(t_particle_boundary_check_kernel, cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  auto start = std::chrono::high_resolution_clock::now();
  /*collide(particles, cells, m_dt_current, getTau(), t_rng_numtype,
          t_assign_particle_to_collision_cell_kernel, t_rotate_momentum_kernel,
          cl_queue);*/
  collide2(particles, cells, m_dt_current, getTau(), t_rng_numtype, t_rng_int,
           t_assign_particle_to_collision_cell_kernel, t_rotate_momentum_kernel,
           t_reset_collision_cell_kernel,
           t_generate_collision_cell_kernel, cl_queue);

  particles.readParticleEBuffer(cl_queue);
  particles.readdPBuffer(cl_queue);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << duration.count() << std::endl;
  m_dP = 0.;
  numType currentTotalEnergy = 0.;
  for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    m_dP += particles.returnParticledP(i);
    currentTotalEnergy += particles.returnParticleE(i);
  }
  // Calculate energy change for bubble (m_dP is ~ energy change for particles)
  dE = -m_dP * bubble.getSpeed();
  m_dP = m_dP / bubble.calculateArea();

  bubble.evolveWall(m_dt_current, m_dP);
  currentTotalEnergy += bubble.calculateEnergy();

  // bubble.evolveWall2(m_dt_current, dE);
  // currentTotalEnergy += bubble.getEnergy();

  bubble.writeBubbleBuffer(cl_queue);

  m_totalEnergy = currentTotalEnergy;
  m_time += m_dt_current;
  m_step_count += 1;
}