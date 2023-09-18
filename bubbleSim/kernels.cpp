#include "kernels.h"

void MomentumRotationKernel::setBuffers(ParticleCollection& t_particles,
                                        CollisionCellCollection& t_cells) {
  m_kernel.setArg(0, t_particles.getParticleEBuffer());
  m_kernel.setArg(1, t_particles.getParticlepXBuffer());
  m_kernel.setArg(2, t_particles.getParticlepYBuffer());
  m_kernel.setArg(3, t_particles.getParticlepZBuffer());
  m_kernel.setArg(4, t_particles.getParticleCollisionCellIndexBuffer());
  m_kernel.setArg(5, t_cells.getCellThetaAxisBuffer());
  m_kernel.setArg(6, t_cells.getCellPhiAxisBuffer());
  m_kernel.setArg(7, t_cells.getCellThetaRotationBuffer());
  m_kernel.setArg(8, t_cells.getCellEBuffer());
  m_kernel.setArg(9, t_cells.getCellpXBuffer());
  m_kernel.setArg(10, t_cells.getCellpYBuffer());
  m_kernel.setArg(11, t_cells.getCellpZBuffer());
  m_kernel.setArg(12, t_cells.getCellCollideBooleanBuffer());
}

void MomentumRotationKernel::setBuffers(ParticleCollection& t_particles,
                                        CollisionCellCollection& t_cells,
                                        std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_particles, t_cells);
  } else {
    std::cerr << "All buffers for momentum rotation are not initialized."
              << std::endl;
    std::exit(1);
  }
}

void AssignParticleToCollisionCellKernel::setBuffers(
    ParticleCollection& t_particles, CollisionCellCollection& t_cells) {
  m_kernel.setArg(0, t_particles.getParticleXBuffer());
  m_kernel.setArg(1, t_particles.getParticleYBuffer());
  m_kernel.setArg(2, t_particles.getParticleZBuffer());
  m_kernel.setArg(3, t_particles.getParticleCollisionCellIndexBuffer());
  m_kernel.setArg(4, t_cells.getCellCountInOneAxisBuffer());
  m_kernel.setArg(5, t_cells.getCellLengthBuffer());
  m_kernel.setArg(6, t_cells.getShiftVectorBuffer());
}

void AssignParticleToCollisionCellKernel::setBuffers(
    ParticleCollection& t_particles, CollisionCellCollection& t_cells,
    std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_particles, t_cells);
  } else {
    std::cerr << "All buffers for assigning particles to collision cells are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void AssignParticleToCollisionCellTwoPhaseKernel::setBuffers(
    ParticleCollection& t_particles, CollisionCellCollection& t_cells) {
  m_kernel.setArg(0, t_particles.getParticleXBuffer());
  m_kernel.setArg(1, t_particles.getParticleYBuffer());
  m_kernel.setArg(2, t_particles.getParticleZBuffer());
  m_kernel.setArg(3, t_particles.getParticleCollisionCellIndexBuffer());
  m_kernel.setArg(4, t_cells.getCellCountInOneAxisBuffer());
  m_kernel.setArg(5, t_cells.getCellLengthBuffer());
  m_kernel.setArg(6, t_cells.getShiftVectorBuffer());
}

void AssignParticleToCollisionCellTwoPhaseKernel::setBuffers(
    ParticleCollection& t_particles, CollisionCellCollection& t_cells,
    std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_particles, t_cells);
  } else {
    std::cerr << "All buffers for assigning particles to collision cells two "
                 "phase are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void CollisionCellGenerationKernel::setBuffers(
    CollisionCellCollection& t_cells) {
  m_kernel.setArg(0, t_cells.getCellThetaAxisBuffer());
  m_kernel.setArg(1, t_cells.getCellPhiAxisBuffer());
  m_kernel.setArg(2, t_cells.getCellThetaRotationBuffer());
  m_kernel.setArg(3, t_cells.getCellEBuffer());
  m_kernel.setArg(4, t_cells.getCellpXBuffer());
  m_kernel.setArg(5, t_cells.getCellpYBuffer());
  m_kernel.setArg(6, t_cells.getCellpZBuffer());
  m_kernel.setArg(7, t_cells.getCellCollideBooleanBuffer());
  m_kernel.setArg(8, t_cells.getCellLogEBuffer());
  m_kernel.setArg(9, t_cells.getCellParticleCountBuffer());
  m_kernel.setArg(10, t_cells.getSeedBuffer());
  m_kernel.setArg(11, t_cells.getNoCollisionProbabilityBuffer());
}

void CollisionCellGenerationKernel::setBuffers(CollisionCellCollection& t_cells,
                                               std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_cells);
  } else {
    std::cerr << "All buffers for collision cell generation are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void CollisionCellResetKernel::setBuffers(CollisionCellCollection& t_cells) {
  m_kernel.setArg(0, t_cells.getCellThetaAxisBuffer());
  m_kernel.setArg(1, t_cells.getCellPhiAxisBuffer());
  m_kernel.setArg(2, t_cells.getCellThetaRotationBuffer());
  m_kernel.setArg(3, t_cells.getCellEBuffer());
  m_kernel.setArg(4, t_cells.getCellpXBuffer());
  m_kernel.setArg(5, t_cells.getCellpYBuffer());
  m_kernel.setArg(6, t_cells.getCellpZBuffer());
  m_kernel.setArg(7, t_cells.getCellCollideBooleanBuffer());
  m_kernel.setArg(8, t_cells.getCellLogEBuffer());
  m_kernel.setArg(9, t_cells.getCellParticleCountBuffer());
}

void CollisionCellResetKernel::setBuffers(CollisionCellCollection& t_cells,
                                          std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_cells);
  } else {
    std::cerr << "All buffers for collision cell reset are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void ParticleStepLinearKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles) {
  int errNum;
  errNum = m_kernel.setArg(0, t_particles.getParticleXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(1, t_particles.getParticleYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(2, t_particles.getParticleZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(3, t_particles.getParticleEBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's energy buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(4, t_particles.getParticlepXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(5, t_particles.getParticlepYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(6, t_particles.getParticlepZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(7, t_simulation_parameters.getDtBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize timestep buffer." << std::endl;
    std::terminate();
  }
}

void ParticleStepLinearKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_simulation_parameters, t_particles);
  } else {
    std::cerr << "All buffers for liner step are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void ParticleStepWithBubbleKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, PhaseBubble& t_bubble) {
  int errNum;
  std::cout << &m_kernel << std::endl;
  errNum = m_kernel.setArg(0, t_particles.getParticleXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(1, t_particles.getParticleYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(2, t_particles.getParticleZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(3, t_particles.getParticleEBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's energy buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(4, t_particles.getParticlepXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(5, t_particles.getParticlepYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(6, t_particles.getParticlepZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(7, t_particles.getParticleMBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's mass buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(8, t_particles.getdPBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's pressure (dP) buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(9, t_particles.getInteractedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(10, t_particles.getPassedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) and passing buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(11, t_particles.getInteractedBubbleTrueStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "true vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(12, t_bubble.getBubbleBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Bubble buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(13, t_simulation_parameters.getMassInBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass In buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(14, t_simulation_parameters.getMassOutBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass Out buffer." << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(15, t_simulation_parameters.getDeltaMassSquaredBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Delta Mass Squared buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(16, t_simulation_parameters.getDtBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize timestep buffer." << std::endl;
    std::terminate();
  }
}

void ParticleStepWithBubbleKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_simulation_parameters, t_particles, t_bubble);
  } else {
    
    std::cerr << "All buffers for step with bubble are "
                 "not initialized." 
              << std::endl;
    std::cout << std::bitset<64>(t_flags & m_kernel_flags) << std::endl <<
                           std::bitset<64>(m_kernel_flags) << std::endl;
    std::exit(1);
  }
}

void ParticleStepWithBubbleInvertedKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, PhaseBubble& t_bubble) {
  int errNum;
  errNum = m_kernel.setArg(0, t_particles.getParticleXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(1, t_particles.getParticleYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(2, t_particles.getParticleZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(3, t_particles.getParticleEBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's energy buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(4, t_particles.getParticlepXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(5, t_particles.getParticlepYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(6, t_particles.getParticlepZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(7, t_particles.getParticleMBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's mass buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(8, t_particles.getdPBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's pressure (dP) buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(9, t_particles.getInteractedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(10, t_particles.getPassedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) and passing buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(11, t_particles.getInteractedBubbleTrueStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "true vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(12, t_bubble.getBubbleBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Bubble buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(13, t_simulation_parameters.getMassInBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass In buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(14, t_simulation_parameters.getMassOutBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass Out buffer." << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(15, t_simulation_parameters.getDeltaMassSquaredBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Delta Mass Squared buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(16, t_simulation_parameters.getDtBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize timestep buffer." << std::endl;
    std::terminate();
  }
}

void ParticleStepWithBubbleInvertedKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_simulation_parameters, t_particles, t_bubble);
  } else {
    std::cerr << "All buffers for step with bubble inverted are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void ParticleStepWithBubbleReflectKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, PhaseBubble& t_bubble) {
  int errNum;
  errNum = m_kernel.setArg(0, t_particles.getParticleXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(1, t_particles.getParticleYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(2, t_particles.getParticleZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(3, t_particles.getParticleEBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's energy buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(4, t_particles.getParticlepXBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum X coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(5, t_particles.getParticlepYBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Y coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(6, t_particles.getParticlepZBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's momentum Z coordinate buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(7, t_particles.getdPBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's pressure (dP) buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(8, t_particles.getInteractedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(9, t_particles.getPassedBubbleFalseStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "false vacuum) and passing buffer."
              << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(10, t_particles.getInteractedBubbleTrueStateBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Particle's interaction (with bubble from "
                 "true vacuum) buffer."
              << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(11, t_bubble.getBubbleBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Bubble buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(12, t_simulation_parameters.getMassInBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass In buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(13, t_simulation_parameters.getMassOutBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Mass Out buffer." << std::endl;
    std::terminate();
  }
  errNum =
      m_kernel.setArg(14, t_simulation_parameters.getDeltaMassSquaredBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize Delta Mass Squared buffer." << std::endl;
    std::terminate();
  }
  errNum = m_kernel.setArg(15, t_simulation_parameters.getDtBuffer());
  if (errNum != CL_SUCCESS) {
    std::cerr << "Couldn't initialize timestep buffer." << std::endl;
    std::terminate();
  }
}

void ParticleStepWithBubbleReflectKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_simulation_parameters, t_particles, t_bubble);
  } else {
    std::cerr << "All buffers for step with bubble reflect are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void ParticleBoundaryCheckKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles) {
  m_kernel.setArg(0, t_particles.getParticleXBuffer());
  m_kernel.setArg(1, t_particles.getParticleYBuffer());
  m_kernel.setArg(2, t_particles.getParticleZBuffer());
  m_kernel.setArg(3, t_simulation_parameters.getBoundaryBuffer());
}

void ParticleBoundaryCheckKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_simulation_parameters, t_particles);
  } else {
    std::cerr << "All buffers for boundary check are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void ParticleBoundaryCheckMomentumKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles) {
  m_kernel.setArg(0, t_particles.getParticleXBuffer());
  m_kernel.setArg(1, t_particles.getParticleYBuffer());
  m_kernel.setArg(2, t_particles.getParticleZBuffer());
  m_kernel.setArg(3, t_particles.getParticlepXBuffer());
  m_kernel.setArg(4, t_particles.getParticlepYBuffer());
  m_kernel.setArg(5, t_particles.getParticlepZBuffer());
  m_kernel.setArg(6, t_simulation_parameters.getBoundaryBuffer());
}

void ParticleBoundaryCheckMomentumKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_simulation_parameters, t_particles);
  } else {
    std::cerr << "All buffers for momentum boundary check are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void ParticleLabelByCoordinateKernel::setBuffers(
    ParticleCollection& t_particles, PhaseBubble& t_bubble) {
  m_kernel.setArg(0, t_particles.getParticleXBuffer());
  m_kernel.setArg(1, t_particles.getParticleYBuffer());
  m_kernel.setArg(2, t_particles.getParticleZBuffer());
  m_kernel.setArg(3, t_particles.getParticleInBubbleBuffer());
  m_kernel.setArg(4, t_bubble.getBubbleBuffer());
}

void ParticleLabelByCoordinateKernel::setBuffers(
    ParticleCollection& t_particles, PhaseBubble& t_bubble,
    std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_particles, t_bubble);
  } else {
    std::cerr << "All buffers for particle coordinate label are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

void ParticleLabelByMassKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles) {
  m_kernel.setArg(0, t_particles.getParticleMBuffer());
  m_kernel.setArg(1, t_particles.getParticleInBubbleBuffer());
  m_kernel.setArg(2, t_simulation_parameters.getMassInBuffer());
}

void ParticleLabelByMassKernel::setBuffers(
    SimulationParameters& t_simulation_parameters,
    ParticleCollection& t_particles, std::uint32_t t_flags) {
  if (checkFlags(t_flags)) {
    setBuffers(t_simulation_parameters, t_particles);
  } else {
    std::cerr << "All buffers for particle mass label are "
                 "not initialized."
              << std::endl;
    std::exit(1);
  }
}

/*
void Simulation::setBuffersCollisionCellCalculateSummation(
    ParticleCollection& t_particles, CollisionCellCollection& t_cells,
    cl::Kernel& t_kernel) {
  t_kernel.setArg(0, t_particles.getParticleEBuffer());
  t_kernel.setArg(1, t_particles.getParticlepXBuffer());
  t_kernel.setArg(2, t_particles.getParticlepYBuffer());
  t_kernel.setArg(3, t_particles.getParticlepZBuffer());
  t_kernel.setArg(4, t_particles.getParticleCollisionCellIndexBuffer());
  std::cerr << "Simulation::setBuffersCollisionCellCalculateSummation method is
old version. Update the kernel and function."
            << std::endl;
  exit(0);
*/