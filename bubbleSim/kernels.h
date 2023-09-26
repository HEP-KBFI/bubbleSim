#pragma once
#include <bitset>

#include "base.h"
#include "bubble.h"
#include "collision.h"
#include "particle.h"
#include "simulation_parameters.h"

class SimulationKernel {
 protected:
  std::string m_kernel_name;
  cl::Kernel m_kernel;
  std::uint64_t m_kernel_flags;
  bool checkFlags(std::uint64_t t_flags) {
    return m_kernel_flags == (m_kernel_flags & t_flags);
  }

 public:
  SimulationKernel() = default;
  SimulationKernel(cl::Program &t_program, std::string t_name = "") {
    int errNum;
    m_kernel_name = t_name;
    m_kernel = cl::Kernel(t_program, m_kernel_name.c_str(), &errNum);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Failed to create a kernel: " << m_kernel_name
                << ", Error number: " << errNum << std::endl;
      exit(1);
    }
  };
  std::string getKernelName() { return m_kernel_name; }
  cl::Kernel &getKernel() { return m_kernel; }
};

class MomentumRotationKernel : public SimulationKernel {
 private:
 public:
  MomentumRotationKernel() : SimulationKernel(){};
  MomentumRotationKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "rotate_momentum") {
    m_kernel_flags =
        PARTICLE_E_BUFFER | PARTICLE_PX_BUFFER | PARTICLE_PY_BUFFER |
        PARTICLE_PZ_BUFFER | PARTICLE_COLLISION_CELL_IDX_BUFFER |
        CELL_THETA_AXIS_BUFFER | CELL_PHI_AXIS_BUFFER |
        CELL_THETA_ROTATION_BUFFER | CELL_E_BUFFER | CELL_PX_BUFFER |
        CELL_PY_BUFFER | CELL_PZ_BUFFER | CELL_COLLIDE_BUFFER;
  };
  void setBuffers(ParticleCollection &t_particles,
                  CollisionCellCollection &t_cells);
  void setBuffers(ParticleCollection &t_particles,
                  CollisionCellCollection &t_cells, std::uint64_t t_flags);
};

class AssignParticleToCollisionCellKernel : public SimulationKernel {
 private:
 public:
  AssignParticleToCollisionCellKernel() : SimulationKernel(){};
  AssignParticleToCollisionCellKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "assign_particle_to_collision_cell") {
    m_kernel_flags = PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
                     PARTICLE_COLLISION_CELL_IDX_BUFFER |
                     CELL_COUNT_IN_ONE_AXIS_BUFFER | CELL_LENGTH_BUFFER |
                     CELL_SHIFT_VECTOR_BUFFER;
  };
  void setBuffers(ParticleCollection &t_particles,
                  CollisionCellCollection &t_cells);
  void setBuffers(ParticleCollection &t_particles,
                  CollisionCellCollection &t_cells, std::uint64_t t_flags);
};

class AssignParticleToCollisionCellTwoMassStateKernel : public SimulationKernel {
 private:
 public:
  AssignParticleToCollisionCellTwoMassStateKernel() : SimulationKernel(){};
  AssignParticleToCollisionCellTwoMassStateKernel(cl::Program &t_program)
      : SimulationKernel(t_program,
                         "assign_particle_to_collision_cell_two_state") {
    m_kernel_flags = PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
                     PARTICLE_COLLISION_CELL_IDX_BUFFER |
                     CELL_COUNT_IN_ONE_AXIS_BUFFER | CELL_LENGTH_BUFFER |
                     CELL_SHIFT_VECTOR_BUFFER;
  };
  void setBuffers(ParticleCollection &t_particles,
                  CollisionCellCollection &t_cells);
  void setBuffers(ParticleCollection &t_particles,
                  CollisionCellCollection &t_cells, std::uint64_t t_flags);
};

class CollisionCellGenerationKernel : public SimulationKernel {
 private:
 public:
  CollisionCellGenerationKernel() : SimulationKernel(){};
  CollisionCellGenerationKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "collision_cell_calculate_generation") {
    m_kernel_flags = CELL_THETA_AXIS_BUFFER | CELL_PHI_AXIS_BUFFER |
                     CELL_THETA_ROTATION_BUFFER | CELL_E_BUFFER |
                     CELL_PX_BUFFER | CELL_PY_BUFFER | CELL_PZ_BUFFER |
                     CELL_COLLIDE_BUFFER | CELL_LOGE_BUFFER |
                     CELL_PARTICLE_COUNT_BUFFER | CELL_SEED_INT64_BUFFER |
                     CELL_NO_COLLISION_PROBABILITY_BUFFER;
  };
  void setBuffers(CollisionCellCollection &t_cells);
  void setBuffers(CollisionCellCollection &t_cells, std::uint64_t t_flags);
};

class CollisionCellResetKernel : public SimulationKernel {
 private:
 public:
  CollisionCellResetKernel() : SimulationKernel(){};
  CollisionCellResetKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "collision_cell_reset") {
    m_kernel_flags = CELL_THETA_AXIS_BUFFER | CELL_PHI_AXIS_BUFFER |
                     CELL_THETA_ROTATION_BUFFER | CELL_E_BUFFER |
                     CELL_PX_BUFFER | CELL_PY_BUFFER | CELL_PZ_BUFFER |
                     CELL_COLLIDE_BUFFER | CELL_LOGE_BUFFER |
                     CELL_PARTICLE_COUNT_BUFFER;
  };
  void setBuffers(CollisionCellCollection &t_cells);
  void setBuffers(CollisionCellCollection &t_cells, std::uint64_t t_flags);
};

class ParticleStepLinearKernel : public SimulationKernel {
 private:
 public:
  ParticleStepLinearKernel() : SimulationKernel(){};
  ParticleStepLinearKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "particle_step_linear") {
    m_kernel_flags = PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
                     PARTICLE_E_BUFFER | PARTICLE_PX_BUFFER |
                     PARTICLE_PY_BUFFER | PARTICLE_PZ_BUFFER |
                     SIMULATION_DT_BUFFER;
  };
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles);
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, std::uint64_t t_flags);
};

class ParticleStepWithBubbleKernel : public SimulationKernel {
 public:
  ParticleStepWithBubbleKernel() : SimulationKernel(){};
  ParticleStepWithBubbleKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "particle_step_with_bubble") {
    m_kernel_flags =
        PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
        PARTICLE_E_BUFFER | PARTICLE_PX_BUFFER | PARTICLE_PY_BUFFER |
        PARTICLE_PZ_BUFFER | PARTICLE_M_BUFFER | PARTICLE_dP_BUFFER |
        PARTICLE_INTERACTED_FALSE_BUFFER | PARTICLE_INTERACTED_TRUE_BUFFER |
        PARTICLE_PASSED_FALSE_BUFFER | BUBBLE_BUFFER |
        SIMULATION_MASS_IN_BUFFER | SIMULATION_MASS_OUT_BUFFER |
        SIMULATION_DELTA_MASS_BUFFER | SIMULATION_DT_BUFFER;
  };
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, PhaseBubble &t_bubble);
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, PhaseBubble &t_bubble,
                  std::uint64_t t_flags);
};

class ParticleStepWithBubbleInvertedKernel : public SimulationKernel {
 private:
 public:
  ParticleStepWithBubbleInvertedKernel() : SimulationKernel(){};
  ParticleStepWithBubbleInvertedKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "particle_step_with_bubble_inverted") {
    m_kernel_flags =
        PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
        PARTICLE_E_BUFFER | PARTICLE_PX_BUFFER | PARTICLE_PY_BUFFER |
        PARTICLE_PZ_BUFFER | PARTICLE_M_BUFFER | PARTICLE_dP_BUFFER |
        PARTICLE_INTERACTED_FALSE_BUFFER | PARTICLE_INTERACTED_TRUE_BUFFER |
        PARTICLE_PASSED_FALSE_BUFFER | BUBBLE_BUFFER |
        SIMULATION_MASS_IN_BUFFER | SIMULATION_MASS_OUT_BUFFER |
        SIMULATION_DELTA_MASS_BUFFER | SIMULATION_DT_BUFFER;
  };
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, PhaseBubble &t_bubble);
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, PhaseBubble &t_bubble,
                  std::uint64_t t_flags);
};

class ParticleStepWithBubbleReflectKernel : public SimulationKernel {
 private:
 public:
  ParticleStepWithBubbleReflectKernel() : SimulationKernel(){};
  ParticleStepWithBubbleReflectKernel(cl::Program &t_program)
      : SimulationKernel(t_program,
                         "particles_step_with_false_bubble_reflect") {
    m_kernel_flags = PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
                     PARTICLE_E_BUFFER | PARTICLE_PX_BUFFER |
                     PARTICLE_PY_BUFFER | PARTICLE_PZ_BUFFER |
                     PARTICLE_dP_BUFFER | PARTICLE_INTERACTED_FALSE_BUFFER |
                     PARTICLE_INTERACTED_TRUE_BUFFER |
                     PARTICLE_PASSED_FALSE_BUFFER | BUBBLE_BUFFER |
                     SIMULATION_MASS_IN_BUFFER | SIMULATION_MASS_OUT_BUFFER |
                     SIMULATION_DELTA_MASS_BUFFER | SIMULATION_DT_BUFFER;
  };
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, PhaseBubble &t_bubble);
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, PhaseBubble &t_bubble,
                  std::uint64_t t_flags);
};

class ParticleBoundaryCheckKernel : public SimulationKernel {
 private:
 public:
  ParticleBoundaryCheckKernel() : SimulationKernel(){};
  ParticleBoundaryCheckKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "particle_boundary_check") {
    m_kernel_flags = PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
                     SIMULATION_BOUNDARY_BUFFER;
  };
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles);
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, std::uint64_t t_flags);
};

class ParticleBoundaryCheckMomentumKernel : public SimulationKernel {
 private:
 public:
  ParticleBoundaryCheckMomentumKernel() : SimulationKernel(){};
  ParticleBoundaryCheckMomentumKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "particle_boundary_momentum_reflect") {
    m_kernel_flags = PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
                     PARTICLE_PX_BUFFER | PARTICLE_PY_BUFFER |
                     PARTICLE_PZ_BUFFER | SIMULATION_BOUNDARY_BUFFER;
  };
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles);
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, std::uint64_t t_flags);
};

class ParticleLabelByCoordinateKernel : public SimulationKernel {
 private:
 public:
  ParticleLabelByCoordinateKernel() : SimulationKernel(){};
  ParticleLabelByCoordinateKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "label_particles_position_by_coordinate") {
    m_kernel_flags = PARTICLE_X_BUFFER | PARTICLE_Y_BUFFER | PARTICLE_Z_BUFFER |
                     PARTICLE_IN_BUBBLE_BUFFER | BUBBLE_BUFFER;
  };
  void setBuffers(ParticleCollection &t_particles, PhaseBubble &t_bubble);
  void setBuffers(ParticleCollection &t_particles, PhaseBubble &t_bubble,
                  std::uint64_t t_flags);
};

class ParticleLabelByMassKernel : public SimulationKernel {
 private:
 public:
  ParticleLabelByMassKernel() : SimulationKernel(){};
  ParticleLabelByMassKernel(cl::Program &t_program)
      : SimulationKernel(t_program, "label_particles_position_by_mass") {
    m_kernel_flags = PARTICLE_M_BUFFER | PARTICLE_IN_BUBBLE_BUFFER |
                     SIMULATION_MASS_IN_BUFFER;
  };
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles);
  void setBuffers(SimulationParameters &t_simulation_parameters,
                  ParticleCollection &t_particles, std::uint64_t t_flags);
};
