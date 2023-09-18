#pragma once
#include <nlohmann/json.hpp>

#include "base.h"

class SimulationSettings {
 public:
  bool isFlagSet(SimulationFlags flag) const { return flag == (flags & flag); }
  void setFlag(SimulationFlags flag) { flags |= flag; }
  void removeFlag(SimulationFlags flag) { flags &= ~flag; }
  std::uint32_t getFlag() { return flags; }
 private:
  std::uint32_t flags = 0b00000000000000000000000000000000;
};

class StreamSettings {
 public:
  bool isFlagSet(StreamFlags flag) const { return flag == (flags & flag); }
  void setFlag(StreamFlags flag) { flags |= flag; }
  void removeFlag(StreamFlags flag) { flags &= ~flag; }
  std::uint32_t getFlag() { return flags; }
 private:
  std::uint32_t flags = 0b00000000000000000000000000000000;
};

class ConfigReader {
 public:
  SimulationSettings SIMULATION_SETTINGS = SimulationSettings();
  StreamSettings STREAM_SETTINGS = StreamSettings();
  std::string kernelName;
  int m_seed;
  u_int m_maxSteps;
  numType dt;
  numType maxTime;
  bool cyclicBoundaryOn;

  numType cyclicBoundaryRadius;
  /*
   * Physics parameters
   */
  numType parameterAlpha;
  numType parameterEta;
  numType parameterUpsilon;
  numType parameterTau;
  numType parameter_dV;
  /*
   * Particle parameters
   */
  numType particleMassFalse;
  numType particleMassTrue;
  numType particleTemperatureFalse;
  numType particleTemperatureTrue;
  u_int particleCountFalse;
  u_int particleCountTrue;
  /*
   * Collision parameters
   */
  bool collision_on;
  unsigned int collision_cell_count;
  numType collision_cell_length;
  /*
   * Bubble parameters
   */
  bool bubbleOn;
  numType bubbleInitialRadius;
  numType bubbleInitialSpeed;
  bool bubbleIsTrueVacuum;
  bool bubbleInteractionsOn;
  /*
   * Streamer parameters
   */
  bool streamOn;
  bool streamDataOn;
  bool streamDensityOn;
  bool streamEnergyOn;
  bool streamMomentumInOn;
  bool streamMomentumOutOn;
  bool streamRadialVelocityOn;
  bool streamTangentialVelocityOn;

  std::string m_dataSavePath;
  numType streamTime;  // After what simulation time info is saved
  int streamStep;
  int binsCountDensity;
  int binsCountEnergy;
  int binsCountRadialVelocity;
  int binsCountTangentialVelocity;
  int binsCountMomentumIn;
  int binsCountMomentumOut;
  numType minValueMomentumIn;
  numType minValueMomentumOut;
  numType maxValueDensity;
  numType maxValueEnergy;
  numType maxValueRadialVelocity;
  numType maxValueTangentialVelocity;
  numType maxValueMomentumIn;
  numType maxValueMomentumOut;

  ConfigReader(std::string configPath) {
    std::ifstream configStream(configPath);
    nlohmann::json config = nlohmann::json::parse(configStream);

    kernelName = config["kernel"]["name"];
    m_seed = config["simulation"]["seed"];
    m_maxSteps = config["simulation"]["max_steps"];
    if (m_maxSteps == 0) {
      m_maxSteps = std::numeric_limits<int>::max();
    } else if (m_maxSteps < 0) {
      std::cerr << "maxSteps is set to negative value. maxSteps >= 0."
                << std::endl;
      std::terminate();
    }

    dt = config["simulation"]["dt"];
    maxTime = config["simulation"]["max_time"];

    if (config["simulation"]["cyclic_boundary_on"]) {
      SIMULATION_SETTINGS.setFlag(SIMULATION_BOUNDARY_ON);
    }
    cyclicBoundaryOn = config["simulation"]["cyclic_boundary_on"];
    cyclicBoundaryRadius = config["simulation"]["cyclic_boundary_radius"];
    /*
     * Physical parameters
     */
    parameterAlpha = config["parameters"]["alpha"];
    parameterEta = config["parameters"]["eta"];
    parameterUpsilon = config["parameters"]["upsilon"];
    parameterTau = config["parameters"]["tau"];
    parameter_dV = config["parameters"]["dV"];
    /*
     * Particle parameters
     */
    particleMassFalse = config["particles"]["mass_false"];
    particleMassTrue = config["particles"]["mass_true"];
    particleTemperatureFalse = config["particles"]["T_false"];
    particleTemperatureTrue = config["particles"]["T_true"];
    particleCountFalse = config["particles"]["N_false"];
    particleCountTrue = config["particles"]["N_true"];
    /*
     * Collisions
     */
    if (config["collision"]["collision_on"])
      SIMULATION_SETTINGS.setFlag(COLLISION_ON);
    collision_on = config["collision"]["collision_on"];
    collision_cell_count = config["collision"]["N_cells"];
    collision_cell_length = config["collision"]["cell_length"];
    /*
     * Bubble parameters
     */

    if (config["bubble"]["bubble_on"]) SIMULATION_SETTINGS.setFlag(BUBBLE_ON);
    bubbleOn = config["bubble"]["bubble_on"];
    bubbleInitialRadius = config["bubble"]["initial_radius"];
    bubbleInitialSpeed = config["bubble"]["initial_speed"];
    if (config["bubble"]["is_true_vacuum"]) {
      SIMULATION_SETTINGS.setFlag(TRUE_VACUUM_BUBBLE_ON);
    }

    bubbleIsTrueVacuum = config["bubble"]["is_true_vacuum"];
    if (config["bubble"]["interaction_on"]) {
      SIMULATION_SETTINGS.setFlag(BUBBLE_INTERACTION_ON);
    }
    bubbleInteractionsOn = config["bubble"]["interaction_on"];
    /*
     * Streaming parameters
     */


    if (config["stream"]["stream"]) {
      STREAM_SETTINGS.setFlag(STREAM_ON);
    }
    if (config["stream"]["stream_data"]) {
      STREAM_SETTINGS.setFlag(STREAM_DATA);
    }
    if (config["stream"]["stream_density_profile"]) {
      STREAM_SETTINGS.setFlag(STREAM_NUMBER_DENSITY);
    }
    if (config["stream"]["steam_energy_profile"]) {
      STREAM_SETTINGS.setFlag(STREAM_ENERGY_DENSITY);
    }
    if (config["stream"]["stream_momentum"]) {
      STREAM_SETTINGS.setFlag(STREAM_MOMENTUM);
    }
    if (config["stream"]["stream_momentumIn_profile"]) {
      STREAM_SETTINGS.setFlag(STREAM_MOMENTUM_IN);
    }
    if (config["stream"]["stream_momentumOut_profile"]) {
      STREAM_SETTINGS.setFlag(STREAM_MOMENTUM_OUT);
    }
    if (config["stream"]["stream_radial_velocity"]) {
      STREAM_SETTINGS.setFlag(STREAM_RADIAL_VELOCITY);
    }
    if (config["stream"]["stream_tangential_velocity"]) {
      STREAM_SETTINGS.setFlag(STREAM_TANGENTIAL_VELOCITY);
    }

    streamOn = config["stream"]["stream"];
    streamDataOn = config["stream"]["stream_data"];
    streamDensityOn = config["stream"]["stream_density_profile"];
    streamEnergyOn = config["stream"]["steam_energy_profile"];
    streamMomentumInOn = config["stream"]["stream_momentumIn_profile"];
    streamMomentumOutOn = config["stream"]["stream_momentumOut_profile"];
    streamRadialVelocityOn = config["stream"]["stream_radial_velocity"];
    streamTangentialVelocityOn = config["stream"]["stream_tangential_velocity"];
    m_dataSavePath = config["stream"]["data_save_path"];
    streamTime = config["stream"]
                       ["stream_time"];  // time after which simulation is saved
    streamStep =
        config["stream"]
              ["stream_step"];  // Step after which simulation state is saved
    // Bins count
    binsCountDensity = config["stream"]["bins_count_density"];
    binsCountEnergy = config["stream"]["bins_count_energy"];
    binsCountRadialVelocity = config["stream"]["bins_count_radial_velocity"];
    binsCountTangentialVelocity =
        config["stream"]["bins_count_tangential_velocity"];
    binsCountMomentumIn = config["stream"]["bins_count_momentumIn"];
    binsCountMomentumOut = config["stream"]["bins_count_momentumOut"];
    // Minimum unit value for profile (radius, momentum, etc.)
    minValueMomentumIn = config["stream"]["min_value_momentumIn"];
    minValueMomentumOut = config["stream"]["min_value_momentumOut"];
    // Maximum unit value for profile (radius, momentum, etc.)
    maxValueDensity = config["stream"]["max_value_density"];
    maxValueEnergy = config["stream"]["max_value_energy"];
    maxValueRadialVelocity = config["stream"]["max_value_radial_velocity"];
    maxValueTangentialVelocity =
        config["stream"]["max_value_tangential_velocity"];
    maxValueMomentumIn = config["stream"]["max_value_momentumIn"];
    maxValueMomentumOut = config["stream"]["max_value_momentumOut"];
  }
  void print_info() {
    std::string sublabel_prefix = "==== ";
    std::string sublabel_sufix = " ====";
    std::cout << std::setprecision(6);
    std::cout << "=============== Config ===============" << std::endl;
    std::cout << sublabel_prefix + "Simulation" + sublabel_sufix << std::endl;
    std::cout << "seed: " << m_seed << ", max_steps: " << m_maxSteps
              << ", dt: " << dt << ", Stream time: " << streamTime
              << ", Stream step: " << streamStep << std::endl;
    std::cout << "Cyclic boundary on: " << cyclicBoundaryOn
              << ", Cyclic boundary radius: " << cyclicBoundaryRadius
              << std::endl;
    std::cout << sublabel_prefix + "Parameters" << sublabel_sufix << std::endl;
    std::cout << "alpha: " << parameterAlpha << ", eta: " << parameterEta
              << ", upsilon: " << parameterUpsilon << std::endl;

    std::cout << sublabel_prefix + "Bubble" << sublabel_sufix << std::endl;
    std::cout << "R_b: " << bubbleInitialRadius
              << ", V_b: " << bubbleInitialSpeed
              << ", isTrueVacuum: " << bubbleIsTrueVacuum
              << ", Interaction wth bubble: " << bubbleInteractionsOn
              << std::endl;
    std::cout << sublabel_prefix + "Particles" << sublabel_sufix << std::endl;
    std::cout << "Mass false: " << particleMassFalse
              << ", Mass true: " << particleMassTrue << std::endl;
    std::cout << "Temperature false: " << particleTemperatureFalse
              << ", Temperature true: " << particleTemperatureTrue << std::endl;
    std::cout << "Count false: " << particleCountFalse
              << ", Count true: " << particleCountTrue << std::endl;
  }

 private:
};