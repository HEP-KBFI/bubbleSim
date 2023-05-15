#pragma once
#include <nlohmann/json.hpp>

#include "base.h"

class ConfigReader {
 public:
  std::string kernelName;
  int m_seed;
  int m_maxSteps;
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
  bool collisionCellOn;
  numType parameterCoupling;
  unsigned int collisionCellCount;
  numType collisionCellLength;
  /*
   * Bubble parameters
   */
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
    cyclicBoundaryOn = config["simulation"]["cyclic_boundary_on"];
    cyclicBoundaryRadius = config["simulation"]["cyclic_boundary_radius"];
    /*
     * Physical parameters
     */
    parameterAlpha = config["parameters"]["alpha"];
    parameterEta = config["parameters"]["eta"];
    parameterUpsilon = config["parameters"]["upsilon"];
    parameterCoupling = config["parameters"]["coupling"];
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
    collisionCellOn = config["collision"]["collision_on"];
    collisionCellCount = config["collision"]["N_cells"];
    collisionCellLength = config["collision"]["cell_length"];
    /*
     * Bubble parameters
     */
    bubbleInitialRadius = config["bubble"]["initial_radius"];
    bubbleInitialSpeed = config["bubble"]["initial_speed"];
    bubbleIsTrueVacuum = config["bubble"]["is_true_vacuum"];
    bubbleInteractionsOn = config["bubble"]["interaction_on"];
    /*
     * Streaming parameters
     */
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
              << ", upsilon: " << parameterUpsilon
              << ", coupling: " << parameterCoupling << std::endl;

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