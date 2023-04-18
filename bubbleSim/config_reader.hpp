#pragma once
#include <nlohmann/json.hpp>

#include "base.h"

class ConfigReader {
 public:
  int m_seed;
  int m_maxSteps;
  numType dt;
  numType maxTime;
  bool cyclicBoundaryOn;
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
  bool streamMomentumOn;

  std::string m_dataSavePath;
  int streamFreq;
  int streamDensityBinsCount;
  int streamMomentumBinsCount;

  ConfigReader(std::string configPath) {
    std::ifstream configStream(configPath);
    nlohmann::json config = nlohmann::json::parse(configStream);

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
    maxTime = config["simulation"]["maxTime"];
    cyclicBoundaryOn = config["simulation"]["cyclic_boundary_on"];
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
    streamMomentumOn = config["stream"]["stream_momentum_profile"];
    m_dataSavePath = config["stream"]["data_save_path"];
    streamFreq = config["stream"]["stream_freq"];
    streamDensityBinsCount = config["stream"]["profile_density_bins_count"];
    streamMomentumBinsCount = config["stream"]["profile_momentum_bins_count"];
  }
  void print_info() {
    std::string sublabel_prefix = "==== ";
    std::string sublabel_sufix = " ====";
    std::cout << "=============== Config ===============" << std::endl;
    std::cout << sublabel_prefix + "Simulation" + sublabel_sufix << std::endl;
    std::cout << "seed: " << m_seed << ", max_steps: " << m_maxSteps
              << ", dt: " << dt << ", Cyclic boundary on: " << cyclicBoundaryOn
              << std::endl;
    std::cout << sublabel_prefix + "Parameters" << sublabel_sufix << std::endl;
    std::cout << std::setprecision(5) << std::fixed;
    std::cout << "alpha: " << parameterAlpha << ", eta: " << parameterEta
              << ", upsilon: " << parameterUpsilon
              << ", coupling: " << parameterCoupling << std::endl;
    std::cout << std::setprecision(2);
    std::cout << sublabel_prefix + "Bubble" << sublabel_sufix << std::endl;
    std::cout << "R_b: " << bubbleInitialRadius
              << ", V_b: " << bubbleInitialSpeed
              << ", isTrueVacuum: " << bubbleIsTrueVacuum
              << ", Interaction: " << bubbleInteractionsOn << std::endl;
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