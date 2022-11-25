#pragma once
#include <nlohmann/json.hpp>

#include "base.h"

class ConfigReader {
 public:
  int m_seed;
  int m_maxSteps;
  numType m_dt;
  bool m_dVisPositive;
  /*
   * Physics parameters
   */
  numType m_alpha;
  numType m_eta;
  numType m_upsilon;
  numType m_coupling;
  /*
   * Particle parameters
   */
  numType m_massFalse;
  numType m_massTrue;
  numType m_temperatureFalse;
  numType m_temperatureTrue;
  u_int m_countParticlesFalse;
  u_int m_countParticlesTrue;
  /*
   * Bubble parameters
   */
  numType m_initialBubbleRadius;
  numType m_initialBubbleSpeed;
  bool m_isBubbleTrueVacuum;
  bool m_interactionsOn;
  /*
   * Streamer parameters
   */
  bool m_toStream;
  bool m_streamData;
  bool m_streamProfiles;

  std::string m_dataSavePath;
  int m_streamFreq;
  int m_densityBinsCount;
  int m_momentumBinsCount;

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

    m_dt = config["simulation"]["dt"];
    m_dVisPositive = config["simulation"]["dV_positive"];

    /*
     * Physical parameters
     */
    m_alpha = config["parameters"]["alpha"];
    m_eta = config["parameters"]["eta"];
    m_upsilon = config["parameters"]["upsilon"];
    m_coupling = config["parameters"]["coupling"];

    /*
     * Particle parameters
     */
    m_massFalse = config["particles"]["mass_false"];
    m_massTrue = config["particles"]["mass_true"];
    m_temperatureFalse = config["particles"]["T_false"];
    m_temperatureTrue = config["particles"]["T_true"];
    m_countParticlesFalse = config["particles"]["N_false"];
    m_countParticlesTrue = config["particles"]["N_true"];

    /*
     * Bubble parameters
     */
    m_initialBubbleRadius = config["bubble"]["initial_radius"];
    m_initialBubbleSpeed = config["bubble"]["initial_speed"];
    m_isBubbleTrueVacuum = config["bubble"]["is_true_vacuum"];
    m_interactionsOn = config["bubble"]["interaction_on"];

    /*
     * Streaming parameters
     */
    m_toStream = config["stream"]["stream"];
    m_streamData = config["stream"]["stream_data"];
    m_streamProfiles = config["stream"]["stream_profile"];
    m_dataSavePath = config["stream"]["data_save_path"];
    m_streamFreq = config["stream"]["stream_freq"];
    m_densityBinsCount = config["stream"]["profile_density_bins_count"];
    m_momentumBinsCount = config["stream"]["profile_momentum_bins_count"];
  }

  void print_info() {
    std::string sublabel_prefix = "==== ";
    std::string sublabel_sufix = " ====";
    std::cout << "=============== Config ===============" << std::endl;
    std::cout << sublabel_prefix + "Simulation" + sublabel_sufix << std::endl;
    std::cout << "seed: " << m_seed << ", max_steps: " << m_maxSteps
              << ", dt: " << m_dt << ", dV_isPositive: " << m_dVisPositive
              << std::endl;
    std::cout << sublabel_prefix + "Parameters" << sublabel_sufix << std::endl;
    std::cout << std::setprecision(5) << std::fixed;
    std::cout << "alpha: " << m_alpha << ", eta: " << m_eta
              << ", upsilon: " << m_upsilon << ", coupling: " << m_coupling
              << std::endl;
    std::cout << std::setprecision(2);
    std::cout << sublabel_prefix + "Bubble" << sublabel_sufix << std::endl;
    std::cout << "R_b: " << m_initialBubbleRadius
              << ", V_b: " << m_initialBubbleSpeed
              << ", isTrueVacuum: " << m_isBubbleTrueVacuum
              << ", Interaction: " << m_interactionsOn << std::endl;
    std::cout << sublabel_prefix + "Particles" << sublabel_sufix << std::endl;
    std::cout << "Mass false: " << m_massFalse << ", Mass true: " << m_massTrue
              << std::endl;
    std::cout << "Temperature false: " << m_temperatureFalse
              << ", Temperature true: " << m_temperatureTrue << std::endl;
    std::cout << "Count false: " << m_countParticlesFalse
              << ", Count true: " << m_countParticlesTrue << std::endl;
  }

 private:
};