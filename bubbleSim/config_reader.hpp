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
    std::cout << "Config path: " << configPath << std::endl;
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

 private:
};