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
  int m_seed;
  u_int m_max_steps;
  numType dt;
  u_int timestep_resolution;
  bool cyclicBoundaryOn;

  /*
   * Physics parameters
   */
  numType alpha;
  numType eta;
  numType upsilon;
  numType sigma;
  numType tau;
  numType lambda;
  numType v;
  numType y;
  numType etaV;
  numType Tn;
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
  bool collision_two_mass_state_on;
  unsigned int collision_cell_count;
  unsigned int collision_cell_duplication;

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
  std::string m_dataSavePath;

  int stream_step;

  bool streaming_on;
  bool stream_timeseries;
  bool stream_profile;
  bool stream_momentum;
  bool stream_momentum_profile;

  u_int profile_bins_count_in;
  u_int profile_bins_count_out;

  u_int momentum_bins_count_in;
  u_int momentum_bins_count_out;

  u_int momentum_profile_momentum_bins_count;
  u_int momentum_profile_radius_bins_count;

  ConfigReader(std::string configPath) {
    std::ifstream configStream(configPath);
    nlohmann::json config = nlohmann::json::parse(configStream);

    m_seed = config["simulation"]["seed"];
    m_max_steps = config["simulation"]["max_steps"];
    if (m_max_steps == 0) {
      m_max_steps = std::numeric_limits<int>::max();
    } else if (m_max_steps < 0) {
      std::cerr << "maxSteps is set to negative value. maxSteps >= 0."
                << std::endl;
      std::terminate();
    }

    dt = config["simulation"]["dt"];
    timestep_resolution = config["simulation"]["timestep_resolution"];

    if (config["simulation"]["cyclic_boundary_on"]) {
      SIMULATION_SETTINGS.setFlag(SIMULATION_BOUNDARY_ON);
    }
    cyclicBoundaryOn = config["simulation"]["cyclic_boundary_on"];
    /*
     * Physical parameters
     */
    alpha = config["parameters"]["alpha"];
    eta = config["parameters"]["eta"];
    upsilon = config["parameters"]["upsilon"];
    tau = config["parameters"]["tau"];
    lambda = config["parameters"]["lambda"];
    v = config["parameters"]["v"];
    y = config["parameters"]["y"];
    etaV = config["parameters"]["etaV"];
    sigma = config["parameters"]["sigma"];
    Tn = config["parameters"]["Tn"];
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
    if (config["collision"]["collision_on"]) {
      SIMULATION_SETTINGS.setFlag(COLLISION_ON);
    }
    collision_on = config["collision"]["collision_on"];
    if (config["collision"]["two_mass_state_on"]) {
      SIMULATION_SETTINGS.setFlag(COLLISION_MASS_STATE_ON);
    }
    collision_two_mass_state_on = config["collision"]["two_mass_state_on"];
    collision_cell_count = config["collision"]["N_cells"];
    collision_cell_duplication = config["collision"]["cell_duplication"];

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

    streaming_on = config["stream"]["bool_stream"];
    if (config["stream"]["bool_stream"]) {
      STREAM_SETTINGS.setFlag(STREAM_ON);
    }
    stream_timeseries = config["stream"]["bool_stream_timeseries"];
    if (config["stream"]["bool_stream_timeseries"]) {
      STREAM_SETTINGS.setFlag(STREAM_TIMESERIES);
    }

    stream_profile = config["stream"]["bool_stream_profile"];
    profile_bins_count_in = config["stream"]["profile_bins_count_in"];
    profile_bins_count_out = config["stream"]["profile_bins_count_out"];
    if (config["stream"]["bool_stream_profile"]) {
      STREAM_SETTINGS.setFlag(STREAM_PROFILE);
    }

    stream_momentum = config["stream"]["bool_stream_momentum"];
    momentum_bins_count_in = config["stream"]["momentum_bins_count_in"];
    momentum_bins_count_out = config["stream"]["momentum_bins_count_out"];
    if (config["stream"]["bool_stream_momentum"]) {
      STREAM_SETTINGS.setFlag(STREAM_MOMENTUM);
    }

    stream_momentum_profile = config["stream"]["bool_stream_momentum_profile"];
    momentum_profile_momentum_bins_count =
        config["stream"]["momentum_profile_momentum_bins_count"];
    momentum_profile_radius_bins_count =
        config["stream"]["momentum_profile_radius_bins_count"];
    if (config["stream"]["bool_stream_momentum_profile"]) {
      STREAM_SETTINGS.setFlag(STREAM_MOMENTUM_PROFILE);
    }

    m_dataSavePath = config["stream"]["data_save_path"];
    stream_step =
        config["stream"]
              ["stream_step"];  // Step after which simulation state is saved
  }
  void print_info() {
    std::string sublabel_prefix = "==== ";
    std::string sublabel_sufix = " ====";
    std::cout << std::setprecision(6);
    std::cout << "=============== Config ===============" << std::endl;
    std::cout << sublabel_prefix + "Simulation" + sublabel_sufix << std::endl;
    std::cout << "seed: " << m_seed << ", max_steps: " << m_max_steps
              << ", dt: " << dt << ", Stream step: " << stream_step << std::endl;
    std::cout << "Cyclic boundary on: " << cyclicBoundaryOn
              << std::endl;
    std::cout << sublabel_prefix + "Parameters" << sublabel_sufix << std::endl;
    std::cout << "alpha: " << alpha << ", eta: " << eta
              << ", upsilon: " << upsilon << std::endl;

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