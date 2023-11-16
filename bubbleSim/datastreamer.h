#pragma once
#define _USE_MATH_DEFINES

#include <chrono>
#include <filesystem>
#include <fstream>

#include "base.h"
#include "bubble.h"
#include "opencl_kernels.h"
#include "simulation.h"

class DataStreamerBinary {
  std::filesystem::path m_file_path;
  std::ofstream m_stream_data;
  std::ofstream m_stream_profile;
  std::ofstream m_stream_momentum;
  std::ofstream m_stream_momentum_radial_profile;

  bool b_stream_momentum_in = false;
  bool b_stream_momentum_out = false;
  bool b_stream_momentum_radial_profile = false;

  numType m_stream_data_time;
  numType m_stream_dt;
  numType m_stream_data_dP;
  numType m_stream_data_radius;
  numType m_stream_data_velocity;
  numType m_stream_data_bubble_energy;
  numType m_stream_data_particle_energy;
  numType m_stream_data_particle_in_energy;
  numType m_stream_data_energy_conservation;
  uint32_t m_stream_data_particle_count_in;
  uint32_t m_stream_data_particle_interacted_false_count;
  uint32_t m_stream_data_particle_passed_false_count;
  uint32_t m_stream_data_particle_interacted_true_count;
  uint32_t m_stream_data_active_particles_in_collision;
  uint32_t m_stream_data_active_cells_in_collision;

  numType m_dr_in;
  numType m_dr_out;
  uint32_t m_N_bins_in_profile;
  uint32_t m_N_bins_out_profile;
  std::vector<uint32_t> m_particle_count;
  std::vector<numType> m_T00;
  std::vector<numType> m_T01;
  std::vector<numType> m_T02;
  std::vector<numType> m_T03;
  std::vector<numType> m_T11;
  std::vector<numType> m_T22;
  std::vector<numType> m_T33;
  std::vector<numType> m_T12;
  std::vector<numType> m_T13;
  std::vector<numType> m_T23;
  std::vector<numType> m_radial_velocity;
  std::vector<numType> m_momentum_value;
  std::vector<numType> m_momentum_change;
  std::vector<numType> m_mean_velocity;
  std::vector<numType> m_square_mean_velocity;

  numType m_dp_in;
  numType m_dp_out;
  numType m_p_min_factor = 1e-3;
  numType m_p_max_factor = 1e+3;
  numType m_p_min;
  numType m_p_max;

  uint32_t m_N_bins_momentum_in;
  uint32_t m_N_bins_momentum_out;
  std::vector<uint32_t> m_momentum;

  // Momenutm-radial profile
  numType m_dp_pr;
  numType m_p_min_factor_pr = 1e-3;
  numType m_p_max_factor_pr = 1e+3;
  numType m_p_min_pr;
  numType m_p_max_pr;
  uint32_t m_N_bins_momentum_pr;
  numType m_dr_in_pr;
  numType m_dr_out_pr;
  uint32_t m_N_bins_in_profile_pr;
  uint32_t m_N_bins_out_profile_pr;
  std::vector<uint32_t> m_momentum_radius_profile;

  numType calculateParticleRadialMomentum(ParticleCollection& particles,
                                          numType& radius, size_t i);

  numType calculateParticlePolarMomentum(ParticleCollection& particles,
                                         numType& theta, numType& phi,
                                         size_t i);

  numType calculateParticleAzimuthalMomentum(ParticleCollection& particles,
                                             numType& phi, size_t i);

 public:
  DataStreamerBinary(){};
  DataStreamerBinary(std::string t_file_path);
  void initStream_Data();
  void initialize_profile_streaming(uint32_t t_N_bins_in,
                                    uint32_t t_N_bins_out);
  void initialize_momentum_streaming(uint32_t t_N_bins_in,
                                     uint32_t t_N_bins_out,
                                     numType energy_scale);

  void initialize_momentum_radial_profile_streaming(
      uint32_t t_N_momentum_bins, uint32_t t_N_radius_bins_in,
      uint32_t t_N_radius_bins_out, numType energy_scale);

  void stream(Simulation& simulation, ParticleCollection& particleCollection,
              PhaseBubble& bubble, SimulationSettings& settings,
              cl::CommandQueue& cl_queue);

  void write_data();
  void write_profile();
  void write_momentum();
  void write_momentum_profile();
};
