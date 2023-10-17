#pragma once
#define _USE_MATH_DEFINES

#include <chrono>
#include <filesystem>
#include <fstream>

#include "base.h"
#include "bubble.h"
#include "opencl_kernels.h"
#include "simulation.h"

class DataStreamer {
  /*
   * Currently it is possible to save following variables:
   *     > Data (R, V, E, E_f, E_p ...)
   *        -
   *     > Momentum profile inside and outside the bubble
   *        - Define: 1) Max momentum 2) number of bins (~ delta P)
   *     > Number density
   *        - Define: 1) Max radius 2) number of bins (~ delta R)
   *     > Energy density
   *        - Define: 1) Max radius 2) number of bins (~ delta R)
   *     > Average radial velocity (radial profile)
   *        - Define: 1) Max radius 2) number of bins (~ delta R)
   *     > Average tangential velocity (radial profile)
   *        - Define: 1) Max radius 2) number of bins (~ delta R)
   *     > Momentum profile (radial profile) !!!
   */
 public:
  DataStreamer(std::string filePath);

  void initStream_Data();
  void initStream_MomentumIn(size_t t_binsCount, numType t_minMomentumValue,
                             numType t_maxMomentumValue, bool t_log_scale_on);
  void initStream_MomentumOut(size_t t_binsCount, numType t_minMomentumValue,
                              numType t_maxMomentumValue, bool t_log_scale_on);
  void initStream_Density(size_t t_binsCount, numType t_maxRadiusValue);
  void initStream_EnergyDensity(size_t t_binsCount, numType t_maxRadiusValue);
  void initStream_RadialVelocity(size_t t_binsCount, numType t_maxRadiusValue);
  void initStream_TangentialVelocity(size_t t_binsCount,
                                     numType t_maxRadiusValue);
  void initStream_radialMomentum(size_t t_binsCount, numType t_maxRadiusValue);
  void initStream_Momentum(size_t t_binsCount, numType t_maxMomentumValue);

  void stream(Simulation& simulation, ParticleCollection& particleCollection,
              PhaseBubble& bubble, bool t_log_scale_on,
              cl::CommandQueue& cl_queue);

  // These functions are meant to be used if only certain step is wanted to be
  // saved. Because otherwise this would mean different for cycles which makes
  // code slower.

  void streamMomentumIn(std::ofstream& t_stream, size_t t_binsCount,
                        numType t_minMomentumValue, numType t_maxMomentumValue,
                        ParticleCollection& particleCollection,
                        PhaseBubble& bubble, cl::CommandQueue& cl_queue);

  void streamMomentumOut(std::ofstream& t_stream, size_t t_binsCount,
                         numType t_minMomentumValue, numType t_maxMomentumValue,
                         ParticleCollection& particleCollection,
                         PhaseBubble& bubble, cl::CommandQueue& cl_queue);

  void streamNumberDensity(std::ofstream& t_stream, size_t t_binsCount,
                           numType t_minRadiusValue, numType t_maxRadiusValue,
                           ParticleCollection& particleCollection,
                           cl::CommandQueue& cl_queue);
  void streamEnergyDensity(std::ofstream& t_stream, size_t t_binsCount,
                           numType t_minRadiusValue, numType t_maxRadiusValue,
                           ParticleCollection& particleCollection,
                           cl::CommandQueue& cl_queue);
  void streamRadialVelocity(std::ofstream& t_stream, size_t t_binsCount,
                            numType t_minRadiusValue, numType t_maxRadiusValue,
                            ParticleCollection& particleCollection,
                            cl::CommandQueue& cl_queue);
  void streamTangentialVelocity(std::ofstream& t_stream, size_t t_binsCount,
                                numType t_minRadiusValue,
                                numType t_maxRadiusValue,
                                ParticleCollection& particleCollection,
                                cl::CommandQueue& cl_queue);
  void streamRadialMomentumProfile(
      std::ofstream& t_stream, size_t t_binsCountRadius,
      size_t t_binsCountMomentum, numType t_minRadiusValue,
      numType t_maxRadiusValue, numType t_minMomentumValue,
      numType t_maxMomentumValue, ParticleCollection& particleCollection,
      cl::CommandQueue& cl_queue);

 private:
  std::filesystem::path m_filePath;

  std::ofstream m_stream_data;
  std::ofstream m_stream_MomentumX;
  std::ofstream m_stream_MomentumY;
  std::ofstream m_stream_MomentumZ;
  std::ofstream m_stream_MomentumIn;
  std::ofstream m_stream_MomentumOut;

  std::ofstream m_stream_Density;
  std::ofstream m_stream_EnergyDensity;

  std::ofstream m_stream_RadialMomentum;
  std::ofstream m_stream_RadialVelocity;
  std::ofstream m_stream_TangentialVelocity;

  // General data about simulation state
  bool m_initialized_Data = false;
  // Momentum profiles in and outside the bubble
  bool m_initialized_Momentum = false;
  bool m_initialized_MomentumIn = false;
  bool m_initialized_MomentumOut = false;
  // Energy and number desnity profiles
  bool m_initialized_Density = false;
  bool m_initialized_EnergyDensity = false;
  // Radial profiling
  bool m_initialized_RadialVelocity = false;
  bool m_initialized_TangentialVelocity = false;
  bool m_initialized_RadialMomentum = false;

  size_t m_binsCount_Momentum;
  size_t m_binsCount_MomentumIn;
  size_t m_binsCount_MomentumOut;
  size_t m_binsCount_Density;
  size_t m_binsCount_EnergyDensity;
  size_t m_binsCount_RadialVelocity;
  size_t m_binsCount_TangentialVelocity;

  numType m_minMomentum_MomentumIn;
  numType m_minMomentum_MomentumOut;

  numType m_maxMomentum_Momentum;
  numType m_maxMomentum_MomentumIn;
  numType m_maxMomentum_MomentumOut;
  numType m_maxRadius_Density;
  numType m_maxRadius_EnergyDensity;
  numType m_maxRadius_RadialVelocity;
  numType m_maxRadius_TangentialVelocity;

  numType m_dp_Momentum;
  numType m_dp_MomentumIn;
  numType m_dp_MomentumOut;
  numType m_dr_Density;
  numType m_dr_EnergyDensity;
  numType m_dr_RadialVelocity;
  numType m_dr_TangentialVelocity;
};

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
