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
  void initStream_MomentumIn(size_t t_binsCount, numType t_maxMomentumValue);
  void initStream_MomentumOut(size_t t_binsCount, numType t_maxMomentumValue);
  void initStream_Density(size_t t_binsCount, numType t_maxRadiusValue);
  void initStream_EnergyDensity(size_t t_binsCount, numType t_maxRadiusValue);
  void initStream_RadialVelocity(size_t t_binsCount, numType t_maxRadiusValue);
  void initStream_TangentialVelocity(size_t t_binsCount,
                                     numType t_maxRadiusValue);
  void initStream_radialMomentum(size_t t_binsCount, numType t_maxRadiusValue);
  void initStream_Momentum(size_t t_binsCount, numType t_maxMomentumValue);
  void stream(Simulation& simulation, ParticleCollection& particleCollection,
              PhaseBubble& bubble, cl::CommandQueue& cl_queue);

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
  void StreamRadialMomentumProfile(
      std::ofstream& t_stream, size_t t_binsCountRadius,
      size_t t_binsCountMomentum, numType t_minRadiusValue,
      numType t_maxRadiusValue, numType t_minMomentumValue,
      numType t_maxMomentumValue, ParticleCollection& particleCollection,
      cl::CommandQueue& cl_queue);

 private:
  std::filesystem::path m_filePath;

  std::ofstream m_stream_Data;
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
