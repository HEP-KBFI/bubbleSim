#pragma once
#define _USE_MATH_DEFINES

#include <filesystem>
#include <fstream>

#include "base.h"
#include "bubble.h"
#include "opencl_kernels.h"
#include "simulation.h"

class DataStreamer {
 public:
  DataStreamer(){};
  DataStreamer(std::string filePath);

  void initMomentumProfile(size_t t_binsCount, numType t_maxMomentumValue);
  void initDensityProfile(size_t t_binsCount, numType t_maxRadiusValue);
  void initData();
  void stream(Simulation& simulation, ParticleCollection& particleCollection,
              PhaseBubble& bubble, cl::CommandQueue& cl_queue);

 private:
  std::filesystem::path m_filePath;

  std::fstream m_fileMomentumIn;
  std::fstream m_fileMomentumOut;
  std::fstream m_fileDensity;
  std::fstream m_fileData;

  bool m_momentumInitialized = false;
  bool m_densityInitialized = false;
  bool m_dataInitialized = false;

  size_t m_momentumBinsCount;
  size_t m_densityBinsCount;

  numType m_maxMomentumValue;
  numType m_maxRadiusValue;
};
