#pragma once
#define _USE_MATH_DEFINES

#include <chrono>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>

#include "base.h"
#include "collision.h"
#include "config_reader.hpp"
#include "datastreamer.h"
#include "objects.h"
#include "opencl_kernels.h"
#include "simulation.h"
#include "stack"

using my_clock = std::chrono::steady_clock;

template <class Rep, std::intmax_t num, std::intmax_t denom>
std::string convertTimeToHMS(
    std::chrono::duration<Rep, std::ratio<num, denom>> d) {
  auto h = std::chrono::duration_cast<std::chrono::hours>(d);
  d -= h;
  auto m = std::chrono::duration_cast<std::chrono::minutes>(d);
  d -= m;
  auto s = std::chrono::duration_cast<std::chrono::seconds>(d);
  // std::string result = std::to_string(h.count()) + ":" +
  // std::to_string(m.count()) + ":" + std::to_string(s.count()) + "s";
  
  std::string hh =
      (h.count() < 10) ? "0" + std::to_string(int(h.count())) : std::to_string(int(h.count()));
  std::string mm = (m.count() < 10) ? "0" + std::to_string(int(m.count()))
                                    : std::to_string(int(m.count()));
  std::string ss = (s.count() < 10) ? "0" + std::to_string(int(s.count()))
                                    : std::to_string(int(s.count()));

  std::string result = hh + ":" + mm + ":" + ss;  // std::format("{:%T}", s);
  return result;
}

std::string createFileNameFromCurrentDate() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);
  static std::uniform_int_distribution<> dis2(8, 11);

  char c_name[18];

  int i;
  std::stringstream ss;
  std::string result;

  std::time_t t_cur = time(NULL);
  const auto* t_local = localtime(&t_cur);
  std::strftime(c_name, 18, "%j_%Y_%H_%M_%S", t_local);

  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  result = std::string(c_name) + "_" + ss.str();
  return result;
}

std::filesystem::path createSimulationFilePath(std::string& t_dataPath,
                                               std::string& t_fileName) {
  std::filesystem::path dataPath(t_dataPath);
  std::filesystem::path filePath = dataPath / t_fileName;
  if (!std::filesystem::is_directory(dataPath) ||
      !std::filesystem::exists(dataPath)) {       // Check if src folder exists
    std::filesystem::create_directory(dataPath);  // create src folder
  }
  if (!std::filesystem::is_directory(filePath) ||
      !std::filesystem::exists(filePath)) {       // Check if src folder exists
    std::filesystem::create_directory(filePath);  // create src folder
  }
  return filePath;
}

void createSimulationInfoFile(std::ofstream& infoStream,
                              std::filesystem::path& filePath,
                              ConfigReader& t_config, numType t_dV) {
  infoStream << "file_name,seed,alpha,eta,upsilon,tau,m-,T-,N-,m+,T+,N+,"
                "bubbleInteraction,selfInteraction,"
                "radius,speed,Rb,Rc,"
                "dV,deltaN,runtime"
             << std::endl;
  numType critical_radius =
      2 * t_config.parameterUpsilon * t_config.bubbleInitialRadius;

  infoStream << filePath.filename() << "," << t_config.m_seed << ","
             << t_config.parameterAlpha << "," << t_config.parameterEta << ",";
  infoStream << t_config.parameterUpsilon << "," << t_config.parameterTau;
  infoStream << t_config.particleMassFalse << ","
             << t_config.particleTemperatureFalse << ","
             << t_config.particleCountFalse << ",";
  infoStream << t_config.particleMassTrue << ","
             << t_config.particleTemperatureTrue << ","
             << t_config.particleCountTrue << ",";
  infoStream << t_config.bubbleInteractionsOn << "," << t_config.collision_on
             << ",";
  infoStream << t_config.bubbleInitialRadius << ","
             << t_config.bubbleInitialSpeed << ","
             << t_config.cyclicBoundaryRadius * t_config.cyclicBoundaryOn << ","
             << critical_radius << ",";
  infoStream << t_dV << ",";
}

void appendSimulationInfoFile(std::ofstream& infoStream,
                              int t_postionDifference, int t_programRuntime) {
  infoStream << t_postionDifference << "," << t_programRuntime << std::endl;
}
