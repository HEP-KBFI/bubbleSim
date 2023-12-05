#pragma once
#define _USE_MATH_DEFINES

#define CL_MINIMUM_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#if __has_include("CL/opencl.hpp")
# include <CL/opencl.hpp>
#elif __has_include("CL/cl2.hpp")
# include <CL/cl2.hpp>
#else
# include <CL/cl.hpp>
#endif
 
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ankerl/ankerl_map.hpp"

// #define TIME_COLLIDE_DEBUG
// #define LOG_DEBUG
// #define DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE

typedef cl_double cl_numType;
typedef double numType;
typedef unsigned int u_int;

enum SimulationFlags : std::uint32_t {
  BUBBLE_ON = 0b00000000000000000000000000000001,
  TRUE_VACUUM_BUBBLE_ON = 0b00000000000000000000000000000010,
  BUBBLE_INTERACTION_ON = 0b00000000000000000000000000000100,
  COLLISION_ON = 0b00000000000000000000000000001000,
  SIMULATION_BOUNDARY_ON = 0b00000000000000000000000000010000,
  COLLISION_MASS_STATE_ON = 0b00000000000000000000000000100000,
};

inline SimulationFlags operator|(SimulationFlags a, SimulationFlags b) {
  return static_cast<SimulationFlags>(static_cast<std::uint32_t>(a) |
                                      static_cast<std::uint32_t>(b));
}

enum StreamFlags : std::uint32_t {
  STREAM_ON = 0b00000000000000000000000000000001,
  STREAM_TIMESERIES = 0b00000000000000000000000000000010,
  STREAM_PROFILE = 0b00000000000000000000000000000100,
  STREAM_MOMENTUM = 0b00000000000000000000000000001000,
  STREAM_MOMENTUM_PROFILE = 0b00000000000000000000000000010000,
};

inline StreamFlags operator|(StreamFlags a, StreamFlags b) {
  return static_cast<StreamFlags>(static_cast<std::uint32_t>(a) |
                                  static_cast<std::uint32_t>(b));
}

enum SimulationBufferFlags : std::uint64_t {
  PARTICLE_X_BUFFER = (std::uint64_t)1 << 0,
  PARTICLE_Y_BUFFER = (std::uint64_t)1 << 1,
  PARTICLE_Z_BUFFER = (std::uint64_t)1 << 2,
  PARTICLE_E_BUFFER = (std::uint64_t)1 << 3,
  PARTICLE_PX_BUFFER = (std::uint64_t)1 << 4,
  PARTICLE_PY_BUFFER = (std::uint64_t)1 << 5,
  PARTICLE_PZ_BUFFER = (std::uint64_t)1 << 6,
  PARTICLE_M_BUFFER = (std::uint64_t)1 << 7,
  PARTICLE_COLLIDE_BUFFER = (std::uint64_t)1 << 8,
  PARTICLE_IN_BUBBLE_BUFFER = (std::uint64_t)1 << 9,
  PARTICLE_COLLISION_CELL_IDX_BUFFER = (std::uint64_t)1 << 10,
  PARTICLE_dP_BUFFER = (std::uint64_t)1 << 11,
  PARTICLE_INTERACTED_FALSE_BUFFER = (std::uint64_t)1 << 12,
  PARTICLE_INTERACTED_TRUE_BUFFER = (std::uint64_t)1 << 13,
  PARTICLE_PASSED_FALSE_BUFFER = (std::uint64_t)1 << 14,
  CELL_THETA_ROTATION_BUFFER = (std::uint64_t)1 << 15,
  CELL_THETA_AXIS_BUFFER = (std::uint64_t)1 << 16,
  CELL_PHI_AXIS_BUFFER = (std::uint64_t)1 << 17,
  CELL_E_BUFFER = (std::uint64_t)1 << 18,
  CELL_LOGE_BUFFER = (std::uint64_t)1 << 19,
  CELL_PX_BUFFER = (std::uint64_t)1 << 20,
  CELL_PY_BUFFER = (std::uint64_t)1 << 21,
  CELL_PZ_BUFFER = (std::uint64_t)1 << 22,
  CELL_COLLIDE_BUFFER = (std::uint64_t)1 << 23,
  CELL_PARTICLE_COUNT_BUFFER = (std::uint64_t)1 << 24,
  CELL_LENGTH_BUFFER = (std::uint64_t)1 << 25,
  CELL_COUNT_IN_ONE_AXIS_BUFFER = (std::uint64_t)1 << 26,
  CELL_SHIFT_VECTOR_BUFFER = (std::uint64_t)1 << 27,
  CELL_SEED_INT64_BUFFER = (std::uint64_t)1 << 28,
  CELL_NO_COLLISION_PROBABILITY_BUFFER = (std::uint64_t)1 << 29,
  BUBBLE_BUFFER = (std::uint64_t)1 << 30,
  SIMULATION_DT_BUFFER = (std::uint64_t)1 << 31,
  SIMULATION_BOUNDARY_BUFFER = (std::uint64_t)1 << 32,
  SIMULATION_MASS_IN_BUFFER = (std::uint64_t)1 << 33,
  SIMULATION_MASS_OUT_BUFFER = (std::uint64_t)1 << 34,
  SIMULATION_DELTA_MASS_BUFFER = (std::uint64_t)1 << 35
};

inline SimulationBufferFlags operator|(SimulationBufferFlags a,
                                       SimulationBufferFlags b) {
  return static_cast<SimulationBufferFlags>(static_cast<std::uint64_t>(a) |
                                            static_cast<std::uint64_t>(b));
}
