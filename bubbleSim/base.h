#pragma once
#define _USE_MATH_DEFINES

#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>
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

//#define TIME_COLLIDE_DEBUG
// #define LOG_DEBUG

typedef cl_double cl_numType;
typedef double numType;
typedef unsigned int u_int;
