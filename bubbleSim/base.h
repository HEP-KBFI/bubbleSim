#pragma once
#define _USE_MATH_DEFINES

#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "ankerl/ankerl_map.hpp"

// #define LOG_DEBUG

typedef cl_double cl_numType;
typedef double numType;
typedef unsigned int u_int;