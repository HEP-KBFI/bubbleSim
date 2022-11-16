#pragma once
#define _USE_MATH_DEFINES

#include <chrono>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>

#include "base.h"
#include "config_reader.hpp"
#include "datastreamer.h"
#include "objects.h"
#include "opencl_kernels.h"
#include "simulation.h"
