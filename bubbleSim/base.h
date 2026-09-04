#pragma once
#define _USE_MATH_DEFINES

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "layout.h"   // → precision.h : sim::real, sim::accum, BSIM_HD

// #define LOG_DEBUG

// Transitional — delete at the end of phase 3.
using numType = sim::real;
using u_int   = uint32_t;