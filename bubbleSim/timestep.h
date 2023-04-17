#pragma once
#include "base.h"

class TimestepAdapter {
 protected:
  numType m_current_dt;

 public:
  TimestepAdapter() { m_current_dt = 0; }
  TimestepAdapter(numType dt) { m_current_dt = dt; }

  numType getTimestep() { return m_current_dt; }
};

class TimestepAdapterDependentOnRadius : public TimestepAdapter {
 public:
  TimestepAdapterDependentOnRadius() { m_current_dt = 0; }
  TimestepAdapterDependentOnRadius(numType dt) { m_current_dt = dt; }
  void calculateNewTimeStep(numType bubbleRadius, numType radius_factor) {
    // dt = R_b * radius_factor
    m_current_dt = bubbleRadius * radius_factor;
  }
};

class TimestepAdapterDependentOnRadiusSpeed : public TimestepAdapter {
 public:
  TimestepAdapterDependentOnRadiusSpeed() { m_current_dt = 0; }
  TimestepAdapterDependentOnRadiusSpeed(numType dt) { m_current_dt = dt; }
  void calculateNewTimeStep(numType bubbleRadius, numType radius_factor) {
    // dt = R_b * radius_factor
    m_current_dt = bubbleRadius * radius_factor;
  }
  void calculateNewTimeStep(numType speedChange, numType bubbleRadius,
                            numType radius_factor) {
    if (speedChange >= 0.01) {
      m_current_dt =
          m_current_dt * 0.9 * std::pow(speedChange / 0.01, -1. / 2.);
      // std::cout << "Speed change > 0.05: dt = " << getTimestep() <<
      // std::endl;
    } else if (speedChange <= 0.01 * 0.5) {
      m_current_dt =
          std::min(m_current_dt * 0.9 * std::pow(speedChange / 0.01, -1. / 3.),
                   bubbleRadius * radius_factor);
      // std::cout << "Speed change < 0.05: dt = " << getTimestep() <<
      // std::endl;
    }
  }
};