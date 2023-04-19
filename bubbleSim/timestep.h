#pragma once
#include "base.h"

class TimestepAdapter {
 protected:
  numType m_current_dt;
  numType m_max_dt;

 public:
  TimestepAdapter() {
    m_current_dt = 0.;
    m_max_dt = 1.;
  }
  TimestepAdapter(numType dt, numType max_dt) {
    m_current_dt = dt;
    m_max_dt = max_dt;
  }
  numType getTimestep() { return m_current_dt; }

  void calculateNewTimeStep() { m_current_dt = m_current_dt / 10; }

  void claculateNewTimeStep(numType bubbleSpeedChange, numType bubbleRadius,
                            numType initialBubbleRadius, numType radiusFactor,
                            numType bubbleSpeed) {
    // https://www.gnu.org/software/gsl/doc/html/ode-initval.html#adaptive-step-size-control
    if (bubbleSpeedChange >= 0.02) {
      m_current_dt =
          m_current_dt * 0.9 * std::pow(bubbleSpeedChange / 0.02, -1. / 2.);
    } else if (bubbleSpeedChange <= 0.01 * 0.5) {
      m_current_dt = std::min(
          m_current_dt * 0.9 * std::pow(bubbleSpeedChange / 0.02, -1. / 3.),
          bubbleRadius / std::abs(100));
      m_current_dt = std::min(m_current_dt, m_max_dt);
    }
  }
};