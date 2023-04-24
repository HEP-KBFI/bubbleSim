#pragma once
#include "base.h"

class TimestepAdapter {
 protected:
  numType m_current_dt;
  numType m_max_dt;

  u_int m_repeatCounter = 0;
  u_int m_repeatValue = 0;

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

  void calculateNewTimeStep() { m_current_dt = m_current_dt / 2; }

  void calculateNewTimeStep(PhaseBubble& bubble) {
    m_current_dt = std::min(bubble.getRadius() / 1000, m_max_dt);
  }
};