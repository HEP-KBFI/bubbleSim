#pragma once
#include "base.h"
#include "components.h"

class PhaseBubble {
  // Input parameters
  /*
   Phase transition bubble. Has information about:
  radius, radius2, speed, gamma, gammaXspeed, radiusAfterStep2
  */
  Bubble m_bubble;
  numType m_dV;
  numType m_sigma;
  // Calculated parameters from input

  // 1) radius, speed, gamma, gamma*speed, radius^2, (radius*v*dt)^2 are used by
  // GPU

 public:
  numType getRadius() { return m_bubble.radius; }
  numType getSpeed() { return m_bubble.speed; }
  numType getGamma() { return m_bubble.gamma; }
  numType getGammaSpeed() { return m_bubble.gammaXspeed; }
  numType getRadius2() { return m_bubble.radius2; }
  numType getRadiusAfterDt2() { return m_bubble.radiusAfterStep2; }
  numType getdV() { return m_dV; }
  numType getSigma() { return m_sigma; }

  Bubble& getRef_Bubble() { return m_bubble; }
  numType& getRef_dV() { return m_dV; }
  numType& getRef_Sigma() { return m_sigma; }

  PhaseBubble() {}
  PhaseBubble(numType t_initialRadius, numType t_initialSpeed, numType t_dV,
              numType t_sigma);

  void evolveWall(numType dt, numType dP);
  numType calculateArea();
  numType calculateVolume();
  numType calculateRadiusAfterStep2(numType dt);
  numType calculateEnergy();
};