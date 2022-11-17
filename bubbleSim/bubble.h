#pragma once
#include "base.h"

typedef struct Bubble {
  cl_numType radius;
  cl_numType radius2;           // Squared
  cl_numType radiusAfterStep2;  // (radius + speed * dt)^2

  cl_numType speed;

  cl_numType gamma;
  cl_numType gammaXspeed;  // gamma * speed
} Bubble;

class PhaseBubble {
  // Input parameters
  /*
   Phase transition bubble. Has information about:
  radius, radius2, speed, gamma, gammaXspeed, radiusAfterStep2
  */
  Bubble m_bubble;
  cl::Buffer m_bubbleBuffer;
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

  cl::Buffer& getBubbleBuffer() { return m_bubbleBuffer; }
  void writeBubbleBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_bubbleBuffer, CL_TRUE, 0, sizeof(Bubble),
                                &m_bubble);
  }
  void readBubbleBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_bubbleBuffer, CL_TRUE, 0, sizeof(Bubble),
                               &m_bubble);
  }

  PhaseBubble() {}
  PhaseBubble(numType t_initialRadius, numType t_initialSpeed, numType t_dV,
              numType t_sigma, cl::Context& cl_context);

  void evolveWall(numType dt, numType dP);
  numType calculateArea();
  numType calculateVolume();
  numType calculateRadiusAfterStep2(numType dt);
  numType calculateEnergy();
};