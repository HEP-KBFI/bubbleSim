#pragma once
#include "base.h"
#include "config_reader.hpp"

typedef struct Bubble {
  cl_numType radius;
  cl_numType radius2;           // Squared
  cl_numType radiusAfterStep2;  // (radius + speed * dt)^2

  cl_numType speed;

  cl_numType gamma;
  cl_numType gammaXspeed;  // gamma * speed
} Bubble;

class PhaseBubble {
 public:
  numType getRadius() { return m_bubble.radius; }
  numType getSpeed() { return m_bubble.speed; }
  numType getGamma() { return m_bubble.gamma; }
  numType getGammaSpeed() { return m_bubble.gammaXspeed; }
  numType getRadius2() { return m_bubble.radius2; }
  numType getRadiusAfterDt2() { return m_bubble.radiusAfterStep2; }
  numType getdV() { return m_dV; }
  numType getSigma() { return m_sigma; }

  PhaseBubble(numType t_initialRadius, numType t_initialSpeed, numType t_dV,
              numType t_sigma, cl::Context& cl_context);

  void evolveWall(numType dt, numType dP);
  numType calculateArea();
  numType calculateVolume();
  numType calculateRadiusAfterStep2(numType dt);
  numType calculateEnergy();

  cl::Buffer& getBubbleBuffer() { return m_bubbleBuffer; }
  void writeBubbleBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_bubbleBuffer, CL_TRUE, 0, sizeof(Bubble),
                                &m_bubble);
  }
  void readBubbleBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_bubbleBuffer, CL_TRUE, 0, sizeof(Bubble),
                               &m_bubble);
  }

  void writeAllBuffersToKernel(cl::CommandQueue& cl_queue) {
    writeBubbleBuffer(cl_queue);
  }

  void print_info(ConfigReader& t_config);

 private:
  Bubble m_bubble;
  cl::Buffer m_bubbleBuffer;
  numType m_dV;
  numType m_sigma;
};