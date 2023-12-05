#pragma once
#include "base.h"
#include "timestep.h"


class SimulationParameters {
  numType m_boundaryRadius;
  cl::Buffer m_boundaryRadiusBuffer;
  numType m_dt;
  cl::Buffer m_dtBuffer;
  numType m_mass_in;
  cl::Buffer m_mass_in_buffer;
  numType m_mass_out;
  cl::Buffer m_mass_out_buffer;
  numType m_delta_mass_squared;
  cl::Buffer m_delta_mass_squared_buffer;

  TimestepAdapter m_timestepAdapter;
  

 public:
  SimulationParameters(){};
  SimulationParameters(numType t_dt,
                       std::uint64_t& t_buffer_flags, cl::Context& cl_context);
  SimulationParameters(numType t_dt, numType t_mass_in, numType t_mass_out,
                       std::uint64_t& t_buffer_flags, cl::Context& cl_context);
  SimulationParameters(numType t_dt, numType t_mass_in, numType t_mass_out,
                       numType t_boundaryRadius,
                       std::uint64_t& t_buffer_flags,
                       cl::Context& cl_context);
  numType getDt() { return m_dt; }
  numType getBoundaryRadius() { return m_boundaryRadius; }
  TimestepAdapter& getTimestepAdapter() { return m_timestepAdapter; }
  cl::Buffer &getDtBuffer() { return m_dtBuffer; }
  cl::Buffer &getBoundaryBuffer() { return m_boundaryRadiusBuffer; }
  cl::Buffer& getMassInBuffer() { return m_mass_in_buffer; }
  cl::Buffer& getMassOutBuffer() { return m_mass_out_buffer; }
  cl::Buffer& getDeltaMassSquaredBuffer() { return m_delta_mass_squared_buffer; }

  /*
   * ================================================================
   * ================================================================
   *                        Buffer writers
   * ================================================================
   * ================================================================
   */
  void writeBoundaryRadiusBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_boundaryRadiusBuffer, CL_TRUE, 0, sizeof(numType), &m_boundaryRadius);
  }

  void writeDtBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_dtBuffer, CL_TRUE, 0, sizeof(numType),
                                &m_dt);
  }

  void writeMassInBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_mass_in_buffer, CL_TRUE, 0, sizeof(numType), &m_mass_in);
  }

  void writeMassOutBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_mass_out_buffer, CL_TRUE, 0, sizeof(numType),
                                &m_mass_out);
  }

  /*
   * ================================================================
   * ================================================================
   *                        Buffer readers
   * ================================================================
   * ================================================================
   */
  void readDtBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_dtBuffer, CL_TRUE, 0, sizeof(numType), &m_dt);
  }

};