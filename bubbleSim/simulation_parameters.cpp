#include "simulation_parameters.h"

SimulationParameters::SimulationParameters(
    numType t_dt, std::uint64_t& t_buffer_flags,
                                           cl::Context& cl_context) {
  int openCLerrNum = 0;
  m_dt = t_dt;
  m_timestepAdapter = TimestepAdapter(t_dt, t_dt);
  m_dtBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(numType), &m_dt, &openCLerrNum);
  t_buffer_flags |= SIMULATION_DT_BUFFER;
}

SimulationParameters::SimulationParameters(numType t_dt,
    numType t_mass_in,
                                           numType t_mass_out,
                                           std::uint64_t& t_buffer_flags,
                                           cl::Context& cl_context)
    : SimulationParameters(t_dt, t_buffer_flags,
                           cl_context) {
  int openCLerrNum = 0;
  m_mass_in = t_mass_in;
  m_mass_out = t_mass_out;
  m_delta_mass_squared = std::abs(std::pow(t_mass_in, (numType)2.) -
                                  std::pow(t_mass_out, (numType)2.));
  m_mass_in_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_mass_in, &openCLerrNum);
  t_buffer_flags |= SIMULATION_MASS_IN_BUFFER;
  m_mass_out_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_mass_out, &openCLerrNum);
  t_buffer_flags |= SIMULATION_MASS_OUT_BUFFER;
  m_delta_mass_squared_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_delta_mass_squared, &openCLerrNum);
  t_buffer_flags |= SIMULATION_DELTA_MASS_BUFFER;
}

SimulationParameters::SimulationParameters(numType t_dt,
    numType t_mass_in,
    numType t_mass_out,
                                           numType m_boundaryRadius,
                                           std::uint64_t& t_buffer_flags,
                                           cl::Context& cl_context)
    : SimulationParameters(t_dt, t_mass_in, t_mass_out, t_buffer_flags,
                           cl_context) {
  
  int openCLerrNum = 0;
  m_boundaryRadiusBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_boundaryRadius, &openCLerrNum);
  t_buffer_flags |= SIMULATION_BOUNDARY_BUFFER;
}