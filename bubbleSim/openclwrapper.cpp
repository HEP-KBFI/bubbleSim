#include "openclwrapper.h"

OpenCLWrapper::OpenCLWrapper(std::string kernelPath, std::string kernelName,
                             std::vector<Particle> t_particles,
                             std::vector<numType>& t_dP, numType& t_dt,
                             numType& t_massTrue, numType& t_massFalse,
                             numType t_massDelta2, Bubble t_bubble,
                             std::vector<int8_t>& t_interactedFalse,
                             std::vector<int8_t>& t_passedFalse,
                             std::vector<int8_t>& t_interactedTrue,
                             bool t_isBubbleTrueVacuum) {
  int errNum;

  createContext(m_devices);
  createProgram(m_context, m_deviceUsed, kernelPath);
  createKernel(m_program, kernelName.c_str());  // cl::Kernel
  createQueue(m_context, m_deviceUsed);

  // Set buffers

  m_bufferParticle = cl::Buffer(
      m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      t_particles.size() * sizeof(Particle), t_particles.data(), &errNum);
  m_buffer_dP =
      cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 t_particles.size() * sizeof(numType), t_dP.data(), &errNum);

  m_buffer_dt = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(numType), &t_dt, &errNum);
  if (t_isBubbleTrueVacuum) {
    m_bufferMassIn =
        cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(numType), &t_massTrue, &errNum);
    m_bufferMassOut =
        cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(numType), &t_massFalse, &errNum);
  } else {
    m_bufferMassIn =
        cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(numType), &t_massFalse, &errNum);
    m_bufferMassOut =
        cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(numType), &t_massTrue, &errNum);
  }

  m_bufferMassDelta2 =
      cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &t_massDelta2, &errNum);

  m_bufferBubble =
      cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(Bubble), &t_bubble, &errNum);

  m_bufferInteractedFalse = cl::Buffer(
      m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      t_particles.size() * sizeof(int8_t), t_interactedFalse.data(), &errNum);
  m_bufferPassedFalse = cl::Buffer(
      m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      t_particles.size() * sizeof(int8_t), t_passedFalse.data(), &errNum);
  m_bufferInteractedTrue = cl::Buffer(
      m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      t_particles.size() * sizeof(int8_t), t_interactedTrue.data(), &errNum);

  errNum = m_kernel.setArg(0, m_bufferParticle);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize X buffer." << std::endl;
  }
  errNum = m_kernel.setArg(1, m_buffer_dP);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize dP buffer." << std::endl;
  }
  errNum = m_kernel.setArg(2, m_buffer_dt);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize dt buffer." << std::endl;
  }
  errNum = m_kernel.setArg(3, m_bufferMassIn);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize mass_in buffer." << std::endl;
  }
  errNum = m_kernel.setArg(4, m_bufferMassOut);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize mass_out buffer." << std::endl;
  }
  errNum = m_kernel.setArg(5, m_bufferMassDelta2);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize Dm2 buffer." << std::endl;
  }
  errNum = m_kernel.setArg(6, m_bufferBubble);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize R buffer." << std::endl;
  }
  errNum = m_kernel.setArg(7, m_bufferInteractedFalse);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize InteractedFalse buffer." << std::endl;
  }
  errNum = m_kernel.setArg(8, m_bufferPassedFalse);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize PassedFalse buffer." << std::endl;
  }
  errNum = m_kernel.setArg(9, m_bufferInteractedTrue);
  if (errNum != CL_SUCCESS) {
    std::cout << "Couldn't initialize InteractedTrue buffer." << std::endl;
  }
}

void OpenCLWrapper::createContext(std::vector<cl::Device>& devices) {
  int errNum;
  std::vector<cl::Platform> platforms;

  errNum = cl::Platform::get(&platforms);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to find any OpenCL platforms." << std::endl;
    exit(1);
  }

  for (const auto& value : platforms) {
    std::cout << "Available platform: " << value.getInfo<CL_PLATFORM_NAME>()
              << std::endl;
  }

  // try to find NVIDIA device
  for (const auto& value : platforms) {
    const auto& platform_name = value.getInfo<CL_PLATFORM_NAME>();
    if (platform_name.find("NVIDIA") != std::string::npos) {
      std::vector<cl::Device> temp_devices;
      value.getDevices(CL_DEVICE_TYPE_GPU, &temp_devices);
      for (const auto& value2 : temp_devices) {
        devices.push_back(value2);
      }
      break;
    }
  }

  // try to find CPU device
  if (devices.size() == 0) {
    for (const auto& value : platforms) {
      const auto& platform_name = value.getInfo<CL_PLATFORM_NAME>();
      if ((platform_name.find("Portable Computing Language") !=
           std::string::npos) ||
          (platform_name.find("Oclgrind") != std::string::npos)) {
        std::vector<cl::Device> temp_devices;
        value.getDevices(CL_DEVICE_TYPE_CPU, &temp_devices);
        for (const auto& value2 : temp_devices) {
          devices.push_back(value2);
        }
        break;
      }
    }
  }

  if (devices.size() == 0) {
    std::cerr << "Devices list is empty." << std::endl;
    exit(1);
  }
  if (devices.size() > 1) {
    std::cerr << "Found more than one device." << std::endl;
    exit(1);
  }

  std::cout << "Device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

  m_context = cl::Context(devices, NULL, NULL, NULL, &errNum);
  m_deviceUsed = devices[0];

  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to create a context." << std::endl;
    exit(1);
  }
}

void OpenCLWrapper::createProgram(cl::Context& context, cl::Device& device,
                                  std::string& kernelFile) {
  int errNum;

  std::ifstream kernel_file(kernelFile);
  std::string kernel_code(std::istreambuf_iterator<char>(kernel_file),
                          (std::istreambuf_iterator<char>()));

  m_program = cl::Program(context, kernel_code, false, &errNum);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to create program from source file. Cehck if kernel "
                 "file location is correct. ("
              << kernelFile << ")" << std::endl;
    exit(1);
  }

  errNum = m_program.build("-cl-std=CL1.2");
  if (errNum != CL_BUILD_SUCCESS) {
    std::cerr << "Error!\nBuild Status: "
              << m_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
              << "\nBuild Log:\t "
              << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    exit(1);
  }
}

void OpenCLWrapper::createKernel(cl::Program& program, const char* name) {
  int errNum;
  m_kernel = cl::Kernel(program, name, &errNum);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to create a kernel: " << name << std::endl;
    exit(1);
  }
}

void OpenCLWrapper::createQueue(cl::Context& context, cl::Device& device) {
  int errNum;
  cl::CommandQueue queue;
  cl_command_queue_properties properties = 0;
  m_queue = cl::CommandQueue(context, device, properties, &errNum);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to create a CommandQueue: " << std::endl;
  }
}

void OpenCLWrapper::makeStep1(Bubble& t_bubble) {
  m_queue.enqueueWriteBuffer(m_bufferBubble, CL_TRUE, 0, sizeof(Bubble),
                             &t_bubble);
}

void OpenCLWrapper::makeStep2(int& particleCount) {
  m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange,
                               cl::NDRange(particleCount));
}

void OpenCLWrapper::makeStep3(int& particleCount, std::vector<numType>& t_dP) {
  m_queue.enqueueReadBuffer(m_buffer_dP, CL_TRUE, 0,
                            particleCount * sizeof(numType), t_dP.data());
}

void OpenCLWrapper::makeStep4(Bubble& t_bubble) {
  m_queue.enqueueWriteBuffer(m_bufferBubble, CL_TRUE, 0, sizeof(Bubble),
                             &t_bubble);
}

void OpenCLWrapper::readBufferParticle(
    std::vector<Particle>& t_vectorParticle) {
  m_queue.enqueueReadBuffer(m_bufferParticle, CL_TRUE, 0,
                            t_vectorParticle.size() * sizeof(Particle),
                            t_vectorParticle.data());
}

void OpenCLWrapper::readBuffer_dP(std::vector<numType>& t_data_dP) {
  m_queue.enqueueReadBuffer(m_buffer_dP, CL_TRUE, 0,
                            t_data_dP.size() * sizeof(numType),
                            t_data_dP.data());
}

void OpenCLWrapper::readBufferInteractedFalse(
    std::vector<int8_t>& t_dataInteractedFalse) {
  m_queue.enqueueReadBuffer(m_bufferInteractedFalse, CL_TRUE, 0,
                            t_dataInteractedFalse.size() * sizeof(int8_t),
                            t_dataInteractedFalse.data());
}

void OpenCLWrapper::readBufferPassedFalse(
    std::vector<int8_t>& t_dataPassedFalse) {
  m_queue.enqueueReadBuffer(m_bufferPassedFalse, CL_TRUE, 0,
                            t_dataPassedFalse.size() * sizeof(int8_t),
                            t_dataPassedFalse.data());
}

void OpenCLWrapper::readBufferInteractedTrue(
    std::vector<int8_t>& t_dataInteractedTrue) {
  m_queue.enqueueReadBuffer(m_bufferInteractedTrue, CL_TRUE, 0,
                            t_dataInteractedTrue.size() * sizeof(int8_t),
                            t_dataInteractedTrue.data());
}

void OpenCLWrapper::writeResetInteractedFalseBuffer(
    std::vector<int8_t>& v_interacted) {
  std::fill(v_interacted.begin(), v_interacted.end(), 0);
  m_queue.enqueueWriteBuffer(m_bufferInteractedFalse, CL_TRUE, 0,
                             v_interacted.size() * sizeof(int8_t),
                             v_interacted.data());
};
void OpenCLWrapper::writeResetPassedFalseBuffer(std::vector<int8_t>& v_passed) {
  std::fill(v_passed.begin(), v_passed.end(), 0);
  m_queue.enqueueWriteBuffer(m_bufferPassedFalse, CL_TRUE, 0,
                             v_passed.size() * sizeof(int8_t), v_passed.data());
};
void OpenCLWrapper::writeResetInteractedTrueBuffer(
    std::vector<int8_t>& v_interacted) {
  std::fill(v_interacted.begin(), v_interacted.end(), 0);
  m_queue.enqueueWriteBuffer(m_bufferInteractedTrue, CL_TRUE, 0,
                             v_interacted.size() * sizeof(int8_t),
                             v_interacted.data());
};
