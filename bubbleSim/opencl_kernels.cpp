#include "opencl_kernels.h"

OpenCLLoader::OpenCLLoader(std::string kernelPath) {
  int errNum;

  std::string particleBubbleStepKernelName = "particle_bubble_step";
  std::string cellAssignKernelName = "assign_cell_index_to_particle";
  std::string transformKernelName = "transform_momentum";
  std::string particleStepKernelName = "particle_step";
  std::string particleBounceKernelName = "particle_bounce";

  createContext(m_devices);
  createProgram(m_context, m_deviceUsed, kernelPath);
  createKernel(m_program, m_particleBubbleStepKernel,
               particleBubbleStepKernelName.c_str());  // cl::Kernel
  createKernel(m_program, m_rotationKernel, transformKernelName.c_str());
  createKernel(m_program, m_cellAssignmentKernel, cellAssignKernelName.c_str());
  createKernel(m_program, m_particleStepKernel, particleStepKernelName.c_str());
  createKernel(m_program, m_particleBounceKernel,
               particleBounceKernelName.c_str());

  createQueue(m_context, m_deviceUsed);
}

void OpenCLLoader::createContext(std::vector<cl::Device>& devices) {
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

void OpenCLLoader::createProgram(cl::Context& context, cl::Device& device,
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

void OpenCLLoader::createKernel(cl::Program& program, cl::Kernel& kernel,
                                const char* name) {
  int errNum;
  kernel = cl::Kernel(program, name, &errNum);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to create a kernel: " << name << std::endl;
    exit(1);
  }
}

void OpenCLLoader::createQueue(cl::Context& context, cl::Device& device) {
  int errNum;
  cl::CommandQueue queue;
  cl_command_queue_properties properties = 0;
  m_queue = cl::CommandQueue(context, device, properties, &errNum);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to create a CommandQueue: " << std::endl;
  }
}
