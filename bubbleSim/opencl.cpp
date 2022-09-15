#include "opencl.h"

cl::Context CreateContext(std::vector<cl::Device>& devices) {
    int errNum;
    std::vector<cl::Platform> platforms;
    cl::Context context;

    errNum = cl::Platform::get(&platforms);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        exit(1);
    }

    for (const auto& value : platforms) {
        if (value.getInfo<CL_PLATFORM_NAME>().find("NVIDIA") != std::string::npos) {
            std::vector<cl::Device> temp_devices;
            value.getDevices(CL_DEVICE_TYPE_GPU, &temp_devices);
            for (const auto& value2 : temp_devices) {
                devices.push_back(value2);
            }
        }
    }

    if (devices.size() == 0) {
        std::cerr << "Devices list is empty." << std::endl;
    }

    context = cl::Context::Context(devices, NULL, NULL, NULL, &errNum);

    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed to create a context." << std::endl;
        exit(1);
    }

    return context;
}

cl::Program CreateProgram(cl::Context& context, cl::Device& device, std::string& kernelFile) {

    int errNum;
    cl::Program program;

    std::ifstream kernel_file(kernelFile);
    std::string kernel_code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

    program = cl::Program::Program(context, kernel_code, false, &errNum);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed to create program from source file." << std::endl;
        exit(1);
    }

    errNum = program.build("-cl-std=CL1.2");
    if (errNum != CL_BUILD_SUCCESS) {
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
            << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    return program;
}

cl::Kernel CreateKernel(cl::Program& program, const char* name) {
    int errNum;
    cl::Kernel kernel;
    kernel = cl::Kernel::Kernel(program, name, &errNum);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed to create a kernel: " << name << std::endl;
        exit(1);
    }

    return kernel;
}

cl::CommandQueue CreateQueue(cl::Context& context, cl::Device& device) {
    int errNum;
    cl::CommandQueue queue;
    cl_command_queue_properties properties = 0;
    queue = cl::CommandQueue(context, device, properties, &errNum);
    // queue = cl::CommandQueue(context, device);
    //if (errNum != CL_SUCCESS) {
    //    std::cerr << "Failed to create a CommandQueue: " << std::endl;
    //}

    return queue;
}
