#include "openclwrapper.h"


OpenCLWrapper::OpenCLWrapper(
	std::string fileName, std::string kernelName, u_int t_particleCount,
	std::vector<numType>& t_X, std::vector<numType>& t_P, std::vector<numType>& t_E, std::vector<numType>& t_M,
	std::vector<numType>& t_dP, numType& t_dt, numType& t_massTrue, numType& t_massFalse, numType t_massDelta2,
	numType& t_bubbleRadius, numType& t_bubbleRadius2, numType& t_bubbleRadiusAfterDt2, numType& t_bubbleSpeed,
	numType& t_bubbleGamma, numType& t_bubbleGammaSpeed,
	std::vector<int8_t>& t_interactedFalse, std::vector<int8_t>& t_passedFalse,
	std::vector<int8_t>& t_interactedTrue, bool t_isBubbleTrueVacuum
) {
    int errNum;

    m_platform.getDevices(CL_DEVICE_TYPE_GPU, &m_devices);
    createContext(m_devices);
    createProgram(m_context, m_deviceUsed, fileName);
    createKernel(m_program, kernelName.c_str()); // cl::Kernel 
    createQueue(m_context, m_deviceUsed);

    // Set buffers

    m_bufferX = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 3 * t_particleCount * sizeof(numType), t_X.data(), &errNum);
    m_bufferP = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 3 * t_particleCount * sizeof(numType), t_P.data(), &errNum);
    m_bufferE = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, t_particleCount * sizeof(numType), t_E.data(), &errNum);
    m_bufferM = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, t_particleCount * sizeof(numType), t_M.data(), &errNum);
    m_buffer_dP = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, t_particleCount * sizeof(numType), t_dP.data(), &errNum);

    m_buffer_dt = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_dt, &errNum);
	if (t_isBubbleTrueVacuum) {
		m_bufferMassIn = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_massTrue, &errNum);
		m_bufferMassOut = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_massFalse, &errNum);
	}
	else {
		m_bufferMassIn = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_massFalse , &errNum);
		m_bufferMassOut = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_massTrue, &errNum);
	}
    
    m_bufferMassDelta2 = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_massDelta2, &errNum);
    
    m_bufferBubbleRadius = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_bubbleRadius, &errNum);
    m_bufferBubbleRadius2 = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_bubbleRadius2, &errNum);
    m_bufferBubbleRadiusSpeedDt2 = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_bubbleRadiusAfterDt2, &errNum);
    m_bufferBubbleSpeed = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_bubbleSpeed, &errNum);
    m_bufferBubbleGamma = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_bubbleGamma, &errNum);
    m_bufferBubbleGammaSpeed = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(numType), &t_bubbleGammaSpeed, &errNum);

    m_bufferInteractedFalse = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, t_particleCount * sizeof(int8_t), t_interactedFalse.data(), &errNum);
    m_bufferPassedFalse = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, t_particleCount * sizeof(int8_t), t_passedFalse.data(), &errNum);
    m_bufferInteractedTrue = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, t_particleCount * sizeof(int8_t), t_interactedTrue.data(), &errNum);

	errNum = m_kernel.setArg(0, m_bufferX);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize X buffer." << std::endl;
	}
	errNum = m_kernel.setArg(1, m_bufferP);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize P buffer." << std::endl;
	}
	errNum = m_kernel.setArg(2, m_bufferE);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize E buffer." << std::endl;
	}
	errNum = m_kernel.setArg(3, m_bufferM);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize M buffer." << std::endl;
	}
	errNum = m_kernel.setArg(4, m_buffer_dP);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize dP buffer." << std::endl;
	}
	errNum = m_kernel.setArg(5, m_buffer_dt);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize dt buffer." << std::endl;
	}
	errNum = m_kernel.setArg(6, m_bufferMassIn);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize mass_in buffer." << std::endl;
	}
	errNum = m_kernel.setArg(7, m_bufferMassOut);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize mass_out buffer." << std::endl;
	}
	errNum = m_kernel.setArg(8, m_bufferMassDelta2);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize Dm2 buffer." << std::endl;
	}
	errNum = m_kernel.setArg(9, m_bufferBubbleRadius);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize R buffer." << std::endl;
	}
	errNum = m_kernel.setArg(10, m_bufferBubbleRadius2);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize R2 buffer." << std::endl;
	}
	errNum = m_kernel.setArg(11, m_bufferBubbleRadiusSpeedDt2);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize R_dt2 buffer." << std::endl;
	}
	errNum = m_kernel.setArg(12, m_bufferBubbleSpeed);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize V buffer." << std::endl;
	}
	errNum = m_kernel.setArg(13, m_bufferBubbleGamma);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize Gamma buffer." << std::endl;
	}
	errNum = m_kernel.setArg(14, m_bufferBubbleGammaSpeed);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize Gamma_v buffer." << std::endl;
	}
	errNum = m_kernel.setArg(15, m_bufferInteractedFalse);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize InteractedFalse buffer." << std::endl;
	}
	errNum = m_kernel.setArg(16, m_bufferPassedFalse);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize PassedFalse buffer." << std::endl;
	}
	errNum = m_kernel.setArg(17, m_bufferInteractedTrue);
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
	std::cout << "Available platform: " << value.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    //try to find NVIDIA device
    for (const auto& value : platforms) {
	if (value.getInfo<CL_PLATFORM_NAME>().find("NVIDIA") != std::string::npos) {
            std::vector<cl::Device> temp_devices;
            value.getDevices(CL_DEVICE_TYPE_GPU, &temp_devices);
            for (const auto& value2 : temp_devices) {
                devices.push_back(value2);
            }
	    break;
        }
    }

    //try to find CPU device
    if (devices.size() == 0) {
    	for (const auto& value : platforms) {
    	    if (value.getInfo<CL_PLATFORM_NAME>().find("Portable Computing Language") != std::string::npos) {
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

void OpenCLWrapper::createProgram(cl::Context& context, cl::Device& device, std::string& kernelFile) {
    int errNum;

    std::ifstream kernel_file(kernelFile);
    std::string kernel_code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

    m_program = cl::Program(context, kernel_code, false, &errNum);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed to create program from source file. Cehck if kernel file location is correct. (" << kernelFile << ")" << std::endl;
        exit(1);
    }

    errNum = m_program.build("-cl-std=CL1.2");
    if (errNum != CL_BUILD_SUCCESS) {
        std::cerr << "Error!\nBuild Status: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
            << "\nBuild Log:\t " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
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

void OpenCLWrapper::makeStep1(numType& t_radius, numType& t_radius2, numType& t_speed, numType& t_gamma, numType& t_gammaSpeed, numType& radiusAfterDt2) {
	m_queue.enqueueWriteBuffer(m_bufferBubbleRadius, CL_TRUE, 0, sizeof(numType), &t_radius);
	m_queue.enqueueWriteBuffer(m_bufferBubbleRadius2, CL_TRUE, 0, sizeof(numType), &t_radius2);
	m_queue.enqueueWriteBuffer(m_bufferBubbleSpeed, CL_TRUE, 0, sizeof(numType), &t_speed);
	m_queue.enqueueWriteBuffer(m_bufferBubbleGamma, CL_TRUE, 0, sizeof(numType), &t_gamma);
	m_queue.enqueueWriteBuffer(m_bufferBubbleGammaSpeed, CL_TRUE, 0, sizeof(numType), &t_gammaSpeed);
	m_queue.enqueueWriteBuffer(m_bufferBubbleRadiusSpeedDt2, CL_TRUE, 0, sizeof(numType), &radiusAfterDt2);
}

void OpenCLWrapper::makeStep2(int& particleCount) {
	m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, cl::NDRange(particleCount));
}

void OpenCLWrapper::makeStep3(int& particleCount, std::vector<numType>& t_dP) {
	m_queue.enqueueReadBuffer(m_buffer_dP, CL_TRUE, 0, particleCount * sizeof(numType), t_dP.data());
}

void OpenCLWrapper::readBufferX(std::vector<numType>& t_dataX) {
	m_queue.enqueueReadBuffer(m_bufferX, CL_TRUE, 0, t_dataX.size()*sizeof(numType), t_dataX.data());
}

void OpenCLWrapper::readBufferP(std::vector<numType>& t_dataP) {
	m_queue.enqueueReadBuffer(m_bufferP, CL_TRUE, 0, t_dataP.size() * sizeof(numType), t_dataP.data());
}

void OpenCLWrapper::readBuffer_dP(std::vector<numType>& t_data_dP) {
	m_queue.enqueueReadBuffer(m_buffer_dP, CL_TRUE, 0, t_data_dP.size() * sizeof(numType), t_data_dP.data());
}

void OpenCLWrapper::readBufferM(std::vector<numType>& t_dataM) {
	m_queue.enqueueReadBuffer(m_bufferM, CL_TRUE, 0, t_dataM.size() * sizeof(numType), t_dataM.data());
}

void OpenCLWrapper::readBufferE(std::vector<numType>& t_dataE) {
	m_queue.enqueueReadBuffer(m_bufferE, CL_TRUE, 0, t_dataE.size() * sizeof(numType), t_dataE.data());
}

void OpenCLWrapper::readBufferInteractedFalse(std::vector<int8_t>& t_dataInteractedFalse) {
	m_queue.enqueueReadBuffer(m_bufferInteractedFalse, CL_TRUE, 0, t_dataInteractedFalse.size() * sizeof(int8_t), t_dataInteractedFalse.data());
}

void OpenCLWrapper::readBufferPassedFalse(std::vector<int8_t>& t_dataPassedFalse) {
	m_queue.enqueueReadBuffer(m_bufferPassedFalse, CL_TRUE, 0, t_dataPassedFalse.size() * sizeof(int8_t), t_dataPassedFalse.data());
}

void OpenCLWrapper::readBufferInteractedTrue(std::vector<int8_t>& t_dataInteractedTrue) {
	m_queue.enqueueReadBuffer(m_bufferInteractedTrue, CL_TRUE, 0, t_dataInteractedTrue.size() * sizeof(int8_t), t_dataInteractedTrue.data());
}

void OpenCLWrapper::readBufferR(numType& t_dataR) {
	m_queue.enqueueReadBuffer(m_bufferBubbleRadius, CL_TRUE, 0, sizeof(numType), &t_dataR);
}

void OpenCLWrapper::readBufferSpeed(numType& t_dataSpeed) {
	m_queue.enqueueReadBuffer(m_bufferBubbleSpeed, CL_TRUE, 0, sizeof(numType), &t_dataSpeed);
}

void OpenCLWrapper::readBufferBubble(numType& t_dataR, numType& t_dataSpeed) {
	readBufferR(t_dataR);
	readBufferSpeed(t_dataSpeed);
}
