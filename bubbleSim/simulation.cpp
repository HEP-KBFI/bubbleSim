#include "simulation.h"

Bubble::Bubble(MainType t_radius, MainType t_speed, MainType t_dVT, MainType t_dV, MainType t_sigma) {
	m_radius = t_radius;
	m_speed = t_speed;
	m_gamma = 1 / sqrt(1 - m_speed * m_speed);
	m_dVT = t_dVT;
	m_dV = t_dV;
	m_sigma = t_sigma;
	m_gammaSpeed = m_speed * m_gamma;
	m_radius2 = m_radius * m_radius;
	m_area = 4 * M_PI * m_radius2;
	m_volume = 4 * M_PI * m_radius * m_radius * m_radius / 3;
	m_energy = calculateEnergy();
}

void Bubble::evolveWall(MainType t_dt, MainType t_dP) {
	MainType new_radius = m_radius + t_dt * m_speed;

	MainType velocity_elem = 1 - m_speed * m_speed;

	m_speed += velocity_elem * sqrt(velocity_elem) * (-m_dV * t_dt + t_dP) / m_sigma - 2 * velocity_elem * t_dt / m_radius;
	m_radius = new_radius;
	
	m_radius2 = new_radius * new_radius;
	m_area = 4 * M_PI * m_radius2;
	m_volume = 4 * M_PI * m_radius * m_radius * m_radius / 3;

	m_gamma = 1 / sqrt(1 - m_speed * m_speed);
	m_gammaSpeed = m_speed * m_gamma;

	m_energy = calculateEnergy();
}

MainType Bubble::calculateEnergy() {
	m_energy = m_area * m_sigma / sqrt(1 - m_speed * m_speed) + 4 * (MainType)M_PI / 3 * m_radius * m_radius * m_radius * m_dV;
	return m_energy;
}

MainType Bubble::calculateRadiusSpeedDt2(MainType dt) {
	MainType temp = m_radius + m_speed * dt;
	m_radiusSpeedDt2 = temp * temp;
	return m_radiusSpeedDt2;
}

Simulation::Simulation(
	MainType t_alpha, MainType t_massIn, MainType t_massOut, MainType t_temperatureIn, MainType t_temperatureOut,
	unsigned int t_particleCountIn, unsigned int t_particleCountOut, MainType t_coupling
) {
	m_alpha = t_alpha;
	m_coupling = t_coupling;

	m_massIn = t_massIn;
	m_massOut = t_massOut;
	m_massDelta2 = abs(m_massIn * m_massIn - m_massOut * m_massOut);
	m_temperatureIn = t_temperatureIn;
	m_temperatureOut = t_temperatureOut;

	m_particleCount = t_particleCountIn + t_particleCountOut;
	m_particleCountIn = t_particleCountIn;
	m_particleCountOut = t_particleCountOut;
	
	m_energyParticlesInitial = 0.;
	m_energyParticles = 0.;
	m_energyBubble = 0.;
	m_energyBubbleInitial = 0.;
	m_energyTotal = 0.;
	m_energyTotalInitial = 0.;

	m_time = (MainType) 0.;

	MainType dpIn = (MainType)1e-4 * m_temperatureIn;
	MainType dpOut = (MainType)1e-4 * m_temperatureOut;
	MainType upperLimitIn = (MainType)30 * m_temperatureIn;
	MainType upperLimitOut = (MainType)30 * m_temperatureOut;
	const int sizeElementsIn = (int)(upperLimitIn / dpIn) + 1;
	const int sizeElementsOut = (int)(upperLimitOut / dpOut) + 1;
	std::cout << "Calculating distribution(s) and statistical values for the simulation..." << std::endl;
	// Calculate distribution values from temperature
	if ((m_temperatureIn != 0) && (m_particleCountIn > 0)) {
		m_nIn = calculateNumberDensity(m_massIn, m_temperatureIn, dpIn, upperLimitIn);
		m_rhoIn = calculateEnergyDensity(m_massIn, m_temperatureIn, dpIn, upperLimitIn);
		calculateCPD(m_massIn, m_temperatureIn, sizeElementsIn, m_cpdIn, m_pValuesIn, dpIn, upperLimitIn);
		std::cout << "\t\"Inside\" values calculated." << std::endl;
	}
	else {
		std::cout << "\t\"Inside\" values not calculated." << std::endl;
	}
	if ((m_temperatureOut != 0) && (m_particleCountOut > 0)) {
		m_nOut = calculateNumberDensity(m_massOut, m_temperatureOut, dpOut, upperLimitOut);
		m_rhoOut = calculateEnergyDensity(m_massOut, m_temperatureOut, dpOut, upperLimitOut);
		calculateCPD(m_massOut, m_temperatureOut, sizeElementsOut, m_cpdOut, m_pValuesOut, dpOut, upperLimitOut);
		std::cout << "\t\"Outside\" values calculated." << std::endl;
	}
	else {
		std::cout << "\t\"Outside\" values not calculated." << std::endl;
	}

	// Reserve memory
	m_X.reserve(m_particleCount);
	m_P.reserve(m_particleCount);
	m_E.reserve(m_particleCount);
	m_M.reserve(m_particleCount);
	m_dP = std::vector<MainType>(m_particleCount, (MainType)0.);
	m_InteractedFalse = std::vector<int8_t>(m_particleCount, 0);
	m_PassedFalse = std::vector<int8_t>(m_particleCount, 0);
	m_InteractedTrue = std::vector<int8_t>(m_particleCount, 0);
	m_PassedTrue = std::vector<int8_t>(m_particleCount, 0);
}

MainType Simulation::countParticlesEnergy() {
	m_energyParticles = 0;
	for (unsigned int i = 0; i < m_particleCount; i++) {
		m_energyParticles += m_E[i];
	}
	return m_energyParticles;
}

MainType Simulation::countParticlesEnergyIn(Bubble& bubble) {
	MainType energyParticles_in = 0;
	for (unsigned int i = 0; i < m_particleCount; i++) {
		if (getParticleRadius(i) < bubble.m_radius2) {
			energyParticles_in += m_E[i];
		}
	}
	return energyParticles_in;
}

MainType Simulation::calculateTotalEnergy(Bubble& bubble) {
	m_energyParticles = countParticlesEnergy();
	m_energyBubble = bubble.calculateEnergy();
	m_energyTotal = m_energyParticles + m_energyBubble;
	return m_energyTotal;
}

MainType Simulation::calculateTotalEnergyInitial(Bubble& bubble) {
	m_energyParticlesInitial = countParticlesEnergy();
	m_energyParticles = m_energyParticlesInitial;
	m_energyBubbleInitial = bubble.calculateEnergy();
	m_energyBubble = m_energyBubbleInitial;
	m_energyTotalInitial = m_energyParticlesInitial + m_energyBubbleInitial;
	m_energyTotal = m_energyTotalInitial;
	
	return m_energyTotalInitial;
}

MainType Simulation::countParticleEnergyDensity(u_int t_startParticleIndex, u_int t_endParticleIndex, MainType t_volume) {
	MainType totalEnergy = (MainType) 0.;
	for (u_int i = t_startParticleIndex; i < t_endParticleIndex; i++) {
		totalEnergy += m_E[i];
	}
	return totalEnergy / t_volume;
}

MainType Simulation::countParticleNumberDensity(u_int t_particleCount, MainType t_volume) {
	return (MainType) t_particleCount / t_volume;
}

MainType Simulation::getParticleRadius(u_int& i) {
	if (i >= m_particleCount) {
		throw std::invalid_argument("Index i is equal/bigger than number of particles.");
	}
	return sqrt(m_X[3 * i] * m_X[3 * i] + m_X[3 * i + 1] * m_X[3 * i + 1] + m_X[3 * i + 2] * m_X[3 * i + 2]);
}

MainType Simulation::getParticleMomentum(u_int& i) {
	if (i >= m_particleCount) {
		throw std::invalid_argument("Index i is equal/bigger than number of particles.");
	}
	return sqrt(m_P[3 * i] * m_P[3 * i] + m_P[3 * i + 1] * m_P[3 * i + 1] + m_P[3 * i + 2] * m_P[3 * i + 2]);
}

MainType Simulation::getParticleEnergy1(u_int& i) {
	// Get energy from already calculated data (from buffer)
	if (i >= m_particleCount) {
		throw std::invalid_argument("Index i is equal/bigger than number of particles.");
	}
	return m_E[i];
}

MainType Simulation::getParticleEnergy2(u_int& i) {
	// Gett energy by calculating it from energy - momentum relation.
	if (i >= m_particleCount) {
		throw std::invalid_argument("Index i is equal/bigger than number of particles.");
	}
	return sqrt(m_P[3 * i] * m_P[3 * i] + m_P[3 * i + 1] * m_P[3 * i + 1] + m_P[3 * i + 2] * m_P[3 * i + 2] + m_M[i] * m_M[i]);
}

MainType Simulation::calculateNumberDensity(MainType& t_mass, MainType& t_temperature,  MainType& t_dp, MainType& t_pUpperLimit) {
	MainType n = 0;
	MainType p = 0;
	MainType m2 = t_mass * t_mass;

	for (; p <= t_pUpperLimit; p += t_dp) {
		n += t_dp * p * p * std::exp(-sqrt(p * p + m2) / t_temperature);
	}
	n = n / ((MainType)2 * (MainType)M_PI * (MainType)M_PI);
	return n;
}

MainType Simulation::calculateEnergyDensity(MainType& t_mass, MainType& t_temperature, MainType& t_dp, MainType& t_pUpperLimit) {
	MainType density = 0;
	MainType p = 0;
	MainType m2 = t_mass * t_mass;

	for (; p <= t_pUpperLimit; p += t_dp) {
		density += t_dp * p * p * sqrt(p * p + m2) * std::exp(-sqrt(p * p + m2) / t_temperature);
	}
	density = density / ((MainType)2 * (MainType)M_PI * (MainType)M_PI);
	return density;
}

void Simulation::calculateCPD(MainType& t_mass, MainType& t_temperature, const int& t_sizeVector, std::vector<MainType>& v_cpd, std::vector<MainType>& v_p, MainType& t_dp, MainType& t_pUpperLimit) {
	v_p.reserve(t_sizeVector);
	v_cpd.reserve(t_sizeVector);

	// Precalculate squares as this makes couple of calculations less
	MainType m2 = t_mass * t_mass;

	MainType last_cpd= 0;
	MainType last_p = 0;

	v_p.push_back(0);
	v_cpd.push_back(0);

	for (int i = 1; i < t_sizeVector; i++) {
		v_p.push_back(last_p + t_dp);
		v_cpd.push_back(last_cpd + t_dp * last_p * last_p * std::exp(-sqrt(last_p * last_p + m2) / t_temperature)); 
		last_cpd = v_cpd[i];
		last_p = v_p[i];
	}

	for (int i = 0; i < t_sizeVector; i++) {
		v_cpd[i] = v_cpd[i] / last_cpd;
	}
}

MainType Simulation::interp(MainType& t_value, std::vector<MainType>& t_x, std::vector<MainType>& t_y) {
	if (t_value < 0) {
		return t_y[0];
	}
	else if (t_value > t_x.back()) {
		return t_y.back();
	}
	else {
		unsigned int k1 = 0;
		unsigned int k2 = static_cast<unsigned int>(t_x.size() - 1);
		unsigned int k = (k2 + k1) / 2;
		for (; k2 - k1 > 1; ) {
			if (t_x[k] > t_value) {
				k2 = k;
			}
			else {
				k1 = k;
			}
			k = (k2 + k1) / 2;
		}
		return t_y[k1] + (t_value - t_x[k1]) * (t_y[k2] - t_y[k1]) / (t_x[k2] - t_x[k1]);
	}
}

std::array<MainType, 3> Simulation::generateRandomDirectionArray(MainType& t_magnitude, std::uniform_real_distribution<MainType>& t_uniformDistribution, std::mt19937_64& t_generator) {
	MainType phi = std::acos(1 - 2 * t_uniformDistribution(t_generator)); // inclination
	MainType theta = (MainType)2 * (MainType)M_PI * t_uniformDistribution(t_generator); // asimuth
	return std::array<MainType, 3> {
		t_magnitude * std::sin(phi)* std::cos(theta), // x
		t_magnitude * std::sin(phi)* std::sin(theta), // y
		t_magnitude * std::cos(phi) // z
	};
}

void Simulation::generateRandomDirection(MainType& t_magnitude, std::vector < MainType>& t_vector, std::uniform_real_distribution<MainType>& t_uniformDistribution, std::mt19937_64& t_generator) {
	MainType phi = std::acos(1 - 2 * t_uniformDistribution(t_generator)); // inclination
	MainType theta = (MainType)2 * (MainType)M_PI * t_uniformDistribution(t_generator); // asimuth 
	t_vector.push_back(t_magnitude * std::sin(phi) * std::cos(theta)); // x
	t_vector.push_back(t_magnitude * std::sin(phi) * std::sin(theta)); // y
	t_vector.push_back(t_magnitude * std::cos(phi)); // z
}

std::array<MainType, 3> Simulation::generatePointInBoxArray(MainType& t_lengthX, MainType& t_lengthY, MainType& t_lengthZ, std::uniform_real_distribution<MainType>& t_uniformDistribution, std::mt19937_64& t_generator) {
	MainType x = t_lengthX - (2 * t_lengthX * t_uniformDistribution(t_generator));
	MainType y = t_lengthY - (2 * t_lengthY * t_uniformDistribution(t_generator));
	MainType z = t_lengthZ - (2 * t_lengthZ * t_uniformDistribution(t_generator));
	return std::array<MainType, 3> {x, y, z};
}

void Simulation::generatePointInBox(MainType& t_lengthX, MainType& t_lengthY, MainType& t_lengthZ, std::vector <MainType>& t_vector, std::uniform_real_distribution<MainType>& t_uniformDistribution, std::mt19937_64& t_generator) {
	t_vector.push_back(t_lengthX - (2 * t_lengthX * t_uniformDistribution(t_generator)));
	t_vector.push_back(t_lengthY - (2 * t_lengthY * t_uniformDistribution(t_generator)));
	t_vector.push_back(t_lengthZ - (2 * t_lengthZ * t_uniformDistribution(t_generator)));
}

void Simulation::generateParticleMomentum(u_int t_particleCount, MainType t_mass, std::vector<MainType>& t_cpd, std::vector<MainType>& t_pValues) {
	std::random_device rand_dev;
	std::mt19937_64 generator(rand_dev());
	std::uniform_real_distribution<MainType> distribution(0, 1);
	MainType particleEnergy;

	MainType mass2 = t_mass * t_mass;

	for (u_int i = 0; i < t_particleCount; i++) {
		MainType generated_prob = distribution(generator);
		MainType p_value = interp(generated_prob, t_cpd, t_pValues);
		generateRandomDirection(p_value, m_P, distribution, generator);
		particleEnergy = sqrt(mass2 + p_value * p_value);
		m_E.push_back(particleEnergy);
		m_M.push_back(t_mass);
		m_energyParticlesInitial += particleEnergy;
	}
	// Clear cpd memory... Actually not required usually
	t_cpd.clear();
	t_pValues.clear();
	t_cpd.shrink_to_fit();
	t_pValues.shrink_to_fit();
}

void Simulation::generateParticleCoordinateCube(u_int t_particleCount, MainType t_radiusSphere, MainType t_radiusCube) {
	std::random_device rand_dev;
	std::mt19937_64 generator(rand_dev());
	std::uniform_real_distribution<MainType> distribution(0, 1);
	std::array<MainType, 3> coordinate_vector;
	if (t_radiusSphere == 0.0) {
		for (u_int i = 0; i < t_particleCount; i++) {
			generatePointInBox(t_radiusCube, t_radiusCube, t_radiusCube, m_X, distribution, generator);
		}
	}
	else{
		for (u_int i = 0; i < t_particleCount; i++) {
			coordinate_vector = generatePointInBoxArray(t_radiusCube, t_radiusCube, t_radiusCube, distribution, generator);
			while (sqrt(coordinate_vector[0] * coordinate_vector[0] + coordinate_vector[1] * coordinate_vector[1] + coordinate_vector[2] * coordinate_vector[2]) <= t_radiusSphere) {
				coordinate_vector = generatePointInBoxArray(t_radiusCube, t_radiusCube, t_radiusCube, distribution, generator);
			}
			m_X.push_back(coordinate_vector[0]);
			m_X.push_back(coordinate_vector[1]);
			m_X.push_back(coordinate_vector[2]);
		}
	}
}

void Simulation::generateParticleCoordinateSphere(u_int t_particleCount, MainType t_radiusSphere1, MainType t_radiusSphere2) {
	std::random_device rand_dev;
	std::mt19937_64 generator(rand_dev());
	std::uniform_real_distribution<MainType> distribution((MainType)0, (MainType)1);

	MainType r;
	std::array<MainType, 3> coordinate_vector;
	if (t_radiusSphere1 >= t_radiusSphere2) {
		throw std::invalid_argument("Invalid values. t_radiusSphere2 must be bigger then t_radiusSphere1.");
	}

	for (u_int i = 0; i < t_particleCount; i++) {
		coordinate_vector = generatePointInBoxArray(t_radiusSphere2, t_radiusSphere2, t_radiusSphere2, distribution, generator);
		r = sqrt(coordinate_vector[0] * coordinate_vector[0] + coordinate_vector[1] * coordinate_vector[1] + coordinate_vector[2] * coordinate_vector[2]);
		while ((r < t_radiusSphere1) || (r > t_radiusSphere2)) {
			coordinate_vector = generatePointInBoxArray(t_radiusSphere2, t_radiusSphere2, t_radiusSphere2, distribution, generator);
			r = sqrt(coordinate_vector[0] * coordinate_vector[0] + coordinate_vector[1] * coordinate_vector[1] + coordinate_vector[2] * coordinate_vector[2]);
		}

		m_X.push_back(coordinate_vector[0]);
		m_X.push_back(coordinate_vector[1]);
		m_X.push_back(coordinate_vector[2]);		
	}
};

MultiprocessingCL::MultiprocessingCL(std::string kernelFile, std::string kernelName, Simulation& sim, Bubble& bubble) {
	int errNum;
	u_int N = sim.m_particleCount;

	m_platform.getDevices(CL_DEVICE_TYPE_GPU, &m_devices);
	m_context = CreateContext(m_devices);
	m_program = CreateProgram(m_context, m_devices[0], kernelFile);

	if (std::is_same<double, MainType>::value) {
		m_kernel = CreateKernel(m_program, "step_double"); // cl::Kernel 
		std::cout << std::endl << "NB! Using 'step_double' kernel." << std::endl << std::endl;
	}
	else if (std::is_same<float, MainType>::value) {
		m_kernel = CreateKernel(m_program, "step_float"); // cl::Kernel 
		std::cout << std::endl << "NB! Using 'step_float' kernel." << std::endl << std::endl;

	}
	else {
		std::cout << "Could not find correcty type. Exiting." << std::endl;
		exit(1);
	}
	m_queue = CreateQueue(m_context, m_devices[0]);

	m_bufferX = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 3 * N * sizeof(MainType), sim.m_X.data(), &errNum);
	m_bufferP = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 3 * N * sizeof(MainType), sim.m_P.data(), &errNum);
	m_bufferE = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(MainType), sim.m_E.data(), &errNum);
	m_bufferM = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * sizeof(MainType), sim.m_M.data(), &errNum);
	m_bufferDP = cl::Buffer(m_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(MainType), sim.m_dP.data(), &errNum);

	// Scalar value buffers:
	m_bufferDt = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &sim.m_dt, &errNum);
	m_bufferMassIn = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &sim.m_massIn, &errNum);
	m_bufferMassOut = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &sim.m_massOut, &errNum);
	m_bufferMassDelta2 = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &sim.m_massDelta2, &errNum);

	// Bubble parameters
	m_bufferBubbleRadius = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &bubble.m_radius, &errNum);
	m_bufferBubbleRadius2 = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &bubble.m_radius2, &errNum);
	m_bufferBubbleRadiusSpeedDt2 = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &bubble.m_radiusSpeedDt2, &errNum);
	m_bufferBubbleSpeed = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &bubble.m_speed, &errNum);
	m_bufferBubbleGamma = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &bubble.m_gamma, &errNum);
	m_bufferBubbleGammaSpeed = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(MainType), &bubble.m_gammaSpeed, &errNum);

	m_bufferInteractedFalse = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(int8_t), sim.m_InteractedFalse.data(), &errNum);
	m_bufferPassedFalse = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(int8_t), sim.m_PassedFalse.data(), &errNum);
	m_bufferInteractedTrue = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(int8_t), sim.m_InteractedTrue.data(), &errNum);
	m_bufferPassedTrue = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof(int8_t), sim.m_PassedTrue.data(), &errNum);

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
	errNum = m_kernel.setArg(4, m_bufferDP);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize dP buffer." << std::endl;
	}
	errNum = m_kernel.setArg(5, m_bufferDt);
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
	errNum = m_kernel.setArg(18, m_bufferPassedTrue);
	if (errNum != CL_SUCCESS) {
		std::cout << "Couldn't initialize PassedTrue buffer." << std::endl;
	}

}

void MultiprocessingCL::runStep(u_int particleCount) {
	m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, cl::NDRange(particleCount));
	// Read in results: We need to know only energy change to evolve bubble
}
void MultiprocessingCL::readStep(u_int particleCount, std::vector<MainType>& dP) {
	m_queue.enqueueReadBuffer(m_bufferDP, CL_TRUE, 0, particleCount * sizeof(MainType), dP.data());
}
void MultiprocessingCL::writeStep(Bubble& bubble) {
	m_queue.enqueueWriteBuffer(m_bufferBubbleRadius, CL_TRUE, 0, sizeof(MainType), &bubble.m_radius);
	m_queue.enqueueWriteBuffer(m_bufferBubbleRadius2, CL_TRUE, 0, sizeof(MainType), &bubble.m_radius2);
	m_queue.enqueueWriteBuffer(m_bufferBubbleSpeed, CL_TRUE, 0, sizeof(MainType), &bubble.m_speed);
	m_queue.enqueueWriteBuffer(m_bufferBubbleGamma, CL_TRUE, 0, sizeof(MainType), &bubble.m_gamma);
	m_queue.enqueueWriteBuffer(m_bufferBubbleGammaSpeed, CL_TRUE, 0, sizeof(MainType), &bubble.m_gammaSpeed);
}
void MultiprocessingCL::writeBubbleRadiusSpeedDt2Buffer(MainType& dt, Bubble& bubble) {
	bubble.calculateRadiusSpeedDt2(dt);
	m_queue.enqueueWriteBuffer(m_bufferBubbleRadiusSpeedDt2, CL_TRUE, 0, sizeof(MainType), &bubble.m_radiusSpeedDt2);
}

