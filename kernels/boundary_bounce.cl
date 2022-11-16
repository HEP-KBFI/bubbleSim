#pragma OPENCL EXTENSION cl_khr_fp64 : enable


typedef struct Particle {
  double x;
  double y;
  double z;

  double p_x;
  double p_y;
  double p_z;

  double E;
  double m;
} Particle;


__kernel void bounce_from_boundary(
	Particle *t_particles,
	double *boundaries, // [x_delta, y_delta, z_delta]
	double *dt
	){
	unsigned int = get_global_id(0);
	
	Particle particle = t_particles[gid];
	double timeToBoundary;
	double dt_value = dt[0];
	
	double v_x = particle.p_x/particle.E;
	double v_y = particle.p_y/particle.E;
	double v_z = particle.p_z/particle.E;

	// Check if bounces first time
	if ((particle.x < -boundaries[0]) || (particle.x > boundaries[0])){
		timeToBoundary = (abs(particle.x) - boundaries[0])/abs(v_x);
		particle.p_x = -particle.p_x;
	}
	else if ((particle.y < -boundaries[1]) || (particle.y > boundaries[1])){
		timeToBoundary = (abs(particle.y) - boundaries[1])/abs(v_y);
		particle.p_y = -particle.p_y;
	}
	else if ((particle.z < -boundaries[2]) || (particle.z > boundaries[2])){
		timeToBoundary = (abs(particle.z) - boundaries[2])/abs(v_z);
		particle.p_z = -particle.p_z;
	}
	// It's rewind time
	particle.x = particle.x - v_x * timeToBoundary;
	particle.y = particle.y - v_y * timeToBoundary;
	particle.z = particle.z - v_z * timeToBoundary;
	// Update velocity
	v_x = particle.p_x/particle.E;
	v_y = particle.p_y/particle.E;
	v_z = particle.p_z/particle.E;
	// Move particles forward
	dt_value = dt_value - timeToBoundary;
	particle.x = particle.x + v_x * dt_value;
	particle.y = particle.y + v_y * dt_value;
	particle.z = particle.z + v_z * dt_value;
	
	// Check for second bounce
	timeToBoundary = 0.;
	
	if ((particle.x < -boundaries[0]) || (particle.x > boundaries[0])){
		timeToBoundary = (abs(particle.x) - boundaries[0])/abs(v_x);
		particle.p_x = -particle.p_x;
	}
	else if ((particle.y < -boundaries[1]) || (particle.y > boundaries[1])){
		timeToBoundary = (abs(particle.y) - boundaries[1])/abs(v_y);
		particle.p_y = -particle.p_y;
	}
	else if ((particle.z < -boundaries[2]) || (particle.z > boundaries[2])){
		timeToBoundary = (abs(particle.z) - boundaries[2])/abs(v_z);
		particle.p_z = -particle.p_z;
	}
	// It's rewind time
	particle.x = particle.x - v_x * timeToBoundary;
	particle.y = particle.y - v_y * timeToBoundary;
	particle.z = particle.z - v_z * timeToBoundary;
	// Update velocity
	v_x = particle.p_x/particle.E;
	v_y = particle.p_y/particle.E;
	v_z = particle.p_z/particle.E;
	// Move particles forward
	dt_value = dt_value - timeToBoundary;
	particle.x = particle.x + v_x * dt_value;
	particle.y = particle.y + v_y * dt_value;
	particle.z = particle.z + v_z * dt_value;
	
	// Update result
	t_particles[gid] = particle;
}