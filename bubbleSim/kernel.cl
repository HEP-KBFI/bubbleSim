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
  char b_collide;
  char b_inBubble;
  int idxCollisionCell;
} Particle;

typedef struct Bubble {
  double radius;
  double radius2;           // Squared
  double radiusAfterStep2;  // (radius + speed * dt)^2
  double speed;
  double gamma;
  double gammaXspeed;  // gamma * speed
} Bubble;

void moveLinear(
	// X_new = X_old + V * dt
	Particle *particle, 
	double t_v1, double t_v2, double t_v3,
	double t_dt
	) {
	// x = x + v*dt
	particle->x = fma(t_v1, t_dt, particle->x);
	particle->y = fma(t_v2, t_dt, particle->y);
	particle->z = fma(t_v3, t_dt, particle->z);
	}
double calculateTimeToWall(
		Particle particle, Bubble bubble, double t_dt
	) {
	double time1, time2;
	
	//double a = fma(t_p1, t_p1/pow(t_E, 2), fma(t_p2, t_p2/pow(t_E, 2), fma(t_p3, t_p3/pow(t_E, 2), - t_vb * t_vb)));
	double a = fma(particle.p_x, particle.p_x,
					fma(particle.p_y, particle.p_y,
						fma(particle.p_z, particle.p_z, 0.)))/pow(particle.E, 2)
							- bubble.speed * bubble.speed;
	//double b = fma(t_x1, t_p1/t_E, fma(t_x2, t_p2/t_E, fma(t_x3, t_p3/t_E, - t_rb * t_vb)));
	double b = fma(particle.x, particle.p_x/particle.E,
					fma(particle.y, particle.p_y/particle.E,
						fma(particle.z, particle.p_z/particle.E, - bubble.radius * bubble.speed)));
	//double c = fma(t_x1, t_x1, fma(t_x2, t_x2, fma(t_x3, t_x3, - t_rb * t_rb)));
	double c = fma(particle.x, particle.x,
					fma(particle.y, particle.y,
						fma(particle.z, particle.z, - bubble.radius * bubble.radius)));
	double d = fma(b, b, - a * c);
	// b*b - a*c;
	// double d = fma(-a, c, pow(b,2)); 
	if (d < 0){
		return 0;
	}
	
	time1 = (-b - sqrt(d))/a;
	time2 = (-b + sqrt(d))/a;
	if ((0 < time1) && (time1 <= t_dt)){
		return time1;
	}
	else if ((0 < time2) && (time2 <= t_dt)){
		return time2;
	}
	else{
		return 0.;
	}
}

void calculateNormal(
	double *t_n1, double *t_n2, double *t_n3, 
	Particle particle, Bubble bubble, double t_X2, double t_mass_in, double t_mass_out
	){	
	// If M_in > M_out then normal is towards center of bubble
	if (t_mass_in > t_mass_out){
		*t_n1 = -particle.x * bubble.gamma/sqrt(t_X2);
		*t_n2 = -particle.y * bubble.gamma/sqrt(t_X2);
		*t_n3 = -particle.z * bubble.gamma/sqrt(t_X2);
		}
	// If M_in < M_out then normal is towards outside the bubble
	else {
		*t_n1 = particle.x * bubble.gamma/sqrt(t_X2);
		*t_n2 = particle.y * bubble.gamma/sqrt(t_X2);
		*t_n3 = particle.z * bubble.gamma/sqrt(t_X2);
		}
	}

double calculateRadius(Particle particle){
	return fma(particle.x, particle.x, fma(particle.y, particle.y, particle.z * particle.z));
}

double calculateEnergy(Particle particle, double mass){
	return sqrt(fma(particle.p_x, particle.p_x, 
					fma(particle.p_y, particle.p_y, 
						fma(particle.p_z, particle.p_z, pow(mass, 2)))));
}

__kernel void step_double(	
	__global Particle *t_particles,
	__global double *t_dP,
	__global char *t_interactedFalse,
	__global char *t_passedFalse,
	__global char *t_interactedTrue,
	__constant Bubble *t_bubble,
	__constant double *t_dt,
	__constant double *t_m_in,
	__constant double *t_m_out,
	__constant double *t_delta_m2
	){
	
	unsigned int gid = get_global_id(0);
	// dE - dP is not actual energy difference. dE = ΔE/R_b -> to avoid singularities/noise near R_b ~ 0
	
	// Bubble parameters	
	struct Bubble bubble = t_bubble[0];
	// Particle
	struct Particle particle = t_particles[gid];
	
	double M_in = t_m_in[0];
	double M_out = t_m_out[0];
	double Delta_M2 = t_delta_m2[0];
	
	// Particle parameters
	double dt = t_dt[0];

	double V_1 = particle.p_x/particle.E;
	double V_2 = particle.p_y/particle.E;
	double V_3 = particle.p_z/particle.E;
	
	// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
	// fma(a, b, c) = a * b + c
	double X2 = calculateRadius(particle);
	
	double X_dt_1 = fma(V_1, dt, particle.x);
	double X_dt_2 = fma(V_2, dt, particle.y);
	double X_dt_3 = fma(V_3, dt, particle.z);
	double X_dt2 = fma(X_dt_1, X_dt_1, fma( X_dt_2, X_dt_2, X_dt_3 * X_dt_3));
	
	double n_1, n_2, n_3;
	double timeToWall, np;
			
	// If M_in < M_out -> Check if particle starts inside
	// If M_in > M_out -> Check if particle starts outside
	if (((X2 < bubble.radius2) && (M_in < M_out)) || ((X2 > bubble.radius2) && (M_in > M_out))){
		// If M_in < M_out -> Check if particle stays in
		// If M_in > M_out -> Check if particle stays out
		if (((X_dt2 < bubble.radiusAfterStep2) && (M_in < M_out)) || ((X_dt2 > bubble.radiusAfterStep2) && (M_in > M_out))) {
			// X_1 = fma(V_1, dt, X_1);
			// X_2 = fma(V_2, dt, X_2);
			// X_3 = fma(V_3, dt, X_3);
			moveLinear(&particle, V_1, V_2, V_3, dt);
			t_dP[gid] = 0.;
			// t_interactedFalse[gid] += 0;
			// t_passedFalse[gid] += 0;
			// t_interactedTrue[gid] += 0;
		}
		// Maybe get outside
		else {
			timeToWall = calculateTimeToWall(particle, bubble, dt);
			// X_1 = fma(V_1, timeToWall, X_1);
			// X_2 = fma(V_2, timeToWall, X_2);
			// X_3 = fma(V_3, timeToWall, X_3);
			moveLinear(&particle, V_1, V_2, V_3, timeToWall);
			// Update X2
			// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
			X2 = calculateRadius(particle);
			// n_1 = X_1 * Gamma/sqrt(X2);
			// n_2 = X_2 * Gamma/sqrt(X2);
			// n_3 = X_3 * Gamma/sqrt(X2);
			
			// If M_in > M_out then normal is towards inside
			// If M_in < M_out then normal is towards outside
			calculateNormal(&n_1, &n_2, &n_3, particle, bubble, X2, M_in, M_out);
					
			np = bubble.speed * bubble.gamma * particle.E - n_1*particle.p_x - n_2*particle.p_y - n_3*particle.p_z;
			
			// ========== Interaction with the bubble ==========
			// Particle bounces from the wall and stays where it was -> Stays in lower mass region
			// Particle keeps lower mass
			if (np*np < Delta_M2){
				// P_i = P_i + 2 * np * n_i
				particle.p_x = fma(np*2., n_1, particle.p_x);
				particle.p_y = fma(np*2., n_2, particle.p_y);
				particle.p_z = fma(np*2., n_3, particle.p_z);
				t_dP[gid] = bubble.gamma * np * 2. ;
				
				// E_particle = sqrt(pow(M[gid], 2.) + P_1*P_1 + P_2*P_2 + P_3*P_3);
				
				particle.E = calculateEnergy(particle, particle.m);
				
				t_interactedFalse[gid] += 1;
				//t_passedFalse[gid] += 0;
				//t_interactedTrue[gid] += 0;
			}
			// Particle gets through the bubble wall -> Gets higher mass
			// Particle gets higher mass
			else {
				if (M_in < M_out){
					particle.m = M_out;
				}
				else {
					particle.m = M_in;
				}
				
				// P_i = P_i + np * n_i * sqrt(1-sqrt(1-Δm^2/np^2))
				particle.p_x = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_1, particle.p_x);
				particle.p_y = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_2, particle.p_y);
				particle.p_z = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_3, particle.p_z);
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.-Delta_M2/pow(np, 2.)));
				
				// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
				
				particle.E = calculateEnergy(particle, particle.m);

				t_interactedFalse[gid] += 1;
				t_passedFalse[gid] += 1;
				//t_interactedTrue[gid] += 0;
				
			}
			// Update velocity vector
			V_1 = particle.p_x / particle.E;
			V_2 = particle.p_y / particle.E;
			V_3 = particle.p_z / particle.E;
			// ========== Movement after the bubble interaction ==========
			//X_1 = fma(V_1, dt - timeToWall, X_1);
			//X_2 = fma(V_2, dt - timeToWall, X_2);
			//X_3 = fma(V_3, dt - timeToWall, X_3);
			moveLinear(&particle, V_1, V_2, V_3, dt - timeToWall);
		}
	}
	
	// If M_in < M_out -> Then particle starts outside
	// If M_in > M_out -> Then particle starts inside
	else {
		// If M_in < M_out -> Particle stays outside
		// If M_in > M_out -> Particle stays inside
		if (((X_dt2 > bubble.radiusAfterStep2) && (M_in < M_out)) || ((X_dt2 < bubble.radiusAfterStep2) && (M_in > M_out))) {
			// X_1 = fma(V_1, dt, X_1);
			// X_2 = fma(V_2, dt, X_2);
			// X_3 = fma(V_3, dt, X_3);
			moveLinear(&particle, V_1, V_2, V_3, dt);
			t_dP[gid] = 0.;
			t_interactedFalse[gid] += 0;
			t_passedFalse[gid] += 0;
			t_interactedTrue[gid] += 0;
		}
		// Particle gets lower mass
		// If M_in < M_out -> Particle goes inside from out of the bubble
		// If M_in > M_out -> Particle goes outside from the bubble to inside
		else {
			timeToWall = calculateTimeToWall(particle, bubble, dt);
			// X_1 = fma(V_1, timeToWall, X_1);
			// X_2 = fma(V_2, timeToWall, X_2);
			// X_3 = fma(V_3, timeToWall, X_3);
			moveLinear(&particle, V_1, V_2, V_3, timeToWall);
			// Update X2
			// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
			X2 = calculateRadius(particle);
			// n_1 = X_1 * Gamma/sqrt(X2);
			// n_2 = X_2 * Gamma/sqrt(X2);
			// n_3 = X_3 * Gamma/sqrt(X2);
			calculateNormal(&n_1, &n_2, &n_3, particle, bubble, X2, M_in, M_out);
			
			np = bubble.speed * bubble.gamma*particle.E - n_1*particle.p_x - n_2*particle.p_y - n_3*particle.p_z;
			
			// P_i = P_i + np * n_i * sqrt(1-sqrt(1+Δm^2/np^2))
			particle.p_x = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_1, particle.p_x);
			particle.p_y = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_2, particle.p_y);
			particle.p_z = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_3, particle.p_z);
			t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.+Delta_M2/pow(np, 2.)));
			
			if (M_in < M_out) {
				particle.m = M_in;
				}
			else {
				particle.m = M_out;
				}
			
			// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
			particle.E = calculateEnergy(particle, particle.m);
			
			
			// Update velocity vector
			V_1 = particle.p_x / particle.E;
			V_2 = particle.p_y / particle.E;
			V_3 = particle.p_z / particle.E;
			// Movement after the bubble interaction
			//X_1 = fma(V_1, dt - timeToWall, X_1);
			//X_2 = fma(V_2, dt - timeToWall, X_2);
			//X_3 = fma(V_3, dt - timeToWall, X_3);
			moveLinear(&particle, V_1, V_2, V_3, dt - timeToWall);
			t_interactedFalse[gid] += 0;
			t_passedFalse[gid] += 0;
			t_interactedTrue[gid] += 1;
		}
	}
		
	t_particles[gid].x = particle.x;
	t_particles[gid].y = particle.y;
	t_particles[gid].z = particle.z;
	
	t_particles[gid].p_x = particle.p_x;
	t_particles[gid].p_y = particle.p_y;
	t_particles[gid].p_z = particle.p_z;
	
	t_particles[gid].E = particle.E;
	t_particles[gid].m = particle.m;
}
