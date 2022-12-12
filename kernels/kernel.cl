#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct Particle {
  double x;
  double y;
  double z;

  double pX;
  double pY;
  double pZ;

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

typedef struct CollisionCell {
  double vX;
  double vY;
  double vZ;
  
  double x;
  double y;
  double z;
  double theta;
  
  double p_E;
  double p_x;
  double p_y;
  double p_z;
  
  double v2; // v2 = Sum: v_i^2 
  double gamma;
  double mass;  
  unsigned int particle_count;
} CollisionCell;

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
	double a = fma(particle.pX, particle.pX,
					fma(particle.pY, particle.pY,
						fma(particle.pZ, particle.pZ, 0.)))/pow(particle.E, 2)
							- bubble.speed * bubble.speed;
	//double b = fma(t_x1, t_p1/t_E, fma(t_x2, t_p2/t_E, fma(t_x3, t_p3/t_E, - t_rb * t_vb)));
	double b = fma(particle.x, particle.pX/particle.E,
					fma(particle.y, particle.pY/particle.E,
						fma(particle.z, particle.pZ/particle.E, - bubble.radius * bubble.speed)));
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
	return sqrt(fma(particle.pX, particle.pX, 
					fma(particle.pY, particle.pY, 
						fma(particle.pZ, particle.pZ, pow(mass, 2)))));
}

__kernel void particle_bubble_step(	
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

	double V_1 = particle.pX/particle.E;
	double V_2 = particle.pY/particle.E;
	double V_3 = particle.pZ/particle.E;
	
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
					
			np = bubble.speed * bubble.gamma * particle.E - n_1*particle.pX - n_2*particle.pY - n_3*particle.pZ;
			
			// ========== Interaction with the bubble ==========
			// Particle bounces from the wall and stays where it was -> Stays in lower mass region
			// Particle keeps lower mass
			if (np*np < Delta_M2){
				// P_i = P_i + 2 * np * n_i
				particle.pX = fma(np*2., n_1, particle.pX);
				particle.pY = fma(np*2., n_2, particle.pY);
				particle.pZ = fma(np*2., n_3, particle.pZ);
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
				particle.pX = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_1, particle.pX);
				particle.pY = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_2, particle.pY);
				particle.pZ = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_3, particle.pZ);
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.-Delta_M2/pow(np, 2.)));
				
				// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
				
				particle.E = calculateEnergy(particle, particle.m);

				t_interactedFalse[gid] += 1;
				t_passedFalse[gid] += 1;
				//t_interactedTrue[gid] += 0;
				
			}
			// Update velocity vector
			V_1 = particle.pX / particle.E;
			V_2 = particle.pY / particle.E;
			V_3 = particle.pZ / particle.E;
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
			// t_interactedFalse[gid] += 0;
			// t_passedFalse[gid] += 0;
			// t_interactedTrue[gid] += 0;
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
			
			np = bubble.speed * bubble.gamma*particle.E - n_1*particle.pX - n_2*particle.pY - n_3*particle.pZ;
			
			// P_i = P_i + np * n_i * sqrt(1-sqrt(1+Δm^2/np^2))
			particle.pX = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_1, particle.pX);
			particle.pY = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_2, particle.pY);
			particle.pZ = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_3, particle.pZ);
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
			V_1 = particle.pX / particle.E;
			V_2 = particle.pY / particle.E;
			V_3 = particle.pZ / particle.E;
			// Movement after the bubble interaction
			//X_1 = fma(V_1, dt - timeToWall, X_1);
			//X_2 = fma(V_2, dt - timeToWall, X_2);
			//X_3 = fma(V_3, dt - timeToWall, X_3);
			moveLinear(&particle, V_1, V_2, V_3, dt - timeToWall);
			
			//t_interactedFalse[gid] += 0;
			//t_passedFalse[gid] += 0;
			t_interactedTrue[gid] += 1;
		}
	}
		
	t_particles[gid].x = particle.x;
	t_particles[gid].y = particle.y;
	t_particles[gid].z = particle.z;
	
	t_particles[gid].pX = particle.pX;
	t_particles[gid].pY = particle.pY;
	t_particles[gid].pZ = particle.pZ;
	t_particles[gid].E = particle.E;
	t_particles[gid].m = particle.m;
	
}

__kernel void particle_bubble_step_cyclic(	
	__global Particle *t_particles,	
	__global double *t_dP,
	__global char *t_interactedFalse,
	__global char *t_passedFalse,
	__global char *t_interactedTrue,
	__constant Bubble *t_bubble,
	__constant double *t_dt,
	__constant double *t_m_in,
	__constant double *t_m_out,
	__constant double *t_delta_m2,
	__constant double *t_cyclicRadius
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

	double V_1 = particle.pX/particle.E;
	double V_2 = particle.pY/particle.E;
	double V_3 = particle.pZ/particle.E;
	
	// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
	// fma(a, b, c) = a * b + c
	double X2 = calculateRadius(particle);
	
	double X_dt_1 = fma(V_1, dt, particle.x);
	double X_dt_2 = fma(V_2, dt, particle.y);
	double X_dt_3 = fma(V_3, dt, particle.z);
	double X_dt2 = fma(X_dt_1, X_dt_1, fma( X_dt_2, X_dt_2, X_dt_3 * X_dt_3));
	
	double n_1, n_2, n_3;
	double timeToWall, np;
			
	// Check if particle in false vacuum
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
					
			np = bubble.speed * bubble.gamma * particle.E - n_1*particle.pX - n_2*particle.pY - n_3*particle.pZ;
			
			// ========== Interaction with the bubble ==========
			// Particle bounces from the wall and stays where it was -> Stays in lower mass region
			// Particle keeps lower mass
			if ((0 < -np) && (-np < sqrt(Delta_M2))){
				// P_i = P_i + 2 * np * n_i
				particle.pX = fma(np*2., n_1, particle.pX);
				particle.pY = fma(np*2., n_2, particle.pY);
				particle.pZ = fma(np*2., n_3, particle.pZ);
				t_dP[gid] = bubble.gamma * np * 2. ;
				
				// E_particle = sqrt(pow(M[gid], 2.) + P_1*P_1 + P_2*P_2 + P_3*P_3);
				
				particle.E = calculateEnergy(particle, particle.m);
				
				t_interactedFalse[gid] += 1;
				//t_passedFalse[gid] += 0;
				//t_interactedTrue[gid] += 0;
			}
			// Particle gets through the bubble wall -> Gets higher mass
			// Particle gets higher mass
			else if (-np >= sqrt(Delta_M2)) {
				if (M_in < M_out){
					particle.m = M_out;
				}
				else {
					particle.m = M_in;
				}
				
				// P_i = P_i + np * n_i * sqrt(1-sqrt(1-Δm^2/np^2))
				particle.pX = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_1, particle.pX);
				particle.pY = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_2, particle.pY);
				particle.pZ = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_3, particle.pZ);
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.-Delta_M2/pow(np, 2.)));
				// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
				
				particle.E = calculateEnergy(particle, particle.m);

				t_interactedFalse[gid] += 1;
				t_passedFalse[gid] += 1;
				//t_interactedTrue[gid] += 0;
				
			}
			else {
				printf("Error -np<0, np=%.2f\n", np);
				moveLinear(&particle, V_1, V_2, V_3, timeToWall);
			}
			
			// Update velocity vector
			V_1 = particle.pX / particle.E;
			V_2 = particle.pY / particle.E;
			V_3 = particle.pZ / particle.E;
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
			// t_interactedFalse[gid] += 0;
			// t_passedFalse[gid] += 0;
			// t_interactedTrue[gid] += 0;
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
			
			np = bubble.speed * bubble.gamma*particle.E - n_1*particle.pX - n_2*particle.pY - n_3*particle.pZ;
			if (np > 0){
				// P_i = P_i + np * n_i * sqrt(1-sqrt(1+Δm^2/np^2))
				particle.pX = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_1, particle.pX);
				particle.pY = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_2, particle.pY);
				particle.pZ = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_3, particle.pZ);
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.+Delta_M2/pow(np, 2.)));
				
				if (M_in < M_out) {
					particle.m = M_in;
					}
				else {
					particle.m = M_out;
					}
				
				// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
				particle.E = calculateEnergy(particle, particle.m);
			}
			
			// Update velocity vector
			V_1 = particle.pX / particle.E;
			V_2 = particle.pY / particle.E;
			V_3 = particle.pZ / particle.E;
			// Movement after the bubble interaction
			//X_1 = fma(V_1, dt - timeToWall, X_1);
			//X_2 = fma(V_2, dt - timeToWall, X_2);
			//X_3 = fma(V_3, dt - timeToWall, X_3);
			moveLinear(&particle, V_1, V_2, V_3, dt - timeToWall);
			
			//t_interactedFalse[gid] += 0;
			//t_passedFalse[gid] += 0;
			t_interactedTrue[gid] += 1;
		}
	}
	/*
	* ========== Cyclic condition ==========
	* If -R_cyclic < particle.x < R_cyclic 	-> Then leave coordinate same
	* If particle.x > R_cyclic 				-> Then particle.x = particle.x - 2 * R_cyclic
	* If particle.x < -R_cyclic				-> Then particle.x = particle.x + 2 * R_cyclic
	*/
	
	double cyclicRadius = t_cyclicRadius[0];
	particle.x = (cyclicRadius > fabs(particle.x)) * particle.x +
				 (particle.x > cyclicRadius) * (particle.x - 2*cyclicRadius) + 
				 (particle.x < -cyclicRadius) * (particle.x + 2*cyclicRadius);
	particle.y = (cyclicRadius > fabs(particle.y)) * particle.y +
				 (particle.y > cyclicRadius) * (particle.y - 2*cyclicRadius) + 
				 (particle.y < -cyclicRadius) * (particle.y + 2*cyclicRadius);
	particle.z = (cyclicRadius > fabs(particle.z)) * particle.z +
				 (particle.z > cyclicRadius) * (particle.z - 2*cyclicRadius) + 
				 (particle.z < -cyclicRadius) * (particle.z + 2*cyclicRadius);
	t_particles[gid].x = particle.x;
	t_particles[gid].y = particle.y;
	t_particles[gid].z = particle.z;
	
	t_particles[gid].pX = particle.pX;
	t_particles[gid].pY = particle.pY;
	t_particles[gid].pZ = particle.pZ;
	t_particles[gid].E = particle.E;
	t_particles[gid].m = particle.m;
}


__kernel void particle_step(
	__global Particle *t_particles,
	__constant double *boundaryDistanceFromCenter,
	__constant double *t_dt
	){
	unsigned int gid = get_global_id(0);
	
	Particle particle = t_particles[gid];
	double V_1 = particle.pX/particle.E;
	double V_2 = particle.pY/particle.E;
	double V_3 = particle.pZ/particle.E;	
	
	double dt = t_dt[0];
	
	moveLinear(&particle, V_1, V_2, V_3, dt);
	
	t_particles[gid] = particle;
}

__kernel void assign_cell_index_to_particle(
	__global Particle *t_particles,
	__global const unsigned int *maxCellIndex,
	__global const double *cellLength,
	__global const double *cuboidShift
	){
	unsigned int gid = get_global_id(0);
	
	Particle particle = t_particles[gid];
	// Find cell numbers
	int x_index = (int) ((particle.x + cellLength[0]*maxCellIndex[0]/2 + cuboidShift[0]) / cellLength[0]);
	int y_index = (int) ((particle.y + cellLength[0]*maxCellIndex[0]/2 + cuboidShift[1]) / cellLength[0]);
	int z_index = (int) ((particle.z + cellLength[0]*maxCellIndex[0]/2 + cuboidShift[2]) / cellLength[0]);
	// Idx = 0 -> if particle is outside of the cuboid cell structure	
	// Convert cell number into 1D vector
	if ((x_index < 0) || (x_index >= maxCellIndex[0])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else if ((y_index < 0) || (y_index >= maxCellIndex[0])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else if ((z_index < 0) || (z_index >= maxCellIndex[0])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else {
		t_particles[gid].idxCollisionCell = 1 + x_index + y_index * maxCellIndex[0] + z_index * maxCellIndex[0] * maxCellIndex[0];
	}
	
	//t_particles[gid].idxCollisionCell = x_index + y_index + z_index;

}

__kernel void assign_particle_cell_index_two_phase(
	__global Particle *t_particles,
	__global const int *maxCellIndex,
	__global const double *cellLength,
	__global const double *cuboidShift // Random particle location shift
	
	){
	unsigned int gid = get_global_id(0);
	
	Particle particle = t_particles[gid];
	// Find cell numbers
	int a = (int) ((particle.x + cuboidShift[0]) / cellLength[0]);
	int b = (int) ((particle.y + cuboidShift[1]) / cellLength[1]);
	int c = (int) ((particle.z + cuboidShift[2]) / cellLength[2]);
	// Idx = 0 -> if particle is outside of the cuboid cell structure
	// Convert cell number into 1D vector. First half of the vector is for outisde the bubble and second half is inside the bubble
	if ((a < 0) || (a >= maxCellIndex[0])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else if ((b < 0) || (b >= maxCellIndex[1])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else if ((c < 0) || (c >= maxCellIndex[2])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else {
		t_particles[gid].idxCollisionCell = 1 + a + b * maxCellIndex[0] + c * maxCellIndex[0] * maxCellIndex[1] + particle.b_inBubble * maxCellIndex[0] * maxCellIndex[1]*maxCellIndex[2];
	}
	
}

__kernel void label_particles_position_by_coordinate(
	__global Particle *t_particles,
	__global Bubble *t_bubble
	){
	unsigned int gid = get_global_id(0);
	
	// If R_b^2 > R_x^2 then particle is inside the bubble
	t_particles[gid].b_inBubble = fma(
									t_particles[gid].x, t_particles[gid].x,
										fma(t_particles[gid].y, t_particles[gid].y,
											t_particles[gid].z * t_particles[gid].z )) < t_bubble[0].radius2;
}

__kernel void label_particles_position_by_mass(
	__global Particle *t_particles,
	__global double *mass_in
	){
	unsigned int gid = get_global_id(0);
	
	// If R_b^2 > R_x^2 then particle is inside the bubble
	t_particles[gid].b_inBubble = t_particles[gid].m == mass_in[0];
}

__kernel void transform_momentum(
	__global Particle *t_particles,
	__global CollisionCell *t_cells,	
	__global unsigned int *number_of_cells
	
	){
		unsigned int gid = get_global_id(0);
		Particle particle = t_particles[gid];
		// If in bubble then cell number is doubled and second half is in bubble cells.
		if (particle.idxCollisionCell != 0){
			CollisionCell cell = t_cells[particle.idxCollisionCell];
			
			double gamma_minus_one = cell.gamma - 1;
			double cos_theta = cos(cell.theta);
			double one_minus_cos_theta = 1 - cos_theta;
			double sin_theta = sin(cell.theta);
			double p0, p1, p2, p3;
			
			
			
			if ((cell.particle_count > 1) && (cell.mass != 0)){
				// Lorentz boost
				
				p0 = cell.gamma * (
					particle.E 
					- particle.pX * cell.vX 
					- particle.pY * cell.vY 
					- particle.pZ * cell.vZ
					);
				p1 = - particle.E * cell.gamma  * cell.vX 
					+ particle.pX * (1 + cell.vX * cell.vX * gamma_minus_one/cell.v2)
					+ particle.pY * cell.vX * cell.vY * gamma_minus_one/cell.v2
					+ particle.pZ * cell.vX * cell.vZ * gamma_minus_one/cell.v2;
				
				p2 = -cell.gamma * particle.E * cell.vY 
					+ particle.pY * (1 + cell.vY * cell.vY * gamma_minus_one/cell.v2)
					+ particle.pX * cell.vX * cell.vY * gamma_minus_one/cell.v2
					+ particle.pZ * cell.vY * cell.vZ * gamma_minus_one/cell.v2;
				
				p3 = -cell.gamma * particle.E * cell.vZ 
					+ particle.pZ * (1 + cell.vZ * cell.vZ * gamma_minus_one/cell.v2)
					+ particle.pX * cell.vX * cell.vZ * gamma_minus_one/cell.v2
					+ particle.pY * cell.vY * cell.vZ * gamma_minus_one/cell.v2;
				
				particle.E = p0;
				particle.pX = p1;
				particle.pY = p2;
				particle.pZ = p3;

				// Rotate momentum
				p1 = particle.pX * (cos_theta + pow(cell.x, 2) * one_minus_cos_theta) + 
					 particle.pY * (cell.x * cell.y * one_minus_cos_theta - cell.z * sin_theta) +
					 particle.pZ * (cell.x * cell.z * one_minus_cos_theta + cell.y * sin_theta);
				p2 = particle.pX * (cell.x * cell.y * one_minus_cos_theta + cell.z * sin_theta) + 
					 particle.pY * (cos_theta + pow(cell.y, 2) * one_minus_cos_theta) +
					 particle.pZ * (cell.y*cell.z*one_minus_cos_theta - cell.x * sin_theta);
				p3 = particle.pX * (cell.x * cell.z * one_minus_cos_theta - cell.y * sin_theta) + 
					 particle.pY * (cell.y * cell.z * one_minus_cos_theta + cell.x * sin_theta) +
					 particle.pZ * (cos_theta + pow(cell.z, 2) * one_minus_cos_theta);
				
				particle.pX = p1;
				particle.pY = p2;
				particle.pZ = p3;
				
				// Lorentz inverse transformation
				p0 = cell.gamma * (
					particle.E 
					+ particle.pX * cell.vX 
					+ particle.pY * cell.vY 
					+ particle.pZ * cell.vZ
					);
				p1 = cell.gamma * particle.E * cell.vX 
					+ particle.pX * (1 + cell.vX * cell.vX * gamma_minus_one/cell.v2)
					+ particle.pY * cell.vX * cell.vY * gamma_minus_one/cell.v2
					+ particle.pZ * cell.vX * cell.vZ * gamma_minus_one/cell.v2;
				
				p2 = cell.gamma * particle.E * cell.vY 
					+ particle.pY * (1 + cell.vY * cell.vY * gamma_minus_one/cell.v2)
					+ particle.pX * cell.vX * cell.vY * gamma_minus_one/cell.v2
					+ particle.pZ * cell.vY * cell.vZ * gamma_minus_one/cell.v2;
				
				p3 = cell.gamma * particle.E * cell.vZ 
					+ particle.pZ * (1 + cell.vZ * cell.vZ * gamma_minus_one/cell.v2)
					+ particle.pX * cell.vX * cell.vZ * gamma_minus_one/cell.v2
					+ particle.pY * cell.vY * cell.vZ * gamma_minus_one/cell.v2;
				
				particle.E = p0;
				particle.pX = p1;
				particle.pY = p2;
				particle.pZ = p3;
				
				t_particles[gid] = particle;
			}
		}
	}

__kernel void particle_bounce(
	__global Particle *t_particles,
	__global double *boundaryDistanceFromCenter // [x_delta]
	){
	unsigned int gid = get_global_id(0);
	
	Particle particle = t_particles[gid];
	
		// If Particle is inside the boundary leave value same. Otherwise change the sign
	
	// abs(x) < Boundary -> leave momentum
	// abs(x) > Boundary and x < -Boundary -> Set momentum positive
	// abs(x) > Boundary and x > Boundary -> Set momentum negative
	particle.pX = ( boundaryDistanceFromCenter[0] > fabs(particle.x)) * particle.pX + // If inside, leave momentum alone
				  ((boundaryDistanceFromCenter[0] < fabs(particle.x)) && (-boundaryDistanceFromCenter[0] > particle.x)) * fabs(particle.pX) - // particle too -, chane to + 
				  ((boundaryDistanceFromCenter[0] < fabs(particle.x)) && (boundaryDistanceFromCenter[0] < particle.x)) * fabs(particle.pX); // particle too +, chane to - 
	particle.pY = ( boundaryDistanceFromCenter[0] > fabs(particle.y)) * particle.pY +
				  ((boundaryDistanceFromCenter[0] < fabs(particle.y)) && (-boundaryDistanceFromCenter[0] > particle.y)) * fabs(particle.pY) -
				  ((boundaryDistanceFromCenter[0] < fabs(particle.y)) && (boundaryDistanceFromCenter[0] < particle.y)) * fabs(particle.pY);
	particle.pZ = ( boundaryDistanceFromCenter[0] > fabs(particle.z)) * particle.pZ +
				  ((boundaryDistanceFromCenter[0] < fabs(particle.z)) && (-boundaryDistanceFromCenter[0] > particle.z)) * fabs(particle.pZ) -
				  ((boundaryDistanceFromCenter[0] < fabs(particle.z)) && (boundaryDistanceFromCenter[0] < particle.z)) * fabs(particle.pZ);
	
	// Update result
	t_particles[gid] = particle;
}
