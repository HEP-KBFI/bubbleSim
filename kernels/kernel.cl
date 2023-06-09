#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Particle struct (also defined in C++ code)
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

// Bubble struct (also defined in C++ code)
typedef struct Bubble {
  double radius;
  double radius2;           // Squared
  double radiusAfterStep2;  // (radius + speed * dt)^2
  double speed;
  double gamma;
  double gammaXspeed;  // gamma * speed
} Bubble;

// In development
typedef struct CollisionCell {
  // 
  double gamma;
  double vX;
  double vY;
  double vZ;
  
  // Rotation vector
  double x;
  double y;
  double z;
  double theta;
  
  double pE;
  double pX;
  double pY;
  double pZ;
  
  double v2; // v2 = Sum: v_i^2 
  double mass;  
  unsigned int particle_count;
} CollisionCell;

/*
 * Given particle and it's velocity move it's location by time=dt.
 * Also this function already updates particle location.
*/
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

/*
 * Assuming sphere and point collision calculate time when these two collide.
 * |r + v * dt|^2 = (R_b+V_b*dt)^2
*/
double calculateTimeToWall(
		Particle particle, Bubble bubble, double t_dt
	) {
	double time1, time2;
	/*
	 * p = 4-vector
	 * P (capital) = 3-vector
	*/
	
	// P*P/E^2 - V_b^2
	double a = fma(particle.pX, particle.pX,
					fma(particle.pY, particle.pY,
						fma(particle.pZ, particle.pZ, 0.)))/pow(particle.E, 2)
							- bubble.speed * bubble.speed;
	// X*P/E^2 - V_b^R_b
	double b = fma(particle.x, particle.pX/particle.E,
					fma(particle.y, particle.pY/particle.E,
						fma(particle.z, particle.pZ/particle.E, - bubble.radius * bubble.speed)));
	// X*X - R_b^2
	double c = fma(particle.x, particle.x,
					fma(particle.y, particle.y,
						fma(particle.z, particle.z, - bubble.radius * bubble.radius)));

	double d = fma(b, b, - a * c);
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

/*
 * Calculate normal for bubble wall. 
 * Direction is defined such that normal shows in the direction of mass growth.
*/
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

/*
 * Take in particle and calculate it's radius from (0,0,0)=center of bubble.
*/
double calculateRadiusSquared(Particle particle){
	return fma(particle.x, particle.x, fma(particle.y, particle.y, particle.z * particle.z));
}

/*
 * Calculate particle energy
*/
double calculateEnergy(Particle particle){
	return sqrt(fma(particle.pX, particle.pX, 
					fma(particle.pY, particle.pY, 
						fma(particle.pZ, particle.pZ, pow(particle.m, 2)))));
}

__kernel void particle_bubble_step(	
	__global Particle *particles,	
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
	/*
	 * t_dP is a array where each element is "pressure" by particle respective to it's index.
	 * t_dP is not actual pressure but actually energy change ΔE. Afterwards ΔP = ΔE/Area
	*/

	// Bubble parameters	
	struct Bubble bubble = t_bubble[0];
	// Particle
	struct Particle particle = particles[gid];
	
	double M_in = t_m_in[0];
	double M_out = t_m_out[0];
	double Delta_M2 = t_delta_m2[0];
	double dt = t_dt[0];
	
	/* 
	 * Particle parameters
	*/
	
	// Veloscity is needed to calculate if particle crosses bubble wall
	double Vx = particle.pX/particle.E;
	double Vy = particle.pY/particle.E;
	double Vz = particle.pZ/particle.E;
	// Particle radius -> check if particle inside the bubble
	double X2 = calculateRadiusSquared(particle);
	
	// Particle coordinates after time dt assuming it's movement is linear.
	double X_dt_x = fma(Vx, dt, particle.x); // x component
	double X_dt_y = fma(Vy, dt, particle.y); // y component
	double X_dt_z = fma(Vz, dt, particle.z); // z component
	// fma(a, b, c) = a*b + c (optimized)
	double X_dt2 = fma(X_dt_x, X_dt_x, fma( X_dt_y, X_dt_y, X_dt_z * X_dt_z));
	
	double n_x, n_y, n_z;
	double timeToWall, np;
			
	// If M_in < M_out -> Check if particle starts inside
	// If M_in > M_out -> Check if particle starts outside
	if (((X2 < bubble.radius2) && (M_in < M_out)) || ((X2 > bubble.radius2) && (M_in > M_out))){
		// If M_in < M_out -> Check if particle stays in
		// If M_in > M_out -> Check if particle stays out
		if (((X_dt2 < bubble.radiusAfterStep2) && (M_in < M_out)) || ((X_dt2 > bubble.radiusAfterStep2) && (M_in > M_out))) {
			// If Particle stays inside the bubble then just move linear
			moveLinear(&particle, Vx, Vy, Vz, dt);
			t_dP[gid] = 0.; // Applies zero pressure.
		}
		// Particle collides with the bubble
		else {
			// Calculate time to wall and move the particle to the wall
			timeToWall = calculateTimeToWall(particle, bubble, dt);
			moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			// X2 must be used to calculate normal because "bubble" radius changes as well
			// and current object bubble radius can't be used
			X2 = calculateRadiusSquared(particle); 
			
			// If M_in > M_out then normal is towards inside
			// If M_in < M_out then normal is towards outside
			calculateNormal(&n_x, &n_y, &n_z, particle, bubble, X2, M_in, M_out);
					
			// Calculate collision term
			np = bubble.speed * bubble.gamma * particle.E - n_x*particle.pX - n_y*particle.pY - n_z*particle.pZ;
			
			// ========== Interaction with the bubble ==========
			// Particle doesn't have enough energy to cross the bubble wall
			// Particle keeps lower mass
			if (np*np < Delta_M2){
				// Update particle 4-momentum after collision
				particle.pX = fma(np*2., n_x, particle.pX);
				particle.pY = fma(np*2., n_y, particle.pY);
				particle.pZ = fma(np*2., n_z, particle.pZ);
				// Calculate applied pressure
				t_dP[gid] = bubble.gamma * np * 2. ;
				particle.E = calculateEnergy(particle);
				
				//Particle only interacts
				t_interactedFalse[gid] += 1;
			}
			// Particle crosses the bubble wall and gets new mass
			else {
				if (M_in < M_out){
					particle.m = M_out;
				}
				else {
					particle.m = M_in;
				}
				// Calculate particle momentum after collision
				particle.pX = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_x, particle.pX);
				particle.pY = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_y, particle.pY);
				particle.pZ = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_z, particle.pZ);
				// Calculate applied pressure
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.-Delta_M2/pow(np, 2.)));
				particle.E = calculateEnergy(particle);

				// Particle interacts and passes through
				t_interactedFalse[gid] += 1;
				t_passedFalse[gid] += 1;
				
			}
			// Update particle velocity to move it amount of "dt - timeToWall"
			Vx = particle.pX / particle.E;
			Vy = particle.pY / particle.E;
			Vz = particle.pZ / particle.E;
			// ========== Movement after the bubble interaction ==========
			moveLinear(&particle, Vx, Vy, Vz, dt - timeToWall);
		}
	}
	// If M_in < M_out -> Then particle starts outside
	// If M_in > M_out -> Then particle starts inside
	else {
		// If M_in < M_out -> Particle stays outside
		// If M_in > M_out -> Particle stays inside
		if (((X_dt2 > bubble.radiusAfterStep2) && (M_in < M_out)) || ((X_dt2 < bubble.radiusAfterStep2) && (M_in > M_out))) {
			// If particle doesn't interact with the wall during time interval dt then move linear
			moveLinear(&particle, Vx, Vy, Vz, dt);
			t_dP[gid] = 0.;
		}
		// If M_in < M_out -> Particle goes inside from out of the bubble
		// If M_in > M_out -> Particle goes outside from the bubble to inside
		else {
			// Calculate time till collision and move up to this point
			timeToWall = calculateTimeToWall(particle, bubble, dt);
			moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			X2 = calculateRadiusSquared(particle);
			
			// Calculate normal during the collision
			calculateNormal(&n_x, &n_y, &n_z, particle, bubble, X2, M_in, M_out);
			
			// Calculate collision term
			np = bubble.speed * bubble.gamma*particle.E - n_x*particle.pX - n_y*particle.pY - n_z*particle.pZ;
			
			// As particle moves from higher mass region to lower mass region
			// only crossing wall is possible.
			if (M_in < M_out) {
				particle.m = M_in;
				}
			else {
				particle.m = M_out;
				}
			
			// Update momentum
			particle.pX = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_x, particle.pX);
			particle.pY = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_y, particle.pY);
			particle.pZ = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_z, particle.pZ);
			// Calculate applied pressure
			t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.+Delta_M2/pow(np, 2.)));
			particle.E = calculateEnergy(particle);
			
			// Update particle velocity and move amount of (dt - timeToWall)
			Vx = particle.pX / particle.E;
			Vy = particle.pY / particle.E;
			Vz = particle.pZ / particle.E;
			// Movement after the bubble interaction
			moveLinear(&particle, Vx, Vy, Vz, dt - timeToWall);
			// Particle interacts from the higher mass side
			t_interactedTrue[gid] += 1;
		}
	}
	
	// Update particle information
	particles[gid].x = particle.x;
	particles[gid].y = particle.y;
	particles[gid].z = particle.z;
	particles[gid].pX = particle.pX;
	particles[gid].pY = particle.pY;
	particles[gid].pZ = particle.pZ;
	particles[gid].E = particle.E;
	particles[gid].m = particle.m;
}

__kernel void particle_bubble_step_cyclic(	
	__global Particle *particles,	
	__global double *t_dP,
	__global char *t_interactedFalse,
	__global char *t_passedFalse,
	__global char *t_interactedTrue,
	__constant Bubble *t_bubble,
	__constant double *t_dt,
	__constant double *t_m_in,
	__constant double *t_m_out,
	__constant double *t_delta_m2,
	__constant double *boundaryRadius
	){
	
	unsigned int gid = get_global_id(0);
	// dE - dP is not actual energy difference. dE = ΔE/R_b -> to avoid singularities/noise near R_b ~ 0


	// Bubble parameters	
	struct Bubble bubble = t_bubble[0];
	// Particle
	struct Particle particle = particles[gid];
	
	double M_in = t_m_in[0];
	double M_out = t_m_out[0];
	double Delta_M2 = t_delta_m2[0];
	
	// Particle parameters
	double dt = t_dt[0];

	double Vx = particle.pX/particle.E;
	double Vy = particle.pY/particle.E;
	double Vz = particle.pZ/particle.E;
	
	double X2 = calculateRadiusSquared(particle);
	
	double X_dt_x = fma(Vx, dt, particle.x);
	double X_dt_y = fma(Vy, dt, particle.y);
	double X_dt_z = fma(Vz, dt, particle.z);
	double X_dt2 = fma(X_dt_x, X_dt_x, fma( X_dt_y, X_dt_y, X_dt_z * X_dt_z));
	
	double n_x, n_y, n_z;
	double timeToWall, np;

	// Check if particle in false vacuum
	// If M_in < M_out -> Check if particle starts inside
	// If M_in > M_out -> Check if particle starts outside
	if (((X2 < bubble.radius2) && (M_in < M_out)) || ((X2 > bubble.radius2) && (M_in > M_out))){
		// If M_in < M_out -> Check if particle stays in
		// If M_in > M_out -> Check if particle stays out
		if (((X_dt2 < bubble.radiusAfterStep2) && (M_in < M_out)) || ((X_dt2 > bubble.radiusAfterStep2) && (M_in > M_out))) {
			// X_1 = fma(Vx, dt, X_1);
			// X_2 = fma(Vy, dt, X_2);
			// X_3 = fma(Vz, dt, X_3);
			moveLinear(&particle, Vx, Vy, Vz, dt);
			t_dP[gid] = 0.;
			// t_interactedFalse[gid] += 0;
			// t_passedFalse[gid] += 0;
			// t_interactedTrue[gid] += 0;
		}
		// Maybe get outside
		else {
			timeToWall = calculateTimeToWall(particle, bubble, dt);
			// X_1 = fma(Vx, timeToWall, X_1);
			// X_2 = fma(Vy, timeToWall, X_2);
			// X_3 = fma(Vz, timeToWall, X_3);
			moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			// Update X2
			// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
			X2 = calculateRadiusSquared(particle);
			// n_x = X_1 * Gamma/sqrt(X2);
			// n_y = X_2 * Gamma/sqrt(X2);
			// n_z = X_3 * Gamma/sqrt(X2);
			
			// If M_in > M_out then normal is towards inside
			// If M_in < M_out then normal is towards outside
			calculateNormal(&n_x, &n_y, &n_z, particle, bubble, X2, M_in, M_out);
					
			np = bubble.speed * bubble.gamma * particle.E - n_x*particle.pX - n_y*particle.pY - n_z*particle.pZ;
			
			// ========== Interaction with the bubble ==========
			// Particle bounces from the wall and stays where it was -> Stays in lower mass region
			// Particle keeps lower mass
			if ((0 < -np) && (-np < sqrt(Delta_M2))){
				// P_i = P_i + 2 * np * n_i
				particle.pX = fma(np*2., n_x, particle.pX);
				particle.pY = fma(np*2., n_y, particle.pY);
				particle.pZ = fma(np*2., n_z, particle.pZ);
				t_dP[gid] = bubble.gamma * np * 2. ;
				
				// E_particle = sqrt(pow(M[gid], 2.) + P_1*P_1 + P_2*P_2 + P_3*P_3);
				
				particle.E = calculateEnergy(particle);
				
				t_interactedFalse[gid] += 1;
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
				
				particle.pX = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_x, particle.pX);
				particle.pY = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_y, particle.pY);
				particle.pZ = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_z, particle.pZ);
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.-Delta_M2/pow(np, 2.)));
				
				particle.E = calculateEnergy(particle);

				t_interactedFalse[gid] += 1;
				t_passedFalse[gid] += 1;
				
			}
			else {
				printf("Error -np<0, np=%.2f, i:%i\n", np, gid);
				moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			}
			
			// Update velocity vector
			Vx = particle.pX / particle.E;
			Vy = particle.pY / particle.E;
			Vz = particle.pZ / particle.E;
			// ========== Movement after the bubble interaction ==========
			moveLinear(&particle, Vx, Vy, Vz, dt - timeToWall);
		}
	}
	
	// If M_in < M_out -> Then particle starts outside
	// If M_in > M_out -> Then particle starts inside
	else {
		// If M_in < M_out -> Particle stays outside
		// If M_in > M_out -> Particle stays inside
		if (((X_dt2 > bubble.radiusAfterStep2) && (M_in < M_out)) || ((X_dt2 < bubble.radiusAfterStep2) && (M_in > M_out))) {
			// X_1 = fma(Vx, dt, X_1);
			// X_2 = fma(Vy, dt, X_2);
			// X_3 = fma(Vz, dt, X_3);
			moveLinear(&particle, Vx, Vy, Vz, dt);
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
			// X_1 = fma(Vx, timeToWall, X_1);
			// X_2 = fma(Vy, timeToWall, X_2);
			// X_3 = fma(Vz, timeToWall, X_3);
			moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			// Update X2
			// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
			X2 = calculateRadiusSquared(particle);
			// n_x = X_1 * Gamma/sqrt(X2);
			// n_y = X_2 * Gamma/sqrt(X2);
			// n_z = X_3 * Gamma/sqrt(X2);
			calculateNormal(&n_x, &n_y, &n_z, particle, bubble, X2, M_in, M_out);
			
			np = bubble.speed * bubble.gamma*particle.E - n_x*particle.pX - n_y*particle.pY - n_z*particle.pZ;
			if (np > 0){
				// P_i = P_i + np * n_i * sqrt(1-sqrt(1+Δm^2/np^2))
				particle.pX = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_x, particle.pX);
				particle.pY = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_y, particle.pY);
				particle.pZ = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_z, particle.pZ);
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.+Delta_M2/pow(np, 2.)));
				
				if (M_in < M_out) {
					particle.m = M_in;
					}
				else {
					particle.m = M_out;
					}
				
				// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
				particle.E = calculateEnergy(particle);
			}
			
			// Update velocity vector
			Vx = particle.pX / particle.E;
			Vy = particle.pY / particle.E;
			Vz = particle.pZ / particle.E;
			// Movement after the bubble interaction
			//X_1 = fma(Vx, dt - timeToWall, X_1);
			//X_2 = fma(Vy, dt - timeToWall, X_2);
			//X_3 = fma(Vz, dt - timeToWall, X_3);
			moveLinear(&particle, Vx, Vy, Vz, dt - timeToWall);
			
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
	
	double r_boundaryRadius = boundaryRadius[0];
	particle.x = (r_boundaryRadius > fabs(particle.x)) * particle.x +
				 (particle.x > r_boundaryRadius) * (particle.x - 2*r_boundaryRadius) + 
				 (particle.x < -r_boundaryRadius) * (particle.x + 2*r_boundaryRadius);
	particle.y = (r_boundaryRadius > fabs(particle.y)) * particle.y +
				 (particle.y > r_boundaryRadius) * (particle.y - 2*r_boundaryRadius) + 
				 (particle.y < -r_boundaryRadius) * (particle.y + 2*r_boundaryRadius);
	particle.z = (r_boundaryRadius > fabs(particle.z)) * particle.z +
				 (particle.z > r_boundaryRadius) * (particle.z - 2*r_boundaryRadius) + 
				 (particle.z < -r_boundaryRadius) * (particle.z + 2*r_boundaryRadius);
				 
	particles[gid].x = particle.x;
	particles[gid].y = particle.y;
	particles[gid].z = particle.z;
	
	particles[gid].pX = particle.pX;
	particles[gid].pY = particle.pY;
	particles[gid].pZ = particle.pZ;
	particles[gid].E = particle.E;
	particles[gid].m = particle.m;
}

__kernel void particle_bubble_step_cyclic_mass_inverted(	
	__global Particle *particles,	
	__global double *t_dP,
	__global char *t_interactedFalse,
	__global char *t_passedFalse,
	__global char *t_interactedTrue,
	__constant Bubble *t_bubble,
	__constant double *t_dt,
	__constant double *t_m_in,
	__constant double *t_m_out,
	__constant double *t_delta_m2,
	__constant double *boundaryRadius
	){
	
	unsigned int gid = get_global_id(0);
	// Mass inverted: Simualte situation where mass inside is higher
	// dE - dP is not actual energy difference. dE = ΔE/R_b -> to avoid singularities/noise near R_b ~ 0

	// Bubble parameters	
	struct Bubble bubble = t_bubble[0];
	// Particle
	struct Particle particle = particles[gid];
	
	double M_in = t_m_in[0];
	double M_out = t_m_out[0];
	double Delta_M2 = t_delta_m2[0];
	
	// Particle parameters
	double dt = t_dt[0];

	double Vx = particle.pX/particle.E;
	double Vy = particle.pY/particle.E;
	double Vz = particle.pZ/particle.E;
	
	double X2 = calculateRadiusSquared(particle);
	
	double X_dt_x = fma(Vx, dt, particle.x);
	double X_dt_y = fma(Vy, dt, particle.y);
	double X_dt_z = fma(Vz, dt, particle.z);
	double X_dt2 = fma(X_dt_x, X_dt_x, fma( X_dt_y, X_dt_y, X_dt_z * X_dt_z));
	
	double n_x, n_y, n_z;
	double timeToWall, np;

	// Check if particle in false vacuum
	// If M_in < M_out -> Check if particle starts inside
	// If M_in > M_out -> Check if particle starts outside
	if (((X2 < bubble.radius2) && (M_out < M_in)) || ((X2 > bubble.radius2) && (M_out > M_in))){
		// If M_in < M_out -> Check if particle stays in
		// If M_in > M_out -> Check if particle stays out
		if (((X_dt2 < bubble.radiusAfterStep2) && (M_out < M_in)) || ((X_dt2 > bubble.radiusAfterStep2) && (M_out > M_in))) {
			// X_1 = fma(Vx, dt, X_1);
			// X_2 = fma(Vy, dt, X_2);
			// X_3 = fma(Vz, dt, X_3);
			moveLinear(&particle, Vx, Vy, Vz, dt);
			t_dP[gid] = 0.;
			// t_interactedFalse[gid] += 0;
			// t_passedFalse[gid] += 0;
			// t_interactedTrue[gid] += 0;
		}
		// Maybe get outside
		else {
			timeToWall = calculateTimeToWall(particle, bubble, dt);
			// X_1 = fma(Vx, timeToWall, X_1);
			// X_2 = fma(Vy, timeToWall, X_2);
			// X_3 = fma(Vz, timeToWall, X_3);
			moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			// Update X2
			// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
			X2 = calculateRadiusSquared(particle);
			// n_x = X_1 * Gamma/sqrt(X2);
			// n_y = X_2 * Gamma/sqrt(X2);
			// n_z = X_3 * Gamma/sqrt(X2);
			
			// If M_in > M_out then normal is towards inside
			// If M_in < M_out then normal is towards outside
			calculateNormal(&n_x, &n_y, &n_z, particle, bubble, X2, M_out, M_in);
					
			np = bubble.speed * bubble.gamma * particle.E - n_x*particle.pX - n_y*particle.pY - n_z*particle.pZ;
			
			// ========== Interaction with the bubble ==========
			// Particle bounces from the wall and stays where it was -> Stays in lower mass region
			// Particle keeps lower mass
			
			if (-np > 0){
				// P_i = P_i + np * n_i * sqrt(1-sqrt(1+Δm^2/np^2))
				particle.pX = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_x, particle.pX);
				particle.pY = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_y, particle.pY);
				particle.pZ = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_z, particle.pZ);
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.+Delta_M2/pow(np, 2.)));
				
				if (M_out < M_in) {
					particle.m = M_out;
					}
				else {
					particle.m = M_in;
					}
				
				// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
				particle.E = calculateEnergy(particle);
				t_interactedTrue[gid] += 1;
			}
			else {
				printf("Error -np<0, np=%.2f, i:%i\n", np, gid);
				moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			}
			
			
			// Update velocity vector
			Vx = particle.pX / particle.E;
			Vy = particle.pY / particle.E;
			Vz = particle.pZ / particle.E;
			// ========== Movement after the bubble interaction ==========
			moveLinear(&particle, Vx, Vy, Vz, dt - timeToWall);
		}
	}
	
	// If M_in < M_out -> Then particle starts outside
	// If M_in > M_out -> Then particle starts inside
	else {
		// If M_in < M_out -> Particle stays outside
		// If M_in > M_out -> Particle stays inside
		if (((X_dt2 > bubble.radiusAfterStep2) && (M_out < M_in)) || ((X_dt2 < bubble.radiusAfterStep2) && (M_out > M_in))) {
			// X_1 = fma(Vx, dt, X_1);
			// X_2 = fma(Vy, dt, X_2);
			// X_3 = fma(Vz, dt, X_3);
			moveLinear(&particle, Vx, Vy, Vz, dt);
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
			// X_1 = fma(Vx, timeToWall, X_1);
			// X_2 = fma(Vy, timeToWall, X_2);
			// X_3 = fma(Vz, timeToWall, X_3);
			moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			// Update X2
			// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
			X2 = calculateRadiusSquared(particle);
			// n_x = X_1 * Gamma/sqrt(X2);
			// n_y = X_2 * Gamma/sqrt(X2);
			// n_z = X_3 * Gamma/sqrt(X2);
			calculateNormal(&n_x, &n_y, &n_z, particle, bubble, X2, M_out, M_in);
			
			np = bubble.speed * bubble.gamma*particle.E - n_x*particle.pX - n_y*particle.pY - n_z*particle.pZ;
			if ((0 < np) && (np < sqrt(Delta_M2))){
				printf("Error  i:%f\n", np);
				// P_i = P_i + 2 * np * n_i
				particle.pX = fma(np*2., n_x, particle.pX);
				particle.pY = fma(np*2., n_y, particle.pY);
				particle.pZ = fma(np*2., n_z, particle.pZ);
				t_dP[gid] = bubble.gamma * np * 2. ;
				
				// E_particle = sqrt(pow(M[gid], 2.) + P_1*P_1 + P_2*P_2 + P_3*P_3);
				
				particle.E = calculateEnergy(particle);
				t_interactedFalse[gid] += 1;
			}
			// Particle gets through the bubble wall -> Gets higher mass
			// Particle gets higher mass
			else if (np >= sqrt(Delta_M2)) {
				if (M_out < M_in){
					particle.m = M_in;
				}
				else {
					particle.m = M_out;
				}
				
				particle.pX = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_x, particle.pX);
				particle.pY = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_y, particle.pY);
				particle.pZ = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_z, particle.pZ);
				t_dP[gid] = bubble.gamma * np * (1.-sqrt(1.-Delta_M2/pow(np, 2.)));
				
				particle.E = calculateEnergy(particle);

				t_interactedFalse[gid] += 1;
				t_passedFalse[gid] += 1;
				
			}
			else {
				printf("Error -np<0, np=%.2f, i:%i\n", np, gid);
				moveLinear(&particle, Vx, Vy, Vz, timeToWall);
			}
			
			// Update velocity vector
			Vx = particle.pX / particle.E;
			Vy = particle.pY / particle.E;
			Vz = particle.pZ / particle.E;
			// Movement after the bubble interaction
			//X_1 = fma(Vx, dt - timeToWall, X_1);
			//X_2 = fma(Vy, dt - timeToWall, X_2);
			//X_3 = fma(Vz, dt - timeToWall, X_3);
			moveLinear(&particle, Vx, Vy, Vz, dt - timeToWall);
			
			//t_interactedFalse[gid] += 0;
			//t_passedFalse[gid] += 0;
		}
	}
	/*
	* ========== Cyclic condition ==========
	* If -R_cyclic < particle.x < R_cyclic 	-> Then leave coordinate same
	* If particle.x > R_cyclic 				-> Then particle.x = particle.x - 2 * R_cyclic
	* If particle.x < -R_cyclic				-> Then particle.x = particle.x + 2 * R_cyclic
	*/
	
	double r_boundaryRadius = boundaryRadius[0];
	particle.x = (r_boundaryRadius > fabs(particle.x)) * particle.x +
				 (particle.x > r_boundaryRadius) * (particle.x - 2*r_boundaryRadius) + 
				 (particle.x < -r_boundaryRadius) * (particle.x + 2*r_boundaryRadius);
	particle.y = (r_boundaryRadius > fabs(particle.y)) * particle.y +
				 (particle.y > r_boundaryRadius) * (particle.y - 2*r_boundaryRadius) + 
				 (particle.y < -r_boundaryRadius) * (particle.y + 2*r_boundaryRadius);
	particle.z = (r_boundaryRadius > fabs(particle.z)) * particle.z +
				 (particle.z > r_boundaryRadius) * (particle.z - 2*r_boundaryRadius) + 
				 (particle.z < -r_boundaryRadius) * (particle.z + 2*r_boundaryRadius);
				 
	particles[gid].x = particle.x;
	particles[gid].y = particle.y;
	particles[gid].z = particle.z;
	
	particles[gid].pX = particle.pX;
	particles[gid].pY = particle.pY;
	particles[gid].pZ = particle.pZ;
	particles[gid].E = particle.E;
	particles[gid].m = particle.m;
}

__kernel void particle_step(
	__global Particle *particles,
	__constant double *boundaryRadius,
	__constant double *t_dt
	){
	unsigned int gid = get_global_id(0);
	
	Particle particle = particles[gid];
	// double Vx = particle.pX/particle.E;
	// double Vy = particle.pY/particle.E;
	// double Vz = particle.pZ/particle.E;	
	// double dt = t_dt[0];
	
	moveLinear(&particle, particle.pX/particle.E, particle.pY/particle.E, particle.pZ/particle.E, t_dt[0]);
	
				 	
	particles[gid] = particle;
}

__kernel void particles_with_false_bubble_step_reflect(
    __global Particle *particles, __global double *t_dP,
    __global char *t_interactedFalse, __global char *t_passedFalse,
    __global char *t_interactedTrue, __constant Bubble *t_bubble,
    __constant double *t_dt, __constant double *t_m_in,
    __constant double *t_m_out, __constant double *t_delta_m2, __constant double *boundaryRadius) {
  unsigned int gid = get_global_id(0);
  // dE - dP is not actual energy difference. dE = ΔE/R_b -> to avoid
  // singularities/noise near R_b ~ 0

  // Bubble parameters
  struct Bubble bubble = t_bubble[0];
  // Particle
  struct Particle particle = particles[gid];

  double M_in = t_m_in[0];
  double M_out = t_m_out[0];
  double Delta_M2 = t_delta_m2[0];

  // Particle parameters
  double dt = t_dt[0];

  // Calculate particle velocity
  double Vx = particle.pX / particle.E;
  double Vy = particle.pY / particle.E;
  double Vz = particle.pZ / particle.E;

  // fma(a, b, c) = a * b + c
  double X2 = calculateRadiusSquared(particle);

  // Calculate new coordinates if particle would move linearly
  // after time dt
  double X_dt_x = fma(Vx, dt, particle.x);
  double X_dt_y = fma(Vy, dt, particle.y);
  double X_dt_z = fma(Vz, dt, particle.z);

  // Find new particle radius (from the center) squared
  double X_dt2 = fma(X_dt_x, X_dt_x, fma(X_dt_y, X_dt_y, X_dt_z * X_dt_z));

  // Placholders for normal, time to wall, n·p (n and p are 4-vectors)
  double n_x, n_y, n_z;
  double timeToWall, np;
  double collide_radius;

  if (((X2 < bubble.radius2) && (M_in < M_out))) {
    // move particles that stay in
    if ((X_dt2 < bubble.radiusAfterStep2) && (M_in < M_out)) {
      particle.x = X_dt_x;
      particle.y = X_dt_y;
      particle.z = X_dt_z;
      t_dP[gid] = 0.;

    }
    // reflect particles that would to get out
    else {
		
      timeToWall = calculateTimeToWall(particle, bubble, dt);
      moveLinear(&particle, Vx, Vy, Vz, timeToWall);

      // Update particle's radius squared value to calculate normal vector

      // Relativistic reflection algorithm
      // Calculate normal at the location where particle and bubble interact
      collide_radius = pow(bubble.radius + timeToWall * bubble.speed, 2.);
      calculateNormal(&n_x, &n_y, &n_z, particle, bubble, collide_radius, M_in,
                      M_out);
      np = bubble.speed * bubble.gamma * particle.E - n_x * particle.pX -
           n_y * particle.pY - n_z * particle.pZ;
	  
      particle.pX = fma(np * 2., n_x, particle.pX);
      particle.pY = fma(np * 2., n_y, particle.pY);
      particle.pZ = fma(np * 2., n_z, particle.pZ);

      t_dP[gid] = bubble.gamma * np * 2.;
      // Update particle energy
      particle.E = calculateEnergy(particle);
      // Count particle collision with the bubble wall
      t_interactedFalse[gid] += 1;

      // Update velocity vector
      Vx = particle.pX / particle.E;
      Vy = particle.pY / particle.E;
      Vz = particle.pZ / particle.E;

      // ========== Movement after the bubble interaction ==========
      moveLinear(&particle, Vx, Vy, Vz, dt - timeToWall);
    }
  } else {
    particle.x = 100000;
    particle.y = 100000;
    particle.z = 100000;
  }

  particles[gid].x = particle.x;
  particles[gid].y = particle.y;
  particles[gid].z = particle.z;

  particles[gid].pX = particle.pX;
  particles[gid].pY = particle.pY;
  particles[gid].pZ = particle.pZ;
  particles[gid].E = particle.E;
  particles[gid].m = particle.m;
}


__kernel void assign_particle_to_collision_cell(
	__global Particle *particles,
	__global const unsigned int *maxCellIndex,
	__global const double *cellLength,
	__global const double *cuboidShift
	){
	unsigned int gid = get_global_id(0);

	// Read variables to local registers
	Particle particle = particles[gid];

	if (gid == 0){
		printf("%p \n", &particle);
	}
	if (gid == 100){
		printf("%p \n", &particle.x);
	}
	unsigned int r_maximumCellIndex = maxCellIndex[0];
	double r_cellLength = cellLength[0];

	// Find cell number in 3D cell
	int x_index = (int) ((particle.x + r_cellLength*r_maximumCellIndex/2 + cuboidShift[0]) / r_cellLength);
	int y_index = (int) ((particle.y + r_cellLength*r_maximumCellIndex/2 + cuboidShift[1]) / r_cellLength);
	int z_index = (int) ((particle.z + r_cellLength*r_maximumCellIndex/2 + cuboidShift[2]) / r_cellLength);
	
	// Assign particles which don't fit to the cell structure at index = 0.
	int isIndexZero = ((x_index < 0) || (x_index >= r_maximumCellIndex) || (y_index < 0) || (y_index >= r_maximumCellIndex) || (z_index < 0) || (z_index >= r_maximumCellIndex));
	// If index not zero -> !isIndexZero
	particles[gid].idxCollisionCell = 0 + !isIndexZero*(1 + x_index + y_index * r_maximumCellIndex + z_index * r_maximumCellIndex * r_maximumCellIndex);

	// Old implementation
	/*
	if ((x_index < 0) || (x_index >= r_maximumCellIndex)){
		particles[gid].idxCollisionCell = 0;
	}
	else if ((y_index < 0) || (y_index >= r_maximumCellIndex)){
		particles[gid].idxCollisionCell = 0;
	}
	else if ((z_index < 0) || (z_index >= r_maximumCellIndex)){
		particles[gid].idxCollisionCell = 0;
	}
	else {
		particles[gid].idxCollisionCell = 1 + x_index + y_index * r_maximumCellIndex + z_index * r_maximumCellIndex * r_maximumCellIndex;
	}
	*/
}

__kernel void is_particle_in_bubble(
	__global Particle *particles,
	__global Bubble *t_bubble
	){
	unsigned int gid = get_global_id(0);
	
	// If R_b^2 > R_x^2 then particle is inside the bubble
	particles[gid].b_inBubble = fma(
									particles[gid].x, particles[gid].x,
										fma(particles[gid].y, particles[gid].y,
											particles[gid].z * particles[gid].z )) < t_bubble[0].radius2;
}

__kernel void collide_particles(
	__global Particle *particles,
	__global CollisionCell *cells,	
	__global unsigned int *number_of_cells
	
	){
		/*
		* Collision algorith based on multi particle collision algorithm.
		* 
		*/
		unsigned int gid = get_global_id(0);
		Particle particle = particles[gid];
		// If in bubble then cell number is doubled and second half is in bubble cells.
		if (particle.idxCollisionCell != 0){
			CollisionCell cell = cells[particle.idxCollisionCell];
			
			double gamma_minus_one = cell.gamma - 1;
			double cos_theta = cos(cell.theta);
			double one_minus_cos_theta = 1 - cos_theta;
			double sin_theta = sin(cell.theta);
			double p0, p1, p2, p3;
			
			if ((cell.particle_count > 1) && (cell.mass != 0)){
				// Lorentz boost
				//if (gid==0){
				//	printf("p0: %.8f\np1: %.8f\np2: %.8f\np3: %.8f\nv1: %.8f\nv2: %.8f\nv3: %.8f\nn1: %.8f\nn2: %.8f\nn3: %.8f\ntheta: %.8f\n",
				//		particle.E, particle.pX, particle.pY, particle.pZ, cell.vX, cell.vY, cell.vZ, cell.x, cell.y, cell.z, cell.theta
				//		);
				//}
				/*
				* Lorentz transformation to zero momentum frame
				*/

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

				/*
				* Rotate particle momentum
				*/
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
				
				/*
				* Lorentz transformation back to initial momentum frame
				*/
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
				// if (gid==0){
				//	printf("E: %.8f, pX: %.8f, pY: %.8f, pZ: %.8f\n", p0, p1, p2, p3);
				// }
				
				particles[gid] = particle;
			}
		}
}

__kernel void collide_particles2(
	__global Particle *particles,
	__global CollisionCell *cells,	
	__global unsigned int *number_of_cells
	
	){
		/*
		* Collision algorith based on multi particle collision algorithm.
		* 
		*/
		unsigned int gid = get_global_id(0);
		Particle particle = particles[gid];
		// If in bubble then cell number is doubled and second half is in bubble cells.
		if (particle.idxCollisionCell != 0){
			CollisionCell cell = cells[particle.idxCollisionCell];
			
			double gamma_minus_one = cell.gamma - 1;
			double cos_theta = cos(cell.theta);
			double one_minus_cos_theta = 1 - cos_theta;
			double sin_theta = sin(cell.theta);
			double p0, p1, p2, p3;
			
			if ((cell.particle_count > 1) && (cell.mass != 0)){
				// Lorentz boost
				//if (gid==0){
				//	printf("p0: %.8f\np1: %.8f\np2: %.8f\np3: %.8f\nv1: %.8f\nv2: %.8f\nv3: %.8f\nn1: %.8f\nn2: %.8f\nn3: %.8f\ntheta: %.8f\n",
				//		particle.E, particle.pX, particle.pY, particle.pZ, cell.vX, cell.vY, cell.vZ, cell.x, cell.y, cell.z, cell.theta
				//		);
				//}
				/*
				* Lorentz transformation to zero momentum frame
				*/

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

				/*
				* Rotate particle momentum
				*/
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
				
				/*
				* Lorentz transformation back to initial momentum frame
				*/
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
				// if (gid==0){
				//	printf("E: %.8f, pX: %.8f, pY: %.8f, pZ: %.8f\n", p0, p1, p2, p3);
				// }
				
				particles[gid] = particle;
			}
		}
}

__kernel void particle_boundary_check(
	__global Particle *particles,
	__global double *boundaryRadius // [x_delta]
	){
	/*
	* If particle is out of boundaries then update it's location.
	* Update coordinate that particle goes to the other side of the simulation space.
	*/
	unsigned int gid = get_global_id(0);
	
	Particle particle = particles[gid];
	double r_boundaryRadius = boundaryRadius[0];
	double doubleBoundaryRadius = 2 * r_boundaryRadius;

	// If Particle is inside the boundary leave value same. Otherwise change the sign
	
	// abs(x) < r_boundaryRadius 	-> leave just as it is
	// x > r_boundaryRadius 		-> x - 2 * r_boundaryRadius
	// x < r_boundaryRadius 		-> x + 2 * r_boundaryRadius
	particle.x = (r_boundaryRadius > fabs(particle.x)) * particle.x +
				 (particle.x > r_boundaryRadius) * (particle.x - doubleBoundaryRadius) + 
				 (particle.x < -r_boundaryRadius) * (particle.x + doubleBoundaryRadius);
	particle.y = (r_boundaryRadius > fabs(particle.y)) * particle.y +
				 (particle.y > r_boundaryRadius) * (particle.y - doubleBoundaryRadius) + 
				 (particle.y < -r_boundaryRadius) * (particle.y + doubleBoundaryRadius);
	particle.z = (r_boundaryRadius > fabs(particle.z)) * particle.z +
				 (particle.z > r_boundaryRadius) * (particle.z - doubleBoundaryRadius) + 
				 (particle.z < -r_boundaryRadius) * (particle.z + doubleBoundaryRadius);
	// Update result
	particles[gid] = particle;
}
