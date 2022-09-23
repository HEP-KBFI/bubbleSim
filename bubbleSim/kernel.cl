#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

void moveLinear(
	// X_new = X_old + V * dt
	double *t_x1, double *t_x2, double *t_x3, 
	double t_v1, double t_v2, double t_v3,
	double t_dt
	) {
	// x = x + v*dt
	*t_x1 = fma(t_v1, t_dt, *t_x1);
	*t_x2 = fma(t_v2, t_dt, *t_x2);
	*t_x3 = fma(t_v3, t_dt, *t_x3);
	}
double calculateTimeToWall(
		double t_x1, double t_x2, double t_x3, 
		double t_p1, double t_p2, double t_p3,
		double t_E, double t_dt, double t_rb, double t_vb
	) {
	double time1, time2;
	
	double a = fma(t_p1, t_p1/pow(t_E, 2), fma(t_p2, t_p2/pow(t_E, 2), fma(t_p3, t_p3/pow(t_E, 2), - t_vb * t_vb)));
	double b = fma(t_x1, t_p1/t_E, fma(t_x2, t_p2/t_E, fma(t_x3, t_p3/t_E, - t_rb * t_vb)));
	double c = fma(t_x1, t_x1, fma(t_x2, t_x2, fma(t_x3, t_x3, - t_rb * t_rb)));
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
	double t_x1, double t_x2, double t_x3,
	double t_X2, double t_gamma, double t_mass_in, double t_mass_out
	){	
	// If M_in > M_out then normal is towards center of bubble
	if (t_mass_in > t_mass_out){
		*t_n1 = -t_x1 * t_gamma/sqrt(t_X2);
		*t_n2 = -t_x2 * t_gamma/sqrt(t_X2);
		*t_n3 = -t_x3 * t_gamma/sqrt(t_X2);
		}
	// If M_in < M_out then normal is towards outside the bubble
	else {
		*t_n1 = t_x1 * t_gamma/sqrt(t_X2);
		*t_n2 = t_x2 * t_gamma/sqrt(t_X2);
		*t_n3 = t_x3 * t_gamma/sqrt(t_X2);
		}
	}

__kernel void step_double(	
	__global double *X,
	__global double *P,
	__global double *E,
	__global double *M,
	__global double *dP,
	__constant double *dt,
	__constant double *m_in,
	__constant double *m_out,
	__constant double *delta_m2,
	__constant double *bubble_radius,
	__constant double *bubble_radius2,
	__constant double *bubble_radius_dt2,
	__constant double *bubble_speed,
	__constant double *gamma,
	__constant double *gamma_v,
	__global char *interactedFalse,
	__global char *passedFalse,
	__global char *interactedTrue,
	__global double *time2wall
	){
		
	unsigned int gid = get_global_id(0);
	
	// dE - dP is not actual energy difference. dE = ΔE/R_b -> to avoid singularities/noise near R_b ~ 0
	
	// Bubble parameters	
	double R = bubble_radius[0];
	double R2 = bubble_radius2[0];
	double R_dt2 = bubble_radius_dt2[0];
	double V_b = bubble_speed[0];
	double M_in = m_in[0];
	double M_out = m_out[0];
	double Delta_M2 = delta_m2[0];
	double Gamma = gamma[0];
	double Gamma_v = gamma_v[0];
	
	// Particle parameters
	double Dt = dt[0];
	double X_1 = X[3*gid];
	double X_2 = X[3*gid+1];
	double X_3 = X[3*gid+2];
	double P_1 = P[3*gid];
	double P_2 = P[3*gid+1];
	double P_3 = P[3*gid+2];
	double E_particle = E[gid];
	double E_particle_old = E[gid];

	double V_1 = P_1/E_particle;
	double V_2 = P_2/E_particle;
	double V_3 = P_3/E_particle;
	
	// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
	double X2 = fma(X_1, X_1, fma(X_2, X_2, X_3 * X_3));
	
	double X_dt_1 = fma(V_1, Dt, X_1);
	double X_dt_2 = fma(V_2, Dt, X_2);
	double X_dt_3 = fma(V_3, Dt, X_3);
	double X_dt2 = fma(X_dt_1, X_dt_1, fma( X_dt_2, X_dt_2, X_dt_3 * X_dt_3));
	
	double n_1, n_2, n_3;
	double timeToWall, np;
			
	// If M_in < M_out -> Check if particle starts inside
	// If M_in > M_out -> Check if particle starts outside
	if (((X2 < R2) && (M_in < M_out)) || ((X2 > R2) && (M_in > M_out))){
		// If M_in < M_out -> Check if particle stays in
		// If M_in > M_out -> Check if particle stays out
		if (((X_dt2 < R_dt2) && (M_in < M_out)) || ((X_dt2 > R_dt2) && (M_in > M_out))) {
			// X_1 = fma(V_1, Dt, X_1);
			// X_2 = fma(V_2, Dt, X_2);
			// X_3 = fma(V_3, Dt, X_3);
			moveLinear(&X_1, &X_2, &X_3, V_1, V_2, V_3, Dt);
			dP[gid] = 0.;
			interactedFalse[gid] += 0;
			passedFalse[gid] += 0;
			interactedTrue[gid] += 0;
			time2wall[gid] = 0.;
		}
		// Maybe get outside
		else {
			timeToWall = calculateTimeToWall(X_1, X_2, X_3, P_1, P_2, P_3, E_particle, Dt, R, V_b);
			time2wall[gid] = timeToWall;
			// X_1 = fma(V_1, timeToWall, X_1);
			// X_2 = fma(V_2, timeToWall, X_2);
			// X_3 = fma(V_3, timeToWall, X_3);
			moveLinear(&X_1, &X_2, &X_3, V_1, V_2, V_3, timeToWall);
			// Update X2
			// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
			double X2 = fma(X_1, X_1, fma(X_2, X_2, X_3 * X_3));
			// n_1 = X_1 * Gamma/sqrt(X2);
			// n_2 = X_2 * Gamma/sqrt(X2);
			// n_3 = X_3 * Gamma/sqrt(X2);
			
			// If M_in > M_out then normal is towards inside
			// If M_in < M_out then normal is towards outside
			calculateNormal(&n_1, &n_2, &n_3, X_1, X_2, X_3, X2, Gamma, M_in, M_out);
					
			np = V_b * Gamma*E_particle - n_1*P_1 - n_2*P_2 - n_3*P_3;
			
			// ========== Interaction with the bubble ==========
			// Particle bounces from the wall and stays where it was -> Stays in lower mass region
			// Particle keeps lower mass
			if (np*np < Delta_M2){
				// P_i = P_i + 2 * np * n_i
				P_1 = fma(np*2., n_1, P_1);
				P_2 = fma(np*2., n_2, P_2);
				P_3 = fma(np*2., n_3, P_3);
				dP[gid] = Gamma * np * 2. ;
				
				// E_particle = sqrt(pow(M[gid], 2.) + P_1*P_1 + P_2*P_2 + P_3*P_3);
				if (M_in < M_out){
					E_particle = sqrt(fma(P_1, P_1, fma(P_2, P_2, fma(P_3, P_3, pow(M_in, 2)))));
				}
				else {
					E_particle = sqrt(fma(P_1, P_1, fma(P_2, P_2, fma(P_3, P_3, pow(M_out, 2)))));
				}
				
				interactedFalse[gid] += 1;
				passedFalse[gid] += 0;
				interactedTrue[gid] += 0;
			}
			// Particle gets through the bubble wall -> Gets higher mass
			// Particle gets higher mass
			else {
				if (M_in < M_out){
					M[gid] = M_out;
				}
				else {
					M[gid] = M_in;
				}
				
				// P_i = P_i + np * n_i * sqrt(1-sqrt(1-Δm^2/np^2))
				P_1 = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_1, P_1);
				P_2 = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_2, P_2);
				P_3 = fma(np * (1.-sqrt(1.-Delta_M2/pow(np, 2.))), n_3, P_3);
				dP[gid] = Gamma * np * (1.-sqrt(1.-Delta_M2/pow(np, 2.)));
				
				// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
				if (M_in < M_out){
					E_particle = sqrt(fma(P_1, P_1, fma(P_2, P_2, fma(P_3, P_3, pow(M_out, 2)))));
				}
				else {
					E_particle = sqrt(fma(P_1, P_1, fma(P_2, P_2, fma(P_3, P_3, pow(M_in, 2)))));
				}
				interactedFalse[gid] += 1;
				passedFalse[gid] += 1;
				interactedTrue[gid] += 0;
				
			}
			// Update velocity vector
			V_1 = P_1 / E_particle;
			V_2 = P_2 / E_particle;
			V_3 = P_3 / E_particle;
			// ========== Movement after the bubble interaction ==========
			//X_1 = fma(V_1, Dt - timeToWall, X_1);
			//X_2 = fma(V_2, Dt - timeToWall, X_2);
			//X_3 = fma(V_3, Dt - timeToWall, X_3);
			moveLinear(&X_1, &X_2, &X_3, V_1, V_2, V_3, Dt - timeToWall);
			
		}
	}
	
	// If M_in < M_out -> Then particle starts outside
	// If M_in > M_out -> Then particle starts inside
	else {
		// If M_in < M_out -> Particle stays outside
		// If M_in > M_out -> Particle stays inside
		if (((X_dt2 > R_dt2) && (M_in < M_out)) || ((X_dt2 < R_dt2) && (M_in > M_out))) {
			// X_1 = fma(V_1, Dt, X_1);
			// X_2 = fma(V_2, Dt, X_2);
			// X_3 = fma(V_3, Dt, X_3);
			moveLinear(&X_1, &X_2, &X_3, V_1, V_2, V_3, Dt);
			dP[gid] = 0.;
			interactedFalse[gid] += 0;
			passedFalse[gid] += 0;
			interactedTrue[gid] += 0;
			time2wall[gid] = 0.;
		}
		// Particle gets lower mass
		// If M_in < M_out -> Particle goes inside from out of the bubble
		// If M_in > M_out -> Particle goes outside from the bubble to inside
		else {
			timeToWall = calculateTimeToWall(X_1, X_2, X_3, P_1, P_2, P_3, E_particle, Dt, R, V_b);
			time2wall[gid] = timeToWall;
			// X_1 = fma(V_1, timeToWall, X_1);
			// X_2 = fma(V_2, timeToWall, X_2);
			// X_3 = fma(V_3, timeToWall, X_3);
			moveLinear(&X_1, &X_2, &X_3, V_1, V_2, V_3, timeToWall);
			// Update X2
			// double X2 = X_1 * X_1 + X_2 * X_2 + X_3 * X_3;
			double X2 = fma(X_1, X_1, fma(X_2, X_2, X_3 * X_3));
			// n_1 = X_1 * Gamma/sqrt(X2);
			// n_2 = X_2 * Gamma/sqrt(X2);
			// n_3 = X_3 * Gamma/sqrt(X2);
			calculateNormal(&n_1, &n_2, &n_3, X_1, X_2, X_3, X2, Gamma, M_in, M_out);
			
			np = V_b * Gamma*E_particle - n_1*P_1 - n_2*P_2 - n_3*P_3;
			
			// P_i = P_i + np * n_i * sqrt(1-sqrt(1+Δm^2/np^2))
			P_1 = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_1, P_1);
			P_2 = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_2, P_2);
			P_3 = fma(np * (1-sqrt(1+Delta_M2/pow(np, 2))), n_3, P_3);
			dP[gid] = Gamma * np * (1.-sqrt(1.+Delta_M2/pow(np, 2.)));
			
			if (M_in < M_out) {
				M[gid] = M_in;
				}
			else {
				M[gid] = M_out;
				}
			
			// E_particle = sqrt(M[gid]*M[gid] + P_1*P_1 + P_2*P_2 + P_3*P_3);
			if (M_in < M_out){
				E_particle = sqrt(fma(P_1, P_1, fma(P_2, P_2, fma(P_3, P_3, pow(M_in, 2)))));
			}
			else {
				E_particle = sqrt(fma(P_1, P_1, fma(P_2, P_2, fma(P_3, P_3, pow(M_out, 2)))));
			}
			
			// Update velocity vector
			V_1 = P_1 / E_particle;
			V_2 = P_2 / E_particle;
			V_3 = P_3 / E_particle;
			// Movement after the bubble interaction
			//X_1 = fma(V_1, Dt - timeToWall, X_1);
			//X_2 = fma(V_2, Dt - timeToWall, X_2);
			//X_3 = fma(V_3, Dt - timeToWall, X_3);
			moveLinear(&X_1, &X_2, &X_3, V_1, V_2, V_3, Dt - timeToWall);
			interactedFalse[gid] += 0;
			passedFalse[gid] += 0;
			interactedTrue[gid] += 1;
		}
	}
	// dE[gid] = E_particle - E_particle_old;
	X[3*gid] = X_1;
	X[3*gid+1] = X_2;
	X[3*gid+2] = X_3;
	
	P[3*gid] = P_1;
	P[3*gid+1] = P_2;
	P[3*gid+2] = P_3;
	
	E[gid] = E_particle;
}
