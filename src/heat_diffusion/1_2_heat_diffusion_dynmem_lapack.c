/* This is a 2D heat transfer simulation using finite-difference method (FDM) made as a toy problem.
* Code is referenced from 1_0_heat_diffusion.c and refactored for NVPL compatibility and rather 
* calculating the derivative manually, we used LAPACK. Some comments re removed for more elaboration
* with what is changed from the original code and the difference between using NVPL and regular C.
*
* For this case, we demonstrated that using LAPACK to solve the heat equation is probable, but with 
* extremely high memory usage (> 120GB). Please proceed with extreme caution.
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <sys/resource.h> 
#include "nvpl_blas_cblas.h" 
#include <nvpl_lapack.h>			// LAPACK


// Heat transfer simulation
double hte(const nvpl_int_t steps){

	// Begin timer
	double time_start;
	time_start = clock();

	// Floating variables
	const double alpha_squared = 0.01;
	const double dt = 9e-7;

	// World variables 
	// Please do not change anything
	const nvpl_int_t domain_length = 1;  
	const nvpl_int64_t domain_interval = 500;  
	const double hotspot_length = 0.4;  
	
	// You can change this one tho...
	const double ambient_temp = 25.0;
	const double hotspot_temp = 75.0;

	// World-related variables
	const nvpl_int64_t world_size = domain_interval * domain_interval;
	const double delta_x = domain_length / domain_interval;
	const double r = alpha_squared / (delta_x * delta_x);

	// Dynamic memory for matrices and LAPACK indices
	float * temperature_map = NULL;
	float * temperature_map_rhs = NULL;  	// rhs
	float * new_temperature_map = NULL;
	nvpl_int_t * ipiv = NULL;  				// lapack indices

	// Dynamic memory allocation
	temperature_map = (float *)malloc(world_size * sizeof(float));
	temperature_map_rhs = (float *)malloc(world_size * sizeof(float));  
	new_temperature_map = (float *)malloc(world_size * world_size * sizeof(float));  // double the domain_interval since lapack
	ipiv = (nvpl_int_t *)malloc(world_size * sizeof(nvpl_int_t));  // lapack indices

	// Defining the temperature map
	const nvpl_int_t hotspot_array_length = domain_interval * (hotspot_length / domain_length);  // Assigning the "hot" area
	const nvpl_int_t hotspot_lower_bound = (domain_interval / 2) - (hotspot_array_length / 2);  // The "not hot" area
	const nvpl_int_t hotspot_upper_bound = (domain_interval / 2) + (hotspot_array_length / 2);  // Ditto

	// Row major, because C
	for (nvpl_int_t i = 0; i < domain_interval; i++){ 
		for (nvpl_int_t j = 0; j < domain_interval; j++){ 
			if ((i >= hotspot_lower_bound && i <= hotspot_upper_bound) && 
			    (j >= hotspot_lower_bound && j <= hotspot_upper_bound)){  
				temperature_map[i * domain_interval + j] = hotspot_temp;  // It's hot
			} else {
				temperature_map[i * domain_interval + j] = ambient_temp;  // It's cold 
			}
		}
	}

	// Assigning neumann boundary
	for (nvpl_int_t i = 0; i < world_size; ++i) {
        	nvpl_int_t row = i / domain_interval;
        	nvpl_int_t col = i % domain_interval;
        	new_temperature_map[i * world_size + i] = 1 + 4 * r * dt;

        	if (row > 0){      
				new_temperature_map[i * world_size+ i - domain_interval] = -r * dt;  	// Top
			} else {              
				new_temperature_map[i * world_size + i] += -r * dt;     				// Neumann
			}

        	if (row < domain_interval - 1){  
				new_temperature_map[i * world_size + i + domain_interval] = -r * dt;  	// Bottom
        	} else {              
				new_temperature_map[i * world_size + i] += -r * dt;     				// Neumann
			}

        	if (col > 0){    
				new_temperature_map[i * world_size + i - 1] = -r * dt;  				// Left
			} else {            
				new_temperature_map[i * world_size + i] += -r * dt;     				// Neumann
			}

        	if (col < domain_interval - 1){  
				new_temperature_map[i * world_size + i + 1] = -r * dt;  				// Right
			} else {              
				new_temperature_map[i * world_size + i] += -r * dt;     				// Neumann
    		}
	}


	// Computing timesteps w/ LAPACK
	for (nvpl_int_t i = 0; i < steps; i++){
		memcpy(temperature_map_rhs, temperature_map, world_size * sizeof(double));
		
		// LAPACK operations
		LAPACKE_sgesv(LAPACK_ROW_MAJOR, world_size, 1, new_temperature_map, world_size, ipiv, temperature_map_rhs, 1);

		double *temp = temperature_map;
		temperature_map = temperature_map_rhs;
		temperature_map_rhs = temp;

	}

	// End timer
	double elapsed_time;
	elapsed_time = (clock() - time_start)/CLOCKS_PER_SEC;
	
	// Memory cleanup
	free(temperature_map);
	free(temperature_map_rhs);
	free(new_temperature_map);
	free(ipiv);

	return elapsed_time;

}

// Memory profiling functions
long get_peak_memory_kb(){
	
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);
	return usage.ru_maxrss;
}


// Main
int main(){
	
	const nvpl_int_t steps = 10;
	double time_elapsed = hte(steps);
	long peak_mem = get_peak_memory_kb();

	printf("Simulation finished in: %f\n", time_elapsed);
	printf("Peak memory usage: %ld KB (%.2f MB)\n", peak_mem, peak_mem / 1024.0);
	
	return 0;
}

