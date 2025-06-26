/* This is a 2D heat transfer simulation using finite-difference method (FDM) made as a toy problem.
* Code is referenced from 1_0_heat_diffusion.c and refactored for NVPL compatibility. Some comments 
* are removed for more elaboration with what is changed from the original code and the difference 
* between using NVPL and regular C.
*
* For this case, we demonstrated how replacing datatypes and memory allocation allows the code to be 
* executed faster. Doing so gives us 50% faster execution time using the same configuration.
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>	
#include <sys/resource.h>  		
#include "nvpl_blas_cblas.h"  	// For NVPL compatibility since we are using its datatypes


// Heat transfer simulation
double hte(const nvpl_int_t steps){

	// Begin timer
	double time_start;
	time_start = clock();

	// Floating variables
	const double alpha_squared = 0.01;
	const double rate_of_change = 9e-7;

	// World variables
	const nvpl_int_t domain_length = 1;  	
	const nvpl_int64_t domain_interval = 5000;  // Needs to be int64 or it will overflow and segfault
	const double hotspot_length = 0.4;
	const double ambient_temp = 25.0;
	const double hotspot_temp = 75.0;

	// Calculating the distance between intervals
	const double delta_x = domain_length / domain_interval;
	const double delta_y = delta_x;

	// Dynamic memory initialization
	// Since this is now dynamic, we assume that the entire area is now a (n * n) length vector
	double * temperature_map = NULL;
	double * new_temperature_map = NULL;

	temperature_map = (double *)malloc(domain_interval * domain_interval * sizeof(double)); 
	new_temperature_map = (double *)malloc(domain_interval * domain_interval * sizeof(double));

	// Defining the temperature map
	const nvpl_int_t hotspot_array_length = domain_interval * (hotspot_length / domain_length);  
	const nvpl_int_t hotspot_lower_bound = (domain_interval / 2) - (hotspot_array_length / 2);  
	const nvpl_int_t hotspot_upper_bound = (domain_interval / 2) + (hotspot_array_length / 2); 

	// Row major, because C
	for (nvpl_int_t i = 0; i < domain_interval; i++){  
		for (nvpl_int_t j = 0; j < domain_interval; j++){  
			if ((i >= hotspot_lower_bound && i <= hotspot_upper_bound) && 
			    (j >= hotspot_lower_bound && j <= hotspot_upper_bound)){  
				temperature_map[i * domain_interval + j] = hotspot_temp;  
			} else {
				temperature_map[i * domain_interval + j] = ambient_temp; 
			}
		}
	}
	
	// Computing timesteps
	for (nvpl_int_t i = 0; i < steps; i++){
		for (nvpl_int_t j = 0; j < domain_interval; j++){
			for (nvpl_int_t k = 0; k < domain_interval; k++){
				double left_temp, right_temp, top_temp, bottom_temp;
				
				// Left area
				if (j == 0){  
					left_temp = temperature_map[j * domain_interval + k];  
				} else {
					left_temp = temperature_map[(j - 1) * domain_interval + k];  
				}

				// Right area
				if (j == (domain_interval - 1)){  
					right_temp = temperature_map[j * domain_interval + k];  
				} else {
					right_temp = temperature_map[(j + 1) * domain_interval + k];  
				}

				// Top area
				if (k == 0){  
				       top_temp = temperature_map[j * domain_interval + k];  
				} else {
			 		top_temp = temperature_map[j * domain_interval + (k - 1)];  
				}

				// Bottom area 
				if (k == (domain_interval - 1)){  
					bottom_temp = temperature_map[j * domain_interval + k];  
				} else { 
					bottom_temp = temperature_map[j * domain_interval + (k + 1)];  
				}
				
				// Calculating derivatives
				double horizontal_derivative = alpha_squared * ((left_temp + right_temp - 2 * temperature_map[j * domain_interval + k]) / (delta_x * delta_x));
				double vertical_derivative = alpha_squared * ((top_temp + bottom_temp - 2 * temperature_map[j * domain_interval + k]) / (delta_y * delta_y));
				double total_derivative = horizontal_derivative + vertical_derivative;

				// New temperatures
				new_temperature_map[j * domain_interval + k] = temperature_map[j * domain_interval + k] + total_derivative * rate_of_change;
			}
		}

		double *temp = temperature_map;
		temperature_map = new_temperature_map;
		new_temperature_map = temp;

	}

	// End timer
	double elapsed_time;
	elapsed_time = (clock() - time_start)/CLOCKS_PER_SEC;
	
	// Memory cleanup
	free(temperature_map);
	free(new_temperature_map);

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
	
	const nvpl_int_t steps = 100;
	double time_elapsed = hte(steps);
	long peak_mem = get_peak_memory_kb();

	printf("Simulation finished in: %f\n", time_elapsed);
	printf("Peak memory usage: %ld KB (%.2f MB)\n", peak_mem, peak_mem / 1024.0);
	
	return 0;
}
