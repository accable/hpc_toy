/* This is a 2D heat transfer simulation using finite-difference method (FDM) made as a toy problem.
* Code is referenced from https://github.com/eddireeder/heat-transfer-simulation and refactored for 
* performance measurement purposes.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>  			// For timekeeping
#include <sys/resource.h>  	// For memory usage information


// Heat transfer simulation
double hte(const int steps){

	// Begin timer
	double time_start;
	time_start = clock();

	// Floating variables
	const float alpha_squared = 0.01;
	const float rate_of_change = 9e-7;

	// World variables
	const float domain_length = 1;  	// n * n surface (higher is bigger)
	const int domain_interval = 5000;  	// Divided into n * n intervals (higher is denser)
	const float hotspot_length = 0.4;  	// How big is the hotspot
	const float ambient_temp = 25.0;
	const float hotspot_temp = 75.0;

	// Calculating the distance between intervals
	const double delta_x = domain_length / domain_interval;
	const double delta_y = delta_x;

	// Initialize and filling the temperature map
	float temperature_map[domain_interval][domain_interval];  								// n * n matrix
	const int hotspot_array_length = domain_interval * (hotspot_length / domain_length);  	// The "hot" area
	const int hotspot_lower_bound = (domain_interval / 2) - (hotspot_array_length / 2);  	// The "not hot" area
	const int hotspot_upper_bound = (domain_interval / 2) + (hotspot_array_length / 2);  	// Ditto

	for (int i = 0; i < domain_interval; i++){ 
		for (int j = 0; j < domain_interval; j++){
			// If the loop index is above the lower bound but below the upper bound
			if ((i >= hotspot_lower_bound && i <= hotspot_upper_bound) && 
			    (j >= hotspot_lower_bound && j <= hotspot_upper_bound)){  
				temperature_map[i][j] = hotspot_temp;  // It's hot
			} else {
				temperature_map[i][j] = ambient_temp;  // It's cold 
			}
		}
	}

	// Defining a new array to store the updated values rather than spamming the thing repeatedly during timestep
	float new_temperature_map[domain_interval][domain_interval];

	// Computing timesteps
	for (int i = 0; i < steps; i++){
		// Assigning surrounding temperatures and applying Neumann boundary condition
		// Will be replaced with a more elegant solution
		for (int j = 0; j < domain_interval; j++){
			for (int k = 0; k < domain_interval; k++){
				float left_temp, right_temp, top_temp, bottom_temp;
				
				// Left area
				if (j == 0){  									// When on most left edge
					left_temp = temperature_map[j][k];  		// Use self
				} else {
					left_temp = temperature_map[j - 1][k];  	// Use neighbor
				}

				// Right area
				if (j == (domain_interval - 1)){  				// When on most right edge
					right_temp = temperature_map[j][k];  		// Use self
				} else {
					right_temp = temperature_map[j + 1][k];  	// Use neighbor
				}

				// Top area
				if (k == 0){  									// When on most top
				       top_temp = temperature_map[j][k];  		// Use self
				} else {
			 		top_temp = temperature_map[j][k - 1];  		// Use neighbor
				}

				// Bottom area 
				if (k == (domain_interval - 1)){  				// When on most bottom
					bottom_temp = temperature_map[j][k];  		// Use self
				} else { 
					bottom_temp = temperature_map[j][k + 1];  	// Use neighbor
				}
				
				// Calculating derivatives
				float horizontal_derivative = alpha_squared * ((left_temp + right_temp - 2 * temperature_map[j][k]) / (delta_x * delta_x));
				float vertical_derivative = alpha_squared * ((top_temp + bottom_temp - 2 * temperature_map[j][k]) / (delta_y * delta_y));
				float total_derivative = horizontal_derivative + vertical_derivative;

				// Assigning new temperatures
				new_temperature_map[j][k] = temperature_map[j][k] + total_derivative * rate_of_change;
			}
		}

		// Assigning the new temperature map to the old temperature map by replacing the values
		// This was due to 
		for (int i = 0; i < domain_interval; i++){
			for (int j = 0; j < domain_interval; j++){
				temperature_map[i][j] = new_temperature_map[i][j];
			}
		}
	}

	// End timer
	double elapsed_time;
	elapsed_time = (clock() - time_start)/CLOCKS_PER_SEC;
	
	return elapsed_time;
}
	

// Memory profiling functions
// Because our cluster doesn't have any diagnostic tools
long get_peak_memory_kb(){
	
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);
	return usage.ru_maxrss;
}


int main(){
	
	const int steps = 100;
	double time_elapsed = hte(steps);
	long peak_mem = get_peak_memory_kb();

	printf("Simulation finished in: %f\n", time_elapsed);
	printf("Peak memory usage: %ld KB (%.2f MB)\n", peak_mem, peak_mem / 1024.0);
	
	return 0;
}
