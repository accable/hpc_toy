#include <iostream>
#include <cmath>
#include <arm_neon.h>  // So we can use half precision numbers (oh and this also requires ARM CPU so if you don't have it, lol)

// Our task is to take a double precision number, split it into 4 half precision numbers, and reconstruct it back

struct fp16_4 {__fp16 x0, x1,  x2, x3;};  // Struct values are half-precision

// Splitting alg
inline fp16_4 split_to_4_fp16(double x){
        fp16_4 out{};
        double curr = x;  // We're keeping the original value via x

        out.x0 = static_cast<__fp16>(curr);  // We turn the current value to __half precision and save it on the struct 
        curr -= static_cast<double>(out.x0);  // We then subtract the original x with the extracted half-precision number

        out.x1 = static_cast<__fp16>(curr);
        curr -= static_cast<double>(out.x1);

        out.x2 = static_cast<__fp16>(curr);
        curr -= static_cast<double>(out.x2);

        out.x3 = static_cast<__fp16>(curr);

        return out;
}


// Reconstruct
inline double reconstruct_fp16_4(const fp16_4& p) {
    return static_cast<double>(p.x0)
         + static_cast<double>(p.x1)
         + static_cast<double>(p.x2)
         + static_cast<double>(p.x3);
}


int main(){
        double val = 1.23456789123456;  // Test number

        std::cout.precision(15);  // To get the double precision numbers
        std::cout << "Original number is: " << val << std::endl;

        // We now split the nuumbers into 4 fp16's
        fp16_4 halves = split_to_4_fp16(val);
        double reconstructed = reconstruct_fp16_4(halves);
        std::cout << "Reconstructed number is: " << reconstructed << std::endl;

        // Check the difference
        double diff = std::abs(val - reconstructed);
        double relative_diff = diff / std::abs(val);
        std::cout << "Absolute difference: " << diff << std::endl;
        std::cout << "Relative difference: " << relative_diff << std::endl;
        return 0;
}