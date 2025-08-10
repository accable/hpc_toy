#include <iostream>
#include <cmath>
#include <bit>  // To make FP21
#include <cstdint>

// Our task is to take a single precision number, split it into 3 FP21 numbers, and reconstruct it back

// For the context, please check this paper: https://dl.acm.org/doi/fullHtml/10.1145/3492805.3492813
// Original impl is in fortran but we ported it to C++. Might not be entirely accurate. But the numbers 
// we ran seems fine so should be correct.
struct fp21{
       // Storing the 21 bits for FP21
       uint32_t v = 0;

       static constexpr uint32_t F32_to_F21 = 0xFFFFF800u;
       static constexpr uint32_t F21_mask = 0x001FFFFFu;

       fp21() = default;  // Default constructor

       explicit fp21(float x){
               uint32_t bits = std::bit_cast<uint32_t>(x);  // Get the bits

               // Round to nearest even before chopping the rest of the bits
               bits += 0x00000400u;
               bits &= F32_to_F21;
               v = (bits >> 11) & F21_mask;
        }

       explicit operator float() const{
               uint32_t bits = (v & F21_mask) << 11;
               return std::bit_cast<float>(bits);
        }
};


// Now we split it to 3 Fp21
struct fp21_3 {fp21 x0; fp21 x1; fp21 x2;};
// Splitting alg
inline fp21_3 split_to_3_fp21(double x){
        fp21_3 out{};
        double curr = x;  // We're keeping the original value via x

        out.x0 = fp21(static_cast<float>(curr));  // We turn the current value to fp32 first then to fp21 precision and save it on the struct
        curr -= static_cast<double>(static_cast<float>(out.x0));  // We then subtract the original x with the extracted half-precision number

        out.x1 = fp21(static_cast<float>(curr));
        curr -= static_cast<double>(static_cast<float>(out.x1));

        out.x2 = fp21(static_cast<float>(curr));

        return out;
}


// Reconstruct
inline double reconstruct_fp21_3(const fp21_3& p) {
    return static_cast<double>(static_cast<float>(p.x0))
         + static_cast<double>(static_cast<float>(p.x1))
         + static_cast<double>(static_cast<float>(p.x2));
}


int main(){
        double val = 1.23456789123456;  // Test number

        std::cout.precision(15);  // To get the double precision numbers
        std::cout << "Original number is: " << val << std::endl;

        // We now turn it into FP21
        fp21_3 tries = split_to_3_fp21(val);
        float reconstructed = reconstruct_fp21_3(tries);
        std::cout << "Reconstructed number is: " << reconstructed << std::endl;

        // Check the difference
        double diff = std::abs(val - reconstructed);
        double relative_diff = diff / std::abs(val);
        std::cout << "Absolute difference: " << diff << std::endl;
        std::cout << "Relative difference: " << relative_diff << std::endl;
        return 0;
}