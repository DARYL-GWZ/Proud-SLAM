#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>


#define MAX_BITS 21
// #define SCALE_MASK ((uint64_t)0x1FF)
#define SCALE_MASK ((uint64_t)0x1)

using namespace std;

template <class T>
struct Vector3
{
    Vector3() : x(0), y(0), z(0) {}
    Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    Vector3<T> operator+(const Vector3<T> &b)
    {
        return Vector3<T>(x + b.x, y + b.y, z + b.z);
    }

    Vector3<T> operator-(const Vector3<T> &b)
    {
        return Vector3<T>(x - b.x, y - b.y, z - b.z);
    }

    T x, y, z;
};


struct Point {
  int x;
  int y;
};

typedef Vector3<int> Vector3i;
typedef Vector3<float> Vector3f;

/*
 * Mask generated with:
   MASK[0] = 0x7000000000000000,
   for(int i = 1; i < 21; ++i) {
   MASK[i] = MASK[i-1] | (MASK[0] >> (i*3));
   std::bitset<64> b(MASK[i]);
   std::cout << std::hex << b.to_ullong() << std::endl;
   }
 *
*/
constexpr uint64_t MASK[] = {
    0x7000000000000000,
    0x7e00000000000000,
    0x7fc0000000000000,
    0x7ff8000000000000,
    0x7fff000000000000,
    0x7fffe00000000000,
    0x7ffffc0000000000,
    0x7fffff8000000000,
    0x7ffffff000000000,
    0x7ffffffe00000000,
    0x7fffffffc0000000,
    0x7ffffffff8000000,
    0x7fffffffff000000,
    0x7fffffffffe00000,
    0x7ffffffffffc0000,
    0x7fffffffffff8000,
    0x7ffffffffffff000,
    0x7ffffffffffffe00,
    0x7fffffffffffffc0,
    0x7ffffffffffffff8,
    0x7fffffffffffffff};

inline uint64_t expand(unsigned long long value)
{
    uint64_t x = value & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline uint64_t compact(uint64_t value)
{
    uint64_t x = value & 0x1249249249249249;
    x = (x | x >> 2) & 0x10c30c30c30c30c3;
    x = (x | x >> 4) & 0x100f00f00f00f00f;
    x = (x | x >> 8) & 0x1f0000ff0000ff;
    x = (x | x >> 16) & 0x1f00000000ffff;
    x = (x | x >> 32) & 0x1fffff;
    return x;
}

inline uint64_t compute_morton(uint64_t x, uint64_t y, uint64_t z)
{
    uint64_t code = 0;

    x = expand(x);
    y = expand(y) << 1;
    z = expand(z) << 2;

    code = x | y | z;
    return code;
}

inline Eigen::Vector3i decode(const uint64_t code)
{
    return {
        compact(code >> 0ull),
        compact(code >> 1ull),
        compact(code >> 2ull)};
}

inline uint64_t encode(const int x, const int y, const int z)
{
    return (compute_morton(x, y, z) & MASK[MAX_BITS - 1]);
}



// =========================================
// -------------daryl code ---------------
// ----------Hilbert encoder/decoder ----
// ===========================================

namespace hilbert
{
    namespace v1
    {
        namespace internal
        {
            // Extract bits from transposed form.
            //
            // e.g.
            //
            // a d g j    a b c d
            // b e h k -> e f g h
            // c f i l    i j k l
            //
            template<typename T, size_t N>
            std::array<T, N>
            UntransposeBits1(std::array<T, N> const &in)
            {
                const size_t bits = std::numeric_limits<T>::digits;
                const T high_bit(T(1) << (bits - 1));
                const size_t bit_count(bits * N);

                std::array<T, N> out;

                std::fill(out.begin(), out.end(), 0);

                // go through all bits in input, msb first.  Shift distances are
                // from msb.
                for(size_t b=0;b<bit_count;b++)
                {
                    size_t src_bit, dst_bit, src, dst;
                    src = b % N;
                    dst = b / bits;
                    src_bit = b / N;
                    dst_bit = b % bits;

                    out[dst] |= (((in[src] << src_bit) & high_bit) >> dst_bit);
                }

                return out;
            }

            // Pack bits into transposed form.
            //
            // e.g.
            //
            // a b c d    a d g j
            // e f g h -> b e h k
            // i j k l    c f i l
            //
            template<typename T, size_t N>
            std::array<T, N>
            TransposeBits1(std::array<T, N> const &in)
            {
                const size_t bits = std::numeric_limits<T>::digits;
                const T high_bit(T(1) << (bits - 1));
                const size_t bit_count(bits * N);

                std::array<T, N> out;

                std::fill(out.begin(), out.end(), 0);

                // go through all bits in input, msb first.  Shift distances
                // are from msb.
                for(size_t b=0;b<bit_count;b++)
                {
                    size_t src_bit, dst_bit, src, dst;
                    src = b / bits;
                    dst = b % N;
                    src_bit = b % bits;
                    dst_bit = b / N;

                    out[dst] |= (((in[src] << src_bit) & high_bit) >> dst_bit);
                }

                return out;
            }
        }

        //
        // Public interfaces.
        //

        // Find the position of a point on an N dimensional Hilbert Curve.
        //
        // Based on the paper "Programming the Hilbert Curve" by John Skilling.
        //
        // Index is encoded with most significant objects first.  Lexographic
        // sort order.
        template<typename T, size_t N>
        std::array<T, N>
        IndexToPosition1(
                std::array<T, N> const &in)
        {
            // First convert index to transpose.
            std::array<T, N> out(internal::TransposeBits1(in));

            // Initial gray encoding of transposed vector.
            {
                T tmp = out[N-1] >> 1;

                for(size_t n=N-1;n;n--)
                {
                    out[n]^= out[n-1];
                }

                out[0]^= tmp;
            }

            // Apply transforms to gray code.
            {
                T cur_bit(2),
                  low_bits;

                while(cur_bit)
                {
                    low_bits = cur_bit - 1;

                    size_t n(N);

                    do
                    {
                        n--;
                        if(out[n] & cur_bit)
                        {
                            // flip low bits of X
                            out[0]^= low_bits;
                        }
                        else
                        {
                            // swap low bits with X
                            T t((out[n] ^ out[0]) & low_bits);
                            out[n]^= t;
                            out[0]^= t;
                        }
                    }
                    while(n);

                    cur_bit<<= 1;
                }
            }

            return out;
        }

        // Find the index of a point on an N dimensional Hilbert Curve.
        //
        // Based on the paper "Programming the Hilbert Curve" by John Skilling.
        //
        // Index is encoded with most significant objects first.  Lexographic
        // sort order.
        template<typename T, size_t N>
        std::array<T, N>
        PositionToIndex1(std::array<T, N> const &in)
        {
            const size_t bits = std::numeric_limits<T>::digits;

            std::array<T, N> out(in);

            // reverse transforms to convert into transposed gray code.
            {
                T cur_bit(T(1) << (bits - 1)),
                  low_bits;

                do
                {
                    low_bits = cur_bit - 1;

                    for(size_t n=0;n<N;n++)
                    {
                        if(out[n] & cur_bit)
                        {
                            // flip low bits of X
                            out[0]^= low_bits;
                        }
                        else
                        {
                            // swap low bits with X
                            T t((out[n] ^ out[0]) & low_bits);
                            out[n]^= t;
                            out[0]^= t;
                        }
                    }

                    cur_bit>>= 1;
                } while(low_bits > 1);
            }

            // Remove gray code from transposed vector.
            {
                T cur_bit(T(1) << (bits - 1)),
                  t(0);

                for(size_t n=1;n<N;n++)
                {
                    out[n]^= out[n-1];
                }

                do
                {
                    if(out[N-1] & cur_bit)
                    {
                        t^= (cur_bit - 1);
                    }
                    cur_bit>>= 1;
                } while(cur_bit > 1);

                for(auto &v : out)
                {
                    v^= t;
                }
            }

            return internal::UntransposeBits1(out);
        }
    } // namespace v1
} // namespace hilbert
inline Eigen::Vector3i hilbert_decode(uint32_t idx)
// std::array<uint8_t, 3> hilbert_decode(uint32_t idx)
{
    std::array<uint8_t, 3> tmp;

    tmp[0] = uint8_t((idx >> 16) & 0xff);
    tmp[1] = uint8_t((idx >> 8) & 0xff);
    tmp[2] = uint8_t(idx & 0xff);
    std::array<uint8_t, 3> a = hilbert::v1::IndexToPosition1(tmp);
    return {a[0],a[1],a[2]};
}

// Convert 8 bit 3D position to 24 bit index using v1.
inline uint64_t hilbert_encode(const int x, const int y, const int z)
{
    std::array<uint8_t, 3> pos;
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;
    std::array<uint8_t, 3> tmp = hilbert::v1::PositionToIndex1(pos);

    uint64_t idx = (uint32_t(tmp[0]) << 16) +
                   (uint32_t(tmp[1]) << 8) +
                   (uint32_t(tmp[2]));
    return idx;
}

