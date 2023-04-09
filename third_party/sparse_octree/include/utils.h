#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
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


// uint64_t morton3D(float x, float y, float z)
// {
//     x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
//     y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
//     z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
//     uint64_t xx = deinterleave(x);
//     uint64_t yy = deinterleave(y);
//     uint64_t zz = deinterleave(z);
//     return xx * 4 + yy * 2 + zz;
// }

// uint64_t deinterleave(float x)
// {
//     uint32_t xx = (uint32_t)x;
//     xx = (xx | (xx << 8)) & 0x00FF00FF;
//     xx = (xx | (xx << 4)) & 0x0F0F0F0F;
//     xx = (xx | (xx << 2)) & 0x33333333;
//     xx = (xx | (xx << 1)) & 0x55555555;
//     return xx;
// }
// =========================================
// -------------daryl code ---------------
// ----------Hilbert encoder/decoder ----
// ===========================================


uint64_t xy2d( uint64_t x, uint64_t y) {
    int rx, ry, s, d=0;
    int n = 16;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) {
                x = n-1 - x;
                y = n-1 - y;
            }
            swap(x, y);
        }
    }
    return d;
}

inline uint64_t hilbert_encode(uint64_t x, uint64_t y, uint64_t z)
{
    int xx = xy2d(x, y);
    return (xy2d(xx,z));
}

Point d2xy(int d) {
    int rx, ry, s, t=d;
    int n = 16;
    Point p;
    p.x = 0;
    p.y = 0;
    for (s=1; s<n; s*=2) {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        if (ry == 0) {
            if (rx == 1) {
                p.x = s-1 - p.x;
                p.y = s-1 - p.y;
            }
            swap(p.x, p.y);
        }
        p.x += s * rx;
        p.y += s * ry;
        t /= 4;
    }
    return p;
}

inline Eigen::Vector3i hilbert_decode(uint64_t index)
{
    Point p = d2xy(index);
    Point pp = d2xy(p.x);
    return {
        pp.x,
        pp.x,
        p.y};
}