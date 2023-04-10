#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <iostream>
#include <eigen3/Eigen/Dense>


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
            UntransposeBits(std::array<T, N> const &in)
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
            TransposeBits(std::array<T, N> const &in)
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
        IndexToPosition(
                std::array<T, N> const &in)
        {
            // First convert index to transpose.
            std::array<T, N> out(internal::TransposeBits(in));

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
        PositionToIndex(std::array<T, N> const &in)
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

            return internal::UntransposeBits(out);
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
    std::array<uint8_t, 3> a = hilbert::v1::IndexToPosition(tmp);
    return {a[0],a[1],a[2]};
}

// Convert 8 bit 3D position to 24 bit index using v1.
inline uint64_t hilbert_encode(const int x, const int y, const int z)
{
    std::array<uint8_t, 3> pos;
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;
    std::array<uint8_t, 3> tmp = hilbert::v1::PositionToIndex(pos);

    uint64_t idx = (uint32_t(tmp[0]) << 16) +
                   (uint32_t(tmp[1]) << 8) +
                   (uint32_t(tmp[2]));
    return idx;
}



// int main(int, char **)
// {
//     // std::array<uint8_t, 3> tmp;
//     for(int i=0;i<255;i++){

//             // tmp[0] = i;
//             // tmp[1] = i;
//             // tmp[2] = i;
//             printf(" 原始position= (%i , %i, %i) \n", i, i, i);
//             uint64_t tmp2 = hilbert_encode(i,i,i);  
//             printf(" index= %i  \n", tmp2);
//             Eigen::Vector3i tmp3 = hilbert_decode(tmp2);
//             printf("解码position: (%i , %i, %i)\n",tmp3[0],tmp3[1],tmp3[2]);
//             if(i != tmp3[0] )
//                 {
//                     printf("FAIL: %i != %i\n",i, tmp3[0]);
//                     break;
//                 }
        
//     }
//     printf("success! \n");
// }


