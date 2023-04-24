// #include <algorithm>
// #include <array>
// #include <cstdint>
// #include <limits>
// #include <type_traits>
// #include <iostream>
// #include <eigen3/Eigen/Dense>


// namespace hilbert
// {
//     namespace v1
//     {
//         namespace internal
//         {
//             // Extract bits from transposed form.
//             //
//             // e.g.
//             //
//             // a d g j    a b c d
//             // b e h k -> e f g h
//             // c f i l    i j k l
//             //
//             template<typename T, size_t N>
//             std::array<T, N>
//             UntransposeBits(std::array<T, N> const &in)
//             {
//                 const size_t bits = std::numeric_limits<T>::digits;
//                 const T high_bit(T(1) << (bits - 1));
//                 const size_t bit_count(bits * N);

//                 std::array<T, N> out;

//                 std::fill(out.begin(), out.end(), 0);

//                 // go through all bits in input, msb first.  Shift distances are
//                 // from msb.
//                 for(size_t b=0;b<bit_count;b++)
//                 {
//                     size_t src_bit, dst_bit, src, dst;
//                     src = b % N;
//                     dst = b / bits;
//                     src_bit = b / N;
//                     dst_bit = b % bits;

//                     out[dst] |= (((in[src] << src_bit) & high_bit) >> dst_bit);
//                 }

//                 return out;
//             }

//             // Pack bits into transposed form.
//             //
//             // e.g.
//             //
//             // a b c d    a d g j
//             // e f g h -> b e h k
//             // i j k l    c f i l
//             //
//             template<typename T, size_t N>
//             std::array<T, N>
//             TransposeBits(std::array<T, N> const &in)
//             {
//                 const size_t bits = std::numeric_limits<T>::digits;
//                 const T high_bit(T(1) << (bits - 1));
//                 const size_t bit_count(bits * N);

//                 std::array<T, N> out;

//                 std::fill(out.begin(), out.end(), 0);

//                 // go through all bits in input, msb first.  Shift distances
//                 // are from msb.
//                 for(size_t b=0;b<bit_count;b++)
//                 {
//                     size_t src_bit, dst_bit, src, dst;
//                     src = b / bits;
//                     dst = b % N;
//                     src_bit = b % bits;
//                     dst_bit = b / N;

//                     out[dst] |= (((in[src] << src_bit) & high_bit) >> dst_bit);
//                 }

//                 return out;
//             }
//         }

//         //
//         // Public interfaces.
//         //

//         // Find the position of a point on an N dimensional Hilbert Curve.
//         //
//         // Based on the paper "Programming the Hilbert Curve" by John Skilling.
//         //
//         // Index is encoded with most significant objects first.  Lexographic
//         // sort order.
//         template<typename T, size_t N>
//         std::array<T, N>
//         IndexToPosition(
//                 std::array<T, N> const &in)
//         {
//             // First convert index to transpose.
//             std::array<T, N> out(internal::TransposeBits(in));

//             // Initial gray encoding of transposed vector.
//             {
//                 T tmp = out[N-1] >> 1;

//                 for(size_t n=N-1;n;n--)
//                 {
//                     out[n]^= out[n-1];
//                 }

//                 out[0]^= tmp;
//             }

//             // Apply transforms to gray code.
//             {
//                 T cur_bit(2),
//                   low_bits;

//                 while(cur_bit)
//                 {
//                     low_bits = cur_bit - 1;

//                     size_t n(N);

//                     do
//                     {
//                         n--;
//                         if(out[n] & cur_bit)
//                         {
//                             // flip low bits of X
//                             out[0]^= low_bits;
//                         }
//                         else
//                         {
//                             // swap low bits with X
//                             T t((out[n] ^ out[0]) & low_bits);
//                             out[n]^= t;
//                             out[0]^= t;
//                         }
//                     }
//                     while(n);

//                     cur_bit<<= 1;
//                 }
//             }

//             return out;
//         }

//         // Find the index of a point on an N dimensional Hilbert Curve.
//         //
//         // Based on the paper "Programming the Hilbert Curve" by John Skilling.
//         //
//         // Index is encoded with most significant objects first.  Lexographic
//         // sort order.
//         template<typename T, size_t N>
//         std::array<T, N>
//         PositionToIndex(std::array<T, N> const &in)
//         {
//             const size_t bits = std::numeric_limits<T>::digits;

//             std::array<T, N> out(in);

//             // reverse transforms to convert into transposed gray code.
//             {
//                 T cur_bit(T(1) << (bits - 1)),
//                   low_bits;

//                 do
//                 {
//                     low_bits = cur_bit - 1;

//                     for(size_t n=0;n<N;n++)
//                     {
//                         if(out[n] & cur_bit)
//                         {
//                             // flip low bits of X
//                             out[0]^= low_bits;
//                         }
//                         else
//                         {
//                             // swap low bits with X
//                             T t((out[n] ^ out[0]) & low_bits);
//                             out[n]^= t;
//                             out[0]^= t;
//                         }
//                     }

//                     cur_bit>>= 1;
//                 } while(low_bits > 1);
//             }

//             // Remove gray code from transposed vector.
//             {
//                 T cur_bit(T(1) << (bits - 1)),
//                   t(0);

//                 for(size_t n=1;n<N;n++)
//                 {
//                     out[n]^= out[n-1];
//                 }

//                 do
//                 {
//                     if(out[N-1] & cur_bit)
//                     {
//                         t^= (cur_bit - 1);
//                     }
//                     cur_bit>>= 1;
//                 } while(cur_bit > 1);

//                 for(auto &v : out)
//                 {
//                     v^= t;
//                 }
//             }

//             return internal::UntransposeBits(out);
//         }
//     } // namespace v1
// } // namespace hilbert
// inline Eigen::Vector3i hilbert_decode(uint32_t idx)
// // std::array<uint8_t, 3> hilbert_decode(uint32_t idx)
// {
//     std::array<uint8_t, 3> tmp;

//     tmp[0] = uint8_t((idx >> 16) & 0xff);
//     tmp[1] = uint8_t((idx >> 8) & 0xff);
//     tmp[2] = uint8_t(idx & 0xff);
//     std::array<uint8_t, 3> a = hilbert::v1::IndexToPosition(tmp);
//     return {a[0],a[1],a[2]};
// }

// // Convert 8 bit 3D position to 24 bit index using v1.
// inline uint64_t hilbert_encode(const int x, const int y, const int z)
// {
//     std::array<uint8_t, 3> pos;
//     pos[0] = x;
//     pos[1] = y;
//     pos[2] = z;
//     std::array<uint8_t, 3> tmp = hilbert::v1::PositionToIndex(pos);

//     uint64_t idx = (uint32_t(tmp[0]) << 16) +
//                    (uint32_t(tmp[1]) << 8) +
//                    (uint32_t(tmp[2]));
//     return idx;
// }



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


#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 定义空间点
struct Point {
    double x, y, z;
};

// 定义子空间
struct Subspace {
    Point center;
    double length;
    Point point;
};

// 将正方体空间分成8个子空间
vector<Subspace> divideSpace(Point center, double length) {
    vector<Subspace> subspaces;
    double half_length = length / 2.0;
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            for(int k=0; k<2; k++) {
                Subspace subspace;
                subspace.center.x = center.x + (i-0.5)*half_length;
                subspace.center.y = center.y + (j-0.5)*half_length;
                subspace.center.z = center.z + (k-0.5)*half_length;
                subspace.point.x = 0.0;
                subspace.point.y = 0.0;
                subspace.point.z = 0.0;
                subspace.length = half_length;
                subspaces.push_back(subspace);
            }
        }
    }
    return subspaces;
}

// 判断点是否在指定子空间中
bool isInSubspace(Point point, Subspace subspace) {
    double half_length = subspace.length / 2.0;
    if(point.x > subspace.center.x-half_length && point.x < subspace.center.x+half_length &&
       point.y > subspace.center.y-half_length && point.y < subspace.center.y+half_length &&
       point.z > subspace.center.z-half_length && point.z < subspace.center.z+half_length) {
        return true;
    }
    return false;
}

int main() {
    // 正方体空间中心和边长
    Point center = {0.0, 0.0, 0.0};
    double length = 10.0;

    // 将空间分成8个子空间
    vector<Subspace> subspaces = divideSpace(center, length);

    // 遍历200个空间点，将每个点放入对应的子空间
    // vector<Subspace> subspacesWithPoints;
    // for(int i=0; i<8; i++) {
    //     subspacesWithPoints.push_back(subspaces[i]);
    // }
    for(int i=0; i<20; i++) {

        // 随机生成一个点
        Point point = { rand() % 10 - 5, rand() % 10 - 5, rand() % 10 - 5};
        printf("point: (%f,%f,%f)\n" ,point.x, point.y,point.z);

        // 判断该点属于哪个子空间
        for(int j=0; j<8; j++) {
            // printf("%i\n" ,isInSubspace(point, subspaces[j]));
            if(isInSubspace(point, subspaces[j])) {
                // 如果该子空间中已经有点，则替换掉该点
                // std::cout << "subspaces.point: " << subspaces[j].point.x<< ","<< subspaces[j].point.y<< ","<< subspaces[j].point.z<< std::endl;
                // printf("subspaces.point: (%f,%f,%f)\n" ,subspaces[j].point.x, subspaces[j].point.y,subspaces[j].point.z);
                // printf("判断: (%f)\n" , subspaces[j].point.x == 0.0 || subspaces[j].point.y != 0.0 || subspaces[j].point.z != 0.0);
                if(subspaces[j].point.x != 0.0 || subspaces[j].point.y != 0.0 || subspaces[j].point.z != 0.0) {
                    subspaces[j].point = point;
                    printf("替换 point: (%f,%f,%f)\n" ,point.x, point.y,point.z);
                }
                // 否则将该点存入对应的子空间中
                else {
                    subspaces[j].point = point;
                    printf("新 point: (%f,%f,%f)\n" ,point.x, point.y,point.z);
                    break;
                }
            }
        }
    }
    printf("结束\n" );
    return 0;
}


void Octree::insert(torch::Tensor pts, torch::Tensor color, torch::Tensor pcd)
{
    // temporal solution
    all_pts.push_back(pts);
    all_colors.push_back(color);
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }
    // std::cout << "MAX_POINTS_PER_LEAF1: " << MAX_POINTS_PER_LEAF<< std::endl;

    auto points = pts.accessor<int, 2>();
    auto colors = color.accessor<int, 2>();
    auto pcds = pcd.accessor<float, 2>();

    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return;
    }
    if (colors.size(1) != 3)
    {
        std::cout << "Colors dimensions mismatch: inputs are " << colors.size(1) << " expect 3" << std::endl;
        return;
    }
    if (pcds.size(1) != 3)
    {
        std::cout << "pcd dimensions mismatch: inputs are " << pcds.size(1) << " expect 3" << std::endl;
        return;
    }
    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            all_keys.insert(key);
            const unsigned int shift = MAX_BITS - max_level_ - 1;
            auto n = root_;
            unsigned edge = size_ / 2;
            
            for (int d = 1; d <= max_level_; edge /= 2, ++d)
            {
                const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
                // std::cout << "Level: " << d << " ChildID: " << childid << std::endl;
                auto tmp = n->child(childid);
                // std::cout << "Level: " << d << " ChildID: " << childid << "tmp: " << tmp << std::endl;
                // std::cout << "tmp" << tmp << std::endl;
                if (!tmp)
                {
                    // 新的点不在现有八叉树中，创建新节点
                    const uint64_t code = key & MASK[d + shift];
                    const bool is_leaf = (d == max_level_);
                    // tmp = std::make_shared<Octant>();
                    tmp = new Octant();
                    tmp->code_ = code;
                    tmp->side_ = edge;
                    tmp->is_leaf_ = is_leaf;
                    tmp->type_ = is_leaf ? (j == 0 ? SURFACE : FEATURE) : NONLEAF;
                    // if (is_leaf) {
                    if (tmp->point_data_color.size() < MAX_POINTS_PER_LEAF){
                        // std::cout << "创建新节点"<< std::endl;
                        // float xyz = encode(pcds[i][0], pcds[i][1], pcds[i][2]);
                        float color = hilbert_encode(colors[i][0], colors[i][1], colors[i][2]);
                        tmp->point_data_x.push_back(pcds[i][0]);  
                        tmp->point_data_y.push_back(pcds[i][1]);  
                        tmp->point_data_z.push_back(pcds[i][2]);  
                        tmp->point_data_color.push_back(color); 
                    }
                    // }

                    n->children_mask_ = n->children_mask_ | (1 << childid);
                    n->child(childid) = tmp;
                }
                else
                {   
                    if (tmp->type_ == FEATURE && j == 0)
                        tmp->type_ = SURFACE;

                    // if (tmp->is_leaf_) {
                    if (tmp->point_data_color.size() < MAX_POINTS_PER_LEAF){
                        // std::cout << "tmp->point_data_xyz.size()" << tmp->point_data_xyz.size()<< std::endl;
                        // float xyz = encode(pcd[i][0], pcd[i][1], pcd[i][2])& MASK[d + shift];
                        // float color = encode(colors[i][0], colors[i][1], colors[i][2])& MASK[d + shift];
                        float color = hilbert_encode(colors[i][0], colors[i][1], colors[i][2]);
                        tmp->point_data_x.push_back(pcds[i][0]);  
                        tmp->point_data_y.push_back(pcds[i][1]);  
                        tmp->point_data_z.push_back(pcds[i][2]);  
                        tmp->point_data_color.push_back(color); 
                        }           
                }
                n = tmp;
            }
        }
    }
}
