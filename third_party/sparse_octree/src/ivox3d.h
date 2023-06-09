//
// Created by xiang on 2021/9/16.
//

#ifndef FASTER_LIO_IVOX3D_H
#define FASTER_LIO_IVOX3D_H

#include <glog/logging.h>
#include <execution>
#include <list>
#include <thread>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "eigen_types.h"
#include "ivox3d_node.hpp"

// using PointType = pcl::PointXYZRGB;
namespace faster_lio {

enum class IVoxNodeType {
    DEFAULT,  // linear ivox
    PHC,      // phc ivox
};

/// traits for NodeType
template <IVoxNodeType node_type, typename PointT, int dim>
struct IVoxNodeTypeTraits {};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::DEFAULT, PointT, dim> {
    using NodeType = IVoxNode<PointT, dim>;
};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::PHC, PointT, dim> {
    using NodeType = IVoxNodePhc<PointT, dim>;
};

template <int dim = 3, IVoxNodeType node_type = IVoxNodeType::DEFAULT, typename PointType = pcl::PointXYZRGB>
class IVox {
   public:
    using KeyType = Eigen::Matrix<int, dim, 1>;
    using PtType = Eigen::Matrix<float, dim, 1>;
    using NodeType = typename IVoxNodeTypeTraits<node_type, PointType, dim*2>::NodeType;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using DistPoint = typename NodeType::DistPoint;

    enum class NearbyType {
        CENTER,  // center only
        NEARBY6,
        NEARBY18,
        NEARBY26,
    };

    struct Options {
        float resolution_ = 0.2;                        // ivox resolution
        float inv_resolution_ = 5.0;                   // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY6;  // nearby range
        std::size_t capacity_ = 1000000;                // capacity
    };

    /**
     * constructor
     * @param options  ivox options
     */
    explicit IVox(Options options) : options_(options) {
        options_.inv_resolution_ = 1.0 / options_.resolution_;
        GenerateNearbyGrids();
    }

    /**
     * add points
     * @param points_to_add
     */
    void AddPoints(const PointVector& points_to_add);
    // get voxel center
    torch::Tensor get_voxel_center();
    // get points
    std::tuple<torch::Tensor,torch::Tensor> get_points_colors();
    // get voxel center
    torch::Tensor get_points();
    /// get nn 根据一个点获得其1个近邻点
    bool GetClosestPoint(const PointType& pt, PointType& closest_pt);

    /// get knn 根据一个点获得其K个近邻点
    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 10, double max_range = 5.0);

    /// get knn in cloud 根据一组点获得其对应的1个近邻点
    bool GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud);
    
    /// get knn in cloud 根据一组点获得其对应的K个近邻点
    std::tuple<torch::Tensor,torch::Tensor> GetClosestPoint_d(const PointVector& cloud, int k_num = 10);


    /// get number of points
    size_t NumPoints() const;

    /// get number of valid grids
    size_t NumValidGrids() const;

    /// get statistics of the points
    std::vector<float> StatGridPoints() const;

   private:
    /// generate the nearby grids according to the given options
    void GenerateNearbyGrids();

    /// position to grid
    KeyType Pos2Grid(const PtType& pt) const;

    Options options_;
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, hash_vec<dim>>
        grids_map_;                                        // voxel hash map
    std::list<std::pair<KeyType, NodeType>> grids_cache_;  // voxel cache
    std::vector<KeyType> nearby_grids_;                    // nearbys
};

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointType& closest_pt) {
    std::cout << "intro func222 "  << std::endl;
    std::vector<DistPoint> candidates;
    auto key = Pos2Grid(ToEigen<float, dim>(pt));
    std::for_each(nearby_grids_.begin(), nearby_grids_.end(), [&key, &candidates, &pt, this](const KeyType& delta) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            DistPoint dist_point;
            bool found = iter->second->second.NNPoint(pt, dist_point);
            if (found) {
                candidates.emplace_back(dist_point);
            }
        }
    });

    if (candidates.empty()) {
        return false;
    }

    auto iter = std::min_element(candidates.begin(), candidates.end());
    closest_pt = iter->Get();
    return true;
}

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num,
                                                      double max_range) {
    // std::cout << "intro func "  << std::endl;
    std::vector<DistPoint> candidates;
    candidates.reserve(max_num * nearby_grids_.size());
    //  计算所属体素的key
    auto key = Pos2Grid(ToEigen<float, dim>(pt));

// #define INNER_TIMER
#ifdef INNER_TIMER
    static std::unordered_map<std::string, std::vector<int64_t>> stats;
    if (stats.empty()) {
        stats["knn"] = std::vector<int64_t>();
        stats["nth"] = std::vector<int64_t>();
    }
#endif
    //找到所有的邻居体素，并获得每个体素内的近邻点
    for (const KeyType& delta : nearby_grids_) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
#ifdef INNER_TIMER
            auto t1 = std::chrono::high_resolution_clock::now();
#endif
            auto tmp = iter->second->second.KNNPointByCondition(candidates, pt, max_num, max_range);
#ifdef INNER_TIMER
            auto t2 = std::chrono::high_resolution_clock::now();
            auto knn = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            stats["knn"].emplace_back(knn);
#endif
        }
    }

    if (candidates.empty()) {
        return false;
    }

#ifdef INNER_TIMER
    auto t1 = std::chrono::high_resolution_clock::now();
#endif
    //对所有候选近邻排序，得到最终的k个
    if (candidates.size() <= max_num) {
    } else {
        std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
        candidates.resize(max_num);
    }
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

#ifdef INNER_TIMER
    auto t2 = std::chrono::high_resolution_clock::now();
    auto nth = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    stats["nth"].emplace_back(nth);

    constexpr int STAT_PERIOD = 100000;
    if (!stats["nth"].empty() && stats["nth"].size() % STAT_PERIOD == 0) {
        for (auto& it : stats) {
            const std::string& key = it.first;
            std::vector<int64_t>& stat = it.second;
            int64_t sum_ = std::accumulate(stat.begin(), stat.end(), 0);
            int64_t num_ = stat.size();
            stat.clear();
            std::cout << "inner_" << key << "(ns): sum=" << sum_ << " num=" << num_ << " ave=" << 1.0 * sum_ / num_
                      << " ave*n=" << 1.0 * sum_ / STAT_PERIOD << std::endl;
        }
    }
#endif

    closest_pt.clear();
    for (auto& it : candidates) {
        closest_pt.emplace_back(it.Get());
    }
    return closest_pt.empty() == false;
}

template <int dim, IVoxNodeType node_type, typename PointType>
size_t IVox<dim, node_type, PointType>::NumValidGrids() const {
    return grids_map_.size();
}



template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY18) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0), KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1), KeyType(1, 0, -1), KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1), KeyType(0, 1, -1), KeyType(0, -1, -1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY26) {
        nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0),  KeyType(0, 0, -1),  KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0),  KeyType(1, -1, 0),  KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1),  KeyType(1, 0, -1),  KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1),  KeyType(0, 1, -1),  KeyType(0, -1, -1), KeyType(1, 1, 1),
                         KeyType(-1, 1, 1),  KeyType(1, -1, 1),  KeyType(1, 1, -1),  KeyType(-1, -1, 1),
                         KeyType(-1, 1, -1), KeyType(1, -1, -1), KeyType(-1, -1, -1)};
    } else {
        LOG(ERROR) << "Unknown nearby_type!";
    }
}

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud) {
    std::cout << "intro func111 "  << std::endl;
    std::vector<size_t> index(cloud.size());
    for (int i = 0; i < cloud.size(); ++i) {
        index[i] = i;
    }
    closest_cloud.resize(cloud.size());

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&cloud, &closest_cloud, this](size_t idx) {
        PointType pt;
        if (GetClosestPoint(cloud[idx], pt)) {
            closest_cloud[idx] = pt;
        } else {
            closest_cloud[idx] = PointType();
        }
    });
    return true;
}

template <int dim, IVoxNodeType node_type, typename PointType>
std::tuple<torch::Tensor,torch::Tensor> IVox<dim, node_type, PointType>::GetClosestPoint_d(const PointVector& cloud, int k_num) {
    // std::vector<size_t> index(cloud.size());
    // for (int i = 0; i < cloud.size(); ++i) {
    //     index[i] = i;
    // }
    int num = cloud.size();
    torch::Tensor all_points = torch::zeros({num,k_num, 3}, dtype(torch::kFloat32));
    torch::Tensor all_colors = torch::zeros({num,k_num, 3}, dtype(torch::kFloat32));
   
    torch::Tensor xyz_array = torch::zeros({k_num, 3}, dtype(torch::kFloat32));
    torch::Tensor color_array = torch::zeros({k_num, 3}, dtype(torch::kFloat32));
    // std::cout << "cloud.size() = " <<cloud.size() << std::endl;
    int j = 0;
    for(int idx = 0; idx < num; idx++) 
    {
        PointVector pt;
        // bool a = GetClosestPoint(cloud[idx], pt, k_num);
        // std::cout << "a = " <<a << std::endl;
        if (GetClosestPoint(cloud[idx], pt, k_num)) {
            // closest_cloud[idx] = pt;
            // std::cout << "pt.size() = " <<pt.size() << std::endl;
            for (int i = 0; i < pt.size(); ++i) {
                xyz_array[i][0]=pt[i].x;
                xyz_array[i][1]=pt[i].y;
                xyz_array[i][2]=pt[i].z;
                color_array[i][0]=pt[i].r;
                color_array[i][1]=pt[i].g;
                color_array[i][2]=pt[i].b;
                // std::cout << "i = " << i << std::endl;
            }
            all_points[j] = xyz_array; 
            all_colors[j] = color_array;  
            j++;
        } else {
            j++;
        }
    };
    return std::make_tuple(all_points, all_colors);
}

template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::AddPoints(const PointVector& points_to_add) {
    std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [this](const auto& pt) {
        // std::cout << "pt = (" << pt.x << ", " << pt.y<< ", " << pt.z<< " )"<< std::endl;
        auto key = Pos2Grid(ToEigen<float, dim>(pt));
        // std::cout << "insert key = (" << key[0] << ", " << key[1]<< ", " << key[2]<< " )"<< std::endl;

        auto iter = grids_map_.find(key);
        if (iter == grids_map_.end()) {
            PointType center;
            center.getVector3fMap() = key.template cast<float>() * options_.resolution_;
            // std::cout << "insert center = (" << center.x << ", " << center.y<< ", " << center.z<< " )"<< std::endl;
            grids_cache_.push_front({key, NodeType(center, options_.resolution_)});
            grids_map_.insert({key, grids_cache_.begin()});

            grids_cache_.front().second.InsertPoint(pt);

            if (grids_map_.size() >= options_.capacity_) {
                grids_map_.erase(grids_cache_.back().first);
                grids_cache_.pop_back();
            }
        } else {
            if(iter->second->second.Size() <= 10){
                iter->second->second.InsertPoint(pt);
                grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
                grids_map_[key] = grids_cache_.begin();
            }
        }
    });
}

template <int dim, IVoxNodeType node_type, typename PointType>
torch::Tensor IVox<dim, node_type, PointType>::get_voxel_center(){
    int num = NumValidGrids();
    torch::Tensor all_voxels = torch::zeros({num, 3}, dtype(torch::kFloat32));
    PointType center;
    int i = 0;
    for (auto& it : grids_cache_) {
        auto key = it.first;
        // std::cout << "get key = (" << key[0]<< ", " << key[1]<< ", " << key[2]<< " )"<< std::endl;
        center.getVector3fMap() = key.template cast<float>() * options_.resolution_;
        // std::cout << "get center = (" << center.x << ", " << center.y<< ", " << center.z<< " )"<< std::endl;
       
        std::vector<float> coords = {center.x, center.y,center.z};
        torch::Tensor voxel = torch::from_blob(coords.data(), {3}, dtype(torch::kFloat32));
        all_voxels[i] = voxel;
        i++;
    }	
    return all_voxels;
}

template <int dim, IVoxNodeType node_type, typename PointType>
size_t IVox<dim, node_type, PointType>::NumPoints() const {
    int valid_num = 0;
    for (auto& it : grids_cache_) {
        int s = it.second.Size();
        valid_num += s;
    }
    return valid_num;
}

template <int dim, IVoxNodeType node_type, typename PointType>
std::tuple<torch::Tensor,torch::Tensor> IVox<dim, node_type, PointType>::get_points_colors(){
    int num = NumPoints();
    // std::cout << "points num = " << num << std::endl;
    torch::Tensor all_points = torch::zeros({num, 3}, dtype(torch::kFloat32));
    torch::Tensor all_colors = torch::zeros({num, 3}, dtype(torch::kFloat32));

    PointType point;
    int i = 0;
    for (auto& it : grids_cache_) {
        int s = it.second.Size();
        // std::cout << "s = " << s << std::endl;
        for(int j = 0; j < s; j++){
            point = it.second.GetPoint(j);
            std::vector<float> coords_p = {point.x, point.y,point.z};
            std::vector<float> coords_c = {point.r, point.g,point.b};
            // std::cout << "coords_p = (" << coords_p[0] << ", " << coords_p[1]<< ", " << coords_p[2]<< " )"<< std::endl;
            // std::cout << "coords_c = (" << coords_c[0] << ", " << coords_c[1]<< ", " << coords_c[2]<< " )"<< std::endl;
            torch::Tensor point = torch::from_blob(coords_p.data(), {3}, dtype(torch::kFloat32));
            torch::Tensor color = torch::from_blob(coords_c.data(), {3}, dtype(torch::kFloat32));
            // std::cout << "i = " << i << std::endl;
            all_points[i] = point;
            all_colors[i] = color;
            // std::cout << "all_colors = " << all_colors<< std::endl;
            i++;
        }
    }	
    return std::make_tuple(all_points, all_colors);
}


template <int dim, IVoxNodeType node_type, typename PointType>
Eigen::Matrix<int, dim, 1> IVox<dim, node_type, PointType>::Pos2Grid(const IVox::PtType& pt) const {
    // std::cout << "insert pt = (" << pt[0] << ", " << pt[1]<< ", " << pt[2]<< " )"<< std::endl;
    // std::cout << "pt = (" << pt[0]* options_.inv_resolution_ << ", " << pt[1]* options_.inv_resolution_<< ", " << pt[2]* options_.inv_resolution_<< " )"<< std::endl;
    
    return (pt * options_.inv_resolution_).array().round().template cast<int>();
}

template <int dim, IVoxNodeType node_type, typename PointType>
std::vector<float> IVox<dim, node_type, PointType>::StatGridPoints() const {
    int num = grids_cache_.size(), valid_num = 0, max = 0, min = 100000000;
    int sum = 0, sum_square = 0;
    for (auto& it : grids_cache_) {
        int s = it.second.Size();
        valid_num += s > 0;
        max = s > max ? s : max;
        min = s < min ? s : min;
        sum += s;
        sum_square += s * s;
    }
    float ave = float(sum) / num;
    float stddev = num > 1 ? sqrt((float(sum_square) - num * ave * ave) / (num - 1)) : 0;
    return std::vector<float>{valid_num, ave, max, min, stddev};
}

}  // namespace faster_lio

#endif
