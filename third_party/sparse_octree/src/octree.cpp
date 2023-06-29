#include "octree.h"
#include "utils.h"
#include <queue>
#include <iostream>

// #define MAX_HIT_VOXELS 10
// #define MAX_NUM_VOXELS 10000

int Octant::next_index_ = 0;
// int Octree::feature_index = 0;

int incr_x[8] = {0, 0, 0, 0, 1, 1, 1, 1};
int incr_y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
int incr_z[8] = {0, 1, 0, 1, 0, 1, 0, 1};

Octree::Octree()
{
}

Octree::Octree(int64_t grid_dim, int64_t feat_dim, double voxel_size, std::vector<torch::Tensor> all_pts, std::vector<torch::Tensor> all_colors, std::vector<torch::Tensor> all_pcds, int64_t max_num)
{
    std::cout << "octree initialization is OK !!!!"   << std::endl;
    // std::cout << "max_num: " << max_num<< std::endl;
    Octant::next_index_ = 0;
    MAX_POINTS_PER_LEAF = max_num;
    // std::cout << "MAX_POINTS_PER_LEAF0002: " << MAX_POINTS_PER_LEAF<< std::endl;
    init(grid_dim, feat_dim, voxel_size, max_num);
    // for (auto &pt : all_pts)
    // {
    //     for (auto &color : all_colors){
    //         for (auto &pcd : all_pcds){
    //             insert(pt,color,pcd);
    //         }
    //     }
    // }
    for (auto &pt : all_pts)
    {
        insert(pt,pt,pt);
    }
}

Octree::~Octree()
{
}

void Octree::init(int64_t grid_dim, int64_t feat_dim, double voxel_size, int64_t max_num)
{
    MAX_POINTS_PER_LEAF = max_num;
    // float size = voxel_size/5;
    // voxel_scan_.setLeafSize(size, size, size);
    // std::cout << "MAX_POINTS_PER_LEAF0: " << MAX_POINTS_PER_LEAF<< std::endl;
    size_ = grid_dim;
    feat_dim_ = feat_dim;
    voxel_size_ = voxel_size; //0.2
    max_level_ = log2(size_);
    // root_ = std::make_shared<Octant>();
    root_ = new Octant();
    root_->side_ = size_;
    // root_->depth_ = 0;
    root_->is_leaf_ = false;
    // MAX_POINTS_PER_LEAF = max_num;
    ivox_ = std::make_shared<IVoxType>(ivox_options_);
    
    // feats_allocated_ = 0;
    // auto options = torch::TensorOptions().requires_grad(true);
    // feats_array_ = torch::randn({MAX_NUM_VOXELS, feat_dim}, options) * 0.01;
}

bool Octree::isInSubspace(Point3 point, Subspace subspace,int j) {
    double half_length = subspace.length / 2.0;
    // double half_length = subspace.length ;
    if(j > 0){
        half_length = subspace.length ;
    }
    if(point.x > subspace.center.x-half_length && point.x < subspace.center.x+half_length &&
       point.y > subspace.center.y-half_length && point.y < subspace.center.y+half_length &&
       point.z > subspace.center.z-half_length && point.z < subspace.center.z+half_length) {
        return true;
    }
    return false;
}

std::vector<Subspace> Octree::divideSpace(Point3 center, double length) {
    vector<Subspace> subspaces;
    double half_length = length / 4.0;
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

void Octree::insert(torch::Tensor pts, torch::Tensor color, torch::Tensor pcd)
{
    std::cout << "Octree::insert !!!!"   << std::endl;

    // temporal solution
    all_pts.push_back(pts);
    all_colors.push_back(color);
    all_pcds.push_back(pcd);

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
            // int x = points[i][0];
            // int y = points[i][1];
            // int z = points[i][2];
            uint64_t key = encode(x, y, z);
            // int j = 0;
            all_keys.insert(key);
            const unsigned int shift = MAX_BITS - max_level_ - 1;
            auto n = root_;
            unsigned edge = size_ / 2;
            // std::cout << "jj=  " << j << std::endl;
            
            for (int d = 1; d <= max_level_; edge /= 2, ++d)
            {
                // std::cout << "层数=  " << d << std::endl;
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
                    // Point3 centre = {
                    //     (x + edge / 2) * voxel_size_,
                    //     (y + edge / 2) * voxel_size_,
                    //     (z + edge / 2) * voxel_size_,
                    // };
                    Point3 centre = {
                        x * voxel_size_,
                        y * voxel_size_,
                        z * voxel_size_
                    };
                    // std::cout << "centre=  " << "("<< centre.x << "," << centre.y << "," << centre.z << ")"
                    //     << "edge=  " << edge << ","<< "voxel_size_=  " << voxel_size_ 
                    //     << "xyz=(" << x << ","<< y << "," << z << ")"<< std::endl;
                    // std::cout << "centre=  " << "("<< centre.x << "," << centre.y << "," << centre.z << ")"

                    // if(flag1 < 10){
                    //     std::cout << "centre=  " << "("<< centre.x << "," << centre.y << "," << centre.z << ")"
                    //     << "edge=  " << edge << ","<< "voxel_size_=  " << voxel_size_ 
                    //     << "xyz=(" << x << ","<< y << "," << z << ")"<< std::endl;
                    //     flag2 ++;
                    // }
                    tmp->subspaces_ = divideSpace(centre, edge*voxel_size_);
                    // std::cout << "2"  << std::endl;
                    if (tmp->point_data_color.size() < MAX_POINTS_PER_LEAF){
                        // std::cout << "3"  << std::endl;
                        for(int k=0; k<8; k++) {
                            
                            Point3 point = {pcds[i][0],pcds[i][1],pcds[i][2]};

                            if(isInSubspace(point, tmp->subspaces_[k],j)) {
                                // if(flag1 < 50){
                                    // std::cout << "points"<< "("<< points[i][0] << "," << points[i][1] << "," << points[i][2] << ")"<< std::endl;
                                    // std::cout << "xyz"<< "("<< x << "," << y << "," << z << ")"<< std::endl;
                                    // std::cout << "centre=  " << "("<< centre.x << "," << centre.y << "," << centre.z << ")" << std::endl;
                                    // std::cout << "新节点isInSubspace=  "  <<isInSubspace(point, tmp->subspaces_[k])                                    << "pcds"<< "("<< pcds[i][0] << "," << pcds[i][1] << "," << pcds[i][2] << ")" 
                                    //     << "edge=  " << edge << std::endl;
                                    // flag1 ++;
                                // }
                                // 如果该子空间中已经有点，则替换掉该点
                                if(tmp->subspaces_[k].point.x != 0.0 || tmp->subspaces_[k].point.y != 0.0 || tmp->subspaces_[k].point.z != 0.0) {
                                    tmp->subspaces_[k].point = point;
                                    // std::cout << "新节点替换pcd"  << std::endl;

                                }
                                // 否则将该点存入对应的子空间中
                                else {
                                    // if(flag1 < 50){
                                    // std::cout << "新节点添加新pcd"  << std::endl;
                                    // std::cout << "------新节点-------"  << std::endl;

                                        // }
                                    tmp->subspaces_[k].point = point;
                                    float color = hilbert_encode(colors[i][0], colors[i][1], colors[i][2]);
                                    tmp->point_data_x.push_back(pcds[i][0]);  
                                    tmp->point_data_y.push_back(pcds[i][1]);  
                                    tmp->point_data_z.push_back(pcds[i][2]);  
                                    tmp->point_data_color.push_back(color); 
                                    // printf("新 point: (%f,%f,%f)\n" ,point.x, point.y,point.z);
                                    break;
                                }
                            }
                        }
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
                    if (tmp->point_data_color.size()  < MAX_POINTS_PER_LEAF ){
                        // std::cout << "6"  << std::endl;
                        for(int k=0; k<8; k++) {
                            
                            Point3 point = {pcds[i][0],pcds[i][1],pcds[i][2]};
                          
                            // std::cout << "centre=  " << "("<< centre.x << "," << centre.y << "," << centre.z << ")" << std::endl;
                            if(isInSubspace(point, tmp->subspaces_[k],j)) {
                                // if(flag2 < 50){
                                // std::cout << "旧节点isInSubspace=  "  <<isInSubspace(point, tmp->subspaces_[k])
                                    // << "("<< pcds[i][0] << "," << pcds[i][1] << "," << pcds[i][2] << ")" << std::endl;
                                    // std::cout << "centre=  " << "("<< centre.x << "," << centre.y << "," << centre.z << ")" << std::endl;
                                    // flag2 ++;
                                // }
                            // 如果该子空间中已经有点，则替换掉该点
                                if(tmp->subspaces_[k].point.x != 0.0 || tmp->subspaces_[k].point.y != 0.0 || tmp->subspaces_[k].point.z != 0.0) {
                                    tmp->subspaces_[k].point = point;
                                    // std::cout << "旧节点替换pcd"  << std::endl;
                                }
                            // 否则将该点存入对应的子空间中
                                else {
                                    // if(flag2 < 50){
                                    // std::cout << "旧节点添加新pcd"  << std::endl;
                                    // }
                                    // std::cout << "8"  << std::endl;
                                    tmp->subspaces_[k].point = point;
                                    float color = hilbert_encode(colors[i][0], colors[i][1], colors[i][2]);
                                    tmp->point_data_x.push_back(pcds[i][0]);  
                                    tmp->point_data_y.push_back(pcds[i][1]);  
                                    tmp->point_data_z.push_back(pcds[i][2]);  
                                    tmp->point_data_color.push_back(color); 
                                    // printf("新 point: (%f,%f,%f)\n" ,point.x, point.y,point.z);
                                    break;
                                }
                            }
                        }
                    }
                    // }
                }
            n = tmp;
            }
        }
    }
}

void Octree::insert_hash(torch::Tensor pts, torch::Tensor color)
{
    std::cout << "hash insert !!!!"   << std::endl;

    // temporal solution
    all_pts.push_back(pts);
    all_colors.push_back(color);

    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }
    // std::cout << "MAX_POINTS_PER_LEAF1: " << MAX_POINTS_PER_LEAF<< std::endl;

    auto points = pts.accessor<float, 2>();
    auto colors = color.accessor<int, 2>();


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

    // CloudPtr pcd_downsample_{new PointCloudType()}; 
    CloudPtr PCD_{new PointCloudType()};
    for (int i = 0; i < points.size(0); ++i){
        PointType point;
        point.x = points[i][0];
        point.y = points[i][1];
        point.z = points[i][2];
        point.r = colors[i][0];
        point.g = colors[i][1];
        point.b = colors[i][2];
        PCD_->points.push_back(point);
        // std::cout << "point " << point << std::endl;
    }
    // voxel_scan_.setInputCloud(PCD_);
    // voxel_scan_.filter(*pcd_downsample_);
    ivox_->AddPoints(PCD_->points);
}

torch::Tensor Octree::get_centres()
{
    auto all_voxels = ivox_->get_voxel_center();
    return all_voxels;
}

std::tuple<torch::Tensor, torch::Tensor> Octree::getPoints()
{
    auto all_points_colors = ivox_->get_points_colors();
    return all_points_colors;
}

std::tuple<torch::Tensor, torch::Tensor> Octree::getClosePoints(torch::Tensor pts)
{
    auto points = pts.accessor<float, 2>();
    if (points.size(1) != 3)
        {
            std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        }
    PointVector cloud;
    int cur_pts = points.size(0);
    cloud.resize(cur_pts);
    for (int i = 0; i < points.size(0); ++i){
        PointType point;
        point.x = points[i][0];
        point.y = points[i][1];
        point.z = points[i][2];
        cloud[i]= point;
        // std::cout << "cloud " << cloud[i].x<< ", "<< cloud[i].y << ", "<< cloud[i].z << std::endl;
    }
    // std::cout << "cloud " << cloud[0].x<< ", "<< cloud[0].y << ", "<< cloud[0].z << std::endl;
    // std::cout << "cloud " << cloud[1].x<< ", "<< cloud[1].y << ", "<< cloud[1].z << std::endl;
    int k_num = 15;
    // std::cout << "cloud.size() = " <<cloud.size() << std::endl;
    auto all_points_colors = ivox_->GetClosestPoint_d(cloud,k_num);
    return all_points_colors;
}

double Octree::try_insert(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return -1.0;
    }

    std::set<uint64_t> tmp_keys;

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            tmp_keys.insert(key);
        }
    }

    std::set<int> result;
    std::set_intersection(all_keys.begin(), all_keys.end(),
                          tmp_keys.begin(), tmp_keys.end(),
                          std::inserter(result, result.end()));

    double overlap_ratio = 1.0 * result.size() / tmp_keys.size();
    return overlap_ratio;
}

Octant *Octree::find_octant(std::vector<float> coord)
{
    int x = int(coord[0]);
    int y = int(coord[1]);
    int z = int(coord[2]);
    // uint64_t key = encode(x, y, z);
    // const unsigned int shift = MAX_BITS - max_level_ - 1;

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return nullptr;

        n = tmp;
    }
    return n;
}

bool Octree::has_voxel(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 1>();
    if (points.size(0) != 3)
    {
        return false;
    }

    int x = int(points[0]);
    int y = int(points[1]);
    int z = int(points[2]);

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return false;

        n = tmp;
    }

    if (!n)
        return false;
    else
        return true;
}

torch::Tensor Octree::get_features(torch::Tensor pts)
{
}

torch::Tensor Octree::get_leaf_voxels()
{
    std::vector<float> voxel_coords = get_leaf_voxel_recursive(root_);

    int N = voxel_coords.size() / 3;
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 3});
    return voxels.clone();
}

std::vector<float> Octree::get_leaf_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    if (n->is_leaf_ && n->type_ == SURFACE)
    {
        auto xyz = decode(n->code_);
        return {xyz[0], xyz[1], xyz[2]};
    }

    std::vector<float> coords;
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_leaf_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

torch::Tensor Octree::get_voxels()
{
    std::vector<float> voxel_coords = get_voxel_recursive(root_);
    int N = voxel_coords.size() / 4;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 4}, options);
    return voxels.clone();
}

std::vector<float> Octree::get_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    auto xyz = decode(n->code_);
    std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(n->side_)};
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

std::pair<int64_t, int64_t> Octree::count_nodes_internal()
{
    return count_recursive_internal(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
std::pair<int64_t, int64_t> Octree::count_recursive_internal(Octant *n)
{
    if (!n)
        return std::make_pair<int64_t, int64_t>(0, 0);

    if (n->is_leaf_)
        return std::make_pair<int64_t, int64_t>(1, 1);

    auto sum = std::make_pair<int64_t, int64_t>(1, 0);

    for (int i = 0; i < 8; i++)
    {
        auto temp = count_recursive_internal(n->child(i));
        sum.first += temp.first;
        sum.second += temp.second;
    }

    return sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Octree::get_centres_and_children()
{
    auto node_count = count_nodes_internal();
    auto total_count = node_count.first;
    auto leaf_count = node_count.second;
    std::cout << "total_count" << total_count<< std::endl;

    auto all_voxels = torch::zeros({total_count, 4}, dtype(torch::kFloat32));
    auto all_pointclouds_xyz = torch::zeros({total_count, MAX_POINTS_PER_LEAF, 4}, dtype(torch::kFloat32));
    auto all_pointclouds_colors = torch::zeros({total_count, MAX_POINTS_PER_LEAF, 3}, dtype(torch::kFloat32));
    auto all_children = -torch::ones({total_count, 8}, dtype(torch::kFloat32));
    auto all_features = -torch::ones({total_count, 8}, dtype(torch::kInt32));
    
    // std::cout << "all_pointclouds_xyz" << all_pointclouds_xyz<< std::endl;
    // std::cout << "all_pointclouds_colors" << all_pointclouds_colors<< std::endl;
    // std::cout << "MAX_POINTS_PER_LEAF2: " << MAX_POINTS_PER_LEAF<< std::endl;
    
    std::queue<Octant *> all_nodes;
    all_nodes.push(root_);
    // int flag1 = 1;
    // int flag5 = 1;

    while (!all_nodes.empty())
    {
        auto node_ptr = all_nodes.front();
        all_nodes.pop();

        auto xyz = decode(node_ptr->code_);
        std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(node_ptr->side_)};
        auto voxel = torch::from_blob(coords.data(), {4}, dtype(torch::kFloat32));
        all_voxels[node_ptr->index_] = voxel;
        // if(flag1 < 3){
        //         std::cout << "node_ptr->side_1: " << node_ptr->side_<< "  flag1: " << flag1 << std::endl;
        //         flag1 ++;
        //     }
        // std::cout << "MAX_POINTS_PER_LEAF: " << MAX_POINTS_PER_LEAF<< std::endl;

        std::vector<std::array<float, 4>> xyz_array(MAX_POINTS_PER_LEAF, {0, 0, 0, 0});
        std::vector<std::array<float, 3>> color_array(MAX_POINTS_PER_LEAF, {0, 0, 0});
        
            // || node_ptr->side_ == size_
        // if (node_ptr->side_ == size_){
        //     auto xyz_ = decode(node_ptr->code_);
        //     for (int i = 0; i < MAX_POINTS_PER_LEAF ; i++) {
                    
        //             xyz_array[i][0]=(xyz_[0] + float(node_ptr->side_) / 2) * 0.2
        //             xyz_array[i][1]=(xyz_[1] + float(node_ptr->side_) / 2) * 0.2
        //             xyz_array[i][2]=(xyz_[2] + float(node_ptr->side_) / 2) * 0.2

        //         // if(flag4 < 5){
        //         //     std::cout << "node_ptr->side_4: " << node_ptr->side_<< "  flag4: " << flag4 << std::endl;
        //         //     flag4 ++;
        //         // }
        //     }
        // }

        for (int i = 0; i < node_ptr->point_data_x.size(); i++) {
            // if(flag5 < 10){
            //     std::cout << "point_data_x.size(): " << node_ptr->point_data_x.size()<< std::endl;
            //     flag5 ++;
            // }
            auto x_ = node_ptr->point_data_x[i];
            auto y_ = node_ptr->point_data_y[i];
            auto z_ = node_ptr->point_data_z[i];
            xyz_array[i][0]=x_;
            xyz_array[i][1]=y_;
            xyz_array[i][2]=z_;
            xyz_array[i][3]=float(node_ptr->side_);
            auto color_ = hilbert_decode(node_ptr->point_data_color[i]);
            // std::cout << "color_[0]: " << color_[0]<< "  flag5: " << flag5 << std::endl;
            color_array[i][0]=color_[0];
            color_array[i][1]=color_[1];
            color_array[i][2]=color_[2];
            // if(flag5 < 5){
            //     std::cout << "color_[0]: " << color_[0]<< "  flag5: " << flag5 << std::endl;
            //     flag5 ++;
            // }
        }
        // std::cout << "Finish"<< std::endl;

        // for (int i = 0; i < 8; ++i) {
            // auto xyz_ = decode(node_ptr->point_data_xyz[i]);
            // auto color_ = decode(node_ptr->point_data_color[i]);
            // color_array[i][0]=color_[0];
            // color_array[i][1]=color_[1];
            // color_array[i][2]=color_[2];
        // }
        // std::cout << "xyz_array" << xyz_array<< std::endl;

        auto pc_position = torch::from_blob(xyz_array.data(), {MAX_POINTS_PER_LEAF, 4}, dtype(torch::kFloat32));
        all_pointclouds_xyz[node_ptr->index_] = pc_position;

        auto pc_rgb = torch::from_blob(color_array.data(), {MAX_POINTS_PER_LEAF, 3}, dtype(torch::kFloat32));
        all_pointclouds_colors[node_ptr->index_] = pc_rgb;


        if (node_ptr->type_ == SURFACE)
        {
            for (int i = 0; i < 8; ++i)
            {
                std::vector<float> vcoords = coords;
                vcoords[0] += incr_x[i];
                vcoords[1] += incr_y[i];
                vcoords[2] += incr_z[i];
                auto voxel = find_octant(vcoords);
                if (voxel)
                    all_features.data_ptr<int>()[node_ptr->index_ * 8 + i] = voxel->index_;
                // if(flag5 < 5){
                //     std::cout << "voxel->index_ " << voxel->index_<< "  flag5: " << flag5 << std::endl;
                //     flag5 ++;
                // }
            } 
        }

        for (int i = 0; i < 8; i++)
        {
            auto child_ptr = node_ptr->child(i);
            if (child_ptr && child_ptr->type_ != FEATURE)
            {
                all_nodes.push(child_ptr);
                all_children[node_ptr->index_][i] = float(child_ptr->index_);
            }
        }
    }

    return std::make_tuple(all_voxels, all_children, all_features, all_pointclouds_xyz,all_pointclouds_colors);
}
// all_voxels是一个N*4的张量，其中N是八叉树中非FEATURE类型的节点的数量。每一行代表一个节点对应的体素，包含了x,y,z坐标和边长。
// all_children是一个N*8的张量，其中N和上面相同。每一行代表一个节点和其八个子节点之间的索引关系。如果某个子节点不存在或者是FEATURE类型，则对应位置为-1。
// all_features是一个N*8的张量，其中N和上面相同。每一行代表一个表面节点对应体素的八个顶点是否有特征。如果某个顶点有特征，则对应位置为该特征节点在八叉树中的索引；否则为-1。
int64_t Octree::count_nodes()
{
    return count_recursive(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
int64_t Octree::count_recursive(Octant *n)
{
    if (!n)
        return 0;

    int64_t sum = 1;

    for (int i = 0; i < 8; i++)
    {
        sum += count_recursive(n->child(i));
    }

    return sum;
}

int64_t Octree::count_leaf_nodes()
{
    return leaves_count_recursive(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
int64_t Octree::leaves_count_recursive(Octant *n)
{
    if (!n)
        return 0;

    if (n->type_ == SURFACE)
    {
        return 1;
    }

    int64_t sum = 0;

    for (int i = 0; i < 8; i++)
    {
        sum += leaves_count_recursive(n->child(i));
    }

    return sum;
}
