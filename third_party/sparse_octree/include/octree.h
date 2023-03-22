#include <memory>
#include <torch/script.h>
#include <torch/custom_class.h>

enum OcType
{
    NONLEAF = -1,
    SURFACE = 0,
    FEATURE = 1
};

class Octant : public torch::CustomClassHolder
{
public:
    inline Octant()
    {
        code_ = 0;
        side_ = 0;
        index_ = next_index_++;
        depth_ = -1;
        is_leaf_ = false;
        children_mask_ = 0;
        type_ = NONLEAF;
        // point_data_xyz = 0;
        // point_data_color = 0;

        // for (unsigned int i = 0; i < 10; i++)
        // {
        //     point_data_[i] = nullptr;
        //     // feature_index_[i] = -1;
        // }

        for (unsigned int i = 0; i < 8; i++)
        {
            child_ptr_[i] = nullptr;
            // feature_index_[i] = -1;
        }

        // for (unsigned int i = 0; i < 8; i++)
        // {
        //     point_data_xyz = nullptr;
        //     point_data_color[i] = nullptr;
        //     // feature_index_[i] = -1;
        // }

    }
    ~Octant() {}
    // int point_indices_;
    // uint64_t *point_data_[10]; 
    std::vector<uint64_t> point_data_xyz; 
    std::vector<uint64_t> point_data_color; 
    // torch::Tensor point_xyz;
    // auto point_data_xyz = point_xyz.accessor<int, 2>();
    // torch::Tensor point_data_color;
    // auto point_data_color = point_color.accessor<int, 2>();
    // std::vector<uint64_t> point_data_xyz;
    // std::vector<uint64_t> point_data_color;
    // std::vector<Point> point_data_xyz;
    // std::vector<Point> point_data_color;


    Octant *&child(const int x, const int y, const int z)
    {
        return child_ptr_[x + y * 2 + z * 4];
    };

    Octant *&child(const int offset)
    {
        return child_ptr_[offset];
    }
    std::vector<torch::Tensor> pts_;
    uint64_t code_;
    bool is_leaf_;
    unsigned int side_;
    unsigned char children_mask_;
    // std::shared_ptr<Octant> child_ptr_[8];
    // int feature_index_[8];
    int index_;
    int depth_;
    int type_;
    // int feat_index_;
    Octant *child_ptr_[8];
    static int next_index_;
};

class Octree : public torch::CustomClassHolder
{
public:
    Octree();
    // temporal solution
    Octree(int64_t grid_dim, int64_t feat_dim, double voxel_size, std::vector<torch::Tensor> all_pts, std::vector<torch::Tensor> all_colors, int64_t max_num);
    ~Octree();
    void init(int64_t grid_dim, int64_t feat_dim, double voxel_size, int64_t max_num);

    // allocate voxels
    void insert(torch::Tensor vox,torch::Tensor color);
    double try_insert(torch::Tensor pts);

    // find a particular octant
    Octant *find_octant(std::vector<float> coord);

    // test intersections
    bool has_voxel(torch::Tensor pose);

    // query features
    torch::Tensor get_features(torch::Tensor pts);

    // get all voxels
    torch::Tensor get_voxels();
    std::vector<float> get_voxel_recursive(Octant *n);

    // get leaf voxels
    torch::Tensor get_leaf_voxels();
    std::vector<float> get_leaf_voxel_recursive(Octant *n);

    // count nodes
    int64_t count_nodes();
    int64_t count_recursive(Octant *n);

    // count leaf nodes
    int64_t count_leaf_nodes();
    // int64_t leaves_count_recursive(std::shared_ptr<Octant> n);
    int64_t leaves_count_recursive(Octant *n);

    // get voxel centres and childrens
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_centres_and_children();

public:
    int size_;
    int feat_dim_;
    int max_level_;
    int64_t MAX_POINTS_PER_LEAF;
    // int64_t max_num_ ;

    // temporal solution
    double voxel_size_;
    std::vector<torch::Tensor> all_pts;
    std::vector<torch::Tensor> all_colors;
  

private:
    std::set<uint64_t> all_keys;
    
    // std::shared_ptr<Octant> root_;
    Octant *root_;
    // static int feature_index;

    // internal count function
    std::pair<int64_t, int64_t> count_nodes_internal();
    std::pair<int64_t, int64_t> count_recursive_internal(Octant *n);


};