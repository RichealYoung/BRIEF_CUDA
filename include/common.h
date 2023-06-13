

#pragma once
#include <vector>
struct SideInfos
{
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint16_t original_min;
    uint16_t original_max;
    uint16_t normalized_min;
    uint16_t normalized_max;
};

std::vector<uint16_t> load_3d_data_uint16(const std::string &filename, uint32_t &depth, uint32_t &height, uint32_t &width);
void save_3d_data_uint16(const std::string filename, std::vector<uint16_t> data, uint32_t depth, uint32_t height, uint32_t width);
std::vector<float> normalize_data(const std::vector<uint16_t> &data, const uint32_t &data_size, const uint16_t &normalized_min, uint16_t &normalized_max, uint16_t &original_min, uint16_t &original_max);
std::vector<uint16_t> inv_normalize_data(const std::vector<float> &data_normalized, const uint32_t &data_size, const uint16_t &normalized_min, const uint16_t &normalized_max, const uint16_t &original_min, const uint16_t &original_max);
std::vector<float> generate_weight(const std::vector<uint16_t> &data,const uint32_t &data_size, const uint16_t &min, const uint16_t &max, const float &weight_val);
cudaTextureObject_t generate_3Dtexture(float* data,const uint32_t &width,const uint32_t &height,const uint32_t &depth, cudaTextureFilterMode filterMode);