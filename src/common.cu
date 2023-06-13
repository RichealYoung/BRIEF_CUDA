
#include "common.h"
#include <vector>
#include "tiffio.h"
#include <iostream>
#include <tiny-cuda-nn/common_device.h>
uint16 largest(const std::vector<uint16_t> &arr, uint32_t n)
{
    uint32_t i;
    uint16 max = arr[0];
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
    return max;
}

uint16_t smallest(const std::vector<uint16_t> &arr, uint32_t n)
{
    uint32_t i;
    uint16_t min = arr[0];
    for (i = 1; i < n; i++)
        if (arr[i] < min)
            min = arr[i];
    return min;
}

std::vector<uint16_t> load_3d_data_uint16(const std::string &filename, uint32_t &depth, uint32_t &height, uint32_t &width)
{
    TIFF *tif = TIFFOpen(filename.c_str(), "r");
    if (tif == nullptr)
    {
        std::cerr << "Wrong data path !" << std::endl;
        abort();
    }
    uint16_t bitdpeh;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitdpeh);
    if (bitdpeh != 16)
    {
        std::cerr << "Only support 16 bit depth data !" << std::endl;
        abort();
    }
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    depth = TIFFNumberOfDirectories(tif);
    printf("Loading target/original data from %s\n", filename.c_str());
    printf("Target/Original data shape (%d,%d,%d)\n", depth, height, width);

    std::vector<uint16_t> data(depth * height * width);
    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            TIFFReadScanline(tif, &data[d * height * width + h * width], h);
        }
        TIFFReadDirectory(tif);
    }
    TIFFClose(tif);
    return data;
}

void save_3d_data_uint16(const std::string filename, std::vector<uint16_t> data, uint32_t depth, uint32_t height, uint32_t width)
{
    TIFF *out = TIFFOpen(filename.c_str(), "w");
    if (out)
    {
        int d = 0;
        do
        {
            TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
            TIFFSetField(out, TIFFTAG_PAGENUMBER, depth);
            TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
            TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
            TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
            TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, height);
            for (int h = 0; h < height; h++)
            {
                TIFFWriteScanline(out, &data[d * height * width + h * width], h, 0);
            }
            d++;
        } while (TIFFWriteDirectory(out) && d < depth);
        TIFFClose(out);
    }
}

std::vector<float> normalize_data(const std::vector<uint16_t> &data, const uint32_t &data_size, const uint16_t &normalized_min, uint16_t &normalized_max, uint16_t &original_min, uint16_t &original_max)
{
    std::vector<float> data_normalized(data_size);
    original_min = smallest(data, data_size);
    original_max = largest(data, data_size);
    float original_dynamic_range = (float)(original_max - original_min);
    float normalized_dynamic_range = (float)(normalized_max - normalized_min);
    float scale = normalized_dynamic_range / original_dynamic_range;
    for (uint32_t i = 0; i < data_size; i++)
        data_normalized[i] = ((float)data[i] - (float)original_min) * scale + (float)normalized_min;
    return data_normalized;
}

std::vector<uint16_t> inv_normalize_data(const std::vector<float> &data_normalized, const uint32_t &data_size, const uint16_t &normalized_min, const uint16_t &normalized_max, const uint16_t &original_min, const uint16_t &original_max)
{
    std::vector<uint16_t> data(data_size);
    float original_dynamic_range = (float)(original_max - original_min);
    float normalized_dynamic_range = (float)(normalized_max - normalized_min);
    float scale = original_dynamic_range / normalized_dynamic_range;
    for (uint32_t i = 0; i < data_size; i++)
        data[i] = (uint16_t)std::min(65535.0f,std::max(0.0f,((data_normalized[i] - (float)normalized_min) * scale + (float)original_min)));
    return data;
}

std::vector<float> generate_weight(const std::vector<uint16_t> &data,const uint32_t &data_size, const uint16_t &min, const uint16_t &max, const float &weight_val)
{
    std::vector<float> weight(data_size);
    for (uint32_t i = 0; i < data_size; i++)
        weight[i] = (data[i]>=min && data[i]<=max)?weight_val:1.0f;
    return weight;
}

template<typename T>
void align_data(T * original_data, char * aligned_data,size_t pitch, uint32_t depth, uint32_t height, uint32_t width){
	size_t slicePitch = pitch * height;
	for (uint32_t d=0;d<depth;d++){
		char * slice = aligned_data + d * slicePitch;
		for (uint32_t h=0;h<height;h++){
			T * row = (T *)(slice + h * pitch);
			for (uint32_t w=0;w<width;w++){
				row[w] = original_data[d*height*width+h*width+w];
			}
		}
	}
}

__global__ void
test_tex_read(uint32_t pix_d, uint32_t pix_h, uint32_t pix_w, uint32_t depth, uint32_t height, uint32_t width, cudaTextureObject_t texture, float *data)
{
	float d = ((float)pix_d + 0.5); // /(float)depth;
	float h = ((float)pix_h + 0.5); // /(float)height;
	float w = ((float)pix_w + 0.5); // /(float)width;
	float tex_val = tex3D<float>(texture, w, h, d);
	float data_val = data[pix_d * height * width + pix_h * width + pix_w];
	printf("d: %f, h: %f, w:%f, tex_val: %f, data_val: %f\n", d, h, w, tex_val, data_val);
}


cudaTextureObject_t generate_3Dtexture(float* data,const uint32_t &width,const uint32_t &height,const uint32_t &depth, cudaTextureFilterMode filterMode)
{
    cudaTextureObject_t texture;
    cudaExtent volumeSizeBytes = make_cudaExtent(width*sizeof(float), height, depth);
    cudaPitchedPtr d_volumeMem; 
    CUDA_CHECK_THROW(cudaMalloc3D(&d_volumeMem, volumeSizeBytes));
    size_t data_aligned_size = d_volumeMem.pitch * height * depth;
    float* data_aligned = (float *)malloc(data_aligned_size);
    align_data<float>(data,(char *)data_aligned,d_volumeMem.pitch,depth,height,width);
    CUDA_CHECK_THROW(cudaMemcpy(d_volumeMem.ptr, data_aligned, data_aligned_size, cudaMemcpyHostToDevice));
    cudaArray *d_volumeArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize = make_cudaExtent(width, height, depth);
    CUDA_CHECK_THROW(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize, 0));
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = d_volumeMem;
    copyParams.dstArray = d_volumeArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK_THROW(cudaMemcpy3D(&copyParams));
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_volumeArray;
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode = filterMode;
    // FIXME normalizedCoords=true will cause a wrong texture fecthing value
    texDesc.normalizedCoords = false;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    // onlt for test
    // test_tex_read<<<1,1>>>(0, 0, 2,depth,height,width,texture,data_device.data());
    // CUDA_CHECK_THROW(cudaDeviceSynchronize());
    // free memory
    free(data_aligned);
    cudaFree(d_volumeMem.ptr);
    return texture;
}