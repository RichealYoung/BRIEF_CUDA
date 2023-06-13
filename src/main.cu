#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <args/args.hxx>
#include "common.h"
#include <chrono>
#include <stdio.h>
using namespace tcnn;

using precision_t = network_precision_t;



template <uint32_t stride>
__global__ void eval_data(
	uint32_t n_elements, uint32_t depth, uint32_t height, uint32_t width,
	float coord_normalized_min, float coord_normalized_range,
	cudaTextureObject_t texture, float *__restrict__ coordinates, float *__restrict__ result)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements)
		return;

	uint32_t output_idx = i * stride;
	uint32_t input_idx = i * 3;
	// tex3D order: w,h,d
	float val = tex3D<float>(
		texture, 
		(coordinates[input_idx + 2]-coord_normalized_min) / coord_normalized_range * (float)width,
		(coordinates[input_idx + 1]-coord_normalized_min) / coord_normalized_range  * (float)height, 
		(coordinates[input_idx]-coord_normalized_min) / coord_normalized_range * (float)depth
		);
	result[output_idx] = val;
}



int main(int argc, char *argv[], char **envp)
{
	uint32_t compute_capability = cuda_compute_capability();
	if (compute_capability < MIN_GPU_ARCH)
	{
		std::cerr
			<< "Warning: Insufficient compute capability " << compute_capability << " detected. "
			<< "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly." << std::endl;
	}

	args::ArgumentParser parser{
		"BRIEF\n"
		"",
	};

	args::HelpFlag help_flag{
		parser,
		"HELP",
		"Display this help menu.",
		{'h', "help"},
	};

	args::ValueFlag<std::string> config_path{
		parser,
		"config_path",
		"config json file path",
		{'p'},
	};

	args::Flag only_decompress_flag{
		parser,
		"ONLY_DECOMPRESS",
		"Disables compress. Load compressed file from compressed_path then decompress.",
		{"only-decompress"},
	};

	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (const args::Help &)
	{
		std::cout << parser;
		return 0;
	}
	catch (const args::ParseError &e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return -1;
	}
	catch (const args::ValidationError &e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return -2;
	}
	if (!config_path)
	{
		std::cout << "Must specify config_path" << std::endl;
		return 1;
	}
	// 1. load compression task config
	std::ifstream f(args::get(config_path));
	json config = json::parse(f);
	std::cout << "Loading config '" << args::get(config_path) << "'." << std::endl;
	uint32_t n_training_samples_upper_limit = (uint32_t)config["n_training_samples_upper_limit"];
	float n_random_training_samples_of_data_size = (float)config["n_random_training_samples_of_data_size"];
	const uint32_t n_training_steps = config["n_training_steps"];
	const uint32_t n_input_dims = 3;
	const uint32_t n_output_dims = 1;
	// 2. load target/original data, generate weight
	SideInfos sideinfos;
	uint32_t data_size;
	std::vector<uint16_t> data;
	std::vector<float> weight;
	data = load_3d_data_uint16(config["data"]["path"], sideinfos.depth, sideinfos.height, sideinfos.width);
	data_size = sideinfos.depth * sideinfos.height * sideinfos.width;
	weight = generate_weight(data,data_size,(uint16_t)config["weight_intensity_min"],(uint16_t)config["weight_intensity_max"],(float)config["weight_val"]);
	// 3. normalize data
	sideinfos.normalized_min = (uint16_t)config["data"]["normalized_min"];
	sideinfos.normalized_max = (uint16_t)config["data"]["normalized_max"];
	std::vector<float> data_normalized;
	data_normalized = normalize_data(data, data_size, sideinfos.normalized_min, sideinfos.normalized_max, sideinfos.original_min, sideinfos.original_max);
	GPUMemory<float> data_device;
	GPUMemory<float> weight_device;
	// 
	uint32_t n_random_training_samples;
	bool train_by_original_coords;
	if (n_random_training_samples_of_data_size==0.0){
		if (data_size<=n_training_samples_upper_limit){
			train_by_original_coords=true;
		}
		else{
			train_by_original_coords=false;
			n_random_training_samples=n_training_samples_upper_limit;
		}
	}
	else{
		train_by_original_coords=false;
		n_random_training_samples=min(n_training_samples_upper_limit,(uint32_t)(n_random_training_samples_of_data_size*(float)data_size));
	}
	// 4. prepare coordinates d,h,w [-1,1]^3
	float coord_normalized_min = (float)config["coord_normalized_min"];
	float coord_normalized_max = (float)config["coord_normalized_max"];
	float coord_normalized_range = coord_normalized_max - coord_normalized_min;
	std::vector<float> coordinates(data_size * 3);
	for (uint32_t d = 0; d < sideinfos.depth; ++d)
	{
		for (uint32_t h = 0; h < sideinfos.height; ++h)
		{
			for (uint32_t w = 0; w < sideinfos.width; ++w)
			{
				uint32_t idx = (d * sideinfos.height * sideinfos.width + h * sideinfos.width + w) * 3;
				// plus 0.5 to hit the pix's center when using "cudaFilterModeLinear"
				// plese refer to https://stackoverflow.com/a/10667426/19514180
				coordinates[idx + 0] = (float)(d + 0.5) / (float)sideinfos.depth * coord_normalized_range + coord_normalized_min;
				coordinates[idx + 1] = (float)(h + 0.5) / (float)sideinfos.height * coord_normalized_range + coord_normalized_min;
				coordinates[idx + 2] = (float)(w + 0.5) / (float)sideinfos.width * coord_normalized_range + coord_normalized_min;
			}
		}
	}
	GPUMemory<float> coordinates_device(data_size * 3);
	coordinates_device.copy_from_host(coordinates);
	// 5. if not train_by_original_coords, create a 3D texture out of this data. 
	//    It'll be used to generate training samples efficiently on the fly
	GPUMatrix<float> training_target;
	GPUMatrix<float> training_batch;
	GPUMatrix<float> training_weight;
	cudaTextureObject_t data_texture;
	cudaTextureObject_t weight_texture;
	if (!only_decompress_flag){
		if (train_by_original_coords)
		{
			std::cout << "Each optimization step use all " << data_size << " original coords." << std::endl;
			data_device.enlarge(data_size);
			weight_device.enlarge(data_size);
			data_device.copy_from_host(data_normalized);
			weight_device.copy_from_host(weight);
			training_target = GPUMatrix<float>((float *)data_device.data(), n_output_dims, data_size);
			training_batch = GPUMatrix<float>((float *)coordinates_device.data(), n_input_dims, data_size);
			training_weight = GPUMatrix<float>((float *)weight_device.data(), n_input_dims, data_size);
		}
		else
		{
			std::cout << "Each optimization step use " << n_random_training_samples << " randomly sampled coords." << std::endl;
			data_texture = generate_3Dtexture(data_normalized.data(),sideinfos.width,sideinfos.height,sideinfos.depth,cudaFilterModeLinear);
			weight_texture = generate_3Dtexture(weight.data(),sideinfos.width,sideinfos.height,sideinfos.depth,cudaFilterModePoint);
			training_target = GPUMatrix<float>(n_output_dims, n_random_training_samples);
			training_batch = GPUMatrix<float>(n_input_dims, n_random_training_samples);
			training_weight = GPUMatrix<float>(n_input_dims, n_random_training_samples);
		}
	}
	// free memory
	data = std::vector<uint16_t>();
	data_normalized = std::vector<float>();
	weight = std::vector<float>();

	cudaStream_t training_stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&training_stream));
	cudaStream_t inference_stream = training_stream;
	default_rng_t rng{42};
	
	json encoding_opts = config.value("encoding", json::object());
	json loss_opts = config.value("loss", json::object());
	json optimizer_opts = config.value("optimizer", json::object());
	json network_opts = config.value("network", json::object());

	std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
	std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
	std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

	auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);
	// 6. load compressed data (optional)
	if (only_decompress_flag){
		std::cout << "Loading compressed data from " << config["compressed_path"] << std::endl;
		std::ifstream f{(std::string)config["compressed_path"], std::ios::in | std::ios::binary};
		json compressed_data = json::from_msgpack(f);
		trainer->deserialize(compressed_data);
	}
	if (!only_decompress_flag){
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		float tmp_loss = 0;
		uint32_t tmp_loss_counter = 0;

		std::cout << "Beginning optimization with " << n_training_steps << " training steps." << std::endl;

		uint32_t interval = (uint32_t)config["n_print_loss_interval"];

		// 6. train
		for (uint32_t i = 0; i < n_training_steps; ++i)
		{
			bool print_loss = i % interval == 0;

			// Training step
			{
				if (!train_by_original_coords)
				{
					generate_random_uniform<float>(training_stream, rng, n_random_training_samples * n_input_dims, training_batch.data(), coord_normalized_min, coord_normalized_max);
					linear_kernel(eval_data<n_output_dims>, 0, training_stream, n_random_training_samples,
								sideinfos.depth, sideinfos.height, sideinfos.width,
								coord_normalized_min, coord_normalized_range,
								data_texture, training_batch.data(), training_target.data());
					linear_kernel(eval_data<n_output_dims>, 0, training_stream, n_random_training_samples,
								sideinfos.depth, sideinfos.height, sideinfos.width,
								coord_normalized_min, coord_normalized_range,
								weight_texture, training_batch.data(), training_weight.data());
				}
				auto ctx = trainer->training_step(training_stream, training_batch, training_target,training_weight);

				if (i % std::min(interval, (uint32_t)100) == 0)
				{
					tmp_loss += trainer->loss(training_stream, *ctx);
					++tmp_loss_counter;
				}
			}

			// Debug outputs

			if (print_loss)
			{
				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				std::cout << "Step#" << i << ": "
						<< "loss=" << tmp_loss / (float)tmp_loss_counter << " time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[mu s]" << std::endl;

				tmp_loss = 0;
				tmp_loss_counter = 0;
			}
		}

		// serialize
		json data_compressed = trainer->serialize();
		std::ofstream fi(config["compressed_path"], std::ios::out | std::ios::binary);
		json::to_msgpack(data_compressed, fi);
		std::cout << "Saving compressed data into " << config["compressed_path"] << std::endl;
	}

	// decompress
	// 	inference
	GPUMatrix<float> inference_batch = GPUMatrix<float>((float *)coordinates_device.data(), n_input_dims, data_size);
	GPUMatrix<float> prediction(n_output_dims, data_size);
	// CUDA_CHECK_THROW(cudaMemsetAsync(prediction.data(), 0, sizeof(float)*data_size, inference_stream));
	const uint32_t n_inference_batch_size = config["n_inference_batch_size"];
	uint32_t n_inference_batchs = std::ceil((float)data_size/(float)n_inference_batch_size);
	clock_t  tic,toc;
	auto begin=std::chrono::high_resolution_clock::now();
	for (uint32_t batch_idx=0;batch_idx<n_inference_batchs;++batch_idx){
		if (batch_idx==n_inference_batchs-1){
			uint32_t n_left = data_size-batch_idx*n_inference_batch_size;
			uint32_t n_offset = data_size-n_left;
			GPUMatrix<float> inference_batch_i = inference_batch.slice_cols(n_offset,n_left);
			GPUMatrix<float> prediction_i(n_output_dims, n_left);
			network->inference(inference_stream, inference_batch_i, prediction_i);
			CUDA_CHECK_THROW(cudaMemcpy(prediction.data()+n_offset,prediction_i.data(), n_left * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		else{
			GPUMatrix<float> inference_batch_i = inference_batch.slice_cols(batch_idx*n_inference_batch_size,n_inference_batch_size);
			GPUMatrix<float> prediction_i(n_output_dims, n_inference_batch_size);
			network->inference(inference_stream, inference_batch_i, prediction_i);
			CUDA_CHECK_THROW(cudaMemcpy(prediction.data()+batch_idx*n_inference_batch_size,prediction_i.data(), n_inference_batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
		}
	}
	auto end=std::chrono::high_resolution_clock::now();
	auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin);
	std::cout<<elapsed.count()*1e-9<<"s"<<std::endl;
	//	save
	std::vector<float> data_decompressed_normalized(sideinfos.depth * sideinfos.height * sideinfos.width);
	std::vector<uint16_t> data_decompressed(sideinfos.depth * sideinfos.height * sideinfos.width);
	CUDA_CHECK_THROW(cudaMemcpy(data_decompressed_normalized.data(), prediction.data(), sideinfos.depth * sideinfos.height * sideinfos.width * sizeof(float), cudaMemcpyDeviceToHost));
	data_decompressed = inv_normalize_data(data_decompressed_normalized, data_decompressed_normalized.size(), sideinfos.normalized_min, sideinfos.normalized_max, sideinfos.original_min, sideinfos.original_max);
	save_3d_data_uint16(config["decompressed_path"], data_decompressed, sideinfos.depth, sideinfos.height, sideinfos.width);
	std::cout << "Saving decompressed data into " << config["decompressed_path"] << std::endl;
	free_all_gpu_memory_arenas();

	// If only the memory arenas pertaining to a single stream are to be freed, use
	// free_gpu_memory_arena(stream);
	return EXIT_SUCCESS;
}