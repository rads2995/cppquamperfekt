#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cmath>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace cppquamperfekt {

enum class ReturnCode : int {
    valid = 0,
    invalid
};

struct Herz {   

    Herz(
        dnnl::engine::kind kind = dnnl::engine::kind::gpu,
        std::size_t index = 0,
        dnnl::stream::flags stream_flags = dnnl::stream::flags::default_flags
    ) : engine(kind, index),
        stream(engine, stream_flags) {

            // Perform validation for constructed Herz object
            if (
                validate_engine_kind() != ReturnCode::valid
            ) {
                std::cout << "Failed to validate constructed Herz object!\n";
            }

            conv1_weights.resize(std::accumulate(
                conv1_weights_tz.begin(), 
                conv1_weights_tz.end(), 
                static_cast<std::size_t>(1), 
                std::multiplies<std::size_t>()
            ));

            conv1_bias.resize(std::accumulate(
                conv1_bias_tz.begin(), 
                conv1_bias_tz.end(), 
                static_cast<std::size_t>(1), 
                std::multiplies<std::size_t>()
            ));
        }

    ReturnCode validate_engine_kind();
    ReturnCode read_from_dnnl_memory(void* handle, dnnl::memory& memory);
    ReturnCode write_to_dnnl_memory(void* handle, dnnl::memory& memory);
    ReturnCode execute();

    // Create an engine and stream
    dnnl::engine engine;
    dnnl::stream stream; 

    // Create a vector for primitives and a vector to hold memory used as arguments
    std::vector<dnnl::primitive> net;
    std::vector<std::unordered_map<int, dnnl::memory>> net_args;

    static constexpr dnnl::memory::dim batch = 1;
    
    // AlexNet: conv1
    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
    // strides: {4, 4}
    dnnl::memory::dims conv1_src_tz {1, 3, 227, 227};
    dnnl::memory::dims conv1_weights_tz {96, 3, 11, 11};
    dnnl::memory::dims conv1_bias_tz {96};
    dnnl::memory::dims conv1_dst_tz {1, 96, 55, 55};
    dnnl::memory::dims conv1_strides {4, 4};
    dnnl::memory::dims conv1_padding {0, 0};

    // Allocate buffers for input and output data, weights, and bias
    std::vector<float> user_src {std::vector<float>(batch * 3 * 227 * 227)};
    std::vector<float> user_dst {std::vector<float>(1 * 1000)};
    std::vector<float> conv1_weights;
    std::vector<float> conv1_bias;

    // Create memory that describes data layout in the buffers
    dnnl::memory user_src_memory {dnnl::memory(
        {{conv1_src_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw},
        engine
    )};
    dnnl::memory user_weights_memory {dnnl::memory(
        {{conv1_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw},
        engine
    )};
    dnnl::memory conv1_user_bias_memory {dnnl::memory(
        {{conv1_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x},
        engine
    )};

    // Create memory descriptors for convolution data w/ no specified layout
    dnnl::memory::desc conv1_src_md = dnnl::memory::desc(
        {conv1_src_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any
    );
    dnnl::memory::desc conv1_bias_md = dnnl::memory::desc(
        {conv1_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any
    );
    dnnl::memory::desc conv1_weights_md = dnnl::memory::desc(
        {conv1_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any
    );
    dnnl::memory::desc conv1_dst_md = dnnl::memory::desc(
        {conv1_dst_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any
    );

    // Create a convolution primitive descriptor
    dnnl::convolution_forward::primitive_desc 
    conv1_prim_desc = dnnl::convolution_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct,
        conv1_src_md,
        conv1_weights_md,
        conv1_bias_md,
        conv1_dst_md,
        conv1_strides,
        conv1_padding,
        conv1_padding
    );

    dnnl::memory conv1_src_memory = user_src_memory;
    dnnl::memory conv1_weights_memory = user_weights_memory;
    
    // Create a memory primitive for output
    dnnl::memory conv1_dst_memory = dnnl::memory(conv1_prim_desc.dst_desc(), engine);
 
    // AlexNet: relu1
    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
    
    // Create ReLU primitive descriptor
    dnnl::eltwise_forward::primitive_desc relu1_prim_desc =  dnnl::eltwise_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::eltwise_relu,
        conv1_dst_memory.get_desc(),
        conv1_dst_memory.get_desc(),
        0.0f
    );

    // AlexNet: lrn1
    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
    // local size: 5
    // alpha1: 0.0001
    // beta1: 0.75
    const dnnl::memory::dim local1_size = 5;
    const float alpha1 = 0.0001f;
    const float beta1 = 0.75f;
    const float k1 = 1.0f;
    
    // Create lrn primitive
    dnnl::lrn_forward::primitive_desc lrn1_prim_desc = dnnl::lrn_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::lrn_across_channels,
        conv1_dst_memory.get_desc(),
        conv1_dst_memory.get_desc(),
        local1_size,
        alpha1,
        beta1,
        k1
    );

    dnnl::memory lrn1_dst_memory = dnnl::memory(lrn1_prim_desc.dst_desc(), engine);

    // AlexNet: pool1
    // {batch, 96, 55, 55} -> {batch, 96, 27, 27}
    // kernel: {3, 3}
    // strides: {2, 2}
    dnnl::memory::dims pool1_dst_tz {1, 96, 27, 27};
    dnnl::memory::dims pool1_kernel {3, 3};
    dnnl::memory::dims pool1_strides {2, 2};
    dnnl::memory::dims pool_dilation {0, 0};
    dnnl::memory::dims pool_padding {0, 0};

    dnnl::memory::desc pool1_dst_md = dnnl::memory::desc(
            {pool1_dst_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    dnnl::pooling_forward::primitive_desc pool1_pd = dnnl::pooling_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference, 
        dnnl::algorithm::pooling_max,
        lrn1_dst_memory.get_desc(), 
        pool1_dst_md, 
        pool1_strides,
        pool1_kernel, 
        pool_dilation, 
        pool_padding, 
        pool_padding
    );
    
    dnnl::memory pool1_dst_memory = dnnl::memory(pool1_pd.dst_desc(), engine);

};

} // namespace cppquamperfekt

inline cppquamperfekt::ReturnCode
cppquamperfekt::Herz::execute() {
    
    this->write_to_dnnl_memory(this->user_src.data(), this->user_src_memory);
    this->write_to_dnnl_memory(this->conv1_weights.data(), this->user_weights_memory);
    this->write_to_dnnl_memory(this->conv1_bias.data(), this->conv1_user_bias_memory);

    // Create reorder primitives between user input and conv src if needed
    if (this->conv1_prim_desc.src_desc() != this->user_src_memory.get_desc()) {
        std::cout << "Data format required by convolution different from user format!\n";
        this->conv1_src_memory = dnnl::memory(this->conv1_prim_desc.src_desc(), this->engine);
        this->net.push_back(dnnl::reorder(this->user_src_memory, this->conv1_src_memory));
        this->net_args.push_back(
            {
                {DNNL_ARG_FROM, this->user_src_memory},
                {DNNL_ARG_TO, this->conv1_src_memory}
            }
        );
    }

    if (this->conv1_prim_desc.weights_desc() != this->user_weights_memory.get_desc()) {
        std::cout << "Weights format required by convolution different from user format!\n";
        this->conv1_weights_memory = dnnl::memory(this->conv1_prim_desc.weights_desc(), this->engine);
        dnnl::reorder(this->user_weights_memory, this->conv1_weights_memory)
            .execute(this->stream, this->user_weights_memory, this->conv1_weights_memory);
    }

    // Create a convolution primitive and add it to the net
    this->net.push_back(dnnl::convolution_forward(this->conv1_prim_desc));
    this->net_args.push_back(
        {
            {DNNL_ARG_SRC, this->conv1_src_memory},
            {DNNL_ARG_WEIGHTS, this->conv1_weights_memory},
            {DNNL_ARG_BIAS, this->conv1_user_bias_memory},
            {DNNL_ARG_DST, this->conv1_dst_memory}
        }
    );

    // Create a relu primitive
    this->net.push_back(dnnl::eltwise_forward(this->relu1_prim_desc));
    this->net_args.push_back(
        {
            {DNNL_ARG_SRC, this->conv1_dst_memory},
            {DNNL_ARG_DST, this->conv1_dst_memory}
        }
    );

    this->net.push_back(dnnl::lrn_forward(this->lrn1_prim_desc));
    this->net_args.push_back(
        {
            {DNNL_ARG_SRC, this->conv1_dst_memory},
            {DNNL_ARG_DST, this->lrn1_dst_memory}
        }
    );

    this->net.push_back(dnnl::pooling_forward(this->pool1_pd));
    this->net_args.push_back(
        {
            {DNNL_ARG_SRC, this->lrn1_dst_memory},
            {DNNL_ARG_DST, this->pool1_dst_memory}
        }
    );

    return ReturnCode::valid;
}

inline cppquamperfekt::ReturnCode 
cppquamperfekt::Herz::validate_engine_kind() {
    if (this->engine.get_kind() == dnnl::engine::kind::gpu) {
        return ReturnCode::valid;
    }
    else {
        return ReturnCode::invalid;
    }   
}

inline cppquamperfekt::ReturnCode
cppquamperfekt::Herz::read_from_dnnl_memory(void* handle, dnnl::memory& memory) {
    
    if (handle == nullptr) {
        return ReturnCode::invalid;
    }

    if (memory.get_engine() != this->engine) {
        return ReturnCode::invalid;
    }

    dnnl::sycl_interop::memory_kind memory_kind = dnnl::sycl_interop::get_memory_kind(memory);
    std::size_t size = memory.get_desc().get_size();

    if (memory_kind == dnnl::sycl_interop::memory_kind::buffer) {
        sycl::buffer buffer = dnnl::sycl_interop::get_buffer<uint8_t>(memory);
        uint8_t* src_ptr = buffer.get_host_access().get_pointer();

        if(src_ptr == nullptr) {
            return ReturnCode::invalid;
        }

        for (std::size_t i = 0; i < size; ++i) {
            static_cast<uint8_t*>(handle)[i] = src_ptr[i];
        }
    }

    else if (memory_kind == dnnl::sycl_interop::memory_kind::usm) {
        uint8_t* src_ptr = static_cast<uint8_t*>(memory.get_data_handle());

        if(src_ptr == nullptr) {
            return ReturnCode::invalid;
        }

        sycl::queue queue = dnnl::sycl_interop::get_queue(this->stream);
        queue.memcpy(handle, src_ptr, size).wait();
    }

    return ReturnCode::valid;
}

inline cppquamperfekt::ReturnCode
cppquamperfekt::Herz::write_to_dnnl_memory(void* handle, dnnl::memory& memory) {

    if (handle == nullptr) {
        return ReturnCode::invalid;
    }

    if (memory.get_engine() != this->engine) {
        return ReturnCode::invalid;
    }

    dnnl::sycl_interop::memory_kind memory_kind = dnnl::sycl_interop::get_memory_kind(memory);
    std::size_t size = memory.get_desc().get_size();

    if (memory_kind == dnnl::sycl_interop::memory_kind::buffer) {
        sycl::buffer buffer = dnnl::sycl_interop::get_buffer<uint8_t>(memory);
        uint8_t* dst_ptr = buffer.get_host_access().get_pointer();

        if(dst_ptr == nullptr) {
            return ReturnCode::invalid;
        }

        for (std::size_t i = 0; i < size; ++i) {
            dst_ptr[i] = static_cast<uint8_t*>(handle)[i];
        }
    }

    else if (memory_kind == dnnl::sycl_interop::memory_kind::usm) {
        uint8_t* dst_ptr = static_cast<uint8_t*>(memory.get_data_handle());

        if(dst_ptr == nullptr) {
            return ReturnCode::invalid;
        }

        sycl::queue queue = dnnl::sycl_interop::get_queue(this->stream);
        queue.memcpy(dst_ptr, handle, size).wait();
    }

    return ReturnCode::valid;
}
