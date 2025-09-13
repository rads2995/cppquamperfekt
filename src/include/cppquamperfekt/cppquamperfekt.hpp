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

            // TODO: move everything after this to init() method?
            conv_weights.resize(std::accumulate(
                conv_weights_tz.begin(), 
                conv_weights_tz.end(), 
                static_cast<std::size_t>(1), 
                std::multiplies<std::size_t>()
            ));
            // TODO: parallelize this to make it faster?
            for (std::size_t i = 0; i < conv_weights.size(); ++i) {
                conv_weights[i] = std::sinf(static_cast<float>(i));
            }

            conv_bias.resize(std::accumulate(
                conv_bias_tz.begin(), 
                conv_bias_tz.end(), 
                static_cast<std::size_t>(1), 
                std::multiplies<std::size_t>()
            ));
            // TODO: parallelize this to make it faster?
            for (std::size_t i = 0; i < conv_bias.size(); ++i) {
                conv_bias[i] = std::sinf(static_cast<float>(i));
            }
        }

    ReturnCode validate_engine_kind();
    ReturnCode read_from_dnnl_memory(void* handle, dnnl::memory& memory);
    ReturnCode write_to_dnnl_memory(void* handle, dnnl::memory& memory);
    ReturnCode execute();

    dnnl::engine engine;
    dnnl::stream stream; 
    dnnl::memory::dims dims {2, 3, 4, 5}; 

    std::vector<dnnl::primitive> net_fwd, net_bwd;
    std::vector<std::unordered_map<int, dnnl::memory>> net_fwd_args, net_bwd_args;

    std::vector<float> net_src = []() {
        std::vector<float> v(32 * 3 * 227 * 227);
        // TODO: parallelize this to make it faster?
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = std::sinf(static_cast<float>(i));
        }
        return v;
    }();
    std::vector<float> net_dst{32 * 96 * 27 * 27};

    dnnl::memory::dims conv_src_tz {32, 3, 227, 227};
    dnnl::memory::dims conv_weights_tz {96, 3, 11, 11};
    dnnl::memory::dims conv_bias_tz {96};
    dnnl::memory::dims conv_dst_tz {32, 96, 55, 55};
    dnnl::memory::dims conv_strides {4, 4};
    dnnl::memory::dims conv_padding {0, 0};

    std::vector<float> conv_weights; 
    std::vector<float> conv_bias;

    // Create memory for user data
    dnnl::memory conv_user_src_memory {dnnl::memory(
        {{conv_src_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw},
        engine
    )};
    dnnl::memory conv_user_weights_memory {dnnl::memory(
        {{conv_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw},
        engine
    )};
    dnnl::memory conv_user_bias_memory {dnnl::memory(
        {{conv_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x},
        engine
    )};

    // Create memory descriptors for convolution data w/ no specified
    dnnl::memory::desc conv_src_md = dnnl::memory::desc(
        {conv_src_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any
    );
    dnnl::memory::desc conv_bias_md = dnnl::memory::desc(
        {conv_bias_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any
    );
    dnnl::memory::desc conv_weights_md = dnnl::memory::desc(
        {conv_weights_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any
    );
    dnnl::memory::desc conv_dst_md = dnnl::memory::desc(
        {conv_dst_tz}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any
    );

    // Create a convolution primitive descriptor
    dnnl::convolution_forward::primitive_desc conv_pd = dnnl::convolution_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward,
        dnnl::algorithm::convolution_direct,
        conv_src_md,
        conv_weights_md,
        conv_bias_md,
        conv_dst_md,
        conv_strides,
        conv_padding,
        conv_padding
    );

    dnnl::memory conv_src_memory = conv_user_src_memory;
    dnnl::memory conv_weights_memory = conv_user_weights_memory;
    dnnl::memory conv_dst_memory = dnnl::memory(conv_pd.dst_desc(), engine);

};

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

inline cppquamperfekt::ReturnCode
cppquamperfekt::Herz::execute() {
    this->write_to_dnnl_memory(this->net_src.data(), this->conv_user_src_memory);
    this->write_to_dnnl_memory(static_cast<void*>(this->conv_weights.data()), this->conv_user_weights_memory);
    this->write_to_dnnl_memory(this->conv_bias.data(), this->conv_user_bias_memory);

    // Create reorder primitives between user input and conv src if needed
    if (this->conv_pd.src_desc() != this->conv_user_src_memory.get_desc()) {
        this->conv_src_memory = dnnl::memory(this->conv_pd.src_desc(), this->engine);
        this->net_fwd.push_back(dnnl::reorder(this->conv_user_src_memory, this->conv_src_memory));
        this->net_fwd_args.push_back(
            {
                {DNNL_ARG_FROM, this->conv_user_src_memory},
                {DNNL_ARG_TO, this->conv_src_memory}
            }
        );
    }

    if (this->conv_pd.weights_desc() != this->conv_user_weights_memory.get_desc()) {
        this->conv_weights_memory = dnnl::memory(this->conv_pd.weights_desc(), this->engine);
        this->net_fwd.push_back(dnnl::reorder(this->conv_user_weights_memory, this->conv_weights_memory));
        this->net_fwd_args.push_back(
            {
                {DNNL_ARG_FROM, this->conv_user_weights_memory},
                {DNNL_ARG_TO, this->conv_weights_memory}
            }
        );
    }

    // Finally, create a convolution primitive
    this->net_fwd.push_back(dnnl::convolution_forward(this->conv_pd));
    this->net_fwd_args.push_back(
        {
            {DNNL_ARG_SRC, this->conv_src_memory},
            {DNNL_ARG_WEIGHTS, this->conv_weights_memory},
            {DNNL_ARG_BIAS, this->conv_user_bias_memory},
            {DNNL_ARG_DST, this->conv_dst_memory}
        }
    );

    

    return ReturnCode::valid;
}
