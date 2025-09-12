#pragma once

#include <iostream>
#include <cstdint>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace cppquamperfekt {

enum class ReturnCode : int {
    valid = 0,
    invalid
};

struct Herz {  

    dnnl::engine engine;
    dnnl::stream stream; 
    dnnl::memory::dims dims {2, 3, 4, 5};  

    Herz(
        dnnl::engine::kind kind = dnnl::engine::kind::gpu,
        std::size_t index = 0,
        dnnl::stream::flags stream_flags = dnnl::stream::flags::default_flags
    ) : engine(kind, index),
        stream(engine, stream_flags) {
            if (
                validate_engine_kind() != ReturnCode::valid
            ) {
                std::cout << "Failed to validate engine!" << std::endl;
            }
        }

    ReturnCode validate_engine_kind();
    ReturnCode read_from_dnnl_memory(void* handle, dnnl::memory& memory);
    ReturnCode write_to_dnnl_memory(void* handle, dnnl::memory& memory);
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
    
    // TODO: check if input memory belongs to the engine (e.g., mem.get_engine())
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

        sycl::queue queue = dnnl::sycl_interop::get_queue(dnnl::stream(this->engine));
        queue.memcpy(handle, src_ptr, size).wait();
    }

    else {
        return ReturnCode::invalid;
    }
}

inline cppquamperfekt::ReturnCode
cppquamperfekt::Herz::write_to_dnnl_memory(void* handle, dnnl::memory& memory) {

    return ReturnCode::invalid;

}
