#pragma once

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
    dnnl::memory::dims dims {};  

    Herz(
        dnnl::engine::kind kind = dnnl::engine::kind::gpu,
        std::size_t index = 0,
        dnnl::stream::flags stream_flags = dnnl::stream::flags::default_flags
    ) : engine(kind, index),
        stream(engine, stream_flags) {}
};

}
