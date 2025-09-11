#include <iostream>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cppquamperfekt/cppquamperfekt.hpp>

int main() {

    dnnl::engine engine {dnnl::engine::kind::gpu, 0};

    dnnl::stream stream {engine, dnnl::stream::flags::default_flags};

    dnnl::memory::dims tz_dims {2, 3, 4, 5};

    const size_t N = std::accumulate(
        tz_dims.begin(), tz_dims.end(), static_cast<size_t>(1), std::multiplies<size_t>()
    );

    dnnl::memory::desc mem_d(
        tz_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw
    );

    dnnl::memory mem = dnnl::sycl_interop::make_memory(
        mem_d, engine, dnnl::sycl_interop::memory_kind::buffer
    );

    sycl::buffer sycl_buffer = dnnl::sycl_interop::get_buffer<float>(mem);

    sycl::queue queue = dnnl::sycl_interop::get_queue(stream);

    queue.submit([&](sycl::handler &cgh) {
        auto a = sycl_buffer.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            int idx = static_cast<int>(i[0]);
            a[idx] = (idx % 2) ? - static_cast<float>(idx) : static_cast<float>(idx);
        });
    });

    auto relu_pd = dnnl::eltwise_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward,
        dnnl::algorithm::eltwise_relu,
        mem_d,
        mem_d,
        0.0f    
    );

    auto relu = dnnl::eltwise_forward(relu_pd);

    relu.execute(
        stream,
        {
            {DNNL_ARG_SRC, mem},
            {DNNL_ARG_DST, mem}
        }
    );
    stream.wait();

    auto host_acc = sycl_buffer.get_host_access();
    for (size_t i = 0; i < N; i++) {
        float exp_value = (i % 2) ? 0.0f : i;
        if (host_acc[i] != static_cast<float>(exp_value))
            std::cout << "Error! Negative value after the ReLU execution!" << '\n';
    }

    return 0;
}
