#include <iostream>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cppquamperfekt/cppquamperfekt.hpp>

int main() {

    cppquamperfekt::Herz herz;

    herz.execute();

    // const std::size_t N = std::accumulate(
    //     herz.dims.begin(), herz.dims.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>()
    // );

    // auto usm_buffer = static_cast<float*>(sycl::malloc_shared(N * sizeof(float),
    //     dnnl::sycl_interop::get_device(herz.engine), dnnl::sycl_interop::get_context(herz.engine)
    // ));

    // dnnl::memory::desc mem_d(
    //     herz.dims, 
    //     dnnl::memory::data_type::f32, 
    //     dnnl::memory::format_tag::nchw
    // );

    // dnnl::memory mem = dnnl::sycl_interop::make_memory(
    //     mem_d, herz.engine, dnnl::sycl_interop::memory_kind::usm, usm_buffer
    // );

    // sycl::queue queue = dnnl::sycl_interop::get_queue(herz.stream);

    // auto fill_e = queue.submit([&](sycl::handler &cgh) {
    //     cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
    //         int idx = static_cast<int>(i[0]);
    //         usm_buffer[idx] = (idx % 2) ? static_cast<float>(-idx) : static_cast<float>(idx);
    //     });
    // });

    // auto relu_pd = dnnl::eltwise_forward::primitive_desc(
    //     herz.engine,
    //     dnnl::prop_kind::forward,
    //     dnnl::algorithm::eltwise_relu,
    //     mem_d,
    //     mem_d,
    //     0.0f    
    // );

    // auto relu = dnnl::eltwise_forward(relu_pd);

    // auto relu_e = dnnl::sycl_interop::execute(
    //     relu,
    //     herz.stream,
    //     {
    //         {DNNL_ARG_SRC, mem},
    //         {DNNL_ARG_DST, mem}
    //     },
    //     {fill_e}
    // );
    // relu_e.wait();

    // sycl::free(static_cast<void*>(usm_buffer), dnnl::sycl_interop::get_context(herz.engine));

    return 0;
}
