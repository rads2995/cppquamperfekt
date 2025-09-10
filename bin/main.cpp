#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cppquamperfekt/cppquamperfekt.hpp>

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
            auto dst = buffer.get_host_access();
            uint8_t *dst_ptr = dst.get_pointer();
            if (!dst_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                dst_ptr[i] = ((uint8_t *)handle)[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
            if (!dst_ptr)
                throw std::runtime_error("get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)handle)[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(dst_ptr, handle, size).wait();
            }
        }
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        void *mapped_ptr = mem.map_data();
        if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
        mem.unmap_data(mapped_ptr);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
            auto src = buffer.get_host_access();
            uint8_t *src_ptr = src.get_pointer();
            if (!src_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                ((uint8_t *)handle)[i] = src_ptr[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *src_ptr = (uint8_t *)mem.get_data_handle();
            if (!src_ptr)
                throw std::runtime_error("get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    ((uint8_t *)handle)[i] = src_ptr[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(handle, src_ptr, size).wait();
            }
        }
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        void *mapped_ptr = mem.map_data();
        if (mapped_ptr) std::memcpy(handle, mapped_ptr, size);
        mem.unmap_data(mapped_ptr);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        if (!src) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            ((uint8_t *)handle)[i] = src[i];
        return;
    }

    assert(!"not expected");
}

int main() {

    dnnl::engine engine {dnnl::engine::kind::gpu, 0};

    dnnl::stream engine_stream {engine};

    // Create a 4D tensor in NHWC format
    constexpr int N = 1, H = 13, W = 13, C = 3;

    // Compute physical strides for each dimension
    constexpr int stride_N = H * W * C;
    constexpr int stride_H = W * C;
    constexpr int stride_W = C;
    constexpr int stride_C = 1;

    // Auxiliary function that maps logical index to the physical offset
    auto offset = [=](int n, int h, int w, int c) {
        return n * stride_N + h * stride_H + w * stride_W + N * stride_C;
    };

    constexpr int image_size = N * H * W * C;

    // Allocato a buffer for the image
    std::vector<float> image(image_size);

    // Initialize the image with some values
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    int off = offset(
                            n, h, w, c); // Get the physical offset of a pixel
                    image[off] = -std::cos(off / 10.f);
                }

    auto src_md = dnnl::memory::desc(
            {N, C, H, W}, // logical dims, the order is defined by a primitive
            dnnl::memory::data_type::f32,   // tensor's data type
            dnnl::memory::format_tag::nhwc  // memory format, NHWC in this case
    );

    // src_mem contains a copy of image after write_to_dnnl_memory function
    auto src_mem = dnnl::memory(src_md, engine);
    write_to_dnnl_memory(image.data(), src_mem);

    auto dst_mem = dnnl::memory(src_md, engine);

    auto relu_pd = dnnl::eltwise_forward::primitive_desc(
        engine, // an engine the primitive will be created for
        dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_relu,
        src_md, // source memory descriptor for an operation to work on
        src_md, // destination memory descriptor for an operation to work on
        0.f, // alpha parameter means negative slope in case of ReLU
        0.f // beta parameter is ignored in case of ReLU
    );

    // ReLU primitive
    auto relu = dnnl::eltwise_forward(relu_pd);

    relu.execute(engine_stream, // The execution stream
            {
                    // A map with all inputs and outputs
                    {DNNL_ARG_SRC, src_mem}, // Source tag and memory obj
                    {DNNL_ARG_DST, dst_mem}, // Destination tag and memory obj
            });

    // Wait the stream to complete the execution
    engine_stream.wait();

    std::vector<float> relu_image(image_size);
    read_from_dnnl_memory(relu_image.data(), dst_mem);

    // Check the results
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    int off = offset(
                            n, h, w, c); // get the physical offset of a pixel
                    float expected = image[off] < 0
                            ? 0.f
                            : image[off]; // expected value
                    if (relu_image[off] != expected) {
                        std::cout << "At index(" << n << ", " << c << ", " << h
                                  << ", " << w << ") expect " << expected
                                  << " but got " << relu_image[off]
                                  << std::endl;
                        throw std::logic_error("Accuracy check failed.");
                    }
                }

    return 0;
}
