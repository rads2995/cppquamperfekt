#include <vector>
#include <iostream>

#include <sycl/sycl.hpp>

#include <cppquamperfekt/cppquamperfekt.hpp>

int main() {

    sycl::device d;
    try {
        d = sycl::device(sycl::gpu_selector_v);
    } catch (sycl::exception const& e) {
        d = sycl::device(sycl::cpu_selector_v);
    }

    sycl::property_list properties{sycl::property::queue::enable_profiling()};
    sycl::queue q = sycl::queue(d, properties);

    std::cout
        << "Platform: "
        << q.get_device().get_platform().get_info<sycl::info::platform::name>()
        << "\n";
    
    constexpr int num_ints = 1024 * 1024;
    constexpr size_t num_bytes = num_ints * sizeof(int);
    constexpr int alignment = 8;

    // Alloc memory on host
    auto src = std::aligned_alloc(alignment, num_bytes);
    std::memset(src, 1, num_bytes);

    // Alloc memory on device
    auto dst = sycl::malloc_device<int>(num_ints, q);
    q.memset(dst, 0, num_bytes).wait();

    // Copy from host to device
    auto event = q.memcpy(dst, src, num_bytes);
    event.wait();

    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    
    std::cout << "Elapsed time: " << (end - start) / 1.0e9 << " seconds\n";

    sycl::free(dst, q);

    return 0;
}
