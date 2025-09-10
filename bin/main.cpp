#include <vector>
#include <iostream>

#include <sycl/sycl.hpp>

#include <cppquamperfekt/cppquamperfekt.hpp>

int main() {

    constexpr int size = 1'000'000;
    
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


    // USM allocator for data of type int in shared memory
    using vec_alloc = sycl::usm_allocator<int, sycl::usm::alloc::shared>;

    // Create allocator for device associated with queue
    vec_alloc allocator(q);

    // Create std::vector using this allocator
    std::vector<int, vec_alloc> a(size, allocator), b(size, allocator), c(size, allocator);

    // Get pointer(s) to vector(s) data for access in kernel
    auto A = a.data();
    auto B = b.data();
    auto C = c.data();

    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i;
        c[i] = i;
    }

    sycl::event event = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(size),
            [=](sycl::id<1> idx) {
                C[idx] = A[idx] + B[idx];
            });
    });
    event.wait();
    
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    
    std::cout << "Elapsed time: " << (end - start) / 1.0e9 << " seconds\n";

    return 0;
}
