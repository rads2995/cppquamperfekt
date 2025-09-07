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

    int* data = sycl::malloc_shared<int>(1024, q);
    sycl::event event = q.parallel_for(1024, 
        [=](sycl::id<1> idx) {
            data[idx] = idx;
        });
    event.wait();

    for (int i = 0; i < 1024; i++) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    
    std::cout << "Elapsed time: " << (end - start) / 1.0e9 << " seconds\n";

    sycl::free(data, q);

    return 0;
}
