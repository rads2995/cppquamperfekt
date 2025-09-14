#include <iostream>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cppquamperfekt/cppquamperfekt.hpp>

int main() {

    cppquamperfekt::Herz herz;

    herz.execute();

    return 0;
}
