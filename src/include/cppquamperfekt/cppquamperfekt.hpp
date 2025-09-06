#pragma once

#include <expected>

#include <sycl/sycl.hpp>

namespace cppquamperfekt {
    
enum class ReturnCode : int {
    valid = 0,
    invalid
};

template<typename T>
auto test_func(T input) -> std::expected<T, ReturnCode> {
    return input * input;
}

}
