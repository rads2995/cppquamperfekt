#include <cassert>

#include <cppquamperfekt/cppquamperfekt.hpp>

int main() {
    
    int num = 2;
    auto value = cppquamperfekt::test_func(num);
    assert(value == 4);    
    return 0;
}
