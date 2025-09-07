include(CTest)
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        52eb8108c5bdec04579160ae17225d66034bd723 # release-1.17.0
)
FetchContent_MakeAvailable(googletest)

target_compile_options(gtest PRIVATE -w)
target_compile_options(gtest_main PRIVATE -w)
target_compile_options(gmock PRIVATE -w)
target_compile_options(gmock_main PRIVATE -w)

add_executable(cppquamperfekt_test)

target_sources(cppquamperfekt_test
    PRIVATE
        test.cpp
)

target_link_libraries(cppquamperfekt_test
    PRIVATE
        cppquamperfekt::cppquamperfekt
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(cppquamperfekt_test)
