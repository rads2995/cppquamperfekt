include(FetchContent)

FetchContent_Declare(
  oneDNN
  GIT_REPOSITORY https://github.com/uxlfoundation/oneDNN.git
  GIT_TAG        80a3a8e745d2f0186e674b0af9332fd6e074c94f # v3.9.1
)
FetchContent_MakeAvailable(oneDNN)

add_library(cppquamperfekt INTERFACE)
add_library(cppquamperfekt::cppquamperfekt ALIAS cppquamperfekt)

target_sources(cppquamperfekt
    PUBLIC
        FILE_SET cppquamperfekt_headers
        TYPE HEADERS
        BASE_DIRS ./include
        FILES include/cppquamperfekt/cppquamperfekt.hpp
)

target_link_libraries(cppquamperfekt
    INTERFACE
        dnnl
)

install(
    TARGETS cppquamperfekt
    EXPORT cppquamperfektTargets
    FILE_SET cppquamperfekt_headers
)

install(
    EXPORT cppquamperfektTargets
    DESTINATION share/cmake/cppquamperfekt
    NAMESPACE cppquamperfekt::
)

install(
    FILES
        ./cmake/cppquamperfektConfig.cmake
    DESTINATION
        share/cmake/cppquamperfekt
)
