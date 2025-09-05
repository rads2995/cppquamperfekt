add_library(cppquamperfekt INTERFACE)
add_library(cppquamperfekt::cppquamperfekt ALIAS cppquamperfekt)

target_sources(cppquamperfekt
    PUBLIC
        FILE_SET cppquamperfekt_headers
        TYPE HEADERS
        BASE_DIRS ./include
        FILES include/cppquamperfekt/cppquamperfekt.hpp
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
