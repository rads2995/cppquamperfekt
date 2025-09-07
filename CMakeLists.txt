cmake_minimum_required(VERSION 3.25)

project(cppquamperfekt)

add_subdirectory(src)
add_subdirectory(bin)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
