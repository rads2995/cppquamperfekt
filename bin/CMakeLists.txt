# find_package(cppquamperfekt REQUIRED)

add_executable(cppquamperfekt_main)

target_sources(cppquamperfekt_main
    PRIVATE
        main.cpp
)

target_link_libraries(cppquamperfekt_main
    PRIVATE
        cppquamperfekt::cppquamperfekt
)
