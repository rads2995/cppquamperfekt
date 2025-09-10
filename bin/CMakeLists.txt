add_executable(cppquamperfekt_main)

target_compile_options(cppquamperfekt_main
    PRIVATE
        ${CXX_HARDENED_FLAGS} -fPIE
)

target_link_options(cppquamperfekt_main
    PRIVATE
        ${LINKER_HARDENED_FLAGS} -pie
)

target_sources(cppquamperfekt_main
    PRIVATE
        main.cpp
)

target_link_libraries(cppquamperfekt_main
    PRIVATE
        cppquamperfekt::cppquamperfekt
)
