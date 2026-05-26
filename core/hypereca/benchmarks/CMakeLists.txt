function(dress_benchmark benchmark_file)
    get_filename_component(benchmark_target_name "${benchmark_file}" NAME_WE)
    add_executable("${benchmark_target_name}" "")
    target_sources("${benchmark_target_name}"
        PRIVATE
        "${benchmark_file}")
    target_link_libraries("${benchmark_target_name}" dress)
    set_property(TARGET "${benchmark_target_name}" PROPERTY CXX_STANDARD 17)
endfunction(dress_benchmark)

dress_benchmark("dedup_combine.cu")
dress_benchmark("combine_ops.cu")
dress_benchmark("copy.cu")
target_link_libraries(copy nccl gomp)
dress_benchmark("dedup.cc")
dress_benchmark("insert.cc")
dress_benchmark("prefetch.cc")
dress_benchmark("hugectr-reader.cu")
