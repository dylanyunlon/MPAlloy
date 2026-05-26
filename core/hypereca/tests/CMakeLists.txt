function(dress_test test_file)
    get_filename_component(test_target_name "${test_file}" NAME_WE)
    add_executable("${test_target_name}" "")
    target_sources("${test_target_name}"
        PRIVATE
        "${test_file}")
    target_link_libraries("${test_target_name}" dress gmock gtest)
    add_test(NAME "${test_target_name}" COMMAND "${test_target_name}")
    set_property(TARGET "${test_target_name}" PROPERTY CXX_STANDARD 17)
endfunction(dress_test)

dress_test("hello_test.cc")
dress_test("pull_test.cc")
dress_test("lookup_test.cc")
dress_test("push_test.cc")
