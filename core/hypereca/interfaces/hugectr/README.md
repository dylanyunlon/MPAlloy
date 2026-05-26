Steps to use DrESS with HugeCTR
===

* CMake
    * Add the cmake file to Modules
    * Require `dress` in `CMakeLists`
    * Add link target `dress` in `HugeCTR/src/CMakeLists`
* Coding
    * Include the `dress_embedding.hpp` in embedding creator
    * Create relative dress host embedding creator
    * Add dress in parser
