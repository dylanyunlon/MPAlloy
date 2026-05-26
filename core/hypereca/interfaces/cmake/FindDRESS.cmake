set(DRESS_INC_PATHS
    /usr/include
    /usr/local/include
    ${DRESS_DIR}/include
    )

set(DRESS_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    ${DRESS_DIR}/lib
    ${DRESS_DIR}/build
    )

find_path(DRESS_INCLUDE_DIR NAMES dress/dress.h PATHS ${DRESS_INC_PATHS})
find_library(DRESS_LIBRARIES NAMES dress PATHS ${DRESS_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DRESS DEFAULT_MSG DRESS_INCLUDE_DIR DRESS_LIBRARIES)

if (DRESS_FOUND)
  message(STATUS "Found DRESS    (include: ${DRESS_INCLUDE_DIR}, library: ${DRESS_LIBRARIES})")
  mark_as_advanced(DRESS_INCLUDE_DIR DRESS_LIBRARIES)
endif ()
