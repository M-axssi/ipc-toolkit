# BVH (https://github.com/geometryprocessing/SimpleBVH)
# License: MIT

if(TARGET simple_bvh::simple_bvh)
    return()
endif()

message(STATUS "Third-party: creating target 'simple_bvh::simple_bvh'")

include(CPM)
CPMAddPackage("gh:M-axssi/SimpleBVH#db244ebdd4ea348849a9f2b6ef7e908ddbf75ff4")