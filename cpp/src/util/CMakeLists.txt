add_library(util
        CudaUtil.cpp
        Math.cpp
        StaticVars.cpp
        ${MY_TINYEXPR_DIR}/tinyexpr.c
        )
