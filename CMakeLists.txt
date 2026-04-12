set(GODUNOV_FIVE_EQ_SRCS
    border_conditions/BCShockBubble.cpp
    border_conditions/BCShockBubble.h
    border_conditions/FillOutside.cpp
    border_conditions/FillOutside.h
    common.h
    eos/EosWrapper.h
    eos/eos_utils.h
    init/init.cpp
    init/init.h
    init/InitDropletAdvection.cpp
    init/InitDropletAdvection.h
    init/InitStaticDroplet.cpp
    init/InitStaticDroplet.h
    init/InitRichtmyerMeshkov.cpp
    init/InitRichtmyerMeshkov.h
    init/InitShockBubble.cpp
    init/InitShockBubble.h
    init/InitTriplePoint.cpp
    init/InitTriplePoint.h
    init/InitTwoFluidShockTube.cpp
    init/InitTwoFluidShockTube.h
    init/InitUnderwaterExplosion.cpp
    init/InitUnderwaterExplosion.h
    models/FiveEq.cpp
    models/FiveEq.h
    models/HydroState.h
    models/RiemannSolvers.h
    models/utils.h
    profiling.h
    scheme/ComputeDtFiveEqFunctor.cpp
    scheme/ComputeDtFiveEqFunctor.h
    scheme/ComputeFluxesAndStoreFunctor.cpp
    scheme/ComputeFluxesAndStoreFunctor.h
    scheme/ComputeFluxesAndStoreTHINCFunctor.cpp
    scheme/ComputeFluxesAndStoreTHINCFunctor.h
    scheme/ComputeLimitedSlopesFunctor.cpp
    scheme/ComputeLimitedSlopesFunctor.h
    scheme/ComputeRefineFlags.cpp
    scheme/ComputeRefineFlags.h
    scheme/ConvertToPrimitivesVariablesFunctor.cpp
    scheme/ConvertToPrimitivesVariablesFunctor.h
    scheme/GodunovImplemBase.h
    scheme/GodunovImplemV0.cpp
    scheme/GodunovImplemV0.h
    scheme/ReadFluxesAndConservativeUpdateFunctor.cpp
    scheme/ReadFluxesAndConservativeUpdateFunctor.h
    SolverGodunovFiveEq.cpp
    SolverGodunovFiveEq.h
    utils/ComputeDerivedQuantities.h
    utils/IntegralMixingMonitor.cpp
    utils/IntegralMixingMonitor.h
    utils/utils_aux.h)

add_library(godunov_five_eq ${GODUNOV_FIVE_EQ_SRCS})
# target_compile_options(godunov_five_eq PUBLIC "-G")
target_include_directories(godunov_five_eq PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src
                                                  ${CMAKE_SOURCE_DIR}/external/better-enums)

target_link_libraries(godunov_five_eq PRIVATE kalypsso_warnings)

target_link_libraries(godunov_five_eq PUBLIC Kokkos::kokkos kalypsso::log kalypsso::core
                                             kalypsso::mpiUtils kalypsso::p4est_wrap)

# declare library alias
add_library(kalypsso::godunov_five_eq ALIAS godunov_five_eq)

# application main
add_subdirectory(app)
