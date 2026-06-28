#
# Kalypsso two-fluid hydrodynamics solver
#
set(EXE_NAME "solver_godunov_five_eq")
add_executable(${EXE_NAME} solver_godunov_five_eq.cpp)

target_include_directories(${EXE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
                                              ${CMAKE_BINARY_DIR}/src)

target_link_libraries(${EXE_NAME} PRIVATE kalypsso_warnings)

target_link_libraries(
  ${EXE_NAME}
  PRIVATE kalypsso::core
          kalypsso::log
          kalypsso::p4est_wrap
          kalypsso::config
          kalypsso::godunov_five_eq
          Kokkos::kokkos
          kalypsso::monitoring)

if(KALYPSSO_CORE_USE_CNPY)
  target_link_libraries(${EXE_NAME} PRIVATE cnpy::cnpy)
endif(KALYPSSO_CORE_USE_CNPY)

if(KALYPSSO_CORE_USE_CPPTRACE)
  target_link_libraries(${EXE_NAME} PRIVATE cpptrace::cpptrace)
endif(KALYPSSO_CORE_USE_CPPTRACE)

configure_file(test_droplet_advection_2d.ini test_droplet_advection_2d.ini COPYONLY)
configure_file(test_richtmyer_meshkov_2d.ini test_richtmyer_meshkov_2d.ini COPYONLY)
configure_file(test_richtmyer_meshkov_3d.ini test_richtmyer_meshkov_3d.ini COPYONLY)

configure_file(test_triple_point_2d.ini test_triple_point_2d.ini COPYONLY)
configure_file(test_shock_bubble_air_r22_2d.ini test_shock_bubble_air_r22_2d.ini COPYONLY)
configure_file(test_shock_bubble_air_r22_2d_multi.ini test_shock_bubble_air_r22_2d_multi.ini
               COPYONLY)
configure_file(test_shock_bubble_air_r22_3d.ini test_shock_bubble_air_r22_3d.ini COPYONLY)
configure_file(test_shock_bubble_air_helium_2d.ini test_shock_bubble_air_helium_2d.ini COPYONLY)
configure_file(test_shock_bubble_air_helium_2d_restart.ini
               test_shock_bubble_air_helium_2d_restart.ini COPYONLY)
configure_file(test_shock_bubble_air_helium_3d.ini test_shock_bubble_air_helium_3d.ini COPYONLY)
configure_file(test_shock_bubble_air_helium_2d_multi.ini test_shock_bubble_air_helium_2d_multi.ini
               COPYONLY)
configure_file(test_shock_bubble_air_helium_2d_multi_thinc.ini
               test_shock_bubble_air_helium_2d_multi_thinc.ini COPYONLY)
configure_file(test_shock_bubble_air_helium_3d_multi.ini test_shock_bubble_air_helium_3d_multi.ini
               COPYONLY)
configure_file(test_shock_bubble_air_water_2d.ini test_shock_bubble_air_water_2d.ini COPYONLY)
configure_file(test_shock_bubble_air_water_2d_thinc.ini test_shock_bubble_air_water_2d_thinc.ini
               COPYONLY)
configure_file(test_shock_bubble_air_water_2d_mossier.ini
               test_shock_bubble_air_water_2d_mossier.ini COPYONLY)
configure_file(test_shock_two_bubbles_air_water_2d_thinc.ini
               test_shock_two_bubbles_air_water_2d_thinc.ini COPYONLY)

configure_file(test_static_droplet_2d.ini test_static_droplet_2d.ini COPYONLY)

configure_file(test_shock_bubble_air_r22_2d_thinc.ini test_shock_bubble_air_r22_2d_thinc.ini
               COPYONLY)

configure_file(test_spherical_implosion_2d_tin_air.ini test_spherical_implosion_2d_tin_air.ini
               COPYONLY)

configure_file(test_triple_point_2d_thinc.ini test_triple_point_2d_thinc.ini COPYONLY)

configure_file(test_underwater_explosion_2d.ini test_underwater_explosion_2d.ini COPYONLY)
configure_file(test_underwater_explosion_3d.ini test_underwater_explosion_3d.ini COPYONLY)
configure_file(test_underwater_explosion_2d_thinc.ini test_underwater_explosion_2d_thinc.ini
               COPYONLY)
configure_file(test_underwater_explosion_3d_thinc.ini test_underwater_explosion_3d_thinc.ini
               COPYONLY)

configure_file(test_shock_bubble_molybdenum_MORB_2d.ini test_shock_bubble_molybdenum_MORB_2d.ini
               COPYONLY)

#
# Gas-Water shock tube
#
add_subdirectory(gas-water-shock-tube)
configure_file(test_shock_tube_2d_cocchi.ini test_shock_tube_2d_cocchi.ini COPYONLY)

#
# shock Air-Helium plot
#
add_subdirectory(test_shock_bubble_air_helium_2d)

#
# circle advection test
#
configure_file(test_circle_advection_2d_vortex.ini test_circle_advection_2d_vortex.ini COPYONLY)

#
# shock tubes
#
add_subdirectory(test_two_fluid_shock_tube_2d)
