import numpy as np

base_bddl = """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language turn on the stove and put the moka pot on it)
    (:regions
      (flat_stove_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.21000000000000002 0.19 -0.19 0.21000000000000002)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (frypan_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.07500000000000001 -0.275 -0.025 -0.225)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (moka_pot_init_region
          (:target kitchen_table)
          (:ranges (
              (0.025 -0.025 0.07500000000000001 0.025)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (handle_region
          (:target moka_pot_1)
      )
      (cook_region
          (:target flat_stove_1)
      )
      (knob_region
          (:target flat_stove_1)
      )
    )

  (:fixtures
    kitchen_table - kitchen_table
    flat_stove_1 - flat_stove
  )

  (:objects
    chefmate_8_frypan_1 - chefmate_8_frypan
    moka_pot_1 - moka_pot
  )

  (:obj_of_interest
    moka_pot_1
    flat_stove_1
  )

  (:init
    (On flat_stove_1 kitchen_table_flat_stove_init_region)
    (On chefmate_8_frypan_1 kitchen_table_frypan_init_region)
    (On moka_pot_1 kitchen_table_moka_pot_init_region)
  )

  (:goal
    (And {})
  )

)
"""

def reach_the_pot():
	bddl = base_bddl.format("(Reach moka_pot_1_handle_region {})")
	return [bddl.format(reach_distance) for reach_distance in np.arange(0.4, -0.0001, -0.05)]

def grasp_the_pot():
	bddl = base_bddl.format("(Grasp moka_pot_1_handle_region {})")
	return [bddl.format(grasp_amount) for grasp_amount in np.arange(0.0, 1.0001, 0.1)]

def lift_the_pot():
	bddl = base_bddl.format("(Lift moka_pot_1 flat_stove_1_cook_region {})")
	return [bddl.format(lift_distance) for lift_distance in np.arange(-0.15, 0.0001, 0.01)]

def align_the_pot_over_the_stove():
	bddl = base_bddl.format("(And (Lift moka_pot_1 flat_stove_1_cook_region 0) (Align moka_pot_1 flat_stove_1_cook_region {}))")
	return [bddl.format(move_distance) for move_distance in np.arange(0.4, -0.0001, -0.01)]

def place_the_pot_on_the_stove():
	bddl = base_bddl.format("(And (On moka_pot_1 flat_stove_1_cook_region) (Not (Grasp moka_pot_1 {})))")
	return [bddl.format(grasp_amount) for grasp_amount in np.arange(1.0, 0.5, -0.1)]

def reach_the_stove():
	bddl = base_bddl.format("(And (On moka_pot_1 flat_stove_1_cook_region) (Not (Grasp moka_pot_1)) (Or (Reach flat_stove_1_knob_region {}) (Turnon flat_stove_1 0.1)))")
	return [bddl.format(reach_distance) for reach_distance in np.arange(0.4, -0.0001, -0.05)]

def turnon_the_stove():
	bddl = base_bddl.format("(And (On moka_pot_1 flat_stove_1_cook_region) (Turnon flat_stove_1 {}))")
	return [bddl.format(reach_distance) for reach_distance in np.arange(0.0, 1.0001, 0.1)]

