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

def reach_the_stove():
	bddl = base_bddl.format("(Reach flat_stove_1_knob_region {})")
	return [bddl.format(reach_distance) for reach_distance in np.arange(0.4, -0.0001, -0.05)]

def turnon_the_stove():
	bddl = base_bddl.format("(Turnon flat_stove_1 {})")
	return [bddl.format(reach_distance) for reach_distance in np.arange(0.0, 1.0001, 0.1)]

def reach_the_pot():
	bddl = base_bddl.format("(And (Turnon flat_stove_1) (Reach moka_pot_1 {}))")
	return [bddl.format(reach_distance) for reach_distance in np.arange(0.4, -0.0001, -0.05)]

def grasp_the_pot():
	return base_bddl.format("(And (Turnon flat_stove_1) (Grasp moka_pot_1))")

def lift_the_pot():
	bddl = base_bddl.format("(And (Turnon flat_stove_1) (Lift moka_pot_1 flat_stove_1_cook_region {}))")
	return [bddl.format(lift_distance) for lift_distance in np.arange(-0.15, 0.1001, 0.01)]

def align_the_mug_over_the_stove():
	bddl = base_bddl.format("(And (Turnon flat_stove_1) (Align moka_pot_1 flat_stove_1_cook_region {}))")
	return [bddl.format(move_distance) for move_distance in np.arange(0.4, -0.0001, -0.01)]

def place_the_mug_on_the_stove():
	return base_bddl.format("(And (Turnon flat_stove_1) (On moka_pot_1 flat_stove_1_cook_region))")
