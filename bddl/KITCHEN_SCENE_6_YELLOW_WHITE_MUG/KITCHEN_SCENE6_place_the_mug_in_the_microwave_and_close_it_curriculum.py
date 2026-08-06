import numpy as np

base_bddl = """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language put the white yellow mug in the microwave and close it)
    (:regions
      (microwave_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.01 0.33999999999999997 0.01 0.36)
            )
          )
          (:yaw_rotation (
              (0 0)
            )
          )
      )
      (white_yellow_mug_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.025 -0.025 0.025 0.025)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (porcelain_mug_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.125 -0.275 -0.07500000000000001 -0.225)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (porcelain_mug_front_region
          (:target kitchen_table)
          (:ranges (
              (-0.05 -0.3 0.05 -0.2)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (top_side
          (:target microwave_1)
      )
      (heating_region
          (:target microwave_1)
      )
      (handle_region
          (:target microwave_1)
      )
    )

  (:fixtures
    kitchen_table - kitchen_table
    microwave_1 - microwave
  )

  (:objects
    porcelain_mug_1 - porcelain_mug
    white_yellow_mug_1 - white_yellow_mug
  )

  (:obj_of_interest
    white_yellow_mug_1
    microwave_1
  )

  (:init
    (On porcelain_mug_1 kitchen_table_porcelain_mug_init_region)
    (On white_yellow_mug_1 kitchen_table_white_yellow_mug_init_region)
    (On microwave_1 kitchen_table_microwave_init_region)
    (Open microwave_1)
  )

  (:goal
    (And {})
  )

)
"""

def reach_the_mug():
	bddl = base_bddl.format("(Reach white_yellow_mug_1 {})")
	return [bddl.format(reach_distance) for reach_distance in np.arange(0.2, -0.0001, -0.05)]

def contact_the_mug():
    return base_bddl.format("(Contact white_yellow_mug_1)")

def grasp_the_mug():
    # return base_bddl.format("(Grasp white_yellow_mug_1)")
    bddl = base_bddl.format("(Grasp white_yellow_mug_1 {})")
    return [bddl.format(grasp_amount) for grasp_amount in np.arange(0.1, 1.00001, 0.1)]

def lift_the_mug():
	bddl = base_bddl.format("(Lift white_yellow_mug_1 microwave_1_heating_region {})")
	return [bddl.format(lift_distance) for lift_distance in np.arange(-0.15, 0.0001, 0.01)]

def move_the_mug_to_the_microwave():
	bddl = base_bddl.format("(Proximity white_yellow_mug_1 microwave_1_heating_region {})")
	return [bddl.format(move_distance) for move_distance in np.arange(0.4, -0.0001, -0.01)]

def place_the_mug_in_the_microwave():
	bddl =  base_bddl.format("(And (In white_yellow_mug_1 microwave_1_heating_region) (Not (Grasp white_yellow_mug_1 {})))")
	return [bddl.format(grasp_amount) for grasp_amount in np.arange(1.0, 0.5, -0.1)]

def place_the_mug_and_reach_the_microwave_handle():
	bddl = base_bddl.format("(And (In white_yellow_mug_1 microwave_1_heating_region) (Or (Reach microwave_1_handle_region {}) (Close microwave_1 0.0)))")
	return [bddl.format(reach_distance) for reach_distance in np.arange(0.4, -0.0001, -0.02)]

def place_the_mug_and_close_the_microwave():
	bddl = base_bddl.format("(And (In white_yellow_mug_1 microwave_1_heating_region) (Close microwave_1 {}))")
	return [bddl.format(close_amount) for close_amount in np.arange(0.1, 1.0001, 0.1)]
