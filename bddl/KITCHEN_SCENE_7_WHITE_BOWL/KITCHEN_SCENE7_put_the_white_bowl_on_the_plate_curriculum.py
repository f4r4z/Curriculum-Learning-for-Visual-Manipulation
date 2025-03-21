import numpy as np

base_bddl = """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language put the white bowl on the plate)
    (:regions
      (microwave_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.01 -0.26 0.01 -0.24)
            )
          )
          (:yaw_rotation (
              (3.141592653589793 3.141592653589793)
            )
          )
      )
      (plate_init_region
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
      (plate_right_region
          (:target kitchen_table)
          (:ranges (
              (-0.05 0.05 0.05 0.15000000000000002)
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
    )

  (:fixtures
    kitchen_table - kitchen_table
    microwave_1 - microwave
  )

  (:objects
    white_bowl_1 - white_bowl
    plate_1 - plate
  )

  (:obj_of_interest
    white_bowl_1
    plate_1
  )

  (:init
    (On white_bowl_1 microwave_1_top_side)
    (On microwave_1 kitchen_table_microwave_init_region)
    (Close microwave_1)
    (On plate_1 kitchen_table_plate_init_region)
  )

  (:goal
    (And {})
  )

)
"""

def reach_the_bowl():
    bddl = base_bddl.format("(Reach white_bowl_1 {})")
    return [bddl.format(reach_distance) for reach_distance in np.arange(0.3, -0.00001, -0.05)]

def contact_the_bowl():
	return base_bddl.format("(Contact white_bowl_1)")
    
def grasp_the_bowl():
    bddl = base_bddl.format("(Grasp white_bowl_1 {})")
    return [bddl.format(grasp_amount) for grasp_amount in np.arange(0.0, 1.00001, 0.1)]
    
def lift_the_bowl():
    bddl = base_bddl.format("(Lift white_bowl_1 plate_1 {})")
    return [bddl.format(lift_distance) for lift_distance in np.arange(-0.15, 0.03001, 0.01)]
    
def align_the_bowl():
    bddl = base_bddl.format("(And (Lift white_bowl_1 plate_1 0) (Align white_bowl_1 plate_1 {}))")
    return [bddl.format(align_distance) for align_distance in np.arange(0.3, -0.00001, -0.01)]

def move_the_bowl_to_the_plate():
    bddl = base_bddl.format("(And (Proximity white_bowl_1 plate_1 {}) (Grasp white_bowl_1))")
    return [bddl.format(align_distance) for align_distance in np.arange(0.3, -0.00001, -0.01)]

def put_the_bowl_on_the_plate():
	return base_bddl.format("(On white_bowl_1 plate_1)")
