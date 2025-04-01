import numpy as np

base_bddl = """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language open the bottom drawer of the cabinet)
    (:regions
      (wooden_cabinet_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.01 -0.31 0.01 -0.29)
            )
          )
          (:yaw_rotation (
              (3.141592653589793 3.141592653589793)
            )
          )
      )
      (akita_black_bowl_init_region
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
      (plate_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.025 0.225 0.025 0.275)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (top_side
          (:target wooden_cabinet_1)
      )
      (top_region
          (:target wooden_cabinet_1)
      )
      (middle_region
          (:target wooden_cabinet_1)
      )
      (bottom_region
          (:target wooden_cabinet_1)
      )
      (top_handle
          (:target wooden_cabinet_1)
      )
      (middle_handle
          (:target wooden_cabinet_1)
      )
      (bottom_handle
          (:target wooden_cabinet_1)
      )
    )

  (:fixtures
    kitchen_table - kitchen_table
    wooden_cabinet_1 - wooden_cabinet
  )

  (:objects
    akita_black_bowl_1 - akita_black_bowl
    plate_1 - plate
  )

  (:obj_of_interest
    wooden_cabinet_1
  )

  (:init
    (On akita_black_bowl_1 kitchen_table_akita_black_bowl_init_region)
    (On plate_1 kitchen_table_plate_init_region)
    (On wooden_cabinet_1 kitchen_table_wooden_cabinet_init_region)
  )

  (:goal
    (And {})
  )

)
"""

def reach_the_cabinet():
    bddl = base_bddl.format("(Reach wooden_cabinet_1_bottom_handle {})")
    return [bddl.format(reach_distance) for reach_distance in np.arange(0.3, -0.00001, -0.05)]

def open_the_cabinet():
    bddl = base_bddl.format("(Open wooden_cabinet_1_bottom_region {})")
    return [bddl.format(open_amount) for open_amount in np.arange(0.1, 1.00001, 0.1)]
