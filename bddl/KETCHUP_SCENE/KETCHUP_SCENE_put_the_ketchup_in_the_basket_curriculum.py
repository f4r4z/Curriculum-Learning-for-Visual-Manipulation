import numpy as np

base_bddl = """
(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language align the ketchup over the basket)
    (:regions
      (basket_init_region
          (:target living_room_table)
          (:ranges (
              (-0.01 0.25 0.01 0.27)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (ketchup_init_region
          (:target living_room_table)
          (:ranges (
              (-0.01 -0.175 0.01 -0.125)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (contain_region
          (:target basket_1)
      )
    )

  (:fixtures
    living_room_table - living_room_table
  )

  (:objects
    ketchup_1 - ketchup
    basket_1 - basket
  )

  (:obj_of_interest
    ketchup_1
    basket_1
  )

  (:init
    (On ketchup_1 living_room_table_ketchup_init_region)
    (On basket_1 living_room_table_basket_init_region)
  )

  (:goal
    (And {})
  )

)
"""

def reach_the_ketchup():
	bddl = base_bddl.format("(Reach ketchup_1 {})")
	return [bddl.format(open_amount) for open_amount in np.arange(0.15, -0.0001, -0.05)]

def grasp_the_ketchup():
	return base_bddl.format("(Grasp ketchup_1)")

def lift_the_ketchup():
	bddl = base_bddl.format("(Lift ketchup_1 basket_1 {})")
	return [bddl.format(open_amount) for open_amount in np.arange(-0.15, 0.1001, 0.01)]

def align_the_ketchup():
	bddl = base_bddl.format("(And (Align ketchup_1 basket_1 {}) (Lift ketchup_1 basket_1 0.1) (Grasp ketchup_1))")
	return [bddl.format(align_distance) for align_distance in np.arange(0.4, -0.0001, -0.05)]

def put_the_ketchup_in_the_basket():
	bddl = base_bddl.format("(And (In ketchup_1 basket_1_contain_region) (Not (Grasp ketchup_1)))")
	return [bddl.format(align_distance) for align_distance in np.arange(0.4, -0.0001, -0.05)]