import numpy as np

def reach_the_ketchup():
	bddl = """
(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language reach the ketchup)
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
    (And (Reach ketchup_1 {}))
  )

)
	"""
	return [bddl.format(open_amount) for open_amount in np.arange(0.2, -0.05, -0.05)]


def grasp_the_ketchup():
	return """
(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language grasp the ketchup)
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
    (And (Grasp ketchup_1))
  )

)
	"""


def lift_the_ketchup():
	bddl = """
(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language lift the ketchup over the basket)
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
    (And (Lift ketchup_1 basket_1 {}))
  )

)
	"""
	return [bddl.format(open_amount) for open_amount in np.arange(-0.3, 0.15, 0.05)]