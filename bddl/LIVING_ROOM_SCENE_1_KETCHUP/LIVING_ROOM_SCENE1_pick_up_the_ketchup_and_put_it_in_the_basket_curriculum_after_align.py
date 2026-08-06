import numpy as np

base_bddl = """
(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language pick up the ketchup and put it in the basket)
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
      (alphabet_soup_init_region
          (:target living_room_table)
          (:ranges (
              (0.025 -0.125 0.07500000000000001 -0.07500000000000001)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (cream_cheese_init_region
          (:target living_room_table)
          (:ranges (
              (-0.175 0.034999999999999996 -0.125 0.08499999999999999)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (tomato_sauce_init_region
          (:target living_room_table)
          (:ranges (
              (0.07500000000000001 -0.225 0.125 -0.17500000000000002)
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
              (-0.225 -0.175 -0.17500000000000002 -0.125)
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
      (grasp_region
          (:target ketchup_1)
      )
    )

  (:fixtures
    living_room_table - living_room_table
  )

  (:objects
    alphabet_soup_1 - alphabet_soup
    cream_cheese_1 - cream_cheese
    tomato_sauce_1 - tomato_sauce
    ketchup_1 - ketchup
    basket_1 - basket
  )

  (:obj_of_interest
    ketchup_1
    basket_1
  )

  (:init
    (On alphabet_soup_1 living_room_table_alphabet_soup_init_region)
    (On cream_cheese_1 living_room_table_cream_cheese_init_region)
    (On tomato_sauce_1 living_room_table_tomato_sauce_init_region)
    (On ketchup_1 living_room_table_ketchup_init_region)
    (On basket_1 living_room_table_basket_init_region)
  )

  (:goal
    (And {})
  )

)
"""

def lower_the_ketchup_to_the_basket():
    bddl = base_bddl.format("(And (Proximity ketchup_1 basket_1_contain_region {}) (Align ketchup_1 basket_1_contain_region))")
    return [bddl.format(align_distance) for align_distance in np.arange(0.4, -0.00001, -0.01)]

def put_the_ketchup_in_the_basket():
    return base_bddl.format("(And (In ketchup_1 basket_1_contain_region) (Not (Grasp ketchup_1)))")