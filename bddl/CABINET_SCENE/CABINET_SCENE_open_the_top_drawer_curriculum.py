import numpy as np

def reach_the_cabinet():
	bddl = """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
	(:domain robosuite)
	(:language reach a distance to the top handle of the cabinet)
		(:regions
			(wooden_cabinet_init_region
					(:target kitchen_table)
					(:ranges (
							(-0.01 -0.21 0.01 -0.19)
						)
					)
					(:yaw_rotation (
							(3.141592653589793 3.141592653589793)
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
	)

	(:obj_of_interest
		wooden_cabinet_1
	)

	(:init
		(On wooden_cabinet_1 kitchen_table_wooden_cabinet_init_region)
	)

	(:goal
		(Reach wooden_cabinet_1_top_handle {})
	)

)
	"""
	return [bddl.format(open_amount) for open_amount in np.arange(0.2, -0.05, -0.05)]


def partial_open_the_cabinet():
	bddl = """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
	(:domain robosuite)
	(:language partially open the top drawer of the cabinet)
		(:regions
			(wooden_cabinet_init_region
					(:target kitchen_table)
					(:ranges (
							(-0.01 -0.21 0.01 -0.19)
						)
					)
					(:yaw_rotation (
							(3.141592653589793 3.141592653589793)
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
		)

	(:fixtures
		kitchen_table - kitchen_table
		wooden_cabinet_1 - wooden_cabinet
	)

	(:objects
	)

	(:obj_of_interest
		wooden_cabinet_1
	)

	(:init
		(On wooden_cabinet_1 kitchen_table_wooden_cabinet_init_region)
	)

	(:goal
		(Open wooden_cabinet_1_top_region {})
	)

)
	"""
	return [bddl.format(open_amount) for open_amount in np.arange(0.1, 1.0, 0.1)]


def open_the_cabinet():
	return """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
	(:domain robosuite)
	(:language open the top drawer of the cabinet)
		(:regions
			(wooden_cabinet_init_region
					(:target kitchen_table)
					(:ranges (
							(-0.01 -0.21 0.01 -0.19)
						)
					)
					(:yaw_rotation (
							(3.141592653589793 3.141592653589793)
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
		)

	(:fixtures
		kitchen_table - kitchen_table
		wooden_cabinet_1 - wooden_cabinet
	)

	(:objects
	)

	(:obj_of_interest
		wooden_cabinet_1
	)

	(:init
		(On wooden_cabinet_1 kitchen_table_wooden_cabinet_init_region)
	)

	(:goal
		(Open wooden_cabinet_1_top_region)
	)

)
	"""