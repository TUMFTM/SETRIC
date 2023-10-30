import numpy as np
import torch


def coord_to_coord(
    new_coord: tuple,
    old_coord: tuple,
    agent_p: tuple,
    agent_v: tuple = None,
    agent_a: tuple = None,
):
    # new_coords: tupel containing the new coordinate system in UTM coordinates (UTM_X, UTM_Y, UTM_ANGLE)
    # old_coords: tupel containing the old coordinate system in UTM coordinates (UTM_X, UTM_Y, UTM_ANGLE)
    # agent_p: tupel containing the agents position and angle in the old coordinate system (x, y, angle)
    # agent_v: tupel containing the agents velocity components in the old coordinate system (v_x, v_y)
    # agent_a: tupel containing the agents acceleration components in the old coordinate system (v_x, v_y)
    # unpack input
    new_coord_utm_x = new_coord[0]
    new_coord_utm_y = new_coord[1]
    new_coord_utm_angle = new_coord[2]
    old_coord_utm_x = old_coord[0]
    old_coord_utm_y = old_coord[1]
    old_coord_utm_angle = old_coord[2]
    agent_old_coord_x = agent_p[0]
    agent_old_coord_y = agent_p[1]
    agent_old_coord_angle = agent_p[2]

    # Calculate length and utm rotation of vector c from new to old coordinate origin
    c_utm_x = old_coord_utm_x - new_coord_utm_x
    c_utm_y = old_coord_utm_y - new_coord_utm_y
    c = torch.sqrt(torch.square(c_utm_x) + torch.square(c_utm_y))
    c_utm_angle = torch.angle(torch.complex(real=c_utm_x, imag=c_utm_y))
    # Calculate length and utm rotation of vector p from origin of old coordinate system to agent position in old
    # coordinate system
    p = torch.sqrt(torch.square(agent_old_coord_x) + torch.square(agent_old_coord_y))
    p_old_angle = torch.angle(
        torch.complex(real=agent_old_coord_x, imag=agent_old_coord_y)
    )
    p_utm_angle = p_old_angle + old_coord_utm_angle
    # Calculate Translation
    agent_new_coord_x = c * torch.cos(
        c_utm_angle - new_coord_utm_angle
    ) + p * torch.cos(p_utm_angle - new_coord_utm_angle)
    agent_new_coord_y = c * torch.sin(
        c_utm_angle - new_coord_utm_angle
    ) + p * torch.sin(p_utm_angle - new_coord_utm_angle)
    # Calculate rotation
    agent_new_coord_angle = (
        agent_old_coord_angle + old_coord_utm_angle - new_coord_utm_angle
    )
    if agent_v is None and agent_a is None:
        return (agent_new_coord_x, agent_new_coord_y, agent_new_coord_angle)
    elif agent_v is not None and agent_a is not None:
        # unpack velocity
        agent_old_coord_v_x = agent_v[0]
        agent_old_coord_v_y = agent_v[1]
        # Calculate length and utm rotation of vector v
        v = torch.sqrt(
            torch.square(agent_old_coord_v_x) + torch.square(agent_old_coord_v_y)
        )
        v_old_angle = torch.angle(
            torch.complex(real=agent_old_coord_v_x, imag=agent_old_coord_v_y)
        )
        v_utm_angle = v_old_angle + old_coord_utm_angle
        agent_new_coord_v_x = v * torch.cos(v_utm_angle - new_coord_utm_angle)
        agent_new_coord_v_y = v * torch.sin(v_utm_angle - new_coord_utm_angle)
        # unpack acceleration
        agent_old_coord_a_x = agent_a[0]
        agent_old_coord_a_y = agent_a[1]
        # Calculate length and utm rotation of vector a
        a = torch.sqrt(
            torch.square(agent_old_coord_a_x) + torch.square(agent_old_coord_a_y)
        )
        a_old_angle = torch.angle(
            torch.complex(real=agent_old_coord_a_x, imag=agent_old_coord_a_y)
        )
        a_utm_angle = a_old_angle + old_coord_utm_angle
        agent_new_coord_a_x = a * torch.cos(a_utm_angle - new_coord_utm_angle)
        agent_new_coord_a_y = a * torch.sin(a_utm_angle - new_coord_utm_angle)
        return (
            (agent_new_coord_x, agent_new_coord_y, agent_new_coord_angle),
            (agent_new_coord_v_x, agent_new_coord_v_y),
            (agent_new_coord_a_x, agent_new_coord_a_y),
        )
    else:
        raise Exception(
            "The case that only one of the velocity and acceleration tupel is not None is not implemented"
        )


def abs_to_rel_coord(curr_pos, curr_orient, abs_coord):
    """Transform absolute coordinate to car-relative coordinate.

    Args:
        curr_pos (Union[tuple, list, np_array]):    current global position of the car.
                                                    format: (x,y)
        curr_orient (float): current global orientation of the car in rad
        abs_coord (Union[tuple, list, np_array]):   absolute coordinate that has to be
                                                    transformed.
                                                    Coord-List also possible

    Returns:
        np_array: transformed car-relative coordinate in the form (x,y)
    """
    abs_coord = np.array(abs_coord)
    curr_pos = np.array(curr_pos)
    if abs_coord.ndim == 2:
        abs_coord = np.transpose(abs_coord)
        curr_pos = curr_pos.reshape(2, 1)
    rot_mat = np.array(
        [
            [np.cos(curr_orient), np.sin(curr_orient)],
            [-np.sin(curr_orient), np.cos(curr_orient)],
        ]
    )
    rel_coord = abs_coord - curr_pos
    rel_coord = np.matmul(rot_mat, rel_coord)
    if rel_coord.ndim == 2:
        rel_coord = np.transpose(rel_coord)
    return rel_coord
