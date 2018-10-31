import numpy as np
import openravepy as rave

from autodh import create_dh_table, DHTable, Joint

rave.RaveInitialize(load_all_plugins=True, level=rave.DebugLevel.Error)

Rave2AutoDH = {
    rave.JointType.Hinge: Joint.Type.Revolute,
    rave.JointType.Revolute: Joint.Type.Revolute,
    rave.JointType.Prismatic: Joint.Type.Prismatic,
    rave.JointType.Slider: Joint.Type.Prismatic,
}

ROBOT_FILE = "robots/pr2-beta-static.zae"
BASE_LINK_NAME = "base_link"
EE_LINK_NAME = "r_gripper_palm_link"


def main(robot_path, base_link_name, ee_link_name, enable_viewer):
    env = rave.Environment()
    env.StopSimulation()
    if enable_viewer:
        env.SetViewer("qtcoin")

    robot = env.ReadRobotURI(robot_path)
    env.Add(robot)

    base_link = robot.GetLink(base_link_name)
    ee_link = robot.GetLink(ee_link_name)
    chain = robot.GetChain(base_link.GetIndex(), ee_link.GetIndex())
    indices = [j.GetJointIndex() for j in chain if not j.IsStatic()]
    lower, upper = robot.GetDOFLimits(indices)
    dof_values = robot.GetDOFValues(indices)

    # Include zero in joint limits and set joint values to zero
    robot.SetDOFLimits(-np.pi * np.ones(len(indices)), np.pi * np.ones(len(indices)), indices)
    robot.SetDOFValues(np.zeros(len(indices)), indices)

    # Create list of Joints
    joints = []
    for j in chain:
        if j.IsStatic():
            continue
        axis = j.GetAxis()
        anchor = j.GetAnchor()
        joint_type = Rave2AutoDH[j.GetType()]
        joints.append(Joint(axis, anchor, joint_type))

    # Get base and ee frames
    base_frame = base_link.GetTransform()
    ee_frame = ee_link.GetTransform()

    # Restore joint limits and previous dof values
    robot.SetDOFLimits(lower, upper, indices)
    robot.SetDOFValues(dof_values, indices)

    # Create DH tables
    dh1 = create_dh_table(joints, base_frame, ee_frame, DHTable.Convention.Standard)
    dh2 = create_dh_table(joints, base_frame, ee_frame, DHTable.Convention.Modified)

    # Test random configuration
    rand_dof_values = np.random.rand(len(joints)) * (upper - lower) + lower
    robot.SetDOFValues(rand_dof_values, indices)
    ee_transform = np.linalg.inv(base_link.GetTransform()).dot(ee_link.GetTransform())

    print(dh1)
    print(dh2)

    assert np.allclose(dh1.forward(rand_dof_values), ee_transform)
    assert np.allclose(dh2.forward(rand_dof_values), ee_transform)


if __name__ == '__main__':
    main(ROBOT_FILE, BASE_LINK_NAME, EE_LINK_NAME, True)
    rave.RaveDestroy()
