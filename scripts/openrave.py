import click
import numpy as np
import openravepy as rave

from autodh import create_dh_table, DHTable, Joint

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

Rave2AutoDH = {
    rave.JointType.Hinge: Joint.Type.Revolute,
    rave.JointType.Revolute: Joint.Type.Revolute,
    rave.JointType.Prismatic: Joint.Type.Prismatic,
    rave.JointType.Slider: Joint.Type.Prismatic,
}


def get_dh_table(robot, base_link, ee_link, modified_dh, validate):
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
    if not modified_dh:
        dh = create_dh_table(joints, base_frame, ee_frame, DHTable.Convention.Standard)
    else:
        dh = create_dh_table(joints, base_frame, ee_frame, DHTable.Convention.Modified)

    # Test random configuration
    if validate:
        rand_dof_values = np.random.rand(len(joints)) * (upper - lower) + lower
        robot.SetDOFValues(rand_dof_values, indices)
        ee_transform = np.linalg.inv(base_link.GetTransform()).dot(ee_link.GetTransform())

        assert np.allclose(dh.forward(rand_dof_values), ee_transform)

    return dh


def _load_and_validate_robot(ctx, param, value):
    env = rave.Environment()
    env.StopSimulation()
    try:
        robot = env.ReadRobotURI(value)
        env.Add(robot)
    except Exception as e:
        raise click.BadParameter("couldn't load robot from \"{}\"".format(value), param=param) from e
    ctx.params['env'] = env
    return robot


def _validate_link(ctx, param, value):
    link = ctx.params['robot'].GetLink(value)
    if link is None:
        raise click.BadParameter("{}".format(value), param=param)
    return link


@click.command()
@click.argument("robot", callback=_load_and_validate_robot)
@click.argument("base_link", callback=_validate_link)
@click.argument("ee_link", callback=_validate_link)
@click.option('--modified', '-m', "modified_dh", is_flag=True, help="Compute modified DH parameters")
@click.option('--validate', '-v', "validate_result", is_flag=True, help="Validate DH parameters before returning")
@click.option("--viewer", "enable_viewer", is_flag=True, help="Enable OpenRAVE viewer")
def cli(robot, base_link, ee_link, modified_dh, validate_result, enable_viewer, env):
    """ Compute DH parameters of the ROBOT kinematic chain defined by BASE_LINK and EE_LINK.

        Example usage: `python openrave.py robots/pr2-beta-static.zae base_link \\ r_gripper_palm_link`
    """
    if enable_viewer:
        env.SetViewer("qtcoin")

    dh = get_dh_table(robot, base_link, ee_link, modified_dh, validate_result)
    click.echo(dh)

    if enable_viewer:
        input("Press any key to quit...")


if __name__ == '__main__':
    try:
        rave.RaveInitialize(load_all_plugins=True, level=rave.DebugLevel.Error)
        cli()
    finally:
        rave.RaveDestroy()
