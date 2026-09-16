import subprocess
import numpy as np
import urdf_parser_py.urdf as urdf
import PyKDL as kdl
from pathlib import Path


def build_kdl_chain(urdf_xacro_path: str = None):
    if urdf_xacro_path is None:
        # Default fallback to environment / standard location
        urdf_xacro_path = "piper_ros/src/piper_moveit/piper_with_gripper_moveit/config/piper.urdf.xacro"
    
    xml = subprocess.check_output(["xacro", str(urdf_xacro_path)]).decode("utf-8")
    robot = urdf.URDF.from_xml_string(xml)

    joint_map = {j.child: j for j in robot.joints}
    link = "link6"
    chain_joints = []
    while link != "base_link":
        j = joint_map[link]
        chain_joints.append(j)
        link = j.parent
    chain_joints.reverse()

    chain = kdl.Chain()
    joint_limits = []
    for j in chain_joints:
        origin = j.origin
        xyz = origin.xyz if origin and origin.xyz else [0, 0, 0]
        rpy = origin.rpy if origin and origin.rpy else [0, 0, 0]
        R_orig = kdl.Rotation.RPY(*rpy)
        p_orig = kdl.Vector(*xyz)
        f = kdl.Frame(R_orig, p_orig)
        if j.type in ("revolute", "continuous"):
            local_axis = j.axis if j.axis else [1, 0, 0]
            v_local = kdl.Vector(local_axis[0], local_axis[1], local_axis[2])
            v_parent_axis = R_orig * v_local
            kdl_joint = kdl.Joint(j.name, p_orig, v_parent_axis, kdl.Joint.RotAxis)
            joint_limits.append((float(j.limit.lower if j.limit else -np.pi), float(j.limit.upper if j.limit else np.pi)))
        else:
            kdl_joint = kdl.Joint(j.name, kdl.Joint.None_)
        chain.addSegment(kdl.Segment(j.child, kdl_joint, f))

    fk_solver = kdl.ChainFkSolverPos_recursive(chain)
    vik_solver = kdl.ChainIkSolverVel_wdls(chain)
    vik_solver.setLambda(0.01)

    return chain, joint_limits, fk_solver, vik_solver

