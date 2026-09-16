#!/usr/bin/env python3
"""
MoveIt 2 Kinematics & Motion Planning Client for Agilex Piper Arm.

Manages base TF placement in world frame, IK feasibility queries,
and collision-free multi-stage trajectory planning via MoveIt 2 services.
"""

import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped, Quaternion, Transform, Vector3
from sensor_msgs.msg import JointState, MultiDOFJointState
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
    RobotState,
    WorkspaceParameters
)
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionFK,
    GetMotionPlan,
    GetCartesianPath,
    GetStateValidity
)
from shape_msgs.msg import SolidPrimitive


def yaw_to_quaternion_msg(yaw_deg: float) -> Quaternion:
    yaw_rad = math.radians(yaw_deg)
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw_rad / 2.0)
    q.w = math.cos(yaw_rad / 2.0)
    return q


class PiperMoveItPlanner:
    """ROS 2 MoveIt client managing base placement, IK solving, and path planning for Piper."""

    def __init__(self, node: Node, group_name: str = "arm", world_frame: str = "world"):
        self.node = node
        self.group_name = group_name
        self.world_frame = world_frame
        self.base_frame = "base_link"
        self.ee_link = "link6"

        # Static / Dynamic TF broadcaster for moving the robot base in the world frame
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)

        # MoveIt Service Clients
        self.ik_client = self.node.create_client(GetPositionIK, "/compute_ik")
        self.fk_client = self.node.create_client(GetPositionFK, "/compute_fk")
        self.plan_client = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.cartesian_client = self.node.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.validity_client = self.node.create_client(GetStateValidity, "/check_state_validity")

    def wait_for_services(self, timeout_sec: float = 10.0) -> bool:
        """Ensure all MoveIt 2 services are ready."""
        clients = [
            ("IK", self.ik_client),
            ("FK", self.fk_client),
            ("MotionPlan", self.plan_client),
            ("Cartesian", self.cartesian_client),
            ("StateValidity", self.validity_client)
        ]
        start_t = time.time()
        for name, client in clients:
            rem = max(0.5, timeout_sec - (time.time() - start_t))
            if not client.wait_for_service(timeout_sec=rem):
                self.node.get_logger().error(f"MoveIt service '{name}' ({client.srv_name}) not available.")
                return False
        return True

    def get_multi_dof_joint_state(self, base_pose: Optional[Tuple[float, float, float, float]] = None) -> MultiDOFJointState:
        """
        Construct MultiDOFJointState connecting world to base_link via world_joint.
        Enforces det(R) = +1 assertion.
        """
        bp = base_pose if base_pose is not None else getattr(self, "base_pose", (0.0, 0.0, 0.0, 0.0))
        bx, by, bz, yaw_deg = float(bp[0]), float(bp[1]), float(bp[2]), float(bp[3])

        md = MultiDOFJointState()
        md.header.frame_id = self.world_frame
        md.header.stamp = self.node.get_clock().now().to_msg()
        md.joint_names = ["world_joint"]

        yaw_rad = math.radians(yaw_deg)
        qz = math.sin(yaw_rad / 2.0)
        qw = math.cos(yaw_rad / 2.0)

        # Assert det(R) = +1 at construction
        R = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad),  math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        det_R = float(np.linalg.det(R))
        assert abs(det_R - 1.0) < 1e-6, f"Assertion failed: det(R) = {det_R} != +1.0"

        tf = Transform()
        tf.translation = Vector3(x=float(bx), y=float(by), z=float(bz))
        tf.rotation = Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw))
        md.transforms = [tf]
        return md

    def set_robot_base_pose(self, base_x: float, base_y: float, h: float, base_yaw_deg: float):
        """Broadcast transform from world frame to robot base_link frame and record base_pose."""
        self.base_pose = (float(base_x), float(base_y), float(h), float(base_yaw_deg))
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = float(base_x)
        t.transform.translation.y = float(base_y)
        t.transform.translation.z = float(h)
        t.transform.rotation = yaw_to_quaternion_msg(base_yaw_deg)

        self.tf_broadcaster.sendTransform(t)

    def world_to_base_pose(self, pose_world: Pose, base_x: float, base_y: float, h: float, base_yaw_deg: float) -> Pose:
        """Transform a 3D pose from world frame into robot base_link frame."""
        yaw_rad = math.radians(base_yaw_deg)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)

        # Translation relative to base
        dx = pose_world.position.x - base_x
        dy = pose_world.position.y - base_y
        dz = pose_world.position.z - h

        # Rotate by -yaw around Z
        bx = dx * c + dy * s
        by = -dx * s + dy * c
        bz = dz

        p_base = Pose()
        p_base.position.x = float(bx)
        p_base.position.y = float(by)
        p_base.position.z = float(bz)

        # Orientation quaternion transformed into base_link frame: q_base = q_base_yaw^-1 * q_world
        qw_inv = math.cos(-yaw_rad / 2.0)
        qz_inv = math.sin(-yaw_rad / 2.0)

        # Quaternion multiplication: (qw_inv, 0, 0, qz_inv) * (q.w, q.x, q.y, q.z)
        qx = pose_world.orientation.x
        qy = pose_world.orientation.y
        qz = pose_world.orientation.z
        qw = pose_world.orientation.w

        p_base.orientation.w = qw_inv * qw - qz_inv * qz
        p_base.orientation.x = qw_inv * qx + qz_inv * qy
        p_base.orientation.y = qw_inv * qy - qz_inv * qx
        p_base.orientation.z = qw_inv * qz + qz_inv * qw

        return p_base

    def solve_ik(
        self,
        target_pose_world: Pose,
        base_x: float = 0.0,
        base_y: float = 0.0,
        h: float = 0.0,
        base_yaw_deg: float = 0.0,
        timeout_sec: float = 0.5,
        avoid_collisions: bool = True
    ) -> Tuple[bool, Optional[JointState], float]:
        """
        Query MoveIt /compute_ik for target Cartesian pose.
        Returns: (success, joint_state, solve_time_ms)
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.robot_state.is_diff = False
        js = JointState()
        js.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
        js.position = [0.0] * 8
        req.ik_request.robot_state.joint_state = js
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.timeout.sec = int(timeout_sec)
        req.ik_request.timeout.nanosec = int((timeout_sec % 1.0) * 1e9)

        # Transform pose into base_link frame
        p_base = self.world_to_base_pose(target_pose_world, base_x, base_y, h, base_yaw_deg)
        req.ik_request.robot_state.multi_dof_joint_state = self.get_multi_dof_joint_state((base_x, base_y, h, base_yaw_deg))

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        pose_stamped.pose = p_base

        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.ik_link_name = self.ee_link

        t0 = time.time()
        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec + 1.0)
        dt_ms = (time.time() - t0) * 1000.0

        if future.done() and future.result() is not None:
            res = future.result()
            success = (res.error_code.val == 1)  # 1 = SUCCESS
            return success, res.solution.joint_state, dt_ms
        return False, None, dt_ms

    def plan_to_pose(
        self,
        target_pose_world: Pose,
        base_x: float = 0.0,
        base_y: float = 0.0,
        h: float = 0.0,
        base_yaw_deg: float = 0.0,
        start_joint_state: Optional[JointState] = None,
        planner_id: str = "RRTConnect",
        num_planning_attempts: int = 5,
        allowed_planning_time: float = 3.0,
        include_orientation: bool = False,
        orientation_tolerance: float = 0.15
    ) -> Tuple[bool, Optional[dict], float]:
        """
        Plan collision-free path to Cartesian target pose via OMPL.
        If include_orientation is True, adds OrientationConstraint for 6-DOF pose goal.
        Returns: (success, plan_metrics, planning_time_sec)
        """
        req = GetMotionPlan.Request()
        mreq = req.motion_plan_request
        mreq.group_name = self.group_name
        mreq.num_planning_attempts = num_planning_attempts
        mreq.allowed_planning_time = allowed_planning_time
        mreq.planner_id = planner_id
        mreq.max_velocity_scaling_factor = 0.5
        mreq.max_acceleration_scaling_factor = 0.5

        # Explicit start state (default to zero/home configuration)
        if start_joint_state is None:
            js = JointState()
            js.header.frame_id = self.base_frame
            js.header.stamp = self.node.get_clock().now().to_msg()
            js.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            js.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            mreq.start_state.joint_state = js
        else:
            mreq.start_state.joint_state = start_joint_state
        mreq.start_state.multi_dof_joint_state = self.get_multi_dof_joint_state((base_x, base_y, h, base_yaw_deg))
        mreq.start_state.is_diff = False

        # Transform target pose into base_link frame
        p_base = self.world_to_base_pose(target_pose_world, base_x, base_y, h, base_yaw_deg)

        # Workspace bounding volume around robot base
        mreq.workspace_parameters.header.frame_id = self.base_frame
        mreq.workspace_parameters.min_corner.x = -2.0
        mreq.workspace_parameters.min_corner.y = -2.0
        mreq.workspace_parameters.min_corner.z = -1.0
        mreq.workspace_parameters.max_corner.x = 2.0
        mreq.workspace_parameters.max_corner.y = 2.0
        mreq.workspace_parameters.max_corner.z = 2.0

        # Goal constraints in base_link frame:
        # PositionConstraint: end-effector link6 position within 2cm tolerance sphere around p_base.position
        goal_constraint = Constraints()
        goal_constraint.name = "goal_position_tolerance"

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.SPHERE
        box.dimensions = [0.02]  # 2cm tolerance sphere (radius = 0.02m)
        pc.constraint_region.primitives.append(box)

        sphere_pose = Pose()
        sphere_pose.position = p_base.position
        sphere_pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(sphere_pose)
        pc.weight = 1.0
        goal_constraint.position_constraints.append(pc)

        if include_orientation:
            oc = OrientationConstraint()
            oc.header.frame_id = self.base_frame
            oc.link_name = self.ee_link
            oc.orientation = p_base.orientation
            oc.absolute_x_axis_tolerance = orientation_tolerance
            oc.absolute_y_axis_tolerance = orientation_tolerance
            oc.absolute_z_axis_tolerance = orientation_tolerance
            oc.parameterization = OrientationConstraint.XYZ_EULER_ANGLES
            oc.weight = 1.0
            goal_constraint.orientation_constraints.append(oc)

        mreq.goal_constraints.append(goal_constraint)

        t0 = time.time()
        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=allowed_planning_time + 2.0)
        plan_duration = time.time() - t0

        if future.done() and future.result() is not None:
            res = future.result()
            success = (res.motion_plan_response.error_code.val == 1)
            if success:
                trajectory = res.motion_plan_response.trajectory
                num_pts = len(trajectory.joint_trajectory.points)
                
                # Calculate trajectory joint path length
                path_length = 0.0
                pts = trajectory.joint_trajectory.points
                for i in range(1, len(pts)):
                    p1 = np.array(pts[i - 1].positions)
                    p2 = np.array(pts[i].positions)
                    path_length += float(np.linalg.norm(p2 - p1))

                metrics = {
                    'num_waypoints': num_pts,
                    'path_length_rad': path_length,
                    'trajectory_duration_sec': pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9 if pts else 0.0,
                    'planning_time_sec': res.motion_plan_response.planning_time,
                    'final_joint_state': pts[-1].positions if pts else None,
                    'waypoints': [list(p.positions) for p in pts]
                }
                return True, metrics, plan_duration
        return False, None, plan_duration

    def plan_to_joint_goal(
        self,
        target_joint_positions: list,
        start_joint_state: Optional[JointState] = None,
        planner_id: str = "RRTConnect",
        num_planning_attempts: int = 5,
        allowed_planning_time: float = 3.0,
        tolerance: float = 0.01
    ) -> Tuple[bool, Optional[dict], float]:
        """
        Plan collision-free path to explicit joint-space target via OMPL.
        Returns: (success, plan_metrics, planning_time_sec)
        """
        req = GetMotionPlan.Request()
        mreq = req.motion_plan_request
        mreq.group_name = self.group_name
        mreq.num_planning_attempts = num_planning_attempts
        mreq.allowed_planning_time = allowed_planning_time
        mreq.planner_id = planner_id
        mreq.max_velocity_scaling_factor = 0.5
        mreq.max_acceleration_scaling_factor = 0.5

        if start_joint_state is None:
            js = JointState()
            js.header.frame_id = self.base_frame
            js.header.stamp = self.node.get_clock().now().to_msg()
            js.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            js.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            mreq.start_state.joint_state = js
        else:
            mreq.start_state.joint_state = start_joint_state
        mreq.start_state.multi_dof_joint_state = self.get_multi_dof_joint_state()
        mreq.start_state.is_diff = False

        goal_constraint = Constraints()
        goal_constraint.name = "goal_joint_constraints"
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        for i, jname in enumerate(joint_names):
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(target_joint_positions[i])
            jc.tolerance_above = tolerance
            jc.tolerance_below = tolerance
            jc.weight = 1.0
            goal_constraint.joint_constraints.append(jc)
        mreq.goal_constraints.append(goal_constraint)

        t0 = time.time()
        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=allowed_planning_time + 2.0)
        plan_duration = time.time() - t0

        if future.done() and future.result() is not None:
            res = future.result()
            success = (res.motion_plan_response.error_code.val == 1)
            if success:
                trajectory = res.motion_plan_response.trajectory
                num_pts = len(trajectory.joint_trajectory.points)
                path_length = 0.0
                pts = trajectory.joint_trajectory.points
                for i in range(1, len(pts)):
                    p1 = np.array(pts[i - 1].positions)
                    p2 = np.array(pts[i].positions)
                    path_length += float(np.linalg.norm(p2 - p1))

                metrics = {
                    'num_waypoints': num_pts,
                    'path_length_rad': path_length,
                    'trajectory_duration_sec': pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9 if pts else 0.0,
                    'planning_time_sec': res.motion_plan_response.planning_time,
                    'final_joint_state': pts[-1].positions if pts else None,
                    'waypoints': [list(p.positions) for p in pts]
                }
                return True, metrics, plan_duration
        return False, None, plan_duration

    def check_state_validity(self, joint_positions: list) -> Tuple[bool, list]:
        """
        Check if a given robot joint configuration is collision-free in the current planning scene.
        Returns: (is_valid, contacts_list)
        """
        req = GetStateValidity.Request()
        req.robot_state.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        req.robot_state.joint_state.position = [float(p) for p in joint_positions]
        req.robot_state.multi_dof_joint_state = self.get_multi_dof_joint_state()
        req.group_name = self.group_name

        future = self.validity_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if future.done() and future.result() is not None:
            res = future.result()
            contacts = []
            for c in res.contacts:
                contacts.append({
                    'body_1': c.contact_body_1,
                    'body_2': c.contact_body_2,
                    'depth': c.depth
                })
            return res.valid, contacts
        return False, []

    def compute_fk(
        self,
        joint_positions: list,
        link_name: str = "link6",
        frame_id: str = "world"
    ) -> Tuple[bool, Optional[Pose]]:
        """
        Compute forward kinematics for a given joint configuration.
        Returns: (success, pose)
        """
        req = GetPositionFK.Request()
        req.header.frame_id = frame_id
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.fk_link_names = [link_name]
        req.robot_state.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        req.robot_state.joint_state.position = [float(p) for p in joint_positions]
        req.robot_state.multi_dof_joint_state = self.get_multi_dof_joint_state()

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if future.done() and future.result() is not None:
            res = future.result()
            if res.error_code.val == 1 and res.pose_stamped:
                return True, res.pose_stamped[0].pose
        return False, None

    def plan_cartesian_approach(
        self,
        start_pose_world: Pose,
        goal_pose_world: Pose,
        base_x: float = 0.0,
        base_y: float = 0.0,
        h: float = 0.0,
        base_yaw_deg: float = 0.0,
        step_size: float = 0.01,
        avoid_collisions: bool = True
    ) -> Tuple[bool, float, float]:
        """
        Plan straight-line Cartesian approach in base_link frame.
        Returns: (success, fraction_achieved, execution_time_sec)
        """
        p_start_base = self.world_to_base_pose(start_pose_world, base_x, base_y, h, base_yaw_deg)
        p_goal_base = self.world_to_base_pose(goal_pose_world, base_x, base_y, h, base_yaw_deg)

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.group_name = self.group_name
        req.link_name = self.ee_link
        req.waypoints = [p_start_base, p_goal_base]
        req.max_step = step_size
        req.jump_threshold = 0.0
        req.avoid_collisions = avoid_collisions

        t0 = time.time()
        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        dt = time.time() - t0

        if future.done() and future.result() is not None:
            res = future.result()
            fraction = float(res.fraction)
            success = (fraction >= 0.90 and res.error_code.val == 1)
            return success, fraction, dt
        return False, 0.0, dt
