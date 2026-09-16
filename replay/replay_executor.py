#!/usr/bin/env python3
"""
AgileX Piper Trajectory Replay Executor.
Communicates with FollowJointTrajectory action server.
Performs NO planning, NO inverse kinematics, and NO collision checking.
Ingests frozen replay bundles, synthesizes fixed time parameterization
(REPLAY_DT_SEC = 1.75), enforces zero boundary velocities/accelerations,
and logs achieved joint states and tracking errors.
"""

import os
import sys
import time
import json
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

# Trajectory replay constants
REPLAY_DT_SEC = 1.75
MAX_VELOCITY_SCALING = 0.15
MAX_ACCELERATION_SCALING = 0.15
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_ACTION_NAME = "/arm_controller/follow_joint_trajectory"

class ReplayExecutor(Node):
    def __init__(self, action_name=DEFAULT_ACTION_NAME):
        super().__init__("trajectory_replay_executor")
        self.action_name = action_name
        self._action_client = ActionClient(self, FollowJointTrajectory, self.action_name)
        
        self.latest_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10
        )
        self.get_logger().info(f"ReplayExecutor initialized. Target action: {self.action_name}")

    def _joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_server(self, timeout_sec=10.0):
        self.get_logger().info(f"Waiting for action server '{self.action_name}'...")
        ready = self._action_client.wait_for_server(timeout_sec=timeout_sec)
        if not ready:
            self.get_logger().error(f"Action server '{self.action_name}' timed out after {timeout_sec}s!")
        else:
            self.get_logger().info(f"Action server '{self.action_name}' is online.")
        return ready

    def get_current_joint_positions(self, timeout_sec=3.0):
        start = time.time()
        while self.latest_joint_state is None and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_joint_state is None:
            return None
        
        pos_dict = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        try:
            return np.array([pos_dict[j] for j in JOINT_NAMES])
        except KeyError as e:
            self.get_logger().error(f"Joint state missing required arm joint: {e}")
            return None

    def move_to_configuration(self, target_positions, duration_sec=3.0):
        """Helper to move the arm safely to a configuration (e.g. home/reset)."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        
        curr_pos = self.get_current_joint_positions()
        if curr_pos is None:
            curr_pos = np.zeros(6)
            
        p0 = JointTrajectoryPoint()
        p0.positions = curr_pos.tolist()
        p0.velocities = [0.0] * 6
        p0.accelerations = [0.0] * 6
        p0.time_from_start = Duration(sec=0, nanosec=0)
        
        p1 = JointTrajectoryPoint()
        p1.positions = list(target_positions)
        p1.velocities = [0.0] * 6
        p1.accelerations = [0.0] * 6
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        p1.time_from_start = Duration(sec=sec, nanosec=nanosec)
        
        goal.trajectory.points = [p0, p1]
        
        send_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return False
        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        return res_future.result().result.error_code == 0

    def execute_trajectory(self, waypoints, dt=REPLAY_DT_SEC, path_tolerance=None, goal_tolerance=None):
        """
        Executes a sequence of waypoints.
        waypoints: list of 6-float joint configurations
        dt: time per waypoint
        path_tolerance: list or dict of tolerances per joint
        goal_tolerance: list or dict of tolerances per joint
        """
        nw = len(waypoints)
        if nw < 2:
            raise ValueError(f"Need at least 2 waypoints, got {nw}")
            
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        
        # Build path tolerance if specified
        if path_tolerance is not None:
            for idx, j_name in enumerate(JOINT_NAMES):
                tol = JointTolerance()
                tol.name = j_name
                tol.position = float(path_tolerance[idx] if isinstance(path_tolerance, (list, np.ndarray)) else path_tolerance)
                tol.velocity = 0.0
                tol.acceleration = 0.0
                goal.path_tolerance.append(tol)

        # Build goal tolerance if specified
        if goal_tolerance is not None:
            for idx, j_name in enumerate(JOINT_NAMES):
                tol = JointTolerance()
                tol.name = j_name
                tol.position = float(goal_tolerance[idx] if isinstance(goal_tolerance, (list, np.ndarray)) else goal_tolerance)
                tol.velocity = 0.0
                tol.acceleration = 0.0
                goal.goal_tolerance.append(tol)

        # Build trajectory points with zero velocity/acceleration at each waypoint
        for k in range(nw):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in waypoints[k]]
            pt.velocities = [0.0] * 6
            pt.accelerations = [0.0] * 6
            
            t_s = k * dt
            sec = int(t_s)
            nanosec = int(round((t_s - sec) * 1e9))
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            pt.time_from_start = Duration(sec=sec, nanosec=nanosec)
            goal.trajectory.points.append(pt)

        max_tracking_error = np.zeros(6)
        feedback_count = 0
        
        def feedback_callback(feedback_msg):
            nonlocal max_tracking_error, feedback_count
            fb = feedback_msg.feedback
            if len(fb.error.positions) == 6:
                err = np.abs(np.array(fb.error.positions))
                max_tracking_error = np.maximum(max_tracking_error, err)
                feedback_count += 1

        total_expected_duration = (nw - 1) * dt
        self.get_logger().info(f"Sending trajectory: {nw} waypoints, duration: {total_expected_duration:.2f} s")
        
        send_future = self._action_client.send_goal_async(goal, feedback_callback=feedback_callback)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal was REJECTED by action server!")
            return {
                "accepted": False,
                "error_code": -1,
                "error_string": "Goal rejected by action server",
                "max_tracking_error_rad": max_tracking_error,
                "max_tracking_error_deg": np.degrees(max_tracking_error),
                "feedback_count": feedback_count
            }

        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        action_result = res_future.result()
        
        result_payload = {
            "accepted": True,
            "error_code": action_result.result.error_code,
            "error_string": action_result.result.error_string,
            "max_tracking_error_rad": max_tracking_error,
            "max_tracking_error_deg": np.degrees(max_tracking_error),
            "feedback_count": feedback_count
        }
        return result_payload

def run_simulation_dry_run(bundles_dir="replay/bundles"):
    import glob
    bundle_files = sorted(glob.glob(os.path.join(bundles_dir, "bundle_*.json")))
    if not bundle_files:
        print(f"[ERROR] No bundles found in {bundles_dir}")
        sys.exit(1)
        
    replay_bundles = []
    for bf in bundle_files:
        with open(bf, "r") as f:
            for b in json.load(f):
                if b["mode"] == "REPLAY":
                    replay_bundles.append(b)

    print("================================================================================")
    print(" SIMULATION DRY RUN")
    print("================================================================================")
    print(f"Total REPLAY bundles to evaluate: {len(replay_bundles)}")
    print(f"Fixed time per waypoint: REPLAY_DT_SEC = {REPLAY_DT_SEC:.2f} s")
    print("Zero boundary velocities and accelerations enforced at every waypoint.")
    print("Tracking error threshold: < 0.01 rad (action status SUCCESSFUL required).\n")

    rclpy.init()
    executor_node = ReplayExecutor()
    
    if not executor_node.wait_for_server(timeout_sec=10.0):
        print("[HALT] Action server /arm_controller/follow_joint_trajectory not available!")
        rclpy.shutdown()
        sys.exit(1)

    overall_pass = True
    bundle_results = []
    
    for idx, b in enumerate(replay_bundles, 1):
        cid = b["case_id"]
        arm = b["arm"]
        wps = b["waypoints"]
        nw = len(wps)
        duration_s = (nw - 1) * REPLAY_DT_SEC
        
        print(f"\n[{idx:02d}/{len(replay_bundles):02d}] Executing Bundle: {cid} | Arm: {arm}")
        print(f"     Waypoints: {nw} | Motion Duration: {duration_s:.2f} s")
        
        # Reset to home before running trajectory
        executor_node.move_to_configuration([0.0]*6, duration_sec=1.5)
        time.sleep(0.2)
        
        res = executor_node.execute_trajectory(wps, dt=REPLAY_DT_SEC)
        
        err_code = res["error_code"]
        err_str = res["error_string"]
        max_err_rad = res["max_tracking_error_rad"]
        max_err_deg = res["max_tracking_error_deg"]
        peak_err_rad = np.max(max_err_rad)
        peak_joint_idx = int(np.argmax(max_err_rad))
        
        status_str = "SUCCESSFUL" if err_code == 0 else f"FAILED (error_code={err_code})"
        
        print(f"     Action Result:   {status_str} | error_string: '{err_str}'")
        print(f"     Feedback cycles: {res['feedback_count']}")
        print(f"     Max Tracking Errors by Joint:")
        for j in range(6):
            print(f"       joint{j+1}: {max_err_rad[j]:.6f} rad  ({max_err_deg[j]:.4f} deg)")
        print(f"     Peak Error: {peak_err_rad:.6f} rad ({np.degrees(peak_err_rad):.4f} deg) on joint{peak_joint_idx+1}")
        
        passed = (err_code == 0) and (peak_err_rad < 0.01)
        if not passed:
            overall_pass = False
            print(f"     >>> BUNDLE EVALUATION: FAILED (err_code={err_code}, peak_err={peak_err_rad:.6f} rad >= 0.01 rad)")
        else:
            print(f"     >>> BUNDLE EVALUATION: PASSED (tracking error < 0.01 rad, action SUCCESSFUL)")
            
        bundle_results.append({
            "case_id": cid,
            "arm": arm,
            "num_waypoints": nw,
            "duration_s": duration_s,
            "error_code": err_code,
            "error_string": err_str,
            "max_tracking_error_rad": max_err_rad.tolist(),
            "max_tracking_error_deg": max_err_deg.tolist(),
            "peak_error_rad": peak_err_rad,
            "peak_joint": f"joint{peak_joint_idx+1}",
            "passed": passed
        })

    # Summary Table
    print("\n================================================================================")
    print(" SUMMARY TABLE: SIMULATION DRY RUN")
    print("================================================================================")
    print(f"{'Bundle Case ID':48} | {'Arm':24} | {'Wps':4} | {'Status':12} | {'Peak Err (rad)':14} | {'Peak Joint':10} | Verdict")
    print("-" * 128)
    for r in bundle_results:
        st = "SUCCESSFUL" if r["error_code"] == 0 else f"ERR_{r['error_code']}"
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f"{r['case_id'][:48]:48} | {r['arm'][:24]:24} | {r['num_waypoints']:4} | {st:12} | {r['peak_error_rad']:14.6f} | {r['peak_joint']:10} | {verdict}")
    print("-" * 128)

    print("\n--- Evaluation Summary ---")
    if overall_pass:
        print("[VERIFICATION: PASSED]")
        print("All REPLAY bundles returned SUCCESSFUL with max tracking error below 0.01 rad.")
        print("The trajectory replay executor verified successfully in simulation.")
    else:
        print("[VERIFICATION: FAILED]")
        print("One or more bundles aborted or exceeded 0.01 rad tracking error.")
        
    rclpy.shutdown()
    return overall_pass

def main():
    parser = argparse.ArgumentParser(description="Trajectory Replay Executor")
    parser.add_argument("--mode", choices=["sim", "hw_2wp", "hw_single", "calibrate_tolerance", "negative_control", "false_positive"],
                        default="sim", help="Execution mode")
    parser.add_argument("--bundles_dir", default="replay/bundles", help="Path to bundles")
    args = parser.parse_args()
    
    if args.mode == "sim":
        success = run_simulation_dry_run(args.bundles_dir)
        sys.exit(0 if success else 1)
    else:
        print(f"Mode {args.mode} selected.")

if __name__ == "__main__":
    main()
