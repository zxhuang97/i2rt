"""
Teleoperation with Record/Replay for YAM robots.

This script combines bilateral teleoperation with trajectory recording and replaying.

Trajectory Format:
- Saved as NumPy .npy files (default: ./teleop_trajectory.npy)
- Dictionary with keys:
  - 'trajectory': np.ndarray of shape (N, 6) - joint positions over time
  - 'timestamps': np.ndarray of shape (N,) - timestamps for each sample
  - 'frequency': float - recording frequency (default: 60.0 Hz)

Usage:
  Follower: python teleop_record_replay.py --mode follower --can-channel can0
  Leader: python teleop_record_replay.py --mode leader --can-channel can1 --bilateral-kp 0.2
"""

import argparse
import time
import numpy as np
import curses
import os
import shutil
import threading
import sys
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Any

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from functools import partial

import portal
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot, GripperType
from i2rt.robots.robot import Robot

# Import camera initialization utilities
try:
    from robots_realtime.utils.portal_utils import launch_remote_get_local_handler
    CAMERA_SUPPORT = True
except ImportError:
    CAMERA_SUPPORT = False
    launch_remote_get_local_handler = None

DEFAULT_ROBOT_PORT = 11333


def print_and_cr(message: str, width: int = 120) -> None:
    """Print a message with carriage return to overwrite the current line.
    
    Args:
        message: The message to print
        width: Width to pad the line to (to clear previous content)
    """
    import sys
    padded_message = message + " " * max(0, width - len(message))
    sys.stdout.write(f"\r{padded_message}")
    sys.stdout.flush()


class ServerRobot:
    """A simple server for a follower robot.
    
    Always returns full DOF (7 DOF: 6 joints + gripper) so that:
    - Single leader mode can record gripper state
    - Dual leader mode can record gripper state for both arms
    - Clients can extract 6 DOF if needed for leader feedback
    """

    def __init__(self, robot: Robot, port: int):
        """
        Args:
            robot: The robot to serve
            port: Port to serve on
        """
        self._robot = robot
        self._server = portal.Server(port)
        print_and_cr(f"Robot Server Binding to {port}, Robot: {robot}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        # Always return full DOF including gripper (7 DOF)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        """Serve the follower robot."""
        self._server.start()


class ClientRobot(Robot):
    """A simple client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self._client.get_observations().result()


class YAMLeaderRobot:
    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> np.ndarray:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def move_joints(self, joint_pos: np.ndarray, time_interval_s: float = 2.0) -> None:
        """Move the robot to target joint positions smoothly."""
        assert joint_pos.shape[0] == 6
        self._robot.move_joints(joint_pos, time_interval_s)
    
    def move_to_target_slowly(self, target_pos: np.ndarray, duration: float = 2.0, control_rate_hz: float = 30.0) -> None:
        """Slowly move the robot to target position using interpolation, similar to RobotEnv.move_to_target_slowly."""
        assert target_pos.shape[0] == 6
        current_pos = self._robot.get_joint_pos()[:6]  # Get current 6 DOF position
        num_steps = int(control_rate_hz * duration)
        dt = 1.0 / control_rate_hz
        
        for i in range(num_steps + 1):
            alpha = i / num_steps if num_steps > 0 else 1.0
            command_pos = target_pos * alpha + current_pos * (1 - alpha)
            self.command_joint_pos(command_pos)
            time.sleep(dt)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


def get_next_save_folder(output_dir):
    """Get the next available numbered folder for saving."""
    # Convert to absolute path
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    folder_num = 0
    while os.path.exists(os.path.join(output_dir, str(folder_num))):
        folder_num += 1
    save_folder = os.path.join(output_dir, str(folder_num))
    print(f"[SAVE] Creating folder: {save_folder}")
    return save_folder


def main(stdscr, args):
    # Reset position from config (hardcoded) - 6 DOF only
    RESET_POS = [0, 0.3973, 0.5384, -0.1410, 0, 0]
    
    # Curses setup
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    # Trajectory recording
    trajectory = []
    timestamps = []
    recording = False
    replaying = False
    replay_idx = 0
    target_freq = 30.0
    dt = 1.0 / target_freq

    # Load trajectory if specified
    if args.load and args.load.strip():
        load_path = args.load
        if os.path.isdir(load_path):
            # If it's a directory, look for trajectory.npy inside
            load_path = os.path.join(load_path, "trajectory.npy")
        if os.path.exists(load_path):
            try:
                data = np.load(load_path, allow_pickle=True).item()
                trajectory = data['trajectory'].tolist()
                timestamps = data['timestamps'].tolist()
                if 'frequency' in data:
                    target_freq = data['frequency']
                    dt = 1.0 / target_freq
            except Exception as e:
                print(f"Error loading trajectory: {e}")
                trajectory = []
                timestamps = []

    instructions = [
        "Controls:",
        "  r : Start/stop recording",
        "  p : Start replay",
        "  s : Save trajectory",
        "  l : Load trajectory from file",
        "  d : Delete most recent saved trajectory",
        "  c : Clear current trajectory (frames)",
        "  q : Quit",
        "",
        "Status:"
    ]

    last_record_time = time.monotonic()
    last_replay_time = time.monotonic()
    message_timeout = 3.0  # Clear messages after 3 seconds
    message_time = None
    message_text = ""

    if args.mode == "follower":
        # Initialize robot for follower mode
        gripper_type = GripperType.from_string_name(args.gripper)
        robot = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type)
        # Always return full 7 DOF (leader can extract 6 DOF if needed)
        server_robot = ServerRobot(robot, args.server_port)
        # Start server in background thread
        server_thread = threading.Thread(target=server_robot.serve, daemon=True)
        server_thread.start()
        
        print(f"Follower robot server started on port {args.server_port}")
        print("Waiting for leader to connect...")
        print(f"Output folder: {args.output}")
        print("Press Ctrl+C to quit")
        
        # Simple follower mode: just record/replay, no curses UI
        try:
            while True:
                current_time = time.monotonic()
                
                # Record trajectory at target frequency (record follower's actual position)
                if recording and (current_time - last_record_time) >= dt:
                    qpos = robot.get_joint_pos()
                    trajectory.append(np.copy(qpos))
                    timestamps.append(current_time)
                    last_record_time = current_time
                    if len(trajectory) % 60 == 0:  # Print every second at 60Hz
                        print(f"Recording... {len(trajectory)} samples")
                
                # Auto-save when recording stops (optional - can be removed if not desired)
                # This is handled by the 's' key in leader mode

                # Replay trajectory at target frequency
                if replaying and len(trajectory) > 0:
                    if replay_idx == 0:
                        print(f"Starting replay of {len(trajectory)} samples...")
                        robot.move_joints(np.array(trajectory[replay_idx]), time_interval_s=1.5)
                    if replay_idx < len(trajectory) and (current_time - last_replay_time) >= dt:
                        robot.command_joint_pos(trajectory[replay_idx])
                        replay_idx += 1
                        last_replay_time = current_time
                        if replay_idx % 60 == 0:  # Print every second
                            print(f"Replaying: {replay_idx}/{len(trajectory)}")
                    elif replay_idx >= len(trajectory):
                        replaying = False
                        print("Replay finished.")

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nShutting down follower...")
            if len(trajectory) > 0:
                print(f"Current trajectory has {len(trajectory)} samples (not saved)")
    
    elif args.mode == "dual_follower":
        # Launch each follower in a separate process to avoid CAN bus conflicts
        # This follows the same pattern as launch.py - each robot is initialized
        # in its own process, and we wait for each server to be ready before
        # starting the next one
        from functools import partial
        import subprocess
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "robots_realtime"))
        from robots_realtime.robots.utils import Timeout
        
        # Reset CAN interfaces first (like launch.py does)
        print("Resetting CAN interfaces...")
        try:
            # Get the project root (go up from dependencies/i2rt/examples to project root)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            script_path = os.path.join(project_root, "robots_realtime", "scripts", "reset_all_can.sh")
            if os.path.exists(script_path):
                subprocess.run(["bash", script_path], check=True, timeout=10)
                time.sleep(0.5)  # Same delay as launch.py
                print("CAN interfaces reset")
            else:
                print(f"Warning: CAN reset script not found at {script_path}")
                # Try alternative path
                alt_script_path = os.path.join(project_root, "scripts", "reset_all_can.sh")
                if os.path.exists(alt_script_path):
                    subprocess.run(["bash", alt_script_path], check=True, timeout=10)
                    time.sleep(0.5)
                    print("CAN interfaces reset (using alternative path)")
                else:
                    print(f"Warning: CAN reset script not found at alternative path {alt_script_path}")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"Warning: Could not reset CAN interfaces: {e}")
            print("Continuing anyway...")
        
        def _launch_follower_server(channel: str, gripper_type_str: str, port: int) -> None:
            """Launch a follower robot server in a separate process.
            
            This function runs in the separate process, so robot initialization
            happens there, avoiding CAN bus conflicts.
            """
            gripper_type = GripperType.from_string_name(gripper_type_str)
            print_and_cr(f"[{channel}] Initializing robot...")
            robot = get_yam_robot(channel=channel, gripper_type=gripper_type)
            print_and_cr(f"[{channel}] Robot initialized, starting server on port {port}...")
            # Always return full 7 DOF (including gripper)
            server_robot = ServerRobot(robot, port)
            print_and_cr(f"[{channel}] Follower robot initialized and server started on port {port}")
            server_robot.serve()
        
        # Initialize robots sequentially, waiting for each to be ready
        # This matches launch.py's order: left (can0) first, then right (can2)
        # Same as config: left: ["robot_configs/yam/left.yaml"], right: ["robot_configs/yam/left.yaml", "robot_configs/yam/right.yaml"]
        
        # Launch left follower (can0) first - matches config file order
        print("Launching left follower robot (can0) in separate process...")
        left_follower_process = portal.Process(
            partial(_launch_follower_server, channel="can0", gripper_type_str="crank_4310", port=args.left_follower_port),
            start=True
        )
        print(f"Left follower (can0) process started, waiting for server to be ready...")
        
        # Wait for server to be ready (same pattern as launch_remote_get_local_handler)
        # ClientRobot.__init__ creates portal.Client which doesn't block, so we need to wait
        # by calling a method that will block until server is ready
        try:
            with Timeout(20, f"launching left follower client at port {args.left_follower_port}"):
                left_test_client = ClientRobot(args.left_follower_port, host=args.server_host)
                # Call num_dofs() which will block until server is ready and robot is initialized
                left_test_client.num_dofs()
                print(f"✓ Left follower server is ready!")
        except Exception as e:
            print(f"ERROR: Left follower server failed to start: {e}")
            left_follower_process.kill()
            raise
        
        # Verify robot is actually operational by calling a method that uses CAN bus
        try:
            left_test_client.get_joint_pos()  # This actually communicates with the robot
            print(f"✓ Left follower robot is operational!")
        except Exception as e:
            print(f"Warning: Could not verify left follower robot operation: {e}")
        
        # Launch right follower (can2) after left is fully operational
        print("Launching right follower robot (can2) in separate process...")
        right_follower_process = portal.Process(
            partial(_launch_follower_server, channel="can2", gripper_type_str="crank_4310", port=args.right_follower_port),
            start=True
        )
        print(f"Right follower (can2) process started, waiting for server to be ready...")
        
        # Wait for server to be ready - same pattern as left follower
        try:
            with Timeout(20, f"launching right follower client at port {args.right_follower_port}"):
                right_test_client = ClientRobot(args.right_follower_port, host=args.server_host)
                # Call num_dofs() which will block until server is ready and robot is initialized
                right_test_client.num_dofs()
                print(f"✓ Right follower server is ready!")
        except Exception as e:
            print(f"ERROR: Right follower server failed to start: {e}")
            right_follower_process.kill()
            left_follower_process.kill()
            raise
        
        print(f"\nDual follower mode:")
        print(f"  Left follower (can0): server on port {args.left_follower_port}")
        print(f"  Right follower (can2): server on port {args.right_follower_port}")
        print("Waiting for dual leader to connect...")
        print(f"Output folder: {args.output}")
        print("Press Ctrl+C to quit")
        
        # Wait for both processes
        try:
            left_follower_process.join()
            right_follower_process.join()
        except KeyboardInterrupt:
            print("\nShutting down dual followers...")
            left_follower_process.kill()
            right_follower_process.kill()

    elif args.mode == "leader":
        # Initialize robot for leader mode
        gripper_type = GripperType.from_string_name(args.gripper)
        robot_raw = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type)
        robot = YAMLeaderRobot(robot_raw)
        robot_current_kp = robot._robot._kp
        client_robot = ClientRobot(args.server_port, host=args.server_host)

        # Sync the robot state
        current_joint_pos, current_button = robot.get_info()
        try:
            current_follower_joint_pos = client_robot.get_joint_pos()
            # Follower returns 7 DOF (6 joints + gripper), extract only 6 DOF for leader
            current_follower_joint_pos_6dof = current_follower_joint_pos[:6]
        except RuntimeError as e:
            if "boolean index" in str(e):
                # If we can't get the position initially, use zeros as fallback
                print(f"Warning: Could not get initial follower position: {e}")
                current_follower_joint_pos_6dof = np.zeros(6)
            else:
                raise
        print(f"Current leader joint pos: {current_joint_pos}")
        print(f"Current follower joint pos: {current_follower_joint_pos_6dof}")

        def slow_move(joint_pos: np.ndarray, duration: float = 1.0) -> None:
            # joint_pos is 7 DOF (6 joints + gripper), follower needs 7 DOF
            # Pad current_follower_joint_pos_6dof to 7 DOF for interpolation
            # Use current gripper position if available, otherwise default to 1.0 (open)
            try:
                current_follower_full = client_robot.get_joint_pos()
                follower_gripper = current_follower_full[6] if len(current_follower_full) == 7 else 1.0
            except:
                follower_gripper = 1.0  # Default to open if we can't get current position
            follower_start_7dof = np.concatenate([current_follower_joint_pos_6dof, [follower_gripper]])
            joint_pos_7dof = joint_pos if len(joint_pos) == 7 else np.concatenate([joint_pos[:6], [1.0]])
            for i in range(100):
                follower_command_joint_pos = joint_pos_7dof * i / 100 + follower_start_7dof * (1 - i / 100)
                client_robot.command_joint_pos(follower_command_joint_pos)
                time.sleep(0.03)

        synchronized = False
        while True:
            current_time = time.monotonic()
            key = stdscr.getch()

            if key != -1:
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not synchronized:
                        message_text = "Must be synchronized (press yellow button) to record."
                        message_time = current_time
                        print(f"\n[RECORD] Cannot record - not synchronized. Press yellow button first.")
                    else:
                        recording = not recording
                        replaying = False
                        if recording:
                            trajectory = []
                            timestamps = []
                            last_record_time = current_time
                            message_text = "Recording started."
                            message_time = current_time
                            print(f"\n[RECORD] Started recording. Output folder: {args.output}")
                        else:
                            message_text = f"Recording stopped. {len(trajectory)} points recorded."
                            message_time = current_time
                            print(f"\n[RECORD] Stopped recording. Trajectory has {len(trajectory)} points")
                elif key == ord('p'):
                    if len(trajectory) > 0:
                        replaying = True
                        recording = False
                        synchronized = False  # Stop teleop during replay
                        replay_idx = 0
                        last_replay_time = current_time
                        message_text = "Replay started."
                        message_time = current_time
                    else:
                        message_text = "No trajectory to replay."
                        message_time = current_time
                elif key == ord('s'):
                    if len(trajectory) > 0:
                        try:
                            save_folder = get_next_save_folder(args.output)
                            os.makedirs(save_folder, exist_ok=True)
                            save_path = os.path.join(save_folder, "trajectory.npy")
                            num_points = len(trajectory)
                            data = {
                                'trajectory': np.array(trajectory),
                                'timestamps': np.array(timestamps),
                                'frequency': target_freq
                            }
                            np.save(save_path, data, allow_pickle=True)
                            # Verify file was created
                            if os.path.exists(save_path):
                                message_text = f"Saved to {save_path} ({num_points} points)"
                                message_time = current_time
                                print(f"\n[SAVE] Successfully saved {num_points} points to {save_path}")
                                # Clear trajectory after saving
                                trajectory = []
                                timestamps = []
                                recording = False
                                replaying = False
                                print(f"[SAVE] Cleared trajectory from memory")
                            else:
                                message_text = f"ERROR: File not created at {save_path}"
                                message_time = current_time
                                print(f"\n[SAVE ERROR] File not created at {save_path}")
                        except Exception as e:
                            error_msg = f"Save error: {str(e)}"
                            message_text = error_msg[:60]
                            message_time = current_time
                            print(f"\n[SAVE ERROR] {e}")
                    else:
                        message_text = "No trajectory to save."
                        message_time = current_time
                        print("\n[SAVE] No trajectory to save (trajectory is empty)")
                elif key == ord('l'):
                    message_text = "Enter filename to load: "
                    message_time = current_time
                    stdscr.refresh()
                    filename = ""
                    while True:
                        key = stdscr.getch()
                        if key == ord('\n'):
                            break
                        elif key == ord('\x1b'):
                            filename = ""
                            break
                        elif key == ord('\x7f'):
                            if filename:
                                filename = filename[:-1]
                        elif 32 <= key <= 126:
                            filename += chr(key)
                        message_text = f"Enter filename to load: {filename}"
                        message_time = current_time
                        stdscr.refresh()

                    if filename:
                        load_path = filename
                        if os.path.isdir(load_path):
                            load_path = os.path.join(load_path, "trajectory.npy")
                        if os.path.exists(load_path):
                            try:
                                data = np.load(load_path, allow_pickle=True).item()
                                trajectory = data['trajectory'].tolist()
                                timestamps = data['timestamps'].tolist()
                                if 'frequency' in data:
                                    target_freq = data['frequency']
                                    dt = 1.0 / target_freq
                                # Check trajectory dimensions
                                if len(trajectory) > 0:
                                    first_point = np.array(trajectory[0])
                                    print(f"[LOAD] Loaded trajectory with {len(trajectory)} points, first point shape: {first_point.shape}")
                                    if len(first_point) == 7:
                                        print(f"[LOAD] Trajectory includes gripper (7 DOF), first gripper value: {first_point[6]}")
                                    else:
                                        print(f"[LOAD] Trajectory is 6 DOF (legacy format), will pad gripper to 1.0 during replay")
                                message_text = f"Loaded {load_path} successfully"
                                message_time = current_time
                            except Exception as e:
                                message_text = f"Error loading {load_path}"
                                message_time = current_time
                        else:
                            message_text = f"File {load_path} not found"
                            message_time = current_time
                elif key == ord('c'):
                    # Clear current trajectory (frames)
                    if len(trajectory) > 0:
                        num_points = len(trajectory)
                        trajectory = []
                        timestamps = []
                        recording = False
                        replaying = False
                        message_text = f"Cleared {num_points} frames from trajectory."
                        message_time = current_time
                        print(f"\n[CLEAR] Cleared {num_points} frames from trajectory")
                    else:
                        message_text = "No trajectory to clear."
                        message_time = current_time
                elif key == ord('d'):
                    # Delete the most recent saved trajectory folder
                    # Note: This only deletes saved files, not the in-memory trajectory
                    try:
                        output_dir = os.path.abspath(args.output)
                        if not os.path.exists(output_dir):
                            message_text = "No saved trajectories found."
                            message_time = current_time
                            print("\n[DELETE] No output directory found")
                        else:
                            # Find all numbered folders
                            folders = []
                            for item in os.listdir(output_dir):
                                item_path = os.path.join(output_dir, item)
                                if os.path.isdir(item_path) and item.isdigit():
                                    folders.append((int(item), item_path))
                            
                            if len(folders) == 0:
                                message_text = "No saved trajectories to delete."
                                message_time = current_time
                                print("\n[DELETE] No saved trajectory folders found")
                            else:
                                # Sort by folder number and get the most recent (highest number)
                                folders.sort(key=lambda x: x[0], reverse=True)
                                most_recent_num, most_recent_path = folders[0]
                                
                                # Delete the folder and its contents
                                shutil.rmtree(most_recent_path)
                                message_text = f"Deleted trajectory {most_recent_num}"
                                message_time = current_time
                                print(f"\n[DELETE] Deleted most recent trajectory: {most_recent_path}")
                                print(f"[DELETE] Recording state unchanged - you can still record new trajectories")
                    except Exception as e:
                        error_msg = f"Delete error: {str(e)}"
                        message_text = error_msg[:60]
                        message_time = current_time
                        print(f"\n[DELETE ERROR] {e}")
                        import traceback
                        traceback.print_exc()

            # UI
            stdscr.erase()
            for i, line in enumerate(instructions):
                stdscr.addstr(i, 0, line)
            
            # Status line with replay progress if replaying
            if replaying and len(trajectory) > 0:
                try:
                    status_line = f"Recording: {recording}  Replaying: {replay_idx}/{len(trajectory)}  Synced: {synchronized}"
                except NameError:
                    status_line = f"Recording: {recording}  Replaying: {replaying}  Synced: {synchronized}"
            else:
                status_line = f"Recording: {recording}  Replaying: {replaying}  Synced: {synchronized}"
            stdscr.addstr(len(instructions), 0, status_line)
            stdscr.addstr(len(instructions)+1, 0, f"Trajectory length: {len(trajectory)} samples")
            
            # Clear message after timeout
            if message_time is not None and (current_time - message_time) > message_timeout:
                message_text = ""
                message_time = None
            
            # Display message if any
            if message_text:
                stdscr.addstr(len(instructions)+2, 0, message_text)
            else:
                stdscr.addstr(len(instructions)+2, 0, " " * 80)  # Clear the line
            
            stdscr.addstr(len(instructions)+3, 0, "Press yellow button to sync, 'q' to quit.")

            # Button control for synchronization and reset
            current_joint_pos, current_button = robot.get_info()
            if current_button[0] > 0.5:
                # Every time button is pressed, reset both leader and follower to reset position
                reset_pos = np.array(RESET_POS, dtype=np.float64)
                print(f"[BUTTON] Resetting leader and follower to reset position: {reset_pos}")
                
                # Move leader to reset position (6 DOF)
                robot.command_joint_pos(reset_pos)
                
                # Move follower to reset position (7 DOF: 6 joints + 1.0 for gripper to keep it open)
                # Gripper: 1.0 = open, 0.0 = closed (for linear grippers)
                reset_pos_7dof = np.concatenate([reset_pos, [1.0]])
                client_robot.command_joint_pos(reset_pos_7dof)
                time.sleep(2.0)  # Wait for movement to complete
                
                # Update current follower position after reset
                current_follower_joint_pos_6dof = reset_pos.copy()
                
                if not synchronized:
                    # Enable bilateral feedback - leader can now move freely
                    robot.update_kp_kd(kp=robot_current_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                    print("[BUTTON] Synchronized - bilateral feedback enabled")
                else:
                    # Disable bilateral feedback
                    print("[BUTTON] Desynchronized - clearing bilateral PD")
                    robot.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                    t1 = time.time()
                    # Get current follower position and ensure it's 6 DOF
                    try:
                        current_follower_joint_pos = client_robot.get_joint_pos()
                        current_follower_joint_pos_6dof = current_follower_joint_pos[:6] if len(current_follower_joint_pos) > 6 else current_follower_joint_pos
                        robot.command_joint_pos(current_follower_joint_pos_6dof)
                    except RuntimeError as e:
                        if "boolean index" in str(e):
                            # Skip if we can't get position
                            pass
                        else:
                            raise
                    t2 = time.time()
                    print(f"[BUTTON] Time to command joint pos: {t2 - t1}")
                
                synchronized = not synchronized
                while current_button[0] > 0.5:
                    time.sleep(0.03)
                    current_joint_pos, current_button = robot.get_info()

            # Get follower position only if not replaying (to avoid errors during replay)
            if not replaying:
                try:
                    current_follower_joint_pos = client_robot.get_joint_pos()
                    # Follower returns 7 DOF (6 joints + gripper), extract only 6 DOF for leader
                    current_follower_joint_pos_6dof = current_follower_joint_pos[:6]
                except RuntimeError as e:
                    if "boolean index" in str(e):
                        # Skip this iteration if we get the error
                        time.sleep(0.01)
                        continue
                    raise
            else:
                # During replay, use the trajectory position directly (don't query follower)
                if len(trajectory) > 0 and replay_idx < len(trajectory):
                    traj_pos = np.array(trajectory[replay_idx])
                    # Trajectory is now 7 DOF, extract 6 DOF for leader feedback
                    current_follower_joint_pos_6dof = traj_pos[:6]
                else:
                    # Keep previous value if available
                    if 'current_follower_joint_pos_6dof' not in locals():
                        current_follower_joint_pos_6dof = np.zeros(6)

            if synchronized and not replaying:
                # Teleoperation: 7 DOF to 7 DOF mapping
                # current_joint_pos is 7 DOF (6 joints + gripper), send all 7 to follower
                client_robot.command_joint_pos(current_joint_pos)
                # Send follower's 6 DOF back to leader for bilateral feedback
                robot.command_joint_pos(current_follower_joint_pos_6dof)
                
                # Record trajectory during teleoperation (record follower's position including gripper)
                if recording and (current_time - last_record_time) >= dt:
                    try:
                        follower_pos = client_robot.get_joint_pos()
                        # Follower returns 7 DOF, record all 7 DOF (6 joints + gripper)
                        if len(follower_pos) == 7:
                            trajectory.append(np.copy(follower_pos))
                            timestamps.append(current_time)
                            last_record_time = current_time
                            # Debug: print gripper value occasionally
                            if len(trajectory) % 60 == 0:  # Print every ~1 second at 60Hz
                                print_and_cr(f"[RECORD] Recording point {len(trajectory)}, gripper: {follower_pos[6]:.3f}")
                        else:
                            # If follower returns 6 DOF, pad with current leader gripper
                            leader_gripper = current_joint_pos[6] if len(current_joint_pos) == 7 else 1.0
                            follower_pos_7dof = np.concatenate([follower_pos[:6], [leader_gripper]])
                            trajectory.append(np.copy(follower_pos_7dof))
                            timestamps.append(current_time)
                            last_record_time = current_time
                    except Exception as e:
                        # If we can't get follower position, skip this recording
                        print(f"[RECORD ERROR] Failed to get follower position: {e}")
                        pass

            # Replay trajectory
            if replaying and len(trajectory) > 0:
                if replay_idx == 0:
                    # Trajectory is now 7 DOF (6 joints + gripper), use directly
                    traj_pos = np.array(trajectory[replay_idx])
                    if len(traj_pos) == 6:
                        # Legacy: pad with 1.0 (open) if old 6 DOF format
                        traj_7dof = np.concatenate([traj_pos, [1.0]])
                        print(f"[REPLAY] Legacy 6 DOF format detected, padding gripper to 1.0")
                    else:
                        # Use the recorded gripper value (7 DOF)
                        traj_7dof = traj_pos
                        print(f"[REPLAY] Using recorded gripper value: {traj_7dof[6]}")
                    client_robot.command_joint_pos(traj_7dof)
                    time.sleep(1.5)
                if replay_idx < len(trajectory) and (current_time - last_replay_time) >= dt:
                    # Trajectory is now 7 DOF (6 joints + gripper), use directly
                    traj_pos = np.array(trajectory[replay_idx])
                    if len(traj_pos) == 6:
                        # Legacy: pad with 1.0 (open) if old 6 DOF format
                        traj_7dof = np.concatenate([traj_pos, [1.0]])
                    else:
                        # Use the recorded gripper value (7 DOF)
                        traj_7dof = traj_pos
                    client_robot.command_joint_pos(traj_7dof)
                    replay_idx += 1
                    last_replay_time = current_time
                    # Replay status shown in main status line
                    pass
                elif replay_idx >= len(trajectory):
                    replaying = False
                    message_text = "Replay finished."
                    message_time = current_time

            time.sleep(0.01)
    
    elif args.mode == "dual_leader":
        # Initialize cameras (hardcoded configuration)
        camera_dict = {}
        camera_server_processes = []
        if CAMERA_SUPPORT and launch_remote_get_local_handler is not None:
            try:
                print("Initializing cameras...")
                from functools import partial
                _launch_remote_get_local_handler = partial(
                    launch_remote_get_local_handler,
                    launch_remote=True,
                    process_pool=camera_server_processes,
                )
                
                # Hardcoded camera configuration matching yam_molmoact_bimanual.yaml
                cameras_config = {
                    "top_camera": {
                        "_target_": "robots_realtime.sensors.cameras.camera.CameraNode",
                        "camera": {
                            "_target_": "robots_realtime.sensors.cameras.realsense_camera.RealSenseCamera",
                            "serial_number": "230322274714",
                            "name": "top_camera"
                        }
                    },
                    "right_camera": {
                        "_target_": "robots_realtime.sensors.cameras.camera.CameraNode",
                        "camera": {
                            "_target_": "robots_realtime.sensors.cameras.realsense_camera.RealSenseCamera",
                            "serial_number": "230322277156",
                            "name": "right_camera"
                        }
                    },
                    "left_camera": {
                        "_target_": "robots_realtime.sensors.cameras.camera.CameraNode",
                        "camera": {
                            "_target_": "robots_realtime.sensors.cameras.realsense_camera.RealSenseCamera",
                            "serial_number": "230322271210",
                            "name": "left_camera"
                        }
                    }
                }
                
                for camera_name, camera_config in cameras_config.items():
                    print(f"Initializing camera: {camera_name}")
                    camera_config["camera"]["name"] = camera_name
                    _, client = _launch_remote_get_local_handler(camera_config)
                    camera_dict[camera_name] = client
                print(f"Initialized {len(camera_dict)} cameras: {list(camera_dict.keys())}")
            except Exception as e:
                print(f"Warning: Failed to initialize cameras: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: Camera support not available (missing dependencies)")
        
        # Initialize both left and right leader robots
        left_leader_gripper = GripperType.from_string_name("yam_teaching_handle")
        right_leader_gripper = GripperType.from_string_name("yam_teaching_handle")
        
        print("Initializing left leader robot (can1)...")
        left_leader_robot_raw = get_yam_robot(channel="can1", gripper_type=left_leader_gripper)
        print("Left leader robot initialized")
        
        print("Initializing right leader robot (can3)...")
        right_leader_robot_raw = get_yam_robot(channel="can3", gripper_type=right_leader_gripper)
        print("Right leader robot initialized")
        
        left_leader_robot = YAMLeaderRobot(left_leader_robot_raw)
        right_leader_robot = YAMLeaderRobot(right_leader_robot_raw)
        
        left_leader_kp = left_leader_robot._robot._kp
        right_leader_kp = right_leader_robot._robot._kp
        left_original_kd = left_leader_robot._robot._kd
        right_original_kd = right_leader_robot._robot._kd
        # Connect to both followers (use left_leader_port/right_leader_port for backward compatibility)
        left_port = args.left_leader_port if hasattr(args, 'left_leader_port') else args.left_follower_port
        right_port = args.right_leader_port if hasattr(args, 'right_leader_port') else args.right_follower_port
        left_client_robot = ClientRobot(left_port, host=args.server_host)
        right_client_robot = ClientRobot(right_port, host=args.server_host)
        
        print("Dual-arm teleoperation mode")
        print(f"Left leader: can1, connecting to follower on port {left_port}")
        print(f"Right leader: can3, connecting to follower on port {right_port}")
        
        # Sync both robots
        left_joint_pos, left_button = left_leader_robot.get_info()
        right_joint_pos, right_button = right_leader_robot.get_info()
        
        try:
            left_follower_pos = left_client_robot.get_joint_pos()
            left_follower_pos_6dof = left_follower_pos[:6] if len(left_follower_pos) > 6 else left_follower_pos
        except RuntimeError as e:
            if "boolean index" in str(e):
                print(f"Warning: Could not get initial left follower position: {e}")
                left_follower_pos_6dof = np.zeros(6)
            else:
                raise
        
        try:
            right_follower_pos = right_client_robot.get_joint_pos()
            right_follower_pos_6dof = right_follower_pos[:6] if len(right_follower_pos) > 6 else right_follower_pos
        except RuntimeError as e:
            if "boolean index" in str(e):
                print(f"Warning: Could not get initial right follower position: {e}")
                right_follower_pos_6dof = np.zeros(6)
            else:
                raise
        
        print(f"Left leader joint pos: {left_joint_pos}")
        print(f"Left follower joint pos: {left_follower_pos_6dof}")
        print(f"Right leader joint pos: {right_joint_pos}")
        print(f"Right follower joint pos: {right_follower_pos_6dof}")
        
        # Trajectory recording (store both arms)
        trajectory = []  # List of dicts: {'left': 7dof, 'right': 7dof} (kept for compatibility)
        timestamps = []
        # New organized recording structure
        action_left_pos = []  # left_joint_pos (leader actions)
        action_right_pos = []  # right_joint_pos (leader actions)
        left_joint_pos_list = []  # left_follower_pos (follower states)
        right_joint_pos_list = []  # right_follower_pos (follower states)
        left_gripper_pos = []  # left_joint_pos[-1] (leader left gripper)
        right_gripper_pos = []  # right_joint_pos[-1] (leader right gripper)
        states_actions = []  # Concatenated: [left_follower_pos, right_follower_pos, left_joint_pos, right_joint_pos]
        camera_images = {}  # Dictionary to store camera images: {camera_name: [list of rgb images]}
        recording = False
        replaying = False
        replay_idx = 0
        target_freq = 30.0
        dt = 1.0 / target_freq
        synchronized = False
        last_record_time = time.monotonic()
        last_replay_time = time.monotonic()
        message_timeout = 3.0
        message_time = None
        message_text = ""
        button_status_text = ""  # Status message for button actions (displayed in curses UI)
        button_status_time = None  # Time when button status was set
        button_status_timeout = 3.0  # How long to show button status (seconds)
        
        instructions = [
            "Controls:",
            "  r : Start/stop recording",
            "  p : Start replay",
            "  s : Save trajectory",
            "  l : Load trajectory from file",
            "  d : Delete most recent saved trajectory",
            "  c : Clear current trajectory (frames)",
            "  q : Quit",
            "",
            "Status:"
        ]
        
        # Load trajectory if specified
        if args.load:
            load_path = args.load
            if os.path.isdir(load_path):
                load_path = os.path.join(load_path, "trajectory.npy")
            if os.path.exists(load_path):
                try:
                    data = np.load(load_path, allow_pickle=True).item()
                    # Handle both old format (single arm) and new format (dual arm)
                    if 'left' in data and 'right' in data:
                        # New dual-arm format
                        left_traj = data['left'].tolist()
                        right_traj = data['right'].tolist()
                        trajectory = [{'left': np.array(l), 'right': np.array(r)} for l, r in zip(left_traj, right_traj)]
                    else:
                        # Old single-arm format - duplicate for both arms
                        old_traj = data['trajectory'].tolist()
                        trajectory = [{'left': np.array(t), 'right': np.array(t)} for t in old_traj]
                    timestamps = data['timestamps'].tolist()
                    if 'frequency' in data:
                        target_freq = data['frequency']
                        dt = 1.0 / target_freq
                    print(f"[LOAD] Loaded dual-arm trajectory with {len(trajectory)} points")
                except Exception as e:
                    print(f"Error loading trajectory: {e}")
                    trajectory = []
                    timestamps = []
        
        # Initialize button state tracking variables before the loop
        last_left_button_state = False
        last_right_button_state = False
        last_right_button1_state = False
        
        # Initialize follower position variables to avoid undefined variable errors
        left_follower_pos_6dof = np.zeros(6)
        right_follower_pos_6dof = np.zeros(6)
        
        while True:
            current_time = time.monotonic()
            key = stdscr.getch()
            
            # Get current positions
            left_joint_pos, left_button = left_leader_robot.get_info()
            right_joint_pos, right_button = right_leader_robot.get_info()
            
            # Render UI early so it shows immediately, even if later code blocks
            stdscr.erase()
            for i, line in enumerate(instructions):
                stdscr.addstr(i, 0, line)
            
            if replaying and len(trajectory) > 0:
                try:
                    status_line = f"Recording: {recording}  Replaying: {replay_idx}/{len(trajectory)}  Synced: {synchronized}"
                except NameError:
                    status_line = f"Recording: {recording}  Replaying: {replaying}  Synced: {synchronized}"
            else:
                status_line = f"Recording: {recording}  Replaying: {replaying}  Synced: {synchronized}"
            stdscr.addstr(len(instructions), 0, status_line)
            stdscr.addstr(len(instructions)+1, 0, f"Trajectory length: {len(trajectory)} samples (both arms)")
            
            if message_time is not None and (current_time - message_time) > message_timeout:
                message_text = ""
                message_time = None
            
            if message_text:
                stdscr.addstr(len(instructions)+2, 0, message_text)
            else:
                stdscr.addstr(len(instructions)+2, 0, " " * 80)
            
            # Display button status on a dedicated line (overwrites previous status)
            # Clear button status after timeout
            if button_status_time is not None and (current_time - button_status_time) > button_status_timeout:
                button_status_text = ""
                button_status_time = None
            
            if button_status_text:
                stdscr.addstr(len(instructions)+3, 0, button_status_text + " " * (80 - len(button_status_text)))
            else:
                stdscr.addstr(len(instructions)+3, 0, "Press yellow button (either arm) to sync, 'q' to quit.")
            
            stdscr.refresh()
            
            if key != -1:
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not synchronized:
                        message_text = "Must be synchronized (press yellow button) to record."
                        message_time = current_time
                        print(f"\n[RECORD] Cannot record - not synchronized. Press yellow button first.")
                    else:
                        recording = not recording
                        replaying = False
                        if recording:
                            trajectory = []
                            timestamps = []
                            action_left_pos = []
                            action_right_pos = []
                            left_joint_pos_list = []
                            right_joint_pos_list = []
                            left_gripper_pos = []
                            right_gripper_pos = []
                            states_actions = []
                            camera_images = {}
                            last_record_time = current_time
                            message_text = "Recording started (both arms)."
                            message_time = current_time
                            print(f"\n[RECORD] Started recording both arms. Output folder: {args.output}")
                        else:
                            message_text = f"Recording stopped. {len(trajectory)} points recorded."
                            message_time = current_time
                            print(f"\n[RECORD] Stopped recording. Trajectory has {len(trajectory)} points")
                elif key == ord('p'):
                    if len(trajectory) > 0:
                        replaying = True
                        recording = False
                        synchronized = False
                        replay_idx = 0
                        last_replay_time = current_time
                        message_text = "Replay started."
                        message_time = current_time
                    else:
                        message_text = "No trajectory to replay."
                        message_time = current_time
                elif key == ord('s'):
                    if len(trajectory) > 0:
                        try:
                            save_folder = get_next_save_folder(args.output)
                            os.makedirs(save_folder, exist_ok=True)
                            num_points = len(trajectory)
                            
                            # Save each array to a separate .npy file
                            np.save(os.path.join(save_folder, "action-left-pos.npy"), np.array(action_left_pos))
                            np.save(os.path.join(save_folder, "action-right-pos.npy"), np.array(action_right_pos))
                            np.save(os.path.join(save_folder, "left-joint_pos.npy"), np.array(left_joint_pos_list))
                            np.save(os.path.join(save_folder, "right-joint_pos.npy"), np.array(right_joint_pos_list))
                            np.save(os.path.join(save_folder, "left-gripper_pos.npy"), np.array(left_gripper_pos))
                            np.save(os.path.join(save_folder, "right-gripper_pos.npy"), np.array(right_gripper_pos))
                            np.save(os.path.join(save_folder, "states_actions.npy"), np.array(states_actions))
                            
                            # Save camera images as MP4 videos if available
                            if camera_images and CV2_AVAILABLE:
                                for camera_name, images_list in camera_images.items():
                                    if len(images_list) > 0:
                                        video_path = os.path.join(save_folder, f"{camera_name}-images-rgb.mp4")
                                        # Get video dimensions and FPS from first image
                                        height, width = images_list[0].shape[:2]
                                        fps = target_freq
                                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
                                        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))  # type: ignore
                                        for img in images_list:
                                            # Convert RGB to BGR for OpenCV
                                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # type: ignore
                                            writer.write(img_bgr)
                                        writer.release()
                            elif camera_images and not CV2_AVAILABLE:
                                print("Warning: cv2 not available, skipping camera video saves")
                            
                            if os.path.exists(os.path.join(save_folder, "states_actions.npy")):
                                message_text = f"Saved to {save_folder} ({num_points} points)"
                                message_time = current_time
                                print(f"\n[SAVE] Successfully saved {num_points} points to {save_folder}")
                                trajectory = []
                                timestamps = []
                                action_left_pos = []
                                action_right_pos = []
                                left_joint_pos_list = []
                                right_joint_pos_list = []
                                left_gripper_pos = []
                                right_gripper_pos = []
                                states_actions = []
                                camera_images = {}
                                recording = False
                                replaying = False
                                print(f"[SAVE] Cleared trajectory from memory")
                            else:
                                message_text = f"ERROR: File not created at {save_path}"
                                message_time = current_time
                                print(f"\n[SAVE ERROR] File not created at {save_path}")
                        except Exception as e:
                            error_msg = f"Save error: {str(e)}"
                            message_text = error_msg[:60]
                            message_time = current_time
                            print(f"\n[SAVE ERROR] {e}")
                    else:
                        message_text = "No trajectory to save."
                        message_time = current_time
                elif key == ord('c'):
                    if len(trajectory) > 0:
                        num_points = len(trajectory)
                        trajectory = []
                        timestamps = []
                        action_left_pos = []
                        action_right_pos = []
                        left_joint_pos_list = []
                        right_joint_pos_list = []
                        left_gripper_pos = []
                        right_gripper_pos = []
                        states_actions = []
                        camera_images = {}
                        recording = False
                        replaying = False
                        message_text = f"Cleared {num_points} frames from trajectory."
                        message_time = current_time
                        print(f"\n[CLEAR] Cleared {num_points} frames from trajectory")
                    else:
                        message_text = "No trajectory to clear."
                        message_time = current_time
                elif key == ord('d'):
                    try:
                        output_dir = os.path.abspath(args.output)
                        if not os.path.exists(output_dir):
                            message_text = "No saved trajectories found."
                            message_time = current_time
                        else:
                            folders = []
                            for item in os.listdir(output_dir):
                                item_path = os.path.join(output_dir, item)
                                if os.path.isdir(item_path) and item.isdigit():
                                    folders.append((int(item), item_path))
                            if len(folders) == 0:
                                message_text = "No saved trajectories to delete."
                                message_time = current_time
                            else:
                                folders.sort(key=lambda x: x[0], reverse=True)
                                most_recent_num, most_recent_path = folders[0]
                                shutil.rmtree(most_recent_path)
                                message_text = f"Deleted trajectory {most_recent_num}"
                                message_time = current_time
                                print(f"\n[DELETE] Deleted most recent trajectory: {most_recent_path}")
                    except Exception as e:
                        error_msg = f"Delete error: {str(e)}"
                        message_text = error_msg[:60]
                        message_time = current_time
                        print(f"\n[DELETE ERROR] {e}")
            
            # Button control: left button for reset/sync, right button for recording toggle
            # Track button states to detect presses (not holds)
            left_button_pressed = left_button[0] > 0.5 and not last_left_button_state
            right_button_pressed = right_button[0] > 0.5 and not last_right_button_state
            save_button_pressed = right_button[1] > 0.5 and not last_right_button1_state
            
            # Right button[1]: save trajectory (same as 's' key)
            if save_button_pressed:
                if len(trajectory) > 0:
                    try:
                        save_folder = get_next_save_folder(args.output)
                        os.makedirs(save_folder, exist_ok=True)
                        num_points = len(trajectory)
                        
                        # Save each array to a separate .npy file
                        np.save(os.path.join(save_folder, "action-left-pos.npy"), np.array(action_left_pos))
                        np.save(os.path.join(save_folder, "action-right-pos.npy"), np.array(action_right_pos))
                        np.save(os.path.join(save_folder, "left-joint_pos.npy"), np.array(left_joint_pos_list))
                        np.save(os.path.join(save_folder, "right-joint_pos.npy"), np.array(right_joint_pos_list))
                        np.save(os.path.join(save_folder, "left-gripper-pos.npy"), np.array(left_gripper_pos))
                        np.save(os.path.join(save_folder, "right-gripper-pos.npy"), np.array(right_gripper_pos))
                        np.save(os.path.join(save_folder, "states_actions.npy"), np.array(states_actions))
                        
                        # Save camera images as MP4 videos if available
                        if camera_images and CV2_AVAILABLE:
                            for camera_name, images_list in camera_images.items():
                                if len(images_list) > 0:
                                    video_path = os.path.join(save_folder, f"{camera_name}-images-rgb.mp4")
                                    # Get video dimensions and FPS from first image
                                    height, width = images_list[0].shape[:2]
                                    fps = target_freq
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
                                    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))  # type: ignore
                                    for img in images_list:
                                        # Convert RGB to BGR for OpenCV
                                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # type: ignore
                                        writer.write(img_bgr)
                                    writer.release()
                        elif camera_images and not CV2_AVAILABLE:
                            print("Warning: cv2 not available, skipping camera video saves")
                        
                        if os.path.exists(os.path.join(save_folder, "states_actions.npy")):
                            message_text = f"Saved to {save_folder} ({num_points} points)"
                            message_time = current_time
                            print(f"\n[SAVE] Successfully saved {num_points} points to {save_folder}")
                            trajectory = []
                            timestamps = []
                            action_left_pos = []
                            action_right_pos = []
                            left_joint_pos_list = []
                            right_joint_pos_list = []
                            left_gripper_pos = []
                            right_gripper_pos = []
                            states_actions = []
                            camera_images = {}
                            recording = False
                            replaying = False
                            print(f"[SAVE] Cleared trajectory from memory")
                        else:
                            message_text = f"ERROR: Files not created in {save_folder}"
                            message_time = current_time
                            print(f"\n[SAVE ERROR] Files not created in {save_folder}")
                    except Exception as e:
                        error_msg = f"Save error: {str(e)}"
                        message_text = error_msg[:60]
                        message_time = current_time
                        print(f"\n[SAVE ERROR] {e}")
                else:
                    message_text = "No trajectory to save."
                    message_time = current_time
            
            # Right button[0]: toggle recording (same as 'r' key)
            if right_button_pressed:
                if not synchronized:
                    message_text = "Must be synchronized (press left button) to record."
                    message_time = current_time
                    print(f"\n[RECORD] Cannot record - not synchronized. Press left button first.")
                else:
                    recording = not recording
                    replaying = False
                    if recording:
                        trajectory = []
                        timestamps = []
                        action_left_pos = []
                        action_right_pos = []
                        left_joint_pos_list = []
                        right_joint_pos_list = []
                        left_gripper_pos = []
                        right_gripper_pos = []
                        states_actions = []
                        camera_images = {}
                        last_record_time = current_time
                        message_text = "Recording started (both arms)."
                        message_time = current_time
                        print(f"\n[RECORD] Started recording both arms. Output folder: {args.output}")
                    else:
                        message_text = f"Recording stopped. Trajectory length: {len(trajectory)} samples (both arms)."
                        message_time = current_time
                        print(f"\n[RECORD] Stopped recording. Trajectory length: {len(trajectory)} samples (both arms)")
            
            # Left button: reset and toggle sync
            if left_button_pressed:
                reset_pos = np.array(RESET_POS, dtype=np.float64)
                reset_text = f"[BUTTON] Resetting both arms to reset position: {reset_pos}"
                stdscr.addstr(len(instructions)+3, 0, reset_text + " " * (80 - len(reset_text)))
                stdscr.refresh()
                # Temporarily disable teleoperation to prevent it from overriding reset
                # Disable teleoperation to prevent it from interfering with reset movement
                # Save current state so we can toggle it after reset
                was_synchronized = synchronized
                synchronized = False
                
                # Temporarily restore full PD gain for reset movement
                # When synchronized, PD is reduced (kp * bilateral_kp), which may be too weak
                left_current_kp = left_leader_robot._robot._kp.copy()
                right_current_kp = right_leader_robot._robot._kp.copy()
                
                # Set full PD gain and wait longer to ensure it takes effect
                # First, set current position as target to stabilize before reset
                left_current_pos, _ = left_leader_robot.get_info()
                right_current_pos, _ = right_leader_robot.get_info()
                left_leader_robot.update_kp_kd(kp=left_leader_kp, kd=left_original_kd)  # Full gain, original damping
                right_leader_robot.update_kp_kd(kp=right_leader_kp, kd=right_original_kd)  # Full gain, original damping
                # Command current position to stabilize with new PD values
                left_leader_robot.command_joint_pos(left_current_pos[:6])
                right_leader_robot.command_joint_pos(right_current_pos[:6])
                time.sleep(0.15)  # Wait for PD to stabilize at current position
                
                # Reset both leaders using move_to_target_slowly for smoother movement
                button_status_text = "[BUTTON] Resetting leaders to reset position..."
                button_status_time = current_time
                left_leader_robot.move_to_target_slowly(reset_pos, duration=1.5, control_rate_hz=30.0)
                right_leader_robot.move_to_target_slowly(reset_pos, duration=1.5, control_rate_hz=30.0)
                button_status_text = "[BUTTON] Leaders moved to reset position"
                button_status_time = current_time
                
                # Don't restore PD here - let the sync state logic below handle it
                # This avoids a brief moment of wrong PD values
                
                # Reset both followers (need to use move_joints if available, otherwise command_joint_pos)
                reset_pos_7dof = np.concatenate([reset_pos, [1.0]])
                # For followers, we need to interpolate manually since ClientRobot doesn't have move_joints
                try:
                    left_follower_current = left_client_robot.get_joint_pos()
                    right_follower_current = right_client_robot.get_joint_pos()
                    
                    # Handle case where follower might return 6 DOF or 7 DOF
                    if len(left_follower_current) == 6:
                        left_follower_current = np.concatenate([left_follower_current, [1.0]])
                    if len(right_follower_current) == 6:
                        right_follower_current = np.concatenate([right_follower_current, [1.0]])
                    
                    # Ensure both are 7 DOF now
                    assert len(left_follower_current) == 7, f"Left follower should be 7 DOF, got {len(left_follower_current)}"
                    assert len(right_follower_current) == 7, f"Right follower should be 7 DOF, got {len(right_follower_current)}"
                    
                    # Interpolate over 1 second (50 steps) - same duration as leader reset
                    for i in range(51):
                        alpha = i / 50.0
                        left_cmd = reset_pos_7dof * alpha + left_follower_current * (1 - alpha)
                        right_cmd = reset_pos_7dof * alpha + right_follower_current * (1 - alpha)
                        left_client_robot.command_joint_pos(left_cmd)
                        right_client_robot.command_joint_pos(right_cmd)
                        time.sleep(1.0 / 50.0)
                except Exception as e:
                    # Fallback to direct command if interpolation fails
                    print(f"Warning: Could not interpolate follower reset: {e}")
                    import traceback
                    traceback.print_exc()
                    left_client_robot.command_joint_pos(reset_pos_7dof)
                    right_client_robot.command_joint_pos(reset_pos_7dof)
                    time.sleep(1.0)
                
                # Wait for follower reset to complete
                time.sleep(0.1)
                button_status_text = "[BUTTON] Followers moved to reset position"
                button_status_time = current_time
                
                # Update follower positions after reset
                left_follower_pos_6dof = reset_pos.copy()
                right_follower_pos_6dof = reset_pos.copy()
                
                # After reset, always enable synchronization (set to True)
                synchronized = True
                # Enable bilateral feedback - leader can now move freely
                left_leader_robot.update_kp_kd(kp=left_leader_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                right_leader_robot.update_kp_kd(kp=right_leader_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                button_status_text = "[BUTTON] Synchronized - bilateral feedback enabled"
                button_status_time = current_time
                # Refresh UI to show the sync message immediately
                stdscr.addstr(len(instructions)+3, 0, button_status_text + " " * (80 - len(button_status_text)))
                stdscr.refresh()
                
                # Wait for left button release and update positions
                # Note: button_status_text will be cleared after button release below
                while left_button[0] > 0.5:
                    time.sleep(0.03)
                    left_joint_pos, left_button = left_leader_robot.get_info()
                    right_joint_pos, right_button = right_leader_robot.get_info()
            
            # Update button states for next iteration (for edge detection)
            last_left_button_state = left_button[0] > 0.5
            last_right_button_state = right_button[0] > 0.5
            last_right_button1_state = right_button[1] > 0.5
            
            # Wait for right button[0] release if it was pressed
            if right_button[0] > 0.5:
                while right_button[0] > 0.5:
                    time.sleep(0.03)
                    left_joint_pos, left_button = left_leader_robot.get_info()
                    right_joint_pos, right_button = right_leader_robot.get_info()
                last_right_button_state = False
            
            # Wait for right button[1] release if it was pressed
            if right_button[1] > 0.5:
                while right_button[1] > 0.5:
                    time.sleep(0.03)
                    left_joint_pos, left_button = left_leader_robot.get_info()
                    right_joint_pos, right_button = right_leader_robot.get_info()
                last_right_button1_state = False
                
                # After button release, ensure we have current positions
                left_joint_pos, left_button = left_leader_robot.get_info()
                right_joint_pos, right_button = right_leader_robot.get_info()
            
            # Get follower positions
            if not replaying:
                try:
                    left_follower_pos = left_client_robot.get_joint_pos()
                    left_follower_pos_6dof = left_follower_pos[:6] if len(left_follower_pos) > 6 else left_follower_pos
                except RuntimeError as e:
                    if "boolean index" in str(e):
                        pass
                    else:
                        raise
                try:
                    right_follower_pos = right_client_robot.get_joint_pos()
                    right_follower_pos_6dof = right_follower_pos[:6] if len(right_follower_pos) > 6 else right_follower_pos
                except RuntimeError as e:
                    if "boolean index" in str(e):
                        pass
                    else:
                        raise
            else:
                if len(trajectory) > 0 and replay_idx < len(trajectory):
                    traj_left = np.array(trajectory[replay_idx]['left'])
                    traj_right = np.array(trajectory[replay_idx]['right'])
                    left_follower_pos_6dof = traj_left[:6]
                    right_follower_pos_6dof = traj_right[:6]
                else:
                    if 'left_follower_pos_6dof' not in locals():
                        left_follower_pos_6dof = np.zeros(6)
                    if 'right_follower_pos_6dof' not in locals():
                        right_follower_pos_6dof = np.zeros(6)
            
            # Teleoperation
            if synchronized and not replaying:
                # Send both leader positions to followers (7 DOF each)
                left_client_robot.command_joint_pos(left_joint_pos)
                right_client_robot.command_joint_pos(right_joint_pos)
                
                # Send both follower positions back to leaders (6 DOF each)
                left_leader_robot.command_joint_pos(left_follower_pos_6dof)
                right_leader_robot.command_joint_pos(right_follower_pos_6dof)
                
                # Record both arms
                if recording and (current_time - last_record_time) >= dt:
                    try:
                        left_follower_pos = left_client_robot.get_joint_pos()
                        right_follower_pos = right_client_robot.get_joint_pos()
                        
                        # Followers should always return 7 DOF (6 joints + gripper)
                        if len(left_follower_pos) == 7 and len(right_follower_pos) == 7:
                            # Save to organized lists
                            action_left_pos.append(np.copy(left_joint_pos))
                            action_right_pos.append(np.copy(right_joint_pos))
                            left_joint_pos_list.append(np.copy(left_follower_pos[:-1]))
                            right_joint_pos_list.append(np.copy(right_follower_pos[:-1]))
                            left_gripper_pos.append(left_joint_pos[-1])
                            right_gripper_pos.append(right_joint_pos[-1])
                            
                            # Concatenate: [left_follower_pos, right_follower_pos, left_joint_pos, right_joint_pos]
                            states_actions.append(np.concatenate([
                                left_follower_pos,
                                right_follower_pos,
                                left_joint_pos,
                                right_joint_pos
                            ]))
                            
                            # Get camera RGB images if cameras are available
                            # camera_dict is initialized at the start of dual_leader mode from config file
                            
                            if camera_dict:
                                for camera_name, camera_client in camera_dict.items():
                                    try:
                                        # Match the pattern from robot_env.py: camera_client.read() returns camera_data
                                        # For remote clients, read() may return a future, so call .result() if needed
                                        # camera_data structure: {"images": {"rgb": rgb_image}, ...}
                                        camera_data = camera_client.read()
                                        # Handle both synchronous (direct return) and async (future) cases
                                        if hasattr(camera_data, 'result'):
                                            camera_data = camera_data.result()
                                        
                                        if camera_data and isinstance(camera_data, dict):
                                            # Check for rgb in images dict (same as robot_env.py pattern)
                                            if "images" in camera_data and isinstance(camera_data["images"], dict):
                                                if "rgb" in camera_data["images"]:
                                                    rgb_image = camera_data["images"]["rgb"]
                                                    if camera_name not in camera_images:
                                                        camera_images[camera_name] = []
                                                    camera_images[camera_name].append(np.copy(rgb_image))
                                    except Exception as e:
                                        # If camera read fails, skip this camera but continue recording
                                        pass
                            
                            # Keep trajectory for compatibility
                            trajectory.append({
                                'left': np.copy(left_joint_pos),
                                'right': np.copy(right_joint_pos)
                            })
                            timestamps.append(current_time)
                            last_record_time = current_time
                            if len(trajectory) % 60 == 0:
                                print_and_cr(f"[RECORD] Recording point {len(trajectory)}, left gripper: {left_follower_pos[6]:.3f}, right gripper: {right_follower_pos[6]:.3f}")
                        else:
                            print(f"[RECORD ERROR] Unexpected DOF: left={len(left_follower_pos)}, right={len(right_follower_pos)}. Expected 7 DOF for both.")
                    except Exception as e:
                        print(f"[RECORD ERROR] Failed to record: {e}")
                        import traceback
                        traceback.print_exc()
                        pass
            
            # Replay
            if replaying and len(trajectory) > 0:
                if replay_idx == 0:
                    traj_left = np.array(trajectory[replay_idx]['left'])
                    traj_right = np.array(trajectory[replay_idx]['right'])
                    if len(traj_left) == 6:
                        traj_left = np.concatenate([traj_left, [1.0]])
                    if len(traj_right) == 6:
                        traj_right = np.concatenate([traj_right, [1.0]])
                    left_client_robot.command_joint_pos(traj_left)
                    right_client_robot.command_joint_pos(traj_right)
                    time.sleep(1.5)
                if replay_idx < len(trajectory) and (current_time - last_replay_time) >= dt:
                    traj_left = np.array(trajectory[replay_idx]['left'])
                    traj_right = np.array(trajectory[replay_idx]['right'])
                    if len(traj_left) == 6:
                        traj_left = np.concatenate([traj_left, [1.0]])
                    if len(traj_right) == 6:
                        traj_right = np.concatenate([traj_right, [1.0]])
                    left_client_robot.command_joint_pos(traj_left)
                    right_client_robot.command_joint_pos(traj_right)
                    replay_idx += 1
                    last_replay_time = current_time
                elif replay_idx >= len(trajectory):
                    replaying = False
                    message_text = "Replay finished."
                    message_time = current_time
            
            time.sleep(0.01)
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'leader', 'follower', 'dual_follower', or 'dual_leader'")


@dataclass
class Args:
    gripper: str = "yam_teaching_handle"
    mode: Literal["follower", "leader", "dual_follower", "dual_leader"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT
    can_channel: str = "can0"
    bilateral_kp: float = 0.2
    output: str = "./teleop_trajectories"
    load: str = ""
    # Dual arm specific
    left_follower_port: int = DEFAULT_ROBOT_PORT  # Port for left follower (can0)
    right_follower_port: int = DEFAULT_ROBOT_PORT + 1  # Port for right follower (can2)
    left_leader_port: int = DEFAULT_ROBOT_PORT  # Port for left follower (can0) - used by dual_leader
    right_leader_port: int = DEFAULT_ROBOT_PORT + 1  # Port for right follower (can2) - used by dual_leader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gripper", type=str, default="yam_teaching_handle")
    parser.add_argument("--mode", type=str, default="follower", choices=["follower", "leader", "dual_follower", "dual_leader"])
    parser.add_argument("--server-host", type=str, default="localhost")
    parser.add_argument("--server-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--can-channel", type=str, default="can0")
    parser.add_argument("--bilateral-kp", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="./teleop_trajectories", help="Output folder path for saving trajectories")
    parser.add_argument("--load", type=str, default=None, help="Load trajectory from file or folder")
    parser.add_argument("--left-follower-port", type=int, default=DEFAULT_ROBOT_PORT, help="Port for left follower (can0) - used by dual_follower and dual_leader")
    parser.add_argument("--right-follower-port", type=int, default=DEFAULT_ROBOT_PORT + 1, help="Port for right follower (can2) - used by dual_follower and dual_leader")
    parser.add_argument("--left-leader-port", type=int, default=DEFAULT_ROBOT_PORT, help="Port for left follower (can0) - alias for left-follower-port")
    parser.add_argument("--right-leader-port", type=int, default=DEFAULT_ROBOT_PORT + 1, help="Port for right follower (can2) - alias for right-follower-port")
    args = parser.parse_args()
    
    curses.wrapper(main, args)

