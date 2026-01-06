#!/bin/bash

# Script to launch dual-arm teleoperation (left and right arms with their leaders)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Use absolute path for the Python script
PYTHON_SCRIPT="$PROJECT_ROOT/dependencies/i2rt/examples/teleop_record_replay.py"

# Verify the path
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Cannot find teleop_record_replay.py"
    echo "Script dir: $SCRIPT_DIR"
    echo "Project root: $PROJECT_ROOT"
    echo "Looking for: $PYTHON_SCRIPT"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Launching dual-arm teleoperation...${NC}"
echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"
echo -e "${BLUE}Python script: $PYTHON_SCRIPT${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}Cleaning up processes...${NC}"
    kill $DUAL_FOLLOWER_PID $DUAL_LEADER_PID 2>/dev/null
    wait $DUAL_FOLLOWER_PID $DUAL_LEADER_PID 2>/dev/null
    exit
}

# Set up trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Check which terminal emulator is available
if command -v gnome-terminal &> /dev/null; then
    TERMINAL="gnome-terminal"
elif command -v xterm &> /dev/null; then
    TERMINAL="xterm"
elif command -v x-terminal-emulator &> /dev/null; then
    TERMINAL="x-terminal-emulator"
else
    echo "Error: No suitable terminal emulator found (gnome-terminal, xterm, or x-terminal-emulator)"
    exit 1
fi

# Launch dual follower (can0 and can2) in one process
echo -e "${BLUE}Starting dual follower (can0=left, can2=right)...${NC}"
DUAL_FOLLOWER_CMD="cd '$PROJECT_ROOT' && python '$PYTHON_SCRIPT' --mode dual_follower --gripper crank_4310 --left-follower-port 11333 --right-follower-port 11334 --output ./teleop_trajectories; exec bash"

if [ "$TERMINAL" = "gnome-terminal" ]; then
    gnome-terminal --window --title="Dual Follower (can0+can2)" -- bash -c "$DUAL_FOLLOWER_CMD" &
    DUAL_FOLLOWER_PID=$!
elif [ "$TERMINAL" = "xterm" ]; then
    xterm -T "Dual Follower (can0+can2)" -e bash -c "$DUAL_FOLLOWER_CMD" &
    DUAL_FOLLOWER_PID=$!
else
    $TERMINAL -e bash -c "$DUAL_FOLLOWER_CMD" &
    DUAL_FOLLOWER_PID=$!
fi

sleep 2

# Launch dual leader (can1 and can3)
echo -e "${BLUE}Starting dual leader (can1=left, can3=right)...${NC}"
DUAL_LEADER_CMD="cd '$PROJECT_ROOT' && python '$PYTHON_SCRIPT' --mode dual_leader --gripper yam_teaching_handle --bilateral-kp 0.02 --left-follower-port 11333 --right-follower-port 11334 --output ./teleop_trajectories; exec bash"

if [ "$TERMINAL" = "gnome-terminal" ]; then
    gnome-terminal --window --title="Dual Leader (can1+can3)" -- bash -c "$DUAL_LEADER_CMD" &
    DUAL_LEADER_PID=$!
elif [ "$TERMINAL" = "xterm" ]; then
    xterm -T "Dual Leader (can1+can3)" -e bash -c "$DUAL_LEADER_CMD" &
    DUAL_LEADER_PID=$!
else
    $TERMINAL -e bash -c "$DUAL_LEADER_CMD" &
    DUAL_LEADER_PID=$!
fi

echo -e "${GREEN}All processes started!${NC}"
echo -e "${BLUE}Dual follower PID: $DUAL_FOLLOWER_PID${NC}"
echo -e "${BLUE}Dual leader PID: $DUAL_LEADER_PID${NC}"
echo -e "${GREEN}Press Ctrl+C to stop all processes${NC}"

# Wait for processes
wait $DUAL_FOLLOWER_PID $DUAL_LEADER_PID

