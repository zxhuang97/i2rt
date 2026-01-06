#!/bin/bash

# Script to launch leader and follower teleoperation in separate terminals

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up from examples/ to i2rt/ to dependencies/ to robots_realtime/
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

echo -e "${GREEN}Launching teleoperation...${NC}"
echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"
echo -e "${BLUE}Python script: $PYTHON_SCRIPT${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}Cleaning up processes...${NC}"
    kill $FOLLOWER_PID $LEADER_PID 2>/dev/null
    wait $FOLLOWER_PID $LEADER_PID 2>/dev/null
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

# Launch follower in first terminal window
echo -e "${BLUE}Starting follower (can0)...${NC}"
FOLLOWER_CMD="cd '$PROJECT_ROOT' && python '$PYTHON_SCRIPT' --mode follower --gripper crank_4310 --can-channel can0 --server-port 11333 --output ./teleop_trajectories; exec bash"

if [ "$TERMINAL" = "gnome-terminal" ]; then
    gnome-terminal --window --title="Teleop Follower" -- bash -c "$FOLLOWER_CMD" &
    FOLLOWER_PID=$!
elif [ "$TERMINAL" = "xterm" ]; then
    xterm -T "Teleop Follower" -e bash -c "$FOLLOWER_CMD" &
    FOLLOWER_PID=$!
else
    $TERMINAL -e bash -c "$FOLLOWER_CMD" &
    FOLLOWER_PID=$!
fi

# Wait a moment for follower to start
sleep 2

# Launch leader in second terminal window
echo -e "${BLUE}Starting leader (can1)...${NC}"
LEADER_CMD="cd '$PROJECT_ROOT' && python '$PYTHON_SCRIPT' --mode leader --gripper yam_teaching_handle --can-channel can1 --bilateral-kp 0.02 --output ./teleop_trajectories; exec bash"

if [ "$TERMINAL" = "gnome-terminal" ]; then
    gnome-terminal --window --title="Teleop Leader" -- bash -c "$LEADER_CMD" &
    LEADER_PID=$!
elif [ "$TERMINAL" = "xterm" ]; then
    xterm -T "Teleop Leader" -e bash -c "$LEADER_CMD" &
    LEADER_PID=$!
else
    $TERMINAL -e bash -c "$LEADER_CMD" &
    LEADER_PID=$!
fi

echo -e "${GREEN}Both processes started!${NC}"
echo -e "${BLUE}Follower PID: $FOLLOWER_PID${NC}"
echo -e "${BLUE}Leader PID: $LEADER_PID${NC}"
echo -e "${GREEN}Press Ctrl+C to stop both processes${NC}"

# Wait for processes
wait $FOLLOWER_PID $LEADER_PID
