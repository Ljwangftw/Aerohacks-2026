"""
command_mapper.py — Maps controller outputs to drone API calls.

This is a thin helper. Most of the logic lives in hover_controller.py.
Kept as a separate file in case the organizer API changes and command
formatting needs to be isolated.
"""

from src.utils.math_helpers import clamp


def map_commands_to_drone(commands: dict, drone_client) -> None:
    """Send control commands to the drone.

    Args:
        commands: dict with "pitch", "roll", "thrust", "yaw"
        drone_client: DroneClient instance
    """
    drone_client.set_pitch(commands["pitch"])
    drone_client.set_roll(commands["roll"])
    drone_client.set_thrust_uniform(commands["thrust"])
    drone_client.set_yaw(commands["yaw"])
