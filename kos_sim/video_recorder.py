"""Video recording functionality for the KOS simulator."""

import asyncio
import datetime
import os
import time
from pathlib import Path

import mediapy as mp
import numpy as np

from kos_sim import logger
from kos_sim.simulator import MujocoSimulator


class VideoRecorder:
    """Manages video recording for a MuJoCo simulator."""

    def __init__(
        self,
        simulator: MujocoSimulator,
        output_dir: str | Path,
        fps: int = 30,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        self.simulator = simulator
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.last_frame_time = 0.0
        self.writer = None
        self.is_recording = False
        self.recording_task: asyncio.Task[None] | None = None
        self.frames: list[np.ndarray] = []

        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(self, custom_filename: str | None = None) -> str:
        """Start recording video.

        Args:
            custom_filename: Optional custom filename (without extension)

        Returns:
            Path to the output video file
        """
        if self.is_recording:
            self.stop_recording()

        # Generate filename based on datetime if not provided
        if custom_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sim_recording_{timestamp}.mp4"
        else:
            filename = f"{custom_filename}.mp4"

        self.filepath = self.output_dir / filename

        self.frames = []
        self.last_frame_time = time.time()
        self.is_recording = True

        logger.info(f"Started video recording to {self.filepath}")
        return str(self.filepath)

    def stop_recording(self) -> None:
        """Stop recording and save the video to disk."""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.recording_task and not self.recording_task.done():
            self.recording_task.cancel()

        # This will introduce a delay server-side... there is probably a better way to do this
        if self.frames:
            try:
                logger.info(f"Writing {len(self.frames)} frames to {self.filepath}")
                mp.write_video(str(self.filepath), self.frames, fps=self.fps)
                logger.info("Video recording stopped and saved")
            except Exception as e:
                logger.error(f"Error saving video: {e}")
        else:
            logger.warning("No frames captured, no video file created")

        self.frames = []

    async def capture_frame(self) -> None:
        """Capture and add a single frame to the recording."""
        if not self.is_recording:
            return

        try:
            frame, _ = await self.simulator.capture_frame()
            self.frames.append(frame)

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
