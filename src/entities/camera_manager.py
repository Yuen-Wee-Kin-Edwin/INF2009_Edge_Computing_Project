from entities.camera import Camera


class CameraManager:
    def __init__(self):
        # key: camera_id, value: Camera object
        self.cameras = {}

    def add_camera(self, camera_id, source=0):
        """Add a new camera. source can be local index or remote URL"""
        self.cameras[camera_id] = Camera(source)

    def get_frame(self, camera_id):
        """Return current frame as JPEG bytes for the given camera."""
        cam = self.cameras.get(camera_id)
        if cam:
            return cam.get_frame_bytes()
        return None

    def stop_all(self):
        """Stop all camera threads."""
        for cam in self.cameras.values():
            cam.stop()
