import cv2
import threading

# -------------------------
# Camera Class for Streaming
# -------------------------


class Camera:
    """Handles webcam capture and frame generation for streaming."""

    def __init__(self, source=0, width=320, height=240):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        """Continuously capture frames from the webcam."""
        while self.running:
            success, frame = self.cap.read()
            if success:
                self.frame = frame

    def get_frame_bytes(self):
        """Returns current frame as JPEG bytes for streaming."""
        if self.frame is None:
            return None
        ret, buffer = cv2.imencode(".jpg", self.frame)
        return buffer.tobytes()

    def stop(self):
        self.running = False
        self.cap.release()
