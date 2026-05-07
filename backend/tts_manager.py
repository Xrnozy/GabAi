import pyttsx3
import threading
import queue
import time

class TTSManager:
    """
    A thread-safe TTS Manager that handles speech queueing and prevents
    duplicate/spammy alerts from sensors (e.g. YOLO obstacle detections).
    """
    def __init__(self, rate=150, debounce_time=2.0):
        """
        :param rate: Speed of the speech (words per minute).
        :param debounce_time: Cooldown time in seconds to prevent repeating the exact same phrase.
        """
        self.tts_queue = queue.Queue()
        self.is_running = True
        self.current_speech = None
        self.debounce_time = debounce_time
        self.last_spoken_time = {}
        self.rate = rate
        
        # Start the worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
    def _worker_loop(self):
        """
        Internal loop running on a separate thread.
        pyttsx3 is initialized here because some OSs require it to run on the thread it speaks from.
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
        except Exception as e:
            print(f"TTS Initialization Error: {e}")
            return
            
        while self.is_running:
            try:
                # Block until a text is available, timeout allows checking is_running
                text = self.tts_queue.get(timeout=0.5)
                if text is None:
                    continue
                
                self.current_speech = text
                self.engine.say(text)
                self.engine.runAndWait()
                
                self.current_speech = None
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Playback Error: {e}")

    def speak(self, text):
        """
        Add text to the TTS queue if it passes the duplicate and debounce checks.
        """
        if not self.is_running or not text:
            return

        text_lower = text.strip().lower()
        current_time = time.time()
        
        # Check if already speaking the exact same text right now
        if self.current_speech == text_lower:
            return
            
        # Cooldown check (prevent spamming the same command repeatedly)
        last_time = self.last_spoken_time.get(text_lower, 0)
        if current_time - last_time < self.debounce_time:
            return
            
        # Update the time and queue the text
        self.last_spoken_time[text_lower] = current_time
        self.tts_queue.put(text_lower)

    def stop(self):
        """
        Clear the queue to stop upcoming speech. 
        Note: pyttsx3 doesn't easily stop mid-sentence cross-platform, so we clear the queue.
        """
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break
        
        if hasattr(self, 'engine'):
            try:
                self.engine.stop()
            except Exception:
                pass

    def shutdown(self):
        """
        Shutdown the TTS manager cleanly.
        """
        self.is_running = False
        self.stop()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

# --- Example of how to integrate this in your main application: ---
if __name__ == "__main__":
    print("Initializing TTS Manager...")
    tts = TTSManager(debounce_time=2.0)
    
    print("Simulating detection stream...")
    # Simulate a stream of continuous detections
    detections = ["stop", "stop", "stop", "forward", "forward", "stop"]
    
    for det in detections:
        print(f"Received detection: {det}")
        tts.speak(det)
        time.sleep(0.5) # detections happening every half second
        
    print("Waiting for queue to finish...")
    time.sleep(5) # Let the background thread finish speaking
    tts.shutdown()
    print("Done.")
