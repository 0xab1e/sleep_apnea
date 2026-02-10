import os
import numpy as np
import scipy.signal as signal
import librosa
from moviepy import AudioFileClip

class SleepApneaAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.target_sr = 22050  # 22kHz is sufficient for breathing analysis
        
        # --- TUNING PARAMETERS ---
        # Breathing sounds are "air turbulence" (White noise)
        self.low_cut = 200    # Hz: Filter out heartbeats & rumbles
        self.high_cut = 3000  # Hz: Filter out high-pitch hiss
        
        # Apnea Definition (Clinical)
        self.MIN_APNEA_DURATION = 10.0 # Seconds
        
        # Rolling-max analysis parameters (PRIMARY method)
        # Window (seconds) for computing rolling max (activity level)
        self.activity_window = 5.0
        # Fraction of median activity below which = "no breathing"
        # 0.5 is a standard threshold for hypopnea detection
        self.apnea_activity_ratio = 0.5
        # Merge gaps closer than this (seconds) into one event
        self.merge_gap = 5.0
        
        # Peak Detection — higher threshold to only count real breaths
        # (not noise-floor fluctuations)
        self.breath_sensitivity = 0.45

    def load_audio(self):
        """
        Smart loader: Handles .wav, .mp3, AND .mp4 videos.
        Uses direct librosa loading for efficiency, falling back to 
        AudioFileClip for extraction if needed.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        print(f"[-] Processing: {self.file_path}...")
        
        # 1. Try DIRECT Load first (Best for Apple/Mobile files)
        try:
            y, sr = librosa.load(self.file_path, sr=self.target_sr)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"[-] Loaded {duration:.2f} seconds (Direct).")
            return y, sr
        except Exception as e:
            print(f"[!] Direct load failed, trying extraction path...")

        # 2. Extraction Fallback (MoviePy)
        temp_audio = "temp_extracted.wav"
        try:
            print("[-] Extracting audio track...")
            # Use AudioFileClip instead of VideoFileClip to avoid "Frame Error"
            audio_clip = AudioFileClip(self.file_path)
            audio_clip.write_audiofile(temp_audio, fps=self.target_sr, logger=None)
            
            y, sr = librosa.load(temp_audio, sr=self.target_sr)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"[-] Loaded {duration:.2f} seconds.")
            
            audio_clip.close()
            return y, sr
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
                print("[-] Temporary audio file cleaned up.")

    def process_signal(self, audio_data, sr):
        """
        Converts raw audio into a 'Breathing Envelope'.
        """
        # 1. Bandpass Filter (Isolate 'Whoosh')
        nyquist = 0.5 * sr
        low = self.low_cut / nyquist
        high = self.high_cut / nyquist
        
        # Stability check
        if high >= 1.0: high = 0.99
            
        b, a = signal.butter(5, [low, high], btype='band')
        filtered = signal.lfilter(b, a, audio_data)
        
        # 2. Rectify (Absolute Value)
        rectified = np.abs(filtered)
        
        # 3. Smooth (Lowpass Filter)
        # Breathing is slow (~0.3Hz). We cut off anything faster than 2Hz.
        envelope_cutoff = 2.0 
        b_env, a_env = signal.butter(4, envelope_cutoff / nyquist, btype='low')
        envelope = signal.filtfilt(b_env, a_env, rectified)
        
        # Normalize (0 to 1)
        if np.max(envelope) > 0:
            envelope = envelope / np.max(envelope)
            
        return envelope

    def detect_events(self, envelope, sr):
        """
        Detects apnea using TWO complementary methods:
        
        Method 1 (Primary) - Rolling-Max Activity Analysis:
          The key insight: during apnea, the envelope stays near the noise
          floor (~0.1) with NO significant peaks, while normal breathing
          produces sharp spikes (0.5-1.0). We:
          1. Estimate and subtract the noise floor
          2. Compute rolling-max in sliding windows to measure "peak activity"
          3. Flag periods where peak activity drops below a fraction of normal
          
        Method 2 (Secondary) - Inter-Peak Gap Detection:
          Classic approach: finds breath peaks and flags gaps > 10s.
        
        Events from both methods are merged and deduplicated.
        """
        total_duration = len(envelope) / sr
        
        # =============================================
        # METHOD 1: Rolling-max activity analysis
        # =============================================
        
        # Step 1: Estimate noise floor (10th percentile of envelope)
        noise_floor = np.percentile(envelope, 10)
        
        # Step 2: Subtract noise floor to get "activity above noise"
        activity = np.clip(envelope - noise_floor, 0, None)
        
        # Step 3: Compute rolling maximum over sliding windows
        # This captures "how much breathing activity is in each window"
        win_samples = int(self.activity_window * sr)
        if win_samples < 1:
            win_samples = 1
        
        # Use a strided approach for speed — compute max in 1-second steps
        step = sr  # 1-second step
        n_windows = max(1, (len(activity) - win_samples) // step + 1)
        
        rolling_max = np.zeros(n_windows)
        window_times = np.zeros(n_windows)
        for i in range(n_windows):
            start_idx = i * step
            end_idx = min(start_idx + win_samples, len(activity))
            rolling_max[i] = np.max(activity[start_idx:end_idx])
            window_times[i] = (start_idx + end_idx) / 2 / sr
        
        # Step 4: Determine threshold for "breathing present"
        # Use the median of the rolling max as a reference for "normal activity"
        median_activity = np.median(rolling_max)
        activity_threshold = median_activity * self.apnea_activity_ratio
        
        print(f"\n[-] Activity Analysis:")
        print(f"    Noise floor:        {noise_floor:.4f}")
        print(f"    Median peak activity: {median_activity:.4f}")
        print(f"    Apnea threshold:    {activity_threshold:.4f}")
        
        # Step 5: Find contiguous periods of low activity
        is_low = rolling_max < activity_threshold
        
        low_segments = []
        in_low = False
        seg_start = 0
        for i in range(len(is_low)):
            if is_low[i] and not in_low:
                seg_start = i
                in_low = True
            elif not is_low[i] and in_low:
                low_segments.append((seg_start, i))
                in_low = False
        if in_low:
            low_segments.append((seg_start, len(is_low)))
        
        # Convert window indices to seconds
        low_seconds = []
        for s, e in low_segments:
            t_start = max(0, window_times[s] - self.activity_window / 2)
            t_end = min(total_duration, window_times[min(e, len(window_times)) - 1] + self.activity_window / 2)
            low_seconds.append((t_start, t_end))
        
        # Merge nearby segments
        merged = []
        for seg in low_seconds:
            if merged and (seg[0] - merged[-1][1]) < self.merge_gap:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)
        
        # Filter by minimum apnea duration
        amplitude_events = []
        for s, e in merged:
            dur = e - s
            if dur >= self.MIN_APNEA_DURATION:
                amplitude_events.append((s, e, dur))
        
        # =============================================
        # METHOD 2: Inter-peak gap (Secondary)
        # =============================================
        min_dist = int(2.0 * sr)
        peaks, _ = signal.find_peaks(
            envelope, 
            height=self.breath_sensitivity, 
            distance=min_dist
        )
        
        peak_times = peaks / sr
        intervals = np.diff(peak_times)
        
        gap_events = []
        for i, gap_dur in enumerate(intervals):
            if gap_dur >= self.MIN_APNEA_DURATION:
                start = peak_times[i]
                end = peak_times[i+1]
                gap_events.append((start, end, gap_dur))
        
        # =============================================
        # MERGE both method results (deduplicate overlaps)
        # =============================================
        all_events = amplitude_events + gap_events
        all_events.sort(key=lambda x: x[0])
        
        # Deduplicate overlapping events
        final_events = []
        for evt in all_events:
            if final_events and evt[0] <= final_events[-1][1]:
                # Overlapping — extend the existing event
                prev = final_events[-1]
                new_end = max(prev[1], evt[1])
                final_events[-1] = (prev[0], new_end, new_end - prev[0])
            else:
                final_events.append(evt)
        
        print(f"    Amplitude-based events: {len(amplitude_events)}")
        print(f"    Gap-based events:       {len(gap_events)}")
        print(f"    Final merged events:    {len(final_events)}")
                
        return peaks, final_events

    def generate_report(self, audio, envelope, peaks, events, sr):
        """
        Prints stats and plots graphs.
        """
        duration = len(audio) / sr
        if duration == 0: duration = 1 # Avoid div by zero
        
        # Breath Interval Stats
        if len(peaks) > 1:
            peak_times = peaks / sr
            intervals = np.diff(peak_times)
            avg_gap = np.mean(intervals)
            min_gap = np.min(intervals)
            max_gap = np.max(intervals)
        else:
            avg_gap = 0
            min_gap = 0
            max_gap = 0

        bpm = (len(peaks) / duration) * 60
        
        # AHI Estimate (Events per hour)
        hours = duration / 3600
        ahi = len(events) / hours if hours > 0 else 0
        
        print("\n" + "="*40)
        print("       SLEEP APNEA REPORT       ")
        print("="*40)
        print(f" Duration:       {duration:.1f} s")
        print(f" Avg Breath Rate: {bpm:.1f} BPM")
        print(f" Est. AHI:       {ahi:.1f}")
        print("-" * 40)
        print(" BREATHING STATISTICS:")
        print(f"  Avg Gap:       {avg_gap:.2f} s")
        print(f"  Min Gap:       {min_gap:.2f} s")
        print(f"  Max Gap:       {max_gap:.2f} s")
        print("-" * 40)
        print(f" APNEA EVENTS FOUND: {len(events)}")
        for i, (s, e, d) in enumerate(events):
            print(f"  {i+1}. {s:.1f}s -> {e:.1f}s (Gap: {d:.1f}s)")
        print("="*40)


    def export_viewer(self, audio, envelope, peaks, events, sr):
        """
        Exports analysis data and opens an interactive HTML viewer.
        """
        import json, webbrowser, io, base64, wave
        
        duration = len(audio) / sr
        
        # --- Audio WAV for playback (16kHz mono 16-bit) ---
        playback_sr = min(sr, 16000)
        if sr != playback_sr:
            playback_audio = librosa.resample(audio, orig_sr=sr, target_sr=playback_sr)
        else:
            playback_audio = audio
        audio_int16 = (np.clip(playback_audio, -1, 1) * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(playback_sr)
            wf.writeframes(audio_int16.tobytes())
        audio_b64 = base64.b64encode(wav_buf.getvalue()).decode('ascii')
        print(f"[-] Audio encoded ({len(audio_b64)//1024} KB)")
        
        # --- Waveform display data ---
        max_points = 500000
        display_sr = min(100, max_points / max(duration, 1))
        display_sr = max(10, display_sr)
        samples_per_display = int(sr / display_sr)
        n_display = int(len(audio) / samples_per_display)
        
        print(f"[-] Exporting viewer data ({n_display} display samples)...")
        
        audio_min, audio_max, envelope_vals = [], [], []
        for i in range(n_display):
            s = i * samples_per_display
            e = min(s + samples_per_display, len(audio))
            audio_min.append(round(float(np.min(audio[s:e])), 4))
            audio_max.append(round(float(np.max(audio[s:e])), 4))
            envelope_vals.append(round(float(np.max(envelope[s:e])), 4))
        
        peak_times = [round(float(t), 4) for t in peaks / sr]
        peak_values = [round(float(envelope[p]), 4) for p in peaks]
        event_data = [{'start': round(s, 2), 'end': round(e, 2),
                       'duration': round(d, 2)} for s, e, d in events]
        
        if len(peaks) > 1:
            intervals = np.diff(peaks / sr)
            stats = {
                'bpm': round((len(peaks) / duration) * 60, 1),
                'avg_gap': round(float(np.mean(intervals)), 2),
                'min_gap': round(float(np.min(intervals)), 2),
                'max_gap': round(float(np.max(intervals)), 2),
                'ahi': round(len(events) / (duration / 3600), 1) if duration > 0 else 0
            }
        else:
            stats = {'bpm': 0, 'avg_gap': 0, 'min_gap': 0, 'max_gap': 0, 'ahi': 0}
        
        data = {
            'duration': round(duration, 2),
            'display_sr': round(display_sr, 2),
            'audio_min': audio_min, 'audio_max': audio_max,
            'envelope': envelope_vals,
            'peaks': peak_times, 'peak_values': peak_values,
            'events': event_data, 'stats': stats,
            'audio_b64': audio_b64,
            'min_apnea_duration': self.MIN_APNEA_DURATION
        }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, 'viewer.html'), 'r', encoding='utf-8') as f:
            html = f.read()
        html = html.replace('/*DATA_PLACEHOLDER*/null', json.dumps(data))
        output_path = os.path.join(script_dir, 'sleep_report.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[-] Interactive viewer saved: {output_path}")
        webbrowser.open(os.path.abspath(output_path))
        print(f"[-] Opened in browser!")

    def run(self):
        try:
            # Pipeline
            audio, sr = self.load_audio()
            envelope = self.process_signal(audio, sr)
            peaks, events = self.detect_events(envelope, sr)
            self.generate_report(audio, envelope, peaks, events, sr)
            self.export_viewer(audio, envelope, peaks, events, sr)
        except Exception as e:
            print(f"\n[ERROR] {e}")

# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    # REPLACE THIS WITH YOUR FILE PATH (Supports .wav, .mp3, .mp4, .mkv)
    # Example: "C:/Users/Name/Downloads/sleep_recording.mp4"
    INPUT_FILE = "sleep_test.mp4" 
    
    # Create dummy file if not exists (For testing only)
    if not os.path.exists(INPUT_FILE):
        print(f"[!] File {INPUT_FILE} not found.")
        print("[!] Creating a dummy WAV file to demonstrate logic...")
        
        # Generate synthetic data
        sr = 22050
        t = np.linspace(0, 60, 60*sr)
        # Normal breathing + 15s pause + Normal breathing
        sig = np.sin(2*np.pi*0.3*t) * 0.5
        sig[20*sr:35*sr] = 0.01 # Silence
        sig += np.random.normal(0, 0.05, len(t)) # Noise
        
        import soundfile as sf
        sf.write('sleep_test.wav', sig, sr)
        INPUT_FILE = 'sleep_test.wav'

    # Run Analysis
    analyzer = SleepApneaAnalyzer(INPUT_FILE)
    analyzer.run()