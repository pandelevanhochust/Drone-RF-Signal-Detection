"""
BladeRF 2.0 Continuous High-Speed Recorder
Optimized Threading Architecture - Complex Float32 Output (GNU Radio Match)
Target: 2375 MHz @ 60 MSPS Continuous Streaming
"""

import numpy as np
import os
import sys
import time
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QLabel, QFormLayout, QFrame, QMessageBox,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QHBoxLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# Safe conditional backend import
try:
    from bladerf import _bladerf
    BLADERF_DRIVERS_INSTALLED = True
except ImportError:
    BLADERF_DRIVERS_INSTALLED = False

# ============================================================================
# HIGH PERFORMANCE RX STREAM THREAD (Float32 File Sink Architecture)
# ============================================================================

class RXStreamThread(QThread):
    """
    Background worker thread dedicated to pulling samples from the USB 3.0 bus,
    converting them to standard Complex Float32, and writing them to disk.
    """
    progress_signal = pyqtSignal(int, float)  # (samples_captured, file_size_mb)
    finished_signal = pyqtSignal(str, str)    # (data_path, meta_path)
    error_signal = pyqtSignal(str)

    def __init__(self, sdr, filepath, duration_sec, sample_rate, center_freq, gain):
        super().__init__()
        self.sdr = sdr
        self.filepath = filepath
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.running = False

    def run(self):
        self.running = True

        # Buffer dimensions optimized for 60 MSPS continuous transfer
        num_samples_per_read = 32768
        num_buffers = 32
        num_transfers = 16
        timeout_ms = 5000

        total_samples_target = int(self.sample_rate * self.duration_sec)
        samples_captured = 0

        meta_path = self.filepath.rsplit(".", 1)[0] + "_meta.txt"

        try:
            # Configure BladeRF Sync Engine for raw Native 12-bit signed IQ pairs packed in 16-bit ints
            self.sdr.sync_config(
                layout=_bladerf.ChannelLayout.RX_X1,
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=num_buffers,
                buffer_size=num_samples_per_read * 4,
                num_transfers=num_transfers,
                stream_timeout=timeout_ms
            )

            # Pre-allocate the low-level receiver bytearray to prevent memory churn
            rx_buffer = bytearray(num_samples_per_read * 4)

            # Open target streaming file in raw append binary mode
            with open(self.filepath, "wb") as f_sink:
                last_ui_update = time.time()

                while self.running and samples_captured < total_samples_target:
                    # 1. Pull raw SC16_Q11 bytes from the hardware ring buffer
                    self.sdr.sync_rx(rx_buffer, num_samples_per_read)

                    # 2. Vectorized conversion to Complex Float32 (GNU Radio Format)
                    # View raw bytes as int16, convert to float32, and normalize by 2048.0 (Q11 format)
                    raw_samples = np.frombuffer(rx_buffer, dtype=np.int16).astype(np.float32)
                    raw_samples /= 2048.0

                    # 3. Stream the raw float32 memory buffer directly out to disk
                    f_sink.write(raw_samples.tobytes())

                    samples_captured += num_samples_per_read

                    # Throttle UI updates to protect event loop throughput
                    current_time = time.time()
                    if current_time - last_ui_update > 0.20:
                        # 8 bytes per complex sample now (4 bytes Real Float32 + 4 bytes Imag Float32)
                        file_size_mb = (samples_captured * 8) / (1024 * 1024)
                        self.progress_signal.emit(samples_captured, file_size_mb)
                        last_ui_update = current_time

            # Write SigMF style metadata logging context
            with open(meta_path, "w") as f_meta:
                f_meta.write(f"center_freq_hz={int(self.center_freq)}\n")
                f_meta.write(f"sample_rate_hz={int(self.sample_rate)}\n")
                f_meta.write(f"total_samples={samples_captured}\n")
                f_meta.write(f"data_type=complex_float32_interleaved\n")
                f_meta.write(f"gain_db={self.gain}\n")
                f_meta.write(f"timestamp={datetime.now().isoformat()}\n")

            self.finished_signal.emit(self.filepath, meta_path)

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.running = False

    def stop(self):
        self.running = False
        self.wait()

# ============================================================================
# MAIN APPLICATION MANAGEMENT VIEW - PyQt6 UI
# ============================================================================

class HighSpeedRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BladeRF 2.0 High-Speed Float32 Recorder")
        self.setGeometry(200, 200, 550, 480)

        self.sdr = None
        self.worker = None
        self.is_connected = False

        self.init_ui()
        self.apply_stylesheet()
        print("Application ready. Click 'Connect' to initialize BladeRF linkage.")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # ---- Connection Control Panel ----
        conn_group = QGroupBox("Device Connection")
        conn_layout = QFormLayout(conn_group)

        self.connect_btn = QPushButton("Connect to BladeRF")
        self.connect_btn.setStyleSheet("font-weight: bold; min-height: 30px; background-color: #2a2a2a;")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addRow(self.connect_btn)

        self.connection_lbl = QLabel("Disconnected")
        self.connection_lbl.setStyleSheet("color: #ff3333; font-weight: bold;")
        conn_layout.addRow("Link Status:", self.connection_lbl)

        main_layout.addWidget(conn_group)

        # ---- Configuration Panel ----
        self.config_group = QGroupBox("SDR Configuration (Offline)")
        self.config_group.setEnabled(False)
        config_layout = QFormLayout(self.config_group)
        config_layout.setSpacing(8)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(47.0, 6000.0)
        self.freq_spin.setValue(2375.0)
        self.freq_spin.setSuffix(" MHz")
        config_layout.addRow("Center Frequency:", self.freq_spin)

        self.sr_spin = QDoubleSpinBox()
        self.sr_spin.setRange(0.521, 61.44)
        self.sr_spin.setValue(60.0)
        self.sr_spin.setSuffix(" MSPS")
        config_layout.addRow("Sample Rate / BW:", self.sr_spin)

        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 60)
        self.gain_spin.setValue(30)
        self.gain_spin.setSuffix(" dB")
        config_layout.addRow("Manual RX Gain:", self.gain_spin)

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 300)
        self.duration_spin.setValue(5)
        self.duration_spin.setSuffix(" Seconds")
        config_layout.addRow("Capture Duration:", self.duration_spin)

        main_layout.addWidget(self.config_group)

        # ---- Status & Monitoring Panel ----
        status_group = QGroupBox("System Status")
        status_layout = QFormLayout(status_group)

        self.status_lbl = QLabel("Awaiting connection link...")
        self.status_lbl.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        status_layout.addRow("SDR State:", self.status_lbl)

        self.stats_lbl = QLabel("Samples: 0 | Size: 0.00 MB")
        self.stats_lbl.setStyleSheet("font-family: 'Courier New'; font-weight: bold;")
        status_layout.addRow("Storage Pipe:", self.stats_lbl)

        main_layout.addWidget(status_group)

        # ---- Action Buttons ----
        actions_layout = QHBoxLayout()

        self.apply_settings_btn = QPushButton("Apply Hardware Settings")
        self.apply_settings_btn.setEnabled(False)
        self.apply_settings_btn.clicked.connect(self.apply_hardware_settings)
        actions_layout.addWidget(self.apply_settings_btn)

        self.record_btn = QPushButton("Start Stream Capture")
        self.record_btn.setStyleSheet("background-color: #151515; font-weight: bold; color: #555555;")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.start_recording)
        actions_layout.addWidget(self.record_btn)

        main_layout.addLayout(actions_layout)

    def apply_stylesheet(self):
        dark_palette = """
            QMainWindow { background-color: #121212; }
            QGroupBox { color: #ffffff; font-weight: bold; border: 1px solid #333333; margin-top: 5px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            QLabel { color: #aaaaaa; font-size: 13px; }
            QSpinBox, QDoubleSpinBox { background-color: #1e1e1e; color: #ffffff; border: 1px solid #444444; padding: 4px; border-radius: 4px; }
            QPushButton { background-color: #2a2a2a; color: #ffffff; border: 1px solid #444444; padding: 8px; border-radius: 4px; min-height: 25px; }
            QPushButton:hover { background-color: #3a3a3a; }
            QPushButton:disabled { background-color: #151515; color: #555555; border: 1px solid #222222; }
        """
        self.setStyleSheet(dark_palette)

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect_hardware()
        else:
            self.connect_hardware()

    def connect_hardware(self):
        if not BLADERF_DRIVERS_INSTALLED:
            QMessageBox.critical(self, "Driver Error", "The 'bladerf' Python bindings wrapper is missing.")
            return

        try:
            self.connect_btn.setEnabled(False)
            self.connection_lbl.setText("Connecting...")
            self.connection_lbl.setStyleSheet("color: #ffaa00; font-weight: bold;")
            QApplication.processEvents()

            self.sdr = _bladerf.BladeRF()

            self.is_connected = True
            self.connect_btn.setText("Disconnect BladeRF")
            self.connect_btn.setEnabled(True)
            self.connection_lbl.setText("Connected")
            self.connection_lbl.setStyleSheet("color: #00ff66; font-weight: bold;")

            self.config_group.setEnabled(True)
            self.config_group.setTitle("SDR Configuration")
            self.apply_settings_btn.setEnabled(True)

            self.apply_hardware_settings()

        except Exception as e:
            self.is_connected = False
            self.connect_btn.setText("Connect to BladeRF")
            self.connect_btn.setEnabled(True)
            self.connection_lbl.setText("Connection Failed")
            self.connection_lbl.setStyleSheet("color: #ff3333; font-weight: bold;")

            QMessageBox.critical(self, "Link Connection Error", f"Could not initialize hardware:\n\n{str(e)}")

    def disconnect_hardware(self):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.stop()

            if self.sdr:
                try:
                    try:
                        rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
                    except AttributeError:
                        rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX1)
                    rx_ch.enable = False
                except:
                    pass

                self.sdr.close()
                self.sdr = None

            self.is_connected = False
            self.connect_btn.setText("Connect to BladeRF")
            self.connection_lbl.setText("Disconnected")
            self.connection_lbl.setStyleSheet("color: #ff3333; font-weight: bold;")

            self.config_group.setEnabled(False)
            self.config_group.setTitle("SDR Configuration (Offline)")
            self.apply_settings_btn.setEnabled(False)
            self.record_btn.setEnabled(False)
            self.record_btn.setStyleSheet("background-color: #151515; font-weight: bold; color: #555555;")

            self.status_lbl.setText("Awaiting connection link...")
            self.status_lbl.setStyleSheet("color: #aaaaaa; font-weight: bold;")

        except Exception as e:
            print(f"Error during disconnection routing: {e}")

    def apply_hardware_settings(self):
        if not self.sdr:
            return

        try:
            self.apply_settings_btn.setEnabled(False)
            self.record_btn.setEnabled(False)

            freq_hz = int(self.freq_spin.value() * 1e6)
            sr_hz = int(self.sr_spin.value() * 1e6)
            gain_db = self.gain_spin.value()

            try:
                rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
            except AttributeError:
                rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX1)

            rx_ch.sample_rate = sr_hz
            rx_ch.bandwidth = sr_hz
            rx_ch.gain_mode = _bladerf.GainMode.Manual
            rx_ch.gain = gain_db
            rx_ch.frequency = freq_hz
            rx_ch.enable = True

            time.sleep(0.3)

            self.status_lbl.setText("SDR Registers Configured & Locked")
            self.status_lbl.setStyleSheet("color: #00ff66; font-weight: bold;")
            self.record_btn.setEnabled(True)
            self.record_btn.setStyleSheet("background-color: #0055ff; font-weight: bold; color: white;")

        except Exception as e:
            self.status_lbl.setText("Configuration Error")
            self.status_lbl.setStyleSheet("color: #ff3333; font-weight: bold;")
            QMessageBox.warning(self, "Register Error", f"Hardware rejected configuration parameters:\n{e}")
        finally:
            self.apply_settings_btn.setEnabled(True)

    def start_recording(self):
        if not self.sdr:
            return

        output_data_file = os.path.join(os.path.expanduser("~"), "bladerf_high_speed_capture.bin")

        self.record_btn.setEnabled(False)
        self.apply_settings_btn.setEnabled(False)
        self.connect_btn.setEnabled(False)
        self.status_lbl.setText("Streaming to Disk (Float32 Mode)...")
        self.status_lbl.setStyleSheet("color: #0088ff; font-weight: bold;")

        self.worker = RXStreamThread(
            sdr=self.sdr,
            filepath=output_data_file,
            duration_sec=self.duration_spin.value(),
            sample_rate=int(self.sr_spin.value() * 1e6),
            center_freq=int(self.freq_spin.value() * 1e6),
            gain=self.gain_spin.value()
        )

        self.worker.progress_signal.connect(self.update_progress_ui)
        self.worker.finished_signal.connect(self.recording_completed)
        self.worker.error_signal.connect(self.recording_faulted)

        self.worker.start()

    def update_progress_ui(self, samples, size_mb):
        self.stats_lbl.setText(f"Samples: {samples:,} | Size: {size_mb:.2f} MB")

    def recording_completed(self, data_path, meta_path):
        self.status_lbl.setText("Capture Finished Cleanly")
        self.status_lbl.setStyleSheet("color: #00ff66; font-weight: bold;")
        self.record_btn.setEnabled(True)
        self.apply_settings_btn.setEnabled(True)
        self.connect_btn.setEnabled(True)
        QMessageBox.information(self, "Capture Success", f"Float32 stream completed:\n\nBin: {data_path}\nMeta: {meta_path}")

    def recording_faulted(self, error_msg):
        self.status_lbl.setText("Stream Pipe Failure")
        self.status_lbl.setStyleSheet("color: #ff3333; font-weight: bold;")
        self.record_btn.setEnabled(True)
        self.apply_settings_btn.setEnabled(True)
        self.connect_btn.setEnabled(True)
        QMessageBox.critical(self, "I/O Error", f"Worker dropped connection pipeline:\n{error_msg}")

    def closeEvent(self, event):
        self.disconnect_hardware()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HighSpeedRecorder()
    window.show()
    sys.exit(app.exec())