import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, QStyle, QSlider, QRadioButton, QButtonGroup, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, QTimer, Qt
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.lines import Line2D as mlines
import os
import csv
import pandas as pd
from feature_extractor import extract_features, extract_clip_features, extract_mini_clip_features, format_time

class AudioAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.audio = None
        self.sr = None
        self.duration = 0
        self.features = [] 
        self.clip_features = []
        self.mini_clip_features = [] 
        self.silence_threshold = 0.1
        self.frame_length = None
        self.selected_frame = None
        self.is_clips = False

        self.setWindowTitle("Audio Analyzer")
        self.setGeometry(100, 100, 1000, 900)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # loading section
        self.info_layout = QVBoxLayout()
        self.load_button = QPushButton("Load Audio File")
        self.load_button.clicked.connect(self.load_audio)
        self.info_layout.addWidget(self.load_button)

        # player section
        self.player_layout = QVBoxLayout()
        self.audio_name_label = QLabel("")
        self.bold_font = self.audio_name_label.font()
        self.bold_font.setBold(True)
        self.audio_name_label.setFont(self.bold_font)
        self.player_layout.addWidget(self.audio_name_label)
        
        self.player = QMediaPlayer(self)  
        self.audio_output = QAudioOutput() 
        self.player.setAudioOutput(self.audio_output)  
        self.player.playbackStateChanged.connect(self.audio_changed)
        
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_position)

        self.player_label = QLabel(text='0:00:00 / 0:00:00')
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.audio_play)
        self.player_layout.addWidget(self.player_label)
        self.player_layout.addWidget(self.play_button)
        self.info_layout.addLayout(self.player_layout)

        # volume slider - silence threshold
        self.volume_slider_layout = QVBoxLayout()
        self.silence_threshold_label = QLabel("Silence Threshold for Volume")
        self.silence_threshold_label.setFont(self.bold_font)
        self.volume_slider_layout.addWidget(self.silence_threshold_label)
        self.value_label = QLabel("Value: 0.09")
        self.volume_slider_layout.addWidget(self.value_label)
        self.silence_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.silence_threshold_slider.setMinimum(0)
        self.silence_threshold_slider.setMaximum(15)
        self.silence_threshold_slider.setValue(9)
        self.silence_threshold_slider.setTickInterval(1)
        self.silence_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.silence_threshold_slider.valueChanged.connect(self.slider_value_changed)
        self.volume_slider_layout.addWidget(self.silence_threshold_slider)
        self.info_layout.addLayout(self.volume_slider_layout)

        #frame-length buttons
        self.selection_label = QLabel("Frame length:")
        self.selection_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.selection_label)
        self.button_layout = QHBoxLayout()
        self.option1 = QRadioButton("10ms")
        self.option2 = QRadioButton("20ms")
        self.option3 = QRadioButton("30ms")
        self.option4 = QRadioButton("40ms")
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.option1)
        self.button_group.addButton(self.option2)
        self.button_group.addButton(self.option3)
        self.button_group.addButton(self.option4)
        self.option2.setChecked(True)
        self.button_layout.addWidget(self.option1)
        self.button_layout.addWidget(self.option2)
        self.button_layout.addWidget(self.option3)
        self.button_layout.addWidget(self.option4)
        self.info_layout.addLayout(self.button_layout)
        self.option1.toggled.connect(self.update_selection)
        self.option2.toggled.connect(self.update_selection)
        self.option3.toggled.connect(self.update_selection)
        self.option4.toggled.connect(self.update_selection)

        # frames or clips layout
        self.frame_clip_layout = QVBoxLayout()
        self.frame_clip_label = QLabel("Show")
        self.frame_clip_label.setFont(self.bold_font)
        self.frame_clip_layout.addWidget(self.frame_clip_label)
        self.frame_button = QRadioButton("Frames")
        self.clip_button = QRadioButton("Clips")

        self.frame_clip_group = QButtonGroup(self)
        self.frame_clip_group.addButton(self.frame_button)
        self.frame_clip_group.addButton(self.clip_button)

        self.frame_button.setChecked(True)

        self.frame_clip_layout.addWidget(self.frame_button)
        self.frame_clip_layout.addWidget(self.clip_button)
        self.info_layout.addLayout(self.frame_clip_layout)
        self.frame_button.toggled.connect(lambda: self.show_frames())
        self.clip_button.toggled.connect(lambda: self.show_clips())

        # frame/clip info layout
        self.frame_info_label = QLabel("Frame Info")
        self.frame_info_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.frame_info_label)
        self.frame_info_l2 = QLabel("")
        self.info_layout.addWidget(self.frame_info_l2)

        # whole audio info layout
        self.audio_info_label = QLabel("Audio Info")
        self.audio_info_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.audio_info_label)
        self.audio_info_l2 = QLabel("")
        self.info_layout.addWidget(self.audio_info_l2)

        self.is_music_label = QLabel("")
        self.is_music_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.is_music_label)

        # saving features layout
        self.save_csv_button = QPushButton("Save features to csv")
        self.save_csv_button.clicked.connect(lambda: self.save_features(True, self.features, self.clip_features))
        self.save_txt_button = QPushButton("Save features to txt")
        self.save_txt_button.clicked.connect(lambda: self.save_features(False, self.features, self.clip_features))

        self.info_layout.addWidget(self.save_csv_button)
        self.info_layout.addWidget(self.save_txt_button)

        # plots - right side layout
        self.plots_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots(6, 1, figsize=(10, 14), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.plots_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plots_layout.addWidget(self.toolbar)

        self.layout.addLayout(self.info_layout)
        self.layout.addLayout(self.plots_layout)
        self.file_name_path = None

        self.selected_option = "20ms"

    def show_frames(self):
        self.is_clips = False
        self.plot_audio(self.features)
        self.reset_frame_info()
        self.duration = len(self.audio) / self.sr * 1000 

    def show_clips(self):
        self.is_clips = True
        self.plot_audio(self.features)
        self.reset_frame_info()
        self.duration = len(self.audio) / self.sr * 1000 
    
    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "WAV Files (*.wav)")
        if file_path:
            self.option2.setChecked(True)
            self.audio, self.sr = librosa.load(file_path, sr=None)
            self.frame_length = int(0.02 * self.sr)
            self.process_audio()
            self.player.setSource(QUrl.fromLocalFile(file_path))
            self.audio_name_label.setText(os.path.basename(file_path))
            self.file_name_path = os.path.basename(file_path)
    
    def process_audio(self):
        self.features = extract_features(self.audio, self.sr, self.frame_length, self.silence_threshold)
        self.mini_clip_features = extract_mini_clip_features(self.features, self.sr, self.frame_length)
        self.plot_audio(self.features)
        self.reset_frame_info()
        self.duration = len(self.audio) / self.sr * 1000 

    def reset_frame_info(self):
        self.frame_info_label.setText("Frame Info")
        self.frame_info_l2.setText("")
        self.selected_frame = None
    
    def plot_audio(self, features):
        self.figure.clear()
        ste, vol, zcr, silent_ratio, f0, voiced_ratio = zip(*features)

        time_axis = np.arange(len(ste)) * (self.frame_length / self.sr)
        
        ax = self.figure.subplots(5, 1, sharex=True)
        self.ax = ax

        ax[0].plot(np.linspace(0, len(self.audio) / self.sr, num=len(self.audio)), self.audio, label='Waveform')
        ax[0].set_title("Waveform")
        
        if not self.is_clips:
            ax, time_axis = self.plot_wavelength_frames(ax, time_axis, ste, vol, zcr, silent_ratio, f0, voiced_ratio)
        else:
            ax, time_axis = self.plot_wavelength_clips(ax, time_axis)

        self.canvas.mpl_connect('button_press_event', self.on_click_waveform)
        self.canvas.draw()
    
    def plot_wavelength_frames(self, ax, time_axis, ste, vol, zcr, silent_ratio, f0, voiced_ratio):
        '''Plots silent and voiced areas on the waveform plot.'''
        silent_patch = mlines([], [], color='gray', lw=4, label='Silent Area')
        voiced_patch = mlines([], [], color='pink', lw=4, label='Voiced Area')
        ax[0].legend(handles=[silent_patch, voiced_patch])

        for i, silent in enumerate(silent_ratio):
            if silent ==1 :
                ax[0].axvspan(i * self.frame_length / self.sr, (i + 1) * self.frame_length / self.sr, color='gray', alpha=0.5)
            if voiced_ratio[i] == 1:
                ax[0].axvspan(i * self.frame_length / self.sr, (i + 1) * self.frame_length / self.sr, color='pink', alpha=0.5)

        for i in range(5):
            ax[i].set_xlim(0, len(self.audio) / self.sr)
            ax[i].grid(True)
        ax[4].set_xlabel("Time (s)")
        self.canvas.mpl_connect('button_press_event', self.on_click_waveform)
        self.ax[0].callbacks.connect('xlim_changed', self.zoom_on_plot)
     
        ax[1].plot(time_axis, ste, label='Short Time Energy (STE)', color='blue')
        ax[1].set_title("Short Time Energy (STE)")
        ax[1].legend()
        ax[1].legend(loc='upper left')

        # volume
        ax[2].plot(time_axis, vol, label='Volume', color='orange')
        ax[2].set_title("Volume")
        ax[2].legend()
        ax[2].legend(loc='upper left')

        # ZCR - Zero Crossing Rate
        ax[3].plot(time_axis, zcr, label='Zero Crossing Rate (ZCR)', color='green')
        ax[3].set_title("Zero Crossing Rate (ZCR)")
        ax[3].legend()
        ax[3].legend(loc='upper left')

        # F0 - Fundamental Frequency
        ax[4].plot(time_axis, f0, label='Fundamental Frequency (F0)', color='purple')
        ax[4].set_title("Fundamental Frequency (F0)")
        ax[4].legend()
        ax[4].legend(loc='upper left')

        return ax, time_axis
    
    def plot_wavelength_clips(self, ax, time_axis, clip_step=0.5):
        """Plots music and speech areas and clip-level features on the waveform plot."""
        music_patch = mlines([], [], color='yellow', lw=4, label='Music')
        speech_patch = mlines([], [], color='green', lw=4, label='Speech')
        ax[0].legend(handles=[music_patch, speech_patch])

        num_clips = len(self.mini_clip_features)
        total_duration = time_axis[-1]

        vstd_list = []
        vdr_list = []
        energy_entropy_list = []
        hzcrr_list = []
        clip_indices = []
        lstr_list = []

        for i in range(num_clips):
            start_time = i * clip_step
            end_time = min(start_time + 1.0, total_duration)

            if self.is_music(self.mini_clip_features[i]):
                ax[0].axvspan(start_time, end_time, color='yellow', alpha=0.5)
            else:
                ax[0].axvspan(start_time, end_time, color='green', alpha=0.5)

            clip_indices.append(i)
            vstd_list.append(self.mini_clip_features[i].get('VSTD', 0))
            vdr_list.append(self.mini_clip_features[i].get('VDR', 0))
            energy_entropy_list.append(self.mini_clip_features[i].get('Energy_Entropy', 0))
            hzcrr_list.append(self.mini_clip_features[i].get('HZCRR', 0))
            lstr_list.append(self.mini_clip_features[i].get('LSTER', 0))

        clip_time_axis = np.array(clip_indices) * clip_step  

        if len(clip_time_axis) != len(vstd_list):
            print(f"WARNING: Mismatched lengths! clip_time_axis={len(clip_time_axis)}, vstd_list={len(vstd_list)}")
            return ax, time_axis
        # clip features 
        ax[1].plot(clip_time_axis, vstd_list, label='Volume STD (VSTD)', color='red')
        ax[1].set_title("Volume STD (VSTD)")
        ax[1].legend(loc='upper left')

        ax[2].plot(clip_time_axis, vdr_list, label='ZSTD', color='blue')
        ax[2].set_title("ZSTD")
        ax[2].legend(loc='upper left')

        ax[3].plot(clip_time_axis, energy_entropy_list, label='Energy Entropy', color='purple')
        ax[3].set_title("Energy Entropy")
        ax[3].legend(loc='upper left')

        ax[4].plot(clip_time_axis, hzcrr_list, label='LSTER', color='green')
        ax[4].set_title("LSTER")
        ax[4].legend(loc='upper left')

        for i in range(5):
            ax[i].grid(True)
        
        ax[4].set_xlabel("Time (s)")
        self.ax[0].callbacks.connect('xlim_changed', self.zoom_on_plot)

        return ax, time_axis 

    def on_click_waveform(self, event):
        """Handles waveform clicks, updating frame or mini-clip info based on self.is_clips."""
        if event.inaxes is not None:
            time_point = event.xdata
            
            if time_point is not None:
                if self.is_clips:
                    clip_index = self.get_clip_index(time_point)
                    self.update_frame_info(clip_index)
                else:
                    frame_index = self.get_frame_index(time_point)
                    self.update_frame_info(frame_index)

                # Move playback to the clicked position
                self.player.setPosition(int(time_point * 1000))
                self.display_time(int(time_point * 1000))

    def zoom_on_plot(self, event):
        '''Zooms in on the waveform plot and updates the audio-level features.'''
        xlim = self.ax[0].get_xlim() 
        start_time, end_time = xlim  
        #print(f"Zoomed from {start_time:.2f}s to {end_time:.2f}s")

        start_frame = self.get_frame_index(start_time)
        end_frame = self.get_frame_index(end_time)
        
        if start_frame < end_frame:
            ste, vol, zcr, silent_ratio, f0, voiced_ratio = zip(*self.features[start_frame:end_frame])

            features_dict = {
                'STE': np.array(ste),
                'volume': np.array(vol),
                'ZCR': np.array(zcr)
            }
            self.clip_features = extract_clip_features(features_dict)

            audio_info_text = "\n".join([f"{key}: {value:.4f}" for key, value in self.clip_features.items()])
            self.is_music_label.setText("Music" if self.is_music(self.clip_features) else "Speech")
            self.audio_info_label.setText("Audio Info")
            self.audio_info_l2.setText(audio_info_text)

    def is_music(self, clip_features):
        '''Determines if the given clip is music based on the clip features.'''
        lster_value = clip_features['LSTER']
        hzcrr_value = clip_features['HZCRR']
        zstd_value = clip_features['ZSTD']

        if (lster_value <= 0.39 and hzcrr_value < 0.15) or (lster_value <= 0.39 and zstd_value < 0.04) or (hzcrr_value < 0.09 and zstd_value < 0.037):
            return True
        return False

    def get_frame_index(self, time_point):
        """ Returns the index of the frame corresponding to the given time point. """
        frame_index = int(time_point * self.sr / self.frame_length)
        if frame_index >= len(self.features):
            frame_index = len(self.features) - 1
        if frame_index < 0:
            frame_index = 0
        return frame_index
    
    def get_clip_index(self, time_point):
        """Returns the index of the mini-clip corresponding to the given time point."""
        if not self.mini_clip_features:
            print("Error: mini_clip_features is empty!")
            return 0 
        
        clip_step = 0.5
        clip_index = int(time_point // clip_step)
        
        if clip_index >= len(self.mini_clip_features):
            clip_index = len(self.mini_clip_features) - 1
        if clip_index < 0:
            clip_index = 0
            
        return clip_index

    def update_selection(self):
        if self.sr is None:
            return
        self.selected_option = "20ms"
        if self.option1.isChecked():
            self.selected_option = "10ms"
            self.frame_length = int(0.01 * self.sr)
        elif self.option2.isChecked():
            self.selected_option = "20ms"
            self.frame_length = int(0.02 * self.sr)
        elif self.option3.isChecked():
            self.selected_option = "30ms"
            self.frame_length = int(0.03 * self.sr)
        elif self.option4.isChecked():
            self.selected_option = "40ms"
            self.frame_length = int(0.04 * self.sr)
        self.process_audio()
                
    def update_frame_info(self, frame_index):
        """ Updates the frame or clip info based on the selected index. """
        if not self.is_clips:
            ste, vol, zcr, silent_ratio, f0, voiced_ratio = self.features[frame_index]
            self.frame_info_label.setText(f"Frame {frame_index} Info:")
            frame_info_text = (
            f"STE: {ste:.4f}\n"
            f"Volume: {vol:.4f}\n"
            f"ZCR: {zcr:.4f}\n"
            f"Silent Ratio: {silent_ratio}\n"
            f"F0: {f0:.4f}\n"
            f"Voiced Ratio: {voiced_ratio}"
            )
            self.frame_info_l2.setText(frame_info_text)
        else:
            if len(self.mini_clip_features) > frame_index:  
                clip_features = self.mini_clip_features[frame_index]
                self.frame_info_label.setText(f"Clip {frame_index} Info:")
                audio_info_text = (
                    f"VSTD: {clip_features['VSTD']:.4f}\n"
                    f"VDR: {clip_features['VDR']:.4f}\n"
                    f"VU: {clip_features['VU']:.4f}\n"
                    f"LSTER: {clip_features['LSTER']:.4f}\n"
                    f"Energy Entropy: {clip_features['Energy_Entropy']:.4f}\n"
                    f"ZSTD: {clip_features['ZSTD']:.4f}\n"
                    f"HZCRR: {clip_features['HZCRR']:.4f}"
                )
                self.frame_info_l2.setText(audio_info_text)
            else:
                print(f"Error: Invalid clip index {frame_index}")

    def slider_value_changed(self):
        value = self.silence_threshold_slider.value() / 100
        self.silence_threshold = value
        self.value_label.setText(f"Value: {value}")
        self.process_audio()

    def audio_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.timer.stop()
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self.player.play()
            self.timer.start()
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def audio_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        elif state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:  # Paused
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def update_position(self):
        self.display_time(self.player.position())

    def display_time(self, ms):
        self.player_label.setText(f'{format_time(ms)} / {format_time(int(self.duration))}')

    def save_features(self, is_csv, features, txt_filename="features.txt", csv_filename="features.csv", 
                    clip_txt_filename="clip_features.txt", clip_csv_filename="clip_features.csv"):
        """Saves frame-level and clip-level features to text and CSV files with corrected start and end times."""

        # Filenames based on selected option
        txt_filename = f"features_{self.selected_option}_{self.file_name_path}.txt"
        csv_filename = f"features_{self.selected_option}_{self.file_name_path}.csv"
        clip_txt_filename = f"clip_features_{self.selected_option}_{self.file_name_path}.txt"
        clip_csv_filename = f"clip_features_{self.selected_option}_{self.file_name_path}.csv"

        frame_step = self.frame_length
        total_audio_duration = len(self.audio) / self.sr  # Total duration in seconds

        # Fix Clip-Level Start and End Times (Using 0.5s step)
        if self.mini_clip_features is not None:
            for i, clip_feature in enumerate(self.mini_clip_features):
                start_time = i * 0.5
                end_time = min(start_time + 1.0, total_audio_duration)  # Ensure clips don't exceed total duration
                clip_feature['Start Time (s)'] = round(start_time, 3)
                clip_feature['End Time (s)'] = round(end_time, 3)
            clip_df = pd.DataFrame(self.mini_clip_features)  # Convert to DataFrame for saving

        # Frame-Level Feature Saving
        if not is_csv:
            with open(txt_filename, 'w') as f:
                f.write("Frame: Start Time (s), End Time (s), STE, Volume, ZCR, Silent Ratio, F0, Voiced Ratio\n")
                for i, feature in enumerate(self.features):
                    start_time = (i * frame_step) / self.sr  
                    end_time = min((start_time + self.frame_length / self.sr), total_audio_duration)  # Fixed time calculation
                    f.write(f"Frame {i+1}: Start Time: {start_time:.3f}s, End Time: {end_time:.3f}s, " + "\t".join(map(str, feature)) + "\n")

            # Save mini-clip features to a text file
            clip_df.to_csv(clip_txt_filename, sep='\t', index=False)

        else:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Frame", "Start Time (s)", "End Time (s)", "STE", "Volume", "ZCR", "Silent Ratio", "F0", "Voiced Ratio"])  # Header
                for i, feature in enumerate(self.features):
                    start_time = (i * frame_step) / self.sr
                    end_time = (start_time + self.frame_length / self.sr)  # Fixed time calculation
                    writer.writerow([i+1, f"{start_time:.3f}", f"{end_time:.3f}"] + list(feature))

            # Save mini-clip features to a CSV file
            clip_df.to_csv(clip_csv_filename, index=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioAnalyzer()
    window.show()
    sys.exit(app.exec())