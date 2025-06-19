# datasets/gtu_rir.py
import tarfile
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class GTURIRDataset(Dataset):
    def __init__(self, tar_path, mode='raw', max_length=88200, inside_file='RIR.pickle.dat', hop_length=256, n_fft=512, use_spectrogram=False):
        self.data = self._load_from_tar(tar_path, inside_file)
        self.mode = mode
        self.max_length = max_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.use_spectrogram = use_spectrogram

        self.rir_data_field_numbers = {"timestamp": 0, "speakerMotorIterationNo": 1, "microphoneMotorIterationNo": 2,
                                       "speakerMotorIterationDirection": 3, "currentActiveSpeakerNo": 4,
                                       "currentActiveSpeakerChannelNo": 5,
                                       "physicalSpeakerNo": 6, "microphoneStandInitialCoordinateX": 7,
                                       "microphoneStandInitialCoordinateY": 8, "microphoneStandInitialCoordinateZ": 9,
                                       "speakerStandInitialCoordinateX": 10,
                                       "speakerStandInitialCoordinateY": 11, "speakerStandInitialCoordinateZ": 12,
                                       "microphoneMotorPosition": 13, "speakerMotorPosition": 14,
                                       "temperatureAtMicrohponeStand": 15,
                                       "humidityAtMicrohponeStand": 16, "temperatureAtMSpeakerStand": 17,
                                       "humidityAtSpeakerStand": 18, "tempHumTimestamp": 19,
                                       "speakerRelativeCoordinateX": 20, "speakerRelativeCoordinateY": 21,
                                       "speakerRelativeCoordinateZ": 22, "microphoneStandAngle": 23,
                                       "speakerStandAngle": 24, "speakerAngleTheta": 25, "speakerAnglePhi": 26,
                                       "mic_RelativeCoordinateX": 27, "mic_RelativeCoordinateY": 28,
                                       "mic_RelativeCoordinateZ": 29, "mic_DirectionX": 30, "mic_DirectionY": 31,
                                       "mic_DirectionZ": 32, "mic_Theta": 33, "mic_Phi": 34, "essFilePath": 35,
                                       "roomId": 36, "configId": 37, "micNo": 38,
                                       "roomWidth": 39, "roomHeight": 40, "roomDepth": 41,
                                       "rt60": 42,
                                       "rirData": 43
                                       }


    def _load_from_tar(self, tar_path, inside_file):
        with open(tar_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # RIR
        rir = np.array(sample[43], dtype=np.float32)
        if self.max_length:
            rir = np.pad(rir, (0, max(0, self.max_length - len(rir))))[:self.max_length]
        rir = torch.tensor(rir).unsqueeze(0) # [1, T]
        if self.use_spectrogram:
            # Convert to spectrogram if required (2 channels for real and imaginary parts)
            window = torch.hann_window(self.n_fft, device=rir.device)
            rir = torch.stft(rir.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window)
            rir = torch.stack((rir.real, rir.imag), dim=0)  # [2, F, T]
        # RT60
        rt60 = float(sample[int(self.rir_data_field_numbers['rt60'])])
        # Room dimensions [m]
        room_dim = np.array([float(sample[int(self.rir_data_field_numbers['roomDepth'])]),
                             float(sample[int(self.rir_data_field_numbers['roomWidth'])]),
                            float(sample[int(self.rir_data_field_numbers['roomHeight'])])]) / 100 # [depth, width, height] [m]
        # Microphone location [m]
        microphone_coordinates_x = float(
            sample[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateX'])]) + float(
            sample[int(self.rir_data_field_numbers['mic_RelativeCoordinateX'])])
        microphone_coordinates_y = float(
            sample[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateY'])])  + float(
            sample[int(self.rir_data_field_numbers['mic_RelativeCoordinateY'])])
        microphone_coordinates_z = float(sample[int(self.rir_data_field_numbers['mic_RelativeCoordinateZ'])])
        mic_loc = np.array([microphone_coordinates_x, microphone_coordinates_y, microphone_coordinates_z]) / 100  # [m]
        # Speaker location [m]
        speaker_coordinates_x = float(
            sample[int(self.rir_data_field_numbers['speakerStandInitialCoordinateX'])])  + float(
            sample[int(self.rir_data_field_numbers['speakerRelativeCoordinateX'])])
        speaker_coordinates_y = float(
            sample[int(self.rir_data_field_numbers['speakerStandInitialCoordinateY'])]) + float(
            sample[int(self.rir_data_field_numbers['speakerRelativeCoordinateY'])])
        speaker_coordinates_z = float(sample[int(self.rir_data_field_numbers['speakerRelativeCoordinateZ'])])
        speaker_loc = np.array([speaker_coordinates_x, speaker_coordinates_y, speaker_coordinates_z]) / 100 # [m]

        return rir, room_dim, mic_loc, speaker_loc, rt60
