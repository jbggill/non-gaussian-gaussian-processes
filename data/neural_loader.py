import pickle
import numpy as np
import torch

class NeuralDatasetLoader:
    def __init__(self):
        self.file_path = '/Users/jessegill/Desktop/nggp/nggp_lib/data/ST260_Day1.pkl'
        self.batch_size = 10
        self.data = self.load_data()
        self.mode = 'signals'

        self.preprocess_data()
        self.trial_length = 316
        self.training_length = 280
        self.stack = self.generate_datastack()[:self.training_length]
        np.random.shuffle(self.stack)
        self.ptr = 0

    def load_data(self):
        with open(self.file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def preprocess_data(self):
        # Normalize the 'signals' field
        self.data[self.mode] = (self.data[self.mode] - np.mean(self.data[self.mode], axis=0)) / np.std(self.data[self.mode], axis=0)
        # Convert data to torch tensors
        for key in self.data:
            self.data[key] = torch.tensor(self.data[key], dtype=torch.float32 if key == self.mode else torch.int32)

    def generate_datastack(self):
        labels = self.data[self.mode][7][:self.trial_length]
        time_points = self.data['time'][:self.trial_length]
        combined = np.column_stack((time_points, labels))
        return combined

    

    def draw_sample(self):
        if self.ptr >= 280:
            return None  # Signal that all data has been drawn
        sample = tuple(self.stack[self.ptr])
        self.ptr += 1
        return sample

    def get_batch(self):
            if self.ptr + self.batch_size > len(self.stack):
                self.ptr = len(self.stack) - self.ptr
                return None, None  # No more data to form a full batch

            batch_data = self.stack[self.ptr:self.ptr + self.batch_size]
            self.ptr += self.batch_size

            if not batch_data.size:  # If batch_data is empty
                return None, None

            time_points, neuron_activities = batch_data[:, 0], batch_data[:, 1:]
            return torch.FloatTensor(time_points).unsqueeze(-1), torch.FloatTensor(neuron_activities)