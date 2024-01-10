import pickle
import numpy as np
import torch

class NeuralDatasetLoader:
    def __init__(self, batch_size=32):
        self.file_path = '/Users/jessegill/Desktop/nggp/nggp_lib/data/ST260_Day1.pkl'
        self.batch_size = batch_size
        self.data = self.load_data()
        self.mode = 'spikes'
        self.trial_length = 316
        self.stack = self.generate_datastack()
        np.random.shuffle(self.stack)
        self.ptr = 0


    def load_data(self):
        with open(self.file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def preprocess_data(self):
        # Normalize the 'signals' field
        self.data['signals'] = (self.data['signals'] - np.mean(self.data['signals'], axis=0)) / np.std(self.data['signals'], axis=0)
        # Convert data to torch tensors
        for key in self.data:
            self.data[key] = torch.tensor(self.data[key], dtype=torch.float32 if key == 'signals' else torch.int32)

    def generate_batches(self):
        total_samples = len(self.data['signals'])
        for i in range(0, total_samples, self.batch_size):
            batch_indices = slice(i, min(i + self.batch_size, total_samples))
            batch = {key: self.data[key][batch_indices] for key in self.data}
            yield batch
    


    def generate_datastack(self):
        labels = self.data[self.mode][0][:self.trial_length]
        time_points = self.data['time'][:self.trial_length]
        combined = np.column_stack((time_points,labels))
        return combined


        
    def draw_sample(self):
        if self.ptr == self.trial_length-1:
            self.ptr = 0
        self.ptr += 1
        sample = tuple(self.stack[self.ptr-1])
        # Assuming the sample contains continuous data
        return sample

    def get_batch(self):
        batch = []
        batch_labels = []
        for i in range(len(self.stack)):
            time_point,neuron_activity = self.draw_sample()
            batch.append(time_point)
            batch_labels.append(neuron_activity)
        return torch.FloatTensor(batch),torch.FloatTensor(batch_labels)