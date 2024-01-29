import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.priors import UniformPrior
import data
from nggp_lib.data.data_generator import SinusoidalDataGenerator, Nasdaq100padding
import os
#from data.qmul_loader import get_batch, train_people, test_people
# import neural loader
from nggp_lib.data.neural_loader import NeuralDatasetLoader
from nggp_lib.models.kernels import NNKernel, MultiNNKernel
#from data.objects_pose_loader import get_dataset, get_objects_batch
from nggp_lib.training.utils import prepare_for_plots, plot_histograms


def get_transforms(model, use_context):
    if use_context:
        def sample_fn(z, context=None, logpz=None):
            if logpz is not None:
                return model(z, context, logpz, reverse=True)
            else:
                return model(z, context, reverse=True)

        def density_fn(x, context=None, logpx=None):
            if logpx is not None:
                return model(x, context, logpx, reverse=False)
            else:
                return model(x, context, reverse=False)
    else:
        def sample_fn(z, logpz=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z, reverse=True)

        def density_fn(x, logpx=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

    return sample_fn, density_fn


class ANGGP(nn.Module):
    def __init__(self,backbone, device, num_tasks=1, config=None, dataset='QMUL', cnf=None, use_conditional=False, batch_size= 50,
                 add_noise=False, context_type='nn', multi_type=2):

        super(ANGGP, self).__init__()

        ## GP parameters
        
        self.feature_extractor = backbone
        self.device = torch.device(device)
        self.num_tasks = num_tasks
        self.config = config
        self.cnf = cnf
        self.use_conditional = use_conditional
        self.multi_type = multi_type
        self.context_type = context_type
        if self.cnf is not None:
            self.is_flow = True
        else:
            self.is_flow = False
        self.add_noise = add_noise


        #plotting
        self.max_test_plots=5
        self.i_plots=0

        self.batch_size = batch_size
        self.train_x = None
        self.train_y = None
    
    def set_training_data(self, train_x, train_y):
        self.train_x = train_x.to(self.device)
        self.train_y = train_y.to(self.device)
        self.get_model_likelihood_mll(train_x=self.train_x, train_y=self.train_y)
        print('Training data set')

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if train_x is None or train_y is None:
            raise ValueError("Training data must be provided")

        if self.num_tasks == 1:
            # adds some noise for the nasdaq data set and can be ignored
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPLayer( config=self.config, train_x=train_x,
                                train_y=train_y, likelihood=likelihood, kernel=self.config.kernel_type)

                                 #train_y=train_y[:50], likelihood=likelihood, kernel=self.config.kernel_type)
        else:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
            model = MultitaskExactGPLayer(config=self.config, train_x=train_x, train_y=train_y,
                                          likelihood=likelihood,
                                          kernel=self.config.kernel_type, num_tasks=self.num_tasks,
                                          multi_type=self.multi_type)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)
        self.mse = nn.MSELoss()
        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    # looks like nothing needs to be changed here either, just calls the train loop but only after some noise is added to the nasdaq dataset
    def train_loop(self, epoch, optimizer, params, results_logger):
        self._train_loop(epoch, optimizer, params, results_logger)


    # trouble area
    def _train_loop(self, epoch, optimizer, params, results_logger):

        # put the model, feature extractor and likelihood into "training mode"
        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()
        if self.is_flow:
            self.cnf.train()

        dataset = TensorDataset(self.train_x, self.train_y)
        #dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=len(self.train_x), shuffle=True)

        # iterate through each input label pair in batch and performance SGD
        # I don't think I need to do anything here
        total_batches = len(dataloader)  # Get the total number of batches

        for batch_index, (inputs, labels) in enumerate(dataloader):
            #if batch_index == total_batches - 1:  # Check if it's the last batch
             #   break  # Skip the last batch
    
        # Process your inputs and labels here

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

           # break
            #z = self.feature_extractor(inputs)
            z = inputs
            if self.add_noise:
                labels = labels + torch.normal(0, 0.1, size=labels.shape).to(labels)
            if self.is_flow:                
                delta_log_py, labels, y = self.apply_flow(labels, z)
            else:
                y = labels
            self.model.set_train_data(inputs=z, targets=y)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets)
            if self.is_flow:
                loss = loss + torch.mean(delta_log_py)
            loss.backward()
            optimizer.step()

            mse, _ = self.compute_mse(labels, predictions, z)
            if epoch % 10 == 0:
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
                results_logger.log("epoch", epoch)
                results_logger.log("loss", loss.item())
                results_logger.log("MSE", mse.item())
                results_logger.log("noise", self.model.likelihood.noise.item())
        return loss.item()

    # compute the mse between predicted and actual labels for the GP model
    def compute_mse(self, labels, predictions, z):

        if self.is_flow:
            sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
            if self.num_tasks == 1:
                means = predictions.mean.unsqueeze(1)
            else:
                means = predictions.mean
            if self.use_conditional:
                new_means = sample_fn(means, self.get_context(z))
            else:
                new_means = sample_fn(means)
            mse = self.mse(new_means.squeeze(), labels.squeeze())
        else:
            mse = self.mse(predictions.mean, labels)
            new_means = None
        return mse, new_means

    # generates context for conditional models??
    def get_context(self, z):
        if self.context_type == 'nn':
            if self.num_tasks == 1:
                context = self.model.kernel.model(z)
            else:
                if self.multi_type == 3:
                    contexts = []
                    for k in range(len(self.model.kernels)):
                        contexts.append(self.model.kernels[k].model(z))
                    context = sum(contexts)
                else:
                    context = self.model.kernels.model(z)
        elif self.context_type == 'backbone':
            context = z
        else:
            raise ValueError("unknown context type")
        return context

    def apply_flow(self, labels, z):
        if self.num_tasks == 1:
            labels = labels.unsqueeze(1)
        if self.use_conditional:
            y, delta_log_py = self.cnf(labels, self.get_context(z),
                                       torch.zeros(labels.size(0), 1).to(labels))
        else:
            y, delta_log_py = self.cnf(labels, torch.zeros(labels.size(0), 1).to(labels))
        y = y.squeeze()
        return delta_log_py, labels, y


    # the testing part. This is made for meta-learning so it would have to be adapted
    def test_loop(self, n_support, params=None, save_dir=None):

        # each creates a support and query dataset for 0
        if self.dataset == "sines":
            x_all, x_support, y_all, y_support = self.get_support_query_sines(n_support, params)
            x_test, y_test = x_all, y_all
        elif self.dataset == "nasdaq" or self.dataset == "eeg":
            x_all, x_support, y_all, y_support = self.get_support_query_nasdaq(n_support, params)
            x_test, y_test = x_all, y_all
        elif self.dataset == "objects":
            x_all, x_support, x_query, y_all, y_support, y_query = self.get_support_query_objects(n_support, params)
            x_test, y_test = x_query, y_query
        elif self.dataset == "QMUL":
            x_all, x_support, y_all, y_support = self.get_support_query_qmul(n_support)
            x_test, y_test = x_all, y_all
        else:
            raise ValueError("Unknown dataset")

        sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
        # choose a random test person
        n = np.random.randint(0, x_support.shape[0])
        z_support = self.feature_extractor(x_support[n]).detach()
        labels = y_support[n]

        if self.is_flow:
            with torch.no_grad():
                _, labels, y_support = self.apply_flow(labels, z_support)
        else:
            y_support = labels

        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()
        if self.is_flow:
            self.cnf.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_test[n]).detach()
            predictions_query = self.model(z_query)
            pred = self.likelihood(predictions_query)
            context = None
            new_means = None
            if self.is_flow:
                if self.num_tasks == 1:
                    mean_base = pred.mean.unsqueeze(1)
                else:
                    mean_base = pred.mean
                if self.use_conditional:
                    context = self.get_context(z_query)
                    new_means = sample_fn(mean_base, context)
                else:
                    new_means = sample_fn(mean_base)
                delta_log_py, _, y = self.apply_flow(y_test[n], z_query)
                log_py = -self.mll(predictions_query, y.squeeze())
                NLL = log_py + torch.mean(delta_log_py.squeeze())
                # log_py = normal_logprob(y.squeeze(), pred.mean, pred.stddev)
                # NLL = -1.0 * torch.mean(log_py - delta_log_py.squeeze())
            else:
                NLL = -self.mll(predictions_query, y_test[n])
                #log_py = normal_logprob(y_all[n], pred.mean, pred.stddev)
                #NLL = -1.0 * torch.mean(log_py)
            if self.i_plots<self.max_test_plots and save_dir is not None and self.num_tasks == 1:
                samples, true_y, gauss_y, flow_samples, flow_y = prepare_for_plots(pred, y_test[n],
                                                                                    sample_fn, context, new_means)
                plot_histograms(save_dir, samples, true_y, gauss_y, n, flow_samples, flow_y, self.i_plots)
                self.i_plots=self.i_plots+1
                samples_dict = {"gauss_samples": samples, "gauss_y":gauss_y, "flow_samples":flow_samples,
                                "flow_y":flow_y, "true_y":true_y,"true_x":x_test[n]}
                np.save(os.path.join(save_dir,"plot_samples_{}.npy".format(self.i_plots)),samples_dict)

                
            mse, new_means = self.compute_mse(y_test[n], pred, z_query)
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        if self.is_flow:
            return mse, NLL, new_means, lower, upper, x_test[n], y_test[n]
        else:
            return mse, NLL, pred.mean, lower, upper, x_test[n], y_test[n]

    def get_support_query_objects(self, n_support, params):
        inputs, targets = get_objects_batch(self.x_objects_train,
                                            self.y_objects_train,
                                            params.meta_batch_size,
                                            params.update_batch_size,
                                            params.num_tasks)
        inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1, 128, 128))
        support_ind = list(np.random.choice(list(range(30)), replace=False, size=n_support))
        query_ind = [i for i in range(30) if i not in support_ind]
        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)
        x_support = inputs[:, support_ind, :, :, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :, :, :].to(self.device)
        y_query = targets[:, query_ind].to(self.device)
        return x_all, x_support, x_query, y_all, y_support, y_query

    def get_support_query_qmul(self, n_support):
        inputs, targets = get_batch(test_people, data_dir=self.config.data_dir["qmul"])
        support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        query_ind = [i for i in range(19) if i not in support_ind]
        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)
        x_support = inputs[:, support_ind, :, :, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :, :, :].to(self.device)
        y_query = targets[:, query_ind].to(self.device)
        return x_all, x_support, y_all, y_support

    def get_support_query_sines(self, n_support, params):
        batch, batch_labels, amp, phase = SinusoidalDataGenerator(200, params.meta_batch_size, params.num_tasks,
                                                                  params.multidimensional_amp,
                                                                  params.multidimensional_phase, params.noise,
                                                                  params.out_of_range).generate()
        if self.num_tasks == 1:
            inputs = torch.from_numpy(batch)
            targets = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)
        else:
            inputs = torch.from_numpy(batch)
            targets = torch.from_numpy(batch_labels)

        support_ind = list(np.random.choice(list(range(200)), replace=False, size=n_support))
        query_ind = [i for i in range(200) if i not in support_ind]

        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)
        x_support = inputs[:, support_ind, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        return x_all, x_support, y_all, y_support


    def get_support_query_nasdaq(self, n_support, params):
        nasdaq100padding = Nasdaq100padding(directory=self.config.data_dir['nasdaq'], normalize=True, partition="train",
                                            window=params.update_batch_size * 2,
                                            time_to_predict=params.meta_batch_size * 2)
        data_loader = torch.utils.data.DataLoader(nasdaq100padding, batch_size=params.update_batch_size * 2,
                                                  shuffle=True)
        batch, batch_labels = next(iter(data_loader))
        inputs = batch.reshape(params.update_batch_size * 2, params.meta_batch_size * 2, 1)
        targets = batch_labels[:, :, -1].float()

        support_ind = list(np.random.choice(list(range(10)), replace=True, size=n_support))
        query_ind = [i for i in range(10) if i not in support_ind]
        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)
        x_support = inputs[:, support_ind, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        return x_all, x_support, y_all, y_support

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict = self.feature_extractor.state_dict()

        state_dicts = {'gp': gp_state_dict, 'likelihood': likelihood_state_dict,
                       'net': nn_state_dict}
        if self.is_flow:
            cnf_dict = self.cnf.state_dict()
            state_dicts['cnf'] = cnf_dict
        torch.save(state_dicts, checkpoint)

    def load_checkpoint(self, checkpoint, device):
        ckpt = torch.load(checkpoint,map_location=torch.device('cpu')    )
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])
        if self.is_flow:
            self.cnf.load_state_dict(ckpt['cnf'])
            
    def get_test_set(self):
        if self.dataset == 'neural':
            return self.neural_loader.get_test_set()
        else:
            ValueError('Not on neural set')
    
    def standard_test(self, test_data_np):
        """
        Standard test function for the NGGP class.
        Evaluates the model on a fixed test set.

        Args:
            test_data_np (np.array): Test data in the form of a numpy array [(input, label), ...]

        Returns:
            float, float: Average loss and mean squared error (MSE) on the test set.
        """
        # Convert the numpy test data to PyTorch tensors
        test_data = [(torch.tensor(inp), torch.tensor(lbl)) for inp, lbl in test_data_np]

        # Set the model and its components to evaluation mode
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()
        if self.is_flow:
            self.cnf.eval()

        # Initialize metrics
        total_loss = 0.0
        total_mse = 0.0
        count = 0

        # Evaluate the model on the test data
        with torch.no_grad():
            for inputs, labels in test_data:
                # Ensure data is on the correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass through the model
                z = self.feature_extractor(inputs)
                predictions = self.model(z)

                # Calculate loss and MSE
                loss = -self.mll(predictions, labels)  # Negative log-likelihood loss
                mse, _ = self.compute_mse(labels, predictions, z)  # Assuming compute_mse returns MSE and another value

                # Aggregate the metrics
                total_loss += loss.item()
                total_mse += mse.item()
                count += 1

        # Compute average metrics
        avg_loss = total_loss / count
        avg_mse = total_mse / count

        return avg_loss, avg_mse

class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, config, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ## RBF kernel
        if kernel == 'rbf' or kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' Must be RBG for ANGGP'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, config, train_x, train_y, likelihood, kernel='nn', num_tasks=2, multi_type=2):
        super(MultitaskExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
            )
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' is not supported for ANGGP.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
