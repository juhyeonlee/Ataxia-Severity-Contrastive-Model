import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, torch_pad_nan
from scipy.spatial.transform import Rotation as R


class ContrastiveModel:
    '''The contrastive model for ataxia severity assessment'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
    ):
        ''' Initialize a contrastive model.
        
         Parameters:
            input_dims (int): The input dimension.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size

        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.n_epochs = 0
        self.n_iters = 0
    
    def fit(self, train_data, contrast_weight, n_epochs=None, verbose=False, ):
        ''' Training the contrastive model for ataxia severity assessment.
        
        Parameters:
            train_data (numpy.ndarray): The training data. (Velocity input segments) It should have a shape of (n_instance, n_timestamps, 3).
            contrast_weight (numpy.ndarray): The weights used for contrasting. (Clinician-scord Total BARS) It should have a shape of (n_instance).
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''

        assert train_data.ndim == 3  # to apply random rotation

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), torch.from_numpy(contrast_weight).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)

        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            for batch in train_loader:
                
                x = batch[0]
                rot_mat = R.random(32).as_matrix()
                rot_mat = torch.from_numpy(rot_mat).to(torch.float)
                x = x @ rot_mat
                
                x = x.to(self.device)
                w = batch[1].to(self.device)

                ts_l = torch.sum(~torch.isnan(x).all(axis=-1), axis=1)
                min_ts_l = torch.min(ts_l).item()

                crop_l = np.random.randint(low=2, high=min_ts_l+1)
                crop_left = np.random.randint(min_ts_l - crop_l + 1) 
                crop_right = crop_left + crop_l
                
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=min_ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=min_ts_l - crop_eright + 1, size=x.size(0))
                optimizer.zero_grad()
                
                # movement elem
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l] 

                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    contrast_weight=w
                )
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1

            scheduler.step()
                
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose and self.n_epochs % 10 == 0:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Parameters:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
