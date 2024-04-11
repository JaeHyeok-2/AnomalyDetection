import torch 
import torch.nn as nn  
import numpy as np  



class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, input_dropout_rate = 0.0, reccurent_dropout_rate = 0.0):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.input_dropout_rate = np.clip(input_dropout_rate, 0.0, 1.0)
        self.recurrent_dropout_rate = np.clip(reccurent_dropout_rate, 0.0, 1.0)
        
        self.padding = int((kernel_size - 1) / 2)
        
        self.input_dropout = nn.Dropout(p=self.input_dropout_rate) if 0.0 < self.input_dropout_rate < 1.0 else nn.Identity()
        self.recurrent_dropout = nn.Dropout(p=self.recurrent_dropout_rate) if 0.0 < self.input_dropout_rate < 1.0 else nn.Identity()
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1 , self.padding, bias=False) 
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False) 
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wci = None
        self.Wcf = None 
        self.Wco = None 
        
    def forward(self, x, h, c):
        x = self.input_dropout(x) 
        h = self.recurrent_dropout(h)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci) 
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf) 
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc) 
        return ch, cc 
    
    
    def init_hidden(self, batch_size, hidden, shape, use_cuda=False):
        if self.Wci is None: 
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1])
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1])
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1]) 
            
        else: # H W              # B C H W 
            assert shape[0] == self.Wci.size()[2], 'Input height Mismatched!' 
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'

        if use_cuda: 
            self.Wci = self.Wci.cuda() 
            self.Wcf = self.Wcf.cuda()
            self.Wco = self.Wco.cuda() 
            
        h = torch.zeros(batch_size, hidden, shape[0], shape[1]) 
        c = torch.zeros(batch_size, hidden, shape[0], shape[1])
        
        if use_cuda:
            h, c = h.cuda(), c.cuda() 
            
        return (h, c)
    
    
    
    

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, batch_first=False, input_dropout_rate= 0.0, recurrent_dropout_rate = 0.0):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels 
        
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.batch_first = batch_first
        
        if not isinstance(input_dropout_rate, list):
            self.input_dropout_rate = [input_dropout_rate] * self.num_layers
        if not isinstance(recurrent_dropout_rate, list):
            self.recurrent_dropout_rate = [recurrent_dropout_rate] * self.num_layers 
            
        self._all_layers = nn.ModuleList() 
        
        for i in range(self.num_layers):
            name = 'cell{}'.format(i) 
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.input_dropout_rate[i], self.recurrent_dropout_rate[i])
            setattr(self, name, cell) 
            self._all_layers.append(cell) 
            
            
    def forward(self, inputs, hidden_state=None):
        '''
        input Tensor : 5-D tensor shape (t, b, c, h, w) or (b, t, c, h, w)
        '''
        
        if not self.batch_first:
            inputs = inputs.permute(1, 0, 2, 3, 4)
            
        internal_state = [] 
        outputs = [] 
        
        n_steps = inputs.size(1)
        for t in range(n_steps):
            x = inputs[:, t, :, :, :]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i) 
                if t == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size = bsize, hidden= self.hidden_channels[i], shape = (height, width), use_cuda = inputs.is_cuda)
                    internal_state.append((h, c))
                    
                
                (h, c) = internal_state[i] 
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c) 
                
            outputs.append(x) 
        outputs = torch.stack(outputs, dim=1) 

        return outputs, (x, new_c) 