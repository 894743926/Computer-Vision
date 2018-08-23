import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import copy


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            # freeze the parameters
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_rate = 0.5
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=False)
        
        self.dropout = nn.Dropout(self.drop_rate)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def one_hot(self, inp, size):
        """
        One a 2-D tensor to a 3-D tensor
        """
 
        inp_ = torch.unsqueeze(inp, len(inp.size()))
        one_hot = torch.FloatTensor(*inp.size(), size).zero_()
        
        one_hot.scatter_(len(inp.size()), inp_.to(torch.device('cpu')), 1);
        
        return one_hot.to(inp.device)
    
    def forward(self, features, captions):
        
        def feed_lstm(x, hc):
            x, (h, c) = self.lstm(x.unsqueeze(1), hc)
            x = x.view(x.size()[0]*x.size()[1], self.hidden_size)
            x = self.dropout(x)
            x = self.fc(x)
            return x, (h, c)
        
        # init first hidden layers
        init_hc = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(features.device),
                   Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(features.device))
        # get the caption length
        words_len = captions.size()[1]
        
        # first lstm input x_-1 from images encoder
        x, hc = feed_lstm(features, init_hc)
        
        # output content
        output = torch.zeros(captions.shape[0], captions.shape[1], self.vocab_size).to(features.device)
        output[:,0,:] = copy.copy(x)
        
        embedding = self.embed(captions[:,:-1])
        
        # for training
        for i in range(words_len-1):
            x = embedding[:,i,:]
            x, hc = feed_lstm(x, hc)
            # add the x to the output array
            output[:,i+1,:] = copy.copy(x)
            
         
        return output
               

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        def feed_lstm(x, hc):
            x, (h, c) = self.lstm(x, hc)
            x = x.view(x.size()[0]*x.size()[1], self.hidden_size)
            x = self.fc(x)
            return x, (h, c)
        
        # init first hidden layers
        init_hc = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(inputs.device),
                   Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(inputs.device))
        hc = init_hc if states == None else states
        
        
        output = []
        
        # first lstm input x_-1 from images encoder
        x = inputs
        
        for i in range(max_len):
            
            x, hc = feed_lstm(x, hc)

            # add the x to the output array
            predicted_index = x.argmax(1).item()
            output.append(predicted_index)
            if predicted_index == 1:
                break                  
            
            # prepare x for next prediction
            x = self.embed(x.argmax(1)).unsqueeze(1)
            
        return output