import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxEQL(nn.Module):
    def __init__(self, num_classes):
        super(SoftmaxEQL, self).__init__()
        # initialize the class frequencies with ones
        self.class_freqs = torch.ones(num_classes).cuda()
        #self.loss_func = torch.nn.functional.cross_entropy.cuda()
    
    def forward(self , logits , labels):
        # get the batch size and number of classes
        batch_size , num_classes = logits.size()
        
        # update the class frequencies with the current batch labels
        for i in range(num_classes):
            self.class_freqs[i] += (labels == i).sum()

        # normalize the class frequencies to get the class probabilities
        class_probs = self.class_freqs / self.class_freqs.sum()

        # generate a random variable beta that follows a Bernoulli distribution with probability gamma
        gamma = 0.5 # you can tune this parameter
        beta = torch.bernoulli(torch.full((num_classes,), gamma)).cuda()

        # reweigh the class probabilities with beta
        reweighed_probs = beta * (1 - class_probs) + (1 - beta) * class_probs

        # compute the softmax cross-entropy loss with reweighed probabilities as weights
        loss = F.cross_entropy(logits , labels , weight=reweighed_probs)

        return loss