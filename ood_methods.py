import numpy as np 
import torch
from torch.nn.functional import softmax
from torch.autograd import Variable
from sklearn.decomposition import PCA

class ForwardHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output.last_hidden_state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()




class OpenPCS():
    def __init__(self, model_name, model, device = 'cuda', n_components = 3) -> None:
        
        self.device = device 
        self.model_name = model_name
        if self.model_name == 'microsoft/resnet-50':
            self.layer = model.resnet
        elif self.model_name == 'google/vit-base-patch16-224':
            self.layer = model.vit
        else:
            self.layer = model.convnext

        self.model = model
        self.n_components = n_components

    def get_id_features(self, train_loader):

        np_features = None
        np_labels = None

        for batch in train_loader:
            input, label = batch
            input = Variable(input).to(self.device)
            labels = label.numpy()
            self.model.eval()
            #with torch.no_grad():
            with ForwardHook(self.layer) as hook:        

                output = self.model(input)

                act = hook.features.float()
                if self.model_name == 'google/vit-base-patch16-224':
                    features = act.mean(dim=1).view(act.shape[0], -1).detach().cpu().numpy()
                else: 
                    act = act.mean(dim=2)
                    features = act.mean(dim=2).view(act.shape[0], -1).detach().cpu().numpy()
                
                if np_features is None:
                        np_features = features 
                        np_labels = labels
                else:
                    np_features = np.vstack((np_features, features))
                    np_labels = np.concatenate((np_labels, labels))
        return np_features, np_labels

    
    def get_ood_features(self, model, ood_loader):
        ood_features = None

        for batch in ood_loader:
            if len(batch) == 2:
                input, _ = batch
            else:
                input = batch 
            input = Variable(input).to(self.device)
            model.eval()
            with torch.no_grad():
                with ForwardHook(self.layer) as hook:        
                        # Do a forward and backward pass.
                        output = model(input)
                        
                        act = hook.features.float()
                        if self.model_name == 'google/vit-base-patch16-224':
                            features = act.mean(dim=1).view(act.shape[0], -1).detach().cpu().numpy()
                        else: 
                            act = act.mean(dim=2)
                            features = act.mean(dim=2).view(act.shape[0], -1).detach().cpu().numpy()
                        
                        if ood_features is None:
                            ood_features = features 
                        else:
                            ood_features = np.vstack((ood_features, features))

        return ood_features    
    
    def fit_PCA(self, in_scores=None, labels=None):
        self.pca_ = {} 
       
        for i in np.unique(labels):
            pca = PCA(n_components=self.components)
            
            X_fit = in_scores[labels==i]
            print(X_fit.shape)
            pca.fit(X_fit)
            self.pca_[f'pca_{i}'] = pca

    def get_scores(self, states):

        scores_ = []
        for _, estimator in self.pca_.items():
            scores = estimator.score_samples(states)
            scores_.append(scores)
        
        pca_scores = np.max(np.array(scores_), axis=0)

        return pca_scores
    
def msp_get_scores(model, ood_loader, device='cuda'):
    msp_ood_features = None

    for batch in ood_loader:
        if len(batch) == 2:
            input, label = batch
        else:
            input = batch
        input = Variable(input).to(device)
        model.eval()
        with torch.no_grad():
            out_logits = model(input)


            msp, _ = torch.max(softmax(out_logits, dim = 1),axis=1)
            # msp, _ = torch.max(softmax(output.logits, dim = 1),axis=1)
            msp = msp.detach().cpu().numpy()
            
            if msp_ood_features is None:
                    msp_ood_features = msp 
                    
            else:
                msp_ood_features = np.concatenate((msp_ood_features, msp))
    
    return msp_ood_features

from sklearn.mixture import GaussianMixture


class Mahalanobis():
    def __init__(self, n_components=3):
        self.components = n_components
        
    def fit_PCA(self, in_scores=None, labels=None):
        self.pca_ = {} 
       
        for i in np.unique(labels):
            pca = GaussianMixture(n_components=self.components)
            
            X_fit = in_scores[labels==i]
            pca.fit(X_fit)
            self.pca_[f'pca_{i}'] = pca

    def get_scores(self, states):

        scores_ = []
        for _, estimator in self.pca_.items():
            x_mu = states - estimator.means_[0,:]
            inv_covmat = np.linalg.inv(estimator.covariances_[0,:])
            left = np.dot(x_mu, inv_covmat)
            mahal = np.dot(left, x_mu.T)
            scores_.append(mahal.diagonal())
        pca_scores = np.max(np.array(scores_), axis=0)

        return pca_scores

def energy_get_scores(model, ood_loader, device='cuda'):
    ood_features = None

    temp = 1

    for batch in ood_loader:
        if len(batch) == 2:
            input, label = batch
        else:
            input = batch
        input = Variable(input).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(input)

            out = temp*torch.logsumexp(output.logits/temp,dim=1)
            energy = out.detach().cpu().numpy()
            
            if ood_features is None:
                    ood_features = energy 
                    
            else:
                ood_features = np.concatenate((ood_features, energy))

    return ood_features
            