import torch 
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset 

from tqdm import tqdm 
from PIL import Image 
import os 

class HAM10000(Dataset):
    def __init__(self, df, transform) -> None:
        super().__init__()
        self.df = df 
        self.transform = transform 

    def __getitem__(self, idx):
        X = self.transform(Image.open(os.path.join('HAM10000_images',self.df['image_id'][idx])))
        y = torch.tensor(int(self.df['cell_type_idx'][idx]))
        return X, y 
    
    def __len__(self):
        return self.df.shape[0]
    
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_list[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device='cuda'):
    for epoch in range(num_epochs):
        # Initialize metrics
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)

        # Training loop
        model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=num_train_batches, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            
            optimizer.zero_grad()
            out = model(inputs)
            outputs = out.logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in tqdm(enumerate(val_loader), total=num_val_batches, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                
                out = model(inputs)
                outputs = out.logits
                loss = criterion(outputs, labels)

                # Update metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_acc += (predicted == labels).sum().item()

        # Calculate metrics
        train_loss /= num_train_batches
        train_acc /= len(train_loader.dataset)
        val_loss /= num_val_batches
        val_acc /= len(val_loader.dataset)

        # Print metrics and progress bar
        tqdm.write(f'Epoch {epoch + 1}/{num_epochs} - Training accuracy: {train_acc:.4f} - Training loss: {train_loss:.4f} - Validation accuracy: {val_acc:.4f} - Validation loss: {val_loss:.4f}')