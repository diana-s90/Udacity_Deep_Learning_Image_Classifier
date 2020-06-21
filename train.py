import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")                     
    parser.add_argument('--arch',
                        type=str,
                        default='vgg16',
                        choices=['vgg16', 'vgg19'],
                        help="Choose pretrained Neural Network - Default is VGG16")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help="Specify learning rate as Float - Default is 0.001")
    parser.add_argument('--hidden_units',
                        type=int,
                        default=1632,
                        help="Hidden units as int")
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help="Specify number of training epochs - Default is 5")
    parser.add_argument('--gpu',
                        type=bool,
                        nargs='?',
                        default=False,
                        help='Use GPU for training')
    args = parser.parse_args()
    return args

def train_dataloader(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(p=0.4),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    return train_data, trainloader

def validation_dataloader(valid_dir):
    test_transforms = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    return validloader

def load_pretrained_model(model_name):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    else: 
        model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model 

def replace_classifier(model, hidden_units):
    input_features = model.classifier[0].in_features
    classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.3),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    return classifier

def train_model(model, trainloader, validloader, device, learning_rate, epochs):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)        
    model.to(device)
    print('Traning Sequence Initialized...\n')
    steps = 0
    running_loss = 0
    print_every = 12
    print("Print Every: " +str(print_every))
    for e in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                steps += 1
                
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy =0
                    model.eval()
                    with torch.no_grad():
                        for images, labels in validloader:
                            images, labels = images.to(device), labels.to(device)
                            log_ps = model(images)
                            loss = criterion(log_ps, labels)
                            test_loss += loss.item()
                            
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                        print(f"Epoch {e+1}/{epochs} | ",
                              f"Train Loss: {running_loss/print_every:3f} | ",
                              f"Test Loss: {test_loss/len(validloader):3f} | ",
                              f"Accuracy: {accuracy/len(validloader):3f}")
                        running_loss = 0
                        model.train()
    print('Finished Training Sequence.')
    return model

def save_checkpoint(model, train_data):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'model': model,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')
  
def main():
    args = arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    print('Specifications:')
    print('Device used: ' + str(device))
    print('Architecture: ' + args.arch)
    print('Hidden Units: ' + str(args.hidden_units))
    print('Learning Rate: ' + str(args.learning_rate) + '...\n')

    train_data, trainloader = train_dataloader(train_dir)
    validloader = validation_dataloader(valid_dir)
    
    model = load_pretrained_model(args.arch)
    model.classifier = replace_classifier(model, args.hidden_units)
    
    train_model(model, trainloader, validloader, device, args.learning_rate, args.epochs)
    
    save_checkpoint(model, train_data)
    
    
if __name__ == '__main__': main()
