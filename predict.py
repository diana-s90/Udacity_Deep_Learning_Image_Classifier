import argparse
import torch
import numpy as np
from PIL import Image
import json

def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    parser.add_argument('--image', 
                        type=str, 
                        help='Specify path to image to be predicted',
                        required=True)
    parser.add_argument('--checkpoint',
                        type=str,
                        default="checkpoint.pth",
                        help='Specify path to .pth file to load model')
    parser.add_argument('--topk',
                        type=int,
                        default=1,
                        help='Choose top k predicted classes as int')
    parser.add_argument('--category_names',
                        dest='category_names',
                        metavar='C',
                        type=str,
                        default='cat_to_name.json',
                        help='Specify file needed for mapping category numbers to names.')
    parser.add_argument('--gpu',
                        type=bool,
                        nargs='?',
                        default=False,
                        help='Use GPU for inference')
    args = parser.parse_args()
    return args

#Image Processer:  #Scales, crops, and normalizes a PIL image for a PyTorch model,
#returns an Numpy array
def process_image(image): 
    
    image = Image.open(image)
    width, height = image.size
    #set shortest side to 256 pixels while keeping the aspect ratio
    if height < width: 
        resize_size = [256*width/height, 256]
    else:
        resize_size = [256, 256*height/width]  
    image.thumbnail(size=resize_size)
    
    #crop from center
    thumb_width, thumb_height = image.size
    crop_size = 224
    left = (thumb_width-crop_size)/2 
    right = (thumb_width+crop_size)/2
    upper = (thumb_height-crop_size)/2 
    lower = (thumb_height+crop_size)/2
    image = image.crop((left, upper, right, lower))
    
    #convert values
    np_image = np.array(image)/255
    
    #normalize according to instrutions
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    #Reorder Array Dimensions
    image = np_image.transpose(2, 0, 1)
    
    return image
  
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    #load information from the checkpoint
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])    
    return model

#Predict the top k clases of an image using a pretrained deep learning model
def predict(np_image, model, topk, cat_to_name, device):    
    tensor_image = torch.from_numpy(np.expand_dims(np_image, axis=0)).type(torch.FloatTensor).to(device)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(tensor_image))
        probs, top_classes = ps.topk(topk)
        #Convert Class Probabilities to Numpy Array
        probs = np.array(probs).squeeze()
       
        top_classes = np.array(top_classes)
        #Convert Classes to Names
        idx_to_class = dict(map(reversed, model.class_to_idx.items()))
        classes = [idx_to_class[idx] for idx in top_classes[0]]      
        names = []
        for cls in classes:
            names.append(cat_to_name[str(cls)])

    return(probs, names)

def print_results(probs, names):
    for i, j in enumerate(zip(names, probs)):
        print(f"Rank: {i+1}..",
              "Flower: {}..".format(j[0]), 
              "Likelihood: {}%..".format(j[1]*100))
def main():
    args = arg_parser()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
    #Load Category Names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)    
            
    model = load_checkpoint(args.checkpoint)
    np_image = process_image(args.image)
    probs, names = predict(np_image, model, args.topk, cat_to_name, device)
    
    if args.topk > 1:
            print_results(probs, names)
    else: 
        print(f"The top predicted class is {names[0]} with a likelihood of {str(probs*100)}%.")
  

if __name__ == '__main__': main()