from tsnecuda import TSNE
from tsne.resnet import ResNet18
import random
import numpy as np
import os
import torch
import torchvision.models as models
import torch.optim
from torchvision import transforms
model = models.resnet18()
model.eval()

import os
import torch
import torchvision.models as models
import torch.optim
model = models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=1e-4)
if os.path.isfile("checkpoint.pth.tar"):
    print("=> loading checkpoint '{}'".format("checkpoint.pth.tar"))
    loc = 'cuda:{}'.format(0)
    checkpoint = torch.load('checkpoint.pth.tar', map_location=loc)
    start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    best_acc1 = best_acc1.to(torch.device("cuda"))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format("model_best.pth.tar", checkpoint['epoch']))
model.eval()


def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)     

def get_features(model, transform_dataset):        # PyTorch在ImageNet上的pre-trained weight進行特徵萃取
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.eval()
    model.to(device)
    
    dataset = CustomDataset(df_tsne, transform = transform_dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, collate_fn=collate_skip_empty, shuffle=False, num_workers=4)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    labels = []
    image_paths = []
    print("Start extracting Feature")
    for i, (img, target) in enumerate(tqdm(dataloader)):
        images = img.to(device)
        target = target.squeeze().tolist()
        for element in target:
            labels.append(element)

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, labels    

def get_features_trained_weight(model, transform_dataset):        # 透過訓練好的pth檔案進行特徵萃取
    

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    
    model.eval()
    model.to(device)
    
    
    dataset = CustomDataset(df_tsne, transform = transform_dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, collate_fn=collate_skip_empty, shuffle=False, num_workers=4)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None
    
    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    print("Start extracting Feature")
    for i, (img, target) in enumerate(tqdm(dataloader)):
        
        feat_list = []
        def hook(module, input, output): 
            feat_list.append(output.clone().detach())
        
        images = img.to(device)
        target = target.squeeze().tolist()
        
        for element in target:
            labels.append(element)
        
        with torch.no_grad():
            handle=model.avgpool.register_forward_hook(hook) #擷取avgpool的output
            output = model.forward(images)
            feat = torch.flatten(feat_list[0], 1)            #將avgpool的output送入flatten layer
            handle.remove()
        
        current_features = feat.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, labels

# T-SNE    

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)
    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
        zip(images, labels, tx, ty),
        desc='Building the T-SNE plot',
        total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.show()
    plt.savefig('visualize_tsne_image.png')


def visualize_tsne_points(tx, ty, labels):
    print('Plotting TSNE image')
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    class_name = ['good','missing','shift','stand','broke','short']

    colors_per_class = {
        0 : [254, 202, 87],
        1 : [255, 107, 107],
        2 : [10, 189, 227],
        3 : [255, 159, 243],
        4 : [16, 172, 132],
        5 : [128, 80, 128]
    }
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        
        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255
        
        # add a scatter plot with the correponding color and label

        ax.scatter(current_tx, current_ty, c=color, label=label)

        # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()
    plt.savefig('visualize_tsne_points.png')


def visualize_tsne(tsne, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # visualize the plot: samples as images
    #visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)

def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample] # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':

    fix_random_seeds()
    if args.resume:                                                                #使用指定路徑載入訓練好的model
        model = model
        features, labels = get_features_trained_weight(model, transform_dataset)
        tsne = TSNE(n_components=2).fit_transform(features)
        visualize_tsne(tsne, labels)
    else:                                                                          #使用ImageNet上的Pre-trained weight
        model = ResNet18(pretrained=True)
        print("Using ResNet18 as feature extractor")   
        features, labels = get_features(model, transform_dataset)    
        tsne = TSNE(n_components=2).fit_transform(features)        
        visualize_tsne(tsne, labels)
    return