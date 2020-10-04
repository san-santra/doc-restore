from keras.models import load_model
from keras_contrib.losses import DSSIMObjective
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from model import TextExtractor
from data_gen import PatchifyDB


def load_model_weights_from_keras(model, keras_weights):
    import numpy as np

    # model.conv1.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[0], (3, 2, 0, 1)))
    # model.conv1.bias.data = torch.from_numpy(keras_weights[1])
    # model.conv2.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[2], (3, 2, 0, 1)))
    # model.conv2.bias.data = torch.from_numpy(keras_weights[3])
    # model.conv3.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[4], (3, 2, 0, 1)))
    # model.conv3.bias.data = torch.from_numpy(keras_weights[5])
    # model.conv4.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[6], (3, 2, 0, 1)))
    # model.conv4.bias.data = torch.from_numpy(keras_weights[7])

    # model.convt1.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[8], (3, 2, 0, 1)))
    # model.convt1.bias.data = torch.from_numpy(keras_weights[9])
    # model.convt2.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[10], (3, 2, 0, 1)))
    # model.convt2.bias.data = torch.from_numpy(keras_weights[11])
    # model.convt3.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[12], (3, 2, 0, 1)))
    # model.convt3.bias.data = torch.from_numpy(keras_weights[13])
    # model.convt4.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[14], (3, 2, 0, 1)))
    # model.convt4.bias.data = torch.from_numpy(keras_weights[15])
    # model.convt5.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[16], (3, 2, 0, 1)))
    # model.convt5.bias.data = torch.from_numpy(keras_weights[17])
    # model.convt6.weight.data = \
    #     torch.from_numpy(np.transpose(keras_weights[18], (3, 2, 0, 1)))
    # model.convt6.bias.data = torch.from_numpy(keras_weights[19])

    model.conv1.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[0]))
    model.conv1.bias.data = torch.from_numpy(keras_weights[1])
    model.conv2.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[2]))
    model.conv2.bias.data = torch.from_numpy(keras_weights[3])
    model.conv3.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[4]))
    model.conv3.bias.data = torch.from_numpy(keras_weights[5])
    model.conv4.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[6]))
    model.conv4.bias.data = torch.from_numpy(keras_weights[7])

    model.convt1.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[8]))
    model.convt1.bias.data = torch.from_numpy(keras_weights[9])
    model.convt2.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[10]))
    model.convt2.bias.data = torch.from_numpy(keras_weights[11])
    model.convt3.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[12]))
    model.convt3.bias.data = torch.from_numpy(keras_weights[13])
    model.convt4.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[14]))
    model.convt4.bias.data = torch.from_numpy(keras_weights[15])
    model.convt5.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[16]))
    model.convt5.bias.data = torch.from_numpy(keras_weights[17])
    model.convt6.weight.data = \
        torch.from_numpy(np.transpose(keras_weights[18]))
    model.convt6.bias.data = torch.from_numpy(keras_weights[19])


if __name__ == '__main__':
    in_path = '../data/ourdata/X/'
    target_path = '../data/ourdata/Y/'
    patch_size = 256
    patch_per_image = 8
    batch_size = 8
    num_workers = 2

    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()])
    db = PatchifyDB(in_path, target_path, patch_size, patch_per_image,
                    transform=transform)

    # keras model
    model = load_model('../ProjectDigitalHeritage/upto2017_model_ourdata.h5',
                       custom_objects={'DSSIMObjective':
                                       DSSIMObjective(kernel_size=23)})

    keras_wt = model.get_weights()

    # torch model
    p_model = TextExtractor()

    load_model_weights_from_keras(p_model, keras_wt)
    torch.save(p_model.state_dict(), 'upto2017_model_ourdata.pt')

    # test
    sample = db[0]

    with torch.no_grad():
        out = p_model(sample[0].unsqueeze(0))

    plt.subplot(1, 3, 1)
    plt.imshow(sample[0][0])
    plt.subplot(1, 3, 2)
    plt.imshow(sample[1][0])
    plt.subplot(1, 3, 3)
    plt.imshow(out[0, 0])

    # plt.ion()
    plt.show()
