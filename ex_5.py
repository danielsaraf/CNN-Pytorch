import os
import os.path
import sys
from torch.autograd import Variable
import soundfile as sf
import librosa
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

EPOCHS = 100
BATCH_SIZE = 32
class_number = 30
LR = 0.001


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = sf.read(path)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)

        if len(spects) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.len = len(self.spects)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        # print(index)
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        # print (path)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return self.len


def get_data():
    train_dataset = GCommandLoader(sys.argv[1])
    int_to_class_dict = {v: k for k, v in train_dataset.class_to_idx.items()}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    valid_dataset = GCommandLoader(sys.argv[2])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, pin_memory=True)

    test_dataset = GCommandLoader(sys.argv[3])
    test_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=True)
    return train_loader, valid_loader, test_loader, int_to_class_dict


def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = _make_layers(
            [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, class_number)
        self.optimizer = optimizer.Adam(self.parameters(), lr=LR)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


def train_model(model, training_loader, epoc):
    model.train()
    train_loss = 0
    correct = 0
    # iterate once over training_loader (1 epoc)
    for batch_idx, (data_x, target) in enumerate(training_loader):
        print("epoc: " + str(epoc) + " batch:" + str(batch_idx) + " of: " + str(30000 / BATCH_SIZE))
        data_x, target = Variable(data_x), Variable(target)
        model.optimizer.zero_grad()
        output = model(data_x)
        loss = F.nll_loss(output, target)
        loss.backward()
        model.optimizer.step()
        # calculate loss and accuracy for report file
        train_loss += loss
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum()
    train_loss /= len(training_loader.dataset) / BATCH_SIZE
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(training_loader.dataset),
        100. * correct / len(training_loader.dataset)))


def test_model(model, validation_loader, epoc):
    # iterate over validation_loader and calculate loss and accuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data_x, target) in enumerate(validation_loader):
            print("epoc: " + str(epoc) + " valid number:" + str(idx) + " of: 6798")
            output = model(data_x)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(validation_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    return test_loss


def write_predicts(model, test_loader, int_to_class_dict):
    # get predication for each picture in test_x and write its prediction to test_y
    predictions_dics = {}
    model.eval()
    files = [x[0][x[0].rfind('\\')+1:] for x in test_loader.sampler.data_source.spects]
    with torch.no_grad():
        for idx, (data_x, target) in enumerate(test_loader):
            output = model(data_x)
            pred = output.max(1, keepdim=True)[1].item()
            predictions_dics[files[idx]] = str(files[idx]) + "," + str(int_to_class_dict.get(pred))

    with open('test_y', 'w') as out:
        for i in range(len(predictions_dics)):
            p = predictions_dics.get(str(i) + ".wav")
            out.write(str(p))
            if idx < len(predictions_dics) - 1:
                out.write("\n")


def main():
    # get train,validation and test data
    training_loader, validation_loader, test_loader, int_to_class_dict = get_data()

    # train & test model
    cnn_model = CNNModel()

    last_test_loss = None
    for epoc in range(EPOCHS):
        train_model(cnn_model, training_loader, epoc)
        curr_test_loss = test_model(cnn_model, validation_loader, epoc)
        if last_test_loss is not None:
            if curr_test_loss > last_test_loss:
                break
            else:
                last_test_loss = curr_test_loss

    # calculate and write predicts
    write_predicts(cnn_model, test_loader, int_to_class_dict)


if __name__ == '__main__':
    main()
