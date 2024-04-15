import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image


num_classes = 15

font_names = {0:"Aguante-Regular",
1:"ambidexter_regular",
2:"ArefRuqaaInk-Bold",
3:"ArefRuqaaInk-Regular",
4:"better-vcr-5.2",
5:"BrassMono-Bold",
6:"BrassMono-BoldItalic",
7:"BrassMono-Italic",
8:"BrassMono-Regular",
9:"GaneshaType-Regular",
10:"GhastlyPanicCyr",
11:"Realest-Extended",
12:"AlumniSansCollegiateOne-Italic",
13:"AlumniSansCollegiateOne-Regular",
14:"TanaUncialSP"
}

test_transforms =transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU() ,
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)) 
        #self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(54 * 54 * 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

while True:
    print('*'*30)
    print('Введите путь до изображения: ')
    path = input()
    img_for_pred = Image.open(path)
    img_for_pred = test_transforms(img_for_pred)

    loaded_model = ConvNet()
    loaded_model.load_state_dict(torch.load('weights.pt', map_location=torch.device('cpu')))
    font_class = int(torch.argmax(loaded_model(img_for_pred.unsqueeze(0)), dim=1))
    probability = float(torch.softmax(loaded_model(img_for_pred.unsqueeze(0)), dim=1)[0][font_class])
    print('\n')
    print('Шрифт: %s, Вероятность: %.2f' %(font_names[font_class], probability))
    