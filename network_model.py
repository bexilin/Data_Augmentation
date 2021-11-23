from torch import nn

class Aug_Model(nn.Module):
    def __init__(self, class_num):
        super(Aug_Model, self).__init__()
        self.class_num = class_num
        self.augment = nn.Sequential(
            nn.Conv2d(6,16,3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(16,3,3,padding='same')
        )
        self.classify = nn.Sequential(
            nn.Conv2d(3,16,3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(32*16*16,1024),
            nn.Dropout(),
            nn.Linear(1024,self.class_num),
            nn.Softmax(dim=1)
        )

        # minibatch dimension: B x C x H x W
    
    def forward(self,x):
        self.aug_img = self.augment(x)
        return self.classify(self.aug_img)

    def getTargetImg(self):
        return self.aug_img
