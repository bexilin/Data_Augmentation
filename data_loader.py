import cv2
import os
import numpy as np
from typing import List, Tuple

class Data_Handler():
    def __init__(self, data_dir: str, classes: List[str], grayscale=False) -> None:
        
        # TODO: Consider only load image when sampling
        training_dir = os.path.join(data_dir,"train")
        test_dir = os.path.join(data_dir,"train/images")
        
        print("loading data\n")
        self.classes = classes
        self.training_set = []
        self.validation_set = []
        for label in range(len(self.classes)):
            class_train_set = []
            class_val_set = []
            class_dir = os.path.join(training_dir,self.classes[label],"images")
            for idx in range(500):
                img_name = self.classes[label]+"_"+str(idx)+".JPEG"
                if grayscale:
                    img = cv2.imread(os.path.join(class_dir,img_name),0)
                    img = img.astype(float)
                    img = np.array([img])
                else: 
                    img = cv2.imread(os.path.join(class_dir,img_name))
                    assert not isinstance(img,type(None)), "Failed to load an image!"
                    img = img.astype(float)
                    img = np.moveaxis(img,2,0)
                if idx < 400:
                    class_train_set.append(img)
                else:
                    class_val_set.append(img)
            self.training_set.append(class_train_set)
            self.validation_set.append(class_val_set)

    def Samples(self, num: int = 40) -> Tuple[np.ndarray, np.ndarray]: 
        # Since we have same amount of training data in each class, sample number should also be the same
        num_per_class = int(num / len(self.classes))
        labels = np.zeros((num_per_class*len(self.classes),len(self.classes)))
        
        first_sample = True
        for label in range(len(self.classes)):
            for idx in range(num_per_class):
                rnd_1 = np.random.randint(400)
                rnd_2 = np.random.randint(400)
                rnd_3 = np.random.randint(400)

                input_sample = np.concatenate((self.training_set[label][rnd_1],self.training_set[label][rnd_2]))
                target_sample = self.training_set[label][rnd_3]

                if first_sample:
                    input_samples = np.array([input_sample])
                    target_samples = np.array([target_sample])
                    first_sample = False
                else:
                    input_samples = np.concatenate((input_samples,np.array([input_sample])))
                    target_samples = np.concatenate((target_samples,np.array([target_sample])))
                labels[label*num_per_class+idx,label] = 1.0

            # print(self.classes[label]," : ",rnd_1)
            # img = self.training_set[label][rnd_1]
            # img = np.moveaxis(img,0,2).astype(np.uint8)
            # cv2.imshow("1",img)
            # cv2.waitKey(0)

        # Shuffle samples
        idx = np.random.permutation(num)
        input_samples = input_samples[idx,:,:,:]
        target_samples = target_samples[idx,:,:,:]
        labels = labels[idx,:]

        return input_samples, target_samples, labels 

    def GetValSet(self,stack=True):
        labels = np.zeros((100*len(self.classes),len(self.classes)))

        first_sample = True
        for label in range(len(self.classes)):
            for idx in range(100):
                input_sample = self.validation_set[label][idx]
                if stack:
                    # Double the input sample for consistency with NN
                    input_sample = np.concatenate((input_sample,input_sample))
                if first_sample:
                    input_samples = np.array([input_sample])
                    first_sample = False
                else:
                    input_samples = np.concatenate((input_samples,np.array([input_sample])))
                labels[label*100+idx,label] = 1.0

        # Shuffle samples
        idx = np.random.permutation(100)
        input_samples = input_samples[idx,:,:,:]
        labels = labels[idx,:]

        return input_samples, labels
