from torch.utils.data.dataset import Dataset
import torch
import json
import cv2
import os
import numpy as np


class MyCustomDataset(Dataset):
    def __init__(self, category, json_file_path="/Users/mjo/Desktop/WLASL/WLASL_v0.3.json", video_file_path="/Users/mjo/Desktop/WLASL/WLASL2000", frame_location="/Users/mjo/Desktop/WLASL/Processed_data/"):
        self.frame_location = frame_location
        # Defining count_dictionary which contains the number of videos for each class
        # and defining video_id_dictionary which has all of the video id's for each class.
        with open(json_file_path, "r") as read_file:
            data = json.load(read_file)

        self.count_dictionary = {}
        self.video_id_dictionary = {}

        for instance in data:
            current_label = instance['gloss']
            self.count_dictionary[current_label] = 0
            self.video_id_dictionary[current_label] = []
            inner_array = instance['instances']
            for video in inner_array:
                self.count_dictionary[current_label] += 1
                self.video_id_dictionary[current_label].append(video['video_id'])

       
        self.labels_iterated = {}
        self.inv_video_id_dictionary = {}
        self.training_data = []
        self.Categories(category, frame_location)

    def __getitem__(self, index):
        return self.training_data[index]


    def __len__(self):
        total = 0
        folders = ([name for name in os.listdir(os.path.join(self.frame_location,"{}/".format(len(self.labels_x))))
                    if os.path.isdir(os.path.join(self.frame_location,"{}/".format(len(self.labels_x)),name))])
        for folder in folders:
            contents = os.listdir(os.path.join(self.frame_location,"{}/".format(len(self.labels_x)),folder))
            total += len(contents)
        return total


    def total_videos(self):


        sum_count = 0
        for label in self.labels_x:
            for video in self.video_id_dictionary[label]:
                sum_count += 1
        return sum_count


    def make_training_data(self, labels_x,frame_location):


        data_directory = ("{}/{}".format(frame_location, len(labels_x)))
        num_labels = len(labels_x)
        for label in (labels_x):
            for video in self.video_id_dictionary[label]:
                path = os.path.join(data_directory, video)
                for file in (os.listdir(path)):
                    if "jpg" in file:
                        try:
                            path = os.path.join(data_directory, video, file)
                            img = np.array(cv2.imread(path)).astype(np.float64)
                            img = cv2.resize(img,(112,112))
                            self.training_data.append([torch.Tensor(img), self.labels_iterated[label]])
                        except Exception as e:
                            print(e)
                            pass
        



    def Categories(self, category,frame_location):
        # Creating a list for 100, 200, 500, 1000 and 2000 classes with the highest amount of videos.
        count = 0
        labels_100 = []
        labels_200 = []
        labels_500 = []
        labels_1000 = []
        labels_2000 = []

        for key in self.count_dictionary:
            if count < 100:
                labels_100.append(key)
            if count < 200:
                labels_200.append(key)
            if count < 500:
                labels_500.append(key)
            if count < 1000:
                labels_1000.append(key)
            labels_2000.append(key)
            count += 1

        if category == "labels_100":
            self.labels_x = labels_100
        elif category == "labels_200":
            self.labels_x = labels_200
        elif category == "labels_500":
            self.labels_x = labels_500
        elif category == "labels_1000":
            self.labels_x = labels_1000
        elif category == "labels_2000":
            self.labels_x = labels_2000

        # Assigning an integer to each class.
        counter = 0
        for label in self.labels_x:
            self.labels_iterated[label] = counter
            counter += 1

        # Creating a dictionary where given a video whe can fint its class.
        for k, v in self.video_id_dictionary.items():
            for video in v:
                self.inv_video_id_dictionary[video] = k
        self.make_training_data(self.labels_x, frame_location)