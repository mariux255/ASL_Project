from torch.utils.data.dataset import Dataset
import torch
import json
import cv2
import os
import numpy as np
from tqdm import tqdm
import natsort
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class MyCustomDataset(Dataset):
    def __init__(self, category, json_file_path="/scratch/s174411/WLASL_v0.3.json", frame_location="/scratch/s174411/Processed_data/"):
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
        #buffer, label = self.training_data[index]
        #images = []

        # for i in range(64):
        #     path = buffer[i]
        #     #print("PATH: ", path)
        #     img = np.array(cv2.imread(path))
        #     img = cv2.resize(img, (224, 224))
        #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #     #img = img/255
        #     images.append(img)
        # images = np.array(images)
        # images = self.to_tensor(images)
        #temp = buffer.transpose(1,2,3,0)
        #img = temp[0]
        #imgplot = plt.imshow(img)
        #plt.show()
        path, label = self.training_data[index]

        img = np.array(cv2.imread(path))
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        #imgplot = plt.imshow(img)
        #plt.show()

        return [img, label]


    def __len__(self):
        # total = 0
        # folders = ([name for name in os.listdir(os.path.join(self.frame_location,"{}/".format(len(self.labels_x))))
        #             if os.path.isdir(os.path.join(self.frame_location,"{}/".format(len(self.labels_x)),name))])
        # for folder in folders:
        #     contents = os.listdir(os.path.join(self.frame_location,"{}/".format(len(self.labels_x)),folder))
        #     total += len(contents)
        total = self.total_frames
        #print(total)
        return total


    def total_videos(self):


        sum_count = 0
        for label in self.labels_x:
            for video in self.video_id_dictionary[label]:
                sum_count += 1
        return sum_count


    def make_training_data(self, labels_x,frame_location,buffer_size = 64,image_height = 224, image_width = 224):
        self.total_frames = 0
        data_directory = ("{}/{}".format(frame_location, len(labels_x)))
        num_labels = len(labels_x)
        ignored_videos = ['11305', '38536', '28114', '63666', '26741', '63210', '14675', '42971', '06371',
                         '32947', '58363', '10193', '66016', '56699', '05303', '21071', '39630', '59204',
                          '08370', '70250', '65905', '38124', '48038', '58792', '05800', '18295', '28314',
                           '37091', '42253', '55056', '61979', '65115', '10397', '17934', '22636', '32200',
                            '70324', '47498', '55299', '61144', '04173', '09899', '67565', '20383', '25624',
                             '33640', '44058', '51821', '55286', '60856', '02566', '06949', '12593', '20159',
                              '25431', '67839', '38070', '45913', '51691', '58707', '67347', '06316', '11059', 
                              '16432', '20327', '27456', '32866', '40636', '44629', '67150', '53243', '56466',
                               '60380', '02611', '09522', '65454', '20623', '70248', '30754', '36413', '43383',
                                '48159', '51956', '56787', '62519', '01472', '66040', '09620', '14734', '19183', 
                                '24014', '65945', '35770', '41653', '44777', '49342', '53262', '57598', '62345', 
                                '03092', '07513', '11930', '14635', '17964', '22015', '27754', '32424', '36906', 
                                '41262', '44551', '47096', '52114', '56356', '61138']

        counterr = 0
        #countel = 0
        video_list = []
        for label in tqdm(labels_x):
            for video in self.video_id_dictionary[label]:
                #countel += 1
                #if countel % 184 == 0:
                    #video_list.append(video)
                if video in ignored_videos:
                    counterr += 1
                    #print(counterr)
                    #continue
                buffer = []
                path = os.path.join(data_directory, video)
                number_of_frames = len([file for file in os.listdir(path) if "jpg" in file])
                try:
                    save_frequency = np.floor(number_of_frames/buffer_size)
                    if save_frequency == 0:
                        save_frequency = 1
                    if number_of_frames % save_frequency == 0:
                        save_start = 0
                    else:
                        save_start = save_frequency
                    if number_of_frames % save_frequency  != 0 or number_of_frames / save_frequency > buffer_size:
                        save_start = (np.ceil(number_of_frames/save_frequency)-buffer_size)*save_frequency
                except Exception as e:
                    print(e)
                to_repeat = False
                if number_of_frames < buffer_size:
                    repeat = buffer_size - number_of_frames
                    to_repeat = True
                    save_frequency = 1
                    save_start = 1

                counter = 1
                #buffer = np.empty((buffer_size, image_height, image_width, 3), np.dtype('float32'))
                #buffer = []
                index = 0
                fff = natsort.natsorted(os.listdir(path),reverse=False)

                for file in fff:
                    if (counter % save_frequency == 0 and counter > save_start) or (counter % save_frequency == 0 and counter >= save_start and number_of_frames % save_frequency != 0):
                        if "jpg" in file:
                            try:
                                path = os.path.join(data_directory, video, file)
                                #img = np.array(cv2.imread(path))
                                #img = cv2.resize(img, (image_height, image_width))
                                #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                #img = img/255
                                self.training_data.append([path,self.labels_iterated[label]])
                                self.total_frames += 1
                                #print(path)
                                #buffer[index] = img
                                index += 1
                                if counter == number_of_frames and to_repeat == True:
                                    for i in range(repeat+1):
                                        #buffer[index] = img
                                        #buffer.append(img)
                                        #buffer.append(path)
                                        self.training_data.append([path,self.labels_iterated[label]])
                                        self.total_frames += 1
                                        index += 1
                            except Exception as e:
                                print(e)
                                pass
                    counter += 1
                # if len(buffer) != buffer_size:
                #     print("Buffer is not of the right size")
                #     print("video:",video ,len(buffer))
                #     print(f"Video:", video, "Number of frames:", number_of_frames, "Save_frequency:", save_frequency, "Save start:", save_start)
                #self.training_data.append([(np.array(buffer)),self.labels_iterated[label]])





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
