from torch.utils.data.dataset import Dataset
import json
import cv2
import os

class MyCustomDataset(Dataset):
    def __init__(self, json_file_path = "/Users/mjo/Desktop/WLASL/WLASL_v0.3.json", category,
     video_file_path = "/Users/mjo/Desktop/WLASL/WLASL2000", frame_location = "/Users/mjo/Desktop/WLASL/Processed_data"):

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

		# Creating a list for 100, 200, 500, 1000 and 2000 classes with the highest amount of videos.
		count = 0
		self.labels_100 = []
		self.labels_200 = []
		self.labels_500 = []
		self.labels_1000 = []
		self.labels_2000 = []

		for key in count_dictionary:
			if count<100:
				self.labels_100.append(key)
			if count<200:
				self.labels_200.append(key)
			if count<500:
				self.labels_500.append(key)
			if count<1000:
				self.labels_1000.append(key)

			self.labels_2000.append(key)

			count += 1

		# Creating folders where each video has a folder with all of its frames inside.
		exctract_frames(self.labels_100)
		exctract_frames(self.labels_200)
		exctract_frames(self.labels_500)
		exctract_frames(self.labels_1000)
		exctract_frames(self.labels_2000)

		# Assigning an integer to each class.
		self.labels_iterated = {}
		counter = 0
		for label in labels_100:
		    self.labels_iterated[label] = counter
		    counter += 1

		# Creating a dictionary where given a video whe can fint its class.
		self.inv_video_id_dictionary = {}
		for k, v in video_id_dictionary.items(): 
		    for video in v:
		        self.inv_video_id_dictionary[video] = k

		# Defining training_data dependant on choice.
		self.training_data = []
		if category == "labels_100":
			make_training_data(labels_100)
		elif category == "labels_200":
			make_training_data(labels_200)
		elif category == "labels_500":
			make_training_data(labels_500)
		elif category == "labels_1000":
			make_training_data(labels_1000)
		elif category == "labels_2000":
			make_training_data(labels_2000)
		        

    def exctract_frames(self, labels_x, frame_location, video_file_path):
		video_count = 0
		num_classes = len(labels_x)

		if not os.path.exists("{}".format(frame_location)):
					os.mkdir("{}".format(frame_location)) 

		if not os.path.exists("{}/{}".format(frame_location, num_classes)):
					os.mkdir("{}/{}".format(frame_location, num_classes)) 

		for label in labels_x:
			for video in self.video_id_dictionary[label]:
				video_capture = cv2.VideoCapture(os.path.join(video_file_path, video + ".mp4"))
				success,image = video_capture.read()
				count = 0

				if video_count % 10 == 0:
					print(video_count)

				if not os.path.exists("{}/{}/{}".format(frame_location, num_classes, video)):
					os.mkdir("{}/{}/{}".format(frame_location, num_classes, video)) 
				while success:
					cv2.imwrite("{}/{}/{}/{}".format(frame_location, num_classes, video,"frame%d.jpg" % count), image)
					success,image = video_capture.read()
					count += 1
				video_count += 1

	def total_videos(self, labels_x):
		sum_count = 0
		for label in labels_x:
			for video in video_id_dictionary[label]:
				sum_count += 1
		return sum_count


	def make_training_data(self, labels_x):
		data_directory = ("{}/{}/{}/{}".format(frame_location, len(labels_x)))
	    num_labels = len(labels_x)
	    for label in (labels_x):
	        for video in self.video_id_dictionary[label]:
	            path = os.path.join(data_directory, video)
	            for file in (os.listdir(path)):
	                if "jpg" in file:
	                    try:
	                        path = os.path.join(data_directory,video, file)
	                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	                        self.training_data.append([np.array(img),self.labels_iterated[label]])
	                    except Exception as e:
	                        print(e)
	                        pass

	def __getitem__(self, index):
        return training_data(index)

    def __len__(self, labels_x):
        return total_videos(labels_x)