import json
import cv2
import os

json_file_path = "/Users/mjo/Desktop/WLASL/WLASL_v0.3.json"
video_file_path = "/Users/mjo/Desktop/WLASL/WLASL2000"
frame_location = "/Users/mjo/Desktop/WLASL/Processed_data"
file_extension = ".mp4"


def exctract_json_data():
	with open(json_file_path, "r") as read_file:
	    data = json.load(read_file)

	count_dictionary = {}
	video_id_dictionary = {}

	for instance in data:
		current_label = instance['gloss']
		count_dictionary[current_label] = 0
		video_id_dictionary[current_label] = []
		inner_array = instance['instances']
		for video in inner_array:
			count_dictionary[current_label] += 1
			video_id_dictionary[current_label].append(video['video_id'])

	return count_dictionary, video_id_dictionary

def define_categories(count_dictionary):
	count = 0
	labels_100 = []
	labels_200 = []
	labels_500 = []
	labels_1000 = []
	labels_2000 = []

	for key in count_dictionary:
		if count<100:
			labels_100.append(key)
		if count<200:
			labels_200.append(key)
		if count<500:
			labels_500.append(key)
		if count<1000:
			labels_1000.append(key)

		labels_2000.append(key)

		count += 1

	return labels_100, labels_200, labels_500, labels_1000, labels_2000

def exctract_frames(labels_x, video_id_dictionary):
	video_count = 0
	num_classes = len(labels_x)

	if not os.path.exists("{}".format(frame_location)):
				os.mkdir("{}".format(frame_location)) 

	if not os.path.exists("{}/{}".format(frame_location, num_classes)):
				os.mkdir("{}/{}".format(frame_location, num_classes)) 

	for label in labels_x:
		for video in video_id_dictionary[label]:
			video_capture = cv2.VideoCapture(os.path.join(video_file_path, video + file_extension))
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


def total_videos(video_id_dictionary, labels_x):
	sum_count = 0
	for label in labels_x:
		for video in video_id_dictionary[label]:
			sum_count += 1
	return sum_count


def main():
	count_dictionary = {}
	video_id_dictionary = {}

	labels_100 = []
	labels_200 = []
	labels_500 = []
	labels_1000 = []
	labels_2000 = []
	count_dictionary, video_id_dictionary = exctract_json_data()
	labels_100, labels_200, labels_500, labels_1000, labels_2000 = define_categories(count_dictionary)
	print(total_videos(video_id_dictionary, labels_100))
	exctract_frames(labels_100, video_id_dictionary)

