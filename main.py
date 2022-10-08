from Detector import Detector


cases = {
	"instance_segmentation" : 1, 
	"object_detection" : 1,
	"keypoint_detection" : 1
}

if __name__ == "__main__":


	path = input("Video Path: ")


	try:
		open(path, "r")
	except (OSError, IOError) as e:
		print("ERROR: Invalid File")
		exit(1)

	detection = ((input("Detection Type: ")).strip()).lower()

	if detection not in cases:
		print("ERROR: Detection Type invalid!")
		exit(1)

	detector = Detector(detection)

	
	if path.split('.')[-1] == 'mp4':


		detector.onVideo(path)
		exit(0)

	detector.onImage(path)

