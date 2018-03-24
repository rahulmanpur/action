import cv2
import sys
import os

def vidtoimg(path, name):
	#path should end with /
	video = cv2.VideoCapture(path+name)

	frame = video.read()
	frames = 0
	while frame[0]:
	    # get frame by frame
	    frames += 1
	    frame = video.read()
	    cv2.imwrite(name + '_frame{}'.format(frames) + '.jpg',frame[1])

	print 'Created {} frames from {}'.format(frames, name)
	video.release()
	 
	return

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'error'
		exit()
	path = sys.argv[1]
	for filename in os.listdir(path):
		if (filename.endswith('avi')):
			vidtoimg(path, name = filename)