# !/usr/bin/env python
import argparse
import datetime
import imutils
import time
import cv2
from multiprocessing import Process
from multiprocessing import Queue

def motion_detector(detectqueue, showqueue):
	firstFrame = None
	while True:
		image = detectqueue.get()
		if image is None : break
		frame = image
		if frame is None:
			break
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
		if firstFrame is None:
			firstFrame = gray
			continue
		frameDelta = cv2.absdiff(firstFrame, gray)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		upper_threshold_cnt =[]
		for c in cnts:
			if cv2.contourArea(c) < 500:
				continue
			(x, y, w, h) = cv2.boundingRect(c)
			upper_threshold_cnt.append([x,y,w,h])
		showqueue.put([upper_threshold_cnt, frame])



def image_display(showqueue):
	while True:
		detectedData = showqueue.get()
		if detectedData is None : break
		for c in detectedData[0]:
			detectedData[1][c[1]:c[1]+c[3],c[0]:c[0]+c[2]] = cv2.blur(detectedData[1][c[1]:c[1]+c[3],c[0]:c[0]+c[2]],(23,23))
			# cv2.rectangle(detectedData[1], (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (0, 255, 0), 2)

		cv2.putText(detectedData[1], datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
					(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		cv2.imshow('image_display', detectedData[1])
		cv2.waitKey(10)
		continue


def video_streamer(detectqueue, filename):
	vs = cv2.VideoCapture(filename)
	while True:
		flag, image = vs.read()
		if flag:
			detectqueue.put(image)
			time.sleep(0.030)
			continue
		else:
			break
	vs.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", default="1.mp4", help="path to the video file")
	args = vars(ap.parse_args())
	detectqueue = Queue()
	showqueue = Queue()
	streamer = Process(target=video_streamer, args=(detectqueue, args["video"],))
	detector = Process(target=motion_detector, args=(detectqueue, showqueue,))
	viewer = Process(target=image_display, args=(showqueue,))
	streamer.start()
	detector.start()
	viewer.start()

	streamer.join()
	streamer.terminate()
	detector.terminate()
	viewer.terminate()

