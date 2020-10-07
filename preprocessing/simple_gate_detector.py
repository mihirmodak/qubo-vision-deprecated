import cv2
import numpy as np
import argparse
import math
import os
import shutil as sh
from glob import glob

def color_stretching(img):
	result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	avg_a = np.average(result[:,:,1])
	avg_b = np.average(result[:,:,2])
	result[:,:,1] = result[:,:,1] - ((avg_a-128)*(result[:,:,0]/255.0)*1.1)
	result[:,:,2] = result[:,:,2] - ((avg_b-128)*(result[:,:,0]/255.0)*1.1)
	result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
	return result

def find_gate(img, output, frame_counter, CURR_DIR_CUSTOM):
	img = np.int16(img)
	grayscaleImg = np.uint8((255 + img[:,:,2] - ((img[:,:,1] + img[:,:,0])/2))/2)
	cv2.imshow("grayscale", grayscaleImg)
	grayscaleImg = cv2.blur(grayscaleImg, (5,5))
	grayscaleImg = cv2.Canny(grayscaleImg, 10, 25)
	cv2.imshow("canny", grayscaleImg)
	lines = cv2.HoughLinesP(grayscaleImg, rho=1, theta=np.pi/180, threshold = 10, minLineLength=30, maxLineGap = 10)
	#print(lines)
	img = cv2.cvtColor(grayscaleImg.copy(), cv2.COLOR_GRAY2BGR)
	draw_lines(img, lines, (0, 0, 255))
	cv2.imshow("lines", img)
	majorLines = find_major_lines(lines, 60, 0.2)
	#print(majorLines)
	majorHorizontalLines = [[x.getLine()] for x in majorLines[0]]
	majorVerticalLines = [[x.getLine()] for x in majorLines[1]]

	if args["output"]:
		with open(output, 'a') as f:
			f.write('\nFrame' + str(frame_counter))
			f.write("\nMajor Horizontal Lines: " + str(majorHorizontalLines))
			f.write("\nMajor Vertical Lines: " + str(majorVerticalLines))

	line1 = majorVerticalLines[0][0]
	line2 = majorVerticalLines[1][0]
	x_center = (( (line1[0]/2 + line1[2]/2) + (line2[0]/2 + line2[2]/2) ) / 2) / 1080
	y_center = (( (line1[1]/2 + line1[3]/2) + (line2[1]/2 + line2[3]/2) ) / 2) / 720
	width = (abs( (line1[0]/2 + line1[2]/2) - (line2[0]/2 + line2[2]/2) ) / 2) / 1080
	height = (abs( (line1[1] - line1[3]) + (line2[1] - line2[3]) ) / 2) / 720

	with open(CURR_DIR_CUSTOM + "/source_labels/frame"+str(frame_counter).zfill(4) + ".txt", 'w+') as f:
		f.write(f"0 {x_center} {y_center} {width} {height}")

	print(majorVerticalLines)
	majorlineImg = cv2.cvtColor(grayscaleImg, cv2.COLOR_GRAY2BGR)
	draw_lines(majorlineImg, majorHorizontalLines, (0,255,0))
	draw_lines(majorlineImg, majorVerticalLines, (0,0,255))
	cv2.imshow("major lines", majorlineImg)
	return None, None

def find_major_lines(lines, coordThresh, slopeThresh):
	majorHorizontalLines = []
	majorVerticalLines = []
	verticalLines = []
	horizontalLines = []
	for line in lines:
		if abs(line[0][0] - line[0][2]) < abs(line[0][1] - line[0][3]) and abs(line[0][0] - line[0][2])/abs(line[0][1] - line[0][3]) < 2*slopeThresh:
			verticalLines.append(line)
		elif abs(line[0][1] - line[0][3])/abs(line[0][0] - line[0][2]) < 2*slopeThresh:

			horizontalLines.append(line)
	print(len(verticalLines))
	for i in range(len(verticalLines)):
		addedLine = False
		for majorLine in majorVerticalLines:
			if majorLine.checkAddLine(verticalLines[i], coordThresh, slopeThresh):
				addedLine = True
				print("valid")
				break	
		if addedLine == False:
                        majorVerticalLines.append(MajorLine(verticalLines[i], False))

	for i in range(len(horizontalLines)):
		addedLine = False
		for majorLine in majorHorizontalLines:
			if majorLine.checkAddLine(horizontalLines[i], coordThresh, slopeThresh):
				addedLine = True
				break
		if addedLine == False:
			majorHorizontalLines.append(MajorLine(horizontalLines[i], True)) 
	return majorHorizontalLines, majorVerticalLines

class MajorLine:
	avgSlope = 0.0 #x/y or y/x depends on horizontal
	avgPerpCoord = 0.0 #x or y depends on if horizontal
	numLines = 0
	minParrCoord = 0.0
	maxParrCoord = 0.0
	horizontal = False

	def __init__(self, line, horizontal):
		(x1, y1, x2, y2) = line[0]
		self.numLines += 1
		self.horizontal = horizontal
		if horizontal:
			self.minParrCoord = min(x1, x2)
			self.maxParrCoord = max(x1, x2)
			self.avgPerpCoord = (y2+y1)/2.0
			self.avgSlope = (y2-y1)/(float)(x2-x1)
		else:
			self.minParrCoord = min(y1, y2)
			self.maxParrCoord = max(y1, y2)
			self.avgPerpCoord = (x2+x1)/2.0
			self.avgSlope = (x2-x1)/(float)(y2-y1)
	
	def getLine(self):
		if self.horizontal:
			dif = self.avgSlope * ((self.maxParrCoord - self.minParrCoord)/2.0)
			return (self.minParrCoord, self.avgPerpCoord - dif, self.maxParrCoord, self.avgPerpCoord + dif)
		else:
			dif = self.avgSlope * ((self.maxParrCoord - self.minParrCoord)/2.0)
			return (self.avgPerpCoord - dif, self.minParrCoord, self.avgPerpCoord + dif, self.maxParrCoord)
		return((int)(x1),(int)(y1),(int)(x2),(int)(y2))
	
	def checkAddLine(self, line, coordThresh, slopeThresh):
		(x1, y1, x2, y2) = line[0]
		slope = 0.0 	
		p1 = 0
		p2 = 0
		if self.horizontal:
			#print("horizontal: " + str(line[0]))
			slope = (y2-y1)/(float)(x2-x1)
			perpCoord = (y2+y1)/2.0
			p1 = x1
			p2 = x2
		else:
			slope = (x2-x1)/(float)(y2-y1)
			perpCoord = (x2+x1)/2.0
			p1 = y1
			p2 = y2
		if(abs(slope - self.avgSlope) < slopeThresh and abs(perpCoord - self.avgPerpCoord) < coordThresh):
			if self.horizontal:
				print ("horizontal " + str(slope))
			self.avgSlope = ((self.numLines*self.avgSlope) + slope)/(float)(self.numLines + 1)
			self.avgPerpCoord = ((self.numLines*self.avgPerpCoord) + perpCoord)/(float)(self.numLines + 1)
			self.numLines += 1
			self.minParrCoord = min(self.minParrCoord, p1, p2)
			self.maxParrCoord = max(self.maxParrCoord, p1, p2)
			return True
		return False

def getGateFromLines(horizontalLines, verticalLines, thresh):
	for hLine in horizontalLines:
		p1 = (hLine[0], hLine[1])
		p2 = (hLine[2], hLine[3])
		p1Line = None
		p2Line = None
		p1Match = False
		p2Match = False
		foundGate = False
		for vLine in verticalLines:
			topVPoint = None
			if vLine[1] > vLine[3]:
				topVPoint = (vLine[0], vLine[1])
			else:
				topVPoint = (vLine[2], vLine[3])
			if not p1Match and pointDist(topVPoint, p1) < thresh:
				p1Match = True
				p1Line = vLine
			elif not p2Match and pointDist(topVPoint, p2) < thresh:
				p2Match = True
				p2Line = vLine
				foundGate = True
				break
		if foundGate:
			return True, (hLine, p1Line, p2Line)
	return False, (None, None, None)
		
				
def pointDist(p1, p2):
	return math.sqrt((p1[0] - p2[0])^2 + (p1[1] - p2[1])^2)

def draw_lines(img, lines, color):
	if lines is None:
		return
	for line in lines:
		[x1, y1, x2, y2] = line[0]
		x1 = (int)(x1)
		x2 = (int)(x2)
		y1 = (int)(y1)
		y2 = (int)(y2)
		cv2.line(img, (x1, y1), (x2, y2), color, 2)

def remove_recursive(input_path):
	"""
	Recursively removes the given folder and all the files within it
	:param input_path: The path of the file/folder to remove
	:return: None
	"""
	input_path = os.getcwd().join(input_path.split('./'))

	if os.path.isfile(input_path):
		os.remove(input_path)
	else:
		if not input_path[-1] == '/':
			input_path += '/'
		for dir_name in os.listdir(input_path):
			remove_recursive(input_path + "/" + dir_name)
			os.rmdir(input_path + "/" + dir_name + "/")
		os.rmdir(input_path)


def create_dir_system(dir_names, parent_path=os.getenv('CURR_DIR_CUSTOM')):
	"""
	:param dir_names: list of folders to create in the parent dir, e.g. [test, train]
	:param parent_path: path to the folder containing the dir_names e.g. ./source/images ...
			i.e. source/images/test and source/images/train as the final directories
	:return: None
	"""
	if isinstance(dir_names, str):
		dir_names = [dir_names]
	assert isinstance(dir_names, list)

	parent_path = os.path.abspath(parent_path)

	for name in dir_names:
		try:
			os.makedirs(parent_path+"/"+name)
		except FileExistsError:
			try:
				old_files = glob(parent_path+"/"+name+"_OLD/*", recursive=True)
				for filename in old_files:
					remove_recursive(filename)
				remove_recursive(parent_path+"/"+name+"_OLD")
			except FileNotFoundError:
				pass
			os.rename(parent_path+"/"+name, parent_path+"/"+name+"_OLD")
			os.makedirs(parent_path+"/"+name)

if __name__ == "__main__":
	ap=argparse.ArgumentParser()
	ap.add_argument("-v", "--video", required=False, help="path to input video")
	ap.add_argument("-o", "--output", required=False, help="path to folder where the output is saved")
	ap.add_argument("-d", "--cwd", required=False, help="path of current working directory (where yolov5 git repo was cloned); default is the directory where this preprocessing.sh script is run form")
	args = vars(ap.parse_args())

	if args["cwd"]:
		CURR_DIR_CUSTOM = os.path.abspath(args["cwd"] if args["cwd"] is not "" or args["cwd"] is not None else AttributeError)
		os.putenv("CURR_DIR_CUSTOM", CURR_DIR_CUSTOM)
	elif os.getenv("CURR_DIR_CUSTOM"):
		CURR_DIR_CUSTOM=os.path.abspath(os.getenv("CURR_DIR_CUSTOM"))
	else:
		print("Current Directory path could not be automatically detected. Using the preprocessing directory.")
		CURR_DIR_CUSTOM = os.path.abspath(os.getcwd())

	cap = None
	if args["video"]:
		cap = cv2.VideoCapture(args["video"])
	else:
		cap = cv2.VideoCapture(0)	
	cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
	frame_counter = 0
	if not args["output"]:
		args["output"] = CURR_DIR_CUSTOM + "/hough_output_logs.txt"
	with open(args["output"], 'w+') as f:
		pass

	create_dir_system('source')
	create_dir_system("source_labels")

	while True:
		frame_counter += 1
		#If the last frame is reached, reset the capture and the frame_counter
		if args["video"] and frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
			frame_counter = 0
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		ret, frame = cap.read()
		frame = cv2.resize(frame, (1080, 720))
		img = color_stretching(frame)
		cv2.imshow("original", frame)
		cv2.imshow("modified", img)

		cv2.imwrite(CURR_DIR_CUSTOM + "/source/frame" + str(frame_counter).zfill(4) + ".jpg", frame)
		try:
			gateFound, gate = find_gate(img, args["output"], frame_counter, CURR_DIR_CUSTOM) #currently this only works for 1 gate and will find the first valid gate
		except:
			with open(CURR_DIR_CUSTOM + "/source_labels/frame" + str(frame_counter).zfill(4) + ".txt", 'w+') as f:
				pass

		if cv2.waitKey(1) == ord('q'):
			break;
		
	cap.release()
	cv2.destroyAllWindows()
