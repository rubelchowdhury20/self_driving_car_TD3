# standard library imports
import math
import random

# third party imports
import numpy as np
from PIL import Image as PILImage, ImageDraw

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.config import Config

# local imports
from network import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# global variable
brain = TD3(1)

last_angle = 0
last_reward = 0
total_reward = 0
done_bool = False

total_steps = 0

# Initializing the map
first_update = True
def init():
	global sand
	global sand_img
	global car_img
	global goal_x
	global goal_y
	global first_update
	global flas

	sand_img = PILImage.open("./images/MASK1.png").convert('L')
	sand = np.asarray(sand_img)/255
	sand = np.transpose(sand)

	car_img = PILImage.open("./images/car.png").convert('RGBA')
	car_img = car_img.resize((20, 10))

	goal_x = 1420
	goal_y = 38
	first_update = False
	global swap
	swap = 0


# Initializing the last distance
last_distance = 0

count = 0
def get_input_image(x, y, angle):
	y = largeur - y
	global count
	base = 15
	crop_size = 20
	theta = (angle) * math.pi / 180

	x1, y1 = x + (4/3) * base * math.cos(theta), y - (4/3) * base * math.sin(theta)
	x2, y2 = x + (2/3) * base * math.cos(theta + (135*math.pi/180)), y - (2/3) * base * math.sin(theta + (135*math.pi/180))
	x3, y3 = x + (2/3) * base * math.cos(theta + (225*math.pi/180)), y - (2/3) * base * math.sin(theta + (225*math.pi/180))

	# x1, y1 = y + (4/3) * base * math.cos(theta), x - (4/3) * base * math.sin(theta)
	# x2, y2 = y + (2/3) * base * math.cos(theta + (135*math.pi/180)), x - (2/3) * base * math.sin(theta + (135*math.pi/180))
	# x3, y3 = y + (2/3) * base * math.cos(theta + (225*math.pi/180)), x - (2/3) * base * math.sin(theta + (225*math.pi/180))


	sand_img_copy = sand_img.copy()
	draw = ImageDraw.Draw(sand_img_copy)
	draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill = (0))
	# draw.polygon([(x, y), (x+3, y+3), (x+5, y+5)], fill= (0))

	img_patch = sand_img_copy.crop((x-crop_size, y-crop_size, x+crop_size, y+crop_size))
	# img_patch.save("./folder/image_" + str(count) + ".png")
	# count = count + 1
	img_patch = np.asarray(img_patch)/255
	img_patch = np.transpose(img_patch)
	img_patch = np.expand_dims(img_patch, axis=0)
	return img_patch


# def get_input_image(x, y, angle):
# 	car_rotated = car_img.rotate(angle, expand = 1)
# 	car_rotated = car_rotated.convert("L")
# 	sand_img_copy = sand_img
# 	sand_img_copy.paste(car_rotated, (x, y), car_rotated)
# 	img_patch = sand_img_copy.crop((y-20, x-20, y+20, x+20))
# 	img_patch = np.asarray(img_patch)/255
# 	img_patch = np.expand_dims(img_patch, axis=0)
# 	return img_patch



class Car(Widget):
	angle = NumericProperty(0)
	rotation = NumericProperty(0)
	velocity_x = NumericProperty(0)
	velocity_y = NumericProperty(0)
	velocity = ReferenceListProperty(velocity_x, velocity_y)

	def move(self, rotation):
		self.pos = Vector(*self.velocity) + self.pos
		self.rotation = rotation
		self.angle = self.angle - self.rotation

# defining the game class
class Game(Widget):
	car = ObjectProperty(None)

	def serve_car(self):
		self.car.center = self.center
		self.car.velocity = Vector(6, 0)
		self.car.angle = self.car.angle + 180

	def update(self, dt):

		global brain
		global last_reward
		global total_reward
		global done_bool

		global brain
		global scores
		global last_angle
		global last_reward
		global last_distance
		global goal_x
		global goal_y
		global longueur
		global largeur
		global swap
		global total_steps


		longueur = self.width
		largeur = self.height
		
		if first_update:
			init()

		distance = np.sqrt((self.car.x - goal_x)**2 + (largeur - self.car.y - goal_y)**2)
		input_image = get_input_image(int(self.car.x), int(self.car.y), self.car.angle)
		action = brain.update(last_reward, input_image, done_bool)
		# print(action)

		done_bool = False

		# now moving the car based on the action value which will act as rotation and velocity
		if(total_steps < 10000):
			self.car.move(action * 10)
		else:
			self.car.move(action * 50)
		# if(abs(action) < 0.5):
		# 	self.car.move(action * 100)
		# else:
		# 	self.car.move(action)

		# if sand[int(self.car.x),largeur - int(self.car.y)] == 1:
		# 	self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
		# 	# print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y))
		# 	last_reward = -1
		# else: # otherwise
		# 	self.car.velocity = Vector(2, 0).rotate(self.car.angle)
		# 	last_reward = -0.2
		# 	# print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y))
		# 	if distance < last_distance:
		# 		last_reward = 0.1

		if distance < last_distance:
			last_reward = 0.1
			self.car.velocity = Vector(1, 0).rotate(self.car.angle)

		else:
			last_reward = -0.2
			self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)



		if self.car.x < 10:
			self.car.x = 10
			last_reward = -1
		if self.car.x > self.width - 10:
			self.car.x = self.width - 10
			last_reward = -1
		if self.car.y < 10:
			self.car.y = 10
			last_reward = -1
		if self.car.y > self.height - 10:
			self.car.y = self.height - 10
			last_reward = -1

		total_reward += last_reward

		if distance < 25 or total_reward < -5000:
			print("total_steps")
			print(total_steps)
			if swap == 1:
				done_bool = True
				self.car.x = int(np.random.randint(25, self.width-25, 1)[0])
				self.car.y = int(np.random.randint(25, self.height-25, 1)[0])
				self.car.angle = random.randint(0, 360)
				total_reward = 0

				goal_x = 1420
				goal_y = 38
				swap = 0
			else:
				done_bool = True
				self.car.x = int(np.random.randint(25, self.width-25, 1)[0])
				self.car.y = int(np.random.randint(25, self.height-25, 1)[0])
				self.car.angle = random.randint(0, 360)
				total_reward = 0

				goal_x = 9
				goal_y = 575
				swap = 1
			print("one episode is done")
		
		last_distance = distance

		total_steps += 1

class TeslaApp(App):

	def build(self):
		parent = Game()
		parent.serve_car()
		Clock.schedule_interval(parent.update, 1.0/60.0)
		return parent



# Running the whole thing
if __name__ == '__main__':
	TeslaApp().run()
