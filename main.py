# standard library imports


# third party imports
import numpy as np
from PIL import Image as PILImage

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.config import Config


from torchvision import transforms


# local imports
from network import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '768')
Config.set('graphics', 'height', '670')

# global variable
brain = TD3(5)

last_angle = 0
last_reward = 0
total_reward = 0
done_bool = False

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

    sand_img = PILImage.open("./images/map3.png").convert('L')
    sand = np.asarray(sand_img)/255
    car_img = PILImage.open("./images/car.png").convert('RGBA')
    car_img = car_img.resize((20, 10))

    goal_x = 298
    goal_y = 67
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

def get_input_image(x, y, angle):
    car_rotated = car_img.rotate(angle, expand = 1)
    car_rotated = car_rotated.convert("L")
    sand_img_copy = sand_img
    sand_img_copy.paste(car_rotated, (x, y), car_rotated)
    img_patch = sand_img_copy.crop((y-20, x-20, y+20, x+20))
    img_patch = np.asarray(img_patch)/255
    img_patch = np.expand_dims(img_patch, axis=0)
    return img_patch



class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation


# defining the game class
class Game(Widget):
    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

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

        longueur = self.width
        largeur = self.height
        
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        

        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        input_image = get_input_image(int(self.car.x), int(self.car.y), last_angle)
        action = brain.update(last_reward, input_image, orientation, done_bool)

        done_bool = False

        # now moving the car based on the action value which will act as rotation and velocity
        self.car.move(action)


        # updating velocity as well as the reward of the 
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            # print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y))
            
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.2
            # print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y))
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 25:
            self.car.x = 25
            last_reward = -1
        if self.car.x > self.width - 25:
            self.car.x = self.width - 25
            last_reward = -1
        if self.car.y < 25:
            self.car.y = 25
            last_reward = -1
        if self.car.y > self.height - 25:
            self.car.y = self.height - 25
            last_reward = -1

        total_reward += last_reward

        if distance < 25 or total_reward<-1000:
            print("one episode is done")
            done_bool = True
            self.car.x = int(np.random.randint(25, self.width-25, 1)[0])
            self.car.y = int(np.random.randint(25, self.height-25, 1)[0])
            total_reward = 0
        
        last_distance = distance

class TeslaApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent



# Running the whole thing
if __name__ == '__main__':
    TeslaApp().run()
