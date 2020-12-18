import numpy as np
import cv2
import carla
import argparse
import logging
import time
import math
import random
from sklearn.neighbors import KDTree
import sys
#IK this is bad, fix file path stuff later :(
sys.path.append("/scratch/cluster/stephane/Carla_0.9.10/PythonAPI/carla/agents/navigation")
from global_route_planner import GlobalRoutePlanner
from global_route_planner_dao import GlobalRoutePlannerDAO


SHOW_PREVIEW = False
#maybe scale these down
height = 80
width = 80
fov = 10
max_ep_length = 60
FPS = 60

class CarEnv:
    rgb_cam = None

    def __init__(self,host,world_port):
        self.client = carla.Client(host,world_port)
        self.client.set_timeout(10.0)
        #self.world = self.client.get_world()
        self.world = self.client.load_world('Town01')

        self.blueprint_lib = self.world.get_blueprint_library()
        self.car_agent_model = self.blueprint_lib.filter("model3")[0]

        self.command2onehot = {"LEFT":[0,0,0,0,0,1], "RIGHT":[0,0,0,0,1,0], "STRAIGHT":[0,0,0,1,0,0],"LANEFOLLOW":[0,0,1,0,0,0],"CHANGELANELEFT":[0,1,0,0,0,0],"CHANGELANERIGHT":[1,0,0,0,0,0]}

    def process_img(self,img):
        img = np.array(img.raw_data).reshape(height,width,4)
        rgb = img[:,:,:3]
        #norm = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.rgb_cam = rgb

    def reset(self):
        self.collisions = []
        self.actors = []
        #spawn car randomly
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.car_agent = self.world.spawn_actor(self.car_agent_model,self.spawn_point)
        self.actors.append(self.car_agent)
        #get camera
        self.rgb_cam = self.blueprint_lib.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x",f"{width}")
        self.rgb_cam.set_attribute("image_size_y",f"{height}")
        self.rgb_cam.set_attribute("fov",f"{fov}")
        sensor_pos = carla.Transform(carla.Location(x=2.5,z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, sensor_pos, attach_to=self.car_agent)
        self.actors.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        #workaround to get things started sooner
        self.car_agent.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)
        #get collision sensor
        col_sensor = self.blueprint_lib.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_sensor,sensor_pos,attach_to=self.car_agent)
        self.actors.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.collisions.append(event))

        while self.rgb_cam is None:
            print ("camera is not starting!")
            time.sleep(0.01)

        self.episode_start = time.time()
        #workaround to get things started sooner
        self.car_agent.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        #create random target to reach
        self.target = random.choice(self.world.get_map().get_spawn_points())
        self.get_route()
        return self.rgb_cam

    def step (self, action):
        self.car_agent.apply_control(carla.VehicleControl(throttle=action[0],steer=action[1]))
        time.sleep(1)
        velocity = self.car_agent.get_velocity()
        if (len(self.collisions) != 0):
            done = True
            reward = -100
        #velocity_kmh = int(3.6*np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2)))
        # elif velocity_kmh < 40: #might stop car from going in circle
        #     reward = -1
        else:
            done = False
            reward = 1
        if self.episode_start + max_ep_length < time.time():
            done = True
        closest_index = self.route_kdtree.query([[self.car_agent.get_location().x,self.car_agent.get_location().y,self.car_agent.get_location().z]],k=1)[1][0][0]
        command_encoded = self.command2onehot.get(self.route_commands[closest_index])
        d2target = np.sqrt(np.power(self.car_agent.get_location().x-self.target.location.x,2)+np.power(self.car_agent.get_location().y-self.target.location.y,2)+np.power(self.car_agent.get_location().z-self.target.location.z,2))
        velocity_mag = np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2))

        return (self.rgb_cam,command_encoded,velocity_mag,d2target), reward, done, None

    def cleanup(self):
        for actor in self.actors:
            actor.destroy()

    def get_route(self):
        map = self.world.get_map()
        dao = GlobalRoutePlannerDAO(map, 2)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        route = dict(grp.trace_route(self.spawn_point.location, self.target.location))
        
        self.route_waypoints = []
        self.route_commands = []
        for waypoint in route.keys():
            self.route_waypoints.append((waypoint.transform.location.x,waypoint.transform.location.y,waypoint.transform.location.z))
            self.route_commands.append(route.get(waypoint))
        self.route_kdtree = KDTree(np.array(self.route_waypoints))

def run_navigation(host,world_port):
    env = CarEnv(host,world_port)
    done = False
    s = env.reset()
    #while not done:
    i = 0
    while i < 5:
        a = [1,1] #go straight and slow
        s_prime, reward, done, info = env.step(a)
        i+=1
    env.cleanup()



def main(n_vehicles,host,world_port,tm_port):
    run_navigation(host,world_port)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world_port', type=int, required=True)
    parser.add_argument('--tm_port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)

    main(**vars(parser.parse_args()))
