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

from traffic_events import TrafficEventType
from statistics_manager import StatisticManager
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
        self.blocked_start= 0
        self.blocked = False

        self.collisions = []
        self.actors = []
        self.events = []
        self.followed_waypoints = []
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
        self.col_sensor.listen(lambda event: self.handle_collision(event))
        #get obstacle sensor
        obs_sensor = self.blueprint_lib.find("sensor.other.obstacle")
        self.obs_sensor = self.world.spawn_actor(obs_sensor,sensor_pos,attach_to=self.car_agent)
        self.actors.append(self.obs_sensor)
        self.obs_sensor.listen(lambda event: self.handle_obstacle(event))

        while self.rgb_cam is None:
            print ("camera is not starting!")
            time.sleep(0.01)

        self.episode_start = time.time()
        #workaround to get things started sooner
        self.car_agent.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        #create random target to reach
        self.target = random.choice(self.world.get_map().get_spawn_points())
        self.get_route()
        #create statistics manager
        self.statistics_manager = StatisticManager(self.route_waypoints)

        return self.rgb_cam

    def handle_collision (self,event):
        self.collisions.append(event)
        #print (event)
        #print (event.other_actor)
        #print (event.other_actor.type_id)
        #print (event.other_actor.semantic_tags)
        if ("pedestrian" in event.other_actor.type_id):
            self.events.append([TrafficEventType.COLLISION_PEDESTRIAN])
        if ("vehicle" in event.other_actor.type_id):
            self.events.append([TrafficEventType.COLLISION_VEHICLE])
        if ("static" in event.other_actor.type_id):
            self.events.append([TrafficEventType.COLLISION_STATIC])

    def handle_obstacle(self,event):
        if event.distance < 0.5 and self.cur_velocity == 0:
            if self.blocked == False:
                self.blocked = True
                self.blocked_start = time.time()
            else:
                #if the car has been blocked for more that 180 seconds
                if time.time() - self.blocked_start > 180:
                    self.events.append([TrafficEventType.VEHICLE_BLOCKED])
                    #reset
                    self.blocked = False
                    self.blocked_start = 0


    def step (self, action):
        self.car_agent.apply_control(carla.VehicleControl(throttle=action[0],steer=action[1]))
        time.sleep(1)
        velocity = self.car_agent.get_velocity()

        #get state information
        closest_index = self.route_kdtree.query([[self.car_agent.get_location().x,self.car_agent.get_location().y,self.car_agent.get_location().z]],k=1)[1][0][0]
        self.followed_waypoints.append(self.route_waypoints[closest_index])
        command_encoded = self.command2onehot.get(self.route_commands[closest_index])
        #d2target = np.sqrt(np.power(self.car_agent.get_location().x-self.target.location.x,2)+np.power(self.car_agent.get_location().y-self.target.location.y,2)+np.power(self.car_agent.get_location().z-self.target.location.z,2))
        d2target = self.statistics_manager.route_record["route_length"] - self.statistics_manager.compute_route_length(self.followed_waypoints)
        self.d_completed = d2target
        #velocity_kmh = int(3.6*np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2)))
        velocity_mag = np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2))
        self.cur_velocity = velocity_mag
        state = (self.rgb_cam,command_encoded,velocity_mag,d2target)

        #check for traffic light infraction/stoplight infraction
        #TODO: Right now stoplights and traffic lights are handled the same, which is incorrect
        #https://carla.readthedocs.io/en/latest/ref_code_recipes/#traffic-light-recipe
        if self.car_agent.is_at_traffic_light():
            traffic_light = self.car_agent.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red and velocity_mag > 0.2:
                self.events.append([TrafficEventType.TRAFFIC_LIGHT_INFRACTION])

        #get done information
        if self.episode_start + max_ep_length < time.time():
            done = True
            self.events.append([TrafficEventType.ROUTE_COMPLETION, self.d_completed])
        elif d2target < 0.1:
            done = True
            self.events.append([TrafficEventType.ROUTE_COMPLETED])
        else:
            done = False
            self.events.append([TrafficEventType.ROUTE_COMPLETION, self.d_completed])
        
        #get reward information
        self.statistics_manager.compute_route_statistics(time.time(), self.events)
        reward = self.statistics_manager.route_record["score_composed"] - self.statistics_manager.prev_score
        self.statistics_manager.prev_score = self.statistics_manager.route_record["score_composed"]
        #print ("distance 2 target:")
        #print (d2target)
        #reset is blocked if car is moving
        if self.cur_velocity > 0 and self.blocked == True:
            self.blocked = False
            self.blocked_start = 0
        return state, reward, done, None

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
   
    rewards = []
    while not done:
        a = [1,1] #go straight and slow
        s_prime, reward, done, info = env.step(a)
        rewards.append(reward)

    #print (len(env.events))
    print (rewards)
    #for e in env.events:
    #    print (e)
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
