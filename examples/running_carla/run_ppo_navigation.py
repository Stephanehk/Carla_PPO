import numpy as np
import cv2
import carla
import planner
import argparse
import logging
import time
import math
import random

class CarEnv:
    rgb_cam = None

    def __init__(self,host,world_port):
        self.client = carla.Client(host,world_port)
        self.client.set_timeout(10.0)
        #self.world = self.client.get_world()
        self.world = self.client.load_world('Town01')
        self._planner = planner.Planner('Town01')

        self.blueprint_lib = self.world.get_blueprint_library()
        self.car_agent_model = self.blueprint_lib.filter("model3")[0]

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
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)
        #get collision sensor
        col_sensor = self.blueprint_lib.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_sensor,sensor_pos,attach_to=self.car_agent)
        self.actors.append(self.col_sensor)
        col_sensor.listen(lambda event: self.collision_data(event))

        while self.rgb_cam is None:
            print ("camera is not starting!")
            time.sleep(0.01)

        self.episode_start = time.time()
        #workaround to get things started sooner
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        #create random target to reach
        self.target = random.choice(self.world.get_map().get_spawn_points())

        return self.rgb_cam

    def step (self, action):
        self.vehicle.apply_control(carla.VehicleControl(throttle=action[0],steer=action[1]))
        velocity = self.vehicle.get_velocity()
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
        return self.rgb_cam, reward, done, None

    def cleanup(self):
        for actor in actors:
            actor.destroy()

    def sldist(self,c1, c2):
        return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

    def get_path(self):
        """
        Returns the path were the log was saved.
        """
        return self._recording.path

    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self._planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions


    def navigation_info_at_step (self):
        measurements, sensor_data = self.client.read_data()
        directions = self._get_directions(measurements.player_measurements.transform, self.target)

        current_x = measurements.player_measurements.transform.location.x
        current_y = measurements.player_measurements.transform.location.y
        distance = sldist([current_x, current_y],[target.location.x, target.location.y])
        return directions, distance


def run_navigation(host,world_port):
    env = CarEnv(host,world_port)
    dont = False
    s = env.reset()
    while not done:
        a = [0.3,0] #go straight and slow
        s_prime, reward, done, info = env.step(a)


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
