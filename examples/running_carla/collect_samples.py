import numpy as np
import cv2
import carla
import random
import sys

SHOW_PREVIEW = False
#maybe scale these down
height = 80
width = 80
fov = 10
max_ep_length = 60
FPS = 60
iter = 0
actors = []

def start_carla(host,world_port,tm_port):
    client = carla.Client(host,world_port)
    client.set_timeout(10.0)
    #self.world = self.client.get_world()
    world = client.load_world('Town01')

    blueprint_lib = world.get_blueprint_library()
    car_agent_model = blueprint_lib.filter("model3")[0]

    #get traffic light and stop sign info
    _map = world.get_map()

    settings = world.get_settings()
    settings.set(WeatherId=random.randrange(14))
    settings.set(SendNonPlayerAgentsInfo=True,NumberOfVehicles=random.randrange(30),NumberOfPedestrians=random.randrange(30),WeatherId=random.randrange(14))
    settings.randomize_seeds()
    world.apply_settings(self.settings)

    collisions = []
    vehicles = []
    spawn_point = random.choice(world.get_map().get_spawn_points())
    car_agent = world.try_spawn_actor(car_agent_model,spawn_point)

    #handle invalid spwawn point
    while car_agent == None:
        spawn_point = random.choice(world.get_map().get_spawn_points())
        car_agent = world.try_spawn_actor(car_agent_model,spawn_point)

    actors.append(car_agent)
    vehicles.append(car_agent)
    #get camera
    rgb_cam = blueprint_lib.find("sensor.camera.rgb")
    rgb_cam.set_attribute("image_size_x",f"{width}")
    rgb_cam.set_attribute("image_size_y",f"{height}")
    rgb_cam.set_attribute("fov",f"{fov}")
    sensor_pos = carla.Transform(carla.Location(x=2.5,z=0.7))
    sensor = world.spawn_actor(rgb_cam, sensor_pos, attach_to=car_agent)
    actors.append(sensor)
    sensor.listen(lambda data: process_img(data,save_video,iter))

    tm = client.get_trafficmanager(tm_port)
    tm_port = tm.get_port()
    for v in vehicles:
      v.set_autopilot(True,tm_port)
    tm.global_distance_to_leading_vehicle(5)
    tm.global_percentage_speed_difference(80)


def process_img(img,save_video,iter):
    img = np.array(img.raw_data).reshape(height,width,4)
    rgb = img[:,:,:3]
    #norm = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #rgb_cam = rgb
    cv2.imwrite("sample_frames/frame" + str(iter) + ".png",rgb)
    iter+=1
    if iter >10000:
        for agent in actors:
            agent.destroy()
        sys.exit()

def main(n_vehicles,host,world_port,tm_port):
    start_carla(host,world_port,tm_port)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world_port', type=int, required=True)
    parser.add_argument('--tm_port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)

    main(**vars(parser.parse_args()))
