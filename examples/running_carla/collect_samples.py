
import carla
import cv2
import argparse
import logging
import numpy as np
import random

iter = 0
height = 480
width = 640
fov = 10
FPS = 60


def process_img(img,iter):
    print ("processing image...")
    #iter += 1
    img = np.array(img.raw_data).reshape(height,width,4)
    rgb = img[:,:,:3]
    #norm = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    rgb_cam = rgb
    id = random.uniform(0, 1)
    cv2.imwrite("/scratch/cluster/stephane/cluster_quickstart/examples/running_carla/sample_frames/frame" + str(id) + ".png",rgb)
    #iter+=1
    print ("written image " + str(id))

def main(n_vehicles, host, world_port, tm_port):
    vehicles = []

    client = carla.Client(host, world_port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)

        traffic_manager.set_synchronous_mode(True)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        # cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # spawn vehicles.
        batch = []
        sensors = []

        for i, transform in enumerate(world.get_map().get_spawn_points()):
            if i >= n_vehicles:
                break

            blueprints = world.get_blueprint_library().filter('vehicle.*')
            blueprint = random.choice(blueprints)
            blueprint.set_attribute('role_name', 'autopilot')
            #vehicle = SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            vehicle = world.spawn_actor(blueprint,transform)
            vehicle.set_autopilot(True, traffic_manager.get_port())
            batch.append(vehicle)

            rgb_cam = world.get_blueprint_library().find("sensor.camera.rgb")
            rgb_cam.set_attribute("image_size_x",f"{width}")
            rgb_cam.set_attribute("image_size_y",f"{height}")
            rgb_cam.set_attribute("fov",f"{fov}")
            sensor_pos = carla.Transform(carla.Location(x=2.5,z=0.7))
            sensor = world.spawn_actor(rgb_cam, sensor_pos, attach_to=vehicle)
            sensors.append(sensor)
            sensor.listen(lambda data: process_img(data,iter))


#        for response in client.apply_batch_sync(batch, True):
#            if response.error:
#                print(response.error)
#            else:
#                vehicles.append(response.actor_id)

        # let them run around.
        for sample in range(10000):
            #print('Tick: %d' % t)
            #iter+=1
            for i, v in enumerate(world.get_actors().filter('*vehicle*')):
                print('Vehicle %d: id=%d, x=%.2f, y=%.2f' % (
                    i, v.id, v.get_location().x, v.get_location().y))

            world.tick()
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
        client.apply_batch([carla.command.DestroyActor(x) for x in sensors])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world_port', type=int, required=True)
    parser.add_argument('--tm_port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)

    main(**vars(parser.parse_args()))
