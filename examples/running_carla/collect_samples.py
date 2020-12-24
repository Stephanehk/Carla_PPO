import carla

import argparse
import logging
from numpy import random

iter = 0
def process_img(self,img,iter):
    img = np.array(img.raw_data).reshape(height,width,4)
    rgb = img[:,:,:3]
    #norm = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    self.rgb_cam = rgb
    cv2.imwrite ("sample_frames/frame" + str(iter) + ".png")
    iter+=1



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

        for i, transform in enumerate(world.get_map().get_spawn_points()):
            if i >= n_vehicles:
                break

            blueprints = world.get_blueprint_library().filter('vehicle.*')
            blueprint = random.choice(blueprints)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            batch.append()

            self.rgb_cam = self.blueprint_lib.find("sensor.camera.rgb")
            self.rgb_cam.set_attribute("image_size_x",f"{width}")
            self.rgb_cam.set_attribute("image_size_y",f"{height}")
            self.rgb_cam.set_attribute("fov",f"{fov}")
            sensor_pos = carla.Transform(carla.Location(x=2.5,z=0.7))
            self.sensor = self.world.spawn_actor(self.rgb_cam, sensor_pos, attach_to=vehicle)

            self.sensor.listen(lambda data: self.process_img(data,iter))


        for response in client.apply_batch_sync(batch, True):
            if response.error:
                print(response.error)
            else:
                vehicles.append(response.actor_id)

        # let them run around.
        for t in range(100):
            print('Tick: %d' % t)

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world_port', type=int, required=True)
    parser.add_argument('--tm_port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)

    main(**vars(parser.parse_args()))
