import collections
import queue
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
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import wandb
import copy
from PIL import Image, ImageDraw

from shapely.geometry import LineString
from traffic_events import TrafficEventType
from statistics_manager import StatisticManager
#IK this is bad, fix file path stuff later :(
sys.path.append("/scratch/cluster/stephane/Carla_0.9.10/PythonAPI/carla/agents/navigation")
from global_route_planner import GlobalRoutePlanner
from global_route_planner_dao import GlobalRoutePlannerDAO

# from scripts.launch_carla import launch_carla_server
# from scripts.kill_carla import kill_carla


class CarlaEnv(object):
    def __init__(self, args, town='Town01'):
        # Tunable parameters
        self.FRAME_RATE = 30.0  # in Hz
        self.MAX_EP_LENGTH = 60  # in seconds
        self.MAX_EP_LENGTH = self.MAX_EP_LENGTH / (1.0 / self.FRAME_RATE)  # convert to ticks

        self._client = args.client

        self._statistics_manager = StatisticManager

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()
        self._blueprints = self._world.get_blueprint_library()
        self._spectator = self._world.get_spectator()
        self._car_agent_model = self._blueprints.filter("model3")[0]

        self.command2onehot =\
            {"RoadOption.LEFT":             [0, 0, 0, 0, 0, 1],
             "RoadOption.RIGHT":            [0, 0, 0, 0, 1, 0],
             "RoadOption.STRAIGHT":         [0, 0, 0, 1, 0, 0],
             "RoadOption.LANEFOLLOW":       [0, 0, 1, 0, 0, 0],
             "RoadOption.CHANGELANELEFT":   [0, 1, 0, 0, 0, 0],
             "RoadOption.CHANGELANERIGHT":  [1, 0, 0, 0, 0, 0]
             }

        self.DISTANCE_LIGHT = 15
        self.PROXIMITY_THRESHOLD = 50.0  # meters
        self.SPEED_THRESHOLD = 0.1
        self.WAYPOINT_STEP = 1.0  # meters
        self.ALLOWED_OUT_DISTANCE = 1.3          # At least 0.5, due to the mini-shoulder between lanes and sidewalks
        self.MAX_ALLOWED_VEHICLE_ANGLE = 120.0   # Maximum angle between the yaw and waypoint lane
        self.MAX_ALLOWED_WAYPOINT_ANGLE = 150.0  # Maximum change between the yaw-lane angle between frames
        self.WINDOWS_SIZE = 3   # Amount of additional waypoints checked (in case the first on fails)

        self.init()

    def __enter__(self):
        self.frame = self.set_sync_mode(True)
        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self._cleanup()
        self.set_sync_mode(False)

    def init(self, randomize=False, save_video=False, i=0):
        self._settings = self._world.get_settings()
        #get traffic light and stop sign info
        self._list_traffic_lights = []
        self._list_stop_signs = []
        self._last_red_light_id = None
        self._target_stop_sign = None
        self._stop_completed = False
        self._affected_by_stop = False
        self.stop_actual_value = 0
        self.light_actual_value = 0
        self._outside_lane_active = False
        self._wrong_lane_active = False
        self._last_road_id = None
        self._last_lane_id = None
        self._total_distance = 0
        self._wrong_distance = 0
        self._current_index = 0
        self.save_video = save_video

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)
        # self.rgb_img = np.reshape(np.zeros(80*80*3), [1, 80, 80, 3]) # DEBUG

        self._tick = 0
        self._car_agent = None

        # TODO: decide whether to use actor lists or dict
        # Get all static actors in world
        all_actors = self._world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._list_traffic_lights.append((_actor, center, waypoints))
            if 'traffic.stop' in _actor.type_id:
                self._list_stop_signs.append(_actor)
        if(randomize):
            self._settings.set(SendNonPlayerAgentsInfo=True, NumberOfVehicles=random.randrange(30),
                              NumberOfPedestrians=random.randrange(30), WeatherId=random.randrange(14))
            self._settings.randomize_seeds()
            self._world.apply_settings(self._settings)

        self.blocked_start = 0
        self.blocked = False
        self.last_col_time = 0
        self.last_col_id = 0
        self.n_step_cols = 0

        self.collisions = []
        self.events = []
        self.followed_waypoints = []

        # spawn car at random location
        np.random.seed(49)
        self._start_pose = np.random.choice(self._map.get_spawn_points())

        self._current_velocity = None

        self._spawn_car_agent()
        print('car agent spawned')
        self._setup_sensors(iter=i,save_video=self.save_video)

        # create random target to reach
        np.random.seed(6)
        self._target_pose = np.random.choice(self._map.get_spawn_points())
        while(self._target_pose is self._start_pose):
            self.target = np.random.choice(self._map.get_spawn_points())
        self.get_route()
        # create statistics manager
        self.statistics_manager = StatisticManager(self.route_waypoints)
        # get all initial waypoints
        self._pre_ego_waypoint = self._map.get_waypoint(self._car_agent.get_location())
        # some metrics for debugging
        self.colllided_w_static = False
        self.n_collisions = 0
        self.n_tafficlight_violations = 0
        self.n_stopsign_violations = 0
        self.n_route_violations = 0
        self.n_vehicle_blocked = 0
        self._time_start = time.time()

        # create sensor queues
        self._queues = []
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)
        # make_queue(self._world.on_tick)
        for sensor in self._actor_dict['camera']:
            make_queue(sensor.listen)

        self.target_waypoint_idx = 0
        self.at_waypoint = []
        self.followed_target_waypoints = []
        self.dist_to_target_wp_tr = None

    def reset(self, randomize, save_video, i):
        # self._cleanup()
        # self.init()
        return self.step(timeout=2)

    def _spawn_car_agent(self):
        self._car_agent = self._world.try_spawn_actor(self._car_agent_model, self._start_pose)
        # handle invalid spawn point
        while self._car_agent is None:
            self._start_pose = random.choice(self._map.get_spawn_points())
            self._car_agent = self._world.try_spawn_actor(self._car_agent_model, self._start_pose)
        self._actor_dict['car_agent'].append(self._car_agent)

    def _setup_sensors(self, save_video, height=80, width=80, fov=10, FPS=60, iter=0):
        sensor_relative_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        # get camera sensor
        self.rgb_cam = self._blueprints.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{width}")
        self.rgb_cam.set_attribute("image_size_y", f"{height}")
        self.rgb_cam.set_attribute("fov", f"{fov}")
        self._rgb_cam_sensor = self._world.spawn_actor(self.rgb_cam, sensor_relative_transform, attach_to=self._car_agent)
        self._actor_dict['camera'].append(self._rgb_cam_sensor)
        self.out = None
        if save_video:
            print ("saving video turned on")
            #self.cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.out = cv2.VideoWriter("episode_footage/output_"+str(iter)+".avi", fourcc,FPS, (height+60,width))
            self.n_img = 0

        # get collision sensor
        col_sensor_bp = self._blueprints.find("sensor.other.collision")
        self._col_sensor = self._world.spawn_actor(col_sensor_bp, sensor_relative_transform, attach_to=self._car_agent)
        self._actor_dict['col_sensor'].append(self._col_sensor)
        self._col_sensor.listen(lambda event: self.handle_collision(event))

        # get obstacle sensor
        obs_sensor_bp = self._blueprints.find("sensor.other.obstacle")
        self._obs_sensor = self._world.spawn_actor(obs_sensor_bp, sensor_relative_transform, attach_to=self._car_agent)
        self._actor_dict['obs_sensor'].append(self._obs_sensor)
        self._obs_sensor.listen(lambda event: self.handle_obstacle(event))


    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(block=True, timeout=timeout)
            if data.frame == self.frame:
                sensor_queue.task_done()
                # print(self.frame)
                return data

    def step(self, timeout, action=None):
        self.started_sim = True
        # spectator camera with overhead view of ego vehicle
        spectator_rot = self._car_agent.get_transform().rotation
        spectator_rot.pitch -= 10
        self._spectator.set_transform(carla.Transform(self._car_agent.get_transform().location + carla.Location(z=2), spectator_rot))

        if action is not None:
            self._car_agent.apply_control(carla.VehicleControl(throttle=action[0][0], steer=action[0][1]))
        else:
            self._car_agent.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        self.frame = self._world.tick()
        # time.sleep(0.5)
        self._tick += 1

        transform = self._car_agent.get_transform()
        velocity = self._car_agent.get_velocity()

        # set current waypoint to closest waypoint to agent position
        closest_index = self.route_kdtree.query([[self._car_agent.get_location().x, self._car_agent.get_location().y,
                                                  self._car_agent.get_location().z]], k=1)[1][0][0]
        self.at_waypoint = self.route_waypoints[closest_index]
        self.followed_waypoints.append(self.at_waypoint)

        # get command of current waypoint
        command_encoded = self.command2onehot.get(str(self.route_commands[closest_index]))

        # get next target waypoint assuming they are ordered by the route planner
        self.target_waypoint = self.route_waypoints[self.target_waypoint_idx]

        if self.at_waypoint == self.target_waypoint:
            self.followed_target_waypoints.append(self.target_waypoint)
            self.target_waypoint_idx += 1
            self.target_waypoint = self.route_waypoints[self.target_waypoint_idx]
            car_agent_trigger_pos = [self._car_agent.get_location().x, self._car_agent.get_location().y, self._car_agent.get_location().z]
            self.dist_to_target_wp_tr = self.statistics_manager.compute_route_length([car_agent_trigger_pos, self.target_waypoint])

        car_agent_x = self._car_agent.get_location().x
        car_agent_y = self._car_agent.get_location().y
        car_agent_z = self._car_agent.get_location().z

        target_x, target_y, target_z = self.target_waypoint
        dist_x2 = (car_agent_x - target_x)**2
        dist_y2 = (car_agent_y - target_y)**2
        dist_z2 = (car_agent_z - target_z)**2
        dist_to_target_wp = math.sqrt(dist_x2 + dist_y2 + dist_z2)
        # input(dist_to_next_wp)

        if self.dist_to_target_wp_tr:
            # print(self.dist_to_target_wp_tr)
            # print(dist_to_target_wp)
            dist_toward_target_wp = self.dist_to_target_wp_tr - dist_to_target_wp
        else:
            dist_toward_target_wp = 0

        # compute completed distance based on followed target waypoints TODO change to include negative progress
        # print(self.statistics_manager.compute_route_length(self.followed_target_waypoints))
        # print(dist_toward_target_wp)
        self.d_completed = (self.statistics_manager.compute_route_length(self.followed_target_waypoints) + dist_toward_target_wp)
        # if self.at_waypoint == self.target_waypoint:
        #     print('completed distance is:', self.d_completed)
        # print('completed distance is:', self.d_completed)
        # get distance to destination TODO change formula for this
        d2target = self.statistics_manager.route_record["route_length"] - \
                   self.d_completed

        velocity_kmh = int(3.6*np.sqrt(np.power(velocity.x, 2) + np.power(velocity.y, 2) + np.power(velocity.z, 2)))
        velocity_mag = np.sqrt(np.power(velocity.x, 2) + np.power(velocity.y, 2) + np.power(velocity.z, 2))
        self.cur_velocity = velocity_mag


        state = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in state)
        state = [self.process_img(img, 80, 80, self.save_video) for img in state]
        # state = self.rgb_img # DEBUG
        state = [state, velocity_mag, d2target]
        state.extend(command_encoded)

        #check for traffic light infraction/stoplight infraction
        self.check_traffic_light_infraction()
        self.check_stop_sign_infraction()
        #check if the vehicle is either on a sidewalk or at a wrong lane.
        self.check_outside_route_lane()

        #get done information
        if self._tick > self.MAX_EP_LENGTH or self.colllided_w_static:
            done = True
            self.events.append([TrafficEventType.ROUTE_COMPLETION, self.d_completed])
        elif d2target < 0.1:
            done = True
            self.events.append([TrafficEventType.ROUTE_COMPLETED])
        else:
            done = False
            self.events.append([TrafficEventType.ROUTE_COMPLETION, self.d_completed])

        # print(self.events)
        #get reward information
        self.statistics_manager.compute_route_statistics(time.time(), self.events)
        #------------------------------------------------------------------------------------------------------------------
        reward = self.statistics_manager.route_record["score_composed"] - self.statistics_manager.prev_score

        self.statistics_manager.prev_score = self.statistics_manager.route_record["score_composed"]
        #reward = self.statistics_manager.route_record["score_composed"]
        #self.events.clear()
        #------------------------------------------------------------------------------------------------------------------
        # reward = 1000*(self.d_completed - self.statistics_manager.prev_d_completed) + 0.05*(velocity_kmh-self.statistics_manager.prev_velocity_kmh) - 10*self.statistics_manager.route_record["score_penalty"]
        # self.statistics_manager.prev_d_completed = self.d_completed
        # self.statistics_manager.prev_velocity_kmh = velocity_kmh
        #------------------------------------------------------------------------------------------------------------------
        #reset is blocked if car is moving
        if self.cur_velocity > 0 and self.blocked:
            self.blocked = False
            self.blocked_start = 0
        self.n_step_cols = 0
        return state, reward, done, [self.statistics_manager.route_record['route_percentage'], self.n_collisions,
                                     self.n_tafficlight_violations, self.n_stopsign_violations, self.n_route_violations,
                                     self.n_vehicle_blocked]

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        print('cleaning up..')

        # TODO why doesn't this work?
        # self._client.apply_batch_sync([carla.command.DestroyActor(x[0]) for x in self._actor_dict.values()])

        [x[0].destroy() for x in self._actor_dict.values()]

        for q in self._queues:
            with q.mutex:
                q.queue.clear()

        self._actor_dict.clear()
        self._queues.clear()
        self._tick = 0
        self._time_start = time.time()
        self._car_agent = None
        self._spectator = None
        # self._world.tick()

    def set_sync_mode(self, sync):
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 1.0 / self.FRAME_RATE
        frame = self._world.apply_settings(settings)
        return frame

    def get_route(self):
        dao = GlobalRoutePlannerDAO(self._map, 2)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        route = dict(grp.trace_route(self._start_pose.location, self._target_pose.location))

        self.route_waypoints = []
        self.route_commands = []
        self.route_waypoints_unformatted = []
        for waypoint in route.keys():
            self.route_waypoints.append((waypoint.transform.location.x, waypoint.transform.location.y,
                                         waypoint.transform.location.z))
            self.route_commands.append(route.get(waypoint))
            self.route_waypoints_unformatted.append(waypoint)
        self.route_kdtree = KDTree(np.array(self.route_waypoints))

    def process_img(self, img, height, width, save_video):
        img = np.frombuffer(img.raw_data, dtype='uint8').reshape(height, width, 4)
        rgb = img[:, :, :3]
        #rgb_f = rgb[:, :, ::-1]
        if save_video and self.started_sim and 'route_percentage' in self.statistics_manager.route_record:
            #percent complete
            rgb_mat = cv2.UMat(rgb)
            rgb_mat = cv2.copyMakeBorder(rgb_mat, 60,0,0,0, cv2.BORDER_CONSTANT, None, 0)
            cv2.putText(rgb_mat, str(self.statistics_manager.route_record['route_percentage']), (2,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            #high level command
            #cv2.putText(rgb_mat, self.high_level_command, (2,25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            #closest waypoint (x,y,z)
            #cv2.putText(rgb_mat, str(self.closest_waypoint), (2,40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            #distance 2 waypoint
            cv2.putText(rgb_mat, str(self.d_completed), (2,55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            rgb = rgb.reshape(height+60,width,3)
            rgb_mat = cv2.resize(rgb_mat,(height,width))
            self.out.write(rgb_mat)
            #cv2.imwrite("/scratch/cluster/stephane/cluster_quickstart/examples/running_carla/episode_footage/frame_"+str(iter)+str(self.n_img)+".png",rgb)
            self.n_img+=1
        return rgb

    '''Evaluation tools'''

    def handle_collision(self, event):
        distance_vector = self._car_agent.get_location() - event.other_actor.get_location()
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if not (self.last_col_id == event.other_actor.id and time.time() - self.last_col_time < 1) and self.n_step_cols < 2:
            self.n_step_cols += 1
            self.collisions.append(event)
            self.n_collisions += 1
            self.last_col_id = event.other_actor.id
            self.last_col_time = time.time()

            if ("pedestrian" in event.other_actor.type_id):
                self.events.append([TrafficEventType.COLLISION_PEDESTRIAN])
            if ("vehicle" in event.other_actor.type_id):
                self.events.append([TrafficEventType.COLLISION_VEHICLE])
            if ("static" in event.other_actor.type_id):
                self.events.append([TrafficEventType.COLLISION_STATIC])
                self.colllided_w_static = True

    def handle_obstacle(self, event):
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
                    self.n_vehicle_blocked += 1

    def check_traffic_light_infraction(self):
        transform = self._car_agent.get_transform()
        location = transform.location

        veh_extent = self._car_agent.bounding_box.extent.x
        tail_close_pt = self.rotate_point(carla.Vector3D(-0.8 * veh_extent, 0.0, location.z), transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(carla.Vector3D(-veh_extent - 1, 0.0, location.z), transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:
            center_loc = carla.Location(center)
            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue
            for wp in waypoints:
                tail_wp = self._map.get_waypoint(tail_far_pt)
                # Calculate the dot product (Might be unscaled, as only its sign is important)
                ve_dir = self._car_agent.get_transform().get_forward_vector()
                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

                # Check the lane until all the "tail" has passed
                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and dot_ve_wp > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = self.rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp + 90)
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp - 90)
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)):
                        self.light_actual_value += 1
                        # location = traffic_light.get_transform().location
                        #red_light_event = TrafficEvent(event_type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION)
                        self.events.append([TrafficEventType.TRAFFIC_LIGHT_INFRACTION])
                        self._last_red_light_id = traffic_light.id
                        self.n_tafficlight_violations += 1
                        break

    def get_traffic_light_waypoints(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps

    def check_stop_sign_infraction(self):
        transform = self._car_agent.get_transform()
        location = transform.location
        if not self._target_stop_sign:
            # scan for stop signs
            self._target_stop_sign = self._scan_for_stop_sign()
        else:
            # we were in the middle of dealing with a stop sign
            if not self._stop_completed:
                # did the ego-vehicle stop?
                velocity = self._car_agent.get_velocity()
                current_speed = np.sqrt(np.power(velocity.x, 2) + np.power(velocity.y, 2) + np.power(velocity.z, 2))
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True

            if not self._affected_by_stop:
                stop_location = self._target_stop_sign.get_location()
                stop_extent = self._target_stop_sign.trigger_volume.extent

                if self.point_inside_boundingbox(location, stop_location, stop_extent):
                    self._affected_by_stop = True

            if not self.is_actor_affected_by_stop(self._car_agent, self._target_stop_sign):
                # is the vehicle out of the influence of this stop sign now?
                if not self._stop_completed and self._affected_by_stop:
                    # did we stop?
                    self.stop_actual_value += 1
                    #stop_location = self._target_stop_sign.get_transform().location
                    self.events.append([TrafficEventType.STOP_INFRACTION])
                    self.n_stopsign_violations += 1

                # reset state
                self._target_stop_sign = None
                self._stop_completed = False
                self._affected_by_stop = False

    def is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                next_wps = waypoint.next(self.WAYPOINT_STEP)
                if not next_wps:
                    break
                waypoint = next_wps[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self.point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _scan_for_stop_sign(self):
        target_stop_sign = None

        ve_tra = self._car_agent.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in self._list_stop_signs:
                if self.is_actor_affected_by_stop(self._car_agent, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign

    def check_outside_route_lane(self):
        self.test_status = None
        _waypoints = self.route_waypoints_unformatted #not sure if this is correct
        _route_length = len(self.route_waypoints)
        location = self._car_agent.get_location()
         # 1) Check if outside route lanes
        self._is_outside_driving_lanes(location)
        self._is_at_wrong_lane(location)
        if self._outside_lane_active or self._wrong_lane_active:
            self.test_status = "FAILURE"
        # 2) Get the traveled distance
        for index in range(self._current_index + 1,
                           min(self._current_index + self.WINDOWS_SIZE + 1, _route_length)):
            # Get the dot product to know if it has passed this location
            index_location = _waypoints[index].transform.location
            index_waypoint = self._map.get_waypoint(index_location)

            wp_dir = index_waypoint.transform.get_forward_vector()  # Waypoint's forward vector
            wp_veh = location - index_location  # vector waypoint - vehicle
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                # Get the distance traveled
                index_location = _waypoints[index].transform.location
                current_index_location = _waypoints[self._current_index].transform.location
                new_dist = current_index_location.distance(index_location)

                # Add it to the total distance
                self._current_index = index
                self._total_distance += new_dist

                # And to the wrong one if outside route lanes
                if self._outside_lane_active or self._wrong_lane_active:
                    self._wrong_distance += new_dist
        if self.test_status == "FAILURE" and self._total_distance != 0:
            self.events.append([TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION,self._wrong_distance / self._total_distance * 100])
            self.n_route_violations += 1

    def _is_outside_driving_lanes(self, location):
        """
        Detects if the ego_vehicle is outside driving lanes
        """

        current_driving_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving, project_to_road=True)
        current_parking_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Parking, project_to_road=True)

        driving_distance = location.distance(current_driving_wp.transform.location)
        if current_parking_wp is not None:  # Some towns have no parking
            parking_distance = location.distance(current_parking_wp.transform.location)
        else:
            parking_distance = float('inf')

        if driving_distance >= parking_distance:
            distance = parking_distance
            lane_width = current_parking_wp.lane_width
        else:
            distance = driving_distance
            lane_width = current_driving_wp.lane_width

        self._outside_lane_active = bool(distance > (lane_width / 2 + self.ALLOWED_OUT_DISTANCE))

    def _is_at_wrong_lane(self, location):
        """
        Detects if the ego_vehicle has invaded a wrong lane
        """

        current_waypoint = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving, project_to_road=True)
        current_lane_id = current_waypoint.lane_id
        current_road_id = current_waypoint.road_id

        # Lanes and roads are too chaotic at junctions
        if current_waypoint.is_junction:
            self._wrong_lane_active = False
        elif self._last_road_id != current_road_id or self._last_lane_id != current_lane_id:

            # Route direction can be considered continuous, except after exiting a junction.
            if self._pre_ego_waypoint.is_junction:
                yaw_waypt = current_waypoint.transform.rotation.yaw % 360
                yaw_actor = self._car_agent.get_transform().rotation.yaw % 360

                vehicle_lane_angle = (yaw_waypt - yaw_actor) % 360

                if vehicle_lane_angle < self.MAX_ALLOWED_VEHICLE_ANGLE \
                        or vehicle_lane_angle > (360 - self.MAX_ALLOWED_VEHICLE_ANGLE):
                    self._wrong_lane_active = False
                else:
                    self._wrong_lane_active = True

            else:
                # Check for a big gap in waypoint directions.
                yaw_pre_wp = self._pre_ego_waypoint.transform.rotation.yaw % 360
                yaw_cur_wp = current_waypoint.transform.rotation.yaw % 360

                waypoint_angle = (yaw_pre_wp - yaw_cur_wp) % 360

                if self.MAX_ALLOWED_WAYPOINT_ANGLE <= waypoint_angle <= (360 - self.MAX_ALLOWED_WAYPOINT_ANGLE):

                    # Is the ego vehicle going back to the lane, or going out? Take the opposite
                    self._wrong_lane_active = not bool(self._wrong_lane_active)
                else:

                    # Changing to a lane with the same direction
                    self._wrong_lane_active = False

        # Remember the last state
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id
        self._pre_ego_waypoint = current_waypoint

    @staticmethod
    def rotate_point(point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    @staticmethod
    def is_vehicle_crossing_line(seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)

        return not inter.is_empty

    @staticmethod
    def point_inside_boundingbox(point, bb_center, bb_extent):
        """
        X
        :param point:
        :param bb_center:
        :param bb_extent:
        :return:
        """

        # pylint: disable=invalid-name
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=9216):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=9216, z_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=7, stride=3),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

class PPO_Agent(nn.Module):
    def __init__(self, linear_state_dim,encoded_vector_size, action_dim, action_std):
        super(PPO_Agent, self).__init__()
        # action mean range -1 to 1
        self.actor= nn.Sequential(
                nn.Linear(encoded_vector_size + linear_state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )

        self.critic= nn.Sequential(
                nn.Linear(encoded_vector_size + linear_state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def actor_(self, frame, mes):
        frame = frame.to(device)
        mes = mes.to(device)
        if len(list(mes.size())) == 1:
            mes = mes.unsqueeze(0)
        if len(list(frame.size())) == 3:
            frame = frame.squeeze()
        X = torch.cat((frame, mes), 1)
        return self.actor(X)

    def critic_(self, frame, mes):
        frame = frame.to(device)
        mes = mes.to(device)
        if len(list(mes.size())) == 1:
            mes = mes.unsqueeze(0)
        if len(list(frame.size())) == 3:
            frame = frame.squeeze()
        X = torch.cat((frame, mes), 1)
        return self.critic(X)

    def choose_action(self, frame, mes):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            mean = self.actor_(frame, mes)
            cov_matrix = torch.diag(self.action_var)
            gauss_dist = MultivariateNormal(mean, cov_matrix)
            action = gauss_dist.sample()
            action_log_prob = gauss_dist.log_prob(action)
        return action, action_log_prob

    def get_training_params(self, frame, mes, action):
        frame = torch.stack(frame)
        mes = torch.stack(mes)
        if len(list(frame.size())) > 4:
            frame = torch.squeeze(frame)
        if len(list(mes.size())) > 2:
            mes = torch.squeeze(mes)

        action = torch.stack(action)

        mean = self.actor_(frame, mes)
        action_expanded = self.action_var.expand_as(mean)
        cov_matrix = torch.diag_embed(action_expanded).to(device)

        gauss_dist = MultivariateNormal(mean, cov_matrix)
        action_log_prob = gauss_dist.log_prob(action).to(device)
        entropy = gauss_dist.entropy().to(device)
        state_value = torch.squeeze(self.critic_(frame, mes)).to(device)
        return action_log_prob, state_value, entropy

def format_frame(frame,vae):
    frame = frame[0]
    frame = cv2.resize(frame,(127,127))
    frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    frame = frame[:, :, ::-1]
    frame = torch.FloatTensor(frame.copy())
    h, w, c = frame.shape
    frame = frame.unsqueeze(0).view(1, c, h, w)
    encoded_frame,_,_ = vae.encode(frame)
    return encoded_frame


def format_mes(mes):
    mes = torch.FloatTensor(mes)
    return mes


def train_PPO(args):
    wandb.init(project='PPO_Carla_Navigation')
    n_iters = 10000
    n_epochs = 50
    max_steps = 2000
    gamma = 0.99
    lr = 0.0001
    clip_val = 0.2
    avg_t = 0
    moving_avg = 0

    config = wandb.config
    config.learning_rate = lr

    encoded_vector_size = 128
    n_states = 8
    #currently the action array will be [throttle, steer]
    n_actions = 2

    action_std = 0.5
    #init models
    policy = PPO_Agent(n_states,encoded_vector_size, n_actions, action_std).to(device)
    optimizer = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    prev_policy = PPO_Agent(n_states,encoded_vector_size, n_actions, action_std).to(device)
    prev_policy.load_state_dict(policy.state_dict())

    vae = VAE()
    if encoded_vector_size == 128:
        vae.load_state_dict(torch.load("dim=127VAE_state_dictionary.pt"))
    elif encoded_vector_size == 64:
        vae.load_state_dict(torch.load("dim=64VAE_state_dictionary.pt"))
    else:
        print ("no VAE with this dimension")
        return
    vae.eval()

    wandb.watch(prev_policy)

    for iters in range(n_iters):
        # if iters % 50 == 0:
        #     kill_carla()
        #     launch_carla_server(args.world_port, gpu=3, boot_time=5)
        with CarlaEnv(args) as env:
            s, _, _, _ = env.reset(False, False, iters)
            t = 0
            episode_reward = 0
            done = False
            rewards = []
            eps_frames = []
            eps_mes = []
            actions = []
            actions_log_probs = []
            states_p = []
            while not done:
                a, a_log_prob = prev_policy.choose_action(format_frame(s[0],vae), format_mes(s[1:]))
                s_prime, reward, done, info = env.step(action=a.detach().tolist(), timeout=2)

                # if reward != 0:
                #     print('reward is:', reward)
                eps_frames.append(format_frame(s[0],vae).detach().clone())
                eps_mes.append(format_mes(s[1:]).detach().clone())
                actions.append(a.detach().clone())
                actions_log_probs.append(a_log_prob.detach().clone())
                rewards.append(copy.deepcopy(reward))
                states_p.append(copy.deepcopy(s_prime))
                s = s_prime
                t += 1
                episode_reward += reward
        if t == 1:
            continue
        print("Episode reward: " + str(episode_reward))
        print("Percent completed: " + str(info[0]))
        avg_t += t
        moving_avg = (episode_reward - moving_avg) * (2/(iters+2)) + moving_avg

        wandb.log({"episode_reward (suggested reward w/ ri)": episode_reward})
        wandb.log({"average_reward (suggested reward w/ ri)": moving_avg})
        wandb.log({"percent_completed": info[0]})
        wandb.log({"number_of_collisions": info[1]})
        wandb.log({"number_of_trafficlight_violations": info[2]})
        wandb.log({"number_of_stopsign_violations": info[3]})
        wandb.log({"number_of_route_violations": info[4]})
        wandb.log({"number_of_times_vehicle_blocked": info[5]})
        wandb.log({"timesteps before termination": t})
        wandb.log({"iteration": iters})

        if len(eps_frames) == 1:
            continue

        discounted_reward = 0
        for i in range(len(rewards)):
            rewards[len(rewards)-1-i] = rewards[len(rewards)-1-i] + (gamma*discounted_reward)
            discounted_reward = rewards[len(rewards)-1-i]
        # rewards = [episode_reward - r for r in rewards] ToDo: check that this is equivalent to the above

        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards-rewards.mean())/rewards.std()

        actions_log_probs = torch.FloatTensor(actions_log_probs).to(device)
        #train PPO
        for i in range(n_epochs):
            current_action_log_probs, state_values, entropies = policy.get_training_params(eps_frames, eps_mes, actions)

            policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())
            #policy_ratio = current_action_log_probs.detach()/actions_log_probs
            advantage = rewards - state_values.detach()
            advantage = (advantage - advantage.mean()) / advantage.std()
            update1 = (policy_ratio*advantage).float()
            update2 = (torch.clamp(policy_ratio, 1-clip_val, 1+clip_val) * advantage).float()
            loss = -torch.min(update1, update2) + 0.5*mse(state_values.float(), rewards.float()) - 0.001*entropies


            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            if i % 10 == 0:
                print("    on epoch " + str(i))
            #wandb.log({"loss": loss.mean()})

        if iters % 50 == 0:
            torch.save(policy.state_dict(), "policy_state_dictionary.pt")
        prev_policy.load_state_dict(policy.state_dict())


def run_model(args):
    n_iters = 20
    n_epochs = 50
    avg_t = 0
    moving_avg = 0

    encoded_vector_size = 128
    n_states = 8
    #currently the action array will be [throttle, steer]
    n_actions = 2

    action_std = 0.5
    #init models
    policy = PPO_Agent(n_states,encoded_vector_size, n_actions, action_std).to(device)
    policy.load_state_dict(torch.load("policy_state_dictionary.pt"))
    policy.eval()

    vae = VAE()
    if encoded_vector_size == 128:
        vae.load_state_dict(torch.load("dim=127VAE_state_dictionary.pt"))
    elif encoded_vector_size == 64:
        vae.load_state_dict(torch.load("dim=64VAE_state_dictionary.pt"))
    else:
        print ("no VAE with this dimension")
        return
    vae.eval()

    for iters in range(n_iters):
        # if iters % 50 == 0:
        #     kill_carla()
        #     launch_carla_server(args.world_port, gpu=3, boot_time=5)
        with CarlaEnv(args) as env:
            s, _, _, _ = env.reset(False, True, iters)
            t = 0
            episode_reward = 0
            done = False
            rewards = []
            while not done:
                a, a_log_prob = policy.choose_action(format_frame(s[0],vae), format_mes(s[1:]))
                s_prime, reward, done, info = env.step(action=a.detach().tolist(), timeout=2)

                rewards.append(reward)
                s = s_prime
                t += 1
                episode_reward += reward
        if t == 1:
            continue
        print("Episode reward: " + str(episode_reward))
        print("Percent completed: " + str(info[0]))
        avg_t += t
        moving_avg = (episode_reward - moving_avg) * (2/(iters+2)) + moving_avg


def random_baseline(args):
    wandb.init(project='PPO_Carla_Navigation')
    n_iters = 10000
    n_epochs = 50
    avg_t = 0
    moving_avg = 0

    config = wandb.config
    #config.learning_rate = lr

    n_states = 8
    #currently the action array will be [throttle, steer]
    n_actions = 2

    action_std = 0.5

    wandb.watch(policy)

    for iters in range(n_iters):
        # if iters % 50 == 0:
        #     kill_carla()
        #     launch_carla_server(args.world_port, gpu=3, boot_time=5)
        with CarlaEnv(args) as env:
            s, _, _, _ = env.reset(False, False, iters)
            t = 0
            episode_reward = 0
            done = False
            rewards = []
            while not done:
                s_prime, reward, done, info = env.step(action=[[random.uniform(-1, 1),random.uniform(-1, 1)]], timeout=2)

                rewards.append(reward)
                s = s_prime
                t += 1
                episode_reward += reward
        if t == 1:
            continue
        print("Episode reward: " + str(episode_reward))
        print("Percent completed: " + str(info[0]))
        avg_t += t
        moving_avg = (episode_reward - moving_avg) * (2/(iters+2)) + moving_avg

        wandb.log({"episode_reward (suggested reward w/ ri)": episode_reward})
        wandb.log({"average_reward (suggested reward w/ ri)": moving_avg})
        wandb.log({"percent_completed": info[0]})
        wandb.log({"number_of_collisions": info[1]})
        wandb.log({"number_of_trafficlight_violations": info[2]})
        wandb.log({"number_of_stopsign_violations": info[3]})
        wandb.log({"number_of_route_violations": info[4]})
        wandb.log({"number_of_times_vehicle_blocked": info[5]})
        wandb.log({"timesteps before termination": t})
        wandb.log({"iteration": iters})

def launch_client(args):
    client = carla.Client(args.host, args.world_port)
    client.set_timeout(args.client_timeout)
    return client

def main(args):
    # Create client outside of Carla environment to avoid creating zombie clients
    args.client = launch_client(args)
    train_PPO(args)
    #random_baseline(args)
    #run_model(args)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world_port', type=int, required=True)
    parser.add_argument('--tm_port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)
    parser.add_argument('--client_timeout', type=int, default=10)

    main(parser.parse_args())
