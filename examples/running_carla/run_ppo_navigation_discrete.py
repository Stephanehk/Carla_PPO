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
from torch.distributions import Categorical
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import wandb
from shapely.geometry import LineString
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

        self.command2onehot = {"RoadOption.LEFT":[0,0,0,0,0,1], "RoadOption.RIGHT":[0,0,0,0,1,0], "RoadOption.STRAIGHT":[0,0,0,1,0,0],"RoadOption.LANEFOLLOW":[0,0,1,0,0,0],"RoadOption.CHANGELANELEFT":[0,1,0,0,0,0],"RoadOption.CHANGELANERIGHT":[1,0,0,0,0,0]}

        #get traffic light and stop sign info
        self._map = self.world.get_map()
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
        self.DISTANCE_LIGHT = 15
        self.PROXIMITY_THRESHOLD = 50.0  # meters
        self.SPEED_THRESHOLD = 0.1
        self.WAYPOINT_STEP = 1.0  # meters
        self.ALLOWED_OUT_DISTANCE = 1.3          # At least 0.5, due to the mini-shoulder between lanes and sidewalks
        self.MAX_ALLOWED_VEHICLE_ANGLE = 120.0   # Maximum angle between the yaw and waypoint lane
        self.MAX_ALLOWED_WAYPOINT_ANGLE = 150.0  # Maximum change between the yaw-lane angle between frames
        self.WINDOWS_SIZE = 3 # Amount of additional waypoints checked (in case the first on fails)

        all_actors = self.world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._list_traffic_lights.append((_actor, center, waypoints))
            if 'traffic.stop' in _actor.type_id:
                self._list_stop_signs.append(_actor)

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

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    def process_img(self,img,save_video,iter):
        img = np.array(img.raw_data).reshape(height,width,4)
        rgb = img[:,:,:3]
        #norm = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.rgb_cam = rgb
        if save_video:
            self.out.write(rgb)
            #cv2.imwrite("/scratch/cluster/stephane/cluster_quickstart/examples/running_carla/episode_footage/frame_"+str(iter)+str(self.n_img)+".png",rgb)
            self.n_img+=1

    def reset(self, randomize, save_video, iter):
        if (randomize):
            self.settings = world.get_settings()
            self.settings.set(WeatherId=random.randrange(14))
            self.settings.set(SendNonPlayerAgentsInfo=True,NumberOfVehicles=random.randrange(30),NumberOfPedestrians=random.randrange(30),WeatherId=random.randrange(14))
            self.settings.randomize_seeds()
            self.world.apply_settings(self.settings)

        self.blocked_start= 0
        self.blocked = False
        self.last_col_time = 0
        self.last_col_id = 0

        self.collisions = []
        self.actors = []
        self.events = []
        self.followed_waypoints = []
        #spawn car randomly
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.car_agent = self.world.try_spawn_actor(self.car_agent_model,self.spawn_point)
        #handle invalid spwawn point
        while self.car_agent == None:
            self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
            self.car_agent = self.world.try_spawn_actor(self.car_agent_model,self.spawn_point)

        self.actors.append(self.car_agent)
        #get camera
        self.rgb_cam = self.blueprint_lib.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x",f"{width}")
        self.rgb_cam.set_attribute("image_size_y",f"{height}")
        self.rgb_cam.set_attribute("fov",f"{fov}")
        sensor_pos = carla.Transform(carla.Location(x=2.5,z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, sensor_pos, attach_to=self.car_agent)
        self.actors.append(self.sensor)
        #setup record video
        self.out = None
        if save_video:
            print ("saving video turned on")
            #self.cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.out = cv2.VideoWriter("episode_footage/output_"+str(iter)+".avi", fourcc,1, (height,width))
            self.n_img = 0
        self.sensor.listen(lambda data: self.process_img(data,save_video,iter))
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
        while (self.target == self.spawn_point):
            self.target = random.choice(self.world.get_map().get_spawn_points())
        self.get_route()
        #create statistics manager
        self.statistics_manager = StatisticManager(self.route_waypoints)
        #get all initial waypoints
        self._pre_ego_waypoint = self._map.get_waypoint(self.car_agent.get_location())
        #some metrics for debugging
        self.colllided_w_static = False
        self.n_collisions = 0
        self.n_tafficlight_violations = 0
        self.n_stopsign_violations = 0
        self.n_route_violations = 0
        self.n_vehicle_blocked = 0

        return [self.rgb_cam,0,0,0,0,0,0,0,0]

    def handle_collision (self,event):
        distance_vector = self.car_agent.get_location() - event.other_actor.get_location()
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if not (self.last_col_id == event.other_actor.id and time.time()- self.last_col_time < 1):
            self.collisions.append(event)
            self.n_collisions+=1
            self.last_col_id = event.other_actor.id
            self.last_col_time = time.time()

            if ("pedestrian" in event.other_actor.type_id):
                self.events.append([TrafficEventType.COLLISION_PEDESTRIAN])
            if ("vehicle" in event.other_actor.type_id):
                self.events.append([TrafficEventType.COLLISION_VEHICLE])
            if ("static" in event.other_actor.type_id):
                self.events.append([TrafficEventType.COLLISION_STATIC])
                self.colllided_w_static = True

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
                    self.n_vehicle_blocked+=1

    def is_vehicle_crossing_line(self, seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)

        return not inter.is_empty

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

        ve_tra = self.car_agent.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in self._list_stop_signs:
                if self.is_actor_affected_by_stop(self._actor, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign

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
                yaw_actor = self.car_agent.get_transform().rotation.yaw % 360

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

                if waypoint_angle >= self.MAX_ALLOWED_WAYPOINT_ANGLE \
                        and waypoint_angle <= (360 - self.MAX_ALLOWED_WAYPOINT_ANGLE):

                    # Is the ego vehicle going back to the lane, or going out? Take the opposite
                    self._wrong_lane_active = not bool(self._wrong_lane_active)
                else:

                    # Changing to a lane with the same direction
                    self._wrong_lane_active = False

        # Remember the last state
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id
        self._pre_ego_waypoint = current_waypoint


    def check_traffic_light_infraction (self):
        transform = self.car_agent.get_transform()
        location = transform.location

        veh_extent = self.car_agent.bounding_box.extent.x
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
                ve_dir = self.car_agent.get_transform().get_forward_vector()
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
                        self.n_tafficlight_violations+=1
                        break

    def check_stop_sign_infraction (self):
        transform = self.car_agent.get_transform()
        location = transform.location
        if not self._target_stop_sign:
            # scan for stop signs
            self._target_stop_sign = self._scan_for_stop_sign()
        else:
            # we were in the middle of dealing with a stop sign
            if not self._stop_completed:
                # did the ego-vehicle stop?
                velocity = self.car_agent.get_velocity()
                current_speed = np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2))
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True

            if not self._affected_by_stop:
                stop_location = self._target_stop_sign.get_location()
                stop_extent = self._target_stop_sign.trigger_volume.extent

                if self.point_inside_boundingbox(location, stop_location, stop_extent):
                    self._affected_by_stop = True

            if not self.is_actor_affected_by_stop(self._actor, self._target_stop_sign):
                # is the vehicle out of the influence of this stop sign now?
                if not self._stop_completed and self._affected_by_stop:
                    # did we stop?
                    self.stop_actual_value += 1
                    #stop_location = self._target_stop_sign.get_transform().location
                    self.events.append([TrafficEventType.STOP_INFRACTION])
                    self.n_stopsign_violations+=1

                # reset state
                self._target_stop_sign = None
                self._stop_completed = False
                self._affected_by_stop = False

    def check_outside_route_lane (self):
        self.test_status = None
        _waypoints = self.route_waypoints_unformatted #not sure if this is correct
        _route_length = len(self.route_waypoints)
        location = self.car_agent.get_location()
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
            self.n_route_violations+=1

    def step (self, action):
        self.car_agent.apply_control(carla.VehicleControl(throttle=action[0][0],steer=action[0][1]))
        #time.sleep(1)
        velocity = self.car_agent.get_velocity()

        #get state information
        closest_index = self.route_kdtree.query([[self.car_agent.get_location().x,self.car_agent.get_location().y,self.car_agent.get_location().z]],k=1)[1][0][0]
        self.followed_waypoints.append(self.route_waypoints[closest_index])
        command_encoded = self.command2onehot.get(str(self.route_commands[closest_index]))
        #d2target = np.sqrt(np.power(self.car_agent.get_location().x-self.target.location.x,2)+np.power(self.car_agent.get_location().y-self.target.location.y,2)+np.power(self.car_agent.get_location().z-self.target.location.z,2))
        d2target = self.statistics_manager.route_record["route_length"] - self.statistics_manager.compute_route_length(self.followed_waypoints)
        self.d_completed = self.statistics_manager.compute_route_length(self.followed_waypoints)
        velocity_kmh = int(3.6*np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2)))
        velocity_mag = np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2))
        self.cur_velocity = velocity_mag

        state = [self.rgb_cam,velocity_mag,d2target]
        state.extend(command_encoded)

        #check for traffic light infraction/stoplight infraction
        self.check_traffic_light_infraction()
        self.check_stop_sign_infraction()
        #check if the vehicle is either on a sidewalk or at a wrong lane.
        self.check_outside_route_lane()

        #get done information
        if self.episode_start + max_ep_length < time.time() or self.colllided_w_static:
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
        #------------------------------------------------------------------------------------------------------------------
        #reward = self.statistics_manager.route_record["score_composed"] - self.statistics_manager.prev_score
        #self.statistics_manager.prev_score = self.statistics_manager.route_record["score_composed"]
        reward = self.statistics_manager.route_record["score_composed"] - self.statistics_manager.prev_score
        self.statistics_manager.prev_score = self.statistics_manager.route_record["score_composed"]
        #------------------------------------------------------------------------------------------------------------------
        # reward = 1000*(self.d_completed - self.statistics_manager.prev_d_completed) + 0.05*(velocity_kmh-self.statistics_manager.prev_velocity_kmh) - 10*self.statistics_manager.route_record["score_penalty"]
        # self.statistics_manager.prev_d_completed = self.d_completed
        # self.statistics_manager.prev_velocity_kmh = velocity_kmh
        #------------------------------------------------------------------------------------------------------------------
        #reset is blocked if car is moving
        if self.cur_velocity > 0 and self.blocked == True:
            self.blocked = False
            self.blocked_start = 0
        return state, reward, done, [self.statistics_manager.route_record['route_percentage'], self.n_collisions, self.n_tafficlight_violations,self.n_stopsign_violations,self.n_route_violations,self.n_vehicle_blocked]

    def cleanup(self):
        for actor in self.actors:
            actor.destroy()
        #end video
        if self.out != None:
            self.out.release()
            #cv2.destroyAllWindows()

    def get_route(self):
        map = self.world.get_map()
        dao = GlobalRoutePlannerDAO(map, 2)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        route = dict(grp.trace_route(self.spawn_point.location, self.target.location))

        self.route_waypoints = []
        self.route_commands = []
        self.route_waypoints_unformatted = []
        for waypoint in route.keys():
            self.route_waypoints.append((waypoint.transform.location.x,waypoint.transform.location.y,waypoint.transform.location.z))
            self.route_commands.append(route.get(waypoint))
            self.route_waypoints_unformatted.append(waypoint)
        self.route_kdtree = KDTree(np.array(self.route_waypoints))

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class PPO_Agent(nn.Module):
    def __init__(self, linear_state_dim, action_dim, action_std):
        self.action_options = [[[0,-1]],[[0,0]],[[0,1]],[[0.5,-1]],[[0.5,0]],[[0.5,1]],[[1,-1]],[[1,0]],[[1,1]], [[0,-0.5]],[[0,0.5]],[[0.5,-0.5]],[[0.5,0.5]],[[1,-0.5]],[[1,0.5]]]
        super(PPO_Agent, self).__init__()
        # action mean range -1 to 1
        self.actorConv = nn.Sequential(
                nn.Conv2d(3,6,5),
                nn.Tanh(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(6,12,5),
                nn.Tanh(),
                nn.MaxPool2d(2,2),
                nn.Flatten()
                )
        self.actorLin = nn.Sequential(
                nn.Linear(12*17*17 + linear_state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Softmax(dim=-1)
                )

        self.criticConv = nn.Sequential(
                nn.Conv2d(3,6,5),
                nn.Tanh(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(6,12,5),
                nn.Tanh(),
                nn.MaxPool2d(2,2),
                nn.Flatten()
                )
        self.criticLin = nn.Sequential(
                nn.Linear(12*17*17 + linear_state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def actor(self,frame,mes):
        frame = frame.to(device)
        mes = mes.to(device)
        if len(list(mes.size())) == 1:
            mes = mes.unsqueeze(0)
        vec = self.actorConv(frame)
        X = torch.cat((vec,mes),1)
        return self.actorLin(X)

    def critic(self,frame,mes):
        frame = frame.to(device)
        mes = mes.to(device)
        if len(list(mes.size())) == 1:
            mes = mes.unsqueeze(0)
        vec = self.criticConv(frame)
        X = torch.cat((vec,mes),1)
        return self.criticLin(X)

    def choose_action(self,frame,mes):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #state = torch.FloatTensor(state).to(device)
        action_prob = self.actor(frame,mes)
        dist = Categorical(action_prob)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return self.action_options[action], action, action_log_prob

    def get_training_params(self,frame,mes, action):
        frame = torch.stack(frame)
        mes = torch.stack(mes)
        if len(list(frame.size())) > 4:
            frame = torch.squeeze(frame)
        if len(list(mes.size())) > 2:
            mes = torch.squeeze(mes)

        action = torch.stack(action)

        action_prob = self.actor(frame,mes)
        dist = Categorical(action_prob)
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(frame,mes)
        return action_log_prob, state_value, entropy

def format_frame(frame):
    frame = torch.FloatTensor(frame)
    h,w,c = frame.shape
    frame = frame.unsqueeze(0).view(1, c, h, w)
    return frame

def format_mes(mes):
    mes = torch.FloatTensor(mes)
    return mes

def train_PPO(host,world_port):
    wandb.init(project='PPO_Carla_Navigation')
    env = CarEnv(host,world_port)
    n_iters = 10000
    n_epochs = 50
    max_steps = 2000
    gamma = 0.9
    lr = 0.0001
    clip_val = 0.2
    avg_t = 0
    moving_avg = 0

    config = wandb.config
    config.learning_rate = lr


    n_states = 8
    #currently the action array will be [throttle, steer]
    n_actions = 15

    action_std = 0.5
    #init models
    policy = PPO_Agent(n_states, n_actions, action_std).to(device)

    optimizer = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    prev_policy = PPO_Agent(n_states, n_actions, action_std).to(device)
    prev_policy.load_state_dict(policy.state_dict())


    wandb.watch(prev_policy)

    for iters in range (n_iters):
        s = env.reset(False,False,iters)
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
            formatted_a, a, a_log_prob = prev_policy.choose_action(format_frame(s[0]), format_mes(s[1:]))
            s_prime, reward, done, info = env.step(formatted_a)

            eps_frames.append(format_frame(s[0]))
            eps_mes.append(format_mes(s[1:]))
            actions.append(a)
            actions_log_probs.append(a_log_prob)
            rewards.append(reward)
            states_p.append(s_prime)

            s = s_prime
            t+=1
            episode_reward+=reward

        env.cleanup()
        if t == 1:
            continue
        print ("Episode reward: " + str(episode_reward))
        print ("Percent completed: " + str(info[0]))
        avg_t+=t
        moving_avg = (episode_reward - moving_avg) * (2/(iters+2)) + moving_avg

        wandb.log({"episode_reward": episode_reward})
        wandb.log({"average_reward": moving_avg})
        wandb.log({"percent_completed": info[0]})
        wandb.log({"number_of_collisions": info[1]})
        wandb.log({"number_of_trafficlight_violations": info[2]})
        wandb.log({"number_of_stopsign_violations": info[3]})
        wandb.log({"number_of_route_violations": info[4]})
        wandb.log({"number_of_times_vehicle_blocked": info[5]})
        wandb.log({"timesteps before termination": t})
        if (len(eps_frames) == 1):
            continue

        rewards = torch.tensor(rewards).to(device)
        rewards= (rewards-rewards.mean())/rewards.std()

        actions_log_probs = torch.FloatTensor(actions_log_probs).to(device)
        #train PPO
        for i in range(n_epochs):
            current_action_log_probs, state_values, entropies = policy.get_training_params(eps_frames,eps_mes,actions)

            policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())
            #policy_ratio = current_action_log_probs.detach()/actions_log_probs
            advantage = rewards - state_values.detach()

            update1 = (policy_ratio*advantage).float()
            update2 = (torch.clamp(policy_ratio,1-clip_val, 1+clip_val) * advantage).float()
            loss = -torch.min(update1,update2) + 0.5*mse(state_values.float(),rewards.float()) - 0.001*entropies

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            print ("    on epoch " + str(i))
            #wandb.log({"loss": loss.mean()})

        if iters % 50 == 0:
            torch.save(policy.state_dict(),"policy_state_dictionary.pt")
        prev_policy.load_state_dict(policy.state_dict())


def main(n_vehicles,host,world_port,tm_port):
    #train_PPO(host,world_port)
    #random_baseline(host,world_port)
    run_model(host,world_port)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world_port', type=int, required=True)
    parser.add_argument('--tm_port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)

    main(**vars(parser.parse_args()))
