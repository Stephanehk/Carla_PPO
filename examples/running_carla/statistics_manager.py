import carla
import numpy as np
import math

from traffic_events import TrafficEventType

class StatisticManager:

    def __init__(self, trajectory,waypoints):
        self.route_record = {}
        self.route_record['route_length'] = self.compute_route_length(trajectory)
        self._accum_meters = self.compute_accum_length(waypoints)
        self.prev_score = 0
        self.prev_d_completed = 0
        self.prev_velocity_kmh = 0
        self.PENALTY_COLLISION_PEDESTRIAN = 0.50
        self.PENALTY_COLLISION_VEHICLE = 0.60
        self.PENALTY_COLLISION_STATIC = 0.65
        self.PENALTY_TRAFFIC_LIGHT = 0.70
        self.PENALTY_STOP = 0.80
        self.prev_route_infractions = 0
        self.prev_route_completion = 0
        self._current_index = 0
        self._wsize = 2 #Im not sure what this parameter controls
        self.route_record['route_percentage'] = 0

    def compute_route_length(self, trajectory):
        route_length = 0.0
        previous_transform = None
        for transform in trajectory:
            if previous_transform:
                x, y, z = transform
                prev_x, prev_y, prev_z = previous_transform
                dist = math.sqrt((x-prev_x)*(x-prev_x) +
                                 (y-prev_y)*(y-prev_y) +
                                 (z-prev_z)*(z-prev_z))
                route_length += dist
            previous_transform = transform

        return route_length

    def compute_accum_length(self,_waypoints):
        accum_meters = []
        prev_wp = _waypoints[0]
        for i, wp in enumerate(_waypoints):
            d = wp.transform.location.distance(prev_wp.transform.location)
            if i > 0:
                accum = accum_meters[i - 1]
            else:
                accum = 0

            accum_meters.append(d + accum)
            prev_wp = wp
        return accum_meters

    def segment_completed (self,_waypoints,_map,actor_location):
        completed = False
        for index in range(self._current_index, min(self._current_index + self._wsize + 1, self.route_record['route_length'])):
            # Get the dot product to know if it has passed this location
            ref_waypoint = _waypoints[index]
            wp = _map.get_waypoint(ref_waypoint.transform.location)
            wp_dir = wp.transform.get_forward_vector()          # Waypoint's forward vector
            wp_veh = actor_location - ref_waypoint.transform.location                    # vector waypoint - vehicle
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0 or index == self._current_index:
                # good! segment completed!
                self._current_index = index
                self._percentage_route_completed = 100.0 * (
                            float(self._accum_meters[self._current_index]) + float(dot_ve_wp)) \
                                                   / float(self._accum_meters[-1])
                self.route_record['route_percentage'] = self._percentage_route_completed

        d2target = self.route_record['route_length'] - float(self._accum_meters[-1])
        return completed, d2target



    def compute_route_statistics(self, duration, trajector_events):
            """
            Compute the current statistics by evaluating all relevant scenario criteria
            """

            target_reached = False
            score_penalty = 1.0
            score_route = 0.0
            route_infraction = 1

            self.route_record['duration'] = duration

            for event in trajector_events:
                if event[0] == TrafficEventType.COLLISION_STATIC:
                    score_penalty *= self.PENALTY_COLLISION_STATIC
                    #route_record.infractions['collisions_layout'].append(event.get_message())

                elif event[0] == TrafficEventType.COLLISION_PEDESTRIAN:
                    score_penalty *= self.PENALTY_COLLISION_PEDESTRIAN
                    #route_record.infractions['collisions_pedestrian'].append(event.get_message())

                elif event[0] == TrafficEventType.COLLISION_VEHICLE:
                    score_penalty *= self.PENALTY_COLLISION_VEHICLE
                    #route_record.infractions['collisions_vehicle'].append(event.get_message())

                elif event[0] == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                    route_infraction = (1 - (event[1]/100.0))
                    #score_penalty *= (1 - ((event[1]/100.0)-self.prev_route_infractions))
                    self.prev_route_infractions = (1 - (event[1]/100.0))
                    #route_record.infractions['outside_route_lanes'].append(event.get_message())

                elif event[0] == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                    score_penalty *= self.PENALTY_TRAFFIC_LIGHT
                    #route_record.infractions['red_light'].append(event.get_message())

                elif event[0] == TrafficEventType.ROUTE_DEVIATION:
                    #route_record.infractions['route_dev'].append(event.get_message())
                    failure = "Agent deviated from the route"

                elif event[0] == TrafficEventType.STOP_INFRACTION:
                    score_penalty *= self.PENALTY_STOP
                    #route_record.infractions['stop_infraction'].append(event.get_message())

                elif event[0] == TrafficEventType.VEHICLE_BLOCKED:
                    #route_record.infractions['vehicle_blocked'].append(event.get_message())
                    failure = "Agent got blocked"

                elif event[0] == TrafficEventType.ROUTE_COMPLETED:
                    score_route = 100.0
                    target_reached = True
                elif event[0] == TrafficEventType.ROUTE_COMPLETION:
                    if not target_reached:
                        score_route = self.route_record['route_percentage']

            # update route scores
            self.route_record['score_route'] = score_route
            self.route_record['score_penalty'] = score_penalty
            if score_route >= 0:
                self.route_record['score_composed'] = max(score_route*score_penalty*route_infraction, 0.0)
            else:
                try:
                    self.route_record['score_composed'] = score_route / (score_penalty * route_infraction)
                except ZeroDivisionError:
                    self.route_record['score_composed'] = score_route

            # update status
            if target_reached:
                self.route_record["status"] = 'Completed'
            else:
                self.route_record["status"] = 'Failed'
                # if failure:
                #     self.route_record["status"] += ' - ' + failure

            return self.route_record
