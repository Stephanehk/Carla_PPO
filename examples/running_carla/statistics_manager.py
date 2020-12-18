import carla
import numpy as np

from traffic_events import TrafficEventType

class StatisticManager:

    PENALTY_COLLISION_PEDESTRIAN = 0.50
    PENALTY_COLLISION_VEHICLE = 0.60
    PENALTY_COLLISION_STATIC = 0.65
    PENALTY_TRAFFIC_LIGHT = 0.70
    PENALTY_STOP = 0.80

    def __init__(self,trajectory):
        self.route_record = {}
        self.route_record['route_length'] = compute_route_length(trajectory)
        self.prev_score = 0

    def compute_route_length(self, trajectory):
        route_length = 0.0
        previous_transform = None
        for transform in trajectory:
            if previous_transform:
                dist = math.sqrt((transform.location.x-previous_transform.location.x)*(transform.location.x-previous_transform.location.x) +
                                 (transform.location.y-previous_transform.location.y)*(transform.location.y-previous_transform.location.y) +
                                 (transform.location.z-previous_transform.location.z)*(transform.location.z-previous_transform.location.z))
                route_length += dist
            previous_transform = transform

        return route_length

    def compute_route_statistics(self,trajectory, duration):
            """
            Compute the current statistics by evaluating all relevant scenario criteria
            """

            target_reached = False
            score_penalty = 1.0
            score_route = 0.0

            route_record.meta['duration'] = duration
            route_record.meta['route_length'] = compute_route_length(trajectory)

            for node in _master_scenario.get_criteria():
                if node.list_traffic_events:
                    # analyze all traffic events
                    for event in node.list_traffic_events:
                        if event.get_type() == TrafficEventType.COLLISION_STATIC:
                            score_penalty *= PENALTY_COLLISION_STATIC
                            route_record.infractions['collisions_layout'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                            score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                            route_record.infractions['collisions_pedestrian'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                            score_penalty *= PENALTY_COLLISION_VEHICLE
                            route_record.infractions['collisions_vehicle'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                            score_penalty *= (1 - event.get_dict()['percentage'] / 100)
                            route_record.infractions['outside_route_lanes'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                            score_penalty *= PENALTY_TRAFFIC_LIGHT
                            route_record.infractions['red_light'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                            route_record.infractions['route_dev'].append(event.get_message())
                            failure = "Agent deviated from the route"

                        elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                            score_penalty *= PENALTY_STOP
                            route_record.infractions['stop_infraction'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                            route_record.infractions['vehicle_blocked'].append(event.get_message())
                            failure = "Agent got blocked"

                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                            score_route = 100.0
                            target_reached = True
                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                            if not target_reached:
                                if event.get_dict():
                                    score_route = event.get_dict()['route_completed']
                                else:
                                    score_route = 0

            # update route scores
            route_record.scores['score_route'] = score_route
            route_record.scores['score_penalty'] = score_penalty
            route_record.scores['score_composed'] = max(score_route*score_penalty, 0.0)

            # update status
            if target_reached:
                route_record.status = 'Completed'
            else:
                route_record.status = 'Failed'
                if failure:
                    route_record.status += ' - ' + failure

            return route_record


    def compute_route_statistics(self,duration, trajector_events):
            """
            Compute the current statistics by evaluating all relevant scenario criteria
            """

            target_reached = False
            score_penalty = 1.0
            score_route = 0.0

            sefl.route_record['duration'] = duration

            for event in trajector_events:
                if event[0] == TrafficEventType.COLLISION_STATIC:
                    score_penalty *= PENALTY_COLLISION_STATIC
                    #route_record.infractions['collisions_layout'].append(event.get_message())

                elif event[0] == TrafficEventType.COLLISION_PEDESTRIAN:
                    score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                    #route_record.infractions['collisions_pedestrian'].append(event.get_message())

                elif event[0] == TrafficEventType.COLLISION_VEHICLE:
                    score_penalty *= PENALTY_COLLISION_VEHICLE
                    #route_record.infractions['collisions_vehicle'].append(event.get_message())

                elif event[0] == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                    pass
                    #score_penalty *= (1 - percentage bla bla)
                    #route_record.infractions['outside_route_lanes'].append(event.get_message())

                elif event[0] == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                    score_penalty *= PENALTY_TRAFFIC_LIGHT
                    #route_record.infractions['red_light'].append(event.get_message())

                elif event[0] == TrafficEventType.ROUTE_DEVIATION:
                    #route_record.infractions['route_dev'].append(event.get_message())
                    failure = "Agent deviated from the route"

                elif event[0] == TrafficEventType.STOP_INFRACTION:
                    score_penalty *= PENALTY_STOP
                    #route_record.infractions['stop_infraction'].append(event.get_message())

                elif event[0] == TrafficEventType.VEHICLE_BLOCKED:
                    #route_record.infractions['vehicle_blocked'].append(event.get_message())
                    failure = "Agent got blocked"

                elif event[0] == TrafficEventType.ROUTE_COMPLETED:
                    score_route = 100.0
                    target_reached = True
                elif event[0] == TrafficEventType.ROUTE_COMPLETION:
                    if not target_reached:
                        score_route = event[1]/self.route_record['route_length']
                        # if event.get_dict():
                        #     score_route = event.get_dict()['route_completed']
                        # else:
                        #     score_route = 0

            # update route scores
            self.route_record.scores['score_route'] = score_route
            self.route_record.scores['score_penalty'] = score_penalty
            self.route_record.scores['score_composed'] = max(score_route*score_penalty, 0.0)

            # update status
            if target_reached:
                self.route_record["status"] = 'Completed'
            else:
                self.route_record["status"] = 'Failed'
                # if failure:
                #     self.route_record["status"] += ' - ' + failure

            return route_record
