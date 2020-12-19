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

        return [self.rgb_cam,0,0,0,0,0,0,0,0]

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
        self.car_agent.apply_control(carla.VehicleControl(throttle=action[0][0],steer=action[0][1]))
        time.sleep(1)
        velocity = self.car_agent.get_velocity()

        #get state information
        closest_index = self.route_kdtree.query([[self.car_agent.get_location().x,self.car_agent.get_location().y,self.car_agent.get_location().z]],k=1)[1][0][0]
        self.followed_waypoints.append(self.route_waypoints[closest_index])
        command_encoded = self.command2onehot.get(str(self.route_commands[closest_index]))

        print (command_encoded)
        print (str(self.route_commands[closest_index]))

        #d2target = np.sqrt(np.power(self.car_agent.get_location().x-self.target.location.x,2)+np.power(self.car_agent.get_location().y-self.target.location.y,2)+np.power(self.car_agent.get_location().z-self.target.location.z,2))
        d2target = self.statistics_manager.route_record["route_length"] - self.statistics_manager.compute_route_length(self.followed_waypoints)
        self.d_completed = d2target
        #velocity_kmh = int(3.6*np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2)))
        velocity_mag = np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2))
        self.cur_velocity = velocity_mag

        state = [self.rgb_cam,velocity_mag,d2target]
        state.extend(command_encoded)

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
        return state, reward, done, [self.statistics_manager.route_record['score_route']]

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

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class PPO_Agent(nn.Module):
    def __init__(self, linear_state_dim, action_dim, action_std):
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
                nn.Tanh()
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
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def actor(self,frame,mes):
        frame = frame.to(device)
        mes = mes.to(device)
        mes = mes.unsqueeze(0)
        vec = self.actorConv(frame)
        X = torch.cat((vec,mes),1)
        return self.actorLin(X)

    def critic(self,frame,mes):
        frame = frame.to(device)
        mes = mes.to(device)
        mes = mes.unsqueeze(0)
        vec = self.criticConv(frame)
        X = torch.cat((vec,mes),1)
        return self.criticLin(X)

    def choose_action(self,frame,mes):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #state = torch.FloatTensor(state).to(device)
        mean = self.actor(frame,mes)
        cov_matrix = torch.diag(self.action_var)
        gauss_dist = MultivariateNormal(mean,cov_matrix)
        action = gauss_dist.sample()
        action_log_prob = gauss_dist.log_prob(action)
        return action, action_log_prob

    def get_training_params(self,frame,mes, action):
        #state = state.to(device)
        #action = action.to(device)

        frame = torch.stack(frame)
        mes = torch.stack(mes)
        if len(list(frame.size())) > 4:
            frame = torch.squeeze(frame)
            mes = torch.squeeze(mes)
        elif len(list(frame.size())) == 3:
            frame = frame.unsqueeze(0)
            mes = mes.unsqueeze(0)
        action = torch.stack(action)

        mean = self.actor(frame,mes)
        action_expanded = self.action_var.expand_as(mean)
        cov_matrix = torch.diag_embed(action_expanded).to(device)

        gauss_dist = MultivariateNormal(mean,cov_matrix)
        action_log_prob = gauss_dist.log_prob(action).to(device)
        entropy = gauss_dist.entropy().to(device)
        state_value = torch.squeeze(self.critic(frame,mes)).to(device)
        return action_log_prob, state_value, entropy

# def format_(state):
#     frame = torch.FloatTensor(state[0])
#     h,w,c = frame.shape
#     frame = frame.unsqueeze(0).view(1, c, h, w)
#
#     measurements = torch.FloatTensor(state[1:])
#     return [frame,measurements]

def format_frame(frame):
    frame = torch.FloatTensor(frame)
    h,w,c = frame.shape
    frame = frame.unsqueeze(0).view(1, c, h, w)
    return frame

def format_mes(mes):
    measurements = torch.FloatTensor(mes)
    return mes

def train_PPO(host,world_port):
    wandb.init(project='PPO_Carla_Navigation')

    env = CarEnv(host,world_port)
    n_iters = 1000
    n_epochs = 50
    max_steps = 2000
    gamma = 0.9
    lr = 0.0001
    clip_val = 0.2
    avg_t = 0
    avg_r = 0

    config = wandb.config
    config.learning_rate = lr


    n_states = 8

    #currently the action array will be [throttle, steer]
    n_actions = 2

    action_std = 0.5 #maybe try some other values for this
    #init models
    policy = PPO_Agent(n_states, n_actions, action_std).to(device)

    #policy.load_state_dict(torch.load("policy_state_dictionary.pt"))
    #FileNotFoundError

    optimizer = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    prev_policy = PPO_Agent(n_states, n_actions, action_std).to(device)
    #TODO: idk if I should be setting each policies initial states to the same thing or not
    prev_policy.load_state_dict(policy.state_dict())


    wandb.watch(prev_policy)

    for iters in range (n_iters):
        s = env.reset()
        t = 0
        episode_reward = 0
        done = False
        rewards = []
        states = []
        actions = []
        actions_log_probs = []
        states_p = []
        while not done:
            #env.render()
            a, a_log_prob = prev_policy.choose_action(format_frame(s[0]), format_mes(s[1:]))
            s_prime, reward, done, info = env.step(a.detach().tolist())

            states.append(s)
            actions.append(a)
            actions_log_probs.append(a_log_prob)
            rewards.append(reward)
            states_p.append(s_prime)

            s = s_prime
            t+=1
            episode_reward+=reward

        print ("Episode reward: " + str(episode_reward))
        print ("Percept compleyed: " + str(info[0]))
        avg_t+=t
        avg_r+=episode_reward
        env.cleanup()

        #f = open("output.txt","a")
        #f.write("ran episode with reward " + str(episode_reward))
        #f.close()

        wandb.log({"episode_reward": episode_reward})
        wandb.log({"percent_completed": info[0]})
        if (len(states) == 1):
            continue

        #format reward
        for i in range (len(rewards)):
            rewards[len(rewards)-1] = rewards[len(rewards)-1]*np.power(gamma,i)
        rewards = torch.tensor(rewards).to(device)
        rewards= (rewards-rewards.mean())/rewards.std()

        actions_log_probs = torch.FloatTensor(actions_log_probs).to(device)
        #train PPO
        for i in range(n_epochs):
            current_action_log_probs, state_values, entropies = policy.get_training_params(format_frame(state[0]),format_mes(state[1:]),actions)

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
            prev_policy.load_state_dict(policy.state_dict())
            torch.save(policy.state_dict(),"policy_state_dictionary.pt")


def main(n_vehicles,host,world_port,tm_port):
    train_PPO(host,world_port)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world_port', type=int, required=True)
    parser.add_argument('--tm_port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)

    main(**vars(parser.parse_args()))
