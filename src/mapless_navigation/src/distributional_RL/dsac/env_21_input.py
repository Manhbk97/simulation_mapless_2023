#!/usr/bin/env python3
from turtle import position
import rospy
import numpy as np
import math
import os
import time
import random
from math import pi, sqrt, pow, exp
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int64
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
#from repawnGoal import Respawn


def initial_pose():
	# clutter_env 
	# location = np.random.randint(3)

	# if location == 0: 
	# 	x,y =  5.2,8.3
	# elif location == 1:
	# 	x,y =  0.0,0.0
	# else:
	# 	x,y =  -22.35,-11.96
	location = np.random.randint(5)
	if location == 0: 
		x,y =  0.0, 0.0
	elif location == 1:
		x,y =  -7.81, -0.94
	elif location == 2:
		x,y =  7.54, 7.34
	elif location == 3:
		x,y =  6.97, -6.99
	else:
		x,y =  -1.24,9.16
	return x,y


def goal_def(index):

	# obstacles = [[2,0],[4,0],[3,3],[0,2],[0,4],[-2,0],[-4,0],[-3,3],[0,-2],[0,-4],[3,-3],[-3,-3]]
	# while True:
	#     distance = []
	#     print("finding proper goal position...")
	#x, y = np.random.randint(222,265)/10, np.random.randint(-65,8)/10
	#     x, y = np.random.randint(-80,80)/10, np.random.randint(-80,80)/10
	#     for i in range(len(obstacles)):
	#         distance.append(sqrt(pow(obstacles[i][0]-x,2)+pow(obstacles[i][1]-y,2)))
	#     if (min(distance) >= 1.0) and (not(x==0 and y==0)):
	#         break


	# ENV T_dynamic_human_object_env
	# obstacles = [[9,2.94],[5.4,1.0],[3.16,3.3],[5.274,-2.59],[8.21,-1.3],[11.6,-2.42],[18.16,-2.21],[14.67,0.85],[25.69,-0.79],[31.82,2.85],[9.04,2.94]]
	# # 12 obstacle
	# # obstacles = [[2,0],[4,0],[3,3],[0,2],[0,4],[-2,0],[-4,0],[-3,3],[0,-2],[0,-4],[3,-3],[-3,-3]]  
	# obstacles = [[-3.27,5.75],[-1.38,4.88],[-0.25,3.74],[1.7,3.3],[4.2,2.74],[7.22,4.59],[6.26,5.98],[5.5,5.98],[4.5,5.98],[3.5,5.98],[2.8,5.98], ## line 1
	#              [-6.39,-0.24],[-3.26,0.21],[3.15,-1.51],[6.43,-1.04],  ## line 2 has 4 point 
	#              [-6.62,-4.86],[-3.29,-3.2],[-1.77,-2.2],[4.94,-3.43],
	#              [-4.97,-6.47],[-3.97,-6.49],[-0.79,-6.23],[6.66,-7.09]]

	# ## test sequence without reset gazebo
	# obstacles = [[-5.27,3.54],[-1.26,3.48],[1.73,3.33],[4.83,2.08],[4.2,2.74],[7.22,4.59],[6.26,5.98],[5.5,5.98],[4.5,5.98],[3.5,5.98],[2.8,5.98], ## line 1
	#              [-6.39,-0.24],[-3.26,0.21],[3.15,-1.51],  ## testing
	#              [-6.62,-4.86],[-3.29,-3.2],[-0.238,-2.65],[2.96,-3.92],
	#              [-4.97,-6.47],[-3.97,-6.49],[-0.79,-6.23],[6.66,-7.09]]
	# while True:
	#     distance = []
	#     print("finding proper goal position...")
	#     # x, y = np.random.randint(222,265)/10, np.random.randint(-65,8)/10 # StR env
	#     # x, y = np.random.randint(0,180)/10, np.random.randint(-40,43)/10
	#     # x, y = np.random.randint(-80,80)/10, np.random.randint(-80,80)/10
	#     x, y = np.random.randint(-80,80)/10, np.random.randint(-70,70)/10
	#     for i in range(len(obstacles)):
	#         distance.append(sqrt(pow(obstacles[i][0]-x,2)+pow(obstacles[i][1]-y,2)))
	#     if (min(distance) >= 2.5) and (not(x==0 and y==0)):
	#         break
	#ICM env
	# location = np.random.randint(4)
	# # location = index
	# if location == 0: 
	# 	x,y =  14.6, 1.0
	# # 	x,y = np.random.randint(-0,17), 14.8
	# elif location == 1:
	# 	x,y =  0.1, 5.6
	# # 	x,y = 0.0, np.random.randint(5,10)
	# elif location == 2: 
	# 	x,y =  1.5, 14.3
	# # 	# x,y = np.random.randint(0,16), 10.0s
	# # 	x,y = np.random.randint(-0,17), 14.8
	# # elif location == 3: 
	# # 	x,y =  16.8, 14.8
	# 	# x,y = 16.5, np.random.randint(0,10)
	# else:
	# 	x,y =  1.5, 14.3
	# x,y = 0.0,-6.0

	#maze_rectangle
	# location = np.random.randint(5)
	# # location = index
	# if location == 0: 
	# 	# x,y =  8.0, 0.0
	# 	x,y = 8.0, np.random.randint(-90,70)/10
	# elif location == 1:
	# 	# x,y =  0.0, -7.0
	# 	x,y =  np.random.randint(-80,80)/10, -9.0
	# elif location == 2: 
	# 	# x,y =  0.0, 7.0
	# 	x,y = np.random.randint(-80,80)/10, 7.0
	# # 	x,y = np.random.randint(-0,17), 14.8
	# elif location == 3: 
	# 	# x,y = -4.0, 0.0
	# 	x,y = -8.0, np.random.randint(-90,70)/10
	# else:
	# 	x,y =  8.0, np.random.randint(-90,70)/10

#crowed_env PPO , 12obstacle env 
	# location = np.random.randint(4)
	# if location == 0: 
	#     x, y =  8.0, np.random.randint(-75,75)/10
	#     # x, y =  7.0, 0
	# elif location == 1:
	#     x,y = np.random.randint(-75,75)/10, 7.5
	# elif location ==2: 
	#     x,y = np.random.randint(-70,75)/10, -7.5
	# else:
	#     x, y =  -7.5, np.random.randint(-75,75)/10
	# x, y =  7.0, 0.0
#####human_crowed_dynamic.world#### dynamic reward env
	# location = np.random.randint(8)
	# if location == 0: 
	#     x, y =  np.random.randint(15,19), -1.9
	# elif location == 1:
	#     x,y = np.random.randint(-20,0)/10,  np.random.randint(-11,-9)  
	# elif location ==2: 
	#     x, y =  -8.0, np.random.randint(-60,70)/10
	#     if y >=-6 and y <=7:
	#         x = -12
	# elif location==3:
	#     x,y =np.random.randint(-30,10)/10,-8.0
	#     if x >=-3 and x <=1:
	#         y = -11
	# elif location == 4:
	#     x,y =np.random.randint(15,19), -1.6
	# elif location ==5: 
	#     # x,y = 0.0,3.0
	#     x,y =   np.random.randint(-120,6)/10, 13
	# elif location==6:
	#     x,y =-12, np.random.randint(3,64)/10 
	# elif location==7:
	#     x,y = 13, np.random.randint(30,80)/10
	# else:
	#     x,y = 8.0, np.random.randint(-80,80)/10

#### simple dynamic03 env #####
	# location = np.random.randint(5)
	# if location == 0: 
	#     x,y = 15.5,0

	# elif location == 1:
	#     x,y = 15.0, -1.0
	# elif location == 2:
	#     x,y = 15.0, -2.0
	# elif location == 3:
	#     x,y = 15.0, 2.0
	# else: 
	#     x,y = 15.0, 2.0
#complex_env02 training 

	# if index ==0:
	# 	location = np.random.randint(2)
	# 	if location == 0: 
	# 		x, y =  1.27,-2.59
	# 	else:
	# 		x,y = 1.14, 1.58
	# elif index == 100:
	# 	location = np.random.randint(4)
	# 	if location == 0:  
	# 		x, y =  7.5, np.random.randint(-80,80)/10
	# 	elif location == 1: 
	# 		x,y = np.random.randint(-110,95)/10, 12.6
	# 	elif location ==2: 
	# 		# x,y = 0.0,3.0
	# 		x,y = np.random.randint(-80,80)/10, -7.5
	# 	elif location ==3: 
	# 		# x,y = 0.0,3.0
	# 		x,y = np.random.randint(75,150)/10, -0.8 
	# 	else:
	# 		x, y =  -7.9, np.random.randint(-80,80)/10
	# else:
	# 	location = np.random.randint(6)
	# 	if location == 0: 
	# 		x, y =  7.5, np.random.randint(-80,80)/10
	# 		if y >=-3.38 and y <=1:
	# 			x,y = 15,-1.9
	# 	elif location == 1:
	# 		# x,y = 7.0,-1.0
	# 		# x,y = 39,-2.0 
	# 		x,y = np.random.randint(-110,95)/10, 12.6
	# 		if x >= -11 and x <= 2.3: 
	# 			y = 12.8
	# 	elif location ==2: 
	# 		# x,y = 0.0,3.0
	# 		x,y = np.random.randint(-80,80)/10, -7.5
	# 	elif location ==3: 
	# 		# x,y = 0.0,3.0
	# 		x,y = np.random.randint(75,150)/10, -0.8  
	# 	elif location ==4:
	# 		# x,y = 0.0,8.8
	# 		# x,y = 37,0.0 
	# 		x, y =  -7.9, np.random.randint(-80,80)/10
	# 		if y <= 8.0 and y >=-6:
	# 			x = -12
	# 	else: 
	# 		# x,y = 0.0,3.0
	# 		x,y = 11.9, np.random.randint(-70,90)/10  		 
#complex_env02 training 
	# location = np.random.randint(6)
	# if location == 0: 
	# 	# x,y = 8.0,9.0
	# 	# x,y = 38,3.0 
	# 	x, y =  7.5, np.random.randint(-80,80)/10
	# 	if y >=-3.38 and y <=1:
	# 		x,y = 15,-1.9
	# elif location == 1:
	# 	# x,y = 7.0,-1.0
	# 	# x,y = 39,-2.0 
	# 	x,y = np.random.randint(-110,95)/10, 12.6
	# 	if x >= -11 and x <= 2.3: 
	# 		y = 12.8
	# elif location ==2: 
	# 	# x,y = 0.0,3.0
	# 	x,y = np.random.randint(-80,80)/10, -7.5
	# elif location ==3: 
	# 	# x,y = 0.0,3.0
	# 	x,y = np.random.randint(75,150)/10, -0.8  
	# elif location ==4:
	# 	# x,y = 0.0,8.8
	# 	# x,y = 37,0.0 
	# 	x, y =  -7.9, np.random.randint(-80,80)/10
	# 	if y <= 8.0 and y >=-6:
	# 		x = -12
	# else: 
	# 	# x,y = 0.0,3.0
	# 	x,y = 11.9, np.random.randint(-70,90)/10     
	# x,y = 26.2,0.0    
#####human_crowed_dynamic.world####
	# location = np.random.randint(8)
	# if location == 0: 
	#     x, y =  8.0, np.random.randint(-80,80)/10
	#     if y >=-2.5 and y < 0.5:
	#         x = np.random.randint(15,19)
	# elif location == 1:
	#     x,y = np.random.randint(-80,80)/10, 8.0
	#     if x >=-2 and x <=0:
	#         y = np.random.randint(-13,-11)
	# elif location ==2: 
	#     x, y =  -8.0, np.random.randint(-80,80)/10
	#     if y >=-6 and y <=6:
	#         x = -12
	# elif location==3:
	#     x,y =np.random.randint(-80,80)/10,-8.0
	#     if x >=0 and x <=1:
	#         y = -13
	# elif location == 4:
	#     x,y =np.random.randint(15,19), -1.6
	# elif location ==5: 
	#     # x,y = 0.0,3.0
	#     x, y =   np.random.randint(-120,6)/10, 13
	# elif location==6:
	#     x,y =-12, np.random.randint(3,64)/10 
	# else:
	#     x,y =12.9,8.5
# # rastech_env
	# location = np.random.randint(5)
	location = index
	if location == 0: 
		x,y =  6.0, 4.5
	# 	x,y = np.random.randint(-0,17), 14.8
	elif location == 1:
		x,y =  6.0, 10.0
	# 	x,y = 0.0, np.random.randint(5,10)
	elif location == 2: 
		x,y =  15.0, 10.0
	# 	# x,y = np.random.randint(0,16), 10.0s
	# 	x,y = np.random.randint(-0,17), 14.8
	elif location == 3: 
		x,y =  15.0, -4.0
		# x,y = 16.5, np.random.randint(0,10)
	elif location == 4: 
		x,y =  6.0, 4.5
		# x,y = 16.5, np.random.randint(0,10)
	else:
		x,y =  0.0,0.0

# clutter_env022
	# location = np.random.randint(8)
	# # location = index
	# if location == 0: 
	# 	x,y =  -6.86, 4.17
	# # 	x,y = np.random.randint(-0,17), 14.8
	# elif location == 1:
	# 	x,y =  0.9, 4.0
	# # 	x,y = 0.0, np.random.randint(5,10)
	# elif location == 2: 
	# 	x,y =  9.02, 4.21
	# elif location == 3: 
	# 	x,y =  17.7,4.35
	# 	# x,y = 16.5, np.random.randint(0,10)
	# elif location == 4: 
	# 	# x,y =  7.61, 10.69
	# 	x,y = 7.3, np.random.randint(104,124)/10
	# elif location == 5: 
	# 	x,y =  np.random.randint(-14,10)/10, np.random.randint(141,175)/10
	# elif location == 6: 
	# 	# x,y =  7.61, 10.69
	# 	x,y = np.random.randint(181,212)/10, np.random.randint(111,126)/10
	# else:
	# 	x,y =  27.6, 4.11


#####human_crowed_dynamic.world####
	# location = np.random.randint(8)
	# if location == 0: 
	# 	x, y =  8.0, np.random.randint(-150,190)/10
	# 	if y >=-2.5 and y < 0.5:
	# 		x = np.random.randint(15,19)
	# elif location == 1:
	# 	x,y = np.random.randint(-20,0.0)/10, 8.0
	# 	if x >=-2 and x <=0:
	# 		y = np.random.randint(-13,-11)
	# elif location ==2: 
	# 	x, y =  -8.0, np.random.randint(-60,60)/10
	# 	if y >=-6 and y <=6:
	# 		x = -12
	# elif location==3:
	# 	x,y =np.random.randint(0,10)/10,-8.0
	# 	if x >=0 and x <=1:
	# 		y = -13
	# elif location == 4:
	# 	x,y =np.random.randint(15,19), -1.6
	# elif location ==5: 
	# 	# x,y = 0.0,3.0
	# 	x, y =   np.random.randint(-120,6)/10, 13
	# elif location==6:
	# 	x,y =-12, np.random.randint(3,64)/10 
	# elif location==7:
	# 	x,y =13, np.random.randint(30,80)/10
	# else:
	# 	x,y =12.9,8.5
#T_human_env
	# location = np.random.randint(4)
	# if location==0:
	#     x,y =np.random.randint(6,12),11.0
	# elif location==1:
	#     x,y = np.random.randint(6,12),-11.0
	# else:
	#     # x,y = 0.0,8.8
	#     # x,y = 37,0.0 
	#     x, y =  18.0, np.random.randint(-30,30)/10
#### simple dynamic03 env #####
	# location = np.random.randint(5)
	# if location == 0: 
	#     x,y = 15.5,0

	# elif location == 1:
	#     x,y = 15.0, -1.0
	# elif location == 2:
	#     x,y = 15.0, -2.0
	# elif location == 3:
	#     x,y = 15.0, 2.0
	# else: 
	### 12 obstacles 
	# location = np.random.randint(4)
	# if location == 0: 
	# 	x, y =  7.0, np.random.randint(-7,7)
	# elif location == 1:
	# 	x,y = np.random.randint(-7,7), 7.4
	# elif location ==2: 
	# 	x, y =  -7.0, np.random.randint(-7,7)

	# elif location==3:
	# 	x,y =np.random.randint(-7,7), -7.4

	return x, y 


class Env():
	def __init__(self):
	
		#self.respawn_goal = Respawn()
		self.delete = False
		self.goal_x, self.goal_y = 0.0,0.0# goal_def()
		#self.goal_x, self.goal_y = self.RespawnGoal.goal_def(self.delete)
		self.initGoal = True
		self.heading = 0
		self.initGoal = True
		self.get_goalbox = False
		self.position = Pose()
		self.line_error = 0
		
		self.current_obstacle_angle = 0
		self.old_obstacle_angle = 0
		self.current_obstacle_min_range = 0
		self.old_obstacle_min_range = 0
		self.t = 0
		self.old_t = time.time()
		self.dt = 0
		self.index = 0
		self.time_step = 0      
		self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
		self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
		self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
		self.cumulated_reward= 0.0
		# publisher
		self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
		self._marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=5)
		# subcriber
		self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
		self.sub_scan = rospy.Subscriber('laser/scan', LaserScan, self.getScanData) 
		self.scan = LaserScan()
		
	def getScanData(self, scan_msg):
		self.scan = scan_msg
	  

	def getGoalDistace(self):
		goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 5)
		#goal_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
		return goal_distance

	def getOdometry(self, odom):
		self.position = odom.pose.pose.position
		orientation = odom.pose.pose.orientation
		orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
		_, _, yaw = euler_from_quaternion(orientation_list)
		#print("position:" + str(self.position.y))
		goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

		heading = goal_angle - yaw
		if heading > pi:
			heading -= 2 * pi

		elif heading < -pi:
			heading += 2 * pi

		#self.heading = round(heading, 2)
		self.heading = heading



	def getState(self, scan):
		scan_range = []        
		heading = self.heading
		min_range = 0.4
		max_range = 3.0
		done = False
		self.goal = False
		arrival = False
		#target_size = 1.0        
		target_size = 0.2

		scan_array_np = np.array(scan.ranges)
		min_pooled_array = scan_array_np.reshape(21, -1).min(axis=1)
		min_pooled_list = min_pooled_array.tolist()

		for i,item in enumerate(min_pooled_list):
			if item >= max_range or item ==float('Inf') or np.isinf(item):
				scan_range.append(max_range - min_range)
			elif np.isnan(item):
				scan_range.append(0.0)
			else:
				scan_range.append((item - min_range))
			if min_range > item > 0:
				done = True
				print("done_eps:" + str(done))    
				self.move_base(0.0, 0.0, epsilon=0.05, update_rate=10)
			# if min_range+0.1 > item > 0:
			# 	# done = True
			# 	print("done_eps:" + str(done))    
			# 	self.move_base(-0.15, 0.0, epsilon=0.05, update_rate=10)

		obstacle_min_range = min(scan_range) + min_range
		current_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
		# ''' Auxiliary task
		# a = 11 + round(self.heading/0.0305) 
		# print("index a", a)


		# print("dis ", scan.ranges[a])
		# if min(scan.ranges[:])_ < current_distance < 9.0:
		#     current_dis = 0.0
		#     self.agent = 1

		# max_value1 = np.mean(scan_range[0:10])
		# max_value2 = np.mean(scan_range[10:20])
		# if max_value1 >= max_value2 :
		# 	max_value = max(scan_range[0:5])
		# 	max_index = scan_range.index(max_value)
		# else: 
		# 	max_value = max(scan_range[15:20])
		# 	max_index = scan_range.index(max_value)			

		# if min(scan_range[5:15])<0.7 or min(scan_range[0:20])<=0.1:
		# 	if max_value1 > max_value2 and min(scan_range[0:3]) >= min(scan_range[17:20]): #max_value == max_value1:
		# 		differ_angle = -(11-max_index)*0.0997 ## ben phai 
		# 		self.move_base(0.0, -0.15)
		# 		heading = differ_angle
		# 		print("change heading right:", differ_angle)
		# 	elif max_value1 < max_value2 and min(scan_range[0:3]) <= min(scan_range[17:20]):
		# 		differ_angle = (11+max_index)*0.0997  ## ben trai 
		# 		self.move_base(0.0, 0.15)
		# 		print("change heading left:", differ_angle)
		# 		heading = differ_angle
		# 	else:
		# 		time.sleep(0.1)
		# 		self.move_base(-0.15, 0.0)
		# 	# heading = differ_angle

		# if max(scan_range[5:15])<1.3 and  scan.ranges[11] < current_distance:
		# 	print("wrong direction")
		#     heading = -1.57 + heading
		# '''	




		#print("current_distance:",current_distance)
		if current_distance < target_size:
			self.get_goalbox = True
			arrival = True
			done = True

		# if scan.ranges[a] < current_distance < 9.0:
		#     current_distance = 0.0
		#     print("wrong direction")
		# if self.position.x >=21.0 or self.position.x < -16.0 or self.position.y >=16.0 or self.position.y < -13.0:
		# 	print("outside of env")
		# 	done = True 
		# print("time step",self.time_step)
		if self.time_step >= 1200:
			done = True
		# boundary = math.sqrt(pow(self.position.x,2) + pow(self.position.y,2))
		# if boundary > 10:
		#     #done = False
		#     done = True
		
		#print("laser scan:",scan_range)
		constant = (max_range-min_range)*np.ones(21)
		rescale = np.divide(scan_range,constant)
		laser_input = np.round(rescale, decimals=2)
		# print("current distance:" + str(current_distance))

		relationship = [round(heading,3), round(current_distance/10,2), round(obstacle_min_range/10.0,2)]
		#state= rescale_laser + [heading, round(current_distance/15,2), obstacle_min_range/15]
		# t=0
		# for i in laser_input:
		#     if i < 1.0  and i>=0.03:
		#         laser_input[t] = laser_input[t] - 0.04
		#     t += 1
		# print("  laser input of neural:{}".format(laser_input))

		state = np.concatenate((laser_input, relationship), axis=0).tolist()
		#state = [heading, current_distance, obstacle_min_range, obstacle_angle]
		# print(" state", (state))

		return state, done, arrival 

	def show_marker_in_rviz(self, point):     #Ve duong di cua robot
		marker = Marker(
			type=Marker.ARROW,
			id=0,
			lifetime=rospy.Duration(350),
			pose=Pose(point, Quaternion(0, -1.0, 0, 1)),
			scale=Vector3(1.0, 0.2, 0.2),
			header=Header(frame_id='odom'),
			color=ColorRGBA(1.0, 0.0, 0.0, 0.8))
		self._marker_publisher.publish(marker)    

	def setReward(self, state, done, action):
		yaw_reward = []
		obstacle_min_range = state[23]*10
		current_distance = state[22]*10
		heading = state[21]      
		if not done: 
			## adjust## 
			distance_difference = current_distance - self.previous_distance_from_desination
			if distance_difference <= 0.0:
				# rospy.logwarn("DECREASE IN DISTANCE GOOD")
				# distance_difference *= 1.5
				distance_difference += 0.1
			else:
				# rospy.logerr("INCREASE IN DISTANCE BAD")
				# distance_difference *= 1.5    
				distance_difference -= 0.1  

			rotations_cos_sum = math.cos(heading)
			reward = (-2.0 * distance_difference + (math.pi - math.fabs(heading)) / math.pi + 2*rotations_cos_sum)
			# if obstacle_min_range < 0.8:
			# 	reward += -10

			# if obstacle_min_range < 1.0:
			# 	reward += -30*(1-(obstacle_min_range/1.0))

			# if abs(action[0]) >= 0.8: 
			#     #angular_reward= -0.5*exp(0.01*self.time_step) # time 500
			#     angular_reward =-0.1*exp(0.01*self.time_step)  # time 600
			#     reward+= angular_reward
			#     print("angular Reward: " + str(angular_reward))

			# if state[10] == np.amax(state[:-3]):
			#     reward += 2
			# if self.time_step >= 1500:
			#     reward = -500
			#     self._episode_done= True
			#     self.time_step = 0
			## reward dynamic 
			# x = min(state[37:69])*(6-0.7) + 0.7
			# print("distance_dynamic", x)
			# print("reward dynamic", exp(x))

			# if 1.2 <= x <=1.8:
			#     reward = reward - exp(x)  
			# print("distance and orientation Reward: " + str(reward))

		else:
			if self.get_goalbox:
				rospy.loginfo("Goal!!")
				reward = 600
				self._episode_done= True
				# self.goal_x, self.goal_y = goal_def() #4.5, 4.5 # goal_def()
				# print("NEXT GOAL : ", self.goal_x, self.goal_y )
				# desired_point = Point()
				# desired_point.x = float(self.goal_x)
				# desired_point.y = float(self.goal_y) 
				# desired_point.z = 0.0 
				# self.show_marker_in_rviz(desired_point)  
				self.goal_distance = self.getGoalDistace()
				self.get_goalbox = False
				self.time_step = 0
				self.index = self.index + 1 
				#time.sleep(0.2)
			else: 
				rospy.loginfo("Collision!")
				reward = -500
				self.time_step = 0
				self.pub_cmd_vel.publish(Twist())
				self.move_base(0.0, 0.0, epsilon=0.05, update_rate=10)
				self._episode_done= True

		#self.cumulated_reward += reward
		#print("cumulate reward:" + str(self.cumulated_reward))
		# print("total Reward:%0.3f"%reward, "\n")
		# print("Reward : ", reward)
		self.previous_distance_from_desination = current_distance
		return reward

	def step(self, action):
		
		self.time_step += 1

		self._set_action(action)
		#time.sleep(0.2)
		#print("EP:", ep, " Step:", t, " Goal_x:",self.goal_x, "  Goal_y:",self.goal_y)
		
		state, done, arrival = self.getState(self.scan)
		reward = self.setReward(state, done, action)
		
		return np.asarray(state), reward, done, arrival 

	def _set_action(self, action):
		"""
		This set action will Set the linear and angular speed of the turtlebot2
		based on the action number given.
		:param action: The action integer that set s what movement to do next.
		"""
		rospy.logdebug("Start Set Action ==>"+str(action))
		# We convert the actions to speed movements to send to the parent class

		# if action == 0: #FORWARD
		#     linear_speed = 0.3
		#     angular_speed = 1.0
		#     #self.last_action = "FORWARDS"
		# elif action == 1:
		#     linear_speed = 0.5
		#     angular_speed = 0.5
		#     #self.last_action = "TURN_LEFT"
		# elif action == 2:
		#     linear_speed = 0.75
		#     angular_speed = 0.25
		#     #self.last_action = "TURN_RIGHT"
		# elif action == 3:
		#     linear_speed = 1.0
		#     angular_speed = 0.0
		#     self.last_action = "FORWARDS"
		# elif action == 4:
		#     linear_speed = 0.75
		#     angular_speed = -0.25
		# elif action == 5:
		#     linear_speed = 0.5
		#     angular_speed = -0.5          
		# elif action == 6:
		#     linear_speed = 0.3
		#     angular_speed = -1.0

		# Continuous Action
		self.time_step += 1
		linear_speed = action[1]
		angular_speed = action[0]
		# We tell TurtleBot2 the linear and angular speed to set to execute
		self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
		
		rospy.logdebug("END Set Action ==>"+str(action))

	def reset(self):
		
		self.time_step = 0
		self._episode_done = False
		# if arrv == False:
		rospy.wait_for_service('gazebo/reset_simulation')
		time.sleep(0.1)
		try:
			self.reset_proxy()
			# time.sleep(0.2)
		except (rospy.ServiceException) as e:
			print("gazebo/reset_simulation service call failed")
		# if arrv== True:
			
		while True:
			if (len(self.scan.ranges) > 0):
				break

		# srv_client_set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		## set robot position
		# model_state = ModelState()
		# model_state.model_name = 'nana_model_0602_1729'
		# model_state.pose.position.x, model_state.pose.position.y = initial_pose()
		# resp = srv_client_set_model_state(model_state)

		# if self.initGoal:
		#     self.goal_x, self.goal_y = self.respawn_goal.getPosition()
		#     self.initGoal = False
		#error_data = self.line_error
		self.goal_x, self.goal_y = goal_def(self.index) #4.5, 4.5 # goal_def()
		# print("NEXT GOAL : ", self.goal_x, self.goal_y )
		desired_point = Point()
		desired_point.x = float(self.goal_x)
		desired_point.y = float(self.goal_y) 
		desired_point.z = 0.0 
		self.show_marker_in_rviz(desired_point)  
		self.goal_distance = self.getGoalDistace()
		state, done, _ = self.getState(self.scan)

		self.previous_distance_from_desination = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
		self.previous_rotation_to_goal_diff = self.heading
	 
		return np.asarray(state)

	def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10 ):
		cmd_vel_value = Twist()
		cmd_vel_value.linear.x = linear_speed
		cmd_vel_value.angular.z = angular_speed
		rospy.logdebug("Base Twist Cmd>>" + str(cmd_vel_value))
		self.pub_cmd_vel.publish(cmd_vel_value)
		time.sleep(0.1)
	
