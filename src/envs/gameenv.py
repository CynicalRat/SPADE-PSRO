import copy
import pprint
from math import radians, degrees
import numpy as np
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Dict, Union

import gymnasium as gym
from gymnasium import spaces

import orekit
vm = orekit.initVM()
print(orekit.VERSION)
from orekit import JArray_double
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime, download_orekit_data_curdir
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import FramesFactory, Transform, Frame, LocalOrbitalFrame, LOFType
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, CartesianOrbit
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.time import DateComponents, TimeComponents
from org.orekit.utils import Constants, PVCoordinates, IERSConventions, PVCoordinatesProvider, TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D, SphericalCoordinates, Rotation
from org.hipparchus.geometry import Vector
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.maneuvers import Maneuver, ConstantThrustManeuver

from org.orekit.forces.maneuvers.propulsion import BasicConstantThrustPropulsionModel
from org.orekit.forces.maneuvers.trigger import DateBasedManeuverTriggers
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.attitudes import LofOffset
from org.orekit.propagation import SpacecraftState



# 设置Orekit环境
# download_orekit_data_curdir()
setup_orekit_curdir()


# 设置 UTC 时间尺度
utc = TimeScalesFactory.getUTC()


LOWER_BOUND = 1e3
UPPER_BOUND = 50e3
LOWER_BOUND_T = 10e3
UPPER_BOUND_T = 30e3
ANGLE_THRESHOLD = radians(30)
THRUST_FORCE = 500.0
ISP = 30000.0
WINNING_STEP = 360

# 初始化给定轨道的 NumericalPropagator
def initialize_propagator(orbit: CartesianOrbit):
    min_step = 1.0e-6  # 最小步长
    max_step = 1000.0  # 最大步长
    init_step = 60.0
    position_tolerance = 10.0  # 位置误差容忍度

    tolerances = NumericalPropagator.tolerances(position_tolerance, orbit, orbit.getType())

    integrator = DormandPrince853Integrator(min_step, max_step, 
                                            JArray_double.cast_(tolerances[0]),  # Double array of doubles needs to be casted in Python
                                            JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(init_step)

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(orbit.getType())
    propagator.setMu(Constants.WGS84_EARTH_MU)
    # propagator.setInitialState(SpacecraftState(orbit, 3000.0))
    propagator.setInitialState(SpacecraftState(orbit, 1000.0))
    # propagator.setInitialState(SpacecraftState(orbit))

    propagator.setAttitudeProvider(LofOffset(FramesFactory.getEME2000(), LOFType.LVLH))
    
    # 添加地球重力场模型，暂时不需要，可以添加不同的力场模型
    gravity_provider = GravityFieldFactory.getNormalizedProvider(8, 8)
    itrf    = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # International Terrestrial Reference Frame, earth fixed
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                            Constants.WGS84_EARTH_FLATTENING,
                            itrf)
    gravityProvider = GravityFieldFactory.getNormalizedProvider(8, 8)
    earth_gravity = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravityProvider)
    # propagator.addForceModel(earth_gravity)
    
    return propagator
            
class SatelliteGameEnv(gym.Env):
    def __init__(self, #ra, rp, 
                 a = 42166.258669*1000,# 500*1000+Constants.WGS84_EARTH_EQUATORIAL_RADIUS,  
                 e = 0.0003 , 
                 i = 0.113,     # 轨道倾角
                 pa = 0.000,    # Perigee Argument，近地点俯角
                 raan = 82.565,     # 升交点赤经
                 mean_anomaly = 231.903, 
                 r = 5000.0, 
                 speed_factor_range = (0.95, 1.05), 
                 step_size = 10.0,
                 shift_steps= 3000):
        """
        这个gym环境适用于单智能体，只操控work星，target星在其自有轨道上运行
        params:
            r: 工作星和目标星之间的初始化距离范围
            speed_factor_range: 目标星相对于工作星的初始化速度系数
        """

        # 状态空间和动作空间定义
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.agent_1 = 1
        self.agent_2 = 0

        # 时间尺度设置
        self.utc = TimeScalesFactory.getUTC()

        # 初始时间设置
        self.initial_date = AbsoluteDate(2024, 10, 20, 0, 0, 0.000, self.utc)
        self.current_date = self.initial_date

        # 地球参考系
        self.earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

        # 惯性参考系
        self.inertial_frame = FramesFactory.getEME2000()

        # 太阳体
        sun = CelestialBodyFactory.getSun()     # Here we get it as an CelestialBody
        self.sun = PVCoordinatesProvider.cast_(sun)  # But we want the PVCoord interface

        self.a = a  # 半长轴
        self.e = e  # 离心率
        self.i = i  # 轨道倾角
        self.pa = pa
        self.raan = raan
        self.mean_anomaly=mean_anomaly
        self.r = r
        self.speed_factor_range = speed_factor_range

        # 保存每个时刻的字典
        self.current_state = {}


        # 仿真参数
        self.step_size = step_size  # 每步的时间步长，单位为秒
        self.shift_steps = shift_steps
        total_propagation_duration = step_size * shift_steps  # 总仿真时间，单位为秒
        self.final_date = self.initial_date.shiftedBy(total_propagation_duration)

        self.timesteps = 0
        self.winning_steps = {self.agent_1: 0,
                              self.agent_2: 0}
        self.winner = None

        self.game_over = False
        self.prev_shaping = None

        self.flag_d_1 = 0   # 进入侦察距离范围标志
        self.flag_d_2 = 0
        self.flag_a_1 = 0   # 进入侦察角度标志
        self.flag_a_2 = 0
        self.rewards = {}
        self.obss = {}

        self.seed = 42

    def set_seed(self, seed=None):
        self.seed = seed
        np.random.seed(seed)

    def reset(self,        
              *, 
              seed: int | None = None,
              options: dict[str, Any] | None = None,):        
        """
        重置环境到初始状态，并返回初始状态信息。
        20241022修改初始化方法为获取当前时刻的速度位置后，使用CartesianOrbit进行计算
        """
        super().reset(seed=seed if seed is not None else self.seed)
        self.timesteps = 0
        self.winning_steps = {self.agent_1: 0,
                              self.agent_2: 0}
        self.winner = None

        self.game_over = False
        self.prev_shaping = None

        self.flag_d_1 = 0   # 进入侦察距离范围标志
        self.flag_d_2 = 0
        self.flag_a_1 = 0   # 进入侦察角度标志
        self.flag_a_2 = 0

        self.rewards = {}
        self.obss = {}

        work_orbit = KeplerianOrbit(
            self.a, 
            self.e, 
            radians(self.i), 
            radians(self.pa), 
            radians(self.raan), 
            radians(self.mean_anomaly),
            PositionAngleType.MEAN, 
            self.inertial_frame, 
            self.initial_date, 
            Constants.WGS84_EARTH_MU
        )

        # 得到初始化坐标
        work_pv = work_orbit.getPVCoordinates()
        self.pos_w = work_pv.getPosition()
        self.vel_w = work_pv.getVelocity()

        # 从开普勒换到笛卡尔轨道下
        work_orbit_ = CartesianOrbit(work_pv, self.inertial_frame, self.initial_date, Constants.WGS84_EARTH_MU)

        # 生成目标卫星的位置和速度
        # 在以工作卫星为原点、半径为r的球面内随机生成目标卫星的位置
        theta = np.random.uniform(0, 2 * np.pi)  # 方位角，0到2π 
        phi = np.random.uniform(0, np.pi)    # 极角 0到π 

        # 使用 SphericalCoordinates 类生成直角坐标
        spherical_coords = SphericalCoordinates(self.r, theta, phi)
        pos_t_relative = spherical_coords.getCartesian()

        # 生成target在j2000下的位置速度
        self.pos_t = self.pos_w.add(pos_t_relative)
        speed_factor = np.random.uniform(*self.speed_factor_range)
        # self.vel_t = self.vel_w.scalarMultiply(speed_factor)
        self.vel_t = self.vel_w
        
        # 构建target的笛卡尔坐标
        target_pv = PVCoordinates(self.pos_t, self.vel_t)
        target_orbit_ = CartesianOrbit(target_pv, self.inertial_frame, self.initial_date, Constants.WGS84_EARTH_MU)

        # 初始化两颗星的传播器
        self.work_propagator = initialize_propagator(work_orbit_)
        self.target_propagator = initialize_propagator(target_orbit_)

        # 获取太阳在当前时间的坐标
        sun_pv = self.sun.getPVCoordinates(self.initial_date, self.inertial_frame)

        # work所需状态，以target为原点的LVLH坐标系下的work相对位置和速度，太阳位置和夹角
        work_relative_lvlh_target_position,\
            work_relative_lvlh_target_velocity,\
            sun_relative_lvlh_target_position,\
            angle_work_target_sun, \
                angle_work_target_sun_degrees=\
                self.get_LVLH_transform(target_pv, work_pv, sun_pv)
        # target所需状态，以work为原点计算LVLH
        target_relative_lvlh_work_position,\
            target_relative_lvlh_work_velocity,\
            sun_relative_lvlh_work_position,\
            angle_target_work_sun, \
                angle_target_work_sun_degrees=\
                self.get_LVLH_transform(work_pv, target_pv, sun_pv)

        # 初始化当前时间
        self.current_date = self.initial_date
        # print("time reset!", self.current_date)

        # 返回状态信息
        state = {
            "time": self.current_date.toString(),
            "work_position": work_pv.getPosition().toArray(),
            "work_velocity": work_pv.getVelocity().toArray(),
            "target_position": target_pv.getPosition().toArray(),
            "target_velocity": target_pv.getVelocity().toArray(),
            "sun_position":sun_pv.getPosition().toArray(),
            "work_relative_lvlh_target_position": work_relative_lvlh_target_position.toArray(),
            "work_relative_lvlh_target_velocity": work_relative_lvlh_target_velocity.toArray(),
            "sun_relative_lvlh_target_position": sun_relative_lvlh_target_position.toArray(), # this has been already normolized
            "angle_work_target_sun":angle_work_target_sun,
            "angle_work_target_sun_degrees": angle_work_target_sun_degrees,
            "target_relative_lvlh_work_position": target_relative_lvlh_work_position.toArray(),
            "target_relative_lvlh_work_velocity": target_relative_lvlh_work_velocity.toArray(),
            "sun_relative_lvlh_work_position": sun_relative_lvlh_work_position.toArray(), # this has been already normolized
            "angle_target_work_sun":angle_target_work_sun,
            "angle_target_work_sun_degrees": angle_target_work_sun_degrees
        }

        self.current_state = state

        obss = self.get_agent_obs(state)
        self.obss = copy.deepcopy(obss)
        
        info = {'winner':self.winner,
                 'done':False,
                 'timesteps':self.timesteps}
        return obss[self.agent_1], info
    


    def apply_thrust(self, propagator:NumericalPropagator, 
                     action_vector:np.array, 
                     thrust_scale: float, 
                     isp: float, 
                     name: str):
        """
        为指定的传播器应用推力。
        :param propagator: NumericalPropagator，目标传播器。
        :param action_vector: np.array，形状为 (3,) 的推力向量。
        :param thrust_scale: float，推力缩放因子。
        :param isp: float，比冲。
        """
        thrust_vector = Vector3D(
            float(action_vector[0] * thrust_scale),
            float(action_vector[1] * thrust_scale),
            float(action_vector[2] * thrust_scale)
        )
        thrust_magnitude = thrust_vector.getNorm()
        if thrust_magnitude == 0:
            return  # 如果推力为零，则不应用推力模型

        # 定义推力方向
        # Normalize thrust direction with zero check
        thrust_mag = thrust_vector.getNorm()
        if thrust_mag == 0.0:
            thrust_direction = Vector3D(0.0, 0.0, 0.0)
        else:
            thrust_direction = thrust_vector.scalarMultiply(1.0 / thrust_mag)
        """
        # 定义推力模型
        propulsion_model = BasicConstantThrustPropulsionModel(
            thrust_magnitude,
            isp,
            thrust_direction,
            name
        )
        """
        # 定义推力开始时间和持续时间
        maneuver_start = self.current_date
        maneuver_duration = self.step_size  # 在整个时间步内应用推力, 废弃，推力太大推进太久会推到地心里,改成1s

        # 清除上一个推力模型
        propagator.removeForceModels()

        # 创建并添加推力模型
        # maneuver_trigger = DateBasedManeuverTriggers(maneuver_start, 1.0)
        lvlh_lofOffset = LofOffset(self.inertial_frame, LOFType.LVLH)
        # maneuver = Maneuver(lvlh_lofOffset, maneuver_trigger, propulsion_model)
        # maneuver = ConstantThrustManeuver(maneuver_start, maneuver_duration, thrust_magnitude, isp, lvlh_lofOffset, thrust_direction)
        maneuver = ConstantThrustManeuver(maneuver_start, 1.0, thrust_magnitude, isp, lvlh_lofOffset, thrust_direction)
        propagator.addForceModel(maneuver)

    def get_LVLH_transform(self, 
                           origin_pv:TimeStampedPVCoordinates,
                           transform_pv:TimeStampedPVCoordinates,
                           sun_pv: TimeStampedPVCoordinates
                           )-> tuple[Vector3D, Vector3D, Vector3D, float, float]:
        """
        获取以给定轨道状态为参考的 LVLH 参考系的转换。
        :param: 
                origin_pv,作为LVLH参考系原点的飞行器的pv
                transform_pv,需要转换坐标的飞行器的pv
                sun_pv, 此时太阳在惯性系下的pv
        :return:
                lvlh_p, transform_state在以origin_state为原点的LVLH坐标系下的位置矢量
                lvlh_v, 速度矢量
                lvlh_s, sun的位置单位矢量(方向)
                angle, t-o-sun夹角(弧度)
                angle_degrees, t-o-sun夹角(角度)
        """

        # 获取惯性系->LVLH坐标的旋转，使用rotation.applyTo()方法完成坐标转换
        rotation :Rotation = LOFType.LVLH.rotationFromInertial(origin_pv)
        # 计算相对矢量, t-o=o->t
        _p = transform_pv.getPosition().subtract(origin_pv.getPosition())
        _v = transform_pv.getVelocity().subtract(origin_pv.getVelocity())
        lvlh_p = rotation.applyTo(_p)
        lvlh_v = rotation.applyTo(_v)

        # 计算太阳位置和t-o-sun夹角
        _sun = sun_pv.getPosition().subtract(origin_pv.getPosition())
        lvlh_s = rotation.applyTo(_sun)
        # 计算单位向量并防止除以0
        p_norm = _p.getNorm()
        sun_norm = _sun.getNorm()
        if p_norm == 0.0:
            p_unit = Vector3D(0.0, 0.0, 0.0)
        else:
            p_unit = _p.scalarMultiply(1.0 / p_norm)

        if sun_norm == 0.0:
            sun_unit = Vector3D(0.0, 0.0, 0.0)
        else:
            sun_unit = _sun.scalarMultiply(1.0 / sun_norm)

        angle = Vector3D.angle(p_unit, sun_unit)   # target-work-sun的夹角
        angle_degrees = degrees(angle)

        return lvlh_p, lvlh_v, lvlh_s, angle, angle_degrees
    
    def calculate(self, action: Dict):
        """
        执行一步仿真，并返回当前状态信息。
        :param action: 工作卫星和目标卫星的推力向量。
        :return: dict，包含当前时间、卫星位置、速度及其他状态信息。
        """

        assert isinstance(action, dict)
            # a1 = action[self.agent_1]
            # a2 = action[self.agent_2]
            # action = np.vstack((a1, a2))

        # 设置推力参数
        thrust_scale = THRUST_FORCE  # 推力缩放因子，可根据需要调整 300/577
        # 1000kg的飞行器，推进一秒，500N改变0.5m/s，100N改变0.1m/s
        isp = ISP  # 比冲，单位为秒

        # 应用推力
        self.apply_thrust(self.work_propagator, action[self.agent_1], thrust_scale, isp, 'work')
        self.apply_thrust(self.target_propagator, action[self.agent_2], thrust_scale, isp, 'target')

        # 计算下一时间步的状态
        next_date = self.current_date.shiftedBy(self.step_size)
        next_work_state = self.work_propagator.propagate(next_date)
        next_target_state = self.target_propagator.propagate(next_date)

        next_work_pv = next_work_state.getPVCoordinates(self.inertial_frame)
        next_target_pv = next_target_state.getPVCoordinates(self.inertial_frame)

        # 获取太阳在当前时间的坐标provider
        sun_pv = self.sun.getPVCoordinates(next_date, self.inertial_frame)

        # work所需状态，以target为原点的LVLH坐标系下的work相对位置和速度，太阳位置和夹角
        work_relative_lvlh_target_position,\
            work_relative_lvlh_target_velocity,\
            sun_relative_lvlh_target_position,\
            angle_work_target_sun, \
                angle_work_target_sun_degrees=\
                self.get_LVLH_transform(next_target_pv, next_work_pv, sun_pv)
        

        # target所需状态，以work为原点计算LVLH
        target_relative_lvlh_work_position,\
            target_relative_lvlh_work_velocity,\
            sun_relative_lvlh_work_position,\
            angle_target_work_sun, \
                angle_target_work_sun_degrees=\
                self.get_LVLH_transform(next_work_pv, next_target_pv, sun_pv)

        # 更新当前时间
        self.current_date = next_date

        state = {
            "time": self.current_date.toString(),
            "work_position": next_work_pv.getPosition().toArray(),
            "work_velocity": next_work_pv.getVelocity().toArray(),
            "target_position": next_target_pv.getPosition().toArray(),
            "target_velocity": next_target_pv.getVelocity().toArray(),
            "sun_position":sun_pv.getPosition().toArray(),
            "work_relative_lvlh_target_position": work_relative_lvlh_target_position.toArray(),
            "work_relative_lvlh_target_velocity": work_relative_lvlh_target_velocity.toArray(),
            "sun_relative_lvlh_target_position": sun_relative_lvlh_target_position.toArray(), # this has been already normolized
            "angle_work_target_sun":angle_work_target_sun,
            "angle_work_target_sun_degrees": angle_work_target_sun_degrees,
            "target_relative_lvlh_work_position": target_relative_lvlh_work_position.toArray(),
            "target_relative_lvlh_work_velocity": target_relative_lvlh_work_velocity.toArray(),
            "sun_relative_lvlh_work_position": sun_relative_lvlh_work_position.toArray(), # this has been already normolized
            "angle_target_work_sun":angle_target_work_sun,
            "angle_target_work_sun_degrees": angle_target_work_sun_degrees
        }

        self.current_state = state
        return state

    def get_agent_obs(self, state:Dict):
        '''获得两颗卫星此时的观测'''
        # agent1的观测
        p_1 = np.asarray(state['work_relative_lvlh_target_position'])
        # p_1_ = np.sign(p_1) * np.log(np.abs(p_1) + 1)
        v_1 = np.asarray(state['work_relative_lvlh_target_velocity'])
        a_1 = np.asarray([state['angle_work_target_sun']])
        sun_1 = np.asarray(state['sun_relative_lvlh_target_position'])
        sun_1 /= np.linalg.norm(sun_1)
        
        self.flag_d_1 = 1 \
            if LOWER_BOUND_T <= np.linalg.norm(p_1) <= UPPER_BOUND_T\
            else 0
                
        self.flag_a_1 = 1 \
            if a_1 <= ANGLE_THRESHOLD\
            else 0
        
        state_1 = np.concatenate([
            p_1,
            v_1,
            a_1,
            sun_1,
            [self.flag_d_1],
            [self.flag_a_1]            
            ])

        assert len(state_1) == 12

        # agent2的观测
        p_2 = np.asarray(state['target_relative_lvlh_work_position'])
        # p_2_ = np.sign(p_2) * np.log(np.abs(p_2) + 1)
        v_2 = np.asarray(state['target_relative_lvlh_work_velocity'])
        a_2 = np.asarray([state['angle_target_work_sun']])
        sun_2 = np.asarray(state['sun_relative_lvlh_work_position'])
        sun_2 /= np.linalg.norm(sun_2)
        
        self.flag_d_2 = 1 \
            if LOWER_BOUND_T <= np.linalg.norm(p_2) <= UPPER_BOUND_T\
            else 0 
        
        self.flag_a_2 = 1 \
            if a_2 <= ANGLE_THRESHOLD\
            else 0
        
        state_2 = np.concatenate([
            p_2,
            v_2,
            a_2,
            sun_2,
            [self.flag_d_2],
            [self.flag_a_2]            
            ])

        assert len(state_2) == 12

        return {self.agent_1: state_1, 
                self.agent_2: state_2}

    def get_opponent_obs(self):
        obs = self.obss[self.agent_2]
        return obs
    
    def render(self, mode='human'):
        # 可视化代码
        pass

    def calculate_shaping(self, state:np.array):
        # 根据当前状态计算奖励
        # state结构[x,y,z,vx,vy,vz,a,sx,sy,sz,flag1,flag2]
        assert len(state) == 12
        d = np.linalg.norm(state[0:3])
        v = np.linalg.norm(state[3:6])

        distance_shaping = np.abs(d-(UPPER_BOUND_T+LOWER_BOUND_T)/2)

        shaping = (
            - 1e-2 * distance_shaping
            - 100 * abs(state[6])
            + 10 * state[10]
            + 10 * state[11]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward

        return shaping
    
    def calculate_reward(self, states:Dict, actions:Dict):

        shaping_1 = self.calculate_shaping(states[self.agent_1])
        shaping_2 = self.calculate_shaping(states[self.agent_2])

        rewards = {self.agent_1: 0.,
                   self.agent_2: 0.}
        shapings = {
            self.agent_1:shaping_1,
            self.agent_2:shaping_2
        }
        if self.prev_shaping is not None:
            # rewards = dict(map(lambda k: (k, shapings[k] - self.prev_shaping[k]), shapings.keys()))
            rewards = {key: shapings[key] - self.prev_shaping[key] for key in shapings}
        self.prev_shaping = shapings

        rewards = {key: rewards[key] - 0.03*np.sum(np.abs(action)) for key, action in actions.items()}

        # 终局结算
        terminated = False
        truncated = False
        self.check_game_over()
        self.check_game_winner(states)
        if self.timesteps == self.shift_steps:
            truncated = True

        if self.game_over:  # 出界了，给予惩罚
            terminated = True
            rewards = {key: -100 for key in rewards}
        if self.winner is not None:
            rewards[self.winner] = +100
            terminated = True

        return rewards, terminated, truncated

    def check_game_over(self):
        p_1 = np.asarray(self.current_state['work_relative_lvlh_target_position'])
        distance = np.linalg.norm(p_1)
        self.game_over = True \
            if  distance <= LOWER_BOUND  \
                or \
                distance >= UPPER_BOUND\
            else False
        
    def check_game_winner(self, states:Dict):
        for key, value in states.items():
            if value[-1] and value[-2]:
                self.winning_steps[key] +=1 
            else:
                self.winning_steps[key] = 0
            
            if self.winning_steps[key] >= WINNING_STEP:
                self.winner = key



    def step(self, action:np.array):
        # 执行一步并返回状态
        self.timesteps += 1
        if action.shape != (2, 3):
            random_array = np.random.uniform(-1, 1, (3,))
            actions = {self.agent_1:action,
                       self.agent_2:random_array}
        else:
            actions = {self.agent_1:action[0],
                       self.agent_2:action[1]}

        state = self.calculate(actions)
        obss = self.get_agent_obs(state)
        
        reward, terminated, truncated= self.calculate_reward(obss, actions)

        self.rewards = reward
        self.obss = copy.deepcopy(obss)
        info = {'winner':self.winner,
                 'done': terminated or truncated,
                 'timesteps':self.timesteps}
        return obss[self.agent_1], reward[self.agent_1], terminated, truncated, info
    

# '''test code'''
# env = SatelliteGameEnv()
# pprint = pprint.pprint
# state,_ = env.reset()

# while True:
#     action = env.action_space.sample()  # 随机生成动作
#     # action = np.zeros(3)
#     # action2 = np.zeros(3)
#     # action = np.vstack((action, action2))
#     o, r, d1, d2, i = env.step(action)
#     print(r)
#     if d1 or d2:
#         print(o)
#         print(i)
#         break
