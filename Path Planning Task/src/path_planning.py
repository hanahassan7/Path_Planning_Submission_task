from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

from src.models import CarPose, Cone, Path2D

class PathPlanning:
    """Student-implemented path planner.

    You are given the car pose and an array of detected cones, each cone with (x, y, color)
    where color is 0 for yellow (right side) and 1 for blue (left side). The goal is to
    generate a sequence of path points that the car should follow.

    Implement ONLY the generatePath function.
    """
    #Initialing a constant with half of the track width
    HALF_WIDTH = 2.0 
     
    def __init__(self, car_pose: CarPose, cones: List[Cone]):
        self.car_pose = car_pose
        self.cones = cones

    def generatePath(self) -> Path2D:
        """Return a list of path points (x, y) in world frame.

        Requirements and notes:
        - Cones: color==0 (yellow) are on the RIGHT of the track; color==1 (blue) are on the LEFT.
        - You may be given 2, 1, or 0 cones on each side.
        - Use the car pose (x, y, yaw) to seed your path direction if needed.
        - Return a drivable path that stays between left (blue) and right (yellow) cones.
        - The returned path will be visualized by PathTester.

        The path can contain as many points as you like, but it should be between 5-10 meters,
        with a step size <= 0.5. Units are meters.

        Replace the placeholder implementation below with your algorithm.
        """
        #creating a list of blue and yellow cones 
        blue_cones = []
        yellow_cones = []
        #appending the list based on the color
        for c in self.cones:
            if c.color == 1:
                blue_cones.append(c)
            elif c.color == 0:
                yellow_cones.append(c)
       
        # Case 1: No cones detected, generate a straight path in the direction of the car's yaw
        if not blue_cones and not yellow_cones:
            return self.generate_straightpath()

        
        #creating variable to store the car's x and y coordinates
        cx, cy = self.car_pose.x, self.car_pose.y
        #sorting using nearest-neighbor chain (first cone relative to car, subsequent cones relative to previous cone)
        blue_cones = self._chain_sort(blue_cones, cx, cy)
        yellow_cones = self._chain_sort(yellow_cones, cx, cy)

        # creating a list of points(tuples) for the cones
        blue_cone_points=[]
        yellow_cone_points=[]

        for c in blue_cones:
            blue_cone_points.append((c.x, c.y))

        for c in yellow_cones:
            yellow_cone_points.append((c.x, c.y))

        # Case 2: handling if there is cones in one side and empty in the other side (generating a virtual cone)
        if not yellow_cone_points and blue_cone_points:
            blue_cone_points, yellow_cone_points = self.generating_yellow_cones(blue_cone_points, yellow_cone_points)
        elif not blue_cone_points and yellow_cone_points:
            blue_cone_points, yellow_cone_points = self.generating_blue_cones(blue_cone_points, yellow_cone_points)

        # Case 3: If there are no cones on either side, Uneven Arrays
        midpoint =self.midpoints(blue_cone_points, yellow_cone_points)

        #generate the path to start from the car's current position and follow the midpoints
        distance = 0.5
        next_points=(
            cx + math.cos(self.car_pose.yaw) * distance,
            cy + math.sin(self.car_pose.yaw) * distance
        )
        final_path = [(cx, cy), next_points] + midpoint

        #returning the final path after smoothing and resampling the path to ensure a step size <= 0.5m
        return self.smoothandresample(final_path)
        
        



    "METHODS"

    #to generate a straight path in the direction of the car's yaw for scenario 1
    def generate_straightpath(self,target_length: float = 7.5, step: float = 0.4) -> Path2D:
        num_points= int(target_length / step)
        path =[]
        for i in range(1, num_points + 1):
            dx = math.cos(self.car_pose.yaw) * step * i
            dy = math.sin(self.car_pose.yaw) * step * i
            path.append((self.car_pose.x + dx, self.car_pose.y + dy))
        return path
    
    #to generate virtual yellow cones for scenario 2
    def generating_yellow_cones(self, blue_cone_points: List[Tuple[float, float]], yellow_cone_points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        #initialize an empty list to store the generated yellow cone points
        yellow_cone_points = []
        for i, (bx, by) in enumerate(blue_cone_points):
            if len(blue_cone_points) > 1:
                #if the current cone is not the last one
                if i < len(blue_cone_points) - 1:
                    dx, dy = blue_cone_points[i+1][0] - bx, blue_cone_points[i+1][1] - by
                else:
                    dx, dy = bx - blue_cone_points[i-1][0], by - blue_cone_points[i-1][1]
            else:
                dx, dy = math.cos(self.car_pose.yaw), math.sin(self.car_pose.yaw)
                                
            # Normalize Tangent Vector
            length = math.hypot(dx, dy)
            if length > 0: dx, dy = dx / length, dy / length
            else: dx, dy = math.cos(self.car_pose.yaw), math.sin(self.car_pose.yaw)
                
            yellow_cone_points.append((bx + dy * 2.0 * self.HALF_WIDTH, 
                        by - dx * 2.0 * self.HALF_WIDTH))
        return blue_cone_points, yellow_cone_points

    #to generate virtual blue cones for scenario 2
    def generating_blue_cones(self, blue_cone_points: List[Tuple[float, float]], yellow_cone_points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        #initialize an empty list to store the generated blue cone points
        blue_cone_points = []
        for i, (yx, yy) in enumerate(yellow_cone_points):
            if len(yellow_cone_points) > 1:
                if i < len(yellow_cone_points) - 1:
                    dx, dy = yellow_cone_points[i+1][0] - yx, yellow_cone_points[i+1][1] - yy
                else:
                    dx, dy = yx - yellow_cone_points[i-1][0], yy - yellow_cone_points[i-1][1]
            else:
                dx, dy = math.cos(self.car_pose.yaw), math.sin(self.car_pose.yaw)

            length = math.hypot(dx, dy)
            if length > 0: dx, dy = dx / length, dy / length
            else: dx, dy = math.cos(self.car_pose.yaw), math.sin(self.car_pose.yaw)

            blue_cone_points.append((yx - dy * 2.0 * self.HALF_WIDTH, 
                          yy + dx * 2.0 * self.HALF_WIDTH))

        return blue_cone_points, yellow_cone_points
    
    #to generate midpoints between the blue and yellow cones
    def midpoints(self, blue_cone_points: List[Tuple[float, float]], yellow_cone_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            
        midpoints = []
        #minimum length of the two lists to avoid index errors
        min_len = min(len(blue_cone_points), len(yellow_cone_points))
            
        # Symmetrical bisecting calculation
        for i in range(min_len):
            mx = (blue_cone_points[i][0] + yellow_cone_points[i][0]) / 2.0
            my = (blue_cone_points[i][1] + yellow_cone_points[i][1]) / 2.0
            midpoints.append((mx, my))

        # excess cones accurately extending track layout
        if len(blue_cone_points) > min_len:
            for i in range(min_len, len(blue_cone_points)):
                bx, by = blue_cone_points[i]
                dx, dy = (midpoints[-1][0] - midpoints[-2][0], midpoints[-1][1] - midpoints[-2][1]) if len(midpoints) >= 2 else (math.cos(self.car_pose.yaw), math.sin(self.car_pose.yaw))
                l = math.hypot(dx, dy)
                if l > 0: dx, dy = dx/l, dy/l
                midpoints.append((bx + dy * self.HALF_WIDTH, by - dx * self.HALF_WIDTH))
                    
        elif len(yellow_cone_points) > min_len:
            for i in range(min_len, len(yellow_cone_points)):
                yx, yy = yellow_cone_points[i]
                dx, dy = (midpoints[-1][0] - midpoints[-2][0], midpoints[-1][1] - midpoints[-2][1]) if len(midpoints) >= 2 else (math.cos(self.car_pose.yaw), math.sin(self.car_pose.yaw))
                l = math.hypot(dx, dy)
                if l > 0: dx, dy = dx/l, dy/l
                midpoints.append((yx - dy * self.HALF_WIDTH, yy + dx * self.HALF_WIDTH))
    
        return midpoints

    #sort cones using nearest-neighbor chain starting from a reference point
    def _chain_sort(self, cones: List[Cone], start_x: float, start_y: float) -> List[Cone]:
        if not cones:
            return []
        remaining = list(cones)
        sorted_list = []
        cx, cy = start_x, start_y
        while remaining:
            nearest = min(remaining, key=lambda c: math.hypot(c.x - cx, c.y - cy))
            sorted_list.append(nearest)
            cx, cy = nearest.x, nearest.y
            remaining.remove(nearest)
        return sorted_list

    #smooth and resample the path to ensure a step size <= 0.5m
    def smoothandresample(self, finalpoints: List[Tuple[float, float]], target_len: float = 7.5, step: float = 0.4) -> Path2D:
        pts = np.array(finalpoints)
        
      
        dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        keep_idx = [0] + list(np.where(dists > 1e-4)[0] + 1)
        pts = pts[keep_idx]

        if len(pts) < 2:
            return self.generate_straightpath(target_len, step)

        # Compute cumulative arc-length parameterization
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        cum_dist = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        data_length = cum_dist[-1]

        # Enforce minimum total path length
        eval_dist = max(target_len, data_length)
        num_samples = max(2, math.ceil(eval_dist / step) + 1)
        
        # Generate uniformly-spaced sample distances
        sample_dists = np.linspace(0, eval_dist, num_samples)

        # Use shape-preserving piecewise cubic interpolation to avoid overshoots
        if len(pts) == 2:
            fx = interp1d(cum_dist, pts[:, 0], kind='linear', fill_value="extrapolate")
            fy = interp1d(cum_dist, pts[:, 1], kind='linear', fill_value="extrapolate")
        else:
            fx = PchipInterpolator(cum_dist, pts[:, 0])
            fy = PchipInterpolator(cum_dist, pts[:, 1])


        # Compute the tangent direction at the end of the data for linear extension
        eps = min(1e-3, data_length * 0.01)
        end_x = float(fx(data_length))
        end_y = float(fy(data_length))
        near_x = float(fx(data_length - eps))
        near_y = float(fy(data_length - eps))
        dx_end = (end_x - near_x) / eps
        dy_end = (end_y - near_y) / eps

        path = []
        for d in sample_dists:
            if d <= data_length:
                # Within data range: use spline interpolation
                path.append((float(fx(d)), float(fy(d))))
            else:
                # Beyond data range: extend linearly from the last data point
                overshoot = d - data_length
                path.append((end_x + dx_end * overshoot, end_y + dy_end * overshoot))

        # Adaptive subdivision: ensure no segment exceeds step size in Euclidean space
        refined = [path[0]]
        for i in range(1, len(path)):
            px, py = refined[-1]
            qx, qy = path[i]
            seg_dist = math.hypot(qx - px, qy - py)
            if seg_dist > step:
                # Subdivide this segment
                n_sub = math.ceil(seg_dist / step)
                for j in range(1, n_sub):
                    t = j / n_sub
                    # Interpolate parametric distance for this sub-point
                    d_prev = sample_dists[i - 1] if i - 1 < len(sample_dists) else 0
                    d_curr = sample_dists[i] if i < len(sample_dists) else eval_dist
                    d_mid = d_prev + t * (d_curr - d_prev)
                    if d_mid <= data_length:
                        refined.append((float(fx(d_mid)), float(fy(d_mid))))
                    else:
                        overshoot = d_mid - data_length
                        refined.append((end_x + dx_end * overshoot, end_y + dy_end * overshoot))
            refined.append((qx, qy))

        return refined