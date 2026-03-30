
"""
simulation_setup.py

Isolated setup module for setting up the simulation environment required for the given task.

Contains:
  - make_obstacle_texture()   — yellow/black checkerboard texture
  - create_road_and_obstacles() — road surface, lane markings, slalom obstacles, end wall
  - create_car()              — loads and configures the racecar URDF
  - setup_simulation()        — convenience wrapper that initialises everything

Usage (standalone demo):
    python simulation_setup.py

Usage (as a module):
    from simulation_setup import setup_simulation
    carId, steering_joints, motor_joints = setup_simulation()
"""

import cv2
import numpy as np
import tempfile
import pathlib
import pybullet as p
import pybullet_data
import time


def make_obstacle_texture(size: int = 128, tile: int = 10) -> str:
    """
    Creates a yellow (#FFD700) / black checkerboard PNG and returns its path.
    Parameters:
    size : int: Image size in pixels (square).     
    tile : int: Checkerboard tile size in pixels.       

    Returns:
    str : Absolute path to the saved PNG texture file.
    """
    img    = np.zeros((size, size, 3), dtype=np.uint8)
    black  = np.array([0,   0,   0],   dtype=np.uint8)  # BGR black
    yellow = np.array([0,   215, 255], dtype=np.uint8)  # BGR yellow (#FFD700)

    for row in range(size):
        for col in range(size):
            img[row, col] = black if (row // tile + col // tile) % 2 == 0 else yellow

    tex_path = pathlib.Path(tempfile.gettempdir()) / "obs_texture.png"
    ok = cv2.imwrite(str(tex_path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed writing texture to: {tex_path}")

    print(f"[Texture] Saved yellow/black checkerboard → {tex_path}")
    return str(tex_path)


# =============================================================================
# ROAD + OBSTACLES
# =============================================================================

def create_road_and_obstacles():
    """
    Builds the full scene in the currently connected PyBullet world:

      - Road surface     : dark grey box, 33.3 m long × 2.32 m wide
      - Lane markings    : dense white dashes at y = 0, ±0.85 m
      - Slalom obstacles : 5 yellow/black textured boxes in alternating lateral positions
      - End wall         : blue box at x ≈ 31.7 m

    Road half-width is 1.16 m on each side of centre (local frame).
    Obstacle positions (x = 6, 12, 18, 24 m) alternate y = +0.38 / −0.38 m.

    Must be called after p.connect() and p.loadURDF("plane.urdf").
    """
    tex_path = make_obstacle_texture(size=128, tile=10)
    tex_id   = p.loadTexture(tex_path)

    # ── Road surface ──────────────────────────────────────────────────────
    rv = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[16.66, 1.16, 0.01],
        rgbaColor=[0.15, 0.15, 0.15, 1]
    )
    rc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[16.66, 1.16, 0.01])
    p.createMultiBody(0, rc, rv, [16.66, 0, 0.01])

    # ── Lane markings — dense white dashes for background optical flow ────
    for x in np.arange(0, 34, 0.4):
        lv = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.12, 0.03, 0.01],
            rgbaColor=[1, 1, 1, 1]
        )
        p.createMultiBody(0, -1, lv, [x,  0.0,  0.02])
        p.createMultiBody(0, -1, lv, [x,  0.85, 0.02])
        p.createMultiBody(0, -1, lv, [x, -0.85, 0.02])

    # ── Slalom obstacles — white base so yellow/black texture shows fully ─
    obs_extents = [0.25, 0.45, 0.35]
    for i, x in enumerate(range(6, 30, 6)):
        y      = 0.38 if i % 2 == 0 else -0.38
        ov     = p.createVisualShape(
            p.GEOM_BOX, halfExtents=obs_extents,
            rgbaColor=[1, 1, 1, 1]
        )
        oc     = p.createCollisionShape(p.GEOM_BOX, halfExtents=obs_extents)
        obs_id = p.createMultiBody(10, oc, ov, [x, y, 0.35])
        p.changeVisualShape(obs_id, -1, textureUniqueId=tex_id)
        print(f"[Obstacle {i}] x={x}m  y={y:+.2f}m  texture applied")

    # ── End wall (blue — visually distinct from obstacles) ────────────────
    wv = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.5, 1.16, 0.5],
        rgbaColor=[0.1, 0.3, 0.9, 1]
    )
    wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 1.16, 0.5])
    p.createMultiBody(0, wc, wv, [31.66, 0, 0.5])
    print("[End wall] placed at x=31.66 m")


# =============================================================================
# CAR
# =============================================================================

def create_car(start_pos=None, start_orn=None, global_scaling=1.8):
    """
    Loads the PyBullet racecar URDF and returns its ID along with
    categorised joint lists

    Parameters:
    start_pos : list[float], optional: [x, y, z] spawn position. Defaults to [0, 0, 0.25].
    start_orn : list[float], 
    global_scaling : float: URDF scaling factor (1.8 matches the road width).

    Returns:
    car_id : int: PyBullet body ID of the car.
    steering_joints : list[int]: Joint indices whose names contain 'steer'.
    motor_joints : list[int]: Joint indices whose names contain 'wheel'.
    """
    if start_pos is None:
        start_pos = [0, 0, 0.25]
    if start_orn is None:
        start_orn = p.getQuaternionFromEuler([0, 0, 0])

    car_id = p.loadURDF(
        "racecar/racecar.urdf",
        start_pos, start_orn,
        globalScaling=global_scaling
    )
    p.changeDynamics(car_id, -1, ccdSweptSphereRadius=0.1)

    steering_joints, motor_joints = [], []
    for i in range(p.getNumJoints(car_id)):
        name = p.getJointInfo(car_id, i)[1].decode('utf-8')
        if 'steer' in name.lower():
            steering_joints.append(i)
        elif 'wheel' in name.lower():
            motor_joints.append(i)

    print(f"[Car] body_id={car_id}  "
          f"steering_joints={steering_joints}  "
          f"motor_joints={motor_joints}")
    return car_id, steering_joints, motor_joints


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def setup_simulation(dt=1.0 / 60.0, settle_frames=60, gui=True):
    """
    Full simulation initialisation in one call.

    Steps
    -----
    1. Connect to PyBullet (GUI or DIRECT).
    2. Set gravity and load the ground plane.
    3. Build road, lane markings, obstacles, and end wall.
    4. Spawn the racecar and settle its suspension.

    Parameters
    ----------
    dt : float: Physics timestep in seconds. Default 1/60 s.
    settle_frames : int: Number of physics steps to run before returning, so the car
        suspension reaches equilibrium. Default 60.
    gui : bool : If True, opens the PyBullet GUI window. If False, runs headless
        (DIRECT mode) — useful for unit tests or batch runs.

    Returns
    -------
    car_id : int
    steering_joints : list[int]
    motor_joints : list[int]
    """
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    p.loadURDF("plane.urdf")

    create_road_and_obstacles()
    car_id, steering_joints, motor_joints = create_car()

    print(f"[Setup] Settling suspension for {settle_frames} frames …")
    for _ in range(settle_frames):
        p.stepSimulation()
        time.sleep(dt)
    print("[Setup] Ready.")

    return car_id, steering_joints, motor_joints

def cutter(x ,i,w):
    if(x+i>=w):
        return 2*w-x-i-1
    else:
        return x+i
def blurring(img):
    img = img.astype(np.float32)
    h, w = img.shape
    padded_h = np.pad(img, ((0, 0), (2, 2)), mode='edge')
    
    blur_h = (
        0.0625 * padded_h[:, 0:-4] +
        0.25   * padded_h[:, 1:-3] +
        0.375  * padded_h[:, 2:-2] +
        0.25   * padded_h[:, 3:-1] +
        0.0625 * padded_h[:, 4:]
    )
    
    padded_v = np.pad(blur_h, ((2, 2), (0, 0)), mode='edge')
    
    blur_v = (
        0.0625 * padded_v[0:-4, :] +
        0.25   * padded_v[1:-3, :] +
        0.375  * padded_v[2:-2, :] +
        0.25   * padded_v[3:-1, :] +
        0.0625 * padded_v[4:, :]
    )
    
    return blur_v
def pyramiding(img):
    h,w=img.shape
    blurring_img=cv2.GaussianBlur(img.astype(np.float32), (31,31), 10)
    small_img = blurring_img[::2, ::2]
    return small_img

def stack_pyramid(img, levels):
    pyr=[img]
    for i in range(levels):
        img=pyramiding(img)
        pyr.append(img)
    return pyr

def gradient(img):
    img = img.astype(np.float32)
    h,w=img.shape
    ix=np.zeros_like(img)
    iy=np.zeros_like(img)
    ix=ix.astype(np.float32)
    iy=iy.astype(np.float32)
    ix[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    iy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return ix,iy

def make_patch(img, x, y, r,w,h):
    h, w = img.shape
    x0 = int(x)
    y0 = int(y)
    dx = x - x0
    dy = y - y0

    xs = np.arange(-r, r+1)
    ys = np.arange(-r, r+1)
    X, Y = np.meshgrid(xs, ys)

    X = X + x0
    Y = Y + y0


    X = np.clip(X, 0, w-2)
    Y = np.clip(Y, 0, h-2)


    patch = (
        (1-dx)*(1-dy)*img[Y, X] +
        dx*(1-dy)*img[Y, X+1] +
        (1-dx)*dy*img[Y+1, X] +
        dx*dy*img[Y+1, X+1]
    )

    return patch.astype(np.float32)
        
def track_one_point(old_gray, new_gray, Ix, Iy, x0, y0, r, max_iters, eps, det_thresh):
    h,w  = old_gray.shape
    x = float(x0)
    y = float(y0)
    temp = make_patch(old_gray, x0, y0, r,w,h)
    gx = make_patch(Ix, x, y, r,w,h)
    gy = make_patch(Iy, x, y, r,w,h)
    A00 = np.sum(gx * gx)
    A01 = np.sum(gx * gy)
    A11 = np.sum(gy * gy)
    det = A00 * A11 - A01 * A01
    if temp is None:
        return None
    for i in range(max_iters):
        cur = make_patch(new_gray, x, y, r,w,h)
        if abs(det) < det_thresh:
            return None
        if cur is None or gx is None or gy is None:
            return None
        It = temp - cur
        b0 = -np.sum(gx * It)
        b1 = -np.sum(gy * It)
        dx = (A11 * b0 - A01 * b1) / det
        dy = (-A01 * b0 + A00 * b1) / det
        x = x - dx
        y = y - dy
        if (dx * dx + dy * dy) ** 0.5 < eps:
            return (x, y)
    return None

def lucac_kandere(old, new,p0, r, maxlevel, max_cnt, eps,det_thresh):
    p1= []
    ix, iy= gradient(old)
    st=[]
    old_pyr=stack_pyramid(old,maxlevel)
    new_pyr=stack_pyramid(new,maxlevel)
    for t in p0:
        x, y = t.ravel()

        scale = 2 ** maxlevel
        x /= scale
        y /= scale
        valid = True

        for lvl in range(maxlevel, -1, -1):
            old_img = old_pyr[lvl]
            new_img = new_pyr[lvl]
            ix, iy = gradient(old_img)
            p = track_one_point(old_img, new_img, ix, iy, x, y, r, max_cnt, eps, det_thresh)
            if p is None:
                valid = False
                break
            x, y = p
          
            if lvl > 0:
                x *= 2
                y *= 2
        if valid:
            p1.append([x, y])
            st.append(1)
        else:
            p1.append([np.nan, np.nan])
            st.append(0)

        
    return np.array(p1), np.array(st)       
    
def rgbgiver_fixed(car_id, width=640, height=480, distance=0.80, height_offset=0.5):

    pos, orn = p.getBasePositionAndOrientation(car_id)

    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    forward = rot_matrix[:, 0]  
    up = rot_matrix[:, 2]       

    cam_pos = pos + forward * distance + up * height_offset
    cam_target = pos + forward * (distance+1.0) 

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos.tolist(),
        cameraTargetPosition=cam_target.tolist(),
        cameraUpVector=[0, 0, 1]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=100
    )

    img = p.getCameraImage(width, height, view_matrix, proj_matrix)
    return img[2], view_matrix, proj_matrix

def vector_create(rgb,prev_gray,p0):
    foe=None
    obs_mask=None
    Fx=None
    Fy=None
    gamma_x = 30.0
    gamma_y = 100.0
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray.astype(np.float32), (31,31), 10)
    if prev_gray is None:
        prev_gray=gray
        p0=cv2.goodFeaturesToTrack(prev_gray,mask=None,**feature_params)
        foe=None
        return prev_gray, p0,foe,Fx,Fy
    if len(p0)<10:
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    p1,st = lucac_kandere(prev_gray,gray,p0,**lk_params)
    good_new = p1[st == 1]
    good_old = p0.reshape(-1,2)[st == 1]
    foe=compute_foe(good_old,good_new)
    foe = compute_foe(good_old, good_new)
    if foe is None:
        foe = np.array([width//2, height//2])
    flow_map=build_flow_map(good_old,good_new)
    ttc=compute_ttc(flow_map,foe)
    obs_mask=get_obstacle_mask(rgb,flow_map,gray,foe)
    Fxrepel,Fyrepel=compute_repel_force(obs_mask,flow_map,foe,ttc)
    Fxattract,Fyattract,goal_px=goal_pipeline(goal_world,view_matrix,proj_matrix,width,height)
    Fxedge,Fyedge,_,_=edge_pipeline(foe,width,height)

    if Fxedge is None: 
        Fxedge=0
    if Fyedge is None: 
        Fyedge=0
    prev_gray=gray
    Fx=Fxattract-Fxedge-Fxrepel
    Fy=Fyattract-Fyedge-Fyrepel
    
    p0=good_new.reshape(-1,1,2)
    return prev_gray, p0,foe,Fx,Fy

def compute_foe(good_old, good_new):
    A = []
    B = []

    for (new, old) in zip(good_new, good_old):
        x, y = old
        dx = new[0] - old[0]
        dy = new[1] - old[1]

        if dx**2 + dy**2 < 4:
            continue
        a = dy
        b = -dx
        c = dx*y - dy*x
        A.append([a, b])
        B.append([-c])
    if len(A) < 2:
        return None
    A = np.array(A)
    B = np.array(B)
    if len(A) > 30:
        idx = np.random.choice(len(A), 30, replace=False)
        A = A[idx]
        B = B[idx]
   
    sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return sol.flatten()

def build_flow_map(good_old, good_new):
    h, w = height,width
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for (old, new) in zip(good_old, good_new):
        x, y = int(old[0]), int(old[1])
        dx = new[0] - old[0]
        dy = new[1] - old[1]

        for i in range(-6, 7):
            for j in range(-6, 7):
                xi = np.clip(x+i, 0, w-1)
                yj = np.clip(y+j, 0, h-1)

                flow_map[yj, xi] += [dx, dy]
                count[yj, xi] += 1

    count[count == 0] = 1
    flow_map /= count[:, :, None]

    return flow_map

def compute_ttc(flow_map, foe):
    h, w, _ = flow_map.shape
    TTC = np.zeros((h, w), dtype=np.float32)
    if foe is None:
        return None
    fx, fy = foe

    for y in range(h):
        for x in range(w):
            vx, vy = flow_map[y, x]
            mag = np.sqrt(vx*vx + vy*vy)

            if mag < 1e-3:
                TTC[y, x] = 0
                continue

            dist = np.sqrt((x - fx)**2 + (y - fy)**2)
            TTC[y, x] = dist / mag

    return TTC

def get_obstacle_mask(rgb,flow_map, gray, foe):
    flow_mag = np.sqrt(flow_map[:,:,0]**2 + flow_map[:,:,1]**2)
    flow_mag = cv2.GaussianBlur(flow_mag, (11,11), 3)

    flow_mag = flow_mag / (flow_mag.max() + 1e-6)
    
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edges = cv2.GaussianBlur(edges.astype(np.float32), (11,11), 3)
    dist = compute_disturbance(flow_map)
    rad_err = radial_error(flow_map, foe)

    dist = dist / (dist.max() + 1e-6)
    rad_err = rad_err / (rad_err.max() + 1e-6)

    combined = 0.3 * flow_mag +0.4 * rad_err +0.3 * dist
    combined[combined < 0.2] = 0
    combined = (combined * 255).astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.GaussianBlur(combined, (9,9), 2)

    _, clean = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)
    clean = np.zeros_like(combined)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 300:  
            continue

        clean[labels == i] = 255
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)


    h = edges.shape[0]
    edges[:int(0.6*h), :] = 0  

    edges = cv2.dilate(edges, np.ones((3,3), np.uint8))

    strong_motion = (flow_mag > 0.25).astype(np.uint8) * 255
    y_l, y_r = compute_lanes(width, height, foe)
    road_mask = get_road_mask(y_l, y_r, height, width)
    clean = cv2.bitwise_or(clean, strong_motion)
    lane_zone = cv2.dilate(road_mask, np.ones((25,25), np.uint8))
    lane_edges = (
        (edges > 0) &
        (lane_zone > 0) &
        (flow_mag < 0.15)
    )

    lane_edges = lane_edges.astype(np.uint8) * 255

    clean = cv2.bitwise_and(clean, cv2.bitwise_not(lane_edges))

    strong_motion = (flow_mag > 0.25).astype(np.uint8) * 255

    clean = cv2.bitwise_or(clean, strong_motion)
    kernel = np.ones((7,7), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    clean = cv2.medianBlur(clean, 5)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(
        hsv,
        np.array([20, 120, 120]),
        np.array([35, 255, 255])
    )

    clean = cv2.bitwise_or(clean, yellow_mask)

    return clean

def compute_disturbance(flow_map):
    fx = flow_map[:, :, 0]
    fy = flow_map[:, :, 1]

    fx_blur = cv2.GaussianBlur(fx, (21, 21), 5)
    fy_blur = cv2.GaussianBlur(fy, (21, 21), 5)

    dx = fx - fx_blur
    dy = fy - fy_blur

    disturbance = np.sqrt(dx*dx + dy*dy)

    return disturbance

def density_filter(combined, thresh_pixels=10, window=10):

    binary = (combined > 0.05).astype(np.float32)

    kernel = np.ones((window, window), np.float32)

    density = cv2.filter2D(binary, -1, kernel)

    mask = (density > thresh_pixels).astype(np.float32)

    return mask

def radial_error(flow_map, foe):
    fx, fy = foe
    h, w = flow_map.shape[:2]

    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    ex = X - fx
    ey = Y - fy

    v = flow_map
    dot = v[:,:,0]*ex + v[:,:,1]*ey

    mag_v = np.sqrt(v[:,:,0]**2 + v[:,:,1]**2) + 1e-6
    mag_e = np.sqrt(ex**2 + ey**2) + 1e-6

    cos = dot / (mag_v * mag_e)

    return 1 - cos  
    
def compute_repel_force(clean,flow_map,foe,ttc):
    gamma_x=2.0
    gamma_y=20.0
    if clean is None:
        return 0,0
    dist = compute_disturbance(flow_map)
    dir_err = radial_error(flow_map, foe)
    dist = dist / (dist.max() + 1e-6)
    A_mask = (dist > 0.2) & (dir_err > 0.2)
    g = cv2.GaussianBlur(A_mask.astype(np.float32)/255.0, (31,31), 10)
    Fx = gamma_x * np.mean(ttc[A_mask]) if ttc is not None else 0
    Fy = gamma_y * np.mean(g[A_mask]) 
    return Fx,Fy

def detect_lane_edges(gray):
    gray_uint8 = gray.astype(np.uint8)

    edges = cv2.Canny(gray_uint8, 50, 150)

    h = edges.shape[0]
    edges[:int(0.65*h), :] = 0   
    return edges

def get_curvature_params(foe_x, width):
    cx = width / 2
    offset = (foe_x - cx) / width

    if abs(offset) < 0.05:
      
        c2 = 0.005
        n = 1
    else:
        
        n = 2
        if offset > 0:
            c2 = -5e-6   
        else:
            c2 = 5e-6   
    lane_width = width * 0.25
    return c2, n

def compute_lane_params(foe_x, width):
    c2, n = get_curvature_params(foe_x, width)
    c1 = 0
    
    lane_width = width * 0.4
    c0l = -lane_width / 2
    c0r = lane_width / 2
    
    delta_x = (foe_x - width/2) * 0.5
    return c0l, c0r, c1, c2, n, delta_x

def compute_lanes(width, height, foe):
    foe_x=foe[0]
    c2, n = get_curvature_params(foe_x, width)
    c1 = 0
    lane_width = width * 0.4
    c0l = -lane_width / 2
    c0r = lane_width / 2

    delta_x = (foe_x - width/2) * 0.5

    X = np.arange(width)

    y_r = c2 * (X + delta_x)**n + c1*(X + delta_x) + c0r
    y_l = c2 * (X + delta_x)**n + c1*(X + delta_x) + c0l

    return y_l, y_r

def compute_distance_field(y_l, y_r, height, width):
    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    Dl = np.abs(Y - y_l[X])

    Dr = np.abs(Y - y_r[X])

    return Dl, Dr

def compute_edge_potential(Dl, Dr, Y, y_l, y_r):
    A = 10.0
    b = 0.01

    sign_l = np.sign(Y - y_l[np.newaxis, :])
    sign_r = np.sign(Y - y_r[np.newaxis, :])

    Ul = A * (1 - np.exp(b * sign_l * Dl))**2
    Ur = A * (1 - np.exp(-b * sign_r * Dr))**2

    return Ul + Ur

def compute_edge_force(U):
    Ux = np.zeros_like(U)
    Uy = np.zeros_like(U)

    Ux[:,1:-1] = (U[:,2:] - U[:,:-2]) / 2.0
    Uy[1:-1,:] = (U[2:,:] - U[:-2,:]) / 2.0

    Fx = -Ux
    Fy = -Uy
    return Fx, Fy


def edge_pipeline(foe, width, height):
    fx, fy = foe

    # 1. params
    params = compute_lane_params(fx, width)

    # 2. lanes
    y_l, y_r = compute_lanes(width, height,foe)

    # 3. distance
    Dl, Dr = compute_distance_field(y_l, y_r, height, width)

    # 4. meshgrid
    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # 5. potential
    U = compute_edge_potential(Dl, Dr, Y, y_l, y_r)

    # 6. force
    Fx, Fy = compute_edge_force(U)

    return Fx, Fy, y_l, y_r

def get_road_mask(y_l, y_r, height=480, width=640):
    mask = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        yl = int(y_l[x])
        yr = int(y_r[x])

        y_min = max(0, min(yl, yr))
        y_max = min(height-1, max(yl, yr))

        mask[y_min:y_max, x] = 255

    return mask

def project_point(goal_world, view_matrix, proj_matrix):
    point = np.array([*goal_world, 1.0])
    view = np.array(view_matrix).reshape(4,4).T
    proj = np.array(proj_matrix).reshape(4,4).T

    clip = proj @ view @ point
    w = clip[3]
    if not np.isfinite(w) or abs(w) < 1e-6:
        return None
    ndc = clip[:3] / w
    if not np.all(np.isfinite(ndc)):
        return None

    x = int((ndc[0] + 1) * 0.5 * width)
    y = int((1 - ndc[1]) * 0.5 * height)

    return x, y

goal_world = [31.66, 0, 0.5]

def goal_potential(width, height, goal_px, alpha=50):
    gx, gy = goal_px

    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    U = alpha * ((X - gx)**2 + (Y - gy)**2)

    return U

def goal_force(U):
    Ux = np.zeros_like(U)
    Uy = np.zeros_like(U)

    Ux[:,1:-1] = (U[:,2:] - U[:,:-2]) / 2.0
    Uy[1:-1,:] = (U[2:,:] - U[:-2,:]) / 2.0

    Fx = -Ux
    Fy = -Uy

    return Fx, Fy

def goal_pipeline(goal_world, view_matrix, proj_matrix, width, height):
    goal_px = project_point(goal_world, view_matrix, proj_matrix)

    if goal_px is None:
        goal_px = (width//2, int(height*0.2)) 

    U = goal_potential(width, height, goal_px, alpha=0.001)

    Fx, Fy = goal_force(U)

    return Fx, Fy, goal_px

def compute_control(Fx, Fy):
    h, w = Fx.shape

    # focus region (bottom center)
    y1 = int(h * 0.7)
    y2 = h
    x1 = int(w * 0.3)
    x2 = int(w * 0.7)

    Fx_roi = Fx[y1:y2, x1:x2]
    Fy_roi = Fy[y1:y2, x1:x2]

    fx = np.mean(Fx_roi)
    fy = np.mean(Fy_roi)

    return fx, fy

def transform_force(Fx_global, Fy_global, yaw):
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    if Fx_global is None or Fy_global is None:
        return 0,0
    Fx =  cos_y * Fx_global + sin_y * Fy_global
    Fy = -sin_y * Fx_global + cos_y * Fy_global

    return Fx, Fy

def compute_steering(Fy,Fx, lf=0.5, lr=0.5):
    theta_d = np.arctan2(Fy, Fx)
    return np.arctan(((lf + lr) / lr) * np.tan(theta_d))

def draw_force_field(img, Fx, Fy, step=20, scale=10, color=(0, 0, 255)):
    """
    Draw a force field on an image.
    
    Parameters:
    - img : np.array : BGR image
    - Fx, Fy : np.array : force maps (same shape as img height x width)
    - step : int : spacing between vectors
    - scale : float : factor to scale vector length for visualization
    """
    vis = img.copy()
    h, w = Fx.shape
    Fx/=Fx.max()
    Fy/=Fy.max()

    for y in range(0, h, step):
        for x in range(0, w, step):
            fx = Fx[y, x]
            fy = Fy[y, x]
            pt1 = (x, y)
            pt2 = (int(x + fx*scale), int(y + fy*scale))
            cv2.arrowedLine(vis, pt1, pt2, color, 1, tipLength=0.3)
    
    
    cv2.waitKey(1)

feature_params = dict(maxCorners=200, qualityLevel=0.007, minDistance=2, blockSize=5)
lk_params = dict(r=6,maxlevel=3,max_cnt=17,eps=0.005,det_thresh=1e-3)

# =============================================================================
# STANDALONE DEMO
# =============================================================================
width, height = 640, 480
if __name__ == "__main__":
    car_id, steer_j, motor_j = setup_simulation()

    print("\nSimulation running. Close the PyBullet window or press Ctrl+C to exit.")
    dt = 1.0 / 60.0
    try:
        prev_gray=None
        p0=None
        mask=None
        prev_foe=None
        alpha = 0.8
        v = 400.0  
        while True:
            p.stepSimulation()
            time.sleep(dt)
            rgb,view_matrix,proj_matrix=rgbgiver_fixed(car_id)
            rgb = np.reshape(rgb, (height, width, 4)) 
            rgb = rgb[:, :, :3].astype(np.uint8)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            prev_gray,p0,foe,Fx_total,Fy_total= vector_create(rgb,prev_gray,p0)
            pos, orn = p.getBasePositionAndOrientation(car_id)
            _, _, yaw = p.getEulerFromQuaternion(orn)
            # Assume fx_total and fy_total are either None or np.arrays
            fx, fy = 0.0, 0.0 
            if Fx_total is not None and Fy_total is not None:                
                # Ensure pos indices are valid
                x, y = pos[0], pos[1]
                if 0 <= x < Fx_total.shape[0] and 0 <= y < Fx_total.shape[1]:
                    fx, fy = compute_control(Fx_total, Fy_total)
                    fx, fy = transform_force(fx,fy, yaw)
                else:
                    # if pos outside bounds
                    fx, fy = 0.0, 0.0
                
            norm = np.sqrt(fx**2 + fy**2) + 1e-6
            fx /= norm
            fy /= norm
            theta_d = np.arctan2(fy,fx)

            lf,lr=0.5,0.5
            delta_f = np.arctan(((lf + lr)/lr) * np.tan(theta_d))
            delta_f = np.clip(delta_f, -0.5, 0.5)
            speed = 5.0  # base speed
            if fx < 0:
                speed = 1.0
            speed = 2.0 + 2.0 * fx *dt 
            speed = np.clip(speed, 0.5, 5.0)
            print(pos[0])
            print(pos[1])
            # Spin the wheels at a gentle speed so you can see the car move
            for j in steer_j:
                p.setJointMotorControl2(car_id, j,
                    p.POSITION_CONTROL,
                    targetPosition=delta_f)

            for j in motor_j:
                p.setJointMotorControl2(car_id, j,
                    p.VELOCITY_CONTROL,
                    targetVelocity=speed,
                    force=800)
            final=rgb
            if prev_foe is not None and foe is not None:
                foe = alpha * prev_foe + (1 - alpha) * foe
            prev_foe=foe
            cv2.imshow("camera",final)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            p.disconnect()
        except Exception:
            pass
        print("Simulation ended.")