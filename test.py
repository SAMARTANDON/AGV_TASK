import cv2
import numpy as np
def cutter(x ,i,w):
    if(x+i>=w):
        return 2*w-x-i-1
    else:
        return x+i
def blurring(img):
    img = img.astype(np.float32)
    h, w = img.shape
    padded_h = np.pad(img, ((0, 0), (2, 2)), mode='edge')
    
    # Apply your weights: 0.0625, 0.25, 0.375, 0.25, 0.0625
    # Each slice [:, start:end] represents a shift (like your cutter logic)
    blur_h = (
        0.0625 * padded_h[:, 0:-4] +
        0.25   * padded_h[:, 1:-3] +
        0.375  * padded_h[:, 2:-2] +
        0.25   * padded_h[:, 3:-1] +
        0.0625 * padded_h[:, 4:]
    )
    
    # 2. Vertical Blur
    # Pad height by 2 on each side
    padded_v = np.pad(blur_h, ((2, 2), (0, 0)), mode='edge')
    
    # Apply weights vertically
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
    blurring_img=blurring(img)
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
    # grid of offsets
    xs = np.arange(-r, r+1)
    ys = np.arange(-r, r+1)
    X, Y = np.meshgrid(xs, ys)

    X = X + x0
    Y = Y + y0

    # clip to bounds
    X = np.clip(X, 0, w-2)
    Y = np.clip(Y, 0, h-2)

    # bilinear interpolation (vectorized)
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
        # go to coarsest level
        scale = 2 ** maxlevel
        x /= scale
        y /= scale
        valid = True
        # coarse → fine
        for lvl in range(maxlevel, -1, -1):
            old_img = old_pyr[lvl]
            new_img = new_pyr[lvl]
            ix, iy = gradient(old_img)
            p = track_one_point(old_img, new_img, ix, iy, x, y, r, max_cnt, eps, det_thresh)
            if p is None:
                valid = False
                break
            x, y = p
            # upscale for next level
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
    

feature_params = dict(maxCorners=50, qualityLevel=0.01, minDistance=5, blockSize=7)
lk_params = dict(r=6,maxlevel=3,max_cnt=10,eps=0.01,det_thresh=1e-3)
cap = cv2.VideoCapture("OPTICAL_FLOW.mp4")
ret, old = cap.read()
if not ret:
    print("bad video")
mask=np.zeros_like(old)
old_grey=cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
blur_old_grey = blurring(old_grey)
p0 = cv2.goodFeaturesToTrack(blur_old_grey, mask=None, **feature_params)
while True:
    mask = (mask * 0.90).astype(np.uint8)
    if len(p0)<10:
        p0 = cv2.goodFeaturesToTrack(blur_old_grey, mask=None, **feature_params)
    ret, frame = cap.read()
    if not ret:
        break
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame_grey = blurring(frame_grey)
    p1, st = lucac_kandere(blur_old_grey, blur_frame_grey, p0, **lk_params)
    good_new=p1[st==1]
    good_old=p0.reshape(-1,2)[st==1]
    if p1.size ==0 or np.sum(st)==0: 
        blur_old_grey= blur_frame_grey
        p0 = good_new.reshape(-1, 1, 2)
        if cv2.waitKey(25) == 27:  
            break
        continue
    print("Tracked:", np.sum(st), "/", len(st))
    for (pt1,pt0) in zip(good_new,good_old):
        a,b=pt1.ravel()
        c,d=pt0.ravel()
        if (c-a)**2 + (b-d)**2>2:
            cv2.line(mask,(int(a),int(b)),(int(c),int(d)),(255,0,0),5,2,0)
    final= cv2.add(mask,frame)
    cv2.imshow("lk", final)   
    blur_old_grey= blur_frame_grey
    p0 = good_new.reshape(-1, 1, 2)
    if cv2.waitKey(25) == 27:  
        break
cap.release()
cv2.destroyAllWindows()
