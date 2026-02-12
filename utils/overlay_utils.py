# utils/overlay_utils.py (2.5D improved + fast fallback + safe fixes)
import cv2
import numpy as np

# small in-memory cache for resized overlays
_simple_cache = {}
_warp_cache = {}

def clear_overlay_cache():
    _simple_cache.clear()
    _warp_cache.clear()

def _ensure_alpha(img):
    if img is None: return None
    if img.ndim == 2:
        bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA); bgra[:,:,3]=255; return bgra
    if img.shape[2] == 3:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA); bgra[:,:,3]=255; return bgra
    return img.copy()

def _alpha_blend_region(frame, overlay, mask, x, y):
    """Blend overlay into frame at (x,y). overlay & mask are same size."""
    fh, fw = frame.shape[:2]
    h, w = overlay.shape[:2]
    if w == 0 or h == 0: return frame
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(fw, x + w); y2 = min(fh, y + h)
    if x1 >= x2 or y1 >= y2: return frame
    ox1 = x1 - x; oy1 = y1 - y
    ox2 = ox1 + (x2 - x1); oy2 = oy1 + (y2 - y1)
    alpha = (mask[oy1:oy2, ox1:ox2] / 255.0)[:, :, None]
    ov_rgb = overlay[oy1:oy2, ox1:ox2, :3].astype(np.float32)
    bg_rgb = frame[y1:y2, x1:x2, :3].astype(np.float32)
    comp = alpha * ov_rgb + (1 - alpha) * bg_rgb
    frame[y1:y2, x1:x2, :3] = comp.astype(np.uint8)
    return frame

def _warp_perspective(src, src_pts, dst_pts, out_shape):
    """Warp with perspective (4 points) or affine (3 pts). Return warped image and alpha mask."""
    h_out, w_out = out_shape
    s = np.float32(src_pts)
    d = np.float32(dst_pts)
    if s.shape[0] == 4:
        M = cv2.getPerspectiveTransform(s, d)
        warped = cv2.warpPerspective(src, M, (w_out, h_out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    else:
        M = cv2.getAffineTransform(s[:3], d[:3])
        warped = cv2.warpAffine(src, M, (w_out, h_out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    if warped.shape[2] == 4:
        mask = warped[:, :, 3]
    else:
        mask = (cv2.cvtColor(warped[:,:,:3], cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    return warped, mask

def _warp_affine_patch(src, src_tri, dst_tri, out_shape):
    """Warp triangular patch safely and paste into full-size canvas."""
    dst_tri = np.array(dst_tri, dtype=np.float32)
    min_x = int(np.floor(np.min(dst_tri[:,0]))); min_y = int(np.floor(np.min(dst_tri[:,1])))
    max_x = int(np.ceil(np.max(dst_tri[:,0]))); max_y = int(np.ceil(np.max(dst_tri[:,1])))
    if max_x <= min_x or max_y <= min_y:
        return np.zeros((out_shape[0], out_shape[1], 4), dtype=np.uint8), np.zeros((out_shape[0], out_shape[1]), dtype=np.uint8)

    src_tri = np.array(src_tri, dtype=np.float32)
    min_sx = int(np.floor(np.min(src_tri[:,0]))); min_sy = int(np.floor(np.min(src_tri[:,1])))
    max_sx = int(np.ceil(np.max(src_tri[:,0]))); max_sy = int(np.ceil(np.max(src_tri[:,1])))
    src_crop = src[min_sy:max_sy, min_sx:max_sx].copy()
    if src_crop.size == 0:
        return np.zeros((out_shape[0], out_shape[1], 4), dtype=np.uint8), np.zeros((out_shape[0], out_shape[1]), dtype=np.uint8)

    src_tri_rel = src_tri - np.array([min_sx, min_sy], dtype=np.float32)
    dst_rect = dst_tri - np.array([min_x, min_y], dtype=np.float32)
    dst_w = max_x - min_x; dst_h = max_y - min_y
    try:
        M = cv2.getAffineTransform(src_tri_rel[:3], dst_rect[:3])
    except:
        M = np.eye(2,3, dtype=np.float32)
    warped_crop = cv2.warpAffine(src_crop, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    out = np.zeros((out_shape[0], out_shape[1], 4), dtype=np.uint8)
    mask_full = np.zeros((out_shape[0], out_shape[1]), dtype=np.uint8)

    h_crop, w_crop = warped_crop.shape[:2]
    paste_x1 = max(0, min_x); paste_y1 = max(0, min_y)
    paste_x2 = min(out_shape[1], min_x + w_crop); paste_y2 = min(out_shape[0], min_y + h_crop)
    sx1 = max(0, -min_x); sy1 = max(0, -min_y); sx2 = sx1 + (paste_x2 - paste_x1); sy2 = sy1 + (paste_y2 - paste_y1)
    if paste_x1 >= paste_x2 or paste_y1 >= paste_y2:
        return out, mask_full

    out[paste_y1:paste_y2, paste_x1:paste_x2] = warped_crop[sy1:sy2, sx1:sx2]
    if warped_crop.shape[2] == 4:
        mask_full[paste_y1:paste_y2, paste_x1:paste_x2] = warped_crop[sy1:sy2, sx1:sx2, 3]
    else:
        mask_full[paste_y1:paste_y2, paste_x1:paste_x2] = (cv2.cvtColor(warped_crop[sy1:sy2, sx1:sx2, :3], cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    return out, mask_full

# ----------------- Public API -----------------
def overlay_clothes_fast(frame, clothing_img, keypoints):
    """Cheap fallback: compute torso bbox, resize clothing and alpha-blend. Uses cache for speed."""
    cloth = _ensure_alpha(clothing_img)
    fh, fw = frame.shape[:2]
    try:
        ls = keypoints[11]; rs = keypoints[12]; lh = keypoints[23]; rh = keypoints[24]
    except Exception:
        return frame

    lsh = (int(ls.x * fw), int(ls.y * fh)); rsh = (int(rs.x * fw), int(rs.y * fh))
    lhp = (int(lh.x * fw), int(lh.y * fh)); rhp = (int(rh.x * fw), int(rh.y * fh))

    x1 = max(0, min(lsh[0], rsh[0]) - int(0.5 * abs(rsh[0] - lsh[0])))
    x2 = min(fw, max(lsh[0], rsh[0]) + int(0.5 * abs(rsh[0] - lsh[0])))
    y1 = max(0, min(lsh[1], lhp[1]) - int(0.15 * abs(lhp[1] - lsh[1])))
    y2 = min(fh, max(lhp[1], rhp[1]) + int(0.1 * abs(lhp[1] - lsh[1])))
    w = max(2, x2 - x1); h = max(2, y2 - y1)

    key = (id(cloth), w, h)
    if key in _simple_cache:
        resized = _simple_cache[key]
    else:
        try:
            resized = cv2.resize(cloth, (w, h), interpolation=cv2.INTER_AREA)
        except:
            return frame
        _simple_cache[key] = resized

    return _alpha_blend_simple(frame, resized, x1, y1, w, h)

def _alpha_blend_simple(frame, overlay, x1, y1, w, h):
    fh, fw = frame.shape[:2]
    x2, y2 = x1 + w, y1 + h
    if x2 <= 0 or y2 <= 0 or x1 >= fw or y1 >= fh:
        return frame
    # clip overlay region
    ox1 = max(0, -x1); oy1 = max(0, -y1)
    dx1 = max(0, x1); dy1 = max(0, y1)
    dx2 = min(fw, x2); dy2 = min(fh, y2)
    ow = dx2 - dx1; oh = dy2 - dy1
    if ow <= 0 or oh <= 0: return frame
    ov = overlay[oy1:oy1+oh, ox1:ox1+ow]
    alpha = (ov[:,:,3]/255.0)[:,:,None]
    frame[dy1:dy2, dx1:dx2, :3] = (alpha*ov[:,:,:3] + (1-alpha)*frame[dy1:dy2, dx1:dx2, :3]).astype(np.uint8)
    return frame

def overlay_clothes(frame, clothing_img, keypoints):
    """
    2.5D overlay:
     - Warp torso using perspective transform (shoulders->hips).
     - Warp sleeves using two triangular affine patches each side.
    """
    cloth = _ensure_alpha(clothing_img)
    fh, fw = frame.shape[:2]
    ch, cw = cloth.shape[:2]

    try:
        ls = keypoints[11]; rs = keypoints[12]
        lh = keypoints[23]; rh = keypoints[24]
        le = keypoints[13]; re = keypoints[14]
        lw = keypoints[15]; rw = keypoints[16]
    except Exception:
        return frame

    def kp_xy(kp):
        return np.array([int(kp.x * fw), int(kp.y * fh)], dtype=np.int32)

    left_sh = kp_xy(ls); right_sh = kp_xy(rs)
    left_hp = kp_xy(lh); right_hp = kp_xy(rh)
    left_el = kp_xy(le); right_el = kp_xy(re)
    left_wr = kp_xy(lw); right_wr = kp_xy(rw)

    # Torso warp - source quad (fractions of clothing)
    src_torso = np.float32([
        [cw * 0.20, ch * 0.18],
        [cw * 0.80, ch * 0.18],
        [cw * 0.78, ch * 0.88],
        [cw * 0.22, ch * 0.88],
    ])
    dst_torso = np.float32([left_sh, right_sh, right_hp, left_hp])
    warped_torso, mask_torso = _warp_perspective(cloth, src_torso, dst_torso, (fh, fw))
    frame = _alpha_blend_region(frame, warped_torso, mask_torso, 0, 0)

    # sleeve src triangles (fractions)
    src_left_sleeve_tri = [
        [0.02 * cw, 0.26 * ch],
        [0.22 * cw, 0.26 * ch],
        [0.16 * cw, 0.62 * ch],
    ]
    src_left_sleeve_tri2 = [
        [0.02 * cw, 0.26 * ch],
        [0.16 * cw, 0.62 * ch],
        [0.02 * cw, 0.70 * ch],
    ]
    src_right_sleeve_tri = [
        [0.98 * cw, 0.26 * ch],
        [0.78 * cw, 0.26 * ch],
        [0.84 * cw, 0.62 * ch],
    ]
    src_right_sleeve_tri2 = [
        [0.98 * cw, 0.26 * ch],
        [0.84 * cw, 0.62 * ch],
        [0.98 * cw, 0.70 * ch],
    ]

    # destination triangles based on actual arm points
    left_dst_tri = np.array([left_sh, left_el, left_wr], dtype=np.float32)
    left_dst_tri2 = np.array([
        left_sh,
        left_wr,
        [left_wr[0] + int((left_wr[0] - left_el[0]) * 0.25), left_wr[1] + int((left_wr[1] - left_el[1]) * 0.25)]
    ], dtype=np.float32)
    right_dst_tri = np.array([right_sh, right_el, right_wr], dtype=np.float32)
    right_dst_tri2 = np.array([
        right_sh,
        right_wr,
        [right_wr[0] + int((right_wr[0] - right_el[0]) * 0.25), right_wr[1] + int((right_wr[1] - right_el[1]) * 0.25)]
    ], dtype=np.float32)

    # warp and blend sleeve triangles
    lpatch1, lmask1 = _warp_affine_patch(cloth, src_left_sleeve_tri, left_dst_tri, (fh, fw))
    frame = _alpha_blend_region(frame, lpatch1, lmask1, 0, 0)

    lpatch2, lmask2 = _warp_affine_patch(cloth, src_left_sleeve_tri2, left_dst_tri2, (fh, fw))
    frame = _alpha_blend_region(frame, lpatch2, lmask2, 0, 0)

    rpatch1, rmask1 = _warp_affine_patch(cloth, src_right_sleeve_tri, right_dst_tri, (fh, fw))
    frame = _alpha_blend_region(frame, rpatch1, rmask1, 0, 0)

    rpatch2, rmask2 = _warp_affine_patch(cloth, src_right_sleeve_tri2, right_dst_tri2, (fh, fw))
    frame = _alpha_blend_region(frame, rpatch2, rmask2, 0, 0)

    return frame
