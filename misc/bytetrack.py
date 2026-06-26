"""
ByteTrack — a self-contained, dependency-light implementation.

ByteTrack paper: https://arxiv.org/abs/2110.06864
Key idea:
  • High-confidence detections  → matched first  (like SORT)
  • Low-confidence detections   → matched second against *unmatched* tracks
    (rescues occluded/blurry people that SORT would throw away)

This module is designed to slot into simple-HRNet as a drop-in replacement
for find_person_id_associations().  It produces the same output contract:
    boxes (np.ndarray), pts (np.ndarray), person_ids (np.ndarray)

Dependencies: numpy, scipy  (both already required by simple-HRNet)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Kalman Filter  (constant-velocity, axis-aligned bounding box)
# State: [cx, cy, w, h, vcx, vcy, vw, vh]
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Tracks a single person bounding box with a Kalman filter.
    Box format throughout: [x1, y1, x2, y2]
    """

    _count = 0  # global ID counter

    def __init__(self, bbox: np.ndarray):
        # State transition matrix (constant velocity)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix  (we observe cx, cy, w, h)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        # Covariances
        self.R = np.diag([1., 1., 10., 10.]) * 4       # measurement noise
        self.Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.0001, 0.0001])  # process noise
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4, 1e4])       # initial uncertainty

        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float64)

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

        # Last smoothed pose keypoints (or None if not yet assigned)
        self.pts: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Kalman predict / update
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Advance the state estimate one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._cxcywh_to_xyxy(self.x[:4])

    def update(self, bbox: np.ndarray):
        """Correct the state with a new measurement."""
        z = np.array(self._xyxy_to_cxcywh(bbox), dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> np.ndarray:
        return self._cxcywh_to_xyxy(self.x[:4])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _xyxy_to_cxcywh(bbox):
        x1, y1, x2, y2 = bbox[:4]
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    @staticmethod
    def _cxcywh_to_xyxy(cxcywh):
        cx, cy, w, h = cxcywh
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes [x1,y1,x2,y2].
    Returns matrix of shape (len(a), len(b)).
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-6)


def _linear_assignment(cost: np.ndarray, thresh: float):
    """
    Hungarian assignment on a cost matrix.
    Returns (matched_rows, matched_cols, unmatched_rows, unmatched_cols).
    Only keeps matches where cost <= thresh.
    """
    if cost.size == 0:
        return (np.empty((0,), dtype=int),
                np.empty((0,), dtype=int),
                np.arange(cost.shape[0], dtype=int),
                np.arange(cost.shape[1], dtype=int))

    row_ind, col_ind = linear_sum_assignment(cost)
    valid = cost[row_ind, col_ind] <= thresh

    matched_rows = row_ind[valid]
    matched_cols = col_ind[valid]
    unmatched_rows = np.array([r for r in range(cost.shape[0]) if r not in matched_rows], dtype=int)
    unmatched_cols = np.array([c for c in range(cost.shape[1]) if c not in matched_cols], dtype=int)

    return matched_rows, matched_cols, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack multi-person tracker.

    Usage (inside live-demo.py loop)::

        tracker = ByteTracker()          # create once, before the loop

        # inside the frame loop — call instead of find_person_id_associations:
        boxes, pts, person_ids = tracker.update(boxes, pts)

    Args:
        track_thresh   (float): confidence threshold separating high/low detections.
        match_thresh   (float): maximum IoU *cost* (1-IoU) allowed for a match.
        low_match_thresh (float): looser IoU cost threshold for the second-pass
                                  (low-confidence) matching.
        max_time_lost  (int):   frames a track can go unmatched before deletion.
        min_hits       (int):   minimum consecutive hits before a track is confirmed
                                and reported (reduces false positives at startup).
        smoothing_alpha (float): temporal smoothing weight for pose keypoints.
                                  0 = no smoothing, 1 = keep previous pose.
    """

    def __init__(self,
                 track_thresh: float = 0.45,
                 match_thresh: float = 0.8,
                 low_match_thresh: float = 0.5,
                 max_time_lost: int = 30,
                 min_hits: int = 3,
                 smoothing_alpha: float = 0.1):
        self.track_thresh    = track_thresh
        self.match_thresh    = match_thresh
        self.low_match_thresh = low_match_thresh
        self.max_time_lost   = max_time_lost
        self.min_hits        = min_hits
        self.smoothing_alpha = smoothing_alpha

        self.active_tracks: list[KalmanBoxTracker] = []
        self.lost_tracks:   list[KalmanBoxTracker] = []

        KalmanBoxTracker._count = 0  # reset IDs when tracker is re-created

    # ------------------------------------------------------------------

    def update(self,
               boxes: np.ndarray,
               pts: np.ndarray,
               scores: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one tracking step.

        Args:
            boxes  (np.ndarray): shape (N, 4)  —  [x1, y1, x2, y2] from YOLO/HRNet
            pts    (np.ndarray): shape (N, J, 3) — keypoints from HRNet
            scores (np.ndarray): shape (N,) detection confidence scores.
                                  If None, all detections are treated as high-conf.

        Returns:
            boxes      (np.ndarray): shape (M, 4)   tracked & smoothed boxes
            pts        (np.ndarray): shape (M, J, 3) tracked & smoothed keypoints
            person_ids (np.ndarray): shape (M,)      stable integer person IDs
        """
        if scores is None:
            scores = np.ones(len(boxes), dtype=np.float32)

        # ---- split detections into high / low confidence ----------------
        high_mask = scores >= self.track_thresh
        low_mask  = ~high_mask

        det_high  = boxes[high_mask]
        det_low   = boxes[low_mask]
        pts_high  = pts[high_mask]
        pts_low   = pts[low_mask]

        # ---- Kalman predict all existing tracks -------------------------
        pred_boxes = np.array([t.predict() for t in self.active_tracks]) \
            if self.active_tracks else np.empty((0, 4))

        # ---- PASS 1: match high-conf detections → active tracks ---------
        matched_h, unmatched_dets_h, unmatched_trks = \
            self._match(det_high, pred_boxes, self.match_thresh)

        for d, t in matched_h:
            track = self.active_tracks[t]
            track.update(det_high[d])
            if self.smoothing_alpha and track.pts is not None:
                pts_high[d] = ((1 - self.smoothing_alpha) * pts_high[d]
                               + self.smoothing_alpha * track.pts)
            track.pts = pts_high[d]

        # ---- PASS 2: match low-conf detections → *unmatched* tracks -----
        unmatched_trk_boxes = pred_boxes[unmatched_trks] if len(unmatched_trks) else np.empty((0, 4))
        matched_l, _, still_unmatched_trks = \
            self._match(det_low, unmatched_trk_boxes, self.low_match_thresh)

        for d, t_local in matched_l:
            t_global = unmatched_trks[t_local]
            track = self.active_tracks[t_global]
            track.update(det_low[d])
            if self.smoothing_alpha and track.pts is not None:
                pts_low[d] = ((1 - self.smoothing_alpha) * pts_low[d]
                              + self.smoothing_alpha * track.pts)
            track.pts = pts_low[d]

        # unmatched tracks after both passes → move to lost
        truly_unmatched = unmatched_trks[still_unmatched_trks]
        newly_lost = [self.active_tracks[i] for i in truly_unmatched]

        # ---- spawn new tracks for unmatched HIGH-conf detections --------
        new_tracks = []
        for d in unmatched_dets_h:
            t = KalmanBoxTracker(det_high[d])
            t.pts = pts_high[d]
            new_tracks.append(t)

        # ---- update lost pool -------------------------------------------
        for t in newly_lost:
            t.time_since_update += 0  # already incremented in predict()
        self.lost_tracks.extend(newly_lost)
        self.lost_tracks = [t for t in self.lost_tracks
                            if t.time_since_update <= self.max_time_lost]

        # ---- rebuild active tracks list ---------------------------------
        surviving = [self.active_tracks[i]
                     for i in range(len(self.active_tracks))
                     if i not in truly_unmatched]
        self.active_tracks = surviving + new_tracks

        # ---- collect output (only confirmed tracks) ---------------------
        out_boxes, out_pts, out_ids = [], [], []
        for track in self.active_tracks:
            if track.hit_streak >= self.min_hits or track.hits == 1:
                out_boxes.append(track.get_state())
                out_pts.append(track.pts if track.pts is not None
                               else np.zeros_like(pts[0]))
                out_ids.append(track.id)

        if out_boxes:
            return (np.array(out_boxes, dtype=np.float32),
                    np.array(out_pts,   dtype=np.float32),
                    np.array(out_ids,   dtype=np.int32))
        else:
            nof_joints = pts.shape[1] if len(pts) else 17
            return (np.empty((0, 4),           dtype=np.float32),
                    np.empty((0, nof_joints, 3), dtype=np.float32),
                    np.empty((0,),             dtype=np.int32))

    # ------------------------------------------------------------------

    def _match(self, detections, track_boxes, cost_thresh):
        """
        Hungarian matching between detections and track predictions.
        Returns (matched pairs, unmatched_det_indices, unmatched_trk_indices).
        """
        if len(detections) == 0 or len(track_boxes) == 0:
            return (np.empty((0, 2), dtype=int),
                    np.arange(len(detections), dtype=int),
                    np.arange(len(track_boxes), dtype=int))

        iou = _iou_matrix(detections, track_boxes)
        cost = 1 - iou  # Hungarian solver minimises cost

        rows, cols, unmatched_rows, unmatched_cols = \
            _linear_assignment(cost, cost_thresh)

        matched = np.stack([rows, cols], axis=1) if len(rows) else np.empty((0, 2), dtype=int)
        return matched, unmatched_rows, unmatched_cols
