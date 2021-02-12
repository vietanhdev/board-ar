import cv2
import numpy as np

from .config import (AREA_DIFF_HISTORY_MAXLEN, DRAW_MATCHING_POINTS,
                     HOMOGRAPHY_HISTORY_MAXLEN, MIN_MATCHES)
from .utils import overlay_png


class BoardRenderer:
    """Board renderer. Analyze image and overlay port labels
    """

    def __init__(self, board_img_path, board_pos, draw_matching_points=DRAW_MATCHING_POINTS):
        """Initialize board renderer

        Args:
            board_img_path (str): Path to board image. This should be a transparent image (.png).
            board_pos (tuple): The position of the board in the image frame. Format: (x1, y1, x2, y2).
        """

        self.board_pos = board_pos
        self.draw_matching_points = draw_matching_points

        # Save the history of area difference and homography for smoothing
        self.area_diff_history = [0]
        self.homography_history = []

        # Load image
        board_4channels = cv2.imread(board_img_path, cv2.IMREAD_UNCHANGED)
        self.board_alpha_origin = board_4channels[:, :, 3]
        self.board_bgr_origin = board_4channels[:, :, :3]

        # Extract board crop from the image
        self.board_crop = self.board_bgr_origin[board_pos[1]                                                :board_pos[3], board_pos[0]:board_pos[2]]
        self.board_alpha_origin[board_pos[1]                                :board_pos[3], board_pos[0]:board_pos[2]] = 0

        # === Initialize keypoint detector ===
        # Create ORB keypoint detector
        self.orb = cv2.ORB_create()
        # Create BFMatcher object based on hamming distance
        # https://dsp.stackexchange.com/questions/28557/norm-hamming2-vs-norm-hamming
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Compute model keypoints and its descriptors
        self.kp_model, self.des_model = self.orb.detectAndCompute(
            self.board_crop, None)

        self.last_area = 0

    def render(self, frame):
        """Detect the position of the board on image frame and overlay port labels

        Args:
            frame (np.array): Input frame

        Returns:
            np.array: Output image
        """

        draw = frame.copy()

        # Find the keypoints on the frame
        kp_frame, des_frame = self.orb.detectAndCompute(frame, None)
        if len(kp_frame) == 0:
            return frame

        # Match frame descriptors with model descriptors
        matches = self.bf.match(self.des_model, des_frame)

        # Sort them in the order of their distance
        # The lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)

        # Compute Homography if enough matches are found
        if len(matches) < MIN_MATCHES:
            return frame

        # Calculate source and destination points for matching
        src_pts = np.float32(
            [self.kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts[:, :, 0] = src_pts[:, :, 0] + self.board_pos[0]
        src_pts[:, :, 1] = src_pts[:, :, 1] + self.board_pos[1]
        dst_pts = np.float32(
            [kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute and smooth homography
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 10.0)
        if len(self.homography_history) >= HOMOGRAPHY_HISTORY_MAXLEN:
            self.homography_history.pop(0)
        self.homography_history.append(homography)
        homography = np.mean(self.homography_history, axis=0)

        # Project the board position onto frame
        h, w = self.board_crop.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        pts[:, :, 0] = pts[:, :, 0] + self.board_pos[0]
        pts[:, :, 1] = pts[:, :, 1] + self.board_pos[1]
        dst = cv2.perspectiveTransform(pts, homography)

        # Draw the projected board
        draw = cv2.polylines(draw, [np.int32(dst)],
                             True, (0, 255, 0), 1, cv2.LINE_AA)

        # Check for width / height ratio
        match_wh_ratio = True
        peri = cv2.arcLength(dst, True)
        approx = cv2.approxPolyDP(dst, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        if ar > 1:
            ar = 1.0 / ar
        if len(approx) != 4 or ar < 0.1 or ar > 0.9:
            return frame

        # Check for the stability of the area
        area = cv2.contourArea(dst.reshape(-1, 2).astype(np.float32))
        if len(self.area_diff_history) >= AREA_DIFF_HISTORY_MAXLEN:
            self.area_diff_history.pop(0)
        area_diff = abs(area - self.last_area) / (self.last_area + 1e-5)
        self.last_area = area
        self.area_diff_history.append(area_diff)
        avg_area_diff = np.average(area_diff)
        if avg_area_diff > 0.1:
            return frame

        # Project image
        board_alpha = cv2.warpPerspective(
            self.board_alpha_origin, homography, (frame.shape[1], frame.shape[0]))
        board_bgr = cv2.warpPerspective(
            self.board_bgr_origin, homography, (frame.shape[1], frame.shape[0]))
        draw = overlay_png(draw, board_bgr, board_alpha)

        # Draw first 10 matches.
        if self.draw_matching_points:
            draw = cv2.drawMatches(
                self.board_crop, self.kp_model, draw, kp_frame, matches[:10], 0, flags=2)

        return draw
