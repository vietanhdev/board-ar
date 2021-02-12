# Draw matching points between the template and the image 
DRAW_MATCHING_POINTS = False

# Board image. Used for matching points and show ports
BOARD_IMG_PATH = "images/pico.png"

# Minimum number of matched keypoints
MIN_MATCHES = 15

# Position of the board on png image
# x1, y1, x2, y2
BOARD_POS = (306, 100, 496, 550)

# History lengths
AREA_DIFF_HISTORY_MAXLEN = 30
HOMOGRAPHY_HISTORY_MAXLEN = 3