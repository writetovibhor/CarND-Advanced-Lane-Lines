import glob
import pickle
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from collections import deque

nx = 9
ny = 6

c_thresh=(170, 255)
s_thresh=(200, 255)
dir_thresh=(0.7, 1.1)


ym_per_pix = 3.0/72.0 # meters per pixel in y dimension
xm_per_pix = 3.7/660.0 # meters per pixel in x dimension
y_eval = 700
midx = 650

TEST_IMAGES = True

def calibrate_camera():
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    for filename in glob.iglob('camera_cal/*.jpg'):
        # Read image
        img = cv2.imread(filename)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imwrite("output_images/{}".format(filename), img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    for filename in glob.iglob('output_images/camera_cal/*.jpg'):
        img = cv2.imread(filename)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite("{}-undistorted.jpg".format(filename), dst)


    return mtx, dist

def undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def convert_binary(img):
    g_channel = img[:,:,1]
    cv2.imwrite('output_images/test4-green-channel.jpg', g_channel)
    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel >= c_thresh[0]) & (g_channel < c_thresh[1])] = 255
    cv2.imwrite('output_images/test4-green-channel-threshold.jpg', g_binary)

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    cv2.imwrite('output_images/test4-s-channel.jpg', s_channel)

    s_binary = np.zeros_like(s_channel, dtype=np.uint8)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255
    cv2.imwrite('output_images/test4-s-channel-threshold.jpg', s_binary)

    # Sobel gray
    color = np.copy(img)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output_images/test4-gray.jpg', gray)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    cv2.imwrite('output_images/test4-sobel-x.jpg', scaled_sobel)

    sxbinary = np.zeros_like(scaled_sobel, dtype=np.uint8)
    sxbinary[(scaled_sobel >= 10) & (scaled_sobel <= 70)] = 255
    cv2.imwrite('output_images/test4-sobel-x-threshold.jpg', sxbinary)

    color_binary = np.dstack(( sxbinary, s_binary, g_binary)).astype("uint8")
    gray_binary = np.zeros_like(g_binary).astype("uint8")
    BIN_THRESH = 255
    gray_binary[((g_binary >= BIN_THRESH) & (s_binary >= BIN_THRESH)) | ((g_binary >= BIN_THRESH) & (sxbinary >= BIN_THRESH)) | ((s_binary >= BIN_THRESH) & (sxbinary >= BIN_THRESH))] = 255

    return color_binary, gray_binary

def perspective_warp(img):
    img_size = (img.shape[1], img.shape[0])
    width, height = img_size
    offset = 200
    src = np.float32([
        [  586,   446 ],
        [  713,   446 ],
        [ 1119,   683 ],
        [  254 ,  683 ]])
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped, Minv

# # Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = deque(maxlen=5)
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

left_lane = Line()
right_lane = Line()
#lane_widths = deque(maxlen=10)

def refit_histogram(binary_warped, name=None):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    if name is not None:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if name is not None:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    distance = np.mean(right_fitx - left_fitx)
    if distance < 600 or distance > 700:
        invalid = True
    else:
        invalid = False

    if name is not None:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.clf()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig("output_images/{}-4-histogram-fit.jpg".format(name))


    y1 = (2*left_fit[0]*y_eval + left_fit[1])*xm_per_pix/ym_per_pix
    y2 = 2*left_fit[0]*xm_per_pix/(ym_per_pix*ym_per_pix)

    curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)

    return left_fit, right_fit, curvature, invalid

def fit_histogram(binary_warped, name=None):
    if left_lane.detected and right_lane.detected:
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_lane.best_fit[0]*(nonzeroy**2) + left_lane.best_fit[1]*nonzeroy + left_lane.best_fit[2] - margin)) & (nonzerox < (left_lane.best_fit[0]*(nonzeroy**2) + left_lane.best_fit[1]*nonzeroy + left_lane.best_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_lane.best_fit[0]*(nonzeroy**2) + right_lane.best_fit[1]*nonzeroy + right_lane.best_fit[2] - margin)) & (nonzerox < (right_lane.best_fit[0]*(nonzeroy**2) + right_lane.best_fit[1]*nonzeroy + right_lane.best_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        width_ratio = (right_fitx[-1] - left_fitx[-1]) / (right_fitx[0] - left_fitx[0])

        l_y1 = (2*left_fit[0]*y_eval + left_fit[1])*xm_per_pix/ym_per_pix
        l_y2 = 2*left_fit[0]*xm_per_pix/(ym_per_pix*ym_per_pix)

        l_curvature = ((1 + l_y1*l_y1)**(1.5))/np.absolute(l_y2)

        if width_ratio > 0.9 and width_ratio < 1.1:
            left_lane.current_fit.append(left_fit)
            right_lane.current_fit.append(right_fit)
            left_lane.radius_of_curvature = l_curvature
            right_lane.radius_of_curvature = l_curvature
            left_lane.best_fit = np.mean(left_lane.current_fit, axis=0)
            right_lane.best_fit = np.mean(right_lane.current_fit, axis=0)
        else:
            left_lane.detected = False
            right_lane.detected = False
    else:
        left_fit, right_fit, curvature, invalid = refit_histogram(binary_warped, name)
        if not invalid:
            left_lane.current_fit.append(left_fit)
            right_lane.current_fit.append(right_fit)
            left_lane.radius_of_curvature = curvature
            right_lane.radius_of_curvature = curvature
            if not TEST_IMAGES:
                left_lane.detected = True
                right_lane.detected = True
            left_lane.best_fit = np.mean(left_lane.current_fit, axis=0)
            right_lane.best_fit = np.mean(right_lane.current_fit, axis=0)
    return left_lane.best_fit, right_lane.best_fit, left_lane.radius_of_curvature

def plot_lanes(binary_warped, undistorted_img, Minv, left_fit, right_fit, curvature):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_img.shape[1], undistorted_img.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

    cv2.putText(result,'Radius of Curvature: %.2fm' % curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
    x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
    position_from_center = ((x_left_pix + x_right_pix)/2 - midx) * xm_per_pix
    if position_from_center < 0:
        text = 'left'
    else:
        text = 'right'
    cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    return result

def process_image(img):
    dst = undistort(img, mtx, dist)
    c_binary, g_binary = convert_binary(dst)
    warped, Minv = perspective_warp(g_binary)
    left_fit, right_fit, curvature = fit_histogram(warped)
    return plot_lanes(warped, dst, Minv, left_fit, right_fit, curvature)

try:
    calibration = pickle.load(open("calibration.pkl", "rb"))
    mtx = calibration["mtx"]
    dist = calibration["dist"]
except (OSError, IOError) as e:
    mtx, dist = calibrate_camera()
    calibration = {"mtx":mtx, "dist":dist}
    pickle.dump(calibration, open("calibration.pkl", "wb"))

if TEST_IMAGES:
    left_lane.current_fit = deque(maxlen=1)
    right_lane.current_fit = deque(maxlen=1)
    for filename in glob.iglob('test_images/*.jpg'):
        if filename == 'test_images/test4.jpg':
            img = cv2.imread(filename)
            dst = undistort(img, mtx, dist)
            name = os.path.splitext(os.path.basename(filename))[0]
            cv2.imwrite("output_images/{}-0-undistorted.jpg".format(name), dst)
            c_binary, g_binary = convert_binary(dst)
            cv2.imwrite("output_images/{}-1-color_binary.jpg".format(name), c_binary)
            cv2.imwrite("output_images/{}-2-gray_binary.jpg".format(name), g_binary)
            warped, Minv = perspective_warp(g_binary)
            cv2.imwrite("output_images/{}-3-perspective.jpg".format(name), warped)
            left_fit, right_fit, curvature = fit_histogram(warped, name)
            result = plot_lanes(warped, dst, Minv, left_fit, right_fit, curvature)
            cv2.imwrite("output_images/{}-5-final.jpg".format(name), result)
else:
    left_lane.current_fit = deque(maxlen=5)
    right_lane.current_fit = deque(maxlen=5)
    print('Processing video ...')
    clip2 = VideoFileClip("project_video.mp4")
    vid_clip = clip2.fl_image(process_image)
    vid_clip.write_videofile("output_images/project.mp4", audio=False)
