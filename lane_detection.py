import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from utils import get_canny, region_of_interest, display_lines, get_houghlines, make_coordinates,combine_line_img, average_slope_intercept
from multiprocessing import Pool

def main(img_path):
    try:
        img = cv2.imread(img_path)
        img_canny = get_canny(img)
        mask = region_of_interest(img_canny)
        lines = get_houghlines(mask)
        # line_img = display_lines(img, lines)
        # comb_line_img =  combine_line_img(img, line_img)
        avg_line = average_slope_intercept(img, lines)
        line_img = display_lines(img, avg_line)

        comb_line_img = combine_line_img(img, line_img)
        
        if not os.path.exists('output/'):
            os.mkdir('output/')

        img_name = str(img_path).split('/')[-1]
        cv2.imwrite(f'output/{img_name}_out.png', comb_line_img)
        print(f"Done {img_name}")
    except Exception as e:
        print(e)
        pass
if __name__ == "__main__":
    PATH = "data_road/lane_img"
    img_folder = [os.path.join(PATH, img) for img in os.listdir(PATH) if '.png' in img]
    with Pool(processes=4) as pool:
        pool.map(main, img_folder)

    # result = main(img)
    # plt.imshow(result)
    # plt.show()