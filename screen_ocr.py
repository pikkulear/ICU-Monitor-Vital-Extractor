import easyocr
import cv2
import numpy as np
import re


def count_greenary(img, bbox):
    x1,y1,x2,y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
    img_hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)    
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    return np.count_nonzero(mask)


class ScreenOCR(object):
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def match_vitals_by_logic(self, nums, mAPs, sys_dia, others, img):
        # RR -> in nums nearest to 20 (limit by 0-40)
        rrs = [n[0] for n in nums if 0<=n[0]<=45]
        RR = None
        if len(rrs)>0:
            rrs.sort(key=lambda x: abs(x-20))
            RR = int(rrs[0])
            nums = [n for n in nums if rrs[0]!=n[0]]
        # HR -> in nums the most green one lying between 60-140
        HR = None
        if len(nums)>0:
            nums = nums[:5]
            greenary_idx = [count_greenary(img, n[1]) for n in nums]
            for i in range(len(greenary_idx)):
                nums[i] = nums[i]+(greenary_idx[i],)
            nums.sort(key = lambda x: (x[3]), reverse=True)
            for n in nums:
                if n[0] in range(50,180):
                    HR = int(n[0])
                    nums = [num for num in nums if n[0]!=num[0]]
                    break
        # SpO2 -> in nums nearest to 100 and closest to bbox with text 'sp*2' or 'sp*z'
        SpO2 = None
        sps = [n[0] for n in nums if 70<=n[0]<=100]
        if len(sps)>0:
            sps.sort(key=lambda x: abs(x-100))
            SpO2 = int(sps[0])

        # Sys_Dia -> at most 3 digits before and after '/'
        Sys, Dia = None, None
        if len(sys_dia)>0:
            Sys = int(str(sys_dia[0][0][0])[-3:]) 
            Dia = int(str(sys_dia[0][0][1])[:3])
        # mAP -> digits between '()' nearest to 100
        mAP = None
        if len(mAPs)>0:
            candidates = [int(item) for item in re.findall(r'\b\d+\b', mAPs[0][0])]
            candidates.sort(key = lambda x: abs(x-100))
            mAP = candidates[0]
        res = {
            'RR': RR, 'HR': HR, 'SPO2': SpO2,
            'MAP': mAP, 'SBP': Sys, 'DBP': Dia,
        }
        return res

    
    def read_vitals(self, image, image_rgb, get_mAP=True, conf_thres=.4):
        results = self.reader.readtext(image)
        nums, mAPs, sys_dia, others = [], [], [], []
        for result in results:
            bbox, reading, conf = result
            if conf < conf_thres: continue
            try:
                num = float(reading)
                nums.append((num, bbox, conf))
            except ValueError:
                # if get_mAP:
                mAP_match = re.search(r'^\(|\)$', reading)
                sd_match = re.search(r'(\d+)[ /_]+(\d+)', reading)
                if mAP_match:
                    mAPs.append((reading, bbox, conf))
                elif sd_match:
                    sys_dia.append(([int(sd_match.group(1)), int(sd_match.group(2))], bbox, conf))
                else:
                    others.append((reading, bbox, conf))
        nums.sort(key=lambda x: (x[1][2][0]-x[1][0][0]) * (x[1][3][1]-x[1][1][1]), reverse=True)
        mAPs.sort(key=lambda x: (x[1][2][0]-x[1][0][0]) * (x[1][3][1]-x[1][1][1]), reverse=True)
        sys_dia.sort(key=lambda x: (x[1][2][0]-x[1][0][0]) * (x[1][3][1]-x[1][1][1]), reverse=True)
        res = self.match_vitals_by_logic(nums, mAPs, sys_dia, others, image_rgb)
        # print(res)
        return res


    def sanity_check_values(self, value, vital:str):
        """
        SpO2 -> between 75 - 100
        HR -> between 60 - 150
        mAP -> avg of sys and dia and between 80 - 120
        Sys -> 
        """
        if vital=='HR':
            return value 



    def match_vitals_by_position(self, groups, screen_type):
        pass
