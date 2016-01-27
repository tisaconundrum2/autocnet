import pandas as pd
from autocnet.matcher import matcher

# docs
# operates on one set of (src, dest) (kp, image)s
# error if passed in size is even. Don't do this....
# calculate an (x,y, strength) for each keypoint match in edge
# can we assume templates are square?
# look into the keypoint size and something with the descriptors to check physical area....

def subpixel_offset(template_kp, search_kp, template_img, search_img, template_size=9, search_size=27):
    # Get the x,y coordinates
    temp_x, temp_y = map(int, template_kp.pt)
    search_x, search_y = map(int, search_kp.pt)

    # Convert desired template and search sizes to offsets to get the bounding box
    t = int(template_size/2) #index offset for template
    s = int(search_size/2) #index offset for search

    template = template_img[temp_y-t:temp_y+t, temp_x-t:temp_x+t]
    search = search_img[search_y-s:search_y+s, search_x-s:search_x+s]

    # actually do the pattern match
    return matcher.pattern_match(template, search)
