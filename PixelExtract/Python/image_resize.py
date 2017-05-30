import PIL
from PIL import Image

input_file = raw_input('input your file name: ')
basewidth = input('input the width: ')
baseheight = input('input the height: ')
img = Image.open(input_file)

# get rgb value in image
rgb_img = img.convert('RGB')
r, g, b = rgb_img.getpixel((1,1)) # 1st column, 1st row

print 'the first pixel\' RGB value is : ', r, g, b

# resize width using ratio
""""
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
"""

# resize height using ratio
""""
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
"""

print 'image width:',img.size[0], 'image height:',img.size[1]
img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)

img.save('resized_image.bmp')
