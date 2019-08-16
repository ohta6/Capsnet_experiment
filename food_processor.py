#encode:UTF-8
import json
from os.path import join
from os import makedirs
import cv2

SIZE = 96
img_dir = "food-101/images"
train_dir = "food-101/train"+str(SIZE)
test_dir = "food-101/test"+str(SIZE)
foods = ['foie_gras', 'cheese_plate', 'cannoli', 'crab_cakes', 'garlic_bread', 'hamburger', 'steak', 'risotto', 'takoyaki', 'frozen_yogurt']
with open("food-101/meta/train.json", "r") as f:
    train = json.load(f)
    for food in foods:
        food_dir = join(train_dir, food)
        makedirs(food_dir)
        for i, t in enumerate(train[food]):
            img_path = join(img_dir, t+'.jpg')
            img = cv2.imread(img_path)
            # 処理
            w = img.shape[0]
            h = img.shape[1]
            if w == h:
                img = cv2.resize(img, dsize=(SIZE, SIZE))
            elif w > h:
                img = cv2.resize(img[(w-h)//2:(w+h)//2, :, :], dsize=(SIZE, SIZE))
            elif w < h:
                img = cv2.resize(img[:, (h-w)//2:(h+w)//2, :], dsize=(SIZE, SIZE))
            cv2.imwrite(join(train_dir, food, str(i)+'.jpg'), img)
            


with open("food-101/meta/test.json", "r") as f:
    test = json.load(f)
    for food in foods:
        food_dir = join(test_dir, food)
        makedirs(food_dir)
        for i, t in enumerate(test[food]):
            img_path = join(img_dir, t+'.jpg')
            img = cv2.imread(img_path)
            # 処理
            w = img.shape[0]
            h = img.shape[1]
            if w == h:
                img = cv2.resize(img, dsize=(SIZE, SIZE))
            elif w > h:
                img = cv2.resize(img[(w-h)//2:(w+h)//2, :, :], dsize=(SIZE, SIZE))
            elif w < h:
                img = cv2.resize(img[:, (h-w)//2:(h+w)//2, :], dsize=(SIZE, SIZE))
            cv2.imwrite(join(test_dir, food, str(i)+'.jpg'), img)

