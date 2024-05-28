import os 
import shutil

req_imgs = ['IM-000000011378689-AP1.png',
            'IM-000000011378715-AP1.png',
            'IM-000000011378721-AP1.png',
            'IM-000000011378802-AP1.png',
            'IM-000000011378803-AP1.png',
            'IM-000000011378807-AP1.png',
            'IM-000000011378808-AP1.png',
            'IM-000000011378809-AP1.png',
            'IM-000000011378810-AP1.png',
            'IM-000000011378817-AP1.png',
            'IM-000000011378820-AP1.png',
            'IM-000000011378823-AP1.png',
            'IM-000000011378856-AP1.png',
            'IM-000000011378857-AP1.png',
            'IM-000000011378858-AP1.png',
            'IM-000000011378864-AP1.png',
            'IM-000000011378865-AP1.png',
            'IM-000000011378866-AP1.png',
            'IM-000000011378870-AP1.png',
            'IM-000000011378871-AP1.png',
            'IM-000000011378873-AP1.png',
            'IM-000000011378883-AP1.png',
            'IM-000000011378886-AP1.png',
            'IM-000000011379054-AP1.png',
            'IM-000000011379055-AP1.png',
            'IM-000000011379056-AP1.png',
            'IM-000000011387229-AP1.png',
            'IM-000000011387384-AP1.png',
            'IM-000000011387384-AP2.png ',
            'IM-000000011387385-AP1.png',
            'IM-000000011387388-AP1.png',
            'IM-000000011387419-AP1.png',
            'IM-000000011387565-AP1.png',
            'IM-000000011387606-AP1.png',
            'IM-000000011387607-AP1.png',
            'IM-000000011387608-AP1.png',
            'IM-000000011387638-AP1.png',
            'IM-000000011387639-AP1.png',
            'IM-000000011387643-AP1.png'
            ]


row_viz_path = 'row_viz_2'
row_viz = os.listdir(row_viz_path)

bordered_viz = os.path.join(row_viz_path, 'bordered')
os.makedirs(bordered_viz, exist_ok=True)

for img in req_imgs:
    if img in row_viz:
        shutil.copy(
            os.path.join(row_viz_path, img),
            os.path.join(bordered_viz, img)
        )