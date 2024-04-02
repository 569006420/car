home/wei/mycar - рабочая папка автономного вождения, а следующая папка - это каталог папки mycar.
 
logs - Папка журнала 日志文件
models - Обучающие модели 训练模型
tub - Обучающие данные 训练数据，照片和json决策文件
config.py — Это профиль для автономного вождения, который содержит параметры по умолчанию для умных автомобилей, такие как диапазон угла поворота руля, графический размер камеры и т.д.
自动驾驶的配置文件，里面包含小车默认参数，例如转向角度范围、摄像头图形大小等。
1.	import os  
2.	#Каталог хранения данных  数据存放目录
3.	#PATHS  
4.	CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))  
5.	DATA_PATH = os.path.join(CAR_PATH, 'data')  
6.	MODELS_PATH = os.path.join(CAR_PATH, 'models')  
7.	
8.	#VEHICLE  
9.	DRIVE_LOOP_HZ = 20  # Частота камеры 摄像头频率
10.	MAX_LOOPS = 100000  #Скорость выполнения программы  程序运行速率
11.	
12.	#CAMERA  
13.	CAMERA_RESOLUTION = (120, 160) #(height, width)  #Разрешение съемки камеры 摄像头拍摄分辨率
14.	CAMERA_FRAMERATE = DRIVE_LOOP_HZ  
15.	  
16.	#STEERING  
17.	STEERING_CHANNEL = 1  # Канал сервопривода 舵机通道
18.	STEERING_LEFT_PWM = 40  # Предельное значение левого сервопривода 舵机左侧限制值
19.	STEERING_RIGHT_PWM = 150  # Ограничивающее значение для правой стороны сервопривода 舵机右侧限制值
20.	#THROTTLE  
21.	THROTTLE_CHANNEL = 0  # Канал двигателя 电机通道
22.	THROTTLE_FORWARD_PWM = 200  # Максимальное значение ШИМ прямого хода 前进的最大PWM值
23.	THROTTLE_STOPPED_PWM = 100  # Остановить значение PWM 停止PWM值
24.	THROTTLE_REVERSE_PWM = 0  #Значение PWM обратного хода  后退PWM值
25.	
26.	#TRAINING  
27.	BATCH_SIZE = 128  # Количество образцов, отобранных за один раз 一次所选取的样本数
28.	TRAIN_TEST_SPLIT = 0.8 # Пропорциональное значение обучающего набора  训练集比例值
29.	
30.	#JOYSTICK  
31.	USE_JOYSTICK_AS_DEFAULT = False  # Использовать ли режим джойстика по умолчанию 不使用摇杆模式
32.	JOYSTICK_MAX_THROTTLE = 0.25  # Максимальное значение дроссельной заслонки 最大油门值
33.	JOYSTICK_STEERING_SCALE = 1.0  #Значения диапазонов сервокачалки 舵机范围值
34.	AUTO_RECORD_ON_THROTTLE = True  #Автоматическая запись 自动记录
35.	  
36.	  
37.	TUB_PATH = os.path.join(CAR_PATH, 'tub') # Папка для хранения обучающих данных 训练数据存放
Заимствование кодовой ссылки：https://github.com/adison/donkey-car/blob/master/myconfig.py
Изменил некоторые параметры и упростил код.






manage.py — это стартовый файл автономного вождения, точка входа в исходный код автономного вождения. 自动驾驶启动文件，是自动驾驶源码的入口。
1.	#!/usr/bin/env python3  
2.	import os  
3.	from docopt import docopt  
4.	  
5.	#import parts  
6.	from camera import PiCamera  
7.	from transform import Lambda  
8.	from keras import KerasLinear,KerasCategorical  
9.	from actuator import PCA9685, PWMSteering, PWMThrottle  
10.	from datastore import TubGroup, TubWriter  
11.	from controller import LocalWebController, JoystickController  
12.	from clock import Timestamp  
13.	from camera import XRCamera  
14.	  
15.	 # drive()函数  Функция drive()  Запуск управления пулом потоков, получение информации о времени и добавление в пул потоков 启动线程池管理，获取时间信息，并添加到线程池
16.	def drive(cfg, model_path=None, use_joystick=False, use_chaos=False):  
17.	  
18.	    V = vehicle.Vehicle()    # 创建一个车辆  
19.	  
20.	    clock = Timestamp()  
21.	    V.add(clock, outputs='timestamp')  
22.	    #cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)  
23.	    cam = XRCamera(resolution=cfg.CAMERA_RESOLUTION)  
24.	    V.add(cam, outputs=['cam/image_array'], threaded=True)  # Выбор источника видео, т.е. типа камеры, и получение видеопотока для добавления в пул потоков 选择视频源，即摄像头类型，获取视频流添加进线程池
25.	  # use_joystick джойстика 摇杆
26.	  # Запуск управления веб-сервером, добавление информации об изображении входного потока, угол наклона выходного потока, дроссель, режим, запись значений в потоки 启动web服务器控制，将输入流图片信息，输出流角度、油门、模式、记录值添加进线程
27.	    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:  
28.	        ctr = JoystickController(max_throttle=cfg.JOYSTICK_MAX_THROTTLE,  
29.	                                 steering_scale=cfg.JOYSTICK_STEERING_SCALE,  
30.	                                 auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)  
31.	    else:  
32.	        # This web controller will create a web server that is capable  
33.	        # of managing steering, throttle, and modes, and more.  
34.	        ctr = LocalWebController(use_chaos=use_chaos)  
35.	  
36.	    V.add(ctr,  
37.	          inputs=['cam/image_array'],  
38.	          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],  
39.	          threaded=True)  
40.	  
41.	    # See if we should even run the pilot module.  
42.	    # This is only needed because the part run_condition only accepts boolean  
43.	    def pilot_condition(mode):  
44.	        if mode == 'user':  
45.	            return False  
46.	        else:  
47.	            return True  
48.	  
49.	    pilot_condition_part = Lambda(pilot_condition)  
50.	    V.add(pilot_condition_part, inputs=['user/mode'],  
51.	                                outputs=['run_pilot'])  
52.	  # Построение нейросетевой модели 构建神经网络模型
53.	    # Run the pilot if the mode is not user.  
54.	    kl = KerasCategorical()  
55.	    #kl = KerasLinear()  
56.	    if model_path:  
57.	        kl.load(model_path)  
58.	  
59.	    V.add(kl, inputs=['cam/image_array'],  
60.	              outputs=['pilot/angle', 'pilot/throttle'],  
61.	              run_condition='run_pilot')  
62.	  
63.	    # Choose what inputs should change the car.  
64.	    def drive_mode(mode,  
65.	                   user_angle, user_throttle,  
66.	                   pilot_angle, pilot_throttle):  
67.	        if mode == 'user':  
68.	            return user_angle, user_throttle  
69.	  
70.	        elif mode == 'local_angle':  
71.	            return pilot_angle, user_throttle  
72.	  
73.	        else:  
74.	            return pilot_angle, pilot_throttle  
75.	  # Запустите драйверы двигателей и сервоприводов и добавьте их в поток 启动电机和舵机驱动程序，并将其添加进线程
76.	    drive_mode_part = Lambda(drive_mode)  
77.	    V.add(drive_mode_part,  
78.	          inputs=['user/mode', 'user/angle', 'user/throttle',  
79.	                  'pilot/angle', 'pilot/throttle'],  
80.	          outputs=['angle', 'throttle'])  
81.	  
82.	    #steering_controller = PCA9685(cfg.STEERING_CHANNEL)  
83.	    steering = PWMSteering(left_pulse=cfg.STEERING_LEFT_PWM,  
84.	                           right_pulse=cfg.STEERING_RIGHT_PWM)  
85.	  
86.	    #throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)  
87.	    throttle = PWMThrottle(max_pulse=cfg.THROTTLE_FORWARD_PWM,  
88.	                           zero_pulse=cfg.THROTTLE_STOPPED_PWM,  
89.	                           min_pulse=cfg.THROTTLE_REVERSE_PWM)  
90.	  
91.	    V.add(steering, inputs=['angle'])  
92.	    V.add(throttle, inputs=['throttle'])  
93.	  
94.	    # add tub to save data  
95.	    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode', 'timestamp']  
96.	    types = ['image_array', 'float', 'float',  'str', 'str']  
97.	  
98.	    #multiple tubs  
99.	    #th = TubHandler(path=cfg.DATA_PATH)  
100.	    #tub = th.new_tub_writer(inputs=inputs, types=types)  
101.	  # Создание потоков обновления и хранения данных изображений и json-решений, V.start для запуска всех потоков创建图像及json数据更新及存放线程，V.start启动所有线程 
102.	    # single tub  
103.	    tub = TubWriter(path=cfg.TUB_PATH, inputs=inputs, types=types)  
104.	    V.add(tub, inputs=inputs, run_condition='recording')  
105.	  
106.	    # run the vehicle  
107.	    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,  
108.	            max_loop_count=cfg.MAX_LOOPS)  
109.	  
110.	  
111.	def train(cfg, tub_names, new_model_path, base_model_path=None ):  
112.	    X_keys = ['cam/image_array']  
113.	    y_keys = ['user/angle', 'user/throttle']  
114.	    def train_record_transform(record):  
115.	        record['user/angle'] = dk.util.data.linear_bin(record['user/angle'])  
116.	        # TODO add augmentation that doesn't use opencv  
117.	        return record  
118.	  
119.	    def val_record_transform(record):  
120.	        record['user/angle'] = dk.util.data.linear_bin(record['user/angle'])  
121.	        return record  
122.	  
123.	    new_model_path = os.path.expanduser(new_model_path)  
124.	  
125.	    kl = KerasCategorical()  
126.	    if base_model_path is not None:  
127.	        base_model_path = os.path.expanduser(base_model_path)  
128.	        kl.load(base_model_path)  
129.	  
130.	    print('tub_names', tub_names)  
131.	    if not tub_names:  
132.	        tub_names = os.path.join(cfg.DATA_PATH, '*')  
133.	    tubgroup = TubGroup(tub_names)  
134.	    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys,  
135.	                                                    train_record_transform=train_record_transform,  
136.	                                                    val_record_transform=val_record_transform,  
137.	                                                    batch_size=cfg.BATCH_SIZE,  
138.	                                                    train_frac=cfg.TRAIN_TEST_SPLIT)  
139.	  
140.	    total_records = len(tubgroup.df)  
141.	    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)  
142.	    total_val = total_records - total_train  
143.	    print('train: %d, validation: %d' % (total_train, total_val))  
144.	    steps_per_epoch = total_train // cfg.BATCH_SIZE  
145.	    print('steps_per_epoch', steps_per_epoch)  
146.	  
147.	    kl.train(train_gen,  
148.	             val_gen,  
149.	             saved_model_path=new_model_path,  
150.	             steps=steps_per_epoch,  
151.	             train_split=cfg.TRAIN_TEST_SPLIT)  
152.	  
153.	 # Загрузка файлов конфигурации config 加载config配置文件
154.	if __name__ == '__main__':  
155.	    args = docopt(__doc__)  
156.	    cfg = dk.load_config()  
157.	# Переменный параметр [drive], функция drive() активируется в соответствии с параметром [drive], т.е. режимом движения 可变参数[drive],根据带入参数[drive]启动drive()函数即驾驶模式
158.	    if args['drive']:  
159.	        drive(cfg, model_path = args['--model'], use_joystick=args['--js'], use_chaos=args['--chaos'])  
160.	  # Переменный параметр [train], запуск функции train() в соответствии с вводом параметра [train], т.е. режим обучения 可变参数[train],根据带入参数[train]启动train()函数即训练模式
161.	    elif args['train']:  
162.	        tub = args['--tub']  
163.	        new_model_path = args['--model']  
164.	        base_model_path = args['--base_model']  
165.	        cache = not args['--no_cache']  
166.	        train(cfg, tub, new_model_path, base_model_path)  
Заимствование кодовой ссылки：https://github.com/adison/donkey-car/blob/master/manage.py
Удаление и замена некоторых функций по сравнению с исходным кодом, чтобы сделать код более лаконичным











home/wei/parts - Папка для вызова функций модуля 调用函数模块
 

camera.py - Другие типы классов драйверов камер для вызова 调用摄像头驱动函数

1.	import os  
2.	import time  
3.	import numpy as np  
4.	from PIL import Image  
5.	import glob  
6.	  
7.	class BaseCamera:  
8.	  
9.	    def run_threaded(self):  
10.	        return self.frame  
11.	 # Класс камеры Raspberry Pi 树莓派摄像头类
12.	class PiCamera(BaseCamera):  
13.	    def __init__(self, resolution=(120, 160), framerate=20):  
14.	        from picamera.array import PiRGBArray  
15.	        from picamera import PiCamera  
16.	        resolution = (resolution[1], resolution[0])  
17.	        # initialize the camera and stream  
18.	        self.camera = PiCamera()  
19.	        self.camera.resolution = resolution  
20.	        self.camera.framerate = framerate  
21.	        self.rawCapture = PiRGBArray(self.camera, size=resolution)  
22.	        self.stream = self.camera.capture_continuous(self.rawCapture,  
23.	            format="rgb", use_video_port=True)  
24.	  
25.	        self.frame = None  
26.	        self.on = True  
27.	  
28.	        print('PiCamera loaded.. .warming camera')  
29.	        time.sleep(2)  
30.	  
31.	  
32.	    def run(self):  
33.	        f = next(self.stream)  
34.	        frame = f.array  
35.	        self.rawCapture.truncate(0)  
36.	        return frame  
37.	  
38.	    def update(self):    
39.	        for f in self.stream:  
40.	            self.frame = f.array  
41.	            self.rawCapture.truncate(0)  
42.	  
43.	            if not self.on:  
44.	                break  
45.	  
46.	    def shutdown(self):  
47.	        self.on = False  
48.	        print('stoping PiCamera')  
49.	        time.sleep(.5)  
50.	        self.stream.close()  
51.	        self.rawCapture.close()  
52.	        self.camera.close()  
53.	  # Класс веб-камер Web摄像头类
54.	class Webcam(BaseCamera):  
55.	    def __init__(self, resolution = (160, 120), framerate = 20):  
56.	        import pygame  
57.	        import pygame.camera  
58.	  
59.	        super().__init__()  
60.	  
61.	        pygame.init()  
62.	        pygame.camera.init()  
63.	        l = pygame.camera.list_cameras()  
64.	        self.cam = pygame.camera.Camera(l[0], resolution, "RGB")  
65.	        self.resolution = resolution  
66.	        self.cam.start()  
67.	        self.framerate = framerate  
68.	  
69.	        self.frame = None  
70.	        self.on = True  
71.	  
72.	        print('WebcamVideoStream loaded.. .warming camera')  
73.	  
74.	        time.sleep(2)  
75.	  
76.	    def update(self):  
77.	        from datetime import datetime, timedelta  
78.	        import pygame.image  
79.	        while self.on:  
80.	            start = datetime.now()  
81.	  
82.	            if self.cam.query_image():  
83.	                snapshot = self.cam.get_image()  
84.	                snapshot1 = pygame.transform.scale(snapshot, self.resolution)  
85.	                self.frame = pygame.surfarray.pixels3d(pygame.transform.rotate(pygame.transform.flip(snapshot1, True, False), 90))  
86.	  
87.	            stop = datetime.now()  
88.	            s = 1 / self.framerate - (stop - start).total_seconds()  
89.	            if s > 0:  
90.	                time.sleep(s)  
91.	  
92.	        self.cam.stop()  
93.	  
94.	    def run_threaded(self):  
95.	        return self.frame  
96.	  
97.	    def shutdown(self):  
98.	        self.on = False  
99.	        print('stoping Webcam')  
100.	        time.sleep(.5)  
101.	  # Класс виртуальной камеры 虚拟相机类
102.	class MockCamera(BaseCamera):  
103.	    """ 
104.	    Fake camera. Returns only a single static frame 
105.	    """  
106.	    def __init__(self, resolution=(160, 120), image=None):  
107.	        if image is not None:  
108.	            self.frame = image  
109.	        else:  
110.	            self.frame = Image.new('RGB', resolution)  
111.	  
112.	    def update(self):  
113.	        pass  
114.	  
115.	    def shutdown(self):  
116.	        pass  
117.	# Использование локальных данных изображения в качестве потока данных камеры 将本地图片数据作为相机数据流
118.	class ImageListCamera(BaseCamera):  
119.	    def __init__(self, path_mask='~/mycar/data/**/*.jpg'):  
120.	        self.image_filenames = glob.glob(os.path.expanduser(path_mask), recursive=True)  
121.	  
122.	        def get_image_index(fnm):  
123.	            sl = os.path.basename(fnm).split('_')  
124.	            return int(sl[0])  
125.	  
126.	  
127.	        self.image_filenames.sort(key=get_image_index)  
128.	        #self.image_filenames.sort(key=os.path.getmtime)  
129.	        self.num_images = len(self.image_filenames)  
130.	        print('%d images loaded.' % self.num_images)  
131.	        print( self.image_filenames[:10])  
132.	        self.i_frame = 0  
133.	        self.frame = None  
134.	        self.update()  
135.	  
136.	    def update(self):  
137.	        pass  
138.	  
139.	    def run_threaded(self):  
140.	        if self.num_images > 0:  
141.	            self.i_frame = (self.i_frame + 1) % self.num_images  
142.	            self.frame = Image.open(self.image_filenames[self.i_frame])  
143.	  
144.	        return np.asarray(self.frame)  
145.	  
146.	    def shutdown(self):  
147.	        pass  
Заимствование кодовой ссылки：https://github.com/meigrafd/Sample-Code/blob/master/_Tkinter/picamera3.py
Немного измененная информация, такая как путь сохранения фотографий, по сравнению с исходным кодом


clock.py - Класс datetime() модуля datetime в Python.
1.	import datetime  
2.	  
3.	class Timestamp():  
4.	  
5.	    def run(self,):  
6.	        return str(datetime.datetime.utcnow())  

Заимствование кодовой ссылки：https://docs-python.ru/standart-library/modul-datetime-python/klass-datetime-modulja-datetime/





controller.py - Модуль привода джойстика摇杆驱动模块
1.	  
2.	  # Класс привода джойстика 摇杆操纵类
3.	class Joystick():  
4.	    access_url = None #required to be consistent with web controller  
5.	  
6.	    def __init__(self, dev_fn='/dev/input/js0'):  
7.	        self.axis_states = {}  
8.	        self.button_states = {}  
9.	        self.axis_map = []  
10.	        self.button_map = []  
11.	        self.jsdev = None  
12.	        self.dev_fn = dev_fn  
13.	  
14.	        # These constants were borrowed from linux/input.h  
15.	        self.axis_names = {  
16.	            0x00 : 'x',  
17.	            0x01 : 'y',  
18.	            0x02 : 'z',  
19.	            0x03 : 'rx',  
20.	            0x04 : 'ry',  
21.	            0x05 : 'rz',  
22.	            0x06 : 'trottle',  
23.	            0x07 : 'rudder',  
24.	            0x08 : 'wheel',  
25.	            0x09 : 'gas',  
26.	            0x0a : 'brake',  
27.	            0x10 : 'hat0x',  
28.	            0x11 : 'hat0y',  
29.	            0x12 : 'hat1x',  
30.	            0x13 : 'hat1y',  
31.	            0x14 : 'hat2x',  
32.	            0x15 : 'hat2y',  
33.	            0x16 : 'hat3x',  
34.	            0x17 : 'hat3y',  
35.	            0x18 : 'pressure',  
36.	            0x19 : 'distance',  
37.	            0x1a : 'tilt_x',  
38.	            0x1b : 'tilt_y',  
39.	            0x1c : 'tool_width',  
40.	            0x20 : 'volume',  
41.	            0x28 : 'misc',  
42.	        }  
43.	  # Пары клавиш джойстика 手柄键值对
44.	        self.button_names = {  
45.	            0x120 : 'trigger',  
46.	            0x121 : 'thumb',  
47.	            0x122 : 'thumb2',  
48.	            0x123 : 'top',  
49.	            0x124 : 'top2',  
50.	            0x125 : 'pinkie',  
51.	            0x126 : 'base',  
52.	            0x127 : 'base2',  
53.	            0x128 : 'base3',  
54.	            0x129 : 'base4',  
55.	            0x12a : 'base5',  
56.	            0x12b : 'base6',  
57.	  
58.	            #PS3 sixaxis specific  
59.	            0x12c : "triangle",  
60.	            0x12d : "circle",  
61.	            0x12e : "cross",  
62.	            0x12f : 'square',  
63.	  
64.	            0x130 : 'a',  
65.	            0x131 : 'b',  
66.	            0x132 : 'c',  
67.	            0x133 : 'x',  
68.	            0x134 : 'y',  
69.	            0x135 : 'z',  
70.	            0x136 : 'tl',  
71.	            0x137 : 'tr',  
72.	            0x138 : 'tl2',  
73.	            0x139 : 'tr2',  
74.	            0x13a : 'select',  
75.	            0x13b : 'start',  
76.	            0x13c : 'mode',  
77.	            0x13d : 'thumbl',  
78.	            0x13e : 'thumbr',  
79.	  
80.	            0x220 : 'dpad_up',  
81.	            0x221 : 'dpad_down',  
82.	            0x222 : 'dpad_left',  
83.	            0x223 : 'dpad_right',  
84.	  
85.	            # XBox 360 controller uses these codes.  
86.	            0x2c0 : 'dpad_left',  
87.	            0x2c1 : 'dpad_right',  
88.	            0x2c2 : 'dpad_up',  
89.	            0x2c3 : 'dpad_down',  
90.	        }  
91.	  
92.	  #Классы управления джойстиком и преобразования данных 摇杆操作及数据转换类
93.	class JoystickController(object):  
94.	    def __init__(self, poll_delay=0.0,  
95.	                 max_throttle=1.0,  
96.	                 steering_axis='x',  
97.	                 throttle_axis='rz',  
98.	                 steering_scale=1.0,  
99.	                 throttle_scale=-1.0,  
100.	                 dev_fn='/dev/input/js0',  
101.	                 auto_record_on_throttle=True):  
102.	  
103.	  # Ожидание доступа к устройству 等待设备接入
104.	    def update(self):  
105.	        while self.running and not self.init_js():  
106.	            time.sleep(5)  
107.	  
108.	        while self.running:  
109.	            button, button_state, axis, axis_val = self.js.poll()  
110.	             # Получение значения угла и дроссельной заслонки 获取角度和油门值
111.	            if axis == self.steering_axis:  
112.	                self.angle = self.steering_scale * axis_val  
113.	                print("angle", self.angle)  
114.	  
115.	            if axis == self.throttle_axis:  
116.	                self.throttle = (self.throttle_scale * axis_val * self.max_throttle)  
117.	                print("throttle", self.throttle)  
118.	                self.on_throttle_changes()  
119.	              # События клавиш джойстика 手柄按键事件
120.	            if button == 'trigger' and button_state == 1:  
121.	                if self.mode == 'user':  
122.	                    self.mode = 'local_angle'  
123.	                elif self.mode == 'local_angle':  
124.	                    self.mode = 'local'  
125.	                else:  
126.	                    self.mode = 'user'  
127.	                print('new mode:', self.mode)  
128.	  
129.	            if button == 'circle' and button_state == 1:  
130.	                if self.auto_record_on_throttle:  
131.	                    print('auto record on throttle is enabled.')  
132.	                elif self.recording:  
133.	                    self.recording = False  
134.	                else:  
135.	                    self.recording = True  
136.	  
137.	                print('recording:', self.recording)  
138.	  
139.	 
140.	  # Функция обновления данных клавиш джойстика 手柄数据更新函数
141.	    def run_threaded(self, img_arr=None):  
142.	        self.img_arr = img_arr  
143.	        return self.angle, self.throttle, self.mode, self.recording  
144.	  
Заимствование кодовой ссылки： - https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/controller.py
Полностью заимствована из исходного кода и протестирована для работы только с джойстиком PS4. Попытался упростить код, но не смог.

datastore.py - Модуль хранения данных, снимки с камеры, дистанционное управление и параметры управления необходимо сохранить в соответствующих папках. 数据存储模块，来自摄像头，遥控器的图片及控制参数需要存储到对应的文件夹中。

# Конструирование классов структуры набора данных 构建数据集结构类
class Tub(object):  
# Определение существования папки tub, если она существует, определение существования файла mete.json, если нет, создание нового файла mete.json 判断tub文件夹是否存在，如存在则判断mete.json文件是否存在，不存在则创建新的mete.json文件
1.	if exists:  
2.	    logger.info('Tub exists: {}'.format(self.path))  
3.	    if os.path.exists(self.meta_path):  
4.	        with open(self.meta_path, 'r') as f:  
5.	            self.meta = json.load(f)  
6.	        self.current_ix = self.get_last_ix() + 1  
7.	    else:  
8.	        self.meta = {'inputs': inputs, 'types': types}  
9.	        with open(self.meta_path, 'w') as f:  
10.	            json.dump(self.meta, f)  
11.	        self.current_ix = 0  
12.	elif not exists and inputs:  
13.	    logger.info('Tub does NOT exist. Creating new tub...')  
14.	    os.makedirs(self.path, 0o777)  
15.	    self.meta = {'inputs': inputs, 'types': types}  
16.	    with open(self.meta_path, 'w') as f:  
17.	        json.dump(self.meta, f)  
18.	    self.current_ix = 0  
19.	    logger.info('New tub created at: {}'.format(self.path))  
# Запись данных в json-файл 向json文件中写入数据
def write_json_record(self, json_data):  
# Чтение данных из json-файла 读取json文件中数据
def get_json_record(self, ix):  
# Запись данных изображения в локальную папку将图像数据写入本地文件夹
def put_record(self, data):  
# Чтение данных изображения 读取图片数据
def read_record(self, record_dict):  
# Возврат к обучающему проверочному набору 返回训练验证集
def get_train_gen(self, X_keys, Y_keys, batch_size=128, record_transform=None, df=None):  
# Запись данных в папку tub 向tub文件夹中写入数据 
class TubWriter(Tub):  
# Считывание данных из папки tub 读取tub文件夹中数据 
class TubReader(Tub):  
class TubHandler(): 
Заимствование кодовой ссылки：https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/datastore.py
Заимствовано из исходного кода, но значительно оптимизировано

imu.py- Приводной модуль MPU6050 MPU6050驱动模块
Заимствование кодовой ссылки：https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/imu.py
Я хотел добавить модуль датчика для определения ускорения умного автомобиля для предотвращения столкновений, обхода препятствий и т.д. Однако это не увенчалось успехом. Поэтому пока эта функция модуля не используется.

encoder.py - Драйвер подключен к микроконтроллеру 连接到微控制器的驱动程序
Заимствование кодовой ссылки：https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/encoder.py
Не используется.

lidar.py – радар 激光雷达
Заимствование кодовой ссылки：https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/lidar.pyhttps://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/lidar.py
Нет радара

simulation.py – Симулятор 模拟器
Заимствование кодовой ссылки：https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/simulation.py
Использование симулятора для сбора данных, обучения модели и последующего плохого тестирования. Заброшенное использование.

teensy.py – Плата разработки Teensy USB开发板
Заимствование кодовой ссылки：https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/teensy.py
Не используется плата разработки Teensy USB.

transform.py - Классы PID-регуляторов для Python PID 控制器类
Заимствование кодовой ссылки：https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/transform.py
Выполняет расчет PID-регулятора и возвращает управляющее значение. Оно основывается на прошедшем времени (dt) и текущем значении переменной процесса. 执行 PID 计算并返回控制值。 它基于经过时间 (dt) 和过程变量的当前值。






keras.py - Этот модуль является основным для автономного вождения и использует алгоритмы глубокого обучения. 这个模块是自动驾驶的核心模块，采用深度学习算法。
Здесь указывается количество слоев, количество и тип слоев в нейронной сети. Он также включает решения более низкого уровня, такие как выбор функции потерь, функции активации, процесса оптимизации и количества циклов. 其中包含神经网络中的层数，数量和类型。还包括较低级别的决策，如选择损失函数，激活函数，优化过程和周期数。

1.	from tensorflow.python.keras.layers import Input  
2.	from tensorflow.python.keras.models import Model, load_model  
3.	from tensorflow.python.keras.layers import Convolution2D  
4.	from tensorflow.python.keras.layers import Dropout, Flatten, Dense  
5.	from tensorflow.python.keras.callbacks import TensorBoard  
6.	from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping  
7.	import matplotlib.pyplot as plt  
8.	import numpy as np  
9.	import tensorflow as tf  
10.	  
11.	  
12.	  # Часть обучающих данных разбивается на проверочный набор данных, а затем производительность этого проверочного набора данных оценивается каждый цикл. 将训练数据的一部分分成验证数据集，然后评估每个周期该验证数据集的性能。
13.	class KerasPilot:  
14.	  
15.	    def load(self, model_path):  
16.	        self.model = load_model(model_path)  
17.	  
18.	    def shutdown(self):  
19.	        pass  
20.	  
21.	    def train(self, train_gen, val_gen,  
22.	              saved_model_path, epochs=100, steps=100, train_split=0.8,  
23.	              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):  
24.	  
25.	        # checkpoint to save model after each epoch  
26.	        save_best = ModelCheckpoint(saved_model_path,  
27.	                                    monitor='val_loss',  
28.	                                    verbose=verbose,  
29.	                                    save_best_only=True,  
30.	                                    mode='min')  
31.	   #    monitor: значение, которое необходимо отслеживать 需要监视的值
32.	        verbose: режим представления информации, 0 или 1 (информация о сохранении контрольной точки, аналогично Epoch 00001: сохранение модели в ...) 信息展示模式，0或1(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
33.	        save_best_only: если установлено значение True, текущая модель будет сохранена только в том случае, если наблюдаемое значение улучшилось 当设置为True时，监测值有改进时才会保存当前的模型
34.	        mode: один из 'auto', 'min', 'max', когда save_best_only=True определяет критерий наиболее эффективной модели, например, когда значение мониторинга равно val_acc, режим должен быть max, а когда значение мониторинга val_loss , режим должен быть min. В автоматическом режиме критерии оценки автоматически выводятся из имен контролируемых значений. 在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
35.	  
36.	        # stop training if the validation error stops improving.  
37.	        early_stop = EarlyStopping(monitor='val_loss',  
38.	                                   min_delta=min_delta,  
39.	                                   patience=patience,  
40.	                                   verbose=verbose,  
41.	                                   mode='auto')  
42.	  #  monitor: интерфейс данных для мониторинга, с 'acc', 'val_acc', 'loss', 'val_ потеря" и т.д. Обычно, если есть набор валидации, используется 'val_acc' или 'val_loss'. 监控的数据接口，有’acc’,’val_acc’,’loss’,’val_loss’等等。正常情况下如果有验证集，就用’val_acc’或者’val_loss’
43.	 min_delta: порог для увеличения или уменьшения, только часть, превышающая это значение, считается улучшением. размер этого значения зависит от monitor.   min_delta：增大或减小的阈值，只有大于这个部分才算作improvement。这个值的大小取决于monitor
44.	  patience：Эта настройка фактически является компромиссом между дрожанием и реальным снижением точности. если терпение установлено большим, то конечная точность будет немного ниже, чем максимальная точность, которую может достичь модель. если терпение установлено маленьким, то модель, скорее всего, будет дрожать на ранних стадиях и остановится на полном поиске графа, что, как правило, плохо.         Если терпение задано небольшим, то модель, скорее всего, будет дрожать на ранних стадиях и останавливаться, когда она еще находится на стадии поиска полного графа, и точность будет в целом низкой. размер терпения напрямую связан со скоростью обучения. Если скорость обучения задана, модель следует обучить несколько раз, чтобы наблюдать за количеством epoch, и установить терпение немного больше, чем это, и немного меньше, чем максимальное количество epoch  при изменении скорости обучения. 能够容忍多少个epoch内都没有improvement。这个设置其实是在抖动和真正的准确率下降之间做tradeoff。如果patience设的大，那么最终得到的准确率要略低于模型可以达到的最高准确率。如果patience设的小，那么模型很可能在前期抖动，还在全图搜索的阶段就停止了，准确率一般很差。patience的大小和learning rate直接相关。在learning rate设定的情况下，前期先训练几次观察抖动的epoch number，比其稍大些设置patience。在learning rate变化的情况下，建议要略小于最大的抖动epoch number。
45.	mode: один из 'auto', 'min', 'max', тренировка в режиме min и прекращение тренировки, если значение обнаружения перестает падать. В максимальном режиме обучение прекращается, когда значение обнаружения больше не увеличивается. 'auto','min','max'之一，在min模式训练，如果检测值停止下降则终止训练。在max模式下，当检测值不再上升的时候则停止训练。
46.	  # Функции обратного вызова
47.	        callbacks_list = [save_best]  
48.	  
49.	        if use_early_stop:  
50.	            callbacks_list.append(early_stop)  
51.	  
52.	        hist = self.model.fit_generator(  
53.	            train_gen,  
54.	            steps_per_epoch=steps,  
55.	            epochs=epochs,  
56.	            verbose=1,  
57.	            validation_data=val_gen,  
58.	            callbacks=callbacks_list,  
59.	            validation_steps=steps * (1.0 - train_split) / train_split)  
60.	        return hist  
61.	  #   generator: функция генератора, выход генератора должен быть: tuple вида (inputs, targets,sample_weight). все возвращаемые значения должны содержать одинаковое количество выборок. Генератор будет бесконечно перебирать набор данных. Количество выборок, проходящих через модель, достигает samples_per_epoch для каждой epoch, отметить конец epoch. 生成器函数，生成器的输出应该为：一个形如（inputs, targets,sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
62.	  steps_per_epoch：Целое число, когда генератор возвращает значение steps_per_epoch, одна epoch завершается и выполняется следующая epoch. 整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
63.	epochs：Целое число, количество раундов итерации данных 整数，数据迭代的轮数
64.	verbose：Отображение журнала, 0 - отсутствие сообщений журнала в стандартном потоке вывода, 1 - вывод записей прогресс-бара, 2 - вывод одной строки за epoch 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
65.	validation_data: имеет одну из трех форм: генератор, создающий набор для проверки; tuple формы (inputs,targets); tuple формы (inputs,targets,sample_weights). 具有以下三种形式之一:生成验证集的生成器;一个形如（inputs,targets）的tuple; 一个形如（inputs,targets，sample_weights）的tuple
66.	validation_steps: когда validation_data является генератором, этот параметр определяет количество возвратов генератора для набора валидации 当validation_data为生成器时，本参数指定验证集的生成器返回次数
67.	class KerasLinear(KerasPilot):  
68.	    def __init__(self, model=None, num_outputs=None, *args, **kwargs):  
69.	        super(KerasLinear, self).__init__(*args, **kwargs)  
70.	        if model:  
71.	            self.model = model  
72.	        elif num_outputs is not None:  
73.	            self.model = default_linear()  
74.	        else:  
75.	            self.model = default_linear()  
76.	  
77.	    def run(self, img_arr):  
78.	        img_arr = img_arr.reshape((1,) + img_arr.shape)  # reshape() - это метод в массиве array, который служит для изменения формы данных.
79.	        outputs = self.model.predict(img_arr)  
80.	        # print(len(outputs), outputs)  
81.	        steering = outputs[0]  
82.	        throttle = outputs[1]  
83.	        return steering[0][0], throttle[0][0]  
84.	  
85.	  
86.	def default_linear():  
87.	    img_in = Input(shape=(120, 160, 3), name='img_in')  
88.	    x = img_in  
89.	  
90.	    # Convolution2D class name is an alias for Conv2D  
91.	    # filters представляет количество ядер свертки, kernel_size размер ядра свертки, strides шаг, Структура нейронной сети на РПЗ странице 35, Функция активации использует elu.
92.	    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu')(x)  
93.	    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='elu')(x)  
94.	    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='elu')(x)  
95.	    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='elu')(x)  
96.	    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu')(x)  
97.	# Flatten слои используются для "уплощения" входных данных, т.е. для того, чтобы сделать многомерные данные одномерными, и часто используются при переходе от сверточных к полносвязным слоям. Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
98.	    x = Flatten(name='flattened')(x)  
99.	    x = Dense(units=100, activation='linear')(x)     # Предотвращение чрезмерной подгонки, units: нейроны в слое, activation: функция активации, используемая в слое 防止过拟合，神经元和激活函数
100.	    x = Dropout(rate=.1)(x)  
101.	    x = Dense(units=50, activation='linear')(x)  
102.	    x = Dropout(rate=.1)(x)  
103.	    # categorical output of the angle  
104.	    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)  
105.	  
106.	    # continous output of throttle  
107.	    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)  
108.	    sess = tf.Session()  
109.	    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])  
110.	    Tensorboard = TensorBoard(log_dir="/home/wei/parts/logs", histogram_freq=1, write_grads=True)  #Вызов инструмента Tensorboard 调用Tensorboard
111.	  # Целевая функция目标函数
112.	    model.compile(optimizer='adam',  
113.	                  loss={'angle_out': 'mean_squared_error',  
114.	                        'throttle_out': 'mean_squared_error'},  
115.	                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})  
116.	    with tf.name_scope('train'):  
117.	        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=angle_out,  
118.	                                                              logits=logits)  
119.	        loss = tf.reduce_mean(xentropy,name='loss')  
120.	        optimizer = tf.train.AdamOptimizer()  
121.	        training_op = optimizer.minimize(loss)  
122.	  
123.	 #   tf.summary.scalar("angle_out", angle_out )  
124.	 #   tf.summary.scalar("throttle_out", throttle_out )  
125.	    tf.summary.scalar("loss", loss )  
126.	    merge_summary = tf.summary.merge_all()  
127.	    writer = tf.summary.FileWriter("/home/wei/mycar/logs", sess.graph)  
128.	    for i in range(100):  
129.	        train_summary = sess.run(merge_summary, feed_dict= {loss:i+1})  
130.	        writer.add_summary(train_summary, i)  
131.	    #history=model.fit(steps_per_epoch=steps_per_epoch, callbacks=[Tensorboard])  
132.	  
133.	  
134.	    return model  



 
web.py — это модуль запуска веб-сервера и передачи данных. Для управления умным автомобилем через веб-управление необходимо создать небольшой сервер, который можно открыть браузером, набрав адрес сервера. Сервер настроен с использованием фреймворка tornado, который представляет собой легкий веб-фреймворк с асинхронной неблокируемой обработкой ввода-вывода. 是web网络服务器启动以及数据传输模块，通过 web 控制的方式来操控智能车，那么就需要架设一个小型服务器，让浏览器输入服务器地址既可打开。架设服务器使用的是 tornado 框架，作为 Web 框架，是一个轻量级的 Web 框架，其拥有异步非阻塞 IO 的处理方式。

        # Настройка класса веб-сервера, Установить порта и ip-адрес
1.	class LocalWebController(tornado.web.Application):  
2.	    port = 8887  
3.	    def __init__(self, use_chaos=False):  
4.	        """ 
5.	        Create and publish variables needed on many of 
6.	        the web handlers. 
7.	        """  
8.	        print('Starting Donkey Server...')  
9.	  
10.	        this_dir = os.path.dirname(os.path.realpath(__file__))  
11.	        self.static_file_path = os.path.join(this_dir, 'templates', 'static')  
12.	  
13.	        self.angle = 0.0  
14.	        self.throttle = 0.0  
15.	        self.mode = 'user'  
16.	        self.recording = False  
17.	        self.ip_address = web.get_ip_address()  
18.	        self.access_url = 'https://{}:{}'.format(self.ip_address, self.port)  

   	 # Привязка веб-обработчиков
1.	handlers = [  
2.	    (r"/", tornado.web.RedirectHandler, dict(url="/drive")),  
3.	    (r"/drive", DriveAPI),  
4.	    (r"/video", VideoAPI),  
5.	    (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": self.static_file_path}),  
6.	]  

# Запустить слушателя
1.	def update(self):  
2.	    """ Start the tornado web server. """  
3.	    self.port = int(self.port)  
4.	    server = httpserver.HTTPServer(self, ssl_options={  
5.	            "certfile": os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.crt"),  
6.	            "keyfile": os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.key"),  
7.	    })  
8.	    server.listen(self.port)  
9.	    instance = tornado.ioloop.IOLoop.instance()  
10.	    instance.add_callback(self.say_hello)  
11.	    instance.start() 

# Веб-передача предоставленных данных
1.	class DriveAPI(tornado.web.RequestHandler):  
2.	    def get(self):  
3.	        data = {}  
4.	        self.render("templates/vehicle.html", **data)  
5.	  
6.	    def post(self):  
7.	        """ 
8.	        Receive post requests as user changes the angle 
9.	        and throttle of the vehicle on a the index webpage 
10.	        """  
11.	        data = tornado.escape.json_decode(self.request.body)  
12.	        self.application.angle = data['angle']  
13.	        self.application.throttle = data['throttle']  
14.	        self.application.mode = data['drive_mode']  
15.	        self.application.recording = data['recording']  


Заимствование кодовой ссылки：https://github.com/johnisanerd/donkey/blob/master/donkeycar/parts/web_controller/web.py
В исходный код были внесены значительные изменения для создания небольшого сервера, чтобы умный автомобиль мог связаться с Raspberry Pi и открыть веб-страницу для управления умным автомобилем для сбора данных даже без сети.

