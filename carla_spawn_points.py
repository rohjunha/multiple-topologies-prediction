import carla

START_POS_DICT_TR = dict()
START_POS_DICT = dict()
START_POS_DICT[0] = [2179] #[2181, 2177, 2173] # [i for i in range(2181, 2171, -2)]
START_POS_DICT[1] = [2297] #[2297, 2293, 2289] # [i for i in range(2297, 2283, -2)]
START_POS_DICT[2] = [2154] #[2154, 2158, 2162] # [i for i in range(2154, 2168, 2)]
START_POS_DICT[3] = [2270] #[2264, 2268, 2272]# [i for i in range(2264, 2278, 2)]

START_POS_DICT_TR[0] = carla.Transform(carla.Location(216.717407, -245.944016, 0.004333), carla.Rotation(360.000000, 359.605499, 0.000))#[2181] #[2181, 2177, 2173] # [i for i in range(2181, 2171, -2)]
START_POS_DICT_TR[1] = carla.Transform(carla.Location(255.303848, -291.853363, 0.019585), carla.Rotation(pitch=360.000000, yaw=90.176750, roll=0.000000))#[2297] #[2297, 2293, 2289] # [i for i in range(2297, 2283, -2)]
START_POS_DICT_TR[2] = carla.Transform(carla.Location(299.386261, -250.013306, 0.004333), carla.Rotation(pitch=0.000000, yaw=179.605499, roll=0.000000))#[2154] #[2154, 2158, 2162] # [i for i in range(2154, 2168, 2)]
START_POS_DICT_TR[3] = carla.Transform(carla.Location(258.552734, -210.456635, 0.019585), carla.Rotation(pitch=0.000000, yaw=-89.823250, roll=0.000000))#[2270] #[2264, 2268, 2272]# [i for i in range(2264, 2278, 2)]

END_POS_DICT = dict()
END_POS_DICT_TR = dict()
END_POS_DICT[0] = 2180 # [i for i in range(2180, 2168, -2)]
END_POS_DICT[1] = 2296 # [i for i in range(2296, 2282, -2)]
END_POS_DICT[2] = 2155 # [i for i in range(2155, 2169, 2)]
END_POS_DICT[3] = 2265 # [i for i in range(2263, 2277, 2)]

END_POS_DICT_TR[0] = carla.Transform(carla.Location(258.803833, -291.842560, 0.019585), carla.Rotation(pitch=0.000000, yaw=-89.823250, roll=0.000000)) # [i for i in range(2180, 2168, -2)]
END_POS_DICT_TR[1] = carla.Transform(carla.Location(299.410370, -246.513397, 0.004333), carla.Rotation(pitch=360.000000, yaw=359.605499, roll=0.000000)) # [i for i in range(2296, 2282, -2)]
END_POS_DICT_TR[2] = carla.Transform(carla.Location(255.006485, -195.467499, 0.019585), carla.Rotation(pitch=360.000000, yaw=90.176750, roll=0.000000)) # [i for i in range(2155, 2169, 2)]
END_POS_DICT_TR[3] = carla.Transform(carla.Location(216.693298, -249.443924, 0.004333), carla.Rotation(pitch=0.000000, yaw=179.605499, roll=0.000000)) # [i for i in range(2263, 2277, 2)]
