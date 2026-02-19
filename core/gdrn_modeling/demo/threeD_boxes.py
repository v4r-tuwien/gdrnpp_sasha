import numpy as np

TRACEBOT_DATA = {
    "tracebotcanister": 
    {
        "diameter": 0.12890791570113228,
        "min_x": -0.025499999523162842,
        "min_y": -0.02607000060379505,
        "min_z": -0.012668999843299389,
        "max_x": 0.025499999523162842,
        "max_y": 0.02493000030517578,
        "max_z": 0.11233100295066833,
        "size_x": 0.050999999046325684,
        "size_y": 0.05100000090897083,
        "size_z": 0.12500000279396772,
        "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}]
    }
}

fac = 0.001
x_minus = TRACEBOT_DATA["tracebotcanister"]['min_x'] * fac
y_minus = TRACEBOT_DATA["tracebotcanister"]['min_y'] * fac
z_minus = TRACEBOT_DATA["tracebotcanister"]['min_z'] * fac
x_plus = TRACEBOT_DATA["tracebotcanister"]['size_x'] * fac + x_minus
y_plus = TRACEBOT_DATA["tracebotcanister"]['size_y'] * fac + y_minus
z_plus = TRACEBOT_DATA["tracebotcanister"]['size_z'] * fac + z_minus

THREED_BOXES = {
    "tracebotcanister": np.array([
        [x_plus,  y_plus,  z_plus],
        [x_plus,  y_plus,  z_minus],
        [x_plus,  y_minus, z_minus],
        [x_plus,  y_minus, z_plus],
        [x_minus, y_plus,  z_plus],
        [x_minus, y_plus,  z_minus],
        [x_minus, y_minus, z_minus],
        [x_minus, y_minus, z_plus]])
}