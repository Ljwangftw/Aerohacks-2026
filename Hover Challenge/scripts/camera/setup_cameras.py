from pygrabber.dshow_graph import FilterGraph

def get_available_cameras() :

    devices = FilterGraph().get_input_devices()
#{0: 'Brio 101', 1: 'USB2.0 VGA UVC WebCam', 2: 'Brio 101'} just need the brios
    available_cameras = {}
    indexes = []

    camera_name = "brio"

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    print("Cameras found: ")
    print(available_cameras)
    cameras =0
    for index, device_name in enumerate(devices):
        if camera_name in device_name.lower():
            indexes.append(index)
    print("Using camera indexes: ")
    print(indexes)


    return indexes


