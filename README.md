# Stream4NVDS

A programmable template of streaming application using NVIDIA DeepStream platform.

This repositiory does not have any inference module, it only has the module 'pulling streaming'.

## Access the image by numpy.ndarray.

You can access the RGBA image 'rgba_img' in the member function 'tiler_sink_pad_buffer_probe()' of object 'DeepStreamApp' which saved in the [src/streamer.py](src/streamer.py).

And you can implement your customized flow of code in the [src/streamer.py](src/streamer.py) or [src/process.py](src/process.py).

## Reference.

* [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).
* [The example 'deepstream-imagedata-multistream' in deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-imagedata-multistream).
* [The parameter 'drop-frame-interval'](https://forums.developer.nvidia.com/t/drop-frame-interval-explained/234423).
