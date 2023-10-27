# source: https://forums.developer.nvidia.com/t/stop-inference-on-deepstream-python-example/167783
################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import re
import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib
from gi.repository import GstRtspServer
from ctypes import *
import sys
from common.is_aarch_64 import is_aarch64
from common.FPS import GETFPS
import numpy as np
import pyds
import cv2
import time
import datetime
from multiprocessing import  Process, Pipe


# customized python modules.
import logproc
import process


class DeepStreamApp(process.Methods):
    """
    The customized application for Single RTSP used.

    p.s. The personal method/flow/algorithum is defined in the Methods.dispose().

    local_args: it loads from json file.
    """
    def __init__(self, local_args: dict, gpu_index: int):
        self.__local_args = local_args.copy()
        super().__init__(self.__local_args, gpu_index)

        self.__fps_streams = {}
        self.__app_logger = logproc.Logger(self.__local_args["kernel"]["log"]["saved_dir"],
                                           self.__local_args["kernel"]["log"]["title_name"])
        # Standard GStreamer initialization
        #GObject.threads_init()
        Gst.init(None)

        # load deepstream version.
        ds_version_path = "/opt/nvidia/deepstream/deepstream/version"
        with open(ds_version_path, newline='') as csvfile:
            text_data = csvfile.read()
        self.__deepstream_version = re.split(': |\n', text_data)[1]
        self.__app_logger.write("gstreamerproc", "App4RTSP.__init__", "Info", 
                                "DeepStream version is " + str(self.__deepstream_version) + ".")

        # Create gstreamer elements */
        # Create Pipeline element that will form a connection of other elements
        self.__app_logger.write("gstreamerproc", "App4RTSP.__init__", "Info", "Creating Pipeline.")
        self.__pipeline = Gst.Pipeline()

        if not self.__pipeline:
            self.__app_logger.write("gstreamerproc", "App4RTSP.__init__", "Info", "Unable to create Pipeline.")
            sys.exit(0)

        # https://forums.developer.nvidia.com/t/how-to-use-the-cv2-imshow-function-with-deepstream-sdk-5-0/121792/21
        self.__recv_debug_info2buf, self.__send_debug_info2buf = None, None
        if self.__local_args["kernel"]["debug_mode"]:
            self.__app_logger.write("gstreamerproc", "App4RTSP.__init__", "Debug", "Creating debug process.")
            self.__recv_debug_info2buf, self.__send_debug_info2buf = Pipe()  

        self.__abnormal_input = False

    def host_debug_rtspin(self, recv_frame):
        """
        used for debug mode.
        """
        while True:
            frame = recv_frame.recv()
            if frame[0] is not None:
                cv2.imwrite("detection_eqrectimg.bmp",frame[0])

            if frame[1] is not None:
                cv2.imwrite("postproc_eqrectimg.bmp",frame[1])

    def get_ds_frame(self, gst_buffer, frame_meta) -> np.ndarray:
        # the input should be address of buffer and batch_id
        # p.s. the 'n_frame' is returning pointer, and it will have same memory address as image in frame_meta.
        return pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)

    def tiler_sink_pad_buffer_probe(self, pad,info,u_data):
        """
        tiler_sink_pad_buffer_probe() will extract metadata received on tiler src pad
        and update params for drawing rectangle, object information etc.
        """
        #print("callback")
        gst_buffer = info.get_buffer()

        if not gst_buffer:
            print("Unable to get GstBuffer up to 5 secs, PIPELINE NEED TO RESET\n")
            self.__pipeline.send_event(Gst.Event.new_eos())
            return
        
        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        
        l_frame = batch_meta.frame_meta_list
        #while l_frame is not None:

        while l_frame is not None:
            #try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            self.__fps_streams["stream{0}".format(frame_meta.source_id)].update_fps()
            stream_fps = self.__fps_streams["stream{0}".format(frame_meta.source_id)].get_fps()

            if frame_meta.frame_num >= self.__local_args["kernel"]["wait_buffer"]:
                # convert to RGBA with numpy.ndarray.
                rgba_img = self.get_ds_frame(gst_buffer, frame_meta)
    
                # your customized/defined flow.
                #=============================================================
                info_dict = self.dispose({ "rgba_img": rgba_img,
                                           "stream_fps": stream_fps,
                                           "debug_mode": self.__local_args["kernel"]["debug_mode"] })

                # debug in host.
                if self.__local_args["kernel"]["debug_mode"]:
                    self.__send_debug_info2buf.send([info_dict["detection_eqrectimg"], 
                                                     info_dict["postproc_eqrectimg"]])
                #=============================================================             

            l_frame=l_frame.next
                                                        
            #except Exception as e:
            #    print("The RTSP source " + str(frame_meta.source_id) + \
            #          "(url: " + self.__rtsp_url + \
            #          ") has someting wrong: " + str(e))

        return Gst.PadProbeReturn.OK
    

    def bus_call(self, bus, message, loop) -> bool:
        """
        reference from GsStream:
        1. Gst.Message: https://lazka.github.io/pgi-docs/Gst-1.0/classes/Message.html
        2. Gst.MessageType: https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MessageType
        """
        t = message.type
        if t == Gst.MessageType.EOS:
            if not self.__abnormal_input:
                self.__app_logger.write("gstreamerproc", "App4RTSP.bus_call", "End", "End of stream.")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            if not self.__abnormal_input:
                err, debug = message.parse_warning()
                self.__app_logger.write("gstreamerproc", "App4RTSP.bus_call", "Warn", str(err) + "->" + str(debug) + ".")
        elif t == Gst.MessageType.ERROR:
            if not self.__abnormal_input:
                err, debug = message.parse_error()
                self.__app_logger.write("gstreamerproc", "App4RTSP.bus_call", "Error", str(err) + "->" + str(debug) + ".")                
            loop.quit()
        elif t == Gst.MessageType.QOS or t == Gst.MessageType.TAG:
            #print(int(t))
            self.__abnormal_input = False
        return True


    def cb_newpad(self, decodebin, decoder_src_pad, data):
        self.__app_logger.write("gstreamerproc", "App4RTSP.cb_newpad", "Info", "In cb_newpad.")
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        # Need to check if the pad created by the decodebin is for video and not
        # audio.
        print("gstname=", gstname)
        if (gstname.find("video") != -1):
            # Link the decodebin pad only if decodebin has picked nvidia
            # decoder plugin nvdec_*. We do this by checking if the pad caps contain
            # NVMM memory features.
            print("features=", features)
            if features.contains("memory:NVMM"):
                # Get the source bin ghost pad
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    self.__app_logger.write("gstreamerproc", "App4RTSP.cb_newpad", "Error", "Failed to link decoder src pad to source bin ghost pad.")
            else:
                self.__app_logger.write("gstreamerproc", "App4RTSP.cb_newpad", "Error", "Decodebin did not pick nvidia decoder plugin.")

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        if not self.__abnormal_input:
            self.__app_logger.write("gstreamerproc", "App4RTSP.decodebin_child_added", "Info", "Decodebin child added '" + name + "'.")

        if (name.find("decodebin") != -1):
            Object.connect("child-added", self.decodebin_child_added, user_data)
        if (is_aarch64() and name.find("nvv4l2decoder") != -1):
            self.__app_logger.write("gstreamerproc", "App4RTSP.decodebin_child_added", "Info", "Seting bufapi_version.")
            Object.set_property("bufapi-version", True)
        if (name.find("nvv4l2decoder") != -1):
            self.__app_logger.write("gstreamerproc", "App4RTSP.decodebin_child_added", "Info", "Drop frame interval.")
            self.__app_logger.write("gstreamerproc", "App4RTSP.decodebin_child_added", "Info", str(self.__local_args["kernel"]["image_data"]["decoded_input_frames"]) + " input decoded frames.")
            self.__app_logger.write("gstreamerproc", "App4RTSP.decodebin_child_added", "Info", str(self.__local_args["kernel"]["image_data"]["drop_decoded_frames"]) + " dropped frames.")
            drop_decoded_frames = self.__local_args["kernel"]["image_data"]["drop_decoded_frames"]
            decoded_input_frames = self.__local_args["kernel"]["image_data"]["decoded_input_frames"]
            if decoded_input_frames <= drop_decoded_frames:
                self.__app_logger.write("gstreamerproc", "App4RTSP.decodebin_child_added", "Error", 
                                      "The 'decoded_input_frames' must be bigger than 'drop_decoded_frames'.")
                sys.exit(0)

            drop_interval = int(decoded_input_frames / (decoded_input_frames - drop_decoded_frames)) 
            self.__app_logger.write("gstreamerproc", "App4RTSP.decodebin_child_added", "Info", str(drop_interval) + " drop interval.")
            
            Object.set_property("drop-frame-interval", drop_interval)

            if self.__deepstream_version == "6.2.0":
                # Use CUDA unified memory in the pipeline so frames
                # can be easily accessed on CPU in Python.
                Object.set_property("cudadec-memtype", 2)
                self.__app_logger.write("gstreamerproc", "App4RTSP.decodebin_child_added", "Info", "cudadec-memtype for decodebin is unified memory (because of DeepStream version 6.2).")

    def create_source_bin(self, index, uri):
        self.__app_logger.write("gstreamerproc", "App4RTSP.create_source_bin", "Info", "Creating source bin.")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        bin_name = "source-bin-%02d" % index
        self.__app_logger.write("gstreamerproc", "App4RTSP.create_source_bin", "Info", bin_name + ".")
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            self.__app_logger.write("gstreamerproc", "App4RTSP.create_source_bin", "Error", "Unable to create source bin.")

        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        uri_decode_bin = self.make_gst_factory("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            self.__app_logger.write("gstreamerproc", "App4RTSP.create_source_bin", "Error", "Unable to create uri decode bin.")

        # We set the input uri to the source element
        uri_decode_bin.set_property("uri", uri)
        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has beed created by the decodebin
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            self.__app_logger.write("gstreamerproc", "App4RTSP.create_source_bin", "Error", "Failed to add ghost pad in source bin.")
            return None
        return nbin

    def make_gst_factory(self, factoryname: str, name: str) -> object:
        """
        make ElementFactory by gstreamer.

        https://lazka.github.io/pgi-docs/Gst-1.0/classes/ElementFactory.html
        """
        self.__app_logger.write("gstreamerproc", "App4RTSP.make_gst_factory", "Info", "Creating " + factoryname + ".")
        module = Gst.ElementFactory.make(factoryname, name)
        if not module:
            self.__app_logger.write("gstreamerproc", "App4RTSP.make_gst_factory", "Error", "Unable to create " + factoryname + ".")
            sys.exit(0)
        return module

    def make_encoder(self) -> (object, object):
        """
        return: (encoder, rtppay)
        """
        codec_type = self.__local_args["kernel"]["rtsp_output"]["encoder"]
        bitrate = self.__local_args["kernel"]["rtsp_output"]["bitrate"]

        if codec_type == "H264":
            # Make the h264 encoder
            encoder = self.make_gst_factory("nvv4l2h264enc", "h264-encoder")
            # Make the payload-encode video into RTP packets
            rtppay = self.make_gst_factory("rtph264pay", "rtppay-h264")
        elif codec_type == "H265":
            # Make the h264 encoder
            encoder = self.make_gst_factory("nvv4l2h265enc", "h265-encoder")
            # Make the payload-encode video into RTP packets
            rtppay = self.make_gst_factory("rtph265pay", "rtppay-h265")

        encoder.set_property('bitrate', bitrate)
        if is_aarch64():
            encoder.set_property('preset-level', 1)
            encoder.set_property('insert-sps-pps', 1)
            encoder.set_property('bufapi-version', 1)

        return (encoder, rtppay)

    def make_udpsink(self) -> object:
        """
        """
        # Make the UDP sink
        ip = self.__local_args["kernel"]["rtsp_output"]["ip"]
        updsink_port_num = self.__local_args["kernel"]["rtsp_output"]["updsink_port_num"]
        udpsink = self.make_gst_factory("udpsink", "udpsink")

        udpsink.set_property('host', ip)
        udpsink.set_property('port', updsink_port_num)
        udpsink.set_property('async', False)
        udpsink.set_property('sync', 1)

        return udpsink

    def launch_rtsp_output(self):
        ip = self.__local_args["kernel"]["rtsp_output"]["ip"]
        encoder_name = self.__local_args["kernel"]["rtsp_output"]["encoder"]
        rtsp_port_num = self.__local_args["kernel"]["rtsp_output"]["rtsp_port_num"]
        updsink_port_num = self.__local_args["kernel"]["rtsp_output"]["updsink_port_num"]
        rtsp_name = self.__local_args["kernel"]["rtsp_output"]["name"]

        self.server = GstRtspServer.RTSPServer.new()
        self.server.props.service = "%d" % rtsp_port_num
        self.server.attach(None)

        self.factory = GstRtspServer.RTSPMediaFactory.new()
        self.factory.set_launch(
            "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (
            updsink_port_num, encoder_name))
        self.factory.set_shared(True)
        self.server.get_mount_points().add_factory("/" + rtsp_name, self.factory)

        self.__app_logger.write("gstreamerproc", "App4RTSP.launch_rtsp_output", "Info", 
                              "DeepStream: Launched RTSP Streaming at rtsp://" + ip + ":" + str(rtsp_port_num) + "/" + rtsp_name + ".")

    def execute(self, rtsp_url: str):
        """
        excute the decode/process/inference flow.

        p.s. the customized method(such as model inference) is defined in the self.tiler_sink_pad_buffer_probe() or self.flowproc(),
             and you can define your personal method in the self.tiler_sink_pad_buffer_probe() or self.flowproc().
        """
        self.__fps_streams["stream{0}".format(0)] = GETFPS(0)
        

        # Create nvstreammux instance to form batches from one or more sources.
        streammux = self.make_gst_factory("nvstreammux", "Stream-muxer")
        

        self.__pipeline.add(streammux)
        self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Info", 
                              "Creating source_bin " + str(0) + ".")

        source_bin = self.create_source_bin(0, rtsp_url)
        if not source_bin:
            self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Error", "Unable to create source bin.")
        self.__pipeline.add(source_bin)
        
        padname = "sink_%u" % 0
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Error", "Unable to create sink pad bin.")
        
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Error", "Unable to create src pad bin.")
        srcpad.link(sinkpad)

        
        # Add nvvidconv1 and filter1 to convert the frames to RGBA
        # which is easier to work with in Python.
        # RGBA only: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-imagedata-multistream/deepstream_imagedata-multistream.py
        # line 391 to 397.
        nvvidconv1 = self.make_gst_factory("nvvideoconvert", "convertor1")
        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter1 = self.make_gst_factory("capsfilter", "filter1")
        filter1.set_property("caps", caps1)
        self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Info", 
                                "Set the flow for Convert2RGBA video raw.")
        
        # Create nvvidconv to do image processing. Such as cropping.
        nvvidconv = self.make_gst_factory("nvvideoconvert", "convertor")

        # Create OSD to draw on the converted RGBA buffer.
        nvosd = self.make_gst_factory("nvdsosd", "onscreendisplay")

        # Create without drawing on screen using fakesink.
        # https://blog.csdn.net/weixin_34910922/article/details/118400005
        fakesink = self.make_gst_factory("fakesink", "fakesink")
        fakesink.set_property('enable-last-sample', 0)
        fakesink.set_property('sync', 0)

        streammux.set_property('width', self.__local_args["kernel"]["image_data"]["width"])
        streammux.set_property('height', self.__local_args["kernel"]["image_data"]["height"])
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', self.__local_args["kernel"]["image_data"]["batched_push_timeout"])

        # timestamp(system time).
        streammux.set_property('attach-sys-ts', 1)

        rtsp_output_open = self.__local_args["kernel"]["rtsp_output"]["open"]
        if rtsp_output_open:
            nvvidconv_postosd = self.make_gst_factory("nvvideoconvert", "convertor_postosd")
            if not nvvidconv_postosd:
                self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Error", 
                                        "Unable to create nvvidconv_postosd.")

            # Create a caps filter.
            caps = self.make_gst_factory("capsfilter", "filter")
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

            encoder, rtppay = self.make_encoder()
            udpsink = self.make_udpsink()


        if not is_aarch64():
            # Use CUDA unified memory in the pipeline so frames
            # can be easily accessed on CPU in Python.
            mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            streammux.set_property("nvbuf-memory-type", mem_type)
            #nvvidconv.set_property("nvbuf-memory-type", mem_type)
            nvvidconv1.set_property("nvbuf-memory-type", mem_type)
            if rtsp_output_open:
                nvvidconv_postosd.set_property("nvbuf-memory-type", mem_type)

        self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Info", 
                                "Adding elements to Pipeline.")
        
        # RGBA only: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-imagedata-multistream/deepstream_imagedata-multistream.py
        # line 391 to 397.
        self.__pipeline.add(nvvidconv)
        self.__pipeline.add(filter1)
        self.__pipeline.add(nvvidconv1)
        self.__pipeline.add(nvosd)
        self.__pipeline.add(fakesink)

        if rtsp_output_open:
            self.__pipeline.add(nvvidconv_postosd)
            self.__pipeline.add(caps)
            self.__pipeline.add(encoder)
            self.__pipeline.add(rtppay)
            self.__pipeline.add(udpsink)


        self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Info", 
                                "Linking elements in the Pipeline.")
        
        streammux.link(nvvidconv1)
        nvvidconv1.link(filter1)
        filter1.link(nvvidconv)
        nvvidconv.link(nvosd)
        
        if not rtsp_output_open:
            nvosd.link(fakesink)
        else:
            nvosd.link(nvvidconv_postosd)
            nvvidconv_postosd.link(caps)
            caps.link(encoder)
            encoder.link(rtppay)
            rtppay.link(udpsink)

        # create an event loop and feed gstreamer bus mesages to it
        loop = GLib.MainLoop()
        bus = self.__pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, loop)

        if rtsp_output_open:
            self.launch_rtsp_output()

        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Error", 
                                    "Unable to get sink pad of nvosd.")

        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.tiler_sink_pad_buffer_probe, 0)

        # List the sources
        self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Info", 
                              "Now playing is " + rtsp_url + ".")

        debug_mode_proc = None
        if self.__local_args["kernel"]["debug_mode"]:
            self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Debug", "Starting debug process.")
            debug_mode_proc = Process(target=self.host_debug_rtspin, args=(self.__recv_debug_info2buf,))
            debug_mode_proc.start()

        # start play back and listed to events
        self.__app_logger.write("gstreamerproc", "App4RTSP.execute", "Info", 
                                "Starting DeepStream pipeline.")
        
        self.__pipeline.set_state(Gst.State.PLAYING)
        loop.run()
        time.sleep(1)
        self.__pipeline.set_state(Gst.State.NULL)