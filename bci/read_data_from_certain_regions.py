#1.Select data
#2.Initialize pipeline
#3.Import forward


#4.Add atlas viewer to the beamformer node
#5.Initialize pipeline
#6.Select ROI
#7.Add LSL stream output

from cognigraph.gui.controls import PipelineTreeWidget

from cognigraph.gui.node_controls import AtlasViewerControls, LSLStreamOutputControls
from cognigraph.nodes import AtlasViewer, LSLStreamOutput
import PyQt5

#AtlasViewerControls
#PipelineTreeWidget
#LSLStreamOutputControls
AtlasViewerExample=AtlasViewer()

OUTPUT_CLASS = outputs.LSLStreamOutput
CONTROLS_LABEL = "LSL stream"
STREAM_NAME_STR_NAME = "Output stream name: "

stream_name = 'lol'
stream_name_str = parameterTypes.SimpleParameter(
    type="str",
    name=STREAM_NAME_STR_NAME,
    value=stream_name,
    editable=False,
)
stream_name_str.sigValueChanged.connect(self._on_stream_name_changed)
self.stream_name_str = self.addChild(stream_name_str)

LSLStreamOutput.initialize()
#Здесь обновляю его




