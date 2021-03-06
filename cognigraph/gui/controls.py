from PyQt5.QtWidgets import (
    QApplication,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QAction,
    QDialog,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
    QDialogButtonBox,
    QMessageBox,
)
from PyQt5.Qt import QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from collections import namedtuple, OrderedDict

from pyqtgraph.parametertree import parameterTypes, ParameterTree

from cognigraph.nodes.pipeline import Pipeline
from cognigraph import nodes

from cognigraph.gui import node_controls

from cognigraph.utils.pyqtgraph import MyGroupParameter
from cognigraph.utils.misc import class_name_of

from cognigraph.nodes.node import Node
from functools import partial

# import traceback
import logging


node_controls_map = namedtuple(
    "node_controls_map", ["node_class", "controls_class"]
)


class _PipelineTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, parent_item, node, widget):
        QTreeWidgetItem.__init__(self, parent_item, [repr(node)])
        self.node = node
        self.widget = widget


class SourceControls(MyGroupParameter):
    """
    Represents a drop-down list with the names of supported source types.
    Selecting a type creates controls for that type below the drop-down.

    """

    # Order is important.
    # Entries with node subclasses must precede entries with the parent class
    SOURCE_OPTIONS = OrderedDict(
        (
            (
                "LSL stream",
                node_controls_map(
                    nodes.LSLStreamSource,
                    node_controls.LSLStreamSourceControls,
                ),
            ),
            (
                "File data",
                node_controls_map(
                    nodes.FileSource, node_controls.FileSourceControls
                ),
            ),
        )
    )

    SOURCE_TYPE_COMBO_NAME = "Source type: "
    SOURCE_TYPE_PLACEHOLDER = ""
    SOURCE_CONTROLS_NAME = "source controls"

    def __init__(self, source_node, **kwargs):
        self._source_node = source_node
        name = repr(source_node)
        super().__init__(name=name)

        labels = [self.SOURCE_TYPE_PLACEHOLDER] + [
            label for label in self.SOURCE_OPTIONS
        ]

        source_type_combo = parameterTypes.ListParameter(
            name=self.SOURCE_TYPE_COMBO_NAME, values=labels, value=labels[0]
        )

        source_type_combo.sigValueChanged.connect(self._on_source_type_changed)
        self.source_type_combo = self.addChild(source_type_combo)

        if source_node is not None:
            for source_option, classes in self.SOURCE_OPTIONS.items():
                if isinstance(source_node, classes.node_class):
                    self.source_type_combo.setValue(source_option)

                    controls = classes.controls_class(
                        source_node=self._source_node,
                        name=self.SOURCE_CONTROLS_NAME,
                    )
                    self.source_controls = self.addChild(controls)

    def _on_source_type_changed(self, param, value):
        if value != self.SOURCE_TYPE_PLACEHOLDER:
            # Update source controls
            source_classes = self.SOURCE_OPTIONS[value]
            # self.source_controls = self.addChild(controls)

            # Update source
            if not isinstance(self._source_node, source_classes.node_class):
                # self._source_node = self.source_controls.create_node()
                parent = self._source_node.parent
                create_node_dialog = _CreateNodeDialog(
                    source_classes.node_class, parent=None
                )
                create_node_dialog.show()
                create_node_dialog.widget.setSizeAdjustPolicy(1)
                create_node_dialog.widget.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding
                )
                create_node_dialog.adjustSize()
                create_node_dialog.node.disabled = True
                create_node_dialog.exec()
                parent.add_child(create_node_dialog.node)
                # self.create_node_dialog.node.initialize()
                if create_node_dialog.result():
                    new_source_node = create_node_dialog.node
                    controls = source_classes.controls_class(
                        source_node=new_source_node,
                        name=self.SOURCE_CONTROLS_NAME,
                    )
                    self.removeChild(self.source_controls)
                    self.source_controls = self.addChild(controls)
                    for child in self._source_node._children:
                        child.parent = new_source_node
                    parent.remove_child(self._source_node)
                    self._source_node = new_source_node
                    new_source_node.disabled = False
                else:
                    parent.remove_child(create_node_dialog.node)
                    for source_option, classes in self.SOURCE_OPTIONS.items():
                        if isinstance(self._source_node, classes.node_class):
                            self.source_type_combo.setValue(source_option)


node_to_controls_map = {
    "Pipeline": node_controls.PipelineControls,
    "LinearFilter": node_controls.LinearFilterControls,
    "MNE": node_controls.MNEControls,
    "EnvelopeExtractor": node_controls.EnvelopeExtractorControls,
    "Preprocessing": node_controls.PreprocessingControls,
    "Beamformer": node_controls.BeamformerControls,
    "MCE": node_controls.MCEControls,
    "ICARejection": node_controls.ICARejectionControls,
    "AtlasViewer": node_controls.AtlasViewerControls,
    "AmplitudeEnvelopeCorrelations": node_controls.AmplitudeEnvelopeCorrelationsControls,  # noqa
    "Coherence": node_controls.CoherenceControls,
    "SeedCoherence": node_controls.SeedCoherenceControls,
    "LSLStreamOutput": node_controls.LSLStreamOutputControls,
    "BrainViewer": node_controls.BrainViewerControls,
    "SignalViewer": node_controls.SignalViewerControls,
    "FileOutput": node_controls.FileOutputControls,
    "TorchOutput": node_controls.TorchOutputControls,
    "ConnectivityViewer": node_controls.ConnectivityViewerControls,
    "LSLStreamSource": SourceControls,  # node_controls.LSLStreamSourceControls
    "FileSource": SourceControls,  # node_controls.FileSourceControls
}


class MultipleNodeControls(MyGroupParameter):
    """
    Base class for grouping of node settings (processors or outputs).
    Source is supported by a separate class.

    """

    @property
    def SUPPORTED_NODES(self):
        raise NotImplementedError

    def __init__(self, nodes, **kwargs):
        self._nodes = nodes
        super().__init__(**kwargs)

        for node in nodes:
            controls_class = self._find_controls_class_for_a_node(node)
            self.addChild(controls_class(node), autoIncrementName=True)

    @classmethod
    def _find_controls_class_for_a_node(cls, processor_node):
        for node_control_classes in cls.SUPPORTED_NODES:
            if isinstance(processor_node, node_control_classes.node_class):
                return node_control_classes.controls_class

        # Raise an error if processor node is not supported
        msg = (
            "Node of class {0} is not supported by {1}.\n"
            "Add node_controls_map(node_class, controls_class) to"
            " {1}.SUPPORTED_NODES"
        ).format(class_name_of(processor_node), cls.__name__)
        raise ValueError(msg)


class BaseControls(QWidget):
    def __init__(self, pipeline, name="BaseControls", type="BaseControls"):
        super().__init__()
        self._pipeline = pipeline

        # TODO: Change names to delineate source_controls as defined here and
        # source_controls - gui.node_controls.source

        # self.source_controls = self.addChild(source_controls)
        layout = QVBoxLayout()
        for node in pipeline.all_nodes:
            widget = self._create_node_controls_widget(node)
            widget.hide()
            layout.addWidget(widget)
        self.setLayout(layout)


class _CreateNodeDialog(QDialog):
    def __init__(self, node_cls, parent=None):
        QDialog.__init__(self, parent)

        self.widget = ParameterTree(showHeader=False)
        layout = QVBoxLayout()
        layout.addWidget(self.widget)
        params = parameterTypes.GroupParameter(name="Parameters setup")
        self.node = node_cls()
        controls_cls = node_to_controls_map[node_cls.__name__]
        self.controls = controls_cls(self.node)
        params.addChild(self.controls)
        self.widget.setParameters(params)
        dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addWidget(dialog_buttons)
        dialog_buttons.accepted.connect(self._on_ok)
        dialog_buttons.rejected.connect(self.reject)

        self.setLayout(layout)
        self._thread = QThread()

    def _on_ok(self):
        self.accept()

    def _on_worker_error(self, exc):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Failed to create %s" % str(self.node))
        msg.setDetailedText(exc)
        msg.show()


class PipelineTreeWidget(QTreeWidget):
    node_added = pyqtSignal("PyQt_PyObject", "PyQt_PyObject")
    node_removed = pyqtSignal("PyQt_PyObject")

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.clear()
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        self.setItemsExpandable(True)

        self.controls_layout = QVBoxLayout()
        self.resizeColumnToContents(0)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(
            self._on_context_menu_requiested
        )
        self.itemSelectionChanged.connect(self._on_tree_item_selection_changed)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _on_context_menu_requiested(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)

        allowed_children = item.node.ALLOWED_CHILDREN
        actions = []
        if len(allowed_children) > 0:
            submenu = QMenu(menu)
            submenu.setTitle("Add node")
            menu.addMenu(submenu)
            for c in allowed_children:
                child_cls = getattr(nodes, c)
                add_node_action = QAction(repr(child_cls), submenu)
                add_node_action.triggered.connect(
                    partial(
                        self._on_add_node_action,
                        parent=item.node,
                        child_cls=child_cls,
                        item=item,
                    )
                )
                actions.append(add_node_action)
            submenu.addActions(actions)
        if len(item.node._children) == 0:
            remove_node_action = QAction("Remove node", menu)
            menu.addAction(remove_node_action)
            remove_node_action.triggered.connect(
                partial(self._on_remove_node_action, item)
            )

        menu.exec(self.viewport().mapToGlobal(pos))

    def remove_item(self, item: _PipelineTreeWidgetItem):
        item.node.parent = None
        item.parent().removeChild(item)
        self._logger.debug("Removed %s from tree widget", str(item.node))

    def _on_remove_node_action(self, item: _PipelineTreeWidgetItem):
        self.remove_item(item)
        self.node_removed.emit(item)

    def _on_add_node_action(self, t, parent: Node, child_cls, item=None):
        """
        Add node to pipeline

        Parameters
        ----------
        t:
            Something that action.triggered signal passes.
        parent: Node
            Parent Node instance
        chld_cls
            Class of node we're about to add

        """
        self._logger.debug("Adding %s to %s" % (child_cls, parent))
        self.create_node_dialog = _CreateNodeDialog(child_cls, parent=self)
        self.create_node_dialog.show()
        self.create_node_dialog.widget.setSizeAdjustPolicy(1)
        self.create_node_dialog.widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.create_node_dialog.adjustSize()
        self.create_node_dialog.node.disabled = True
        parent.add_child(self.create_node_dialog.node)
        self.create_node_dialog.exec()
        # self.create_node_dialog.node.initialize()
        if self.create_node_dialog.result():
            self.create_node_dialog.node.disabled = False
            if item:
                self.node_added.emit(self.create_node_dialog.node, item)
        else:
            parent.remove_child(self.create_node_dialog.node)
        self.create_node_dialog.controls.deleteLater()
        self.create_node_dialog.widget.deleteLater()
        self.create_node_dialog.deleteLater()

    def _create_node_controls_widget(self, node):
        controls_cls = node_to_controls_map[node.__class__.__name__]
        controls = controls_cls(node)
        params = parameterTypes.GroupParameter(name=repr(node))
        params.addChild(controls)
        widget = ParameterTree(showHeader=False)
        widget.setParameters(params)
        widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        widget.setSizeAdjustPolicy(1)
        return widget

    def _on_tree_item_selection_changed(self):
        iterator = QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            item.widget.hide()
            iterator += 1
        for item in self.selectedItems():
            item.widget.show()

    def fetch_item_by_node(self, node):
        iterator = QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            if item.node is node:
                return item
            iterator += 1
        return None


class Controls(QWidget):
    def __init__(self, pipeline: Pipeline, parent=None):
        QWidget.__init__(self, parent)
        self._pipeline = pipeline  # type: Pipeline

        self.tree_widget = PipelineTreeWidget(pipeline=self._pipeline)
        layout = QVBoxLayout()
        layout.addWidget(self.tree_widget)

        self.params_layout = QVBoxLayout()
        self._add_nodes(pipeline, self.tree_widget)
        layout.addLayout(self.params_layout)

        self.setLayout(layout)

        self.tree_widget.node_added.connect(self._add_nodes)

    def _add_nodes(self, node, parent_item):
        widget = self._create_node_controls_widget(node)
        widget.hide()
        this_item = _PipelineTreeWidgetItem(parent_item, node, widget)
        if parent_item is self.tree_widget:
            self.tree_widget.setCurrentItem(this_item)
            widget.show()
        self.tree_widget.controls_layout.addWidget(widget)
        self.params_layout.addWidget(widget)
        self.tree_widget.expandItem(this_item)
        for child in node._children:
            self._add_nodes(child, this_item)

    def _create_node_controls_widget(self, node):
        controls_cls = node_to_controls_map[node.__class__.__name__]
        controls = controls_cls(node)
        params = parameterTypes.GroupParameter(name=repr(node))
        params.addChild(controls)
        widget = ParameterTree(showHeader=False)
        widget.setParameters(params)
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        widget.setSizeAdjustPolicy(1)
        return widget


if __name__ == "__main__":
    import sys
    from cognigraph.tests.prepare_pipeline_tests import (
        ConcreteSource,
        ConcreteProcessor,
        ConcreteOutput,
    )

    pipeline = Pipeline()
    src = ConcreteSource()
    proc = ConcreteProcessor()
    out = ConcreteOutput()
    src.add_child(proc)
    proc.add_child(out)
    pipeline.add_child(src)
    app = QApplication(sys.argv)
    tree_widget = Controls(pipeline)
    tree_widget.show()
    sys.exit(app.exec_())  # dont need this: tree_widget has event_loop
