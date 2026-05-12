from typing import List, Tuple

from PyQt6.QtWidgets import (
    QDockWidget, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal


class ResultList(QDockWidget):
    """Dockable list showing detected difference regions."""

    bbox_selected = pyqtSignal(int, int, int, int)  # x, y, w, h

    def __init__(self, parent=None):
        super().__init__("差异列表", parent)
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(["#", "X", "Y", "W", "H", "面积"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self.setWidget(self._table)
        self._bboxes: List[Tuple[int, int, int, int]] = []

    def set_bboxes(self, bboxes: List[Tuple[int, int, int, int]]):
        self._bboxes = bboxes
        self._table.setRowCount(len(bboxes))
        for i, (x, y, w, h) in enumerate(bboxes):
            area = w * h
            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(str(x)),
                QTableWidgetItem(str(y)),
                QTableWidgetItem(str(w)),
                QTableWidgetItem(str(h)),
                QTableWidgetItem(str(area)),
            ]
            for j, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(i, j, item)

    def clear(self):
        self._bboxes = []
        self._table.setRowCount(0)

    def _on_selection_changed(self):
        rows = self._table.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
            if 0 <= idx < len(self._bboxes):
                x, y, w, h = self._bboxes[idx]
                self.bbox_selected.emit(x, y, w, h)
