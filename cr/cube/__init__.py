# coding: utf-8

from cr.exporter.xlsx import exporter as xlsx_exporter

FORMATS = {
    'xlsx': xlsx_exporter
}

MIME_TYPES = {
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}


def export(payload, fmt, output, display_settings=None, formats=FORMATS, progress=None, meta=None):

    if display_settings is None:
        display_settings = {}
    exporter = formats[fmt]

    exporter(output, payload, display_settings, progress=progress, meta=meta)
