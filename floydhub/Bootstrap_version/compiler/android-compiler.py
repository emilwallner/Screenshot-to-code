#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys

from os.path import basename
from classes.Utils import *
from classes.Compiler import *

if __name__ == "__main__":
    argv = sys.argv[1:]
    length = len(argv)
    if length != 0:
        input_file = argv[0]
    else:
        print("Error: not enough argument supplied:")
        print("android-compiler.py <input file>")
        exit(0)

TEXT_PLACE_HOLDER = "[TEXT]"
ID_PLACE_HOLDER = "[ID]"

dsl_path = "assets/android-dsl-mapping.json"
compiler = Compiler(dsl_path)


def render_content_with_text(key, value):
    value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
    while value.find(ID_PLACE_HOLDER) != -1:
        value = value.replace(ID_PLACE_HOLDER, Utils.get_android_id(), 1)
    return value

file_uid = basename(input_file)[:basename(input_file).find(".")]
path = input_file[:input_file.find(file_uid)]

input_file_path = "{}{}.gui".format(path, file_uid)
output_file_path = "{}{}.xml".format(path, file_uid)

compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)
