import os, sys, shutil, subprocess

import pprint
from collections import OrderedDict
import numpy as np
import pandas as pd

import hazelbean as hb
import math
from osgeo import gdal
import contextlib
import logging

L = hb.get_logger('hazelbean stats')


def execute_os_command(command):
    # TODOOO This may be useful throughout numdal
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        nextline = process.stdout.readline().decode('ascii')
        if nextline == '' and process.poll() is not None and len(nextline) > 10:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    output = process.communicate()[0]
    exitCode = process.returncode

    if (exitCode == 0):
        return output
    else:
        raise Exception(command, exitCode, output)

def execute_r_string(r_string, output_uri=None, script_save_uri=None, keep_files=True):
    if not output_uri:
        # Learning point: the following line failed because os.path.join put a \ on , which r misinterpreted.
        #output_uri = os.path.join(nd.config.TEMP_FOLDER, nd.ruri('generated_r_output.txt'))
        output_uri = nd.config.TEMP_FOLDER + '/' + nd.ruri('generated_r_output.txt')
        if not keep_files:
            nd.uris_to_delete_at_exit.append(output_uri)
    else:
        if '\\\\' in output_uri:
            output_uri = output_uri.replace('\\\\', '/')
        elif '\\' in output_uri:
            output_uri = output_uri.replace('\\', '/')
        else:
            pass # output_uri  was okay
    if not script_save_uri:
        script_save_uri = os.path.join(nd.config.TEMP_FOLDER, nd.ruri('generated_r_script.R'))
        if not keep_files:
            nd.uris_to_delete_at_exit.append(script_save_uri)

    r_string = r_string.replace('\\', '/')

    print(r_string)
    f = open(script_save_uri, "w")

    f.write('sink(\"' + output_uri + '\")\n')
    f.write(r_string)
    f.close()

    returned = execute_r_script(script_save_uri, output_uri)

    return returned

def execute_r_script(script_uri, output_uri):
    cmd = 'C:\\Program Files\\R\\R-3.3.1\\bin\\Rscript.exe --vanilla --verbose ' + script_uri
    returned = subprocess.check_output(cmd, universal_newlines=True)
    if os.path.exists(output_uri):
        f = open(output_uri, 'r')
        to_return = ''
        for l in f:
            to_return += l +'\n'
        return to_return
    else:
        nd.L.warning('Executed r script but no output was found.')


def convert_af_to_1d_df(af):
    array = af.data.flatten()
    # print('   CALLED convert_af_to_1d_df on array of size' + str(len(array)))
    df = pd.DataFrame(array)
    return df


def concatenate_dfs_horizontally(df_list, column_headers=None):
    """
    Append horizontally, based on index.
    """

    df = pd.concat(df_list, axis=1)

    if column_headers:
        df.columns = column_headers
    return df


def convert_df_to_af_via_index(input_df, column, match_af, output_uri):
    match_df = ge.convert_af_to_1d_df(match_af)
    if match_af.size != len(input_df.index):
        full_df = pd.DataFrame(np.zeros(match_af.size), index=match_df.index)
    else:
        full_df = input_df

    listed_index = np.array(list(input_df.index))

    full_df[0][listed_index] = input_df['0']
    # full_df[0][input_df.index] = input_df[column]

    array = full_df.values.reshape(match_af.shape)
    nd.ArrayFrame(array, match_af, output_uri=output_uri)
    af = nd.ArrayFrame(output_uri)

    return af

