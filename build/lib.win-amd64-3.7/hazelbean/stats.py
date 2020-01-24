import os, sys, shutil, subprocess

import pprint
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
import hazelbean as hb
import math
from osgeo import gdal
import contextlib
import logging
import statsmodels.api as sm
import sklearn
import time
import json
import pickle
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import datasets
import matplotlib.pyplot as plt
from decimal import Decimal
import re

L = hb.get_logger('hazelbean stats')


class RegressionFrame(object):
    def __init__(self, project=None):
        self.p = project
        self.inputs = OrderedDict() # Stores user input information (file locations, set membership) for GLOBAL UNPROCESSED inputs that may be different resolutions, extents, etc.
        self.aligned_inputs = OrderedDict() # Subset of inputs for all spatial variables that have been clipped and resampled to generate identical numpy array shapes.
        self.global_aligned_inputs = OrderedDict() # Subset of inputs for all spatial variables that have been clipped and resampled to generate identical numpy array shapes.
        self.sources = OrderedDict() # Subset of aligned inputs that have been loaded as in-memory arrays.
        self.variables = OrderedDict() # Subset of along with any transforms of something that e.g. logs the sources. Not actually calculated until regression time to minimize memory impact.
        self.df = None
        self.stride_rate = None

        self.dependent_variable_label = None
        self.dependent_variable_path = None

        self.variable_sets = OrderedDict()
        self.loaded_data = OrderedDict()
        self.currently_loaded_masks = None

        # Running list to store results
        self.results = OrderedDict()

    def __str__(self):
        return 'RegressionFrame object: ' \
               '\n    Sources: ' + str([i for i in self.sources.keys()]) + '' \
               '\n    Variables: ' + str([i for i in self.variables.keys()]) + '' \

    def save_to_path(self, output_path):
        with open(output_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_from_path(self, input_path):
        print ('CHOSE NOT to do this as a method and instead is a module level function')

    def add_input(self, input_label, input_path, tags=None):
        new_input = hb.RegressionInput(input_label, input_path, self, tags=tags)
        L.debug('Adding regression input ' + str(new_input))
        self.inputs[input_label] = new_input

    def add_global_aligned_input(self, input_label, input_path, tags=None):
        new_input = hb.RegressionGlobalAlignedInput(input_label, input_path, self, tags=tags)
        L.debug('Adding global, aligned regression input ' + str(new_input))
        self.global_aligned_inputs[input_label] = new_input

    def add_aligned_input(self, input_label, input_path, tags=None):
        new_aligned_input = hb.RegressionAlignedInput(input_label, input_path, self, tags=tags)
        L.debug('Adding aligned regression input ' + str(new_aligned_input))
        self.aligned_inputs[input_label] = new_aligned_input

    def add_source(self, source_label, source_path, tags=None):
        new_source = hb.RegressionSource(source_label, source_path, self, tags=tags)
        L.info('Adding regression source ' + str(new_source))
        self.sources[source_label] = new_source
        return new_source

    def add_variable(self, variable_label, source_label, variable_type=None, tags=None):
        new_variable = hb.RegressionVariable(variable_label, source_label, self, variable_type, tags=tags)
        L.debug('Adding regression variable ' + str(new_variable))
        self.variables[variable_label] = new_variable
        return new_variable

    def add_variables_from_source(self, source):
        L.info('Running add_variables_from_source on ' + str(source))
        for tag in source.tags:

            if tag == 'linear':
                variable_label = source.label
                self.add_variable(variable_label, source.label, 'linear', source.tags)
            elif tag == 'square':
                variable_label = source.label + '_square'
                self.add_variable(variable_label, source.label, 'square', source.tags)
            elif tag == 'dummies':
                self.make_dummies_from_source(source, tags=source.tags)
            elif tag == 'gs4':
                variable_label = source.label
                self.add_variable(variable_label, source.label, 'gs4', source.tags)
                # self.make_dummies_from_source(source, tags=source.tags)
            elif tag == 'direct_spatial' or 'gs' in tag:
                variable_label = source.label
                self.add_variable(variable_label, source.label, 'direct_spatial', source.tags)
                # self.make_dummies_from_source(source, tags=source.tags)
            else:
                variable_label = source.label + '_' + tag

    def add_interactions_from_variables(self, var1, var2, tags=None):
        variable_label = var1.label + '_interact_' + var2.label
        source_label = var1.source.label
        variable_type = 'interaction'
        tags.append('interaction')
        self.add_variable(variable_label, source_label, variable_type, tags)


    def add_variable_set(self, variable_set_label, depvar_label, indvars):
        # Variable sets are a named list of variables where the first is assumed to be the depvar and all the others are
        # the indvars that constitute what will be run in a regression.
        self.variable_sets[variable_set_label] = [depvar_label] + indvars

    def add_variable_set_by_tags(self, variable_set_label, depvar_label, tags):
        # Variable sets are a named list of variables where the first is assumed to be the depvar and all the others are
        # the indvars that constitute what will be run in a regression.

        indvars = []

        for var_name, var_obj in self.variables.items():
            if var_obj.variable_type in tags or any(tag in tags for tag in var_obj.tags):
            # if any(var_tag in var_obj.tags for var_tag in tags):
                if var_obj.label != depvar_label:
                    indvars.append(var_obj.label)

        self.variable_sets[variable_set_label] = [depvar_label] + indvars

    def initialize_df_from_equation(self, equation, current_bounding_box, existing_df=None,):
        if existing_df is None:
            df = pd.DataFrame()
        else:
            df = existing_df

        self.equation_dict = hb.parse_equation_to_dict(equation)

        # hb.pp(self.equation_dict)

        if self.stride_rate is None:
            max_raster_size = 100000

            raster_info = hb.get_raster_info_hb(self.global_aligned_inputs[self.dependent_variable_label].path)

            depvar_size = raster_info['raster_size'][0] * raster_info['raster_size'][1]
            pixel_size = raster_info['pixel_size']

            self.current_stride_rate = 1
            if depvar_size > max_raster_size:
                self.current_stride_rate += 1
                while depvar_size / self.current_stride_rate ** 2 > max_raster_size:
                    self.current_stride_rate += 1
        else:
            self.current_stride_rate = self.stride_rate


        # Load the first array out of order to get the shape of data needed to be created
        first_array = hb.load_geotiff_chunk_by_bb(self.global_aligned_inputs[self.dependent_variable_label].path,
                                                  current_bounding_box,
                                                  self.current_stride_rate)

        first_array_shape = first_array.shape
        n_first_array_obs = first_array.size


        # Create an array that calculates efficiently if ALL values are valid in each pixel-stack.
        all_valid_array = np.full(n_first_array_obs, 1, np.int8)

        # For the depvar write as zero where its not valid based on its NDV value.
        current_valid_mask = first_array != self.global_aligned_inputs[self.dependent_variable_label].ndv
        all_valid_array[~current_valid_mask.flatten()] = 0

        # Create a list of masks read at this step for later incorporation into data matrix
        loaded_masks = {}

        # Update all_valid_array based on mask inputs.
        for c, variable_transform in enumerate(self.equation_dict['variable_transforms']):
            if variable_transform[0] == 'mask':
                parsed_input_name = variable_transform[2].split(',')[0]
                parsed_variable_name = variable_transform[2].split(',')[1]
                values_to_make_mask = [int(i) for i in variable_transform[4].split(',')]
                current_array = hb.load_geotiff_chunk_by_bb(self.global_aligned_inputs[parsed_input_name].path,
                                                            current_bounding_box,
                                                            self.current_stride_rate)
                current_array = np.where(np.isin(current_array, values_to_make_mask), 0, 1)

                # Keep track of where all are valid or not
                all_valid_array *= current_array.flatten()

                # To be used later with data array
                loaded_masks[parsed_variable_name] = current_array

        if self.currently_loaded_masks is not None:
            if loaded_masks.keys() != self.currently_loaded_masks.keys():
                L.info('Reloading DF because new masks were added.')
                df = pd.DataFrame()

        all_valid_mask = np.where(all_valid_array.astype(bool))

        n_obs = len(all_valid_mask[0])

        # Calculate size of data (new or to be added)
        n_new_cols = 0
        for c, necessary_variable in enumerate(self.equation_dict['necessary_variables']):
            if necessary_variable not in df.columns:
                n_new_cols += 1

        # Initialize the data array with a NDV.
        data = np.full((n_obs, n_new_cols), -9999., dtype=np.float64)

        # data[:, 0] = first_array.flatten().astype(np.float64)
        # first_array = None

        added_column_index = 0
        columns_to_add = list(df.columns) # Copy list because want to add new ones to end of list.
        for c, necessary_variable in enumerate(self.equation_dict['necessary_variables']):
            if necessary_variable not in df.columns:
                columns_to_add.append(necessary_variable)
                L.info('Loading ' + str(necessary_variable) + ' to df.')

                current_array = hb.load_geotiff_chunk_by_bb(self.global_aligned_inputs[necessary_variable].path,
                                                              current_bounding_box,
                                                              self.current_stride_rate)
                data[:, added_column_index] = current_array.flatten()[all_valid_mask]
                added_column_index += 1


        for c, variable_transform in enumerate(self.equation_dict['variable_transforms']):
            if variable_transform[0] == 'dummy':

                parsed_input_name = variable_transform[2].split(',')[0]
                parsed_variable_name = variable_transform[2].split(',')[1]
                values_to_make_dummy = variable_transform[4].split(',')

                if parsed_variable_name not in df.columns:
                    columns_to_add.append(parsed_variable_name)
                    L.info('Loading VARIABLE TRANSFORM ' + str(parsed_variable_name) + ' to df.')

                    current_array = hb.load_geotiff_chunk_by_bb(self.global_aligned_inputs[parsed_input_name].path,
                                                                current_bounding_box,
                                                                self.current_stride_rate)
                    current_array = np.where(np.isin(current_array, values_to_make_dummy), 1, 0)
                    data = np.concatenate((data, current_array.flatten()[all_valid_mask][:, np.newaxis]), axis=1)  # Funny newaxis is because needed to pass a 2d array to match data

                    added_column_index += 1
            # elif variable_transform[0] == 'mask':
            #     parsed_input_name = variable_transform[2].split(',')[0]
            #     parsed_variable_name = variable_transform[2].split(',')[1]
            #
            #     values_to_make_mask = [int(i) for i in variable_transform[4].split(',')]
            #
            #     if parsed_variable_name not in df.columns:
            #         columns_to_add.append(parsed_variable_name)
            #         L.info('Loading VARIABLE TRANSFORM ' + str(parsed_variable_name) + ' to df.')
            #
            #         current_array = hb.load_geotiff_chunk_by_bb(self.global_aligned_inputs[parsed_input_name].path,
            #                                                     current_bounding_box,
            #                                                     self.current_stride_rate)
            #
            #         current_array = np.where(np.isin(current_array, values_to_make_mask), 0, 1)
            #
            #         # Keep track of where all are valid or not
            #         all_valid_array *= current_array.flatten()
            #
            #         # current_valid_mask = first_array != self.global_aligned_inputs[self.dependent_variable_label].ndv
            #         # all_valid_array[~current_valid_mask.flatten()] = 0
            #         #
            #         print('loaded_masks', loaded_masks)
            #         data = np.concatenate((data, loaded_masks[parsed_variable_name].flatten()[:, np.newaxis]), axis=1)  # Funny newaxis is because needed to pass a 2d array to match data
            #
            #         added_column_index += 1

            if variable_transform[0] == 'log':
                for cc, i in enumerate(variable_transform):
                    5
            elif variable_transform in self.equation_dict['necessary_variables']:
                for cc, i in enumerate(variable_transform):
                    5

        # if the DF already has data, add cols
        if len(df) > 0:
            df = pd.DataFrame(pd.np.column_stack([df, data]), columns=columns_to_add, copy=False)
        else:
            df = pd.DataFrame(data=data, columns=columns_to_add, copy=False)

        self.currently_loaded_masks = loaded_masks

        # # Drop based on masks
        # masks_list = [i[2].split(',')[1] for i in self.equation_dict['mask_variables']]
        # for i in masks_list:
        #     all_valid_array
        #     # df = df[df[i] == 1]

        return df, all_valid_array
    def initialize_variable_set(self, variable_set_label, depvar_label=None, cols_to_drop_zero_from=None):
        L.info('Starting to initialize_variable_set for variable_set ' + str(variable_set_label))

        self.current_variable_set = variable_set_label

        if depvar_label is None:
            self.dependent_variable_label = self.variable_sets[variable_set_label][0]
        else:
            self.dependent_variable_label = depvar_label

        # Load the first array out of order to get the shape of data needed to be created
        first_array = hb.load_gdal_ds_as_strided_array(self.variables[self.dependent_variable_label].path, self.stride_rate)
        first_array_shape = first_array.shape
        n_obs = first_array.size
        n_vars = len(self.variable_sets[variable_set_label])

        # Create an array that calculates efficiently if ALL values are valid in each pixel-stack.
        all_valid_array = np.full(n_obs, 1, np.int8)

        # For the depvar, and later each indvar, write as zero where its not valid.
        current_valid_mask = first_array != self.sources[self.dependent_variable_label].ndv
        all_valid_array[~current_valid_mask.flatten()] = 0

        # Initialize the data array with a NDV.
        data = np.full((n_obs, n_vars), -9999., dtype=np.float64)
        data[:, 0] = first_array.flatten().astype(np.float64)

        # Iterate through all of the post-depvar variables, overwriting the NDV in the data with their value and
        # also keeping track of pixel-stack validity.
        for c, variable_label in enumerate(self.variable_sets[variable_set_label][1:]):

            L.info('  ' + str(c) + ' ' + variable_label)
            variable = self.variables[variable_label]
            if variable.variable_type is 'dummy':
                L.info('Reading dummy variables from ' + str(variable))

                # For dummy variables, we only want to load the array once, and on this read, we also need to determine validity.
                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)
                    current_valid_mask = variable.source.data != variable.source.ndv
                    all_valid_array[~current_valid_mask.flatten()] = 0

                dummy_valid_mask = variable.source.data == variable.source_value

                # For dummies, there are three possible values, 0, 1, and NDV. Use separate masks to achieve this.
                data[:, c + 1][current_valid_mask.flatten() & ~dummy_valid_mask.flatten()] = 0
                data[:, c + 1][dummy_valid_mask.flatten()] = 1

            elif variable.variable_type is 'linear':
                L.debug('Reading linear variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = variable.source.data.flatten()[current_valid_mask.flatten()]
            elif variable.variable_type is 'gs4':
                L.debug('Reading linear variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = variable.source.data.flatten()[current_valid_mask.flatten()]

            elif variable.variable_type is 'direct_spatial':
                L.debug('Reading linear variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = variable.source.data.flatten()[current_valid_mask.flatten()]

            elif variable.variable_type is 'square':
                L.debug('Reading squared variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = variable.source.data.flatten()[current_valid_mask.flatten()] ** 2
            elif variable.variable_type is 'cube':
                L.debug('Reading cubed variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = variable.source.data.flatten()[current_valid_mask.flatten()] ** 3
            elif variable.variable_type is 'quad':
                L.debug('Reading cubed variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = variable.source.data.flatten()[current_valid_mask.flatten()] ** 4
            elif variable.variable_type is 'quint':
                L.debug('Reading cubed variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = variable.source.data.flatten()[current_valid_mask.flatten()] ** 5
            elif variable.variable_type is 'ln':
                L.debug('Reading cubed variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = np.log(variable.source.data.flatten()[current_valid_mask.flatten()])
            elif variable.variable_type is 'log10':
                L.debug('Reading cubed variable from ' + str(variable))

                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = np.log10(variable.source.data.flatten()[current_valid_mask.flatten()])
            elif variable.variable_type is 'interaction':
                L.debug('Reading interaction variables from ' + str(variable))
                var1_label, var2_label = variable_label.split('_interact_')
                var2 = self.variables[var2_label]
                if variable.source.data is None:
                    variable.source.data = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64)

                if var2.source.data is None:
                    var2.source.data = hb.load_gdal_ds_as_strided_array(var2.source.path, self.stride_rate).astype(np.float64)

                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0

                var2_value = float(var2_label.split('dummy')[1])

                data[:, c + 1][current_valid_mask.flatten()] = variable.source.data.flatten()[current_valid_mask.flatten()] * np.where(var2.source.data.flatten()[current_valid_mask.flatten()] == var2_value, 1., 0.)
            else:
                L.critical('shouldnt get here')
                current_valid_mask = variable.source.data != variable.ndv
                all_valid_array[~current_valid_mask.flatten()] = 0
                data[:, c + 1][current_valid_mask.flatten()] = hb.load_gdal_ds_as_strided_array(variable.source.path, self.stride_rate).astype(np.float64).flatten()[current_valid_mask.flatten()]


        df = pd.DataFrame(data=data, columns=self.variable_sets[variable_set_label], copy=False)

        L.info('n. obs before dropping anything: ' + str(len(df.index)))

        for col in df.columns:
            L.info('Checking col ' + str(col) + ' for things to drop. Currently have ' + str(len(df.index)))
            try:
                if cols_to_drop_zero_from is not None:
                    if col in cols_to_drop_zero_from:
                        if df[col].mean() == 0 or df[col].mean() == -9999.0:
                            L.warning('Mean of ' + col + ' was zero or -9999. Removing it from DF and the variable set.')

                            df.drop(col, axis=1, inplace=True)
                            self.variable_sets[variable_set_label].remove(col)
            except:
                L.critical('df[col].mean() didnt work. This is often because there are duplicate cols. cols used: ' + str(col))


        # PROBLEMATIC LINE, think through logic of setting ndv to zero
        df[df==-9999.0] = 0

        if cols_to_drop_zero_from is not None:
            df = df[(df[cols_to_drop_zero_from] != -9999.0).all(1)]
        else:
            df = df[(df != -9999.0).all(1)]


        L.info('n. obs after dropping NDV from array: ' + str(len(df.index)))
        L.info('number valid in all_valid_array:', np.sum(all_valid_array))

        return df, all_valid_array.reshape(first_array_shape)

    def get_stride_rate_from_desired_sample_size_and_path(self, sample_size, input_path):
        n_rows, n_cols = hb.get_raster_info_hb(input_path)['shape']
        unsampled_size = n_rows * n_cols
        stride_rate = 1

        if unsampled_size > sample_size:
            while True:
                stride_rate += 1
                cur_n_rows = int(math.floor(n_rows / stride_rate))
                cur_n_cols = int(math.floor(n_cols / stride_rate))

                if cur_n_rows * cur_n_cols <= sample_size:
                    return stride_rate
        else:
            return stride_rate

    def make_dummies_from_source(self, source, values_to_use=None, tags=None):

        if values_to_use is None:

            if source.data is None:
                source.data = hb.load_gdal_ds_as_strided_array(source.path, self.stride_rate)
            values_to_use = np.unique(source.data).astype(np.int)

            L.info('Making dummy variables from ', str(source) + ': ' + str(values_to_use))

        tags.append('dummy')
        for value in values_to_use:
            label = source.label + '_dummy' + str(value)


            v = self.add_variable(label, source.label, 'dummy', tags=tags)
            v.source_value = value

    def make_variable_transform_from_source(self, source_label, transform_type):
        if source_label not in self.sources:
            L.warning('Attempted to make transform ' + str(source_label) + ' but this was not in the current sources.')
        else:
            source = self.sources[source_label]

            if source.data is None:
                source.data = hb.load_gdal_ds_as_strided_array(source.path, self.stride_rate)

            if transform_type == 'square':
                label = source.label + '_square'
                v = self.add_variable(label, source.label, 'square')

            if transform_type == 'cube':
                label = source.label + '_cube'
                v = self.add_variable(label, source.label, 'cube')

            if transform_type == 'quad':
                label = source.label + '_quad'
                v = self.add_variable(label, source.label, 'quad')

            if transform_type == 'quint':
                label = source.label + '_quint'
                v = self.add_variable(label, source.label, 'quint')

            if transform_type == 'log':
                label = source.label + '_log'
                v = self.add_variable(label, source.label, 'log')

            if transform_type == 'log10':
                label = source.label + '_log10'
                v = self.add_variable(label, source.label, 'log10')


    def run_sm_lm(self, regression_label, df, equation_dict,  output_dir=None, has_constant=True):
        # self.coeff_labels = self.variable_sets[variable_set_label][1:] # NOTE dropping depvar
        lm_sm = sm.OLS(df[equation_dict['dependent_variable']], df[equation_dict['regression_terms']], hasconst=has_constant).fit()

        self.coeff_values = lm_sm.params

        hb.write_to_file(str(lm_sm.summary()), os.path.join(output_dir, regression_label + '_summary.txt'))
        # lm_sm.params.to_csv(os.path.join(output_dir, regression_label + '_params.csv'))

        result = OrderedDict()
        result['depvar_label'] = self.dependent_variable_label

        result['coefficients'] = OrderedDict(zip(list(lm_sm.params.index), list(lm_sm.params.values)))
        result['pvalues'] = OrderedDict(zip(list(lm_sm.params.index), list(lm_sm.pvalues)))

        result['aic'] = lm_sm.aic
        result['bic'] = lm_sm.bic
        result['bse'] = lm_sm.bse
        result['rsquared'] = lm_sm.rsquared
        result['rsquared_adj'] = lm_sm.rsquared_adj
        result['nobs'] = lm_sm.nobs

        result['coefficient_means'] = {}
        for param_name in list(lm_sm.params.index):
            result['coefficient_means'][param_name] = df[param_name].mean()

        #
        # a.pvalues()
        result['equation_dict'] = equation_dict
        result['regression_label'] = regression_label
        self.results[regression_label] = result

        data_csv_path = os.path.join(output_dir, regression_label + '_data.csv')
        # # Save the all valid mask for later operations, such as projection.
        # if not os.path.exists(data_csv_path):
        #     hb.save_array_as_geotiff(p.all_valid_array, p.all_valid_sample_resolution_path, p.carbon_dependent_variable_path)
        #
        # # Save the DF to csv for potential later use
        # if not os.path.exists(data_csv_path):
        #     p.df.to_csv(data_csv_path)

    def run_lasso(self, variable_set_label, regression_label, df, output_dir=None, has_constant=True):
        self.coeff_labels = self.variable_sets[variable_set_label][1:] # NOTE dropping depvar
        depvar_label = self.variable_sets[variable_set_label][0]

        # Load X and y. For posterity
        df_copy = df.copy()

        X = df[self.coeff_labels]
        y = df.iloc[:, 0]

        # Split into train and test
        train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
        X_train = train[self.coeff_labels]
        X_test = test[self.coeff_labels]
        y_train = train.iloc[:, 0]
        y_test = test.iloc[:, 0]

        ## LassoLars Information Criterion optimization method
        # This could be worth running to show that we did further validation to show that our method for choosing the optimal alpha in the CV method resulted in something similar to an aic method.
        run_lassolarsic = False
        if run_lassolarsic:

            model_bic = LassoLarsIC(criterion='bic')
            t1 = time.time()
            model_bic.fit(X, y)
            t_bic = time.time() - t1
            alpha_bic_ = model_bic.alpha_

            model_aic = LassoLarsIC(criterion='aic', max_iter=45)
            model_aic.fit(X, y)
            alpha_aic_ = model_aic.alpha_

            def plot_ic_criterion(model, name, color):
                alpha_ = model.alpha_
                alphas_ = model.alphas_
                criterion_ = model.criterion_
                plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
                         linewidth=3, label='%s criterion' % name)
                plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                            label='alpha: %s estimate' % name)
                plt.xlabel('-log(alpha)')
                plt.ylabel('criterion')


            plt.figure()
            plot_ic_criterion(model_aic, 'AIC', 'b')
            plot_ic_criterion(model_bic, 'BIC', 'r')
            plt.legend()
            plt.title('Information-criterion for model selection (training time %.3fs)' % t_bic)


        ## LassoLars Cross Validation method
        # Compute paths
        t1 = time.time()
        model_larscv = LassoLarsCV(cv=5).fit(X, y)
        t_lasso_lars_cv = time.time() - t1

        # Display results
        m_log_alphas = -np.log10(model_larscv.cv_alphas_)

        plt.figure()
        endplot = int(len(m_log_alphas) * .25)
        plt.plot(m_log_alphas[endplot:], model_larscv.mse_path_[endplot:], ':')
        plt.plot(m_log_alphas[endplot:], model_larscv.mse_path_.mean(axis=-1)[endplot:], 'k',
                 label='Average across the folds', linewidth=2)
        plt.axvline(-np.log10(model_larscv.alpha_), linestyle='--', color='k',
                    label='alpha CV')
        plt.legend()

        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.title('Cross validation of LASSO-LARS to find optimal alpha')
        plt.axis('tight')
        plt.savefig(os.path.join(output_dir, 'LassoLarsCV-alpha.png'), dpi=350)
        plt.show()

        coefs = model_larscv.coef_path_[:,:]

        xx = np.sum(np.abs(coefs.T), axis=1)
        xx /= xx[-1]


        list_plot = np.asarray(list(sorted(coefs, key=sum)))


        plt.plot(xx, coefs.T)
        # for i in [0, 3, 33, 39, 40, 41, 42, 43, 44, 47, 49]:
        #     plt.plot(xx, coefs[i, :], label=X.columns[i].replace('lulc_esa_2000_dummy', ''))
        ymin, ymax = plt.ylim()
        plt.vlines(xx, ymin, ymax, linestyle='dashed', linewidth=0.2)
        plt.xlabel('Coefficient iteration path: |coef| / max|coef|')
        plt.ylabel('Coefficients')
        plt.title('LASSO Path for selected coefficients')
        # plt.axis('tight')
        # plt.legend( fontsize='8')
        # plt.tight_layout(h_pad=60)
        # plt.legend(bbox_to_anchor=(1.01, .99), fontsize='7', loc=2, borderaxespad=0.)
        plt.savefig(os.path.join(output_dir, 'LassoLarsCV-coef.png'), dpi=350)
        plt.show()

        self.coeff_values = model_larscv.coef_
        try:
            intercept = model_larscv.intercept_
        except:
            intercept = None
        result = OrderedDict()
        if intercept is None:
            result['coefficients_intermediate'] = OrderedDict(zip(list(X.columns), list(self.coeff_values)))
        else:
            result['coefficients_intermediate'] = OrderedDict(zip(['intercept'] + list(X.columns), [intercept] + list(self.coeff_values)))
        result['name'] = regression_label
        result['depvar_label'] = 'asdf'
        self.results[result['name']] = result




        ## OLS reinterpretation using Lasso Lars selected variables
        self.selected_variable_labels = [str(k) for k, v in result['coefficients_intermediate'].items() if v != 0.0 and k != 'intercept']


        X = df_copy[self.coeff_labels]
        y = df_copy.iloc[:, 0]

        # NOTE: This absolutely has to have hasconst if you want to interpret the r2
        model_lmsm = sm.OLS(y_train, X_train[self.selected_variable_labels], hasconst=has_constant).fit() # NOTE dropping intercept via list slice

        self.coeff_values = model_lmsm.params
        hb.write_to_file(str(model_lmsm.summary()), os.path.join(output_dir, regression_label + '_summary.txt'))
        model_lmsm.params.to_csv(os.path.join(output_dir, regression_label + '_params.csv'))


        # NOTE THAT THIS regression generates multiple results. The last one, here, is given the name regression name as it is the final output.
        result = OrderedDict()
        if intercept is None:
            result['coefficients'] = OrderedDict(zip(list(model_lmsm.params.index), list(self.coeff_values)))
        else:
            result['coefficients'] = OrderedDict(zip(['intercept'] + list(model_lmsm.params.index), [intercept] + list(self.coeff_values)))

        result['depvar_label'] = depvar_label
        result['variable_set_label'] = variable_set_label
        result['regression_label'] = regression_label

        # result['coefficients'] = OrderedDict(zip(list(model_lmsm.params.index), list(model_lmsm.params.values)))
        result['name'] = regression_label
        self.results[result['name']] = result

        plt.show()

    def run_skl_lm(self, variable_set_label, regression_label, df, output_dir=None):
        L.info('Starting to fit regression.')

        self.coeff_labels = self.variable_sets[variable_set_label][1:]  # NOTE dropping depvar
        depvar_label = self.variable_sets[variable_set_label][0]

        lm_skl = sklearn.linear_model.LinearRegression(normalize=True, fit_intercept=True)
        lm_skl.fit(df[self.coeff_labels], df.iloc[:, 0])


        self.coeff_values = lm_skl.coef_

        output_df = pd.DataFrame({'linear_regression': self.coeff_values}, index=self.coeff_labels)
        output_df.to_csv(os.path.join(output_dir, variable_set_label + '_skl_params.csv'))
        L.debug('Creating predictions from regression.')

        self.predictions = lm_skl.predict(df[self.coeff_labels])

        result = OrderedDict()
        result['coefficients'] = OrderedDict(zip(list(self.coeff_labels), list(self.coeff_values)))
        result['variable_set_label'] = variable_set_label
        result['regression_label'] = regression_label
        self.results[regression_label] = result

    def predict_output(self, regression_label, output_dir, current_bounding_box, replacement_dict=None):

        # Get dependent variable for this regression
        result = self.results[regression_label]
        depvar_label = result['depvar_label']
        # hb.pp('PREDICTION', result)

        if replacement_dict is None:
            replacement_dict = OrderedDict()

        # variable_set_label = result['variable_set_label']
        regression_label = result['regression_label']

        # TODO Add means etc here.

        depvar_source = self.global_aligned_inputs[self.dependent_variable_label]
        depvar_array = hb.load_geotiff_chunk_by_bb(self.global_aligned_inputs[self.dependent_variable_label].path,
                                                  current_bounding_box,
                                                  1,
                                                   output_path=self.current_dependent_variable_path) # Notice stride rate of 1 now that we are doing projecting.

        projected_array = np.full(depvar_array.shape, 0., dtype=np.float64)
        all_valid_array = np.where(depvar_array != -9999., 1, 0).astype(np.int8)

        for label, output_value in result['coefficients'].items():
            if label in replacement_dict:
                label = replacement_dict[label]

            coeff = np.float64(output_value)
            globally_aligned_input = self.global_aligned_inputs[label]
            array = hb.load_geotiff_chunk_by_bb(globally_aligned_input.path,
                                                              current_bounding_box,
                                                              1)
            # if label != 'intercept':
            #     globally_aligned_input = self.global_aligned_inputs[label]

            # if globally_aligned_input.data is None:
            #         source.data = hb.as_array(source.path).astype(np.float64)
            #         all_valid_array[source.data == -9999.] = 0
            #     array = source.data ** 2

            projected_array += array * coeff
            array = None






        # # TODO Add transforms here.
        # for label, output_value in result['coefficients'].items():
        #     if label in replacement_dict:
        #         label = replacement_dict[label]
        #
        #     coeff = np.float64(output_value)
        #
        #     L.info('Creating projected map from ' + str(label) + ' with coefficient ' + str(coeff))
        #     if label != 'intercept':
        #         source = self.variables[label].source
        #     else:
        #         L.info('intercept', output_value)
        #     # TODOO Potential memory optimization here: iterate throug sources first, THEN variables, unloading as needed.
        #     if label == 'intercept':
        #         # TODO why zero not one?
        #         array = np.zeros_like(projected_array, dtype=np.float64)
        #     # elif '_square' in label:
        #     #     if source.data is None:
        #     #         source.data = hb.as_array(source.path).astype(np.float64)
        #     #         all_valid_array[source.data == -9999.] = 0
        #     #     array = source.data ** 2
        #     # elif '_cube' in label:
        #     #     if source.data is None:
        #     #         source.data = hb.as_array(source.path).astype(np.float64)
        #     #         all_valid_array[source.data == -9999.] = 0
        #     #     array = source.data ** 3
        #     # elif '_quad' in label:
        #     #     if source.data is None:
        #     #         source.data = hb.as_array(source.path).astype(np.float64)
        #     #         all_valid_array[source.data == -9999.] = 0
        #     #     array = source.data ** 4
        #     # elif '_quint' in label:
        #     #     if source.data is None:
        #     #         source.data = hb.as_array(source.path).astype(np.float64)
        #     #         all_valid_array[source.data == -9999.] = 0
        #     #     array = source.data ** 5
        #     # elif '_ln' in label:
        #     #     if source.data is None:
        #     #         source.data = hb.as_array(source.path).astype(np.float64)
        #     #         all_valid_array[source.data == -9999.] = 0
        #     #     array = np.log(source.data)
        #     # elif '_log10' in label:
        #     #     if source.data is None:
        #     #         source.data = hb.as_array(source.path).astype(np.float64)
        #     #         all_valid_array[source.data == -9999.] = 0
        #     #     array = np.log10(source.data)
        #
        #     # NOTE WEIRDNESS, interact has to be before dummy otherwise interactions on dummies wont be picked up as it grabs the first one.
        #     elif 'interact' in label:
        #         var1_label, var2_label = label.split('_interact_')
        #         source1 = self.variables[var1_label].source
        #         source2 = self.variables[var2_label].source
        #         dummy_value = float(var2_label.split('dummy')[1])
        #         if source1.data is None:
        #             source1.data = hb.as_array(source1.path).astype(np.float64)
        #             all_valid_array[source1.data == -9999.] = 0
        #         if source2.data is None:
        #             source2.data = hb.as_array(source2.path).astype(np.float64)
        #             all_valid_array[source2.data == -9999.] = 0
        #         array = source1.data * np.where(source2.data == dummy_value, 1., 0.)
        #     elif '_convolve' in label:
        #         class_id_from_label = int(label.split('_dummy')[1].split('_')[0])
        #         esa_class_id = [k for k, v in hb.esacci_extended_short_class_descriptions.items() if k == class_id_from_label][0]
        #
        #         if source.data is None:
        #             source.data = hb.as_array(source.path).astype(np.float64)
        #             all_valid_array[source.data == -9999.] = 0
        #
        #         array = source.data
        #
        #     elif '_dummy' in label:
        #         class_id_from_label = int(label.split('_dummy')[-1].split('_')[0])
        #         esa_class_id = [k for k, v in hb.esacci_extended_short_class_descriptions.items() if k == class_id_from_label][0]
        #
        #         if source.data is None:
        #             source.data = hb.as_array(source.path).astype(np.float64)
        #             all_valid_array[source.data == -9999.] = 0
        #         array = np.where(source.data == esa_class_id, 1, 0)
        #
        #     else:  # Then its a standard, untransformed variable
        #
        #         if source.data is None:
        #             source.data = hb.as_array(source.path).astype(np.float64)
        #
        #             all_valid_array[source.data == -9999.] = 0
        #         array = source.data
        #     # Only calculate projection for where there is a full pixelstack of observation data
        #     # array = np.where(all_valid_array == 1, array, 0)
        #
        #     projected_array += array * coeff
        #     array = None
        #     source.data = None
        #     L.info('sum added: ' + str(np.sum(array)) + ', projected mean: ' + str(np.mean(projected_array)))


        hb.save_array_as_geotiff(all_valid_array, hb.temp(), self.current_dependent_variable_path)
        projected_array *= all_valid_array
        projected_path = os.path.join(output_dir, regression_label, 'agb_' + regression_label + '.tif')
        hb.save_array_as_geotiff(projected_array, projected_path, self.current_dependent_variable_path, data_type=6, ndv=-9999.)

        residuals_array = (projected_array - depvar_array) * all_valid_array
        residuals_path = os.path.join(output_dir, regression_label, 'residuals_' + regression_label + '.tif')
        hb.save_array_as_geotiff(residuals_array, residuals_path, self.current_dependent_variable_path, data_type=6, ndv=-9999.)

        all_valid_prediction_path = os.path.join(output_dir, regression_label, 'all_valid_' + regression_label + '.tif')
        hb.save_array_as_geotiff(all_valid_array, all_valid_prediction_path, self.current_dependent_variable_path, data_type=6, ndv=-9999.)

# The following regression objects define a heirarcy of inputs at different stages of preprocessing and different sets for later use.

class RegressionInput(object):
    """Only used to assign inputs. This is the element the user modifies on input."""
    def __init__(self, label, path, rf, tags=None):
        self.label = label
        self.path = path
        self.data = None
        self.rf = rf

        if tags is None:
            self.tags = ['linear']
        else:
            self.tags = tags

    def __str__(self):
        return '<RegressionInput, ' + str(self.label) + ', ' + str(self.path) + '>'

    def __repr__(self):
        return '<RegressionInput, ' + str(self.label) + ', ' + str(self.path) + '>'


class RegressionAlignedInput(object):
    """An aligned input references a raster that in the same shape as the regression extent.
    But, it doesnt necessarily have NDG or other checks done"""
    def __init__(self, label, path, rf, tags=None):
        self.label = label
        self.path = path
        self.data = None
        self.rf = rf
        self.tags = tags

    def __str__(self):
        return '<RegressionAlignedInput, ' + str(self.label) + ', ' + str(self.path) + '>'

    def __repr__(self):
        return '<RegressionAlignedInput, ' + str(self.label) + ', ' + str(self.path) + '>'


class RegressionGlobalAlignedInput(object):
    """An global aligned input references a raster that can be referenced by row and col without worrying about projeciton or extent.
    But, it doesnt necessarily have NDG or other checks done"""
    def __init__(self, label, path, rf, tags=None):
        self.label = label
        self.path = path
        self.data = None
        self.rf = rf
        self.tags = tags

        # Check that the path exists
        if not os.path.exists(self.path):
            L.critical('RegressionSource', self.path, 'does not exist')

        self.ndv = hb.get_raster_info_hb(path)['ndv']

        if self.ndv == None:
            raise NameError('No NDV set in ' + str(path))


    def __str__(self):
        return '<RegressionGlobalAlignedInput, ' + str(self.label) + ', ' + str(self.path) + '>'

    def __repr__(self):
        return '<RegressionGlobalAlignedInput, ' + str(self.label) + ', ' + str(self.path) + '>'


class RegressionSource(object):
    """Regression Sources are aligned inputs (or other non-aligned pyramidal objects),
     that are able to access gdal.Band object to get e.g. ndv and check that the file is readable."""
    def __init__(self, label, path, rf, tags=None):
        self.label = label
        self.path = path
        self.data = None
        self.rf = rf
        self.tags = tags

        # Check that the path exists
        if not os.path.exists(self.path):
            L.critical('RegressionSource', self.path, 'does not exist')

        self.ndv = hb.get_raster_info_hb(path)['ndv']

        if self.ndv == None:
            raise NameError('No NDV set in ' + str(path))

    def __str__(self):
        return '<RegressionSource, ' + str(self.label) + ', ' + str(self.path) + ', tags: ' + str(self.tags) + '>'

    def __repr__(self):
        return '<RegressionSource, ' + str(self.label) + ', ' + str(self.path) + ', tags: ' + str(self.tags) + '>'

class RegressionVariable(object):
    """Regression variables reference a source and a transform necessary to fully specify what is needed
    in the regression (but without actually loading any data)."""
    def __init__(self, label, source_label, rf, variable_type=None, tags=None):
        self.label = label
        self.source_label = source_label
        self.rf = rf
        self.source = self.rf.sources[source_label]
        self.path = self.source.path
        self.variable_type = variable_type
        self.tags = tags
        # self.variable_type = variable_type
        self.ndv = self.source.ndv

    def __str__(self):
        return '<RegressionVariable, ' + str(self.label) + ', ' + str(self.path) + ', ' + str(self.variable_type) + '>'

    def __repr__(self):
        return '<RegressionVariable, ' + str(self.label) + ', ' + str(self.path) + ', ' + str(self.variable_type) + '>'

def parse_equation_to_dict(equation):
    L.debug('Parsing equation: ' + str(equation))
    equation_dict = {}
    equation_dict['necessary_variables'] = [] # Things that need to be loaded into the DF
    equation_dict['variable_transforms'] = [] # Modifications of things in the DF
    equation_dict['regression_terms'] = [] # Set of the above that is actually used as a variable. (eg doesn't include the dumy generator)
    equation_dict['mask_variables'] = [] # Variables that are used to drop no data etc. Computationally expensive to add new ones as it regens the whole df.

    operator_to_text = {}
    operator_to_text['*'] = '*'
    operator_to_text['('] = '('
    operator_to_text[')'] = ')'
    operator_to_text['^'] = '^'
    equation = equation.replace('\n', '')
    equation = equation.replace(' ', '')

    dependent_variable, rhs = equation.split('~')
    equation_dict['necessary_variables'].append(dependent_variable)

    equation_dict['dependent_variable'] = dependent_variable
    splitting_operators = ['+', '-']
    internal_operators = [':', '*', '#', '|', 'log', '^', '(', ')', 'dummy', '[', ']']
    numeral_indicator = '#'
    current_string = ''
    current_operators = []

    re_split = re.split('(\+|-|:|\*|\^|\(|log|mask|dummy|\||\)|\[|\])', rhs)
    # hb.pp(re_split)
    # re_split = re.split('(\W|log)', rhs)
    level_1 = []
    re_split = [i for i in re_split if i != '']
    re_split += ['+'] # Hackish way to make sure the last term is included.
    for c, i in enumerate(re_split):
        if i == '+':
            level_1.append(current_operators)
            current_operators = []
        else:
            current_operators.append(i)

    for c, i in enumerate(level_1):
        if len(i) == 1:
            equation_dict['necessary_variables'].append(i[0])
            equation_dict['regression_terms'].append(i[0])
        elif isinstance(i, list):
            if i[0] == 'dummy':
                equation_dict['necessary_variables'].append(i[2].split(',')[0])  #i[3] is the name of the variable being made into a dummy
                equation_dict['regression_terms'].append(i[2].split(',')[1])  #i[3] is the name of the variable being made into a dummy
                equation_dict['variable_transforms'].append(i)
            elif i[0] == 'mask':
                equation_dict['necessary_variables'].append(i[2].split(',')[0])  #i[3] is the name of the variable being made into a dummy
                # equation_dict['regression_terms'].append(i[2].split(',')[1])  #i[3] is the name of the variable being made into a dummy
                equation_dict['variable_transforms'].append(i)
                equation_dict['mask_variables'].append(i)
            else:
                equation_dict['variable_transforms'].append(i)
                for cc, ii in enumerate(i):
                    if isinstance(ii, str) and ii not in internal_operators and len(ii) > 1:
                        equation_dict['necessary_variables'].append(ii)
                generated_name = ''.join(i)
                # generated_name = i[0] + operator_to_text[i[1]] + i[2]
                equation_dict['regression_terms'].append(generated_name)



    new_equation_list = []
    for i in equation_dict['necessary_variables']:
        if i not in new_equation_list:
            new_equation_list.append(i)


    equation_dict['necessary_variables'] = new_equation_list

    # LEARNING POINT, using a set made it lose it's order
    # equation_dict['necessary_variables'] = list(set(equation_dict['necessary_variables']))

    return equation_dict
def generate_gaussian_kernel(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array. kernlen determines the size (always choose ODD numbers unless you're baller cause of asymmetric results.
    nsig is the signma blur. HAving it too small makes the blur not hit zero before the edge."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()

    return kernel


def load_rf_from_path(input_path):
    with open(input_path, 'rb') as fp:
        rf = pickle.load(fp)
    return rf


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
        #output_uri = os.path.join(hb.config.TEMP_FOLDER, hb.ruri('generated_r_output.txt'))
        output_uri = hb.config.TEMPORARY_DIR + '/' + hb.ruri('generated_r_output.txt')
        if not keep_files:
            hb.uris_to_delete_at_exit.append(output_uri)
    else:
        if '\\\\' in output_uri:
            output_uri = output_uri.replace('\\\\', '/')
        elif '\\' in output_uri:
            output_uri = output_uri.replace('\\', '/')
        else:
            pass # output_uri  was okay
    if not script_save_uri:
        script_save_uri = os.path.join(hb.config.TEMPORARY_DIR, hb.ruri('generated_r_script.R'))
        if not keep_files:
            hb.uris_to_delete_at_exit.append(script_save_uri)

    r_string = r_string.replace('\\', '/')

    print (r_string)
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
        hb.L.warning('Executed r script but no output was found.')


def convert_af_to_1d_df(af):
    array = af.data.flatten()
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


def convert_af_to_df(input_af, output_column_name=None):
    if not output_column_name:
        output_column_name = 'f_af_' + hb.random_alphanumeric_string(3)

    data = input_af.data.flatten()

    df = pd.DataFrame(data=data,  # values
                      index=np.arange(0, len(data)),  # 1st column as index
                      columns=[output_column_name])  # 1st row as the column names
    return df

def convert_df_to_af_via_index(input_df, column, match_af, output_uri):
    match_df = hb.convert_af_to_1d_df(match_af)
    if match_af.size != len(input_df.index):
        full_df = pd.DataFrame(np.zeros(match_af.size), index=match_df.index)
    else:
        full_df = input_df

    listed_index = np.array(list(input_df.index))

    full_df[0][listed_index] = input_df['0']
    # full_df[0][input_df.index] = input_df[column]

    array = full_df.values.reshape(match_af.shape)
    hb.ArrayFrame(array, match_af, output_uri=output_uri)
    af = hb.ArrayFrame(output_uri)

    return af

