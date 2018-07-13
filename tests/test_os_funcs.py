from unittest import TestCase
import os, sys

from hazelbean.os_utils import *

class DataStructuresTester(TestCase):
    def setUp(self): # Name has to be exactly this
        pass

    def tearDown(self): # Name has to be exactly this
        pass

    def test_make_and_remove_folders(self):
        folder_list = ['asdf', 'asdf/qwer']
        create_dirs(folder_list)
        remove_dirs(folder_list, safety_check='delete')

    def test_make_run_dir(self):
        pass

    def test_temporary_dir(self):
        pass

    def test_remove_folder(self):
        pass

    def test_temp(self):
        pass

    def test_temp_filename(self):
        pass

    def test_temporary_filename(self):
        pass

    def test_temporary_dir(self):
        pass

    def test_remove_folder(self):
        pass

    def test_quad_split_path(self):
        pass

    def test_random_numerals_string(self):
        pass

    def test_random_lowercase_string(self):
        pass

    def test_random_alphanumeric_string(self):
        pass

    def test_convert_string_to_implied_type(self):
        pass

    def test_random_string(self):
        pass

    def test_pretty_time(self):
        pass

    def test_suri(self):
        pass

    def test_insert_string_before_ext(self):
        pass

    def test_ruri(self):
        pass

    def test_insert_random_string_before_ext(self):
        pass

    def test_rsuri(self):
        pass

    def test_insert_string_and_random_string_before_ext(self):
        pass

    def test_create_dirs(self):
        pass

    def test_remove_dirs(self):
        pass

    def test_execute_2to3_on_folder(self):
        pass

    def test_list_mounted_drive_paths(self):
        pass

    def test_list_files_in_dir_recursively(self):
        pass

    def test_list_dirs_in_dir_recursively(self):
        pass

    def test_assert_file_existance(self):
        pass

    def test_remove_temporary_files(self):
        pass

    def test_remove_uri_at_exit(self):
        pass

    def test_remove_at_exit(self):
        pass

    def test_create_directories(self):
        pass

