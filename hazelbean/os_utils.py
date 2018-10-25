import atexit
import datetime
import errno
import os
import random
import shutil
import sys
import distutils
from distutils import dir_util # NEEDED

import hazelbean as hb
import zipfile
from collections import OrderedDict
import warnings
import stat
L = hb.get_logger('os_utils')

def make_run_dir(base_folder=hb.config.TEMPORARY_DIR, run_name='', just_return_string=False):
    """Create a directory in a preconfigured location. Does not delete by default. Returns path of dir."""
    run_dir = os.path.join(base_folder, ruri(run_name))
    if not os.path.exists(run_dir):
        if not just_return_string:
            os.makedirs(run_dir)
    else:
        raise Exception('This should not happen as the temp file has a random name.')
    return run_dir

# # TODOO Useful
# def get_last_run_dirs(num_to_get=10, override_default_temp_dir=None):
#     if override_default_temp_dir:
#         temp_dir = override_default_temp_dir
#     else:
#         temp_dir = hb.TEMPORARY_DIR
#     all_run_dirs = [i for i in os.listdir(temp_dir) if os.path.isdir(i)]
#
#     for possible_dir in all_run_dirs:
#         get_time_stamp_from_string(possible_dir)
#
# def check_for_file_path_in_last_n_run_dirs():
#     pass


def temp(ext=None, filename_start=None, remove_at_exit=False, folder=None, suffix=''):
    if ext:
        if not ext.startswith('.'):
            ext = '.' + ext

    if filename_start:
        if ext:
            filename = ruri(filename_start + ext)
        else:
            filename = ruri(filename_start + '.tif')
    else:
        if ext:
            filename = ruri('tmp' + ext)
        else:
            filename = ruri('tmp.tif')

    if folder is not None:
        uri = os.path.join(folder, filename)
    else:
        uri = os.path.join(hb.config.TEMPORARY_DIR, filename)

    if remove_at_exit:
        remove_uri_at_exit(uri)

    return uri


def temp_filename(ext=None, filename_start=None, remove_at_exit=True, folder=None, suffix=''):
    return temp(ext=ext, filename_start=filename_start, remove_at_exit=remove_at_exit, folder=folder, suffix=suffix)


def temporary_filename(filename_start=None, ext='', remove_at_exit=True, folder=None, suffix=''):
    return temp(ext=ext, filename_start=filename_start, remove_at_exit=remove_at_exit, folder=folder, suffix=suffix)


def temporary_dir(dirname_prefix=None, dirname_suffix=None, remove_at_exit=True):
    """Get path to new temporary folder that will be deleted on program exit.

    he folder is deleted on exit
    using the atexit register.

    Returns:
        path (string): an absolute, unique and temporary folder path. All underneath will be deleted.
    """
    pre_post_string = 'tmp'
    if dirname_prefix:
        pre_post_string = dirname_prefix + '_' + pre_post_string
    if dirname_suffix:
        pre_post_string += '_' + dir_suffix

    path = os.path.join(hb.config.TEMPORARY_DIR, ruri(pre_post_string))
    if os.path.exists(path):
        raise FileExistsError()
    else:
        os.mkdir(path)

    def remove_folder(path):
        """Function to remove a folder and handle exceptions encountered.  This
        function will be registered in atexit."""
        shutil.rmtree(path, ignore_errors=True)

    if remove_at_exit:
        atexit.register(remove_folder, path)

    return path


def temporary_folder():
    """
    Returns a temporary folder using mkdtemp.  The folder is deleted on exit
    using the atexit register.

    Returns:
        path (string): an absolute, unique and temporary folder path.

    """
    warnings.warn('Deprecated for DIR method.')
    path = tempfile.mkdtemp()

    def remove_folder(path):
        """Function to remove a folder and handle exceptions encountered.  This
        function will be registered in atexit."""
        shutil.rmtree(path, ignore_errors=True)

    atexit.register(remove_folder, path)
    return path


def quad_split_path(input_uri):
    '''
    Splits a path into prior directories path, parent directory, basename (extensionless filename), file extension.
    :return: list of [prior directories path, parent directory, basename (extensionless filename), file extension]
    '''
    a, file_extension = os.path.splitext(input_uri)
    b, file_root = os.path.split(a)
    prior_path, parent_directory = os.path.split(b)

    return [prior_path, parent_directory, file_root, file_extension]




def random_numerals_string(length=6):
    max_value = int(''.join(['9'] * length))
    to_return = str(random.randint(0, max_value)).zfill(length)
    return to_return

def random_lowercase_string(length=6):
    """Returns randomly chosen, lowercase characters as a string with given length. Uses chr(int) to convert random."""
    random_ints = [random.randint(hb.config.start_of_lowercase_letters_ascii_int, hb.config.start_of_lowercase_letters_ascii_int + 26)  for i in range(length)]
    random_chars = [chr(i) for i in random_ints]
    return ''.join(random_chars)

def random_alphanumeric_string(length=6):
    """Returns randomly chosen, lowercase characters as a string with given length. Uses chr(int) to convert random."""
    random_chars = [random.choice(hb.config.alphanumeric_lowercase_ascii_symbols) for i in range(length)]

    return ''.join(random_chars)


def convert_string_to_implied_type(input_string):
    if input_string in ['TRUE', 'True', 'true', 'T', 't']:
        return True
    if input_string in ['FALSE', 'False', 'false', 'F', 'f']:
        return False

    try:
        floated = float(input_string)
    except:
        floated = False

    try:
        inted = int(input_string)
    except:
        inted = False

    if '.'  in input_string and floated:
        return floated

    if inted:
        return inted

    return input_string

def random_string():
    """Return random string of numbers of expected length. Used in uri manipulation."""
    return pretty_time(format='full') + str(random_alphanumeric_string(3))

def pretty_time(format=None):
    # Returns a nicely formated string of YEAR-MONTH-DAY_HOURS-MIN-SECONDS based on the the linux timestamp
    now = str(datetime.datetime.now())
    day, time = now.split(' ')
    day = day.replace('-', '')
    time = time.replace(':', '')
    if '.' in time:
        time, milliseconds = time.split('.')
        milliseconds = milliseconds[0:3]
    else:
        milliseconds = '000'

    if not format:
        return day + '_' + time
    elif format == 'full':
        return day + '_' + time + '_' + milliseconds
    elif format == 'day':
        return day


def explode_uri(input_uri):
    return explode_path(input_uri)

def explode_path(input_path):
    """
    Returns a dictionary with the following key-value pairs:

        path
        dir_name
        filename
        file_root
        file_extension
        parent_directory
        grandparent_path
        grandparent_directory
        great_grandparent_path
        root_directory
        post_root_directories
        post_root_path
        root_child_directory
        post_root_child_directories
        post_root_child_path
        file_root_no_suffix
        file_root_suffix
        file_root_no_timestamp
        file_root_date
        file_root_time
        file_root_no_timestamp_or_suffix
        parent_directory_no_suffix
        parent_directory_suffix
        parent_directory_no_timestamp
        parent_directory_date
        parent_directory_time
        parent_directory_datetime
        parent_directory_no_timestamp_or_suffix
        drive
        drive_no_slash
        post_drive_path
        post_drive_dir_name_and_file_root
        fragments
        path_directories

    """
    # Uses syntax from path.os
    curdir = '.'
    pardir = '..'
    extsep = '.'
    sep = '\\'
    pathsep = ';'
    altsep = '/'
    defpath = '.;C:\\bin'
    suffixsep = '_'

    L.debug_deeper_1('Exploding ' + input_path)

    try:
        os.path.split(input_path)
    except:
        raise NameError('Unable to process input_path of ' + input_path + ' with os.path.split().')

    if pathsep in input_path:
        raise NameError('Usage of semicolon for multiple paths is not yet supported.')

    normcase_uri = os.path.normcase(input_path) # Use normpath if you want path optimizations besides case and //. Makes everything ahve the separator '\\'
    drive, post_drive_path = os.path.splitdrive(normcase_uri)
    if drive:
        drive = drive + '\\'

    if post_drive_path.startswith(sep):
        post_drive_path = post_drive_path[1:]

    post_drive_dir_name_and_file_root, file_extension = os.path.splitext(post_drive_path)

    if file_extension:
        post_drive_dir_name = os.path.split(post_drive_path)[0]
        dir_name_and_file_root = os.path.splitext(normcase_uri)[0]
        dir_name, file_root = os.path.split(dir_name_and_file_root)
    else:
        post_drive_dir_name = post_drive_path
        dir_name_and_file_root, file_extension = os.path.splitext(normcase_uri)
        dir_name, file_root = dir_name_and_file_root, ''

    filename = file_root + file_extension
    grandparent_path, parent_directory = os.path.split(dir_name)
    great_grandparent_path, grandparent_directory = os.path.split(grandparent_path)

    if os.path.splitext(post_drive_path):
        post_drive_path_without_files = os.path.split(post_drive_path)[0]
        n_post_drive_directories = len(post_drive_path.split(sep)) - 1
    else:
        n_post_drive_directories = len(post_drive_path.split(sep))

    if n_post_drive_directories == 0:
        root_directory, post_root_directories, root_child_directory, post_root_child_directories = os.path.join(drive, ''), '', '', ''
    elif n_post_drive_directories == 1:
        root_directory = os.path.join(drive, post_drive_path_without_files)
        post_root_directories, root_child_directory, post_root_child_directories = '', '', ''
    elif n_post_drive_directories == 2:
        root_directory, post_root_directories = post_drive_path_without_files.split(sep, 1)
        root_directory = os.path.join(drive, root_directory)
        root_child_directory, post_root_child_directories = post_root_directories, ''
    elif n_post_drive_directories == 3:
        root_directory, post_root_directories = post_drive_path_without_files.split(sep, 1)
        root_directory = os.path.join(drive, root_directory)
        root_child_directory = post_root_directories
        post_root_child_directories = ''
    else:
        root_directory, post_root_directories = post_drive_path_without_files.split(sep, 1)
        root_child_directory, post_root_child_directories = post_root_directories.split(sep, 1)
    post_root_path = os.path.join(post_root_directories, filename)
    post_root_child_path = os.path.join(post_root_child_directories, filename)

    file_root_split = file_root.split(suffixsep)
    split_file_root_reversed = file_root_split[::-1]

    file_root_has_timestamp = False
    file_root_has_suffix = False

    try:
        if len(split_file_root_reversed[0]) == 6 and len(split_file_root_reversed[1]) == 6 and len(split_file_root_reversed[2]) == 8 and split_file_root_reversed[1].isdigit() and split_file_root_reversed[2].isdigit():
            file_root_has_suffix = False
            file_root_has_timestamp = True
            file_root_has_timestamp_but_no_suffix = True
        elif len(split_file_root_reversed[1]) == 6 and len(split_file_root_reversed[2]) == 6 and len(split_file_root_reversed[3]) == 8 and split_file_root_reversed[2].isdigit() and split_file_root_reversed[3].isdigit():
            file_root_has_suffix = True
            file_root_has_timestamp = True
            file_root_has_timestamp_but_no_suffix = False
        else:
            file_root_has_timestamp = False
            file_root_has_suffix = False
            file_root_has_timestamp_but_no_suffix = False
    except:
        file_root_has_timestamp = False
        file_root_has_suffix = False

    if file_root_has_timestamp:
        if file_root_has_timestamp_but_no_suffix:
            a = file_root.rsplit(suffixsep, 3)
            file_root_no_suffix, file_root_suffix = file_root, ''
            file_root_no_timestamp, file_root_date, file_root_time = a[0], a[1], a[2]
            file_root_no_timestamp_or_suffix = a[0]
        else:
            a = file_root.rsplit(suffixsep, 4)
            file_root_no_suffix, file_root_suffix = a[0] + '_' + a[1] +'_' +  a[2] +'_' +  a[3], a[4]
            file_root_no_timestamp, file_root_date, file_root_time = a[0] + '_' + a[4] , a[1], a[2]
            file_root_no_timestamp_or_suffix, file_root_date, file_root_time = a[0], a[1], a[2]
    else:
        if '_' in file_root:
            file_root_no_timestamp_or_suffix, file_root_suffix = file_root.rsplit('_', 1)
            file_root_date, file_root_time = '', ''
            file_root_no_suffix = file_root_no_timestamp_or_suffix
            file_root_no_timestamp = file_root
        else:
            file_root_no_timestamp_or_suffix, file_root_suffix = file_root, ''
            file_root_date, file_root_time = '', ''
            file_root_no_suffix = file_root_no_timestamp_or_suffix
            file_root_no_timestamp = file_root

    parent_directory_split = parent_directory.split(suffixsep)
    split_parent_directory_reversed = parent_directory_split[::-1]

    parent_directory_has_timestamp = False
    parent_directory_has_suffix = False

    try:
        if len(split_parent_directory_reversed[0]) == 6 and len(split_parent_directory_reversed[1]) == 6 and len(split_parent_directory_reversed[2]) == 8 and split_parent_directory_reversed[1].isdigit() and split_parent_directory_reversed[2].isdigit():
            parent_directory_has_suffix = False
            parent_directory_has_timestamp = True
            parent_directory_has_timestamp_but_no_suffix = True
        elif len(split_parent_directory_reversed[1]) == 6 and len(split_parent_directory_reversed[2]) == 6 and len(split_parent_directory_reversed[3]) == 8 and split_parent_directory_reversed[2].isdigit() and split_parent_directory_reversed[3].isdigit():
            parent_directory_has_suffix = True
            parent_directory_has_timestamp = True
            parent_directory_has_timestamp_but_no_suffix = False
        else:
            parent_directory_has_timestamp = False
            parent_directory_has_suffix = False
            parent_directory_has_timestamp_but_no_suffix = False
    except:
        parent_directory_has_timestamp = False
        parent_directory_has_suffix = False

    if parent_directory_has_timestamp:
        if parent_directory_has_timestamp_but_no_suffix:
            a = parent_directory.rsplit(suffixsep, 3)
            parent_directory_no_suffix, parent_directory_suffix = parent_directory, ''
            parent_directory_no_timestamp, parent_directory_date, parent_directory_time = a[0], a[1], a[2]
            parent_directory_no_timestamp_or_suffix = a[0]
        else:
            a = parent_directory.rsplit(suffixsep, 4)
            parent_directory_no_suffix, parent_directory_suffix = a[0] + '_' + a[1] +'_' +  a[2] +'_' +  a[3], a[4]
            parent_directory_no_timestamp, parent_directory_date, parent_directory_time = a[0] + '_' + a[4] , a[1], a[2]
            parent_directory_no_timestamp_or_suffix, parent_directory_date, parent_directory_time = a[0], a[1], a[2]
    else:
        if '_' in parent_directory:
            parent_directory_no_timestamp_or_suffix, parent_directory_suffix = parent_directory.rsplit('_', 1)
            parent_directory_date, parent_directory_time = '', ''
            parent_directory_no_suffix = parent_directory_no_timestamp_or_suffix
            parent_directory_no_timestamp = parent_directory
        else:
            parent_directory_no_timestamp_or_suffix, parent_directory_suffix = parent_directory, ''
            parent_directory_date, parent_directory_time = '', ''
            parent_directory_no_suffix = parent_directory_no_timestamp_or_suffix
            parent_directory_no_timestamp = parent_directory


    # Fragments is a list of all the parts where if it is joined via eg ''.join(fragments) it will recreate the URI.
    fragments = []
    if drive:
        drive_no_slash = drive[0:2]
        fragments.append(drive) # NOTE Because the print method on str of list returns //, this will look different for the drive.
    else:
        drive_no_slash = ''

    split_dirs = post_drive_dir_name.split(sep)
    for i in range(len(split_dirs)):
        if split_dirs[i]:
            fragments.append(split_dirs[i])
    if filename:
        fragments.append(filename)

    path_directories = fragments[1:len(fragments)-1] # NOTE the implicit - 2

    exploded_uri = OrderedDict()
    exploded_uri['path'] = normcase_uri

    exploded_uri['dir_name'] = dir_name
    exploded_uri['filename'] = filename

    exploded_uri['file_root'] = file_root
    exploded_uri['file_extension'] = file_extension

    exploded_uri['parent_directory'] = parent_directory
    exploded_uri['grandparent_path'] = grandparent_path

    exploded_uri['grandparent_directory'] = grandparent_directory
    exploded_uri['great_grandparent_path'] = great_grandparent_path

    exploded_uri['root_directory'] = root_directory
    exploded_uri['post_root_directories'] = post_root_directories
    exploded_uri['post_root_path'] = post_root_path
    exploded_uri['root_child_directory'] = root_child_directory
    exploded_uri['post_root_child_directories'] = post_root_child_directories
    exploded_uri['post_root_child_path'] = post_root_child_path

    exploded_uri['file_root_no_suffix'] = file_root_no_suffix
    exploded_uri['file_root_suffix'] = file_root_suffix
    exploded_uri['file_root_no_timestamp'] = file_root_no_timestamp
    exploded_uri['file_root_date'] = file_root_date
    exploded_uri['file_root_time'] = file_root_time
    exploded_uri['file_root_no_timestamp_or_suffix'] = file_root_no_timestamp_or_suffix

    exploded_uri['parent_directory_no_suffix'] = parent_directory_no_suffix
    exploded_uri['parent_directory_suffix'] = parent_directory_suffix
    exploded_uri['parent_directory_no_timestamp'] = parent_directory_no_timestamp
    exploded_uri['parent_directory_date'] = parent_directory_date
    exploded_uri['parent_directory_time'] = parent_directory_time
    exploded_uri['parent_directory_datetime'] = parent_directory_date + '_' + parent_directory_time
    exploded_uri['parent_directory_no_timestamp_or_suffix'] = parent_directory_no_timestamp_or_suffix

    exploded_uri['drive'] = drive
    exploded_uri['drive_no_slash'] = drive_no_slash
    exploded_uri['post_drive_path'] = post_drive_path
    exploded_uri['post_drive_dir_name_and_file_root'] = post_drive_dir_name_and_file_root
    exploded_uri['fragments'] = fragments
    exploded_uri['path_directories'] = path_directories

    return exploded_uri


def suri(input_uri, input_string):
    '''Shortcut function to insert_string_before_ext'''
    return insert_string_before_ext(input_uri, input_string)


def insert_string_before_ext(input_uri, input_string):
    # The following helper functions are useful for quickly creating temporary files that resemble but dont overwrite
    # their input. The one confusing point i have so far is that calling this on a folder creates a string representing
    # a subfolder, not an in-situ new folder.

    # split_uri = os.path.splitext(input_uri)
    # return os.path.join(split_uri[0] + '_' + str(input_string) + split_uri[1])
    split_uri = os.path.splitext(input_uri)
    if split_uri[1]:
        output_uri = split_uri[0] + '_' + str(input_string) + split_uri[1]
    else:
        output_uri = split_uri[0] + str(input_string)
    return output_uri


def ruri(input_uri):
    '''Shortcut function to insert_random_string_before_ext'''
    return insert_random_string_before_ext(input_uri)


def insert_random_string_before_ext(input_uri):
    split_uri = os.path.splitext(input_uri)
    if split_uri[1]:
        output_uri = split_uri[0] + '_' + random_string() + split_uri[1]
    else:
        # If it's a folder, just tack it onto the end
        output_uri = split_uri[0] + '_' + random_string()
    return output_uri


def rsuri(input_uri, input_string):
    return insert_string_and_random_string_before_ext(input_uri, input_string)


def insert_string_and_random_string_before_ext(input_uri, input_string):
    split_uri = os.path.splitext(input_uri)
    if split_uri[1]:
        output_uri = split_uri[0] + '_' + str(input_string) + '_' + random_string() + split_uri[1]
    else:
        output_uri = split_uri[0] + '_' + str(input_string) + '_' + random_string()
    return output_uri





def create_dirs(list_of_folders):
    L.critical('Deprecated. Use create_directories.')
    if type(list_of_folders) is str:
        list_of_folders = [list_of_folders]

    for folder in list_of_folders:
        try:
            os.makedirs(folder, exist_ok=True)
        except:
            raise NameError('create_dirs() failed to make ' + folder)

def remove_dirs(list_of_folders, safety_check=''):
    if safety_check == 'delete':
        if list_of_folders is str:
            list_of_folders = list(list_of_folders)
        for folder in list_of_folders:
            if folder == '':
                raise NameError('remove_dirs() told to remove current directory (\'\'). This is not allowed.')
            if folder == '/' or folder == '\\' or folder == '\\\\' or folder == '..' or folder == '.' or '*' in folder:
                raise NameError('remove_dirs() given a protected symbol. This is not allowed.')
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder, ignore_errors=True)
                except:
                    raise NameError('remove_dirs() failed to remove ' + folder)
    else:
        raise NameError('remove_dirs() called but saftety_check did not equal \"delete\"')

def execute_2to3_on_folder(input_folder, do_write=False):
    python_files = hb.list_filtered_paths_recursively(input_folder, include_extensions='.py')

    print ('execute_2to3_on_folder found ' + python_files)

    for file in python_files:
        if do_write:
            command = '2to3 -w ' + file
        else:
            command  = '2to3 ' + file
        system_results = os.system(command)
        print (system_results)

def execute_3to2_on_folder(input_folder, filenames_to_exclude=None, do_write=False):
    print ('filenames_to_exclude', filenames_to_exclude)
    python_files = list_filtered_paths_recursively(input_folder, depth=1, include_extensions='.py', exclude_strings=filenames_to_exclude)

    print (python_files)

    python_3_scripts_dir = 'c:/Anaconda363/scripts'
    sys.path.extend(python_3_scripts_dir)

    for file in python_files:
        if do_write:
            command = '3to2.py -w ' + file
        else:
            command  = '3to2.py ' + file
        system_results = os.system(command)
        print (system_results)


def list_mounted_drive_paths():
    # Iterate through all possible drives to identify which exist.
    drives_to_analyze = list('abcdefghijklmnopqrstuvwxyz')
    drive_paths = []
    for drive in drives_to_analyze:
        drive_path = drive + ':/'
        if os.path.exists(drive_path):
            drive_paths.append(drive_path)

    return drive_paths


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def list_filtered_dirs_recursively(input_dir, include_strings=None, depth=9999):
    if not include_strings:
        include_strings = []
    output_dirs = []

    for cur_dir, dirs_present, files_present in hb.walklevel(input_dir, depth):
        for dir_ in dirs_present:
            if len(include_strings) > 0:
                if any(specific_string in os.path.join(cur_dir, dir_) for specific_string in include_strings):
                    output_dirs.append(dir_)
            else:
                output_dirs.append(dir_)
    return output_dirs



def list_filtered_paths_nonrecursively(input_folder, include_strings=None, include_extensions=None, exclude_strings=None, exclude_extensions=None, return_only_filenames=False):
    # NOTE: the filter strings can be anywhere in the path, not just the filename.

    # ONLY CHANGE
    depth = 1

    # Convert filters to lists
    if include_strings is None:
        include_strings = []
    elif type(include_strings) == str:
        include_strings = [include_strings]
    elif type(include_strings) != list:
        raise TypeError('Must be string or list.')
    if include_extensions is None:
        include_extensions = []
    elif type(include_extensions) == str:
        include_extensions = [include_extensions]
    elif type(include_extensions) != list:
        raise TypeError('Must be string or list.')

    if exclude_strings is None:
        exclude_strings = []
    elif type(exclude_strings) == str:
        exclude_strings = [exclude_strings]
    elif type(exclude_strings) != list:
        raise TypeError('Must be string or list.')

    if exclude_extensions is None:
        exclude_extensions = []
    elif type(exclude_extensions) == str:
        exclude_extensions = [exclude_extensions]
    elif type(exclude_extensions) != list:
        raise TypeError('Must be string or list.')

    iteration_count = 0
    files = []
    for current_folder, folders_present, files_present in os.walk(input_folder):
        if depth is not None:
            if iteration_count >= depth: # CURRENTLY NOT DEACTIVATED because i messed up the logic where the break would end early.
                # NOTE that this only counts the times os.walk is called. Each call may have tons of files. Iterations here are more similar to search depth.
                # Also note this just terminates the walk, missing things that come later in the call.
                return files
        iteration_count += 1
        for filename in files_present:

            include = False
            if not (include_strings or include_extensions):
                include = True
            else:
                if include_strings:
                    if include_extensions:
                        if any(specific_string in os.path.join(current_folder, filename) for specific_string in include_strings) \
                                and any(filename.endswith(specific_extension) for specific_extension in include_extensions) \
                                and not any(specific_string in os.path.join(current_folder, filename) for specific_string in exclude_strings) \
                                and not any(filename.endswith(specific_extension) for specific_extension in exclude_extensions):
                            include = True
                    else:
                        if any(specific_string in filename for specific_string in include_strings):
                            include = True
                else:
                    if include_extensions:
                        if any(filename.endswith(specific_extension) for specific_extension in include_extensions) \
                                and not any(filename.endswith(specific_extension) for specific_extension in exclude_extensions):
                            include = True

            for exclude_string in exclude_strings:
                if exclude_string in os.path.join(current_folder, filename):
                    include = False

            for exclude_extension in exclude_extensions:
                if exclude_extension == os.path.splitext(filename)[1]:
                    include = False

            if include:
                if return_only_filenames:
                    files.append(filename)
                else:
                    files.append(os.path.join(current_folder, filename))
    return files

def split_path_by_timestamp(input_path):
    """Checks a file for having either a 3-part (long) or 2-part (short) timestamp. If found, returns a tuple of (path_before_timestamp, timestamp, extension
     For a timestamp to be valid, it must end in something of the form either for LONGFORM: 20180101_120415_123asd
     or SHORTFORM 20180101_120415
     """
    parent_dir, last_element_in_path = os.path.split(input_path)
    last_element_split = last_element_in_path.split('_')

    pre_extension, extension = os.path.splitext(last_element_split[-1])

    if extension:
        last_element_split[-1] = pre_extension

    # Generate a list where the last three elements are the timestamp elements and everything before is False
    test_split_elements_shortform_intable = list(range(len(last_element_split)))
    test_split_elements_longform_intable = list(range(len(last_element_split)))

    has_short_timestamp = False
    has_long_timestamp = False

    # Test if the last elements are intable FOR SHORTFORM
    for c, i in enumerate(last_element_split):
        try:
            int(i)
            test_split_elements_shortform_intable[c] = i
        except:
            test_split_elements_shortform_intable[c] = False

    # Test if the last elements are intable FOR LONGFORM
    for c, i in enumerate(last_element_split):
        try:
            int(i)
            test_split_elements_longform_intable[c] = i
        except:
            if len(i) == 6:
                try:
                    int(i[0:3])
                    test_split_elements_longform_intable[c] = i
                except:
                    test_split_elements_longform_intable[c] = False
            else:
                test_split_elements_longform_intable[c] = False

    # Test for shortform validity of last 2 elements
    shortform_final_result = []
    if test_split_elements_shortform_intable[-2] is not False:
        if 18000101 < int(test_split_elements_shortform_intable[-2]) < 30180101:
            shortform_final_result.append(True)
        else:
            shortform_final_result.append(False)
    if test_split_elements_shortform_intable[-1] is not False:
        if 0 <= int(test_split_elements_shortform_intable[-1]) <= 245999:
            shortform_final_result.append(True)
        else:
            shortform_final_result.append(False)

    # Test for longform validity of last 2 elements
    longform_final_result = []
    if test_split_elements_longform_intable[-3] is not False:
        if 18000101 < int(test_split_elements_longform_intable[-3]) < 30180101:
            longform_final_result.append(True)
        else:
            longform_final_result.append(False)
    if test_split_elements_longform_intable[-2] is not False:
        if 0 <= int(test_split_elements_longform_intable[-2]) <= 245999:
            longform_final_result.append(True)
        else:
            longform_final_result.append(False)
    if test_split_elements_longform_intable[-1] is not False:
        if 0 <= int(test_split_elements_longform_intable[-1][0:3]) <= 999:
            longform_final_result.append(True)
        else:
            longform_final_result.append(False)

    if shortform_final_result == [True, True]:
        has_short_timestamp = True
    if longform_final_result == [True, True, True]:
        has_long_timestamp = True

    if has_short_timestamp and has_long_timestamp:
         raise NameError('WTF?')
    if not has_short_timestamp and not has_long_timestamp:
        return None

    if has_short_timestamp:
        timestamp = '_'.join(last_element_split[-2:])
        return os.path.join(parent_dir, '_'.join(last_element_split[0: -2])), timestamp, extension

    if has_long_timestamp:
        timestamp = '_'.join(last_element_split[-3:])
        return os.path.join(parent_dir, '_'.join(last_element_split[0: -3])), timestamp, extension


def get_most_recent_timestamped_file_in_dir(input_dir, pre_timestamp_string=None, include_extensions=None, recursive=False):
    if recursive:
        paths_list = list_filtered_paths_recursively(input_dir, pre_timestamp_string, include_extensions=include_extensions)
    else:
        paths_list = list_filtered_paths_nonrecursively(input_dir, pre_timestamp_string, include_extensions=include_extensions)

    sorted_paths = OrderedDict()
    for path in paths_list:
        r = split_path_by_timestamp(path)
        sorted_paths[r[1]] = r[0] + r[1] + r[2]



    print ('NEEDS MINOR FIXING FOR get_most_recent_timestamped_file_in_dir')
    sorted_return_list = sorted(sorted_paths)
    if len(sorted_return_list) > 0:
        most_recent_key = sorted_return_list[-1]
        to_return = sorted_paths[most_recent_key]
    else:
        to_return = []

    to_return = '_'.join(to_return)
    return to_return


def list_filtered_paths_recursively(input_folder, include_strings=None, include_extensions=None, exclude_strings=None, exclude_extensions=None, return_only_filenames=False, depth=5000, only_most_recent=False):
    # NOTE: the filter strings can be anywhere in the path, not just the filename.

    """If only_most_recent is True, will analyze time stampes and only return similar-named files with the most recent."""

    # Convert filters to lists
    if include_strings is None:
        include_strings = []
    elif type(include_strings) == str:
        include_strings = [include_strings]
    elif type(include_strings) != list:
        raise TypeError('Must be string or list.')

    if include_extensions is None:
        include_extensions = []
    elif type(include_extensions) == str:
        include_extensions = [include_extensions]
    elif type(include_extensions) != list:
        raise TypeError('Must be string or list.')

    if exclude_strings is None:
        exclude_strings = []
    elif type(exclude_strings) == str:
        exclude_strings = [exclude_strings]
    elif type(exclude_strings) != list:
        raise TypeError('Must be string or list.')

    if exclude_extensions is None:
        exclude_extensions = []
    elif type(exclude_extensions) == str:
        exclude_extensions = [exclude_extensions]
    elif type(exclude_extensions) != list:
        raise TypeError('Must be string or list.')
    iteration_count = 0
    files = []
    for current_folder, folders_present, files_present in os.walk(input_folder):
        if depth is not None:
            if iteration_count >= depth: # CURRENTLY NOT DEACTIVATED because i messed up the logic where the break would end early.
                # NOTE that this only counts the times os.walk is called. Each call may have tons of files. Iterations here are more similar to search depth.
                # Also note this just terminates the walk, missing things that come later in the call.
                return files
        iteration_count += 1
        for filename in files_present:
            include = False
            if not (include_strings or include_extensions):
                include = True
            else:
                if include_strings:
                    if include_extensions:
                        if any(specific_string in os.path.join(current_folder, filename) for specific_string in include_strings) \
                                and any(filename.endswith(specific_extension) for specific_extension in include_extensions) \
                                and not any(specific_string in os.path.join(current_folder, filename) for specific_string in exclude_strings) \
                                and not any(filename.endswith(specific_extension) for specific_extension in exclude_extensions):
                            include = True
                    else:
                        if any(specific_string in filename for specific_string in include_strings):
                            include = True
                else:
                    if include_extensions:
                        if any(filename.endswith(specific_extension) for specific_extension in include_extensions) \
                                and not any(filename.endswith(specific_extension) for specific_extension in exclude_extensions):
                            include = True

            for exclude_string in exclude_strings:
                if exclude_string in os.path.join(current_folder, filename):
                    include = False

            for exclude_extension in exclude_extensions:
                if exclude_extension == os.path.splitext(filename)[1]:
                    include = False

            if include:
                if return_only_filenames:
                    files.append(filename)
                else:
                    files.append(os.path.join(current_folder, filename))

            if only_most_recent is True:
                print ('NYI only_most_recent')
                # final_files = []
                # for file in files:
                #     input_dir = os.path.split(file)[0]
                #     pre_timestamp_string, unused_timestamp = hb.get_pre_timestamp_file_root(file)
                #     most_recent = get_most_recent_timestamped_file_in_dir(input_dir, pre_timestamp_string=pre_timestamp_string, include_extensions=None)
                #     final_files.append(most_recent)
                # files = final_files
    return files
# Example Usage
# input_folder = 'G:\\IONE-Old\\NATCAP\\bulk_data\\worldclim\\baseline\\30s'
# pp(get_list_of_file_uris_recursively(input_folder, '.bil'))




def unzip_file(input_uri, output_folder=None, verbose=True):
    'Unzip file in place. If no output folder specified, place in input_uris folder'
    if not output_folder:
        output_folder = os.path.join(os.path.split(input_uri)[0], os.path.splitext(input_uri)[0])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    fh = open(input_uri, 'rb')
    z = zipfile.ZipFile(fh)
    for name in z.namelist():
        if verbose:
            pp(name, output_folder)
        z.extract(name, output_folder)
    fh.close()


def unzip_folder(input_folder, output_folder=None, verbose=True):
    if not output_folder:
        output_folder = input_folder
    input_files = os.listdir(input_folder)
    for i in range(len(input_files)):
        input_uri = os.path.join(input_folder, input_files[i])
        unzip_file(input_uri, output_folder, verbose)


def zip_files_from_dir_by_filter(input_dir, zip_uri, include_strings=None, include_extensions=None, exclude_strings=None, exclude_extensions=None, return_only_filenames=False):

    "FLATTENS into target zip"
    if not os.path.exists(input_dir):
        raise NameError('File not found: ' + str(input_dir))
    if not os.path.splitext(zip_uri)[1] == '.zip':
        raise NameError('zip_uri must end with zip')

    zipf = zipfile.ZipFile(zip_uri, 'w', zipfile.ZIP_DEFLATED)
    for i in hb.list_filtered_paths_recursively(input_dir, include_strings, include_extensions, exclude_strings, exclude_extensions, return_only_filenames):
        zipf.write(i, os.path.basename(i))

    zipf.close()


def copy_files_from_dir_by_filter(input_dir, dst_dir, include_strings=None, include_extensions=None, exclude_strings=None, exclude_extensions=None, return_only_filenames=False):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for i in hb.list_filtered_paths_recursively(input_dir, include_strings, include_extensions, exclude_strings, exclude_extensions, return_only_filenames):
        filename = os.path.split(i)[1]
        new_uri = os.path.join(dst_dir, filename)

        shutil.copy(i, new_uri)


def copy_files_from_dir_by_filter_preserving_dir_structure(input_dir, dst_dir, include_strings=None, include_extensions=None, exclude_strings=None, exclude_extensions=None, return_only_filenames=False):
    if not os.path.exists(dst_dir):
        try:
            os.mkdir(dst_dir)
        except:
            'perhaps no dir was specified. using curdir'

    for uri in hb.list_filtered_paths_recursively(input_dir, include_strings, include_extensions, exclude_strings, exclude_extensions, return_only_filenames):
        modified_uri = uri.replace(input_dir, dst_dir, 1)
        try:
            os.makedirs(os.path.split(modified_uri)[0])
        except:
            'exists?'
        # os.makedirs(os.path.split(modified_uri)[0], exist_ok=True)
        shutil.copy(uri, modified_uri)

def zip_files_from_dir_by_filter_preserving_dir_structure(input_dir, zip_dst_uri, include_strings=None, include_extensions=None, exclude_strings=None, exclude_extensions=None, return_only_filenames=False):
    # PRESERVES dir structure inside zip.
    temp_dir = input_dir + '_temp'
    copy_files_from_dir_by_filter_preserving_dir_structure(input_dir, temp_dir, include_strings, include_extensions, exclude_strings, exclude_extensions, return_only_filenames)

    # Because i couldnt figure out how to zip to a non curdir, i had this hack
    new_zip_dst_uri = os.path.split(zip_dst_uri)[1]
    zip_dir(temp_dir, zip_dst_uri)

    # remove temporary dir only, because now it's in a zip
    shutil.rmtree(temp_dir, ignore_errors=True)

def zip_dir(input_dir, zip_uri):
    if not os.path.exists(input_dir):
        raise NameError('File not found: ' + str(input_dir))
    if not os.path.splitext(zip_uri)[1] == '.zip':
        raise NameError('zip_uri must end with zip')

    zipf = zipfile.ZipFile(zip_uri, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            destination_relative_to_zip_archive_path = os.path.join(root, file).replace(input_dir, '')
            current_file_to_zip_path = os.path.join(root, file)
            print ('Zipping ' + current_file_to_zip_path + ' to ' + destination_relative_to_zip_archive_path)
            zipf.write(current_file_to_zip_path, destination_relative_to_zip_archive_path)

    zipf.close()


def zip_list_of_paths(paths_list, zip_path):
    if not os.path.splitext(zip_path)[1] == '.zip':
        raise NameError('zip_path must end with zip')

    zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for i in paths_list:
        print('Zipping ' + str(i))
        if not os.path.exists(i):
            raise NameError('File not found when zipping: ' + str(i))
        zipf.write(i, os.path.basename(i))

    zipf.close()



















def list_files_in_dir_recursively(input_folder, filter_strings=None, filter_extensions=None, max_folders_analyzed=None, return_only_filenames=False):
    print ('Function deprecated (list_files_in_dir_recursively). Consider using list_filtered_paths_recursively.')
    if type(filter_strings) == str:
        filter_strings = [filter_strings]
    if type(filter_extensions) == str:
        filter_extensions = [filter_extensions]

    iteration_count = 0
    files = []
    for current_folder, folders_present, files_present in os.walk(input_folder):
        iteration_count += 1
        if max_folders_analyzed is not None:
            if iteration_count > max_folders_analyzed:
                # NOTE that this only counts the times os.walk is called. Each call may have tons of files. Iterations here are more similar to search depth.
                return files

        for filename in files_present:
            include = False
            if filter_strings:
                if filter_extensions:
                    if any(specific_string in filename for specific_string in filter_strings) \
                            and any(filename.endswith(specific_extension) for specific_extension in filter_extensions):
                        include = True
                else:
                    if any(specific_string in filename for specific_string in filter_strings):
                        include = True
            else:
                if filter_extensions:
                    if any(filename.endswith(specific_extension) for specific_extension in filter_extensions):
                        include = True
                else:
                    include = True

            if include:
                if return_only_filenames:
                    files.append(filename)
                else:
                    files.append(os.path.join(current_folder, filename))
    return files

def list_dirs_in_dir_recursively(input_folder, filter_strings=[], max_folders_analyzed=None, return_only_filenames=False):
    if type(filter_strings) == str:
        filter_strings = [filter_strings]

    iteration_count = 0
    folders = []
    for current_folder, folders_present, files_present in os.walk(input_folder):
        iteration_count += 1
        if max_folders_analyzed is not None:
            if iteration_count > max_folders_analyzed:
                # NOTE that this only counts the times os.walk is called. Each call may have tons of files. Iterations here are more similar to search depth.
                return folders

        for current_folder in [current_folder]:
            include = False
            if filter_strings:
                if any(specific_string in current_folder for specific_string in filter_strings):
                    include = True
            else:
                include = True

            folders.append(current_folder)
    return folders


def assert_file_existance(dataset_uri_list):
    """Assert that provided uris exist in filesystem.

    Verify that the uris passed in the argument exist on the filesystem
    if not, raise an exeception indicating which files do not exist

    Args:
        dataset_uri_list (list): a list of relative or absolute file paths to
            validate

    Returns:
        None

    Raises:
        IOError: if any files are not found
    """
    not_found_uris = []
    for uri in dataset_uri_list:
        if not os.path.exists(uri):
            not_found_uris.append(uri)

    if len(not_found_uris) != 0:
        error_message = (
            "The following files do not exist on the filesystem: " +
            str(not_found_uris))
        raise NameError(error_message)
        # raise exceptions.IOError(error_message)


def swap_filenames(left_uri, right_uri):
    left_temp_uri = suri(left_uri, 'temp')
    os.rename(left_uri, left_temp_uri)
    os.rename(right_uri, left_uri)
    os.rename(left_temp_uri, right_uri)

def displace_file(src_uri, to_displace_uri, displaced_uri=None, delete_original=False):
    if not displaced_uri:
        displaced_uri = nd.rsuri(src_uri, 'displaced_by_' + explode_uri(src_uri)['file_root'])
    os.rename(to_displace_uri, displaced_uri)
    os.rename(src_uri, to_displace_uri)

    if delete_original:
        os.remove(displaced_uri)

def rename_with_overwrite(src_path, dst_path):
    if os.path.exists(dst_path):
        hb.remove_path(dst_path)
    os.rename(src_path, dst_path)

def replace_file(src_uri, dst_uri, delete_original=True):
    if os.path.exists(dst_uri):
        if delete_original:
            os.remove(dst_uri)
        else:
            os.rename(dst_uri, rsuri(new_location, 'replaced_by_' + src_uri))

    try:
        os.rename(src_uri, dst_uri)
    except:
        raise Exception('Failed to rename ' + src_uri + ' to ' + dst_uri)


def replace_ext(input_uri, desired_ext):
    if os.path.splitext(input_uri)[1]:
        if desired_ext.startswith('.'):
            modified_uri = os.path.splitext(input_uri)[0] + desired_ext
        else:
            modified_uri = os.path.splitext(input_uri)[0] + '.' + desired_ext
    else:
        raise NameError('Cannot replace extension on the input_uri given because it did not have an extension.')
    return modified_uri

def copy_shapefile(input_uri, output_uri):
    # Because shapefiles have 4+ separate files, use this to smartly copy all of the ones that exist based on versions of input uri.
    for ext in hb.config.possible_shapefile_extensions:
        potential_uri = hb.replace_ext(input_uri, ext)
        if os.path.exists(potential_uri):
            potential_output_uri = hb.replace_ext(output_uri, ext)
            shutil.copyfile(potential_uri, potential_output_uri)

def rename_shapefile(input_uri, output_uri):
    # Because shapefiles have 4+ separate files, use this to smartly rename all of the ones that exist based on versions of input uri.
    for ext in hb.config.possible_shapefile_extensions:
        potential_uri = hb.replace_ext(input_uri, ext)
        if os.path.exists(potential_uri):
            potential_output_uri = hb.replace_ext(output_uri, ext)
            os.rename(potential_uri, potential_output_uri)

def remove_shapefile(input_uri):
    # Because shapefiles have 4+ separate files, use this to smartly rename all of the ones that exist based on versions of input uri.
    for ext in hb.config.possible_shapefile_extensions:
        potential_uri = hb.replace_ext(input_uri, ext)
        if os.path.exists(potential_uri):
            os.remove(potential_uri)

def replace_shapefile(src_uri, dst_uri):
    for ext in hb.config.possible_shapefile_extensions:
        potential_uri = hb.replace_ext(src_uri, ext)
        if os.path.exists(potential_uri):
            potential_output_uri = hb.replace_ext(dst_uri, ext)
            os.replace(potential_uri, potential_output_uri)

def remove_temporary_files():
    for uri_to_delete in hb.config.uris_to_delete_at_exit:
        try:
            if os.path.splitext(uri_to_delete)[1] == '.shp':
                remove_shapefile(uri_to_delete)
            else:
                os.remove(uri_to_delete)
            # L.debug('Deleting temporary file: ' + str(uri_to_delete))
        except:
            pass
            # L.debug('Couldn\'t remove temporary file: ' + str(uri_to_delete))
atexit.register(remove_temporary_files)

def remove_uri_at_exit(input):
    if isinstance(input, str):
        hb.config.uris_to_delete_at_exit.append(input)
    elif isinstance(input, hb.ArrayFrame):
        hb.config.uris_to_delete_at_exit.append(input.uri)


def remove_path(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    filename = os.path.join(root, name)
                    os.chmod(filename, stat.S_IWRITE)
                    os.remove(filename)
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(path)
        else:
            os.remove(path)
    else:
        'couldnt find path, but no worries, we good.'

        # if os.path.isdir(path):
    #     shutil.rmtree(path, ignore_errors=True)
    # else:
    #     shutil.remove

    # if os.path.isdir(path):
    #     os.rmdir(path)
    # else:
    #     os.remove(path)
    # try:
    #     os.remove(path)
    # except:
    #     'Probably just didnt exist.'

def remove_at_exit(uri):
    hb.config.uris_to_delete_at_exit.append(uri)


if __name__=='__main__':
    input_folder = 'C:\\OneDrive\\Projects\\numdal\\natcap'
    # execute_2to3_on_folder(input_folder, do_write=True)

def path_rename_change_dir(input_path, new_dir):
    """Change the directory of a file given its input path, preserving the name. NOTE does not do anything to the file"""
    return os.path.join(new_dir, os.path.split(input_path)[1])


def file_root(input_path):
    return path_file_root(input_path)


def path_file_root(input_path):
    return os.path.splitext(os.path.split(input_path)[1])[0]



def copy_shutil_flex(src, dst, copy_tree=True):
    """Helper util that allows copying of files or dirs in same function"""
    if os.path.isdir(src):
        if not os.path.exists(dst):
            hb.create_directories(dst)
        if copy_tree:
            copy_shutil_copytree(src, dst)
        else:
            dst = os.path.join(dst, os.path.basename(src))
            shutil.copyfile(src, dst)
    else:
        dst_dir = os.path.split(dst)[0]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, dst)

def copy_shutil_copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            distutils.dir_util.copy_tree(s, d)
            # shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def create_directories(directory_list):
    """Make directories provided in list of path strings.

    This function will create any of the directories in the directory list
    if possible and raise exceptions if something exception other than
    the directory previously existing occurs.

    Args:
        directory_list (list/string): a list of string uri paths

    Returns:
        None
    """
    if isinstance(directory_list, str):
        directory_list = [directory_list]
    elif not isinstance(directory_list, list):
        raise TypeError('Must give create_directories either a string or a list.')

    for dir_name in directory_list:
        try:
            os.makedirs(dir_name)
        except OSError as exception:
            #It's okay if the directory already exists, if it fails for
            #some other reason, raise that exception
            if (exception.errno != errno.EEXIST and
                    exception.errno != errno.ENOENT):
                raise

def exists(path):
    # os.path.exists throws an exception rather than False if given None. This version resolves None as False.
    if not path:
        return False
    else:
        try:
            if os.path.exists(path):
                return True
            else:
                return False
        except:
            return False




