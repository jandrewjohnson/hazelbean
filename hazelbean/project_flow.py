import os, sys, types, inspect, logging, collections, time, copy
# import nose
from collections import OrderedDict
from osgeo import gdal, osr, ogr
import numpy as np
import hazelbean as hb
import multiprocessing

try:
    import anytree
except:
    'anytree is probably not needed except for project flow.'

L = hb.get_logger('project_flow')
L.setLevel(logging.INFO)

def run_iterator_in_parallel(p, task, iteration_counter):
    things_returned = []
    for child in task.children:
        L.info('iter ' + str(iteration_counter) + ': Running task ' + str(child.name) + ' with iterator parent ' + child.parent.name + ' in dir ' + str(p.cur_dir))
        r = p.run_task(child)
        things_returned.append(r)


    return things_returned

class Task(anytree.NodeMixin):
    def __init__(self, function, project, parent=None, type='task', **kwargs):
        """
        There are TWO basic types of parallelizataion. Tasks that aren't dependent sequentially,
        or tasks that are defined in a different, unrelated extent, but possibly with sequential tasks.
        This Iterator object is fo the latter type, iterating over some zonal definition.
        """

        self.function = function
        self.p = project
        self.type = type

        # Note that parent here is defined by anytree and it is not possible to set it to None, as parent has to be a Node
        if parent:
            self.parent = parent


        self.name = self.function.__name__
        self.creates_dir = kwargs.get('creates_dir', True)
        self.logging_level = None # Will be inherited from project flow or set explicitly

        self.run = 1
        self.skip_existing = 0 # Will thus overwrite by default.


class ProjectFlow(object):
    def __init__(self, project_dir=None):
        try:
            self.calling_script = inspect.stack()[1][1]
        except:
            L.debug('Could not identify a calling script.')

        ## PROJECT LEVEL ATTRIBUTES
        # Set the project-level logging level. Individual tasks can overwrite this.
        self.logging_level = logging.INFO

        # # WARNING, although this seems logical, it can mess up multiprocessing if L has a handler. Move back outside.
        # self.L = hb.get_logger('project_flow')

        # TODOO Renable run_dir separation.
        # If true, generates a random dirname and creates it in the folder determined by the following options.
        self.make_run_dir = False

        # If project_dir is not defined, use CWD.
        if project_dir:
            self.project_dir = project_dir
        else:
            self.project_dir = os.getcwd() # This may be temporary though because it may be overwritten by UI

        if not os.path.isdir(self.project_dir):
            raise NotADirectoryError('A Project Flow object is based on defining a project_dir as its base.')

        self.ui_agnostic_project_dir = self.project_dir # The project_dir can be overwritten by a UI but it can be useful to know where it would have been for eg decing project_base_data_dir
        self.project_name = hb.file_root(self.project_dir)

        # args is used by UI elements.
        self.args = OrderedDict()

        self.task_paths = OrderedDict()

        self.prepend = '' # for logging

        # Functions are called via their position within a tree data structure.
        # The logic of the project is defined via the flow_tree to which tasks, batches, secenarios etc. are added and run top-to-bottom
        # The tree itself here is initialized by setting the function to be execute
        self.task_tree = Task(self.execute, 'Task tree') # NOTICE that this Task() has no parent. It is the root node.

        self.jobs = [] # Allow only 1 jobs pipeline

        # State variables that are passed into the task's funciton via p. attribtues
        self.cur_task = None
        self.run_this = None
        self.skip_existing = None

        self.task_names_defined = [] # Store a list of tasks defined somewhere in the target script. For convenience, e.g., when setting runtime conditionals based on function names existence.

    def __str__(self):
        return 'Hazelbean ProjectFlow object. ' + hb.pp(self.__dict__, return_as_string=True)

    def __repr__(self):
        return 'Hazelbean ProjectFlow object. ' + hb.pp(self.__dict__, return_as_string=True)

    def write_args_to_project(self, args):
        L.debug('write_args_to_project.')
        for k, v in args.items():
            if k in self.__dict__:
                L.debug('Arg given to P via UI was already in P. Overwritting: ' + str(k) + ', ' + str(v) +', Original value: ' + str(self.__dict__[k]))
            self.__setattr__(k, v)

    def show_tasks(self):
        for pre, fill, task in anytree.RenderTree(self.task_tree):
            if task.name == 'ProjectFlow':
                L.info(pre + task.name)
            else:
                L.info(pre + task.name + ', running: ' + str(task.run) + ', overwriting: ' + str(task.skip_existing))

    def add_task(self, function, project=None, parent=None, type='task', **kwargs):

        if not project:
            project = self
        if not parent:
            parent = self.task_tree
        if not isinstance(function, collections.Callable):
            raise TypeError(
                'Fuction passed to add_task() must be callable. ' + str(function.__name__) + ' was not.')

        task = Task(function, self, parent=parent, type=type, **kwargs)

        self.task_names_defined.append(function.__name__)

        # Add attribute to the parent object (the ProjectFlow object) referencing the iterator_object
        setattr(self, task.name, task)

        # Tasks inherit by default the projects' logging level.
        task.logging_level = self.logging_level

        return task

    def add_iterator(self, function, project=None, parent=None, type='iterator', **kwargs):
        if not project:
            project = self
        if not parent:
            parent = self.task_tree

        if not isinstance(function, collections.Callable):
            raise TypeError('Fuction passed to add_iterator() must be callable. ' + str(function.__name__) + ' was not.')

        # Create the iterator object
        iterator = Task(function, self, parent=parent, type=type, **kwargs)

        # Add attribute to the parent object (the ProjectFlow object) referencing the iterator_object
        setattr(self, iterator.name, iterator)

        # Tasks inherit by default the projects' logging level.
        iterator.logging_level = self.logging_level

        return iterator


    def run_task(self, current_task):
        for task in anytree.LevelOrderIter(current_task, maxlevel=1): # We ALWAYS have maxlevel = 1 even if there are nested things because it handles all nested children recursively and we don't want the tree iterator to find them. This is sorta stupid instead of just giving the tree itself at the top  node.

            # If the function is not the root execute function, go ahead and run it. Can't run execute this way because it doesn't have a parent.
            if not task.function.__name__ == 'execute':

                # Set task_dirs and cur_dirs based on tree position
                if task.parent.type == 'task':
                    if task.parent is not None and getattr(task.parent, 'task_dir', None):
                        task.task_dir = os.path.join(task.parent.task_dir, task.function.__name__)
                    else:
                        task.task_dir = os.path.join(self.run_dir, task.function.__name__)
                    self.cur_dir = task.task_dir
                elif task.parent.type == 'iterator':
                    task.task_dir = os.path.join(self.cur_dir_parent_dir, task.name)
                    self.cur_dir = task.task_dir
                else:
                    raise NameError('Unknown Node type')

                # Set the project level task_dirs object to have an attribute equal to the current name. This makes it possible for functions later in the analysis  script to have access to
                # previous task_dir locations.
                setattr(self, task.name + '_dir', task.task_dir)



                # In addition to self.cur_dir, there are also these two project-level convenience funcitons.
                self.cur_task = task
                self.run_this = task.run # NYI, task skipping enabled here.
                self.skip_existing = task.skip_existing

                if self.skip_existing:
                    if os.path.exists(self.cur_dir):
                        self.run_this = 0

                if not os.path.exists(self.cur_dir) and task.creates_dir:
                    hb.create_directories(self.cur_dir)

                # # NYI, but I want to implement task-level logging conditionals.
                # L.setLevel(task.logging_level)

                if task.type == 'task':
                    if self.run_this:
                        # If the task's parent is an iterator, we want to report different info, otherwise these are the same.
                        if task.parent.type == 'iterator':
                            r = task.function(self)
                        else:
                            self.prepend = ''
                            L.info(self.prepend + 'Running task ' + str(task.name) + ' in dir ' + str(self.cur_dir))
                            task.function(self)  # Running the Task including anyting in p.run_this

                        if task.creates_dir:
                            hb.create_directories(self.cur_dir)
                            assert os.path.exists(self.cur_dir)

                    # NYI, task skipping enabled here.
                    else:
                        if os.path.isdir(self.cur_dir):
                            if task.run:
                                L.info('Skipping task ' + str(task.name) + ' because dir existed at ' + str(self.cur_dir))
                            else:
                                L.info('Instructed to skip task ' + str(task.name) + ' and loading from ' + str(self.cur_dir))

                            # CALL THE TASK FUNCTION
                            task.function(self)  # Running the Task EXCLUDING anyting in p.run_this

                        else:
                            L.debug('Skipping task ' + str(task.name) + ' but its task_dir does not exist at ' + str(self.cur_dir))
                elif task.type == 'iterator':
                    self.prepend += '    '
                    L.info('Creating iterator ' + str(task.name))

                    # Run the function for defining the iterator
                    if task.run:

                        # HACK, I failed to understand why sometiems the dirs weren't created in time. Thus I force it here.
                        hb.create_directories(self.cur_dir)
                        assert os.path.exists(self.cur_dir)

                        task.function(self)

                    else:
                        # NYI, task skipping enabled here.
                        L.info('Skipping running Iterator.')

            # Whether run or not, search for children
            if len(task.children) > 0:

                # If the current task is an iterator, then check for replacements before calling the child task.
                # Definition of the projects' self.iterator_replacements is the one part of ProjectFlow that the analysis script needs to be aware of,
                # creating a dict of key-value pairs that are replaced with each step in the iterator.
                if task.type == 'iterator':

                    # First check dimensions of iterator_replacements:
                    replacement_lengths = []
                    for replacement_attribute_name, replacement_attribute_value in self.iterator_replacements.items():
                        replacement_lengths.append(len(replacement_attribute_value))
                        assert(len(set(replacement_lengths))==1) # Check that all of the same size.
                    num_iterations = replacement_lengths[0]

                    self.run_in_parallel = True # TODOO Connect to UI
                    n_workers = 10
                    if self.run_in_parallel:
                        # OPTIMIZATION NOTE: It's slow to spawn 460 processes when they are just going to be skipped, thus run_this for iterators needs to be improved.
                        worker_pool = multiprocessing.Pool(n_workers) # NOTE, worker pool and results are LOCAL variabes so that they aren't pickled when we pass the project object.
                    results = []

                    # Iterate through all children of the iterator with new replacement values.
                    for iteration_counter in range(num_iterations):

                        # NOTICE strange dimensionality here: even within a single iteration, we have to iterate through self.iterator_replacements because we might have more than 1 var that needs replacing
                        replacements = OrderedDict()
                        for replacement_attribute_name, replacement_attribute_values in self.iterator_replacements.items():
                            current_replacement_value = self.iterator_replacements[replacement_attribute_name][iteration_counter]
                            replacements[replacement_attribute_name] = replacement_attribute_values
                            setattr(self, replacement_attribute_name, current_replacement_value)
                            project_copy = copy.copy(self)# Freeze it in place (necessary for parallelizing)

                        if self.run_in_parallel:
                            L.info('Initializing PARALLEL task ' + str(iteration_counter) + ' ' + task.name + ' with replacements: ' + str(replacements))

                            # We use apply_async, which immediately lets the next line calculate. It is blocked below with results.get()
                            result = worker_pool.apply_async(func=run_iterator_in_parallel, args=(project_copy, task, iteration_counter))

                            results.append(result)
                        else:
                            L.info('Starting task ' + str(iteration_counter) + ' ' + task.name + ' while replacing ' + str(replacement_attribute_name))
                            for child in task.children:
                                self.run_task(child)

                    # Once all the iterations are done, iterate through the stored results and call their get functions, which blocks running past this point until all are done.
                    # SUPER CONFUSING POINT. the project object will be modified independently by all tasks. Cant think of a good way ro rejoin them
                    returns_from_parallel_tasks = []
                    for i in results:
                        for j in i.get():
                            if j is not None:
                                returns_from_parallel_tasks.append(j)
                    for i in returns_from_parallel_tasks:
                        if i[1] == 'append_to_list':
                            if isinstance(i[2], list):
                                getattr(self, i[0]).extend(i[2])
                            else:
                                getattr(self, i[0]).append(i[2])
                        # p.layers_to_stitch.append(5)



                # Task is not an iterator, thus we just call it's child directly
                else:
                    for child in task.children:
                        self.run_task(child)# Run the child found by iterating the task-node's children

        try:
            if(len(r)) > 0:
                return r
        except:
            'nothing needed returning'


    def execute(self, args=None):
        self.show_tasks()

        if not isinstance(self.task_tree, hb.Task):
            raise NameError('Execute was called in ProjectFlow but no tasks were in task_tree.')

        # Execute can be passed an args dict that can be, for instance, generated by a UI. If args exists, write each
        # key value pair as project object level attributes.
        if args:
            self.write_args_to_project(args)

        # Check to see if any args have been set that change runtime conditionals.
        if args:
            for k, v in args.items():
                if k.startswith('run_') and k.split('_', 1)[1] in self.task_names_defined:
                    a = getattr(self, k.split('_', 1)[1], None)
                    if a:
                        a.run = v

            # try:
            #     if k.startswith('run_') and k.split('_', 1)[1] in self.task_names_defined:
            #         a = getattr(self, k.split('_', 1)[1], None)
            #         a.run = v
            #         print(a)
            # except:
            #     'parsing didnt find any matching functions.'


        # TRICKY NOTE: Workspace_dir and project_dir are often but not always the same. Project dir is defined by the model code while workspace dir is defined by the script or UI that is calling the model code. If workspace_dir is defined, it will overwrite project_dir? Is this right?
        self.workspace_dir = getattr(self, 'workspace_dir', None)
        if self.workspace_dir: # Then there IS a UI, so use itz
            self.project_dir = self.workspace_dir

        # If no additional dirs are specified, assume inputs, intermediates and outputs all go in CWD
        # These are the DEFAULT LOCATIONS, but they could be changed by the UI or script calling it for example to make batch runs happen.
        self.input_dir = getattr(self, 'input_dir', os.path.join(self.project_dir, 'input'))
        self.intermediate_dir =  getattr(self, 'intermediate_dir', os.path.join(self.project_dir, 'intermediate'))

        # Because the UI will give '', need to overwrite this manually.
        if self.intermediate_dir == '':
            self.intermediate_dir = os.path.join(self.project_dir, 'intermediate')

        self.output_dir = getattr(self, 'output_dir', os.path.join(self.project_dir, 'output'))

        self.model_base_data_dir = os.path.join(self.ui_agnostic_project_dir, '../../base_data')  # Data that must be redistributed with this project for it to work. Do not put actual base data here that might be used across many projects.
        self.project_base_data_dir = os.path.join(self.project_dir, 'base_data')  # Data that must be redistributed with this project for it to work. Do not put actual base data here that might be used across many projects.
        self.temporary_dir = getattr(self, 'temporary_dir', os.path.join(hb.PRIMARY_DRIVE, 'temp'))  # Generates new run_dirs here. Useful also to set the numdal temporary_dir to here for the run.

        self.run_string = hb.pretty_time()  # unique string with time-stamp. To be used on run_specific identifications.
        self.basis_name = ''  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.
        self.basis_dir = os.path.join(self.intermediate_dir, self.basis_name)  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.

        if self.make_run_dir:
            self.run_dir = getattr(self, 'run_dir', hb.make_run_dir(self.intermediate_dir))
        else:
            self.run_dir = self.intermediate_dir

        # hb.create_directories([self.input_dir, self.intermediate_dir, self.output_dir])

        L.info('\nRunning Project Flow')
        self.run_task(self.task_tree) # LAUNCH the task tree. Everything else will be called via recursive task calls.

        L.info('Script complete.')


if __name__=='__main__':
    print ('cannot run by itself.')