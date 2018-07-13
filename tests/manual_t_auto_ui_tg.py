import os, sys, math
import hazelbean as hb
from hazelbean import project_flow_tg

def function_a(p):
    p.attr_from_func_a = 'blarg'

def function_b(p):
    p.attribute_from_func_b = 'BBB'


if __name__=='__main__':

    p = project_flow_tg.ProjectFlowTG()
