#! /usr/bin/env python

descr   = """
"""

from os.path import join
import os
import sys, compileall

DISTNAME            = 'openopt'
DESCRIPTION         = 'A python module for numerical optimization'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Dmitrey Kroshko',
MAINTAINER_EMAIL    = 'dmitrey-at-openopt-dot-org',
URL                 = 'http://openopt.org',
LICENSE             = 'new BSD'

sys.path.append(os.getcwd() + os.sep + 'openopt')
from ooVersionNumber import __version__ as ooVer
openopt_version = ooVer

DOWNLOAD_URL        = 'http://openopt.org/images/3/33/OpenOpt.zip'

try:
    import setuptools
except:
    print('you should have setuptools installed (http://pypi.python.org/pypi/setuptools), for some Linux distribs you can get it via [sudo] apt-get install python-setuptools')
    print('press Enter for exit...')
    raw_input()
    exit()
    
from distutils.errors import DistutilsError
#from numpy.distutils.system_info import system_info, NotFoundError, dict_append, so_ext
from numpy.distutils.core import setup, Extension

DOC_FILES = []

def configuration(parent_package='',top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')
    pkg_prefix_dir = 'openopt'

    # Get the version

    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name,parent_package,top_path,
        version     = openopt_version,
        maintainer  = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        license = LICENSE,
        url = URL,
        download_url = DOWNLOAD_URL,
        long_description = LONG_DESCRIPTION)


    # XXX: once in SVN, should add svn version...
    #print config.make_svn_version_py()

    # package_data does not work with sdist for setuptools 0.5 (setuptools bug),
    # so we need to add them here while the bug is not solved...

    return config


if __name__ == "__main__":
#    solverPaths = {}
#    #File = string.join(__file__.split(os.sep)[:-1], os.sep)
#    for root, dirs, files in os.walk('openopt'+os.sep +'solvers'):
#        #for root, dirs, files in os.walk(os.path.dirname(file)+os.sep+'solvers'):
#        rd = root.split(os.sep)
#        if '.svn' in rd: continue
#        rd = rd[rd.index('solvers')+1:]
#        for file in files:
#            if len(file)>6 and file[-6:] == '_oo.py':
#                solverPaths[file[:-6]] = string.join(rd,'.') + '.'+file[:-3]
#    f = open('solverPaths.py', 'w')
#    f.write('solverPaths = ' + str(solverPaths))
#    f.close()
#    shutil.move('solverPaths.py', 'openopt' + os.sep + 'kernel' + os.sep + 'solverPaths.py')



    # setuptools version of config script

    # package_data does not work with sdist for setuptools 0.5 (setuptools bug)
    # So we cannot add data files via setuptools yet.

    #data_files = ['test_data/' + i for i in TEST_DATA_FILES]
    #data_files.extend(['docs/' + i for i in doc_files])
    setup(configuration = configuration,
        install_requires=['numpy','setproctitle'], # can also add version specifiers   #namespace_packages=['kernel'],
        #setup_requires = 'setproctitle', 
        #py_modules = ['kernel', 'tests', 'examples', 'solvers'],
        packages=setuptools.find_packages(),
        include_package_data = True,
        #package_data = '*.txt',
        test_suite='',#"openopt.tests", # for python setup.py test
        zip_safe=False, # the package can run out of an .egg file
        #FIXME url, download_url, ext_modules
        classifiers =
            [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering']
    )
